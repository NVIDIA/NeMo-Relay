// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const lib = require('./index.js');
const plugin = require('./plugin.js');

const ADAPTIVE_PLUGIN_KIND = 'adaptive';

/**
 * Create a default adaptive component config.
 *
 * Returns the minimal top-level adaptive config shape with `version = 1` so
 * callers can add state, telemetry, and scheduler settings incrementally.
 *
 * @returns {object} A new adaptive config object.
 * @remarks The returned object is detached from runtime state until it is
 * wrapped with `ComponentSpec` and activated through the plugin system.
 */
function defaultConfig() {
  return {
    version: 1,
  };
}

/**
 * Create an in-memory adaptive state backend spec.
 *
 * Produces the backend descriptor for ephemeral adaptive state stored inside
 * the current process rather than an external datastore.
 *
 * @returns {object} An adaptive backend spec using in-memory storage.
 * @remarks This backend does not persist state across process restarts.
 */
function inMemoryBackend() {
  return {
    kind: 'in_memory',
    config: {},
  };
}

/**
 * Create a Redis-backed adaptive state backend spec.
 *
 * Produces the backend descriptor expected by the adaptive plugin when state
 * should be shared or persisted through Redis.
 *
 * @param {string} url - Redis connection URL for the backend.
 * @param {string} [keyPrefix='nemo_relay:'] - Prefix applied to Redis keys.
 * @returns {object} An adaptive backend spec using Redis storage.
 * @remarks The default key prefix namespaces runtime records under
 * `nemo_relay:` unless a different prefix is supplied.
 */
function redisBackend(url, keyPrefix = 'nemo_relay:') {
  return {
    kind: 'redis',
    config: {
      url,
      key_prefix: keyPrefix,
    },
  };
}

/**
 * Create adaptive telemetry settings with runtime defaults applied.
 *
 * Merges caller-supplied overrides onto the built-in telemetry config shape
 * used by the adaptive plugin.
 *
 * @param {object} [config={}] - Partial telemetry settings to override.
 * @returns {object} A normalized adaptive telemetry config object.
 * @remarks An empty `learners` array is supplied by default so callers can
 * append learner names without checking for initialization first.
 */
function telemetryConfig(config = {}) {
  return {
    learners: [],
    ...config,
  };
}

function mergeDefined(defaults, config = {}) {
  const merged = { ...defaults };
  for (const [key, value] of Object.entries(config)) {
    if (value !== undefined) {
      merged[key] = value;
    }
  }
  return merged;
}

/**
 * Create topology-aware hint load-shedding settings with defaults applied.
 *
 * @param {object} [config={}] - Partial governor settings to override.
 * @returns {object} A normalized governor config object.
 */
function governorConfig(config = {}) {
  return mergeDefined(
    {
      enabled: false,
      epsilon: 1.0,
    },
    config,
  );
}

/**
 * Create topology-aware tool-plan drift detection settings with defaults applied.
 *
 * @param {object} [config={}] - Partial drift settings to override.
 * @returns {object} A normalized drift config object.
 */
function driftConfig(config = {}) {
  return mergeDefined(
    {
      enabled: false,
      threshold: 0.75,
    },
    config,
  );
}

/**
 * Create topological convergence detector settings with defaults applied.
 *
 * @param {object} [config={}] - Partial convergence settings to override.
 * @returns {object} A normalized convergence config object.
 */
function convergenceConfig(config = {}) {
  return mergeDefined(
    {
      enabled: false,
      epsilon: 0.001,
      stability_window: 3,
    },
    config,
  );
}

/**
 * Create adaptive hint-injection settings with defaults applied.
 *
 * Merges caller-supplied overrides onto the default config used by the
 * adaptive hints injector.
 *
 * @param {object} [config={}] - Partial adaptive hints settings to override.
 * @returns {object} A normalized adaptive hints config object.
 * @remarks By default the injector runs at priority `100`, preserves the rest
 * of the chain, and writes hints to `nvext.agent_hints`.
 */
function adaptiveHintsConfig(config = {}) {
  return {
    priority: 100,
    break_chain: false,
    inject_header: true,
    inject_body_path: 'nvext.agent_hints',
    ...config,
  };
}

/**
 * Create adaptive tool-parallelism settings with defaults applied.
 *
 * Merges caller-supplied overrides onto the scheduler config shape used by the
 * adaptive plugin's tool parallelism component.
 *
 * @param {object} [config={}] - Partial tool scheduling settings to override.
 * @returns {object} A normalized tool-parallelism config object.
 * @remarks The default mode is `observe_only`, so recommendations are produced
 * without changing execution behavior unless the caller opts in.
 */
function toolParallelismConfig(config = {}) {
  return {
    priority: 100,
    mode: 'observe_only',
    ...config,
  };
}

/**
 * Create adaptive cache-governor settings with defaults applied.
 *
 * Merges caller-supplied overrides onto the ACG config shape used by the
 * adaptive plugin's LLM execution intercept.
 *
 * @param {object} [config={}] - Partial ACG settings to override.
 * @returns {object} A normalized adaptive cache-governor config object.
 * @remarks Nested `stability_thresholds` values are defaulted individually so
 * callers can override only the thresholds they need.
 */
function acgConfig(config = {}) {
  const { stability_thresholds, ...rest } = config;
  return {
    provider: 'passthrough',
    observation_window: 100,
    priority: 50,
    stability_thresholds: {
      stable_threshold: 0.95,
      semi_stable_threshold: 0.5,
      min_observations_for_full_confidence: 20,
      ...stability_thresholds,
    },
    ...rest,
  };
}

/**
 * Wrap adaptive config as a top-level plugin component.
 *
 * Produces the plugin component entry that can be inserted directly
 * into `plugin.defaultConfig().components`.
 *
 * @param {object} config - Adaptive component configuration document.
 * @param {{ enabled?: boolean }} [options={}] - Optional component-level flags.
 * @returns {object} A plugin component spec for the adaptive plugin.
 * @remarks Setting `enabled` to `false` keeps the config in the plugin document
 * for validation while skipping runtime activation.
 */
function ComponentSpec(config, { enabled = true } = {}) {
  return plugin.ComponentSpec(ADAPTIVE_PLUGIN_KIND, config, {
    enabled,
  });
}

/**
 * Validate an adaptive config document without constructing a runtime.
 *
 * @param {object} config - Adaptive runtime configuration document.
 * @returns {object} A structured validation report with diagnostics.
 */
function validateConfig(config) {
  return lib.validateAdaptiveConfig(config);
}

/**
 * Build one adaptive cache telemetry event from normalized usage.
 *
 * @param {object} options - Cache telemetry event inputs.
 * @returns {object|null} A telemetry event, or `null` when usage lacks prompt tokens.
 */
function buildCacheTelemetryEvent(options) {
  return lib.buildCacheTelemetryEvent(options) ?? null;
}

/**
 * Set manual latency sensitivity on the current scope.
 *
 * @param {number} value - Positive sensitivity value to store on the current scope.
 * @returns {void} Nothing.
 */
function setLatencySensitivity(value) {
  return lib.setLatencySensitivity(value);
}

module.exports = {
  AdaptiveRuntime: lib.AdaptiveRuntime,
  ADAPTIVE_PLUGIN_KIND,
  defaultConfig,
  inMemoryBackend,
  redisBackend,
  telemetryConfig,
  governorConfig,
  driftConfig,
  convergenceConfig,
  adaptiveHintsConfig,
  toolParallelismConfig,
  acgConfig,
  ComponentSpec,
  validateConfig,
  buildCacheTelemetryEvent,
  setLatencySensitivity,
};
