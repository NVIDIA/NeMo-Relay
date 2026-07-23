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
 * Create response-cache settings with defaults applied.
 *
 * Merges caller-supplied overrides onto the opt-in LLM response-cache config
 * shape (exact-match) used by the adaptive plugin. This is a section of
 * the adaptive component, not a standalone plugin kind.
 *
 * @param {object} [config={}] - Partial response-cache settings to override.
 * @returns {object} A normalized response-cache config object.
 * @remarks The default backend is in-memory; pass a `backend` (e.g.
 * `redisBackend(url)`) for a shared cache. `bypassRate` defaults to `0.0`
 * (always reuse / exact replay).
 */
function responseCacheConfig(config = {}) {
  const { backend, ...rest } = config;
  return {
    ttlSeconds: 3600,
    namespace: '',
    priority: 50,
    bypassRate: 0.0,
    cacheNondeterministic: true,
    keyStrategy: 'exact_request',
    headerAllowlist: [],
    skipKeys: [],
    backend: backend ?? inMemoryBackend(),
    ...rest,
  };
}

const RESPONSE_CACHE_PLUGIN_FIELDS = {
  ttlSeconds: 'ttl_seconds',
  bypassRate: 'bypass_rate',
  cacheNondeterministic: 'cache_nondeterministic',
  keyStrategy: 'key_strategy',
  headerAllowlist: 'header_allowlist',
  skipKeys: 'skip_keys',
};

function toPluginConfig(config) {
  const { responseCache, ...rest } = config;
  if (responseCache === undefined) return config;
  const serialized =
    responseCache !== null && typeof responseCache === 'object' && !Array.isArray(responseCache)
      ? Object.fromEntries(
          Object.entries(responseCache).map(([key, value]) => [RESPONSE_CACHE_PLUGIN_FIELDS[key] ?? key, value]),
        )
      : responseCache;
  return { ...rest, response_cache: serialized };
}

class AdaptiveRuntime extends lib.AdaptiveRuntime {
  constructor(config) {
    super(toPluginConfig(config));
  }
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
  return plugin.ComponentSpec(ADAPTIVE_PLUGIN_KIND, toPluginConfig(config), {
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
  return lib.validateAdaptiveConfig(toPluginConfig(config));
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
  AdaptiveRuntime,
  ADAPTIVE_PLUGIN_KIND,
  defaultConfig,
  inMemoryBackend,
  redisBackend,
  telemetryConfig,
  adaptiveHintsConfig,
  toolParallelismConfig,
  acgConfig,
  responseCacheConfig,
  ComponentSpec,
  validateConfig,
  buildCacheTelemetryEvent,
  setLatencySensitivity,
};
