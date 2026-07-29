// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const plugin = require('./plugin.js');

const OBSERVABILITY_PLUGIN_KIND = 'observability';

/**
 * Create a default observability component config.
 *
 * @returns {object} The minimal observability config with schema version 3.
 */
function defaultConfig() {
  return {
    version: 3,
  };
}

/**
 * Create multi-sink ATOF settings with defaults applied.
 *
 * @param {object} [config={}] - Partial ATOF settings to override.
 * @returns {object} A normalized ATOF config object.
 */
function atofConfig(config = {}) {
  return {
    enabled: false,
    ...config,
  };
}

/**
 * Create per-agent ATIF trajectory settings with defaults applied.
 *
 * @param {object} [config={}] - Partial ATIF settings to override.
 * @returns {object} A normalized ATIF config object.
 */
function atifConfig(config = {}) {
  return {
    enabled: false,
    agent_name: 'NeMo Relay',
    model_name: 'unknown',
    filename_template: 'nemo-relay-atif-{session_id}.json',
    ...config,
  };
}

/**
 * Create one typed OpenTelemetry endpoint.
 *
 * @param {object} config - Endpoint settings including required `type` and `endpoint`.
 * @returns {object} A normalized endpoint config object.
 */
function openTelemetryEndpoint(config) {
  if (!config || typeof config !== 'object') {
    throw new TypeError('OpenTelemetry endpoint config is required');
  }
  if (!['full', 'gen_ai', 'openinference'].includes(config.type)) {
    throw new TypeError('OpenTelemetry endpoint type must be "full", "gen_ai", or "openinference"');
  }
  if (typeof config.endpoint !== 'string' || config.endpoint.trim() === '') {
    throw new TypeError('OpenTelemetry endpoint must be a nonblank string');
  }
  return {
    transport: 'http_binary',
    service_name: 'unknown_service',
    instrumentation_scope: 'opentelemetry',
    timeout_millis: 3000,
    headers: {},
    header_env: {},
    resource_attributes: {},
    ...config,
  };
}

/**
 * Create multi-endpoint OpenTelemetry settings.
 *
 * @param {object} [config={}] - Partial section settings.
 * @returns {object} A normalized OpenTelemetry section.
 */
function openTelemetryConfig(config = {}) {
  return {
    enabled: false,
    endpoints: [],
    ...config,
  };
}

/**
 * Wrap observability config as a top-level plugin component.
 *
 * @param {object} config - Observability component configuration document.
 * @param {{ enabled?: boolean }} [options={}] - Optional component-level flags.
 * @returns {object} A plugin component spec for the observability plugin.
 */
function ComponentSpec(config, { enabled = true } = {}) {
  return plugin.ComponentSpec(OBSERVABILITY_PLUGIN_KIND, config, {
    enabled,
  });
}

module.exports = {
  OBSERVABILITY_PLUGIN_KIND,
  defaultConfig,
  atofConfig,
  atifConfig,
  openTelemetryEndpoint,
  openTelemetryConfig,
  ComponentSpec,
};
