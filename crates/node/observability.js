// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const plugin = require('./plugin.js');

const OBSERVABILITY_PLUGIN_KIND = 'observability';

/**
 * Create a default observability component config.
 *
 * @returns {object} The minimal observability config with schema version 2.
 */
function defaultConfig() {
  return {
    version: 2,
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
 * Create OTLP exporter settings for OpenTelemetry or OpenInference.
 *
 * @param {object} [config={}] - Partial OTLP settings to override.
 * @returns {object} A normalized OTLP config object.
 */
function otlpConfig(config = {}) {
  return {
    enabled: false,
    mark_projection: 'inherit',
    mark_exclude_names: ['llm.chunk'],
    attribute_mappings: [],
    transport: 'http_binary',
    headers: {},
    resource_attributes: {},
    service_name: 'nemo-relay',
    timeout_millis: 3000,
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
  otlpConfig,
  ComponentSpec,
};
