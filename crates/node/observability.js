// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const plugin = require('./plugin.js');

const OBSERVABILITY_PLUGIN_KIND = 'observability';
const DEFAULT_COMPLETED_SPAN_CONTEXT_TTL_MILLIS = 60_000;

/**
 * Create a default observability component config.
 *
 * @returns {object} The minimal observability config with schema version 4.
 */
function defaultConfig() {
  return {
    version: 4,
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
    completed_span_context_ttl_millis: DEFAULT_COMPLETED_SPAN_CONTEXT_TTL_MILLIS,
    headers: {},
    header_env: {},
    resource_attributes: {},
    promote_metadata_prefixes: [],
    promote_resource_metadata_prefixes: [],
    ...config,
  };
}

/**
 * Create one signal-specific OpenTelemetry endpoint for logs or metrics.
 *
 * @param {object} config - Endpoint settings including required `endpoint`.
 * @returns {object} A normalized signal endpoint config object.
 */
function openTelemetrySignalEndpoint(config) {
  if (!config || typeof config !== 'object') {
    throw new TypeError('OpenTelemetry signal endpoint config is required');
  }
  if (typeof config.endpoint !== 'string' || config.endpoint.trim() === '') {
    throw new TypeError('OpenTelemetry signal endpoint must be a nonblank string');
  }
  return {
    transport: 'http_binary',
    headers: {},
    header_env: {},
    resource_attributes: {},
    service_name: 'unknown_service',
    instrumentation_scope: 'opentelemetry',
    timeout_millis: 3000,
    ...config,
  };
}

/**
 * Create OTLP log pipeline settings with defaults applied.
 *
 * @param {object} [config={}] - Partial log pipeline settings.
 * @returns {object} A normalized OpenTelemetry log section.
 */
function openTelemetryLogConfig(config = {}) {
  return {
    enabled: false,
    minimum_severity: 'info',
    max_queue_size: 2048,
    max_export_batch_size: 512,
    scheduled_delay_millis: 1000,
    completed_span_context_ttl_millis: DEFAULT_COMPLETED_SPAN_CONTEXT_TTL_MILLIS,
    ...config,
  };
}

/**
 * Create OTLP metric pipeline settings with defaults applied.
 *
 * @param {object} [config={}] - Partial metric pipeline settings.
 * @returns {object} A normalized OpenTelemetry metric section.
 */
function openTelemetryMetricConfig(config = {}) {
  return {
    enabled: false,
    export_interval_millis: 60000,
    temporality: 'cumulative',
    max_instruments: 256,
    cardinality_limit: 2000,
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
  openTelemetrySignalEndpoint,
  openTelemetryLogConfig,
  openTelemetryMetricConfig,
  openTelemetryConfig,
  ComponentSpec,
};
