// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as plugin from './plugin.js';

export const OBSERVABILITY_PLUGIN_KIND = 'observability';

/**
 * Create a default observability component config.
 *
 * @returns {object} The minimal observability config with schema version 1.
 */
export function defaultConfig() {
  return {
    version: 1,
  };
}

/**
 * Create filesystem-backed ATOF JSONL settings with defaults applied.
 *
 * @param {object} [config={}] - Partial ATOF settings to override.
 * @returns {object} A normalized ATOF config object.
 */
export function atofConfig(config = {}) {
  return {
    enabled: false,
    mode: 'append',
    ...config,
  };
}

/**
 * Create per-agent ATIF trajectory settings with defaults applied.
 *
 * @param {object} [config={}] - Partial ATIF settings to override.
 * @returns {object} A normalized ATIF config object.
 */
export function atifConfig(config = {}) {
  return {
    enabled: false,
    agent_name: 'NeMo Flow',
    model_name: 'unknown',
    filename_template: 'nemo-flow-atif-{session_id}.json',
    ...config,
  };
}

/**
 * Create OTLP exporter settings for OpenTelemetry or OpenInference.
 *
 * @param {object} [config={}] - Partial OTLP settings to override.
 * @returns {object} A normalized OTLP config object.
 */
export function otlpConfig(config = {}) {
  return {
    enabled: false,
    transport: 'http_binary',
    headers: {},
    resource_attributes: {},
    service_name: 'nemo-flow',
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
export function ComponentSpec(config, { enabled = true } = {}) {
  return plugin.ComponentSpec(OBSERVABILITY_PLUGIN_KIND, config, {
    enabled,
  });
}
