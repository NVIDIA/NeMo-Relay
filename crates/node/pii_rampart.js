// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const plugin = require('./plugin.js');

const RAMPART_PII_PLUGIN_KIND = 'pii_rampart';
const RAMPART_MODEL_ID = 'nationaldesignstudio/rampart';
const RAMPART_MODEL_REVISION = 'b1993e4e68b082835b80ffc65acc03325ea2e501';

/**
 * Create Rampart PII settings with runtime defaults applied.
 *
 * At least one exact path or path pattern is required because Relay does not
 * send unselected observability fields to the model.
 *
 * @param {string} modelPath - Absolute path to the pinned Rampart snapshot.
 * @param {object} config - Partial settings including explicit target selectors.
 * @returns {object} A normalized Rampart PII config object.
 */
function defaultConfig(modelPath, config) {
  const hasTargetPaths = Array.isArray(config?.target_paths) && config.target_paths.length > 0;
  const hasTargetPathPatterns =
    Array.isArray(config?.target_path_patterns) && config.target_path_patterns.length > 0;
  if (!hasTargetPaths && !hasTargetPathPatterns) {
    throw new TypeError('Rampart PII config requires target_paths or target_path_patterns');
  }
  return {
    version: 1,
    input: true,
    output: true,
    mark: true,
    tool_input: true,
    tool_output: true,
    priority: 100,
    target_paths: [],
    target_path_patterns: [],
    min_score: 0.4,
    excluded_labels: [],
    replacement: '[REDACTED]',
    max_windows_per_payload: 4,
    inference_batch_size: 16,
    ...config,
    model_path: modelPath,
  };
}

/**
 * Wrap Rampart PII config as a top-level plugin component.
 *
 * @param {object} config - Rampart PII component configuration.
 * @param {{ enabled?: boolean }} [options={}] - Optional component flags.
 * @returns {object} A shared plugin component spec.
 */
function ComponentSpec(config, { enabled = true } = {}) {
  return plugin.ComponentSpec(RAMPART_PII_PLUGIN_KIND, config, { enabled });
}

/**
 * Validate Rampart PII configuration without loading model files.
 *
 * @param {object} config - Rampart PII component configuration.
 * @returns {object} A structured validation report with diagnostics.
 */
function validateConfig(config) {
  return plugin.validate({
    version: 1,
    components: [ComponentSpec(config)],
  });
}

module.exports = {
  RAMPART_PII_PLUGIN_KIND,
  RAMPART_MODEL_ID,
  RAMPART_MODEL_REVISION,
  defaultConfig,
  ComponentSpec,
  validateConfig,
};
