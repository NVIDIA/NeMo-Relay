// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as plugin from './plugin.js';

export const PII_REDACTION_PLUGIN_KIND = 'pii_redaction';

/**
 * Create a default PII redaction component config.
 *
 * @returns {object} The minimal PII redaction config with schema version 1.
 */
export function defaultConfig() {
  return {
    version: 1,
    mode: 'builtin',
    input: true,
    output: true,
    tool_input: true,
    tool_output: true,
    priority: 100,
  };
}

/**
 * Create deterministic built-in redaction backend settings with defaults applied.
 *
 * @param {object} [config={}] - Partial built-in settings to override.
 * @returns {object} A normalized built-in backend config object.
 */
export function builtinConfig(config = {}) {
  return {
    action: 'remove',
    ...config,
  };
}

/**
 * Create future local-model backend settings with defaults applied.
 *
 * @param {object} [config={}] - Partial local-model settings to override.
 * @returns {object} A normalized local-model backend config object.
 */
export function localModelConfig(config = {}) {
  return {
    ...config,
  };
}

/**
 * Wrap PII redaction config as a top-level plugin component.
 *
 * @param {object} config - PII redaction component configuration document.
 * @param {{ enabled?: boolean }} [options={}] - Optional component-level flags.
 * @returns {object} A plugin component spec for the PII redaction plugin.
 */
export function ComponentSpec(config, { enabled = true } = {}) {
  return plugin.ComponentSpec(PII_REDACTION_PLUGIN_KIND, config, {
    enabled,
  });
}
