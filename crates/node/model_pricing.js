// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const plugin = require('./plugin.js');

const PRICING_PLUGIN_KIND = 'pricing';

/**
 * Create a default model pricing component config.
 *
 * @returns {object} The minimal model pricing config with no sources.
 */
function defaultConfig() {
  return {
    sources: [],
  };
}

/**
 * Create per-token model pricing rates with defaults applied.
 *
 * @param {object} [config={}] - Partial token-rate fields to override.
 * @returns {object} A normalized token model pricing rates object.
 */
function tokenRates(config = {}) {
  return {
    input_per_million: 0,
    output_per_million: 0,
    ...config,
  };
}

/**
 * Create prompt-cache accounting settings with defaults applied.
 *
 * @param {object} [config={}] - Partial prompt-cache settings to override.
 * @returns {object} A normalized prompt-cache config object.
 */
function promptCache(config = {}) {
  return {
    read_accounting: 'included_in_prompt_tokens',
    ...config,
  };
}

/**
 * Create one prompt-token threshold rate tier.
 *
 * @param {object} rates - Token rates selected by this tier.
 * @param {object} [config={}] - Optional threshold fields to include.
 * @returns {object} A normalized token rate tier object.
 */
function tokenRateTier(rates, config = {}) {
  return {
    ...config,
    rates,
  };
}

/**
 * Create a prompt-token threshold rate schedule.
 *
 * @param {object[]} [tiers=[]] - Ordered threshold tiers.
 * @param {object} [config={}] - Optional schedule fields to override.
 * @returns {object} A normalized rate schedule object.
 */
function promptTokenThresholdRateSchedule(tiers = [], config = {}) {
  return {
    type: 'prompt_token_threshold',
    applies_to: 'full_request',
    tiers,
    ...config,
  };
}

/**
 * Create one model pricing catalog entry with defaults applied.
 *
 * @param {object} config - Required and optional model pricing fields.
 * @returns {object} A normalized model pricing entry.
 */
function catalogEntry(config) {
  const { prompt_cache: promptCacheConfig, ...entryConfig } = config;
  return {
    aliases: [],
    currency: 'USD',
    unit: 'per_token',
    ...entryConfig,
    prompt_cache: promptCache(promptCacheConfig),
  };
}

/**
 * Create an inline model pricing catalog payload.
 *
 * @param {object[]} [entries=[]] - Pricing catalog entries.
 * @param {object} [config={}] - Optional catalog fields to override.
 * @returns {object} A normalized model pricing catalog object.
 */
function inlineCatalog(entries = [], config = {}) {
  return {
    version: 1,
    entries,
    ...config,
  };
}

/**
 * Create an inline model pricing source.
 *
 * @param {object} catalog - Pricing catalog payload.
 * @returns {object} A normalized inline source config.
 */
function inlineSource(catalog) {
  return {
    type: 'inline',
    catalog,
  };
}

/**
 * Create a file-backed model pricing source.
 *
 * @param {string} path - JSON catalog file path.
 * @returns {object} A normalized file source config.
 */
function fileSource(path) {
  return {
    type: 'file',
    path,
  };
}

/**
 * Create a model pricing config from ordered sources.
 *
 * @param {object[]} [sources=[]] - Pricing sources in precedence order.
 * @returns {object} A normalized model pricing config object.
 */
function pricingConfig(sources = []) {
  return {
    sources,
  };
}

/**
 * Wrap model pricing config as a top-level plugin component.
 *
 * @param {object} config - Model pricing component configuration document.
 * @param {{ enabled?: boolean }} [options={}] - Optional component-level flags.
 * @returns {object} A plugin component spec for the model pricing plugin.
 */
function ComponentSpec(config, { enabled = true } = {}) {
  return plugin.ComponentSpec(PRICING_PLUGIN_KIND, config, {
    enabled,
  });
}

/**
 * Validate a model pricing config document without activating it.
 *
 * @param {object} config - Model pricing component configuration document.
 * @returns {object} A structured validation report with diagnostics.
 */
function validateConfig(config) {
  return plugin.validate({
    version: 1,
    components: [ComponentSpec(config)],
  });
}

module.exports = {
  PRICING_PLUGIN_KIND,
  defaultConfig,
  tokenRates,
  promptCache,
  tokenRateTier,
  promptTokenThresholdRateSchedule,
  catalogEntry,
  inlineCatalog,
  inlineSource,
  fileSource,
  pricingConfig,
  ComponentSpec,
  validateConfig,
};
