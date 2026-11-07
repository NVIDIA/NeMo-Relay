// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ConfigReport, JsonObject } from '../binding.js';
import type { ComponentSpecFor } from './plugin.js';
import * as lib from './index.js';

export type { Json } from '../binding.js';

/** Plugin kind of the model pricing component. */
export const PRICING_PLUGIN_KIND = 'pricing';

/** Prompt-cache write price for a duration in seconds, or the default price without `duration`. */
export interface CacheWriteRate {
  duration?: number;
  per_million: number;
}

/** Token rates per million tokens. */
export interface TokenPricingRates {
  input_per_million: number;
  output_per_million: number;
  cache_read_per_million?: number;
  cache_write_rates?: CacheWriteRate[];
}

/** Prompt-cache accounting for a catalog entry. */
export interface PromptCachePricing {
  read_accounting?: 'included_in_prompt_tokens' | 'separate';
}

/** One prompt-token threshold tier. */
export interface TokenRateTier {
  min_prompt_tokens?: number;
  max_prompt_tokens?: number;
  rates: TokenPricingRates;
}

/** Token rate schedule selected by prompt-token thresholds. */
export interface PromptTokenThresholdRateSchedule {
  type: 'prompt_token_threshold';
  applies_to?: 'full_request';
  tiers: TokenRateTier[];
}

/** One model pricing catalog entry. */
export interface ModelPricing {
  provider: string;
  model_id: string;
  aliases?: string[];
  currency?: string;
  unit?: 'per_token' | 'per_request' | 'per_second' | 'gpu_hour';
  rates?: TokenPricingRates;
  rate_schedule?: PromptTokenThresholdRateSchedule | JsonObject;
  prompt_cache: PromptCachePricing;
  pricing_as_of: string;
  pricing_source: string;
}

/** Inline pricing catalog. */
export interface PricingCatalog {
  version?: number;
  entries: Array<ModelPricing | JsonObject>;
}

/** Pricing source holding an inline catalog. */
export interface InlineSource {
  type: 'inline';
  catalog: PricingCatalog | JsonObject;
}

/** Pricing source read from a JSON file. */
export interface FileSource {
  type: 'file';
  path: string;
}

/** Pricing source. */
export type PricingSource = InlineSource | FileSource | JsonObject;

/** Model pricing config. */
export interface PricingConfig {
  sources?: PricingSource[];
}

/** Creates a default model pricing config with no sources. */
export function defaultConfig(): PricingConfig {
  return {
    sources: [],
  };
}

/** Creates token rates with defaults applied. */
export function tokenRates(config: Partial<TokenPricingRates> = {}): TokenPricingRates {
  return {
    input_per_million: 0,
    output_per_million: 0,
    ...config,
  };
}

/** Creates prompt-cache accounting with defaults applied. */
export function promptCache(config: Partial<PromptCachePricing> = {}): PromptCachePricing {
  return {
    read_accounting: 'included_in_prompt_tokens',
    ...config,
  };
}

/** Creates one rate tier. */
export function tokenRateTier(
  rates: TokenPricingRates,
  config: Omit<Partial<TokenRateTier>, 'rates'> = {},
): TokenRateTier {
  return {
    ...config,
    rates,
  };
}

/** Creates a rate schedule from tiers. */
export function promptTokenThresholdRateSchedule(
  tiers: TokenRateTier[] = [],
  config: Omit<Partial<PromptTokenThresholdRateSchedule>, 'type' | 'tiers'> = {},
): PromptTokenThresholdRateSchedule {
  return {
    type: 'prompt_token_threshold',
    applies_to: 'full_request',
    tiers,
    ...config,
  };
}

/** Creates one catalog entry with defaults applied. */
export function catalogEntry(config: Omit<ModelPricing, 'prompt_cache'> & Partial<ModelPricing>): ModelPricing {
  const { prompt_cache: promptCacheConfig, ...entryConfig } = config;
  return {
    aliases: [],
    currency: 'USD',
    unit: 'per_token',
    ...entryConfig,
    prompt_cache: promptCache(promptCacheConfig),
  };
}

/** Creates an inline catalog from entries. */
export function inlineCatalog(
  entries: Array<ModelPricing | JsonObject> = [],
  config: Omit<Partial<PricingCatalog>, 'entries'> = {},
): PricingCatalog {
  return {
    version: 1,
    entries,
    ...config,
  };
}

/** Creates a pricing source from an inline catalog. */
export function inlineSource(catalog: PricingCatalog | JsonObject): InlineSource {
  return {
    type: 'inline',
    catalog,
  };
}

/** Creates a pricing source from a JSON file path. */
export function fileSource(path: string): FileSource {
  return {
    type: 'file',
    path,
  };
}

/** Creates a model pricing config from sources in precedence order. */
export function pricingConfig(sources: PricingSource[] = []): PricingConfig {
  return {
    sources,
  };
}

/** Wraps model pricing config as a plugin component. */
export function ComponentSpec(
  config: PricingConfig,
  { enabled = true }: { enabled?: boolean } = {},
): ComponentSpecFor<'pricing'> {
  return {
    kind: PRICING_PLUGIN_KIND,
    enabled,
    config,
  };
}

/** Validates model pricing config without activating it. */
export function validateConfig(config: PricingConfig | JsonObject): ConfigReport {
  return lib.validateExact({
    version: 1,
    components: [{ kind: PRICING_PLUGIN_KIND, enabled: true, config }],
  }).config;
}

/** Validates an untyped model pricing document and returns it as a config; throws on error diagnostics. */
export function parseConfig(document: JsonObject): PricingConfig {
  const errors = validateConfig(document).diagnostics.filter((diagnostic) => diagnostic.level === 'error');
  if (errors.length > 0) {
    throw new Error(`invalid model pricing config: ${errors.map((diagnostic) => diagnostic.message).join('; ')}`, {
      cause: errors,
    });
  }
  return document as PricingConfig;
}
