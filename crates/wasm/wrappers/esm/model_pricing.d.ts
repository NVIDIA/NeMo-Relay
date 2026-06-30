// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Json, JsonObject } from './index';
import type { ComponentSpec as PluginComponentSpec, ConfigReport } from './plugin';

/** Plugin kind used by the model pricing component. */
export declare const PRICING_PLUGIN_KIND = 'pricing';

/** Token rates expressed per one million tokens. */
export interface TokenPricingRates {
  input_per_million: number;
  output_per_million: number;
  cache_read_per_million?: number;
  cache_write_per_million?: number;
}

/** Prompt-cache accounting settings for a catalog entry. */
export interface PromptCachePricing {
  read_accounting?: 'included_in_prompt_tokens' | 'separate';
}

/** One prompt-token threshold tier. */
export interface TokenRateTier {
  min_prompt_tokens?: number;
  max_prompt_tokens?: number;
  rates: TokenPricingRates;
}

/** Threshold-based token rate schedule. */
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

/** Inline model pricing catalog payload. */
export interface PricingCatalog {
  version?: number;
  entries: Array<ModelPricing | JsonObject>;
}

/** Inline model pricing source config. */
export interface InlineSource {
  type: 'inline';
  catalog: PricingCatalog | JsonObject;
}

/** File-backed model pricing source config. */
export interface FileSource {
  type: 'file';
  path: string;
}

/** Pricing source config. */
export type PricingSource = InlineSource | FileSource | JsonObject;

/** Canonical model pricing plugin config. */
export interface PricingConfig {
  sources?: PricingSource[];
}

/** Create a default model pricing component config. */
export declare function defaultConfig(): PricingConfig;
/** Create per-token model pricing rates with defaults applied. */
export declare function tokenRates(config?: Partial<TokenPricingRates>): TokenPricingRates;
/** Create prompt-cache accounting settings with defaults applied. */
export declare function promptCache(config?: Partial<PromptCachePricing>): PromptCachePricing;
/** Create one prompt-token threshold rate tier. */
export declare function tokenRateTier(
  rates: TokenPricingRates,
  config?: Omit<Partial<TokenRateTier>, 'rates'>,
): TokenRateTier;
/** Create a prompt-token threshold rate schedule. */
export declare function promptTokenThresholdRateSchedule(
  tiers?: TokenRateTier[],
  config?: Omit<Partial<PromptTokenThresholdRateSchedule>, 'type' | 'tiers'>,
): PromptTokenThresholdRateSchedule;
/** Create one model pricing catalog entry with defaults applied. */
export declare function catalogEntry(config: Omit<ModelPricing, 'prompt_cache'> & Partial<ModelPricing>): ModelPricing;
/** Create an inline model pricing catalog payload. */
export declare function inlineCatalog(
  entries?: Array<ModelPricing | JsonObject>,
  config?: Omit<Partial<PricingCatalog>, 'entries'>,
): PricingCatalog;
/** Create an inline model pricing source. */
export declare function inlineSource(catalog: PricingCatalog | JsonObject): InlineSource;
/** Create a file-backed model pricing source. */
export declare function fileSource(path: string): FileSource;
/** Create a model pricing config from ordered sources. */
export declare function pricingConfig(sources?: PricingSource[]): PricingConfig;
/** Wrap model pricing config as a top-level plugin component. */
export declare function ComponentSpec(
  config: PricingConfig | JsonObject,
  options?: {
    enabled?: boolean;
  },
): PluginComponentSpec;
/** Validate a model pricing config document without activating it. */
export declare function validateConfig(config: PricingConfig | JsonObject): ConfigReport;

export type { Json };
