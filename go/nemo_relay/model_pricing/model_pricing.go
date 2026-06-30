// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package model_pricing

import nemo_relay "github.com/NVIDIA/NeMo-Relay/go/nemo_relay"

// Config is the canonical model pricing config document.
type Config = nemo_relay.PricingConfig

// SourceConfig is implemented by model pricing source config structs.
type SourceConfig = nemo_relay.PricingSourceConfigurer

// InlineSourceConfig embeds a model pricing catalog directly in plugin config.
type InlineSourceConfig = nemo_relay.PricingInlineSourceConfig

// FileSourceConfig loads a model pricing catalog from a JSON file path.
type FileSourceConfig = nemo_relay.PricingFileSourceConfig

// Catalog is an inline model pricing catalog payload.
type Catalog = nemo_relay.PricingCatalog

// ModelPricing is one model pricing catalog entry.
type ModelPricing = nemo_relay.ModelPricing

// TokenRates expresses model rates per one million tokens.
type TokenRates = nemo_relay.TokenPricingRates

// PromptCacheConfig configures cache-read token accounting for a model pricing entry.
type PromptCacheConfig = nemo_relay.PromptCachePricing

// TokenRateTier is one prompt-token threshold tier.
type TokenRateTier = nemo_relay.TokenRateTier

// PromptTokenThresholdRateSchedule selects token rates by prompt token thresholds.
type PromptTokenThresholdRateSchedule = nemo_relay.PromptTokenThresholdRateSchedule

// ComponentSpec wraps model pricing config as a top-level model pricing component.
type ComponentSpec = nemo_relay.PricingComponentSpec

// ConfigReport is the validation or activation report for a plugin config.
type ConfigReport = nemo_relay.ConfigReport

// PluginKind is the top-level plugin kind used by the model pricing component.
const PluginKind = nemo_relay.PricingPluginKind

// Pricing unit constants accepted by model pricing catalog entries.
const (
	UnitPerToken   = nemo_relay.PricingUnitPerToken
	UnitPerRequest = nemo_relay.PricingUnitPerRequest
	UnitPerSecond  = nemo_relay.PricingUnitPerSecond
	UnitGPUHour    = nemo_relay.PricingUnitGPUHour
)

// Prompt-cache accounting constants accepted by model pricing catalog entries.
const (
	CacheReadIncludedInPromptTokens = nemo_relay.CacheReadAccountingIncludedInPromptTokens
	CacheReadSeparate               = nemo_relay.CacheReadAccountingSeparate
)

// NewConfig returns an empty model pricing config.
func NewConfig() Config {
	return nemo_relay.NewPricingConfig()
}

// NewInlineSource returns an inline model pricing catalog source.
func NewInlineSource(catalog Catalog) InlineSourceConfig {
	return nemo_relay.NewPricingInlineSource(catalog)
}

// NewFileSource returns a file-backed model pricing catalog source.
func NewFileSource(path string) FileSourceConfig {
	return nemo_relay.NewPricingFileSource(path)
}

// NewCatalog returns an inline model pricing catalog with version 1.
func NewCatalog(entries ...ModelPricing) Catalog {
	return nemo_relay.NewPricingCatalog(entries...)
}

// NewModelPricing returns a model pricing entry with catalog defaults applied.
func NewModelPricing(provider, modelID string) ModelPricing {
	return nemo_relay.NewModelPricing(provider, modelID)
}

// NewTokenRates returns per-token model rates.
func NewTokenRates(inputPerMillion, outputPerMillion float64) TokenRates {
	return nemo_relay.NewTokenPricingRates(inputPerMillion, outputPerMillion)
}

// NewPromptCacheConfig returns default prompt-cache accounting settings.
func NewPromptCacheConfig() PromptCacheConfig {
	return nemo_relay.NewPromptCachePricing()
}

// NewTokenRateTier returns a rate tier with the provided rates.
func NewTokenRateTier(rates TokenRates) TokenRateTier {
	return nemo_relay.NewTokenRateTier(rates)
}

// NewPromptTokenThresholdRateSchedule returns a full-request threshold schedule.
func NewPromptTokenThresholdRateSchedule(tiers ...TokenRateTier) PromptTokenThresholdRateSchedule {
	return nemo_relay.NewPromptTokenThresholdRateSchedule(tiers...)
}

// NewComponentSpec wraps model pricing config as an enabled top-level component.
func NewComponentSpec(config Config) ComponentSpec {
	return nemo_relay.NewPricingComponentSpec(config)
}

// Component converts model pricing config directly into the shared plugin shape.
func Component(config Config) nemo_relay.PluginComponentSpec {
	return nemo_relay.PricingComponent(config)
}

// ValidateConfig validates a model pricing config document without activating it.
func ValidateConfig(config Config) (ConfigReport, error) {
	return nemo_relay.ValidatePricingConfig(config)
}
