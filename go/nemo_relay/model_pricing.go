// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import "encoding/json"

// PricingPluginKind is the top-level plugin kind used by the model pricing component.
const PricingPluginKind = "pricing"

// Pricing unit constants accepted by model pricing catalog entries.
const (
	PricingUnitPerToken   = "per_token"
	PricingUnitPerRequest = "per_request"
	PricingUnitPerSecond  = "per_second"
	PricingUnitGPUHour    = "gpu_hour"
)

// Prompt-cache accounting constants accepted by model pricing catalog entries.
const (
	CacheReadAccountingIncludedInPromptTokens = "included_in_prompt_tokens"
	CacheReadAccountingSeparate               = "separate"
)

// PricingConfig is the canonical Go shape for the model pricing plugin config document.
type PricingConfig struct {
	Sources []PricingSourceConfigurer `json:"sources,omitempty"`
}

// PricingSourceConfigurer is implemented by model pricing source config structs.
type PricingSourceConfigurer interface {
	pricingSourceConfig()
}

// PricingInlineSourceConfig embeds a model pricing catalog directly in plugin config.
type PricingInlineSourceConfig struct {
	Catalog PricingCatalog `json:"catalog"`
}

func (PricingInlineSourceConfig) pricingSourceConfig() {
	// Marker method: inline sources are serialized by MarshalJSON.
}

// MarshalJSON serializes the inline source with the canonical type discriminator.
func (source PricingInlineSourceConfig) MarshalJSON() ([]byte, error) {
	type alias PricingInlineSourceConfig
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
	}{
		Type:  "inline",
		alias: alias(source),
	})
}

// PricingFileSourceConfig loads a model pricing catalog from a JSON file path.
type PricingFileSourceConfig struct {
	Path string `json:"path"`
}

func (PricingFileSourceConfig) pricingSourceConfig() {
	// Marker method: file sources are serialized by MarshalJSON.
}

// MarshalJSON serializes the file source with the canonical type discriminator.
func (source PricingFileSourceConfig) MarshalJSON() ([]byte, error) {
	type alias PricingFileSourceConfig
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
	}{
		Type:  "file",
		alias: alias(source),
	})
}

// PricingCatalog is an inline model pricing catalog payload.
type PricingCatalog struct {
	Version uint32         `json:"version,omitempty"`
	Entries []ModelPricing `json:"entries,omitempty"`
}

// ModelPricing is one model pricing catalog entry.
type ModelPricing struct {
	Provider      string                            `json:"provider"`
	ModelID       string                            `json:"model_id"`
	Aliases       []string                          `json:"aliases,omitempty"`
	Currency      string                            `json:"currency,omitempty"`
	Unit          string                            `json:"unit,omitempty"`
	Rates         *TokenPricingRates                `json:"rates,omitempty"`
	RateSchedule  *PromptTokenThresholdRateSchedule `json:"rate_schedule,omitempty"`
	PromptCache   PromptCachePricing                `json:"prompt_cache"`
	PricingAsOf   string                            `json:"pricing_as_of"`
	PricingSource string                            `json:"pricing_source"`
}

// TokenPricingRates expresses model rates per one million tokens.
type TokenPricingRates struct {
	InputPerMillion      float64  `json:"input_per_million"`
	OutputPerMillion     float64  `json:"output_per_million"`
	CacheReadPerMillion  *float64 `json:"cache_read_per_million,omitempty"`
	CacheWritePerMillion *float64 `json:"cache_write_per_million,omitempty"`
}

// PromptCachePricing configures cache-read token accounting for a model pricing entry.
type PromptCachePricing struct {
	ReadAccounting string `json:"read_accounting"`
}

// TokenRateTier is one prompt-token threshold tier.
type TokenRateTier struct {
	MinPromptTokens *uint64           `json:"min_prompt_tokens,omitempty"`
	MaxPromptTokens *uint64           `json:"max_prompt_tokens,omitempty"`
	Rates           TokenPricingRates `json:"rates"`
}

// PromptTokenThresholdRateSchedule selects token rates by full-request prompt token thresholds.
type PromptTokenThresholdRateSchedule struct {
	AppliesTo string          `json:"applies_to,omitempty"`
	Tiers     []TokenRateTier `json:"tiers"`
}

// MarshalJSON serializes the rate schedule with the canonical type discriminator.
func (schedule PromptTokenThresholdRateSchedule) MarshalJSON() ([]byte, error) {
	type alias PromptTokenThresholdRateSchedule
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
	}{
		Type:  "prompt_token_threshold",
		alias: alias(schedule),
	})
}

// PricingComponentSpec wraps one model pricing config as a top-level plugin component.
type PricingComponentSpec struct {
	Enabled    bool          `json:"enabled,omitempty"`
	Config     PricingConfig `json:"config"`
	enabledSet bool
}

// NewPricingConfig returns an empty model pricing config.
func NewPricingConfig() PricingConfig {
	return PricingConfig{Sources: []PricingSourceConfigurer{}}
}

// NewPricingInlineSource returns an inline model pricing catalog source.
func NewPricingInlineSource(catalog PricingCatalog) PricingInlineSourceConfig {
	return PricingInlineSourceConfig{Catalog: catalog}
}

// NewPricingFileSource returns a file-backed model pricing catalog source.
func NewPricingFileSource(path string) PricingFileSourceConfig {
	return PricingFileSourceConfig{Path: path}
}

// NewPricingCatalog returns an inline model pricing catalog with version 1.
func NewPricingCatalog(entries ...ModelPricing) PricingCatalog {
	return PricingCatalog{
		Version: 1,
		Entries: entries,
	}
}

// NewModelPricing returns a model pricing entry with catalog defaults applied.
func NewModelPricing(provider, modelID string) ModelPricing {
	return ModelPricing{
		Provider:      provider,
		ModelID:       modelID,
		Aliases:       []string{},
		Currency:      "USD",
		Unit:          PricingUnitPerToken,
		PromptCache:   NewPromptCachePricing(),
		PricingAsOf:   "",
		PricingSource: "",
	}
}

// NewTokenPricingRates returns per-token model rates.
func NewTokenPricingRates(inputPerMillion, outputPerMillion float64) TokenPricingRates {
	return TokenPricingRates{
		InputPerMillion:  inputPerMillion,
		OutputPerMillion: outputPerMillion,
	}
}

// NewPromptCachePricing returns default prompt-cache accounting settings.
func NewPromptCachePricing() PromptCachePricing {
	return PromptCachePricing{
		ReadAccounting: CacheReadAccountingIncludedInPromptTokens,
	}
}

// NewTokenRateTier returns a rate tier with the provided rates.
func NewTokenRateTier(rates TokenPricingRates) TokenRateTier {
	return TokenRateTier{Rates: rates}
}

// NewPromptTokenThresholdRateSchedule returns a full-request prompt-token threshold schedule.
func NewPromptTokenThresholdRateSchedule(tiers ...TokenRateTier) PromptTokenThresholdRateSchedule {
	if tiers == nil {
		tiers = []TokenRateTier{}
	}
	return PromptTokenThresholdRateSchedule{
		AppliesTo: "full_request",
		Tiers:     tiers,
	}
}

// NewPricingComponentSpec wraps model pricing config as an enabled top-level component.
func NewPricingComponentSpec(config PricingConfig) PricingComponentSpec {
	return PricingComponentSpec{
		Enabled:    true,
		Config:     config,
		enabledSet: true,
	}
}

// SetEnabled explicitly sets whether this model pricing component is enabled.
func (spec *PricingComponentSpec) SetEnabled(enabled bool) {
	spec.Enabled = enabled
	spec.enabledSet = true
}

// WithEnabled returns a copy with an explicitly present enabled value.
func (spec PricingComponentSpec) WithEnabled(enabled bool) PricingComponentSpec {
	spec.SetEnabled(enabled)
	return spec
}

// MarshalJSON preserves omitted-default and explicit-false enabled semantics.
func (spec PricingComponentSpec) MarshalJSON() ([]byte, error) {
	return marshalComponentWrapper(spec.Enabled, spec.enabledSet, spec.Config)
}

// UnmarshalJSON records whether enabled was explicitly present.
func (spec *PricingComponentSpec) UnmarshalJSON(payload []byte) error {
	enabled, enabledSet, config, err := unmarshalComponentWrapper[PricingConfig](payload)
	if err != nil {
		return err
	}
	*spec = PricingComponentSpec{Enabled: enabled, Config: config, enabledSet: enabledSet}
	return nil
}

// PluginComponent converts the model pricing component wrapper into the shared plugin shape.
func (spec PricingComponentSpec) PluginComponent() PluginComponentSpec {
	component := PluginComponentSpec{
		Kind:    PricingPluginKind,
		Enabled: spec.Enabled,
		Config:  mustConfigMap(spec.Config),
	}
	if spec.enabledSet || spec.Enabled {
		component.SetEnabled(spec.Enabled)
	}
	return component
}

// PricingComponent converts model pricing config directly into a shared plugin component.
func PricingComponent(config PricingConfig) PluginComponentSpec {
	return NewPricingComponentSpec(config).PluginComponent()
}

// ValidatePricingConfig validates a model pricing config document without activating it.
func ValidatePricingConfig(config PricingConfig) (ConfigReport, error) {
	return ValidatePluginConfig(PluginConfig{
		Version:    1,
		Components: []PluginComponentSpec{PricingComponent(config)},
	})
}
