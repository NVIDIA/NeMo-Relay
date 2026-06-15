// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"testing"
)

func TestPricingConfigHelpers(t *testing.T) {
	rates := NewTokenPricingRates(1, 2)
	cacheRead := 0.25
	rates.CacheReadPerMillion = &cacheRead

	entry := NewModelPricing("test", "priced-model")
	entry.PricingAsOf = "2026-06-15"
	entry.PricingSource = "https://example.com/pricing"
	entry.Rates = &rates

	config := NewPricingConfig()
	config.Sources = []PricingSourceConfig{
		NewPricingInlineSource(NewPricingCatalog(entry)),
		NewPricingFileSource("/tmp/pricing.json"),
	}

	component := NewPricingComponentSpec(config).PluginComponent()
	if component.Kind != PricingPluginKind || !component.Enabled {
		t.Fatalf("unexpected pricing component wrapper: %#v", component)
	}
	if len(component.Config["sources"].([]any)) != 2 {
		t.Fatalf("expected two pricing sources, got %#v", component.Config)
	}

	payload, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("marshal pricing config: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(payload, &parsed); err != nil {
		t.Fatalf("unmarshal pricing config: %v", err)
	}
	sources := parsed["sources"].([]any)
	inlineSource := sources[0].(map[string]any)
	fileSource := sources[1].(map[string]any)
	if inlineSource["type"] != "inline" || fileSource["type"] != "file" {
		t.Fatalf("expected source discriminators, got %#v", sources)
	}
	if fileSource["path"] != "/tmp/pricing.json" {
		t.Fatalf("unexpected file source path: %#v", fileSource)
	}
}

func TestPricingRateScheduleHelper(t *testing.T) {
	emptySchedule := NewPromptTokenThresholdRateSchedule()
	emptyPayload, err := json.Marshal(emptySchedule)
	if err != nil {
		t.Fatalf("marshal empty schedule: %v", err)
	}
	var emptyParsed map[string]any
	if err := json.Unmarshal(emptyPayload, &emptyParsed); err != nil {
		t.Fatalf("unmarshal empty schedule: %v", err)
	}
	if tiers, ok := emptyParsed["tiers"].([]any); !ok || len(tiers) != 0 {
		t.Fatalf("expected empty tiers array, got %#v", emptyParsed["tiers"])
	}

	minTokens := uint64(128)
	tier := NewTokenRateTier(NewTokenPricingRates(1, 2))
	tier.MinPromptTokens = &minTokens
	schedule := NewPromptTokenThresholdRateSchedule(tier)

	payload, err := json.Marshal(schedule)
	if err != nil {
		t.Fatalf("marshal schedule: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(payload, &parsed); err != nil {
		t.Fatalf("unmarshal schedule: %v", err)
	}
	if parsed["type"] != "prompt_token_threshold" || parsed["applies_to"] != "full_request" {
		t.Fatalf("unexpected schedule: %#v", parsed)
	}
}

func TestValidatePricingConfig(t *testing.T) {
	entry := NewModelPricing("test", "priced-model")
	entry.PricingAsOf = "2026-06-15"
	entry.PricingSource = "https://example.com/pricing"
	rates := NewTokenPricingRates(1, 2)
	entry.Rates = &rates

	report, err := ValidatePricingConfig(PricingConfig{
		Sources: []PricingSourceConfig{
			NewPricingInlineSource(NewPricingCatalog(entry)),
		},
	})
	if err != nil {
		t.Fatalf("ValidatePricingConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("expected clean report, got %#v", report.Diagnostics)
	}

	rates.InputPerMillion = -1
	entry.Rates = &rates
	invalid, err := ValidatePricingConfig(PricingConfig{
		Sources: []PricingSourceConfig{
			NewPricingInlineSource(NewPricingCatalog(entry)),
		},
	})
	if err != nil {
		t.Fatalf("ValidatePricingConfig invalid failed: %v", err)
	}
	foundInvalidConfig := false
	for _, diagnostic := range invalid.Diagnostics {
		if diagnostic.Code == "pricing.invalid_config" {
			foundInvalidConfig = true
			break
		}
	}
	if !foundInvalidConfig {
		t.Fatalf("expected pricing invalid diagnostic, got %#v", invalid.Diagnostics)
	}
}

func TestListKindsIncludesPricing(t *testing.T) {
	kinds, err := ListPluginKinds()
	if err != nil {
		t.Fatalf("ListPluginKinds failed: %v", err)
	}
	for _, kind := range kinds {
		if kind == PricingPluginKind {
			return
		}
	}
	t.Fatalf("expected %q in plugin kinds: %#v", PricingPluginKind, kinds)
}
