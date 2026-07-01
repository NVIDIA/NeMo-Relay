// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package model_pricing

import (
	"encoding/json"
	"testing"
)

func TestPricingPackageHelpers(t *testing.T) {
	entry := NewModelPricing("test", "priced-model")
	entry.PricingAsOf = "2026-06-15"
	entry.PricingSource = "https://example.com/pricing"
	rates := NewTokenRates(1, 2)
	entry.Rates = &rates

	config := NewConfig()
	config.Sources = []SourceConfig{
		NewInlineSource(NewCatalog(entry)),
	}
	component := NewComponentSpec(config).PluginComponent()
	if component.Kind != PluginKind {
		t.Fatalf("unexpected model pricing component kind: %#v", component)
	}

	report, err := ValidateConfig(config)
	if err != nil {
		t.Fatalf("ValidateConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("expected clean report, got %#v", report.Diagnostics)
	}
}

func TestPricingPackageSourceAndRateHelpers(t *testing.T) {
	fileSource := NewFileSource("/tmp/pricing.json")
	payload, err := json.Marshal(fileSource)
	if err != nil {
		t.Fatalf("marshal file source: %v", err)
	}
	var parsedSource map[string]any
	if err := json.Unmarshal(payload, &parsedSource); err != nil {
		t.Fatalf("unmarshal file source: %v", err)
	}
	if parsedSource["type"] != "file" || parsedSource["path"] != "/tmp/pricing.json" {
		t.Fatalf("unexpected file source: %#v", parsedSource)
	}

	promptCache := NewPromptCacheConfig()
	if promptCache.ReadAccounting != CacheReadIncludedInPromptTokens {
		t.Fatalf("unexpected prompt cache defaults: %#v", promptCache)
	}

	minTokens := uint64(256)
	maxTokens := uint64(1024)
	tier := NewTokenRateTier(NewTokenRates(3, 4))
	tier.MinPromptTokens = &minTokens
	tier.MaxPromptTokens = &maxTokens
	schedule := NewPromptTokenThresholdRateSchedule(tier)
	schedulePayload, err := json.Marshal(schedule)
	if err != nil {
		t.Fatalf("marshal rate schedule: %v", err)
	}
	var parsedSchedule map[string]any
	if err := json.Unmarshal(schedulePayload, &parsedSchedule); err != nil {
		t.Fatalf("unmarshal rate schedule: %v", err)
	}
	if parsedSchedule["type"] != "prompt_token_threshold" || parsedSchedule["applies_to"] != "full_request" {
		t.Fatalf("unexpected rate schedule: %#v", parsedSchedule)
	}
	tiers, ok := parsedSchedule["tiers"].([]any)
	if !ok || len(tiers) != 1 {
		t.Fatalf("unexpected rate schedule tiers: %#v", parsedSchedule["tiers"])
	}
	parsedTier, ok := tiers[0].(map[string]any)
	if !ok {
		t.Fatalf("unexpected rate schedule tier: %#v", tiers[0])
	}
	if parsedTier["min_prompt_tokens"] != float64(minTokens) || parsedTier["max_prompt_tokens"] != float64(maxTokens) {
		t.Fatalf("unexpected token bounds: %#v", parsedTier)
	}
	rates, ok := parsedTier["rates"].(map[string]any)
	if !ok {
		t.Fatalf("unexpected tier rates: %#v", parsedTier["rates"])
	}
	if rates["input_per_million"] != float64(3) || rates["output_per_million"] != float64(4) {
		t.Fatalf("unexpected tier rate values: %#v", rates)
	}

	config := NewConfig()
	component := Component(config)
	if component.Kind != PluginKind || !component.Enabled {
		t.Fatalf("unexpected model pricing component: %#v", component)
	}
}
