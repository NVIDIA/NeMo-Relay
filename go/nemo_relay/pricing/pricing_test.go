// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package pricing

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
		t.Fatalf("unexpected pricing component kind: %#v", component)
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
	tier := NewTokenRateTier(NewTokenRates(3, 4))
	tier.MinPromptTokens = &minTokens
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

	config := NewConfig()
	component := Component(config)
	if component.Kind != PluginKind || !component.Enabled {
		t.Fatalf("unexpected pricing component: %#v", component)
	}
}
