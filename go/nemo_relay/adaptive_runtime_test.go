// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"testing"
)

const (
	testAgentID                 = "go-agent"
	newAdaptiveRuntimeFailedMsg = "NewAdaptiveRuntime failed: %v"
)

func testAdaptiveRuntimeConfig(provider string) AdaptiveConfig {
	config := NewAdaptiveConfig()
	config.AgentID = "go-adaptive-" + provider
	config.State = &AdaptiveStateConfig{
		Backend: NewInMemoryAdaptiveBackend(),
	}
	config.Acg = &AcgConfig{
		Provider: provider,
	}
	return config
}

func uint64Ptr(value uint64) *uint64 {
	return &value
}

func TestValidateAdaptiveConfigAndOwnedRuntime(t *testing.T) {
	report, err := ValidateAdaptiveConfig(NewAdaptiveConfig())
	if err != nil {
		t.Fatalf("ValidateAdaptiveConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("expected clean report, got %#v", report.Diagnostics)
	}

	runtime, err := NewAdaptiveRuntime(NewAdaptiveConfig())
	if err != nil {
		t.Fatalf(newAdaptiveRuntimeFailedMsg, err)
	}
	defer runtime.Shutdown()
	if err := runtime.Register(); err != nil {
		t.Fatalf("Register failed: %v", err)
	}
	runtime.WaitForIdle()
	if report, err := runtime.Report(); err != nil || len(report.Diagnostics) != 0 {
		t.Fatalf("unexpected runtime report: %#v err=%v", report, err)
	}
	if err := runtime.Deregister(); err != nil {
		t.Fatalf("Deregister failed: %v", err)
	}
	if err := runtime.Shutdown(); err != nil {
		t.Fatalf("Shutdown failed: %v", err)
	}
}

func TestBuildCacheTelemetryEvent(t *testing.T) {
	event, err := BuildCacheTelemetryEvent(CacheTelemetryEventInput{
		Provider:  "openai",
		RequestID: "00000000-0000-0000-0000-000000000401",
		Usage: &CacheUsage{
			PromptTokens:     uint64Ptr(100),
			CompletionTokens: uint64Ptr(10),
			CacheReadTokens:  uint64Ptr(25),
		},
		AgentID:         testAgentID,
		TemplateVersion: "v1",
		ToolsetHash:     "tools",
		ModelFamily:     "gpt",
		TenantScope:     "tenant",
		Timestamp:       "2026-06-15T00:00:00Z",
	})
	if err != nil {
		t.Fatalf("BuildCacheTelemetryEvent failed: %v", err)
	}
	if event == nil {
		t.Fatal("expected cache telemetry event")
	}
	if event.Provider != "openai" || event.CacheReadTokens != 25 || event.TotalPromptTokens != 100 || event.HitRate != 0.25 {
		t.Fatalf("unexpected event: %#v", event)
	}
	if event.AgentIdentity.AgentID != testAgentID {
		t.Fatalf("unexpected agent identity: %#v", event.AgentIdentity)
	}

	empty, err := BuildCacheTelemetryEvent(CacheTelemetryEventInput{
		Provider:  "openai",
		RequestID: "00000000-0000-0000-0000-000000000402",
		Usage: &CacheUsage{
			CompletionTokens: uint64Ptr(10),
		},
		AgentID:         testAgentID,
		TemplateVersion: "v1",
		ToolsetHash:     "tools",
		ModelFamily:     "gpt",
		TenantScope:     "tenant",
	})
	if err != nil {
		t.Fatalf("BuildCacheTelemetryEvent without prompt tokens failed: %v", err)
	}
	if empty != nil {
		t.Fatalf("expected nil event without prompt tokens, got %#v", empty)
	}
}

func TestAdaptiveRuntimeBuildCacheRequestFacts(t *testing.T) {
	runtime, err := NewAdaptiveRuntime(testAdaptiveRuntimeConfig("openai"))
	if err != nil {
		t.Fatalf(newAdaptiveRuntimeFailedMsg, err)
	}
	if err := runtime.Register(); err != nil {
		t.Fatalf("Register failed: %v", err)
	}
	defer func() {
		_ = runtime.Shutdown()
	}()

	annotated, err := json.Marshal(map[string]any{
		"messages": []map[string]any{
			{
				"role":    "user",
				"content": "Find sources about caching",
			},
		},
		"model": "gpt-4.1-mini",
	})
	if err != nil {
		t.Fatalf("marshal annotated request: %v", err)
	}

	facts, err := runtime.BuildCacheRequestFacts(CacheRequestFactsInput{
		Provider:         "openai",
		RequestID:        "00000000-0000-0000-0000-000000000403",
		AnnotatedRequest: annotated,
		AgentID:          "go-adaptive-openai",
	})
	if err != nil {
		t.Fatalf("BuildCacheRequestFacts failed: %v", err)
	}
	if facts == nil {
		t.Fatal("expected cache request facts")
	}
	if facts.Provider != "openai" || facts.StablePrefixLength != 0 || len(facts.MissingFacts) != 1 {
		t.Fatalf("unexpected cache request facts: %#v", facts)
	}
	if facts.MissingFacts[0] != "acg_stability_unavailable" {
		t.Fatalf("unexpected missing facts: %#v", facts.MissingFacts)
	}
}

func TestAdaptiveRuntimeBindScopeRejectsNilScope(t *testing.T) {
	runtime, err := NewAdaptiveRuntime(testAdaptiveRuntimeConfig("openai"))
	if err != nil {
		t.Fatalf(newAdaptiveRuntimeFailedMsg, err)
	}
	defer runtime.Shutdown()

	if runtime.BindScope(nil) == nil {
		t.Fatal("expected BindScope to reject nil scope")
	}
}

func TestSetLatencySensitivityRejectsInvalidValue(t *testing.T) {
	if SetLatencySensitivity(0) == nil {
		t.Fatal("expected SetLatencySensitivity(0) to fail")
	}
}

func TestResponseCacheConfigReachesTypedSurface(t *testing.T) {
	backend := NewInMemoryResponseCacheBackend()
	rc := NewResponseCacheConfig()
	rc.Namespace = "go-harness"
	rc.CacheNondeterministic = false // must survive marshal (no omitempty)
	rc.Backend = &backend

	config := NewAdaptiveConfig()
	config.ResponseCache = &rc

	// 1. The typed AdaptiveConfig must carry response_cache through json.Marshal.
	//    The bug this guards is the struct silently DROPPING the section because it
	//    enumerates fields by name with no response_cache field and no catch-all.
	payload, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}
	section, ok := decoded["response_cache"].(map[string]any)
	if !ok {
		t.Fatalf("response_cache missing from marshaled config: %s", payload)
	}
	if section["namespace"] != "go-harness" {
		t.Fatalf("response_cache fields not preserved: %#v", section)
	}
	if v, ok := section["cache_nondeterministic"].(bool); !ok || v {
		t.Fatalf("explicit cache_nondeterministic=false was not preserved: %#v", section["cache_nondeterministic"])
	}
	if b, ok := section["backend"].(map[string]any); !ok || b["kind"] != "in_memory" {
		t.Fatalf("backend not preserved: %#v", section["backend"])
	}

	// 2. A valid section validates clean through the FFI -> Rust adaptive validator.
	report, err := ValidateAdaptiveConfig(config)
	if err != nil {
		t.Fatalf("ValidateAdaptiveConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("expected clean report, got %#v", report.Diagnostics)
	}

	// 3. An invalid section produces a response_cache diagnostic. This proves the
	//    section is actually validated end-to-end (a dropped section would yield no
	//    diagnostic at all), not merely carried in the struct.
	bad := NewResponseCacheConfig()
	bad.BypassRate = 2.0
	badConfig := NewAdaptiveConfig()
	badConfig.ResponseCache = &bad
	badReport, err := ValidateAdaptiveConfig(badConfig)
	if err != nil {
		t.Fatalf("ValidateAdaptiveConfig (invalid bypass_rate) returned error: %v", err)
	}
	found := false
	for _, d := range badReport.Diagnostics {
		if d.Code == "response_cache.invalid_bypass_rate" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected response_cache.invalid_bypass_rate diagnostic, got %#v", badReport.Diagnostics)
	}
}
