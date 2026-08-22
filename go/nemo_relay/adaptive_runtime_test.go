// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

const (
	adaptiveRuntimeClosedMessage    = "adaptive runtime is nil or shut down"
	expectedCleanReportMsg          = "expected clean report, got %#v"
	forcedAdaptiveMarshalFailure    = "forced adaptive JSON marshal failure"
	marshalFailedMsg                = "marshal failed: %v"
	newAdaptiveRuntimeFailedMsg     = "NewAdaptiveRuntime failed: %v"
	responseCacheTestNamespace      = "go-harness"
	testAgentID                     = "go-agent"
	unmarshalFailedMsg              = "unmarshal failed: %v"
	validateAdaptiveConfigFailedMsg = "ValidateAdaptiveConfig failed: %v"
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
		t.Fatalf(validateAdaptiveConfigFailedMsg, err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf(expectedCleanReportMsg, report.Diagnostics)
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
	assertResponseCacheConstructorDefaults(t, rc)
	rc.Namespace = responseCacheTestNamespace
	rc.CacheNondeterministic = true
	rc.KeyStrategy = ResponseCacheKeyStrategyLogical
	rc.Backend = &backend
	assertResponseCacheJSONSurface(t, rc)
	assertResponseCacheValidation(t, rc)
}

func assertResponseCacheConstructorDefaults(t *testing.T, config ResponseCacheConfig) {
	t.Helper()
	if config.TTLSeconds == nil || *config.TTLSeconds != 3600 {
		t.Fatalf("constructor TTL default mismatch: %#v", config.TTLSeconds)
	}
	if config.Priority == nil || *config.Priority != 50 {
		t.Fatalf("constructor priority default mismatch: %#v", config.Priority)
	}
	if config.KeyStrategy != ResponseCacheKeyStrategyExactRequest {
		t.Fatalf("constructor key strategy default mismatch: %#v", config.KeyStrategy)
	}
}

func assertResponseCacheJSONSurface(t *testing.T, responseCache ResponseCacheConfig) {
	t.Helper()
	config := NewAdaptiveConfig()
	config.ResponseCache = &responseCache
	payload, err := json.Marshal(config)
	if err != nil {
		t.Fatalf(marshalFailedMsg, err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf(unmarshalFailedMsg, err)
	}
	section, ok := decoded["response_cache"].(map[string]any)
	if !ok {
		t.Fatalf("response_cache missing from marshaled config: %s", payload)
	}
	if section["namespace"] != responseCacheTestNamespace {
		t.Fatalf("response_cache fields not preserved: %#v", section)
	}
	if _, ok := section["skip_keys"]; ok {
		t.Fatalf("response_cache must not expose arbitrary key omission: %#v", section)
	}
	if v, ok := section["cache_nondeterministic"].(bool); !ok || !v {
		t.Fatalf("explicit cache_nondeterministic=true was not preserved: %#v", section["cache_nondeterministic"])
	}
	if section["key_strategy"] != string(ResponseCacheKeyStrategyLogical) {
		t.Fatalf("logical key strategy was not preserved: %#v", section["key_strategy"])
	}
	if b, ok := section["backend"].(map[string]any); !ok || b["kind"] != "in_memory" {
		t.Fatalf("backend not preserved: %#v", section["backend"])
	}
}

func assertResponseCacheValidation(t *testing.T, responseCache ResponseCacheConfig) {
	t.Helper()
	config := NewAdaptiveConfig()
	config.ResponseCache = &responseCache
	report, err := ValidateAdaptiveConfig(config)
	if err != nil {
		t.Fatalf(validateAdaptiveConfigFailedMsg, err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf(expectedCleanReportMsg, report.Diagnostics)
	}

	bad := NewResponseCacheConfig()
	bad.Namespace = responseCacheTestNamespace
	bad.BypassRate = 2.0
	badConfig := NewAdaptiveConfig()
	badConfig.ResponseCache = &bad
	badReport, err := ValidateAdaptiveConfig(badConfig)
	if err != nil {
		t.Fatalf("ValidateAdaptiveConfig (invalid bypass_rate) returned error: %v", err)
	}
	if !hasAdaptiveDiagnostic(badReport, "response_cache.invalid_bypass_rate") {
		t.Fatalf("expected response_cache.invalid_bypass_rate diagnostic, got %#v", badReport.Diagnostics)
	}
}

func TestResponseCacheToolsConfigReachesTypedSurface(t *testing.T) {
	rc := NewResponseCacheConfig()
	rc.Namespace = "tool-cache-go-test"
	tools := NewResponseCacheToolsConfig()
	if tools.Priority == nil || *tools.Priority != 100 {
		t.Fatalf("constructor tools priority default mismatch: %#v", tools.Priority)
	}
	if tools.CacheErrors {
		t.Fatalf("constructor cache_errors default mismatch: %#v", tools.CacheErrors)
	}
	tools.Enabled = true
	tools.CacheErrors = true
	zero := int32(0)
	classVersion := "class-v1"
	tools.Priority = &zero
	tools.Classes = map[string]ResponseCacheToolClass{
		"read_only": {
			Cacheable:   true,
			ToolVersion: &classVersion,
			Members:     []string{"docs_lookup"},
		},
	}
	rc.Tools = &tools

	config := NewAdaptiveConfig()
	config.ResponseCache = &rc

	payload, err := json.Marshal(config)
	if err != nil {
		t.Fatalf(marshalFailedMsg, err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf(unmarshalFailedMsg, err)
	}
	rcSection, ok := decoded["response_cache"].(map[string]any)
	if !ok {
		t.Fatalf("response_cache missing from marshaled config: %s", payload)
	}
	toolsSection, ok := rcSection["tools"].(map[string]any)
	if !ok {
		t.Fatalf("tools missing from marshaled response_cache: %#v", rcSection)
	}
	if enabled, _ := toolsSection["enabled"].(bool); !enabled {
		t.Fatalf("tools.enabled not preserved: %#v", toolsSection)
	}
	if cacheErrors, ok := toolsSection["cache_errors"].(bool); !ok || !cacheErrors {
		t.Fatalf("tools.cache_errors not preserved: %#v", toolsSection)
	}
	if priority, ok := toolsSection["priority"].(float64); !ok || priority != 0 {
		t.Fatalf("explicit tools.priority = 0 must survive marshal: %#v", toolsSection)
	}
	classes, ok := toolsSection["classes"].(map[string]any)
	if !ok || classes["read_only"] == nil {
		t.Fatalf("tools.classes not preserved: %#v", toolsSection)
	}
	readOnly, ok := classes["read_only"].(map[string]any)
	if !ok || readOnly["tool_version"] != "class-v1" {
		t.Fatalf("class tool_version not preserved: %#v", classes["read_only"])
	}

	report, err := ValidateAdaptiveConfig(config)
	if err != nil {
		t.Fatalf("ValidateAdaptiveConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("expected clean report, got %#v", report.Diagnostics)
	}

}

func TestResponseCacheConfigPreservesOmissionAndExplicitZero(t *testing.T) {
	t.Run("partial config delegates to Rust defaults", testPartialResponseCacheConfig)
	t.Run("missing namespace remains invalid", testMissingResponseCacheNamespace)
	t.Run("explicit TTL zero remains invalid", testExplicitZeroResponseCacheTTL)
	t.Run("explicit priority zero remains valid", testExplicitZeroResponseCachePriority)
}

func marshalResponseCacheConfig(t *testing.T, responseCache ResponseCacheConfig) map[string]any {
	t.Helper()
	payload, err := json.Marshal(responseCache)
	if err != nil {
		t.Fatalf(marshalFailedMsg, err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf(unmarshalFailedMsg, err)
	}
	return decoded
}

func validateResponseCacheConfig(t *testing.T, responseCache ResponseCacheConfig) ConfigReport {
	t.Helper()
	config := NewAdaptiveConfig()
	config.ResponseCache = &responseCache
	report, err := ValidateAdaptiveConfig(config)
	if err != nil {
		t.Fatalf(validateAdaptiveConfigFailedMsg, err)
	}
	return report
}

func hasAdaptiveDiagnostic(report ConfigReport, code string) bool {
	for _, diagnostic := range report.Diagnostics {
		if diagnostic.Code == code {
			return true
		}
	}
	return false
}

func testPartialResponseCacheConfig(t *testing.T) {
	responseCache := ResponseCacheConfig{Namespace: "dev"}
	decoded := marshalResponseCacheConfig(t, responseCache)
	if _, ok := decoded["ttl_seconds"]; ok {
		t.Fatalf("partial config must omit ttl_seconds: %#v", decoded)
	}
	if _, ok := decoded["priority"]; ok {
		t.Fatalf("partial config must omit priority: %#v", decoded)
	}
	if report := validateResponseCacheConfig(t, responseCache); len(report.Diagnostics) != 0 {
		t.Fatalf("expected Rust defaults to validate cleanly, got %#v", report.Diagnostics)
	}
}

func testMissingResponseCacheNamespace(t *testing.T) {
	report := validateResponseCacheConfig(t, ResponseCacheConfig{})
	if !hasAdaptiveDiagnostic(report, "response_cache.missing_namespace") {
		t.Fatalf("expected response_cache.missing_namespace, got %#v", report.Diagnostics)
	}
}

func testExplicitZeroResponseCacheTTL(t *testing.T) {
	zero := uint64(0)
	responseCache := ResponseCacheConfig{TTLSeconds: &zero, Namespace: "dev"}
	if got := marshalResponseCacheConfig(t, responseCache)["ttl_seconds"]; got != float64(0) {
		t.Fatalf("explicit ttl_seconds=0 was not preserved: %#v", got)
	}
	report := validateResponseCacheConfig(t, responseCache)
	if !hasAdaptiveDiagnostic(report, "response_cache.invalid_ttl") {
		t.Fatalf("expected response_cache.invalid_ttl, got %#v", report.Diagnostics)
	}
}

func testExplicitZeroResponseCachePriority(t *testing.T) {
	zero := int32(0)
	responseCache := ResponseCacheConfig{Priority: &zero, Namespace: "dev"}
	decoded := marshalResponseCacheConfig(t, responseCache)
	if got := decoded["priority"]; got != float64(0) {
		t.Fatalf("explicit priority=0 was not preserved: %#v", got)
	}
	if _, ok := decoded["ttl_seconds"]; ok {
		t.Fatalf("unconfigured ttl_seconds must remain omitted: %#v", decoded)
	}
	if report := validateResponseCacheConfig(t, responseCache); len(report.Diagnostics) != 0 {
		t.Fatalf("expected priority=0 to validate cleanly, got %#v", report.Diagnostics)
	}
}

func TestAdaptiveRuntimeLifecycleRejectsUseAfterShutdown(t *testing.T) {
	runtime, err := NewAdaptiveRuntime(testAdaptiveRuntimeConfig("openai"))
	if err != nil {
		t.Fatalf(newAdaptiveRuntimeFailedMsg, err)
	}

	if err := runtime.Shutdown(); err != nil {
		t.Fatalf("Shutdown failed: %v", err)
	}

	assertAdaptiveRuntimeClosed(t, runtime)
}

func assertAdaptiveRuntimeClosed(t *testing.T, runtime *AdaptiveRuntime) {
	t.Helper()

	for _, test := range []struct {
		name string
		err  error
	}{
		{name: "Register", err: runtime.Register()},
		{name: "Deregister", err: runtime.Deregister()},
		{name: "WaitForIdle", err: runtime.WaitForIdle()},
		{name: "BindScope", err: runtime.BindScope(nil)},
	} {
		if test.err == nil || !strings.Contains(test.err.Error(), adaptiveRuntimeClosedMessage) {
			t.Fatalf("expected %s to reject a shut down runtime, got %v", test.name, test.err)
		}
	}

	if _, err := runtime.Report(); err == nil || !strings.Contains(err.Error(), adaptiveRuntimeClosedMessage) {
		t.Fatalf("expected Report to reject a shut down runtime, got %v", err)
	}
	if _, err := runtime.BuildCacheRequestFacts(CacheRequestFactsInput{}); err == nil || !strings.Contains(err.Error(), adaptiveRuntimeClosedMessage) {
		t.Fatalf("expected BuildCacheRequestFacts to reject a shut down runtime, got %v", err)
	}
	if err := runtime.Shutdown(); err == nil || !strings.Contains(err.Error(), adaptiveRuntimeClosedMessage) {
		t.Fatalf("expected repeated Shutdown to reject a shut down runtime, got %v", err)
	}
}

func TestAdaptiveRuntimePublicHelpersPropagateJSONMarshalFailures(t *testing.T) {
	oldMarshal := jsonMarshal
	t.Cleanup(func() { jsonMarshal = oldMarshal })

	jsonMarshal = func(any) ([]byte, error) {
		return nil, errors.New(forcedAdaptiveMarshalFailure)
	}

	if _, err := ValidateAdaptiveConfig(NewAdaptiveConfig()); err == nil || !strings.Contains(err.Error(), forcedAdaptiveMarshalFailure) {
		t.Fatalf("expected ValidateAdaptiveConfig to return marshal failure, got %v", err)
	}
	if _, err := NewAdaptiveRuntime(NewAdaptiveConfig()); err == nil || !strings.Contains(err.Error(), forcedAdaptiveMarshalFailure) {
		t.Fatalf("expected NewAdaptiveRuntime to return marshal failure, got %v", err)
	}
	if _, err := BuildCacheTelemetryEvent(CacheTelemetryEventInput{}); err == nil || !strings.Contains(err.Error(), forcedAdaptiveMarshalFailure) {
		t.Fatalf("expected BuildCacheTelemetryEvent to return marshal failure, got %v", err)
	}
}
