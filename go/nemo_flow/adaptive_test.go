// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_flow

import "testing"

type invalidAdaptiveConfigJSON struct{}

func (invalidAdaptiveConfigJSON) MarshalJSON() ([]byte, error) {
	return []byte("{"), nil
}

func TestNewAdaptiveConfigDefaults(t *testing.T) {
	config := NewAdaptiveConfig()
	if config.Version != 1 {
		t.Fatalf("expected version 1, got %d", config.Version)
	}
	if config.Telemetry != nil || config.AdaptiveHints != nil || config.ToolParallelism != nil {
		t.Fatal("expected adaptive feature sections to default to nil")
	}
}

func TestAdaptiveHelperConstructors(t *testing.T) {
	redis := NewRedisAdaptiveBackend("redis://127.0.0.1:6379", "custom:")
	if redis.Kind != "redis" {
		t.Fatalf("expected redis backend kind, got %q", redis.Kind)
	}
	if redis.Config["url"] != "redis://127.0.0.1:6379" {
		t.Fatalf("unexpected redis url: %#v", redis.Config)
	}
	if redis.Config["key_prefix"] != "custom:" {
		t.Fatalf("unexpected redis key prefix: %#v", redis.Config)
	}

	telemetry := NewTelemetryConfig()
	if telemetry.SubscriberName != "" || len(telemetry.Learners) != 0 {
		t.Fatalf("expected empty telemetry defaults, got %#v", telemetry)
	}

	hints := NewAdaptiveHintsConfig()
	if hints.Priority != 100 || !hints.InjectHeader || hints.InjectBodyPath != "nvext.agent_hints" {
		t.Fatalf("unexpected adaptive hints defaults: %#v", hints)
	}

	parallelism := NewToolParallelismConfig()
	if parallelism.Priority != 100 || parallelism.Mode != "observe_only" {
		t.Fatalf("unexpected tool parallelism defaults: %#v", parallelism)
	}

	component := NewAdaptiveComponentSpec(NewAdaptiveConfig()).PluginComponent()
	if component.Kind != "adaptive" || !component.Enabled {
		t.Fatalf("unexpected adaptive component wrapper: %#v", component)
	}

	zeroComponent := NewAdaptiveComponentSpec(AdaptiveConfig{}).PluginComponent()
	if zeroComponent.Config == nil {
		t.Fatal("expected zero-value adaptive config to normalize to an empty config map")
	}

	nilConfig := mustConfigMap(nil)
	if len(nilConfig) != 0 {
		t.Fatalf("expected nil config to normalize to an empty map, got %#v", nilConfig)
	}
}

func TestMustConfigMapPanicBranches(t *testing.T) {
	assertPanics := func(name string, fn func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("expected %s to panic", name)
			}
		}()
		fn()
	}

	assertPanics("marshal failure", func() {
		_ = mustConfigMap(map[string]any{"unsupported": make(chan int)})
	})

	assertPanics("unmarshal failure", func() {
		_ = mustConfigMap(invalidAdaptiveConfigJSON{})
	})
}

func TestValidatePluginConfigWarnsMissingStateForTelemetry(t *testing.T) {
	report, err := ValidatePluginConfig(PluginConfig{
		Version: 1,
		Components: []PluginComponentSpec{
			AdaptiveComponent(AdaptiveConfig{
				Version:   1,
				Telemetry: &TelemetryConfig{},
			}),
		},
	})
	if err != nil {
		t.Fatalf("ValidatePluginConfig failed: %v", err)
	}
	if len(report.Diagnostics) == 0 {
		t.Fatal("expected compatibility diagnostics")
	}
	if report.Diagnostics[0].Code != "adaptive.section_disabled_missing_state" {
		t.Fatalf("unexpected diagnostic code: %q", report.Diagnostics[0].Code)
	}
}

func TestConfigureAdaptiveComponentLifecycle(t *testing.T) {
	config := NewAdaptiveConfig()
	config.State = &AdaptiveStateConfig{
		Backend: NewInMemoryAdaptiveBackend(),
	}
	config.Telemetry = &TelemetryConfig{
		Learners: []string{"latency_sensitivity"},
	}
	config.AdaptiveHints = &AdaptiveHintsConfig{
		Priority:       100,
		InjectHeader:   true,
		InjectBodyPath: "nvext.agent_hints",
	}
	config.ToolParallelism = &ToolParallelismConfig{
		Priority: 100,
		Mode:     "observe_only",
	}

	report, err := ValidatePluginConfig(PluginConfig{
		Version:    1,
		Components: []PluginComponentSpec{AdaptiveComponent(config)},
	})
	if err != nil {
		t.Fatalf("ValidatePluginConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("expected clean report, got %#v", report.Diagnostics)
	}

	configureReport, err := InitializePlugins(PluginConfig{
		Version:    1,
		Components: []PluginComponentSpec{AdaptiveComponent(config)},
	})
	if err != nil {
		t.Fatalf("InitializePlugins failed: %v", err)
	}
	if len(configureReport.Diagnostics) != 0 {
		t.Fatalf("expected clean configure report, got %#v", configureReport.Diagnostics)
	}

	activeReport, err := ActivePluginReport()
	if err != nil {
		t.Fatalf("ActivePluginReport failed: %v", err)
	}
	if activeReport == nil || len(activeReport.Diagnostics) != 0 {
		t.Fatalf("expected active report with no diagnostics, got %#v", activeReport)
	}

	if err := ClearPluginConfiguration(); err != nil {
		t.Fatalf("ClearPluginConfiguration failed: %v", err)
	}
}
