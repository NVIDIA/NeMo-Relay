// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_flow

import "testing"

func TestNewAdaptiveConfigDefaults(t *testing.T) {
	config := NewAdaptiveConfig()
	if config.Version != 1 {
		t.Fatalf("expected version 1, got %d", config.Version)
	}
	if config.Telemetry != nil || config.AdaptiveHints != nil || config.ToolParallelism != nil {
		t.Fatal("expected adaptive feature sections to default to nil")
	}
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
