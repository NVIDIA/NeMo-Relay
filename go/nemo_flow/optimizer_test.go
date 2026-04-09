// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_flow

import (
	"strings"
	"testing"
)

func TestNewOptimizerConfigDefaults(t *testing.T) {
	config := NewOptimizerConfig()
	if config.Version != 1 {
		t.Fatalf("expected version 1, got %d", config.Version)
	}
	if len(config.Components) != 0 {
		t.Fatalf("expected empty component list, got %d entries", len(config.Components))
	}
}

func TestValidateOptimizerConfigWarnsUnknownComponent(t *testing.T) {
	report, err := ValidateOptimizerConfig(OptimizerConfig{
		Version: 1,
		Components: []OptimizerComponentSpec{
			{
				Kind:    "future_component",
				Enabled: true,
				Config:  map[string]any{},
			},
		},
	})
	if err != nil {
		t.Fatalf("ValidateOptimizerConfig failed: %v", err)
	}
	if len(report.Diagnostics) == 0 {
		t.Fatal("expected compatibility diagnostics")
	}
	if report.Diagnostics[0].Code != "optimizer.unknown_component" {
		t.Fatalf("expected optimizer.unknown_component, got %q", report.Diagnostics[0].Code)
	}
}

func TestNewOptimizerRuntimeRejectsStrictUnknownComponent(t *testing.T) {
	_, err := NewOptimizerRuntime(OptimizerConfig{
		Version: 1,
		Policy: &OptimizerConfigPolicy{
			UnknownComponent: UnsupportedBehaviorError,
		},
		Components: []OptimizerComponentSpec{
			{
				Kind:    "future_component",
				Enabled: true,
				Config:  map[string]any{},
			},
		},
	})
	if err == nil {
		t.Fatal("expected config error")
	}
	if !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("expected unsupported component error, got %v", err)
	}
}

func TestOptimizerRuntimeLifecycle(t *testing.T) {
	config := NewOptimizerConfig()
	config.State = &OptimizerStateConfig{
		Backend: NewInMemoryOptimizerBackend(),
	}
	config.Components = []OptimizerComponentSpec{
		TelemetryComponent(TelemetryComponentConfig{
			Learners: []string{"latency_sensitivity"},
		}),
		DynamoHintsComponent(NewDynamoHintsComponentConfig()),
		ToolParallelismComponent(NewToolParallelismComponentConfig()),
	}

	report, err := ValidateOptimizerConfig(config)
	if err != nil {
		t.Fatalf("ValidateOptimizerConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("expected clean report, got %#v", report.Diagnostics)
	}

	runtime, err := NewOptimizerRuntime(config)
	if err != nil {
		t.Fatalf("NewOptimizerRuntime failed: %v", err)
	}
	defer runtime.Close()

	runtimeReport, err := runtime.Report()
	if err != nil {
		t.Fatalf("Report failed: %v", err)
	}
	if len(runtimeReport.Diagnostics) != 0 {
		t.Fatalf("expected clean runtime report, got %#v", runtimeReport.Diagnostics)
	}

	if err := runtime.Register(); err != nil {
		t.Fatalf("Register failed: %v", err)
	}
	if err := runtime.Deregister(); err != nil {
		t.Fatalf("Deregister failed: %v", err)
	}
	if err := runtime.Shutdown(); err != nil {
		t.Fatalf("Shutdown failed: %v", err)
	}
}
