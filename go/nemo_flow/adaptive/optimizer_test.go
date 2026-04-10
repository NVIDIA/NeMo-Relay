// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package adaptive

import (
	nemo_flow "github.com/NVIDIA/NeMo-Flow/go/nemo_flow"
	"testing"
)

func TestConfigBuilders(t *testing.T) {
	config := NewConfig()
	if config.Version != 1 {
		t.Fatalf("expected version 1, got %d", config.Version)
	}

	config.State = &StateConfig{Backend: NewInMemoryBackend()}
	telemetry := NewTelemetryConfig()
	telemetry.Learners = []string{"latency_sensitivity"}
	config.Telemetry = &telemetry
	adaptiveHints := NewAdaptiveHintsConfig()
	config.AdaptiveHints = &adaptiveHints
	toolParallelism := NewToolParallelismConfig()
	config.ToolParallelism = &toolParallelism

	report, err := nemo_flow.ValidatePluginConfig(nemo_flow.PluginConfig{
		Version:    1,
		Components: []nemo_flow.PluginComponentSpec{Component(config)},
	})
	if err != nil {
		t.Fatalf("ValidatePluginConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("expected no diagnostics, got %+v", report.Diagnostics)
	}
}
