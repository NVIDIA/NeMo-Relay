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

func TestRedisBackendAndComponentSpecBuilders(t *testing.T) {
	backend := NewRedisBackend("redis://127.0.0.1:6379", "adaptive:")
	if backend.Kind != "redis" {
		t.Fatalf("expected redis backend kind, got %q", backend.Kind)
	}
	if backend.Config["url"] != "redis://127.0.0.1:6379" {
		t.Fatalf("expected backend url to round-trip, got %#v", backend.Config["url"])
	}
	if backend.Config["key_prefix"] != "adaptive:" {
		t.Fatalf("expected backend key prefix to round-trip, got %#v", backend.Config["key_prefix"])
	}

	config := NewConfig()
	config.State = &StateConfig{Backend: backend}
	component := NewComponentSpec(config)
	if !component.Enabled {
		t.Fatalf("expected adaptive component to be enabled")
	}
	if component.Config.Version != 1 {
		t.Fatalf("expected adaptive component config version 1, got %d", component.Config.Version)
	}

	wrapped := Component(config)
	if wrapped.Kind != PluginKind {
		t.Fatalf("expected wrapped adaptive component kind %q, got %q", PluginKind, wrapped.Kind)
	}
}
