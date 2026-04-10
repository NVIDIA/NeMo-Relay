// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package adaptive

import nemo_flow "github.com/NVIDIA/NeMo-Flow/go/nemo_flow"

// UnsupportedBehavior controls how adaptive config validation handles unsupported input.
type UnsupportedBehavior = nemo_flow.UnsupportedBehavior

const (
	UnsupportedBehaviorIgnore = nemo_flow.UnsupportedBehaviorIgnore
	UnsupportedBehaviorWarn   = nemo_flow.UnsupportedBehaviorWarn
	UnsupportedBehaviorError  = nemo_flow.UnsupportedBehaviorError
)

// DiagnosticLevel is the severity of one adaptive validation diagnostic.
type DiagnosticLevel = nemo_flow.DiagnosticLevel

const (
	DiagnosticLevelWarning = nemo_flow.DiagnosticLevelWarning
	DiagnosticLevelError   = nemo_flow.DiagnosticLevelError
)

// Config is the canonical adaptive config document.
type Config = nemo_flow.AdaptiveConfig

// ComponentSpec wraps adaptive config as a top-level adaptive component.
type ComponentSpec = nemo_flow.AdaptiveComponentSpec

// StateConfig selects the adaptive state backend.
type StateConfig = nemo_flow.AdaptiveStateConfig

// BackendSpec selects the adaptive state backend kind and backend-specific config.
type BackendSpec = nemo_flow.AdaptiveBackendSpec

// TelemetryConfig configures built-in adaptive telemetry.
type TelemetryConfig = nemo_flow.TelemetryConfig

// AdaptiveHintsConfig configures built-in adaptive hint injection.
type AdaptiveHintsConfig = nemo_flow.AdaptiveHintsConfig

// ToolParallelismConfig configures built-in adaptive tool scheduling.
type ToolParallelismConfig = nemo_flow.ToolParallelismConfig

// PluginKind is the top-level plugin kind used by the adaptive component.
const PluginKind = nemo_flow.AdaptivePluginKind

// NewConfig returns a default adaptive config with version 1.
func NewConfig() Config {
	return nemo_flow.NewAdaptiveConfig()
}

// NewInMemoryBackend returns an in-memory adaptive backend spec.
func NewInMemoryBackend() BackendSpec {
	return nemo_flow.NewInMemoryAdaptiveBackend()
}

// NewRedisBackend returns a Redis adaptive backend spec.
func NewRedisBackend(url, keyPrefix string) BackendSpec {
	return nemo_flow.NewRedisAdaptiveBackend(url, keyPrefix)
}

// NewTelemetryConfig returns default adaptive telemetry settings.
func NewTelemetryConfig() TelemetryConfig {
	return nemo_flow.NewTelemetryConfig()
}

// NewAdaptiveHintsConfig returns default adaptive hints injection settings.
func NewAdaptiveHintsConfig() AdaptiveHintsConfig {
	return nemo_flow.NewAdaptiveHintsConfig()
}

// NewToolParallelismConfig returns default adaptive tool scheduling settings.
func NewToolParallelismConfig() ToolParallelismConfig {
	return nemo_flow.NewToolParallelismConfig()
}

// NewComponentSpec wraps adaptive config as an enabled top-level adaptive component.
func NewComponentSpec(config Config) ComponentSpec {
	return nemo_flow.NewAdaptiveComponentSpec(config)
}

// Component converts adaptive config directly into the shared plugin-host shape.
func Component(config Config) nemo_flow.PluginComponentSpec {
	return nemo_flow.AdaptiveComponent(config)
}
