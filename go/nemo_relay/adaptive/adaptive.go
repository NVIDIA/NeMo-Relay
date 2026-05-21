// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package adaptive

import nemo_relay "github.com/NVIDIA/NeMo-Relay/go/nemo_relay"

// UnsupportedBehavior controls how adaptive config validation handles unsupported input.
type UnsupportedBehavior = nemo_relay.UnsupportedBehavior

const (
	UnsupportedBehaviorIgnore = nemo_relay.UnsupportedBehaviorIgnore
	UnsupportedBehaviorWarn   = nemo_relay.UnsupportedBehaviorWarn
	UnsupportedBehaviorError  = nemo_relay.UnsupportedBehaviorError
)

// DiagnosticLevel is the severity of one adaptive validation diagnostic.
type DiagnosticLevel = nemo_relay.DiagnosticLevel

const (
	DiagnosticLevelWarning = nemo_relay.DiagnosticLevelWarning
	DiagnosticLevelError   = nemo_relay.DiagnosticLevelError
)

// Config is the canonical adaptive config document.
type Config = nemo_relay.AdaptiveConfig

// ComponentSpec wraps adaptive config as a top-level adaptive component.
type ComponentSpec = nemo_relay.AdaptiveComponentSpec

// StateConfig selects the adaptive state backend.
type StateConfig = nemo_relay.AdaptiveStateConfig

// BackendSpec selects the adaptive state backend kind and backend-specific config.
type BackendSpec = nemo_relay.AdaptiveBackendSpec

// TelemetryConfig configures built-in adaptive telemetry.
type TelemetryConfig = nemo_relay.TelemetryConfig

// AdaptiveHintsConfig configures built-in adaptive hint injection.
type AdaptiveHintsConfig = nemo_relay.AdaptiveHintsConfig

// ToolParallelismConfig configures built-in adaptive tool scheduling.
type ToolParallelismConfig = nemo_relay.ToolParallelismConfig

// AcgStabilityThresholds configures ACG prompt-stability classification.
type AcgStabilityThresholds = nemo_relay.AcgStabilityThresholds

// AcgConfig configures the adaptive cache governor.
type AcgConfig = nemo_relay.AcgConfig

// PluginKind is the top-level plugin kind used by the adaptive component.
const PluginKind = nemo_relay.AdaptivePluginKind

// NewConfig returns a default adaptive config with version 1.
func NewConfig() Config {
	return nemo_relay.NewAdaptiveConfig()
}

// NewInMemoryBackend returns an in-memory adaptive backend spec.
func NewInMemoryBackend() BackendSpec {
	return nemo_relay.NewInMemoryAdaptiveBackend()
}

// NewRedisBackend returns a Redis adaptive backend spec.
func NewRedisBackend(url, keyPrefix string) BackendSpec {
	return nemo_relay.NewRedisAdaptiveBackend(url, keyPrefix)
}

// NewTelemetryConfig returns default adaptive telemetry settings.
func NewTelemetryConfig() TelemetryConfig {
	return nemo_relay.NewTelemetryConfig()
}

// NewAdaptiveHintsConfig returns default adaptive hints injection settings.
func NewAdaptiveHintsConfig() AdaptiveHintsConfig {
	return nemo_relay.NewAdaptiveHintsConfig()
}

// NewToolParallelismConfig returns default adaptive tool scheduling settings.
func NewToolParallelismConfig() ToolParallelismConfig {
	return nemo_relay.NewToolParallelismConfig()
}

// NewAcgStabilityThresholds returns default ACG stability thresholds.
func NewAcgStabilityThresholds() AcgStabilityThresholds {
	return nemo_relay.NewAcgStabilityThresholds()
}

// NewAcgConfig returns default adaptive cache governor settings.
func NewAcgConfig() AcgConfig {
	return nemo_relay.NewAcgConfig()
}

// NewComponentSpec wraps adaptive config as an enabled top-level adaptive component.
func NewComponentSpec(config Config) ComponentSpec {
	return nemo_relay.NewAdaptiveComponentSpec(config)
}

// Component converts adaptive config directly into the shared plugin shape.
func Component(config Config) nemo_relay.PluginComponentSpec {
	return nemo_relay.AdaptiveComponent(config)
}
