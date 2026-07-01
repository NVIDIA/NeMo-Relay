// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import "encoding/json"

// AdaptivePluginKind is the top-level plugin kind used by the adaptive component.
const AdaptivePluginKind = "adaptive"

// AdaptiveConfig is the canonical Go shape for the adaptive plugin config document.
type AdaptiveConfig struct {
	Version         uint32                 `json:"version,omitempty"`
	AgentID         string                 `json:"agent_id,omitempty"`
	State           *AdaptiveStateConfig   `json:"state,omitempty"`
	Telemetry       *TelemetryConfig       `json:"telemetry,omitempty"`
	AdaptiveHints   *AdaptiveHintsConfig   `json:"adaptive_hints,omitempty"`
	ToolParallelism *ToolParallelismConfig `json:"tool_parallelism,omitempty"`
	Acg             *AcgConfig             `json:"acg,omitempty"`
	Convergence     *ConvergenceConfig     `json:"convergence,omitempty"`
	Policy          *ConfigPolicy          `json:"policy,omitempty"`
}

// AdaptiveStateConfig selects the adaptive state backend.
type AdaptiveStateConfig struct {
	Backend AdaptiveBackendSpec `json:"backend"`
}

// AdaptiveBackendSpec selects the backend kind and backend-specific config.
type AdaptiveBackendSpec struct {
	Kind   string         `json:"kind"`
	Config map[string]any `json:"config,omitempty"`
}

// TelemetryConfig configures the built-in adaptive telemetry subscriber and learners.
type TelemetryConfig struct {
	SubscriberName string   `json:"subscriber_name,omitempty"`
	Learners       []string `json:"learners,omitempty"`
}

// AdaptiveHintsConfig configures built-in LLM request hint injection.
type AdaptiveHintsConfig struct {
	Priority       int32           `json:"priority,omitempty"`
	BreakChain     bool            `json:"break_chain,omitempty"`
	InjectHeader   bool            `json:"inject_header,omitempty"`
	InjectBodyPath string          `json:"inject_body_path,omitempty"`
	Governor       *GovernorConfig `json:"governor,omitempty"`
}

// ToolParallelismConfig configures built-in adaptive tool scheduling.
type ToolParallelismConfig struct {
	Priority int32        `json:"priority,omitempty"`
	Mode     string       `json:"mode,omitempty"`
	Drift    *DriftConfig `json:"drift,omitempty"`
}

// GovernorConfig configures topology-aware hint load shedding.
type GovernorConfig struct {
	Enabled bool    `json:"enabled,omitempty"`
	Epsilon float64 `json:"epsilon,omitempty"`
}

// DriftConfig configures topology-aware stale-plan invalidation.
type DriftConfig struct {
	Enabled   bool    `json:"enabled,omitempty"`
	Threshold float64 `json:"threshold,omitempty"`
}

// ConvergenceConfig configures topological convergence detection.
type ConvergenceConfig struct {
	Enabled         bool    `json:"enabled,omitempty"`
	Epsilon         float64 `json:"epsilon,omitempty"`
	StabilityWindow uint32  `json:"stability_window,omitempty"`
}

// AcgStabilityThresholds configures prompt stability classification thresholds.
type AcgStabilityThresholds struct {
	StableThreshold                  float64 `json:"stable_threshold,omitempty"`
	SemiStableThreshold              float64 `json:"semi_stable_threshold,omitempty"`
	MinObservationsForFullConfidence uint32  `json:"min_observations_for_full_confidence,omitempty"`
}

// AcgConfig configures the adaptive cache governor.
type AcgConfig struct {
	Provider            string                  `json:"provider,omitempty"`
	ObservationWindow   uint32                  `json:"observation_window,omitempty"`
	Priority            int32                   `json:"priority,omitempty"`
	StabilityThresholds *AcgStabilityThresholds `json:"stability_thresholds,omitempty"`
	Convergence         *ConvergenceConfig      `json:"convergence,omitempty"`
}

// AdaptiveComponentSpec wraps one adaptive config as a top-level plugin component.
type AdaptiveComponentSpec struct {
	Enabled bool           `json:"enabled,omitempty"`
	Config  AdaptiveConfig `json:"config"`
}

// NewAdaptiveConfig returns a default adaptive config with version 1.
func NewAdaptiveConfig() AdaptiveConfig {
	return AdaptiveConfig{Version: 1}
}

// NewInMemoryAdaptiveBackend returns an in-memory adaptive backend spec.
func NewInMemoryAdaptiveBackend() AdaptiveBackendSpec {
	return AdaptiveBackendSpec{
		Kind:   "in_memory",
		Config: map[string]any{},
	}
}

// NewRedisAdaptiveBackend returns a Redis adaptive backend spec.
func NewRedisAdaptiveBackend(url, keyPrefix string) AdaptiveBackendSpec {
	return AdaptiveBackendSpec{
		Kind: "redis",
		Config: map[string]any{
			"url":        url,
			"key_prefix": keyPrefix,
		},
	}
}

// NewTelemetryConfig returns default built-in adaptive telemetry settings.
func NewTelemetryConfig() TelemetryConfig {
	return TelemetryConfig{}
}

// NewGovernorConfig returns default topology-aware hint load-shedding settings.
func NewGovernorConfig() GovernorConfig {
	return GovernorConfig{Epsilon: 1.0}
}

// NewDriftConfig returns default topology-aware stale-plan detection settings.
func NewDriftConfig() DriftConfig {
	return DriftConfig{Threshold: 0.75}
}

// NewConvergenceConfig returns default topological convergence detection settings.
func NewConvergenceConfig() ConvergenceConfig {
	return ConvergenceConfig{Epsilon: 0.001, StabilityWindow: 3}
}

// NewAdaptiveHintsConfig returns default adaptive hints injection settings.
func NewAdaptiveHintsConfig() AdaptiveHintsConfig {
	return AdaptiveHintsConfig{
		Priority:       100,
		InjectHeader:   true,
		InjectBodyPath: "nvext.agent_hints",
	}
}

// NewToolParallelismConfig returns default adaptive tool scheduling settings.
func NewToolParallelismConfig() ToolParallelismConfig {
	return ToolParallelismConfig{
		Priority: 100,
		Mode:     "observe_only",
	}
}

// NewAcgStabilityThresholds returns default ACG stability thresholds.
func NewAcgStabilityThresholds() AcgStabilityThresholds {
	return AcgStabilityThresholds{
		StableThreshold:                  0.95,
		SemiStableThreshold:              0.50,
		MinObservationsForFullConfidence: 20,
	}
}

// NewAcgConfig returns default adaptive cache governor settings.
func NewAcgConfig() AcgConfig {
	thresholds := NewAcgStabilityThresholds()
	return AcgConfig{
		Provider:            "passthrough",
		ObservationWindow:   100,
		Priority:            50,
		StabilityThresholds: &thresholds,
	}
}

// NewAdaptiveComponentSpec wraps adaptive config as an enabled top-level component.
func NewAdaptiveComponentSpec(config AdaptiveConfig) AdaptiveComponentSpec {
	return AdaptiveComponentSpec{
		Enabled: true,
		Config:  config,
	}
}

// PluginComponent converts the adaptive component wrapper into the shared plugin shape.
func (spec AdaptiveComponentSpec) PluginComponent() PluginComponentSpec {
	return PluginComponentSpec{
		Kind:    AdaptivePluginKind,
		Enabled: spec.Enabled,
		Config:  mustConfigMap(spec.Config),
	}
}

// AdaptiveComponent converts adaptive config directly into a shared plugin component.
//
// Prefer NewAdaptiveComponentSpec(config).PluginComponent() when the adaptive
// component wrapper itself is part of the public surface you want to expose.
func AdaptiveComponent(config AdaptiveConfig) PluginComponentSpec {
	return NewAdaptiveComponentSpec(config).PluginComponent()
}

func mustConfigMap(value any) map[string]any {
	payload, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	var out map[string]any
	if err := json.Unmarshal(payload, &out); err != nil {
		panic(err)
	}
	if out == nil {
		return map[string]any{}
	}
	return out
}
