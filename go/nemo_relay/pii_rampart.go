// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

// RampartPiiPluginKind is the in-process Rampart PII component kind.
const RampartPiiPluginKind = "pii_rampart"

// RampartModelID is the pinned Hugging Face model repository.
const RampartModelID = "nationaldesignstudio/rampart"

// RampartModelRevision is the pinned model revision accepted by Relay.
const RampartModelRevision = "b1993e4e68b082835b80ffc65acc03325ea2e501"

// RampartPiiConfig configures in-process Rampart PII redaction.
type RampartPiiConfig struct {
	Version              uint32        `json:"version"`
	ModelPath            string        `json:"model_path"`
	Input                bool          `json:"input"`
	Output               bool          `json:"output"`
	Mark                 bool          `json:"mark"`
	ToolInput            bool          `json:"tool_input"`
	ToolOutput           bool          `json:"tool_output"`
	Priority             int32         `json:"priority"`
	Codec                string        `json:"codec,omitempty"`
	TargetPaths          []string      `json:"target_paths,omitempty"`
	TargetPathPatterns   []string      `json:"target_path_patterns,omitempty"`
	MinScore             float64       `json:"min_score"`
	ExcludedLabels       []string      `json:"excluded_labels,omitempty"`
	Replacement          string        `json:"replacement"`
	MaxWindowsPerPayload int32         `json:"max_windows_per_payload"`
	InferenceBatchSize   int32         `json:"inference_batch_size"`
	Policy               *ConfigPolicy `json:"policy,omitempty"`
}

// RampartPiiComponentSpec wraps one Rampart PII config as a top-level plugin component.
type RampartPiiComponentSpec struct {
	Enabled bool             `json:"enabled,omitempty"`
	Config  RampartPiiConfig `json:"config"`
}

// NewRampartPiiConfig returns Rampart PII settings with runtime defaults.
// Set TargetPaths or TargetPathPatterns before validation or activation.
func NewRampartPiiConfig(modelPath string) RampartPiiConfig {
	return RampartPiiConfig{
		Version:              1,
		ModelPath:            modelPath,
		Input:                true,
		Output:               true,
		Mark:                 true,
		ToolInput:            true,
		ToolOutput:           true,
		Priority:             100,
		TargetPaths:          []string{},
		TargetPathPatterns:   []string{},
		MinScore:             0.4,
		ExcludedLabels:       []string{},
		Replacement:          "[REDACTED]",
		MaxWindowsPerPayload: 4,
		InferenceBatchSize:   16,
	}
}

// NewRampartPiiComponentSpec wraps Rampart PII config as an enabled component.
func NewRampartPiiComponentSpec(config RampartPiiConfig) RampartPiiComponentSpec {
	return RampartPiiComponentSpec{
		Enabled: true,
		Config:  config,
	}
}

// PluginComponent converts the Rampart PII wrapper into the shared plugin shape.
func (spec RampartPiiComponentSpec) PluginComponent() PluginComponentSpec {
	return PluginComponentSpec{
		Kind:    RampartPiiPluginKind,
		Enabled: spec.Enabled,
		Config:  mustConfigMap(spec.Config),
	}
}

// RampartPiiComponent converts config into the shared plugin component.
func RampartPiiComponent(config RampartPiiConfig) PluginComponentSpec {
	return NewRampartPiiComponentSpec(config).PluginComponent()
}

// ValidateRampartPiiConfig validates config without loading model files.
func ValidateRampartPiiConfig(config RampartPiiConfig) (ConfigReport, error) {
	return ValidatePluginConfig(PluginConfig{
		Version:    1,
		Components: []PluginComponentSpec{RampartPiiComponent(config)},
	})
}
