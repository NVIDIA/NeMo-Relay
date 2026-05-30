// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

// ObservabilityPluginKind is the top-level plugin kind used by the core observability component.
const ObservabilityPluginKind = "observability"

// ObservabilityConfig is the canonical Go shape for the observability plugin config document.
type ObservabilityConfig struct {
	Version       uint32                         `json:"version,omitempty"`
	Atof          *ObservabilityAtofConfig       `json:"atof,omitempty"`
	AtofStream    *ObservabilityAtofStreamConfig `json:"atof_stream,omitempty"`
	Atif          *ObservabilityAtifConfig       `json:"atif,omitempty"`
	OpenTelemetry *ObservabilityOtlpConfig       `json:"opentelemetry,omitempty"`
	OpenInference *ObservabilityOtlpConfig       `json:"openinference,omitempty"`
	Policy        *ConfigPolicy                  `json:"policy,omitempty"`
}

// ObservabilityAtofConfig configures filesystem-backed raw ATOF JSONL export.
type ObservabilityAtofConfig struct {
	Enabled         bool   `json:"enabled,omitempty"`
	OutputDirectory string `json:"output_directory,omitempty"`
	Filename        string `json:"filename,omitempty"`
	Mode            string `json:"mode,omitempty"`
}

// ObservabilityAtofStreamConfig configures TCP-backed raw ATOF JSONL stream export.
type ObservabilityAtofStreamConfig struct {
	Enabled bool   `json:"enabled,omitempty"`
	Address string `json:"address,omitempty"`
}

// ObservabilityAtifConfig configures per-top-level-agent ATIF file export.
type ObservabilityAtifConfig struct {
	Enabled          bool             `json:"enabled,omitempty"`
	AgentName        string           `json:"agent_name,omitempty"`
	AgentVersion     string           `json:"agent_version,omitempty"`
	ModelName        string           `json:"model_name,omitempty"`
	ToolDefinitions  []map[string]any `json:"tool_definitions,omitempty"`
	Extra            map[string]any   `json:"extra,omitempty"`
	OutputDirectory  string           `json:"output_directory,omitempty"`
	FilenameTemplate string           `json:"filename_template,omitempty"`
}

// ObservabilityOtlpConfig configures OpenTelemetry or OpenInference OTLP export.
type ObservabilityOtlpConfig struct {
	Enabled              bool              `json:"enabled,omitempty"`
	Transport            string            `json:"transport,omitempty"`
	Endpoint             string            `json:"endpoint,omitempty"`
	Headers              map[string]string `json:"headers,omitempty"`
	ResourceAttributes   map[string]string `json:"resource_attributes,omitempty"`
	ServiceName          string            `json:"service_name,omitempty"`
	ServiceNamespace     string            `json:"service_namespace,omitempty"`
	ServiceVersion       string            `json:"service_version,omitempty"`
	InstrumentationScope string            `json:"instrumentation_scope,omitempty"`
	TimeoutMillis        uint64            `json:"timeout_millis,omitempty"`
}

// ObservabilityComponentSpec wraps one observability config as a top-level plugin component.
type ObservabilityComponentSpec struct {
	Enabled bool                `json:"enabled,omitempty"`
	Config  ObservabilityConfig `json:"config"`
}

// NewObservabilityConfig returns a default observability config with version 1.
func NewObservabilityConfig() ObservabilityConfig {
	return ObservabilityConfig{Version: 1}
}

// NewObservabilityAtofConfig returns disabled ATOF JSONL settings with native defaults.
func NewObservabilityAtofConfig() ObservabilityAtofConfig {
	return ObservabilityAtofConfig{
		Mode: "append",
	}
}

// NewObservabilityAtofStreamConfig returns disabled ATOF stream settings.
func NewObservabilityAtofStreamConfig() ObservabilityAtofStreamConfig {
	return ObservabilityAtofStreamConfig{}
}

// NewObservabilityAtifConfig returns disabled ATIF settings with core defaults.
func NewObservabilityAtifConfig() ObservabilityAtifConfig {
	return ObservabilityAtifConfig{
		AgentName:        "NeMo Relay",
		ModelName:        "unknown",
		FilenameTemplate: "nemo-relay-atif-{session_id}.json",
	}
}

// NewObservabilityOtlpConfig returns disabled OTLP settings with core defaults.
func NewObservabilityOtlpConfig() ObservabilityOtlpConfig {
	return ObservabilityOtlpConfig{
		Transport:          "http_binary",
		Headers:            map[string]string{},
		ResourceAttributes: map[string]string{},
		ServiceName:        "nemo-relay",
		TimeoutMillis:      3000,
	}
}

// NewObservabilityComponentSpec wraps observability config as an enabled top-level component.
func NewObservabilityComponentSpec(config ObservabilityConfig) ObservabilityComponentSpec {
	return ObservabilityComponentSpec{
		Enabled: true,
		Config:  config,
	}
}

// PluginComponent converts the observability component wrapper into the shared plugin shape.
func (spec ObservabilityComponentSpec) PluginComponent() PluginComponentSpec {
	return PluginComponentSpec{
		Kind:    ObservabilityPluginKind,
		Enabled: spec.Enabled,
		Config:  mustConfigMap(spec.Config),
	}
}

// ObservabilityComponent converts observability config directly into a shared plugin component.
func ObservabilityComponent(config ObservabilityConfig) PluginComponentSpec {
	return NewObservabilityComponentSpec(config).PluginComponent()
}
