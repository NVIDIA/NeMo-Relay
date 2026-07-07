// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

// PiiRedactionPluginKind is the top-level plugin kind used by the built-in PII redaction component.
const PiiRedactionPluginKind = "pii_redaction"

// PiiRedactionBuiltinConfig configures deterministic built-in redaction.
type PiiRedactionBuiltinConfig struct {
	Action         string   `json:"action,omitempty"`
	TargetPaths    []string `json:"target_paths,omitempty"`
	Pattern        string   `json:"pattern,omitempty"`
	Detector       string   `json:"detector,omitempty"`
	Replacement    string   `json:"replacement,omitempty"`
	MaskChar       string   `json:"mask_char,omitempty"`
	UnmaskedPrefix *int32   `json:"unmasked_prefix,omitempty"`
	UnmaskedSuffix *int32   `json:"unmasked_suffix,omitempty"`
}

// PiiRedactionLocalModelConfig configures the future local-model redaction backend.
type PiiRedactionLocalModelConfig struct {
	Backend         string `json:"backend,omitempty"`
	ModelID         string `json:"model_id,omitempty"`
	DetectorProfile string `json:"detector_profile,omitempty"`
	AllowNetwork    *bool  `json:"allow_network,omitempty"`
	MaxLatencyMS    *int32 `json:"max_latency_ms,omitempty"`
}

// PiiRedactionConfig is the canonical Go shape for the PII redaction plugin config document.
type PiiRedactionConfig struct {
	Version    uint32                        `json:"version,omitempty"`
	Mode       string                        `json:"mode,omitempty"`
	Input      bool                          `json:"input"`
	Output     bool                          `json:"output"`
	Mark       bool                          `json:"mark"`
	ToolInput  bool                          `json:"tool_input"`
	ToolOutput bool                          `json:"tool_output"`
	Priority   int32                         `json:"priority,omitempty"`
	Codec      string                        `json:"codec,omitempty"`
	Builtin    *PiiRedactionBuiltinConfig    `json:"builtin,omitempty"`
	Local      *PiiRedactionLocalModelConfig `json:"local,omitempty"`
	Policy     *ConfigPolicy                 `json:"policy,omitempty"`
}

// PiiRedactionComponentSpec wraps one PII redaction config as a top-level plugin component.
type PiiRedactionComponentSpec struct {
	Enabled    bool               `json:"enabled,omitempty"`
	Config     PiiRedactionConfig `json:"config"`
	enabledSet bool
}

// NewPiiRedactionConfig returns a default PII redaction config with version 1.
func NewPiiRedactionConfig() PiiRedactionConfig {
	builtin := NewPiiRedactionBuiltinConfig()
	return PiiRedactionConfig{
		Version:    1,
		Mode:       "builtin",
		Input:      true,
		Output:     true,
		Mark:       true,
		ToolInput:  true,
		ToolOutput: true,
		Priority:   100,
		Builtin:    &builtin,
	}
}

// NewPiiRedactionBuiltinConfig returns default built-in redaction settings.
func NewPiiRedactionBuiltinConfig() PiiRedactionBuiltinConfig {
	return PiiRedactionBuiltinConfig{
		Action:      "remove",
		TargetPaths: []string{},
	}
}

// NewPiiRedactionLocalModelConfig returns default local-model redaction settings.
func NewPiiRedactionLocalModelConfig() PiiRedactionLocalModelConfig {
	return PiiRedactionLocalModelConfig{}
}

// NewPiiRedactionComponentSpec wraps PII redaction config as an enabled component.
func NewPiiRedactionComponentSpec(config PiiRedactionConfig) PiiRedactionComponentSpec {
	return PiiRedactionComponentSpec{
		Enabled:    true,
		Config:     config,
		enabledSet: true,
	}
}

// SetEnabled explicitly sets whether this PII redaction component is enabled.
func (spec *PiiRedactionComponentSpec) SetEnabled(enabled bool) {
	spec.Enabled = enabled
	spec.enabledSet = true
}

// WithEnabled returns a copy with an explicitly present enabled value.
func (spec PiiRedactionComponentSpec) WithEnabled(enabled bool) PiiRedactionComponentSpec {
	spec.SetEnabled(enabled)
	return spec
}

// MarshalJSON preserves omitted-default and explicit-false enabled semantics.
func (spec PiiRedactionComponentSpec) MarshalJSON() ([]byte, error) {
	return marshalComponentWrapper(spec.Enabled, spec.enabledSet, spec.Config)
}

// UnmarshalJSON records whether enabled was explicitly present.
func (spec *PiiRedactionComponentSpec) UnmarshalJSON(payload []byte) error {
	enabled, enabledSet, config, err := unmarshalComponentWrapper[PiiRedactionConfig](payload)
	if err != nil {
		return err
	}
	*spec = PiiRedactionComponentSpec{Enabled: enabled, Config: config, enabledSet: enabledSet}
	return nil
}

// PluginComponent converts the PII redaction wrapper into the shared plugin shape.
func (spec PiiRedactionComponentSpec) PluginComponent() PluginComponentSpec {
	component := PluginComponentSpec{
		Kind:    PiiRedactionPluginKind,
		Enabled: spec.Enabled,
		Config:  mustConfigMap(spec.Config),
	}
	if spec.enabledSet || spec.Enabled {
		component.SetEnabled(spec.Enabled)
	}
	return component
}

// PiiRedactionComponent converts PII redaction config directly into a shared plugin component.
func PiiRedactionComponent(config PiiRedactionConfig) PluginComponentSpec {
	return NewPiiRedactionComponentSpec(config).PluginComponent()
}

// ValidatePiiRedactionConfig validates a PII redaction config without activating it.
func ValidatePiiRedactionConfig(config PiiRedactionConfig) (ConfigReport, error) {
	return ValidatePluginConfig(PluginConfig{
		Version:    1,
		Components: []PluginComponentSpec{PiiRedactionComponent(config)},
	})
}
