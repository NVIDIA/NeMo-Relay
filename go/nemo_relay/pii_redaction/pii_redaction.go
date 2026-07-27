// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package pii_redaction

import nemo_relay "github.com/NVIDIA/NeMo-Relay/go/nemo_relay"

// Config is the canonical PII redaction config document.
type Config = nemo_relay.PiiRedactionConfig

// BuiltinConfig configures deterministic built-in redaction.
type BuiltinConfig = nemo_relay.PiiRedactionBuiltinConfig

// LocalModelConfig configures the future local-model redaction backend.
type LocalModelConfig = nemo_relay.PiiRedactionLocalModelConfig

// ComponentSpec wraps PII redaction config as a top-level plugin component.
type ComponentSpec = nemo_relay.PiiRedactionComponentSpec

// ConfigPolicy controls how PII redaction validation handles unsupported input.
type ConfigPolicy = nemo_relay.ConfigPolicy

// ConfigReport is the validation report returned by PII redaction helpers.
type ConfigReport = nemo_relay.ConfigReport

// PluginKind is the top-level plugin kind used by the PII redaction component.
const PluginKind = nemo_relay.PiiRedactionPluginKind

// NewConfig returns a default PII redaction config with version 1.
func NewConfig() Config {
	return nemo_relay.NewPiiRedactionConfig()
}

// NewBuiltinConfig returns default built-in redaction settings.
func NewBuiltinConfig() BuiltinConfig {
	return nemo_relay.NewPiiRedactionBuiltinConfig()
}

// NewLocalModelConfig returns default local-model redaction settings.
func NewLocalModelConfig() LocalModelConfig {
	return nemo_relay.NewPiiRedactionLocalModelConfig()
}

// NewComponentSpec wraps PII redaction config as an enabled component.
func NewComponentSpec(config Config) ComponentSpec {
	return nemo_relay.NewPiiRedactionComponentSpec(config)
}

// Component converts PII redaction config directly into the shared plugin shape.
func Component(config Config) nemo_relay.PluginComponentSpec {
	return nemo_relay.PiiRedactionComponent(config)
}

// ValidateConfig validates a PII redaction config without activating it.
func ValidateConfig(config Config) (ConfigReport, error) {
	return nemo_relay.ValidatePiiRedactionConfig(config)
}
