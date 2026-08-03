// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package pii_rampart

import nemo_relay "github.com/NVIDIA/NeMo-Relay/go/nemo_relay"

type Config = nemo_relay.RampartPiiConfig
type ConfigPolicy = nemo_relay.ConfigPolicy
type ConfigReport = nemo_relay.ConfigReport
type ComponentSpec = nemo_relay.RampartPiiComponentSpec

// PluginKind is the Rampart PII component kind.
const PluginKind = nemo_relay.RampartPiiPluginKind

// ModelID is the pinned Hugging Face model repository.
const ModelID = nemo_relay.RampartModelID

// ModelRevision is the pinned model revision accepted by Relay.
const ModelRevision = nemo_relay.RampartModelRevision

// NewConfig returns Rampart PII settings with runtime defaults.
// Set Preset, TargetPaths, or TargetPathPatterns before validation or activation.
func NewConfig(modelPath string) Config {
	return nemo_relay.NewRampartPiiConfig(modelPath)
}

// NewComponentSpec wraps Rampart PII config as an enabled component.
func NewComponentSpec(config Config) ComponentSpec {
	return nemo_relay.NewRampartPiiComponentSpec(config)
}

// Component converts config into the shared plugin component.
func Component(config Config) nemo_relay.PluginComponentSpec {
	return NewComponentSpec(config).PluginComponent()
}

// ValidateConfig validates config without loading model files.
func ValidateConfig(config Config) (ConfigReport, error) {
	return nemo_relay.ValidateRampartPiiConfig(config)
}
