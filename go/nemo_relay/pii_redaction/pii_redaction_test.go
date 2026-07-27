// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package pii_redaction

import "testing"

func TestPiiRedactionShorthandHelpers(t *testing.T) {
	config := NewConfig()
	config.Codec = "openai_chat"
	builtin := NewBuiltinConfig()
	config.Builtin = &builtin

	component := Component(config)
	if component.Kind != PluginKind || !component.Enabled {
		t.Fatalf("unexpected PII redaction component: %#v", component)
	}
	report, err := ValidateConfig(config)
	if err != nil {
		t.Fatalf("ValidateConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("unexpected diagnostics: %#v", report.Diagnostics)
	}
}

func TestPiiRedactionComponentSpecAndLocalModelHelpers(t *testing.T) {
	config := NewConfig()
	local := NewLocalModelConfig()
	local.Backend = "local"
	local.ModelID = "pii-model"
	config.Mode = "local"
	config.Local = &local

	spec := NewComponentSpec(config)
	if !spec.Enabled || spec.Config.Local == nil || spec.Config.Local.ModelID != "pii-model" {
		t.Fatalf("unexpected PII redaction component spec: %#v", spec)
	}
}
