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
	minScore := 0.75
	replacement := "[PRIVATE]"
	allowNetwork := false
	maxLatencyMS := int32(250)
	local.Backend = "nemo_relay.pii_rampart/detector"
	local.ModelID = "pii-model"
	local.DetectorProfile = "default"
	local.TargetPaths = []string{"/message"}
	local.TargetPathPatterns = []string{"/messages/*/content"}
	local.MinScore = &minScore
	local.ExcludedLabels = []string{"CITY"}
	local.Replacement = &replacement
	local.AllowNetwork = &allowNetwork
	local.MaxLatencyMS = &maxLatencyMS
	config.Mode = "local_model"
	config.Local = &local

	spec := NewComponentSpec(config)
	if !spec.Enabled ||
		spec.Config.Local == nil ||
		spec.Config.Local.ModelID != "pii-model" ||
		spec.Config.Local.DetectorProfile != "default" ||
		len(spec.Config.Local.TargetPaths) != 1 ||
		len(spec.Config.Local.TargetPathPatterns) != 1 ||
		spec.Config.Local.MinScore == nil ||
		*spec.Config.Local.MinScore != minScore ||
		len(spec.Config.Local.ExcludedLabels) != 1 ||
		spec.Config.Local.Replacement == nil ||
		*spec.Config.Local.Replacement != replacement ||
		spec.Config.Local.AllowNetwork == nil ||
		*spec.Config.Local.AllowNetwork ||
		spec.Config.Local.MaxLatencyMS == nil ||
		*spec.Config.Local.MaxLatencyMS != maxLatencyMS {
		t.Fatalf("unexpected PII redaction component spec: %#v", spec)
	}
}
