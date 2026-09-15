// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"strings"
	"testing"
)

func TestPiiRedactionConfigHelpers(t *testing.T) {
	config := NewPiiRedactionConfig()
	if config.Version != 1 || config.Mode != "builtin" || !config.Input || !config.Output || !config.Mark || !config.ToolInput || !config.ToolOutput || config.Priority != 100 {
		t.Fatalf("unexpected PII redaction defaults: %#v", config)
	}
	if config.Builtin == nil || config.Builtin.Action != "remove" || len(config.Builtin.TargetPaths) != 0 || len(config.Builtin.TargetPathGlobs) != 0 || config.Builtin.CustomMarkPayloadPolicy != "" || len(config.Builtin.MetricStringAttributeAllowlist) != 0 {
		t.Fatalf("unexpected default built-in redaction config: %#v", config.Builtin)
	}
	builtin := NewPiiRedactionBuiltinConfig()
	if builtin.Action != "remove" || len(builtin.TargetPaths) != 0 || len(builtin.TargetPathGlobs) != 0 || builtin.CustomMarkPayloadPolicy != "" || len(builtin.MetricStringAttributeAllowlist) != 0 {
		t.Fatalf("unexpected built-in redaction defaults: %#v", builtin)
	}
	local := NewPiiRedactionLocalModelConfig()
	if local != (PiiRedactionLocalModelConfig{}) {
		t.Fatalf("unexpected local model defaults: %#v", local)
	}

	config.Builtin = &builtin
	config.Builtin.Preset = "trajectory_context"
	config.Builtin.CustomMarkPayloadPolicy = "preserve"
	config.Builtin.MetricStringAttributeAllowlist = map[string][]string{"gen_ai.operation.name": {"chat"}}
	component := PiiRedactionComponent(config)
	if component.Kind != PiiRedactionPluginKind || !component.Enabled {
		t.Fatalf("unexpected PII redaction component: %#v", component)
	}
	if component.Config["mode"] != "builtin" || component.Config["priority"] != float64(100) {
		t.Fatalf("unexpected serialized config: %#v", component.Config)
	}
	if component.Config["mark"] != true {
		t.Fatalf("expected mark redaction to be enabled: %#v", component.Config)
	}
	serializedBuiltin, ok := component.Config["builtin"].(map[string]any)
	if !ok {
		t.Fatalf("expected serialized builtin object, got %#v", component.Config["builtin"])
	}
	if _, ok := serializedBuiltin["action"]; ok {
		t.Fatalf("trajectory-context config must omit the legacy action: %#v", serializedBuiltin)
	}
	if serializedBuiltin["preset"] != "trajectory_context" || serializedBuiltin["custom_mark_payload_policy"] != "preserve" {
		t.Fatalf("expected trajectory-context builtin settings: %#v", serializedBuiltin)
	}
	allowlist, ok := serializedBuiltin["metric_string_attribute_allowlist"].(map[string]any)
	if !ok || len(allowlist) != 1 {
		t.Fatalf("expected serialized metric allowlist: %#v", serializedBuiltin)
	}
	report, err := ValidatePiiRedactionConfig(config)
	if err != nil {
		t.Fatalf("ValidatePiiRedactionConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("expected trajectory-context config to validate: %#v", report.Diagnostics)
	}
}

func TestPiiRedactionValidationRejectsBadValues(t *testing.T) {
	config := NewPiiRedactionConfig()
	config.Input = false
	config.Output = false
	builtin := NewPiiRedactionBuiltinConfig()
	builtin.Action = "mask"
	builtin.Detector = "not_a_detector"
	config.Builtin = &builtin

	report, err := ValidatePiiRedactionConfig(config)
	if err != nil {
		t.Fatalf("ValidatePiiRedactionConfig failed: %v", err)
	}
	for _, diagnostic := range report.Diagnostics {
		if diagnostic.Field != nil && *diagnostic.Field == "builtin.detector" {
			return
		}
	}
	t.Fatalf("expected builtin.detector diagnostic, got %#v", report.Diagnostics)
}

func TestPiiRedactionValidationRequiresTrajectoryPresetForTrajectoryOptions(t *testing.T) {
	tests := []struct {
		name    string
		builtin PiiRedactionBuiltinConfig
		field   string
		message string
	}{
		{
			name:    "custom mark payload policy",
			builtin: PiiRedactionBuiltinConfig{CustomMarkPayloadPolicy: "preserve"},
			field:   "builtin.custom_mark_payload_policy",
			message: "requires builtin.preset = 'trajectory_context'",
		},
		{
			name: "metric string attribute allowlist",
			builtin: PiiRedactionBuiltinConfig{
				MetricStringAttributeAllowlist: map[string][]string{"gen_ai.operation.name": {"chat"}},
			},
			field:   "builtin.metric_string_attribute_allowlist",
			message: "requires builtin.preset = 'trajectory_context'",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := NewPiiRedactionConfig()
			config.Builtin = &test.builtin

			report, err := ValidatePiiRedactionConfig(config)
			if err != nil {
				t.Fatalf("ValidatePiiRedactionConfig failed: %v", err)
			}
			for _, diagnostic := range report.Diagnostics {
				if diagnostic.Field != nil && *diagnostic.Field == test.field && strings.Contains(diagnostic.Message, test.message) {
					return
				}
			}
			t.Fatalf("expected %s diagnostic, got %#v", test.field, report.Diagnostics)
		})
	}
}

func TestPiiRedactionListKindIsAutomatic(t *testing.T) {
	kinds, err := ListPluginKinds()
	if err != nil {
		t.Fatalf("ListPluginKinds failed: %v", err)
	}
	for _, kind := range kinds {
		if kind == PiiRedactionPluginKind {
			return
		}
	}
	t.Fatalf("expected %q in registered kinds: %#v", PiiRedactionPluginKind, kinds)
}
