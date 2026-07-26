// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestPiiRedactionConfigHelpers(t *testing.T) {
	config := NewPiiRedactionConfig()
	if config.Version != 1 || config.Mode != "builtin" || !config.Input || !config.Output || !config.Mark || !config.ToolInput || !config.ToolOutput || config.Priority != 100 {
		t.Fatalf("unexpected PII redaction defaults: %#v", config)
	}
	if config.Builtin == nil || config.Builtin.Action != "remove" || len(config.Builtin.TargetPaths) != 0 {
		t.Fatalf("unexpected default built-in redaction config: %#v", config.Builtin)
	}
	builtin := NewPiiRedactionBuiltinConfig()
	if builtin.Action != "remove" || len(builtin.TargetPaths) != 0 {
		t.Fatalf("unexpected built-in redaction defaults: %#v", builtin)
	}
	local := NewPiiRedactionLocalModelConfig()
	if !reflect.DeepEqual(local, PiiRedactionLocalModelConfig{}) {
		t.Fatalf("unexpected local model defaults: %#v", local)
	}
	profile := NewPiiRedactionProfile()
	if !profile.Enabled || profile.Mode != "builtin" || profile.Priority != 100 {
		t.Fatalf("unexpected profile defaults: %#v", profile)
	}

	config.Builtin = &builtin
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
	if serializedBuiltin["action"] != "remove" {
		t.Fatalf("unexpected serialized builtin config: %#v", serializedBuiltin)
	}
}

func TestPiiRedactionProfilesOmitLegacyTopLevelFields(t *testing.T) {
	config := NewPiiRedactionConfig()
	config.Profiles = []PiiRedactionProfile{
		NewPiiRedactionProfile(),
	}
	serialized, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("marshal profile config: %v", err)
	}
	var value map[string]any
	if err := json.Unmarshal(serialized, &value); err != nil {
		t.Fatalf("decode profile config: %v", err)
	}
	if _, present := value["mode"]; present {
		t.Fatalf("profile config retained legacy mode: %#v", value)
	}
	if _, present := value["input"]; present {
		t.Fatalf("profile config retained legacy input: %#v", value)
	}
	if len(value["profiles"].([]any)) != 1 {
		t.Fatalf("unexpected profile config: %#v", value)
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
