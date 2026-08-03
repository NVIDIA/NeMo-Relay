// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package pii_rampart

import (
	"path/filepath"
	"testing"
)

func TestConfigAndComponentHelpers(t *testing.T) {
	config := NewConfig("/models/rampart")
	config.TargetPathPatterns = []string{"/messages/*/content"}
	component := Component(config)
	if component.Kind != PluginKind || !component.Enabled {
		t.Fatalf("unexpected Rampart PII component: %#v", component)
	}
	if component.Config["model_path"] != "/models/rampart" {
		t.Fatalf("unexpected Rampart PII config: %#v", component.Config)
	}
	if ModelID != "nationaldesignstudio/rampart" ||
		ModelRevision != "b1993e4e68b082835b80ffc65acc03325ea2e501" {
		t.Fatalf("unexpected Rampart model identity: %s@%s", ModelID, ModelRevision)
	}

	disabled := NewComponentSpec(config)
	disabled.Enabled = false
	if disabled.PluginComponent().Enabled {
		t.Fatal("disabled Rampart PII component was enabled")
	}
}

func TestValidateConfig(t *testing.T) {
	modelPath, err := filepath.Abs("testdata/rampart")
	if err != nil {
		t.Fatalf("resolve model path: %v", err)
	}
	config := NewConfig(modelPath)
	config.TargetPaths = []string{"/message"}

	report, err := ValidateConfig(config)
	if err != nil {
		t.Fatalf("ValidateConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("unexpected diagnostics: %#v", report.Diagnostics)
	}

	config.TargetPaths = nil
	config.TargetPathPatterns = []string{"/messages/pre*fix/content"}
	report, err = ValidateConfig(config)
	if err != nil {
		t.Fatalf("ValidateConfig rejected diagnostic input: %v", err)
	}
	for _, diagnostic := range report.Diagnostics {
		if diagnostic.Field != nil && *diagnostic.Field == "target_path_patterns" {
			return
		}
	}
	t.Fatalf("expected target_path_patterns diagnostic, got %#v", report.Diagnostics)
}

func TestValidateTrajectoryPreset(t *testing.T) {
	modelPath, err := filepath.Abs("testdata/rampart")
	if err != nil {
		t.Fatalf("resolve model path: %v", err)
	}
	config := NewConfig(modelPath)
	config.Preset = "trajectory_context"

	report, err := ValidateConfig(config)
	if err != nil {
		t.Fatalf("ValidateConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("unexpected diagnostics: %#v", report.Diagnostics)
	}
}
