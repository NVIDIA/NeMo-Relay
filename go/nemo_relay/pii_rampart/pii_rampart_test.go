// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package pii_rampart

import "testing"

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
}
