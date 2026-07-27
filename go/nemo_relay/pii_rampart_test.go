// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"testing"
)

func TestRampartPiiConfigHelpers(t *testing.T) {
	config := NewRampartPiiConfig("/models/rampart")
	config.TargetPathPatterns = []string{"/messages/*/content"}
	component := RampartPiiComponent(config)
	if component.Kind != RampartPiiPluginKind || !component.Enabled {
		t.Fatalf("unexpected Rampart PII component: %#v", component)
	}
	if component.Config["model_path"] != "/models/rampart" {
		t.Fatalf("unexpected Rampart PII config: %#v", component.Config)
	}
	if RampartModelID != "nationaldesignstudio/rampart" ||
		RampartModelRevision != "b1993e4e68b082835b80ffc65acc03325ea2e501" {
		t.Fatalf("unexpected Rampart model identity: %s@%s", RampartModelID, RampartModelRevision)
	}
}

func TestRampartPiiConfigPreservesExplicitZeroValues(t *testing.T) {
	config := NewRampartPiiConfig("/models/rampart")
	config.Version = 0
	config.Priority = 0

	serialized, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("marshal Rampart PII config: %v", err)
	}
	var value map[string]any
	if err := json.Unmarshal(serialized, &value); err != nil {
		t.Fatalf("decode Rampart PII config: %v", err)
	}
	if value["version"] != float64(0) || value["priority"] != float64(0) {
		t.Fatalf("explicit zero values were not preserved: %#v", value)
	}
}
