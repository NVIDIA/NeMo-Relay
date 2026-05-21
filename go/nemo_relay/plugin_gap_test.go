// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import "testing"

func TestPluginConfigSerializationErrorsSurfaceBeforeFFI(t *testing.T) {
	config := PluginConfig{
		Version: 1,
		Components: []PluginComponentSpec{
			{
				Kind:    "go.invalid.plugin",
				Enabled: true,
				Config: map[string]any{
					"unsupported": make(chan int),
				},
			},
		},
	}

	if cConfig, err := pluginConfigCString(config); err == nil {
		t.Fatalf("expected pluginConfigCString serialization error, got %v", cConfig)
	}

	if _, err := ValidatePluginConfig(config); err == nil {
		t.Fatal("expected ValidatePluginConfig serialization error")
	}

	if _, err := InitializePlugins(config); err == nil {
		t.Fatal("expected InitializePlugins serialization error")
	}
}
