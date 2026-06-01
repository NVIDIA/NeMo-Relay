// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"reflect"
	"testing"
)

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

func TestLayerPluginConfigRoundTripsMerge(t *testing.T) {
	// Smoke test only: merge semantics are covered by the core crate. This
	// verifies the cgo boundary forwards both documents and returns merged JSON.
	merged, err := LayerPluginConfig(
		map[string]any{"a": float64(1)},
		map[string]any{"b": float64(2)},
	)
	if err != nil {
		t.Fatalf("LayerPluginConfig failed: %v", err)
	}

	expected := map[string]any{"a": float64(1), "b": float64(2)}
	if !reflect.DeepEqual(merged, expected) {
		t.Fatalf("merged config mismatch:\n got: %#v\nwant: %#v", merged, expected)
	}
}
