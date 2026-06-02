// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"os"
	"path/filepath"
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

func TestInitializePluginsLayersCodeConfigOverProjectPluginsToml(t *testing.T) {
	root := t.TempDir()
	project := filepath.Join(root, "project")
	configDir := filepath.Join(project, ".nemo-relay")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("failed to create config dir: %v", err)
	}
	pluginKind := "go.layered.plugin"
	pluginsToml := `
version = 1

[[components]]
kind = "go.layered.plugin"
enabled = true

[components.config]
source = "file"

[components.config.nested]
file = true
`
	if err := os.WriteFile(filepath.Join(configDir, "plugins.toml"), []byte(pluginsToml), 0o644); err != nil {
		t.Fatalf("failed to write plugins.toml: %v", err)
	}
	oldCwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("failed to read cwd: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(oldCwd)
		_ = ClearPluginConfiguration()
		_ = DeregisterPlugin(pluginKind)
	})
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(root, "xdg"))
	t.Setenv("HOME", filepath.Join(root, "home"))
	if err := os.Chdir(project); err != nil {
		t.Fatalf("failed to change cwd: %v", err)
	}

	var configs []map[string]any
	if err := RegisterPlugin(pluginKind, PluginFuncs{
		ValidateFunc: func(pluginConfig map[string]any) ([]ConfigDiagnostic, error) {
			configs = append(configs, pluginConfig)
			return nil, nil
		},
		RegisterFunc: func(pluginConfig map[string]any, ctx *PluginContext) error {
			configs = append(configs, pluginConfig)
			return nil
		},
	}); err != nil {
		t.Fatalf("RegisterPlugin failed: %v", err)
	}

	report, err := InitializePlugins(PluginConfig{
		Components: []PluginComponentSpec{{
			Kind: pluginKind,
			Config: map[string]any{
				"source": "code",
				"nested": map[string]any{
					"code": true,
				},
			},
		}},
	})
	if err != nil {
		t.Fatalf("InitializePlugins failed: %v", err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("unexpected diagnostics: %#v", report.Diagnostics)
	}
	if len(configs) != 2 {
		t.Fatalf("expected validate and register configs, got %#v", configs)
	}
	for _, config := range configs {
		if config["source"] != "code" {
			t.Fatalf("source mismatch: %#v", config)
		}
		nested, ok := config["nested"].(map[string]any)
		if !ok || nested["file"] != true || nested["code"] != true {
			t.Fatalf("nested config mismatch: %#v", config)
		}
	}
}
