// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"sync"
	"testing"
)

var testPluginHost struct {
	sync.Mutex
	activation *PluginHostActivation
}

func validateTestPluginConfig(config PluginConfig) (ConfigReport, error) {
	report, err := ValidateExact(config)
	return report.Config, err
}

func TestValidateExactUsesExactHostReport(t *testing.T) {
	originalExact := validateExactPluginHostJSON
	originalLayered := validatePluginHostJSON
	t.Cleanup(func() {
		validateExactPluginHostJSON = originalExact
		validatePluginHostJSON = originalLayered
	})

	validatePluginHostJSON = func(string, *string) (string, error) {
		t.Fatal("programmatic validation must not discover plugins.toml layers")
		return "", nil
	}
	validateExactPluginHostJSON = func(string) (string, error) {
		return `{
			"config":{"diagnostics":[]},
			"dynamic_plugins":[{
				"plugin_id":"fixture.dynamic",
				"manifest_ref":"fixture.toml",
				"kind":"worker",
				"status":{
					"manifest":"valid",
					"compatibility":"valid",
					"integrity":"valid",
					"environment":"valid",
					"authenticity":"valid",
					"policy_satisfied":"valid"
				},
				"selected":true
			}]
		}`, nil
	}

	report, err := ValidateExact(PluginConfig{Version: 1})
	if err != nil {
		t.Fatalf("ValidateExact failed: %v", err)
	}
	if len(report.DynamicPlugins) != 1 || report.DynamicPlugins[0].PluginID != "fixture.dynamic" {
		t.Fatalf("expected complete exact host report, got %#v", report)
	}
}

func initializeTestPluginHost(config PluginConfig) (ConfigReport, error) {
	if err := closeTestPluginHost(); err != nil {
		return ConfigReport{}, err
	}
	activation, report, err := Initialize(config, nil)
	if err != nil {
		return ConfigReport{}, err
	}
	testPluginHost.Lock()
	testPluginHost.activation = activation
	testPluginHost.Unlock()
	return report.Config, nil
}

func testPluginHostReport() (*ConfigReport, error) {
	testPluginHost.Lock()
	activation := testPluginHost.activation
	testPluginHost.Unlock()
	if activation == nil {
		return nil, nil
	}
	report, err := activation.Report()
	if err != nil {
		return nil, err
	}
	return &report.Config, nil
}

func closeTestPluginHost() error {
	testPluginHost.Lock()
	activation := testPluginHost.activation
	testPluginHost.activation = nil
	testPluginHost.Unlock()
	return activation.Close()
}
