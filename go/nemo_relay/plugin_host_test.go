// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import "sync"

var testPluginHost struct {
	sync.Mutex
	activation *PluginHostActivation
}

func validateTestPluginConfig(config PluginConfig) (ConfigReport, error) {
	return validateProgrammaticPluginConfig(config)
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
