// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test for the code-driven plugin layer's runtime API. Runs in a dedicated test
//! binary so mutating the process-global layer cannot race other tests. Merge correctness is
//! covered by the pure `merge_plugin_config_layers` unit tests.

use nemo_relay::plugin::{
    PluginComponentSpec, PluginConfig, apply_code_driven_plugin_config,
    set_code_driven_plugin_config,
};

#[test]
fn apply_reflects_the_code_driven_layer() {
    set_code_driven_plugin_config(None);
    assert!(apply_code_driven_plugin_config(&PluginConfig::default()).is_none());

    set_code_driven_plugin_config(Some(PluginConfig {
        components: vec![PluginComponentSpec::new("relay183.example")],
        ..PluginConfig::default()
    }));
    let effective =
        apply_code_driven_plugin_config(&PluginConfig::default()).expect("a layer is active");
    assert!(
        effective
            .components
            .iter()
            .any(|component| component.kind == "relay183.example"),
        "effective config should include the code-driven component: {effective:?}"
    );

    set_code_driven_plugin_config(None);
    assert!(apply_code_driven_plugin_config(&PluginConfig::default()).is_none());
}
