// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the code-driven plugin configuration layer.
//!
//! These run in a dedicated test binary so mutating the process-global
//! code-driven layer cannot contaminate the shared lib-test binary, which runs
//! other `initialize_plugins` tests concurrently. Within this binary the few
//! global-mutating tests are serialized with a local lock.

use std::sync::Mutex;

use nemo_relay::plugin::{
    ConfigPolicy, DiagnosticLevel, PluginComponentSpec, PluginConfig, PluginError,
    UnsupportedBehavior, apply_code_driven_plugin_config, clear_plugin_configuration,
    initialize_plugins, set_code_driven_plugin_config,
};

static LAYER_LOCK: Mutex<()> = Mutex::new(());

// A disabled, unknown-kind component. It is validated (so configuration policy applies to it) but
// skipped during activation (so initialization does not fail on a missing plugin registration).
fn disabled_unknown_component_config() -> PluginConfig {
    PluginConfig {
        components: vec![PluginComponentSpec {
            kind: "relay183.ghost.kind".into(),
            enabled: false,
            config: serde_json::Map::new(),
        }],
        ..PluginConfig::default()
    }
}

#[test]
fn set_clears_and_applies_code_driven_layer() {
    let _guard = LAYER_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    set_code_driven_plugin_config(None);
    // No layer: apply returns None, so the caller's own config is already effective.
    assert!(apply_code_driven_plugin_config(&PluginConfig::default()).is_none());

    let layer = PluginConfig {
        components: vec![PluginComponentSpec::new("relay183.example")],
        ..PluginConfig::default()
    };
    set_code_driven_plugin_config(Some(layer));
    // Layer set: apply returns Some(effective) with the layer's component merged over the base.
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

#[test]
fn initialize_plugins_is_unchanged_without_a_layer() {
    let _guard = LAYER_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    set_code_driven_plugin_config(None);

    // Default policy treats an unknown component as a warning, so a disabled unknown component
    // validates and initializes without error. This is the file-only baseline.
    let report =
        futures::executor::block_on(initialize_plugins(disabled_unknown_component_config()))
            .expect("file-only initialization succeeds");
    assert!(!report.has_errors());
    assert!(
        report.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "plugin.unknown_component"
                && diagnostic.level == DiagnosticLevel::Warning
        }),
        "expected an unknown-component warning: {report:?}"
    );
    clear_plugin_configuration().unwrap();
}

#[test]
fn code_driven_layer_policy_overrides_file_config_before_validation() {
    let _guard = LAYER_LOCK.lock().unwrap_or_else(|err| err.into_inner());

    // The overlay tightens policy so the same disabled unknown component now fails validation.
    // Because the failure occurs at all, the layer must have been merged on top of the caller's
    // config *before* validation, and the overlay's policy won over the file default policy.
    set_code_driven_plugin_config(Some(PluginConfig {
        policy: ConfigPolicy {
            unknown_component: UnsupportedBehavior::Error,
            ..ConfigPolicy::default()
        },
        ..PluginConfig::default()
    }));

    let error =
        futures::executor::block_on(initialize_plugins(disabled_unknown_component_config()))
            .expect_err("overlay policy turns the unknown component into an error");
    assert!(matches!(error, PluginError::InvalidConfig(_)), "{error:?}");

    // The failing activation must not leave a configuration active.
    set_code_driven_plugin_config(None);
    let _ = clear_plugin_configuration();
}
