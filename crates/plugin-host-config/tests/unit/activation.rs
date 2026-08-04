// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Mutex, OnceLock};

use nemo_relay::plugin::PluginConfig;
use nemo_relay::plugin::dynamic::PluginHostActivationPlan;
use tempfile::tempdir;

use super::*;

fn activation_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn resolve_options(root: &std::path::Path) -> PluginFileResolveOptions {
    PluginFileResolveOptions {
        plugin_config_path: Some(root.join("selected/plugins.toml")),
        current_dir: None,
        user_config_dir: None,
        system_config_path: root.join("system/plugins.toml"),
    }
}

#[test]
fn inactive_activation_has_an_empty_report_and_clears_as_a_no_op() {
    let activation = PluginFileActivation::inactive();

    assert!(!activation.is_active());
    assert!(activation.report().diagnostics.is_empty());
    assert!(activation.report().runtime_diagnostics.is_empty());
    activation.clear().unwrap();
}

#[test]
fn file_initialization_distinguishes_no_input_from_explicit_empty_configuration() {
    let _lock = activation_lock();
    let temp = tempdir().unwrap();
    let runtime = runtime();

    let inactive = runtime
        .block_on(initialize_from_plugins_toml_with_options(
            None,
            resolve_options(temp.path()),
        ))
        .unwrap();
    assert!(!inactive.is_active());
    inactive.clear().unwrap();

    let active = runtime
        .block_on(initialize_from_plugins_toml_with_options(
            Some(PluginConfig::default()),
            resolve_options(temp.path()),
        ))
        .unwrap();
    assert!(active.is_active());
    assert!(active.report().diagnostics.is_empty());
    active.clear().unwrap();
}

#[test]
fn resolved_plan_activation_owns_and_releases_the_process_lease() {
    let _lock = activation_lock();
    let runtime = runtime();
    let plan = || PluginHostActivationPlan {
        config: PluginConfig::default(),
        dynamic_plugins: Vec::new(),
        diagnostics: Vec::new(),
    };

    let activation = runtime
        .block_on(PluginFileActivation::activate_plan(plan()))
        .unwrap();
    assert!(activation.is_active());

    let conflict = match runtime.block_on(PluginFileActivation::activate_plan(plan())) {
        Ok(_) => panic!("a second owned activation unexpectedly acquired the process lease"),
        Err(error) => error,
    };
    assert!(matches!(
        conflict,
        PluginHostConfigError::Relay(PluginError::Conflict(_))
    ));

    activation.clear().unwrap();
    let replacement = runtime
        .block_on(PluginFileActivation::activate_plan(plan()))
        .unwrap();
    replacement.clear().unwrap();
}
