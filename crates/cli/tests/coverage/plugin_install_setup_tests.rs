// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};
use std::cell::RefCell;

use super::*;

struct DefaultsOnlyRunner;

#[derive(Default)]
struct GenerationAwareRunner {
    calls: RefCell<Vec<String>>,
}

impl PluginSetupRunner for DefaultsOnlyRunner {
    fn setup(
        &self,
        _host: IntegrationHost,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<(), String> {
        Ok(())
    }

    fn uninstall(
        &self,
        _host: IntegrationHost,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<(), String> {
        Ok(())
    }

    fn doctor(
        &self,
        _host: IntegrationHost,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<(), String> {
        Ok(())
    }

    fn doctor_json(
        &self,
        _host: IntegrationHost,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<Value, String> {
        Ok(json!({}))
    }
}

impl PluginSetupRunner for GenerationAwareRunner {
    fn setup(
        &self,
        _host: IntegrationHost,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<(), String> {
        panic!("generation-aware setup entry point was bypassed")
    }

    fn setup_with_generation(
        &self,
        _host: IntegrationHost,
        _gateway_url: &str,
        _plugin_root: &Path,
        generation_token: Option<&str>,
    ) -> Result<(), String> {
        self.calls
            .borrow_mut()
            .push(format!("setup:{}", generation_token.unwrap_or("missing")));
        Ok(())
    }

    fn uninstall(
        &self,
        _host: IntegrationHost,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<(), String> {
        Ok(())
    }

    fn doctor(
        &self,
        _host: IntegrationHost,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<(), String> {
        panic!("generation-aware doctor entry point was bypassed")
    }

    fn doctor_with_generation(
        &self,
        _host: IntegrationHost,
        _gateway_url: &str,
        _plugin_root: &Path,
        generation_token: Option<&str>,
    ) -> Result<(), String> {
        self.calls
            .borrow_mut()
            .push(format!("doctor:{}", generation_token.unwrap_or("missing")));
        Ok(())
    }

    fn doctor_json(
        &self,
        _host: IntegrationHost,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<Value, String> {
        Ok(json!({}))
    }
}

#[test]
fn setup_runner_defaults_are_explicit_no_ops() {
    let runner = DefaultsOnlyRunner;

    assert!(runner.snapshot(IntegrationHost::Codex).unwrap().is_none());
    runner.restore_snapshot(&PluginSetupSnapshot::Mock).unwrap();
    runner.refresh_gateway().unwrap();
}

#[test]
fn setup_and_doctor_receive_the_installer_verified_generation() {
    let dir = tempfile::tempdir().unwrap();
    let layout = PluginLayout::new(IntegrationHost::Codex, dir.path());
    let options = PluginInstallOptions {
        install_dir: dir.path().to_owned(),
        operation_lock_dir: dir.path().join("locks"),
        force: false,
        dry_run: false,
        skip_doctor: false,
    };
    let runner = GenerationAwareRunner::default();

    run_plugin_setup_with_generation(
        IntegrationHost::Codex,
        &layout,
        &options,
        &runner,
        Some("generation-a"),
    )
    .unwrap();
    run_plugin_doctor_with_generation(
        IntegrationHost::Codex,
        &layout.plugin_root,
        &options,
        &runner,
        Some("generation-a"),
    )
    .unwrap();

    assert_eq!(
        *runner.calls.borrow(),
        ["setup:generation-a", "doctor:generation-a"]
    );
}

#[test]
fn setup_descriptions_reject_unexpanded_hosts_and_unknown_actions() {
    assert!(
        std::panic::catch_unwind(|| setup_action_description(IntegrationHost::All, "configure"))
            .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| setup_action_description(IntegrationHost::Codex, "unknown"))
            .is_err()
    );

    let runner = RealPluginSetupRunner;
    let root = Path::new("unused");
    assert!(std::panic::catch_unwind(|| runner.snapshot(IntegrationHost::All)).is_err());
    assert!(
        std::panic::catch_unwind(|| runner.setup(IntegrationHost::All, DEFAULT_GATEWAY_URL, root))
            .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| runner.uninstall(
            IntegrationHost::All,
            DEFAULT_GATEWAY_URL,
            root
        ))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| runner.doctor(IntegrationHost::All, DEFAULT_GATEWAY_URL, root))
            .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| runner.doctor_json(
            IntegrationHost::All,
            DEFAULT_GATEWAY_URL,
            root
        ))
        .is_err()
    );
}
