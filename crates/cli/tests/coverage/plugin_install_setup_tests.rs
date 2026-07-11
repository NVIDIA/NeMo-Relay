// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};

use super::*;

struct DefaultsOnlyRunner;

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

#[test]
fn setup_runner_defaults_are_explicit_no_ops() {
    let runner = DefaultsOnlyRunner;

    assert!(runner.snapshot(IntegrationHost::Codex).unwrap().is_none());
    runner.restore_snapshot(&PluginSetupSnapshot::Mock).unwrap();
    runner.refresh_gateway().unwrap();
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
