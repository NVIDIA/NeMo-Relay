// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin-shim setup, restore, and doctor delegation.

use crate::config::{CodingAgent, PluginHost};
use crate::plugin_shim;
use serde_json::Value;
use std::path::Path;

use super::DEFAULT_GATEWAY_URL;
use super::state::PluginInstallOptions;
use super::state::PluginLayout;

pub(super) fn run_plugin_setup(
    host: PluginHost,
    layout: &PluginLayout,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    if options.dry_run {
        println!("{}", setup_action_description(host, "configure"));
        return Ok(());
    }
    setup_runner.setup(host, DEFAULT_GATEWAY_URL, &layout.plugin_root)
}

pub(super) fn run_plugin_uninstall(
    host: PluginHost,
    plugin_root: &Path,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    if options.dry_run {
        println!("{}", setup_action_description(host, "restore"));
        return Ok(());
    }
    setup_runner.uninstall(host, DEFAULT_GATEWAY_URL, plugin_root)
}

pub(super) fn run_plugin_doctor(
    host: PluginHost,
    plugin_root: &Path,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    if options.dry_run {
        println!("{}", setup_action_description(host, "doctor"));
        return Ok(());
    }
    setup_runner.doctor(host, DEFAULT_GATEWAY_URL, plugin_root)
}

pub(super) fn run_plugin_doctor_json(
    host: PluginHost,
    plugin_root: &Path,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<Value, String> {
    setup_runner.doctor_json(host, DEFAULT_GATEWAY_URL, plugin_root)
}

pub(super) fn setup_action_description(host: PluginHost, action: &str) -> String {
    match (host, action) {
        (PluginHost::Codex, "configure") => {
            "configure Codex provider and trust plugin-owned hooks".into()
        }
        (PluginHost::Codex, "restore") => "remove Codex provider and plugin hook trust".into(),
        (PluginHost::Codex, "doctor") => "check Codex provider and plugin-owned hooks".into(),
        (PluginHost::ClaudeCode, "configure") => {
            "enable Claude Code provider routing through NeMo Relay".into()
        }
        (PluginHost::ClaudeCode, "restore") => {
            "restore Claude Code provider routing from NeMo Relay backup".into()
        }
        (PluginHost::ClaudeCode, "doctor") => "check Claude Code provider routing".into(),
        (PluginHost::All, _) => unreachable!("all is expanded before plugin setup"),
        (_, _) => unreachable!("unsupported setup action"),
    }
}

pub(super) trait PluginSetupRunner {
    fn snapshot(&self, _host: PluginHost) -> Result<Option<PluginSetupSnapshot>, String> {
        Ok(None)
    }

    fn restore_snapshot(&self, _snapshot: &PluginSetupSnapshot) -> Result<(), String> {
        Ok(())
    }

    fn refresh_gateway(&self, _host: PluginHost) -> Result<(), String> {
        Ok(())
    }

    fn setup(&self, host: PluginHost, gateway_url: &str, plugin_root: &Path) -> Result<(), String>;
    fn uninstall(
        &self,
        host: PluginHost,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String>;
    fn doctor(&self, host: PluginHost, gateway_url: &str, plugin_root: &Path)
    -> Result<(), String>;
    fn doctor_json(
        &self,
        host: PluginHost,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<Value, String>;
}

pub(super) struct RealPluginSetupRunner;

pub(super) enum PluginSetupSnapshot {
    Codex(plugin_shim::CodexSetupSnapshot),
    Claude(plugin_shim::ClaudeSetupSnapshot),
    #[cfg(test)]
    Mock,
}

impl PluginSetupRunner for RealPluginSetupRunner {
    fn snapshot(&self, host: PluginHost) -> Result<Option<PluginSetupSnapshot>, String> {
        match host {
            PluginHost::Codex => plugin_shim::snapshot_codex_setup()
                .map(PluginSetupSnapshot::Codex)
                .map(Some),
            PluginHost::ClaudeCode => plugin_shim::snapshot_claude_setup()
                .map(PluginSetupSnapshot::Claude)
                .map(Some),
            PluginHost::All => unreachable!("all is expanded before plugin setup"),
        }
    }

    fn restore_snapshot(&self, snapshot: &PluginSetupSnapshot) -> Result<(), String> {
        match snapshot {
            PluginSetupSnapshot::Codex(snapshot) => plugin_shim::restore_codex_setup(snapshot),
            PluginSetupSnapshot::Claude(snapshot) => plugin_shim::restore_claude_setup(snapshot),
            #[cfg(test)]
            PluginSetupSnapshot::Mock => Ok(()),
        }
    }

    fn refresh_gateway(&self, host: PluginHost) -> Result<(), String> {
        match host {
            PluginHost::Codex => plugin_shim::stop_plugin_gateway(CodingAgent::Codex),
            PluginHost::ClaudeCode => plugin_shim::stop_plugin_gateway(CodingAgent::ClaudeCode),
            PluginHost::All => unreachable!("all is expanded before plugin setup"),
        }
    }

    fn setup(&self, host: PluginHost, gateway_url: &str, plugin_root: &Path) -> Result<(), String> {
        match host {
            PluginHost::Codex => plugin_shim::install_codex_plugin(gateway_url, plugin_root),
            PluginHost::ClaudeCode => plugin_shim::enable_claude_provider(gateway_url),
            PluginHost::All => unreachable!("all is expanded before plugin setup"),
        }
    }

    fn uninstall(
        &self,
        host: PluginHost,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String> {
        match host {
            PluginHost::Codex => plugin_shim::uninstall_codex_plugin(gateway_url, plugin_root),
            PluginHost::ClaudeCode => plugin_shim::restore_claude_provider(gateway_url),
            PluginHost::All => unreachable!("all is expanded before plugin uninstall"),
        }
    }

    fn doctor(
        &self,
        host: PluginHost,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String> {
        match host {
            PluginHost::Codex => {
                plugin_shim::doctor_plugin(CodingAgent::Codex, gateway_url, plugin_root)
            }
            PluginHost::ClaudeCode => {
                plugin_shim::doctor_plugin(CodingAgent::ClaudeCode, gateway_url, plugin_root)
            }
            PluginHost::All => unreachable!("all is expanded before plugin doctor"),
        }
    }

    fn doctor_json(
        &self,
        host: PluginHost,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<Value, String> {
        match host {
            PluginHost::Codex => {
                plugin_shim::doctor_plugin_json(CodingAgent::Codex, gateway_url, plugin_root)
            }
            PluginHost::ClaudeCode => {
                plugin_shim::doctor_plugin_json(CodingAgent::ClaudeCode, gateway_url, plugin_root)
            }
            PluginHost::All => unreachable!("all is expanded before plugin doctor"),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/plugin_install_setup_tests.rs"]
mod tests;
