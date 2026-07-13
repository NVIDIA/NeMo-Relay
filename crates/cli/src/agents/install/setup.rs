// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host setup, restore, and doctor delegation.

use crate::agents::host;
use crate::configuration::{CodingAgent, IntegrationHost};
use serde_json::Value;
use std::path::Path;

use super::DEFAULT_GATEWAY_URL;
use super::state::PluginInstallOptions;
use super::state::PluginLayout;

#[cfg(test)]
pub(super) fn run_plugin_setup(
    host: IntegrationHost,
    layout: &PluginLayout,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    run_plugin_setup_with_generation(host, layout, options, setup_runner, None)
}

pub(super) fn run_plugin_setup_with_generation(
    host: IntegrationHost,
    layout: &PluginLayout,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
    generation_token: Option<&str>,
) -> Result<(), String> {
    if options.dry_run {
        println!("{}", setup_action_description(host, "configure"));
        return Ok(());
    }
    setup_runner.setup_with_generation(
        host,
        DEFAULT_GATEWAY_URL,
        &layout.plugin_root,
        generation_token,
    )
}

pub(super) fn run_plugin_uninstall(
    host: IntegrationHost,
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

#[cfg(test)]
pub(super) fn run_plugin_doctor(
    host: IntegrationHost,
    plugin_root: &Path,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    run_plugin_doctor_with_generation(host, plugin_root, options, setup_runner, None)
}

pub(super) fn run_plugin_doctor_with_generation(
    host: IntegrationHost,
    plugin_root: &Path,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
    generation_token: Option<&str>,
) -> Result<(), String> {
    if options.dry_run {
        println!("{}", setup_action_description(host, "doctor"));
        return Ok(());
    }
    setup_runner.doctor_with_generation(host, DEFAULT_GATEWAY_URL, plugin_root, generation_token)
}

pub(super) fn run_plugin_doctor_json(
    host: IntegrationHost,
    plugin_root: &Path,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<Value, String> {
    setup_runner.doctor_json(host, DEFAULT_GATEWAY_URL, plugin_root)
}

pub(super) fn setup_action_description(host: IntegrationHost, action: &str) -> String {
    match (host, action) {
        (IntegrationHost::Codex, "configure") => {
            "configure Codex provider and trust plugin-owned hooks".into()
        }
        (IntegrationHost::Codex, "restore") => "remove Codex provider and plugin hook trust".into(),
        (IntegrationHost::Codex, "doctor") => "check Codex provider and plugin-owned hooks".into(),
        (IntegrationHost::ClaudeCode, "configure") => {
            "enable Claude Code provider routing through NeMo Relay".into()
        }
        (IntegrationHost::ClaudeCode, "restore") => {
            "restore Claude Code provider routing from NeMo Relay backup".into()
        }
        (IntegrationHost::ClaudeCode, "doctor") => "check Claude Code provider routing".into(),
        (IntegrationHost::All, _) => unreachable!("all is expanded before plugin setup"),
        (_, _) => unreachable!("unsupported setup action"),
    }
}

pub(super) trait PluginSetupRunner {
    fn snapshot(&self, _host: IntegrationHost) -> Result<Option<PluginSetupSnapshot>, String> {
        Ok(None)
    }

    fn restore_snapshot(&self, _snapshot: &PluginSetupSnapshot) -> Result<(), String> {
        Ok(())
    }

    fn refresh_gateway(&self) -> Result<(), String> {
        Ok(())
    }

    fn setup(
        &self,
        host: IntegrationHost,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String>;
    fn setup_with_generation(
        &self,
        host: IntegrationHost,
        gateway_url: &str,
        plugin_root: &Path,
        _generation_token: Option<&str>,
    ) -> Result<(), String> {
        self.setup(host, gateway_url, plugin_root)
    }
    fn uninstall(
        &self,
        host: IntegrationHost,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String>;
    fn doctor(
        &self,
        host: IntegrationHost,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String>;
    fn doctor_with_generation(
        &self,
        host: IntegrationHost,
        gateway_url: &str,
        plugin_root: &Path,
        _generation_token: Option<&str>,
    ) -> Result<(), String> {
        self.doctor(host, gateway_url, plugin_root)
    }
    fn doctor_json(
        &self,
        host: IntegrationHost,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<Value, String>;
}

pub(super) struct RealPluginSetupRunner;

pub(super) enum PluginSetupSnapshot {
    Codex(host::CodexSetupSnapshot),
    Claude(host::ClaudeSetupSnapshot),
    #[cfg(test)]
    Mock,
}

impl PluginSetupRunner for RealPluginSetupRunner {
    fn snapshot(&self, host: IntegrationHost) -> Result<Option<PluginSetupSnapshot>, String> {
        match host {
            IntegrationHost::Codex => host::snapshot_codex_setup()
                .map(PluginSetupSnapshot::Codex)
                .map(Some),
            IntegrationHost::ClaudeCode => host::snapshot_claude_setup()
                .map(PluginSetupSnapshot::Claude)
                .map(Some),
            IntegrationHost::Hermes | IntegrationHost::All => {
                unreachable!("all is expanded before plugin setup")
            }
        }
    }

    fn restore_snapshot(&self, snapshot: &PluginSetupSnapshot) -> Result<(), String> {
        match snapshot {
            PluginSetupSnapshot::Codex(snapshot) => host::restore_codex_setup(snapshot),
            PluginSetupSnapshot::Claude(snapshot) => host::restore_claude_setup(snapshot),
            #[cfg(test)]
            PluginSetupSnapshot::Mock => Ok(()),
        }
    }

    fn refresh_gateway(&self) -> Result<(), String> {
        host::stop_plugin_gateway()
    }

    fn setup(
        &self,
        host: IntegrationHost,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String> {
        self.setup_with_generation(host, gateway_url, plugin_root, None)
    }

    fn setup_with_generation(
        &self,
        host: IntegrationHost,
        gateway_url: &str,
        plugin_root: &Path,
        generation_token: Option<&str>,
    ) -> Result<(), String> {
        match host {
            IntegrationHost::Codex => host::install_codex_plugin_with_generation(
                gateway_url,
                plugin_root,
                generation_token,
            ),
            IntegrationHost::ClaudeCode => host::enable_claude_provider(gateway_url),
            IntegrationHost::Hermes | IntegrationHost::All => {
                unreachable!("all is expanded before plugin setup")
            }
        }
    }

    fn uninstall(
        &self,
        host: IntegrationHost,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String> {
        match host {
            IntegrationHost::Codex => host::uninstall_codex_plugin(gateway_url, plugin_root),
            IntegrationHost::ClaudeCode => host::restore_claude_provider(gateway_url),
            IntegrationHost::Hermes | IntegrationHost::All => {
                unreachable!("all is expanded before plugin uninstall")
            }
        }
    }

    fn doctor(
        &self,
        host: IntegrationHost,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String> {
        self.doctor_with_generation(host, gateway_url, plugin_root, None)
    }

    fn doctor_with_generation(
        &self,
        host: IntegrationHost,
        gateway_url: &str,
        plugin_root: &Path,
        generation_token: Option<&str>,
    ) -> Result<(), String> {
        match host {
            IntegrationHost::Codex => host::doctor_plugin_with_generation(
                CodingAgent::Codex,
                gateway_url,
                plugin_root,
                generation_token,
            ),
            IntegrationHost::ClaudeCode => {
                host::doctor_plugin(CodingAgent::ClaudeCode, gateway_url, plugin_root)
            }
            IntegrationHost::Hermes | IntegrationHost::All => {
                unreachable!("all is expanded before plugin doctor")
            }
        }
    }

    fn doctor_json(
        &self,
        host: IntegrationHost,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<Value, String> {
        match host {
            IntegrationHost::Codex => {
                host::doctor_plugin_json(CodingAgent::Codex, gateway_url, plugin_root)
            }
            IntegrationHost::ClaudeCode => {
                host::doctor_plugin_json(CodingAgent::ClaudeCode, gateway_url, plugin_root)
            }
            IntegrationHost::Hermes | IntegrationHost::All => {
                unreachable!("all is expanded before plugin doctor")
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/plugin_install_setup_tests.rs"]
mod tests;
