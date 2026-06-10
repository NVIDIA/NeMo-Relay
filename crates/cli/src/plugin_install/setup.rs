// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin-shim setup, restore, and doctor delegation.

use crate::config::{CodingAgent, PluginHost};
use crate::plugin_shim;
use serde_json::Value;

use super::DEFAULT_GATEWAY_URL;
use super::state::PluginInstallOptions;

pub(super) fn run_plugin_setup(
    host: PluginHost,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    if options.dry_run {
        println!("{}", setup_action_description(host, "configure"));
        return Ok(());
    }
    setup_runner.setup(host, DEFAULT_GATEWAY_URL)
}

pub(super) fn run_plugin_uninstall(
    host: PluginHost,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    if options.dry_run {
        println!("{}", setup_action_description(host, "restore"));
        return Ok(());
    }
    setup_runner.uninstall(host, DEFAULT_GATEWAY_URL)
}

pub(super) fn run_plugin_doctor(
    host: PluginHost,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    if options.dry_run {
        println!("{}", setup_action_description(host, "doctor"));
        return Ok(());
    }
    setup_runner.doctor(host, DEFAULT_GATEWAY_URL)
}

pub(super) fn run_plugin_doctor_json(
    host: PluginHost,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<Value, String> {
    setup_runner.doctor_json(host, DEFAULT_GATEWAY_URL)
}

pub(super) fn setup_action_description(host: PluginHost, action: &str) -> String {
    match (host, action) {
        (PluginHost::Codex, "configure") => {
            "configure Codex provider and hook-supervised lazy startup".into()
        }
        (PluginHost::Codex, "restore") => {
            "restore Codex provider and generated hook configuration".into()
        }
        (PluginHost::Codex, "doctor") => "check Codex provider and generated hooks".into(),
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
    fn setup(&self, host: PluginHost, gateway_url: &str) -> Result<(), String>;
    fn uninstall(&self, host: PluginHost, gateway_url: &str) -> Result<(), String>;
    fn doctor(&self, host: PluginHost, gateway_url: &str) -> Result<(), String>;
    fn doctor_json(&self, host: PluginHost, gateway_url: &str) -> Result<Value, String>;
}

pub(super) struct RealPluginSetupRunner;

impl PluginSetupRunner for RealPluginSetupRunner {
    fn setup(&self, host: PluginHost, gateway_url: &str) -> Result<(), String> {
        match host {
            PluginHost::Codex => plugin_shim::install_codex_plugin(gateway_url),
            PluginHost::ClaudeCode => plugin_shim::enable_claude_provider(gateway_url),
            PluginHost::All => unreachable!("all is expanded before plugin setup"),
        }
    }

    fn uninstall(&self, host: PluginHost, gateway_url: &str) -> Result<(), String> {
        match host {
            PluginHost::Codex => plugin_shim::uninstall_codex_plugin(gateway_url),
            PluginHost::ClaudeCode => plugin_shim::restore_claude_provider(gateway_url),
            PluginHost::All => unreachable!("all is expanded before plugin uninstall"),
        }
    }

    fn doctor(&self, host: PluginHost, gateway_url: &str) -> Result<(), String> {
        match host {
            PluginHost::Codex => plugin_shim::doctor_plugin(CodingAgent::Codex, gateway_url),
            PluginHost::ClaudeCode => {
                plugin_shim::doctor_plugin(CodingAgent::ClaudeCode, gateway_url)
            }
            PluginHost::All => unreachable!("all is expanded before plugin doctor"),
        }
    }

    fn doctor_json(&self, host: PluginHost, gateway_url: &str) -> Result<Value, String> {
        match host {
            PluginHost::Codex => plugin_shim::doctor_plugin_json(CodingAgent::Codex, gateway_url),
            PluginHost::ClaudeCode => {
                plugin_shim::doctor_plugin_json(CodingAgent::ClaudeCode, gateway_url)
            }
            PluginHost::All => unreachable!("all is expanded before plugin doctor"),
        }
    }
}
