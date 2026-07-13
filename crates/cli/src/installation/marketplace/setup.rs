// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host setup, restore, and doctor delegation.

use crate::agents::CodingAgent;
use serde_json::Value;
use std::path::Path;

use super::DEFAULT_GATEWAY_URL;
use super::state::PluginInstallOptions;
use super::state::PluginLayout;

#[cfg(test)]
pub(super) fn run_plugin_setup(
    host: CodingAgent,
    layout: &PluginLayout,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    run_plugin_setup_with_generation(host, layout, options, setup_runner, None)
}

pub(super) fn run_plugin_setup_with_generation(
    host: CodingAgent,
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
    host: CodingAgent,
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
    host: CodingAgent,
    plugin_root: &Path,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    run_plugin_doctor_with_generation(host, plugin_root, options, setup_runner, None)
}

pub(super) fn run_plugin_doctor_with_generation(
    host: CodingAgent,
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
    host: CodingAgent,
    plugin_root: &Path,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<Value, String> {
    setup_runner.doctor_json(host, DEFAULT_GATEWAY_URL, plugin_root)
}

pub(super) fn setup_action_description(host: CodingAgent, action: &str) -> String {
    crate::agents::setup_action_description(host, action)
}

pub(super) trait PluginSetupRunner {
    fn snapshot(&self, _host: CodingAgent) -> Result<Option<PluginSetupSnapshot>, String> {
        Ok(None)
    }

    fn restore_snapshot(&self, _snapshot: &PluginSetupSnapshot) -> Result<(), String> {
        Ok(())
    }

    fn refresh_gateway(&self) -> Result<(), String> {
        Ok(())
    }

    fn setup(&self, host: CodingAgent, gateway_url: &str, plugin_root: &Path)
    -> Result<(), String>;
    fn setup_with_generation(
        &self,
        host: CodingAgent,
        gateway_url: &str,
        plugin_root: &Path,
        _generation_token: Option<&str>,
    ) -> Result<(), String> {
        self.setup(host, gateway_url, plugin_root)
    }
    fn uninstall(
        &self,
        host: CodingAgent,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String>;
    fn doctor(
        &self,
        host: CodingAgent,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String>;
    fn doctor_with_generation(
        &self,
        host: CodingAgent,
        gateway_url: &str,
        plugin_root: &Path,
        _generation_token: Option<&str>,
    ) -> Result<(), String> {
        self.doctor(host, gateway_url, plugin_root)
    }
    fn doctor_json(
        &self,
        host: CodingAgent,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<Value, String>;
}

pub(super) struct RealPluginSetupRunner;

pub(super) use crate::agents::SetupSnapshot as PluginSetupSnapshot;

impl PluginSetupRunner for RealPluginSetupRunner {
    fn snapshot(&self, host: CodingAgent) -> Result<Option<PluginSetupSnapshot>, String> {
        crate::agents::snapshot_setup(host).map(Some)
    }

    fn restore_snapshot(&self, snapshot: &PluginSetupSnapshot) -> Result<(), String> {
        crate::agents::restore_setup_snapshot(snapshot)
    }

    fn refresh_gateway(&self) -> Result<(), String> {
        crate::agents::stop_plugin_gateway()
    }

    fn setup(
        &self,
        host: CodingAgent,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String> {
        self.setup_with_generation(host, gateway_url, plugin_root, None)
    }

    fn setup_with_generation(
        &self,
        host: CodingAgent,
        gateway_url: &str,
        plugin_root: &Path,
        generation_token: Option<&str>,
    ) -> Result<(), String> {
        crate::agents::setup_marketplace_plugin(host, gateway_url, plugin_root, generation_token)
    }

    fn uninstall(
        &self,
        host: CodingAgent,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String> {
        crate::agents::uninstall_marketplace_plugin(host, gateway_url, plugin_root)
    }

    fn doctor(
        &self,
        host: CodingAgent,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String> {
        self.doctor_with_generation(host, gateway_url, plugin_root, None)
    }

    fn doctor_with_generation(
        &self,
        host: CodingAgent,
        gateway_url: &str,
        plugin_root: &Path,
        generation_token: Option<&str>,
    ) -> Result<(), String> {
        crate::agents::doctor_marketplace_plugin(host, gateway_url, plugin_root, generation_token)
    }

    fn doctor_json(
        &self,
        host: CodingAgent,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<Value, String> {
        crate::agents::doctor_marketplace_plugin_json(host, gateway_url, plugin_root)
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/plugin_install_setup_tests.rs"]
mod tests;
