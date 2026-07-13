// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-bound setup, restore, and doctor delegation.

use std::path::Path;

use serde_json::Value;

use super::state::{PluginInstallOptions, PluginLayout};
use super::{DEFAULT_GATEWAY_URL, MarketplaceHost, PluginSetupSnapshot};

#[cfg(test)]
pub(super) fn run_plugin_setup(
    host: impl MarketplaceHost,
    layout: &PluginLayout,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    run_plugin_setup_with_generation(host, layout, options, setup_runner, None)
}

pub(super) fn run_plugin_setup_with_generation(
    host: impl MarketplaceHost,
    layout: &PluginLayout,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
    generation_token: Option<&str>,
) -> Result<(), String> {
    if options.dry_run {
        println!("{}", setup_runner.action_description("configure"));
        return Ok(());
    }
    setup_runner.setup_with_generation(
        host.install_arg(),
        DEFAULT_GATEWAY_URL,
        &layout.plugin_root,
        generation_token,
    )
}

pub(super) fn run_plugin_uninstall(
    host: impl MarketplaceHost,
    plugin_root: &Path,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    if options.dry_run {
        println!("{}", setup_runner.action_description("restore"));
        return Ok(());
    }
    setup_runner.uninstall(host.install_arg(), DEFAULT_GATEWAY_URL, plugin_root)
}

#[cfg(test)]
pub(super) fn run_plugin_doctor(
    host: impl MarketplaceHost,
    plugin_root: &Path,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    run_plugin_doctor_with_generation(host, plugin_root, options, setup_runner, None)
}

pub(super) fn run_plugin_doctor_with_generation(
    host: impl MarketplaceHost,
    plugin_root: &Path,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
    generation_token: Option<&str>,
) -> Result<(), String> {
    if options.dry_run {
        println!("{}", setup_runner.action_description("doctor"));
        return Ok(());
    }
    setup_runner.doctor_with_generation(
        host.install_arg(),
        DEFAULT_GATEWAY_URL,
        plugin_root,
        generation_token,
    )
}

pub(super) fn run_plugin_doctor_json(
    host: impl MarketplaceHost,
    plugin_root: &Path,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<Value, String> {
    setup_runner.doctor_json(host.install_arg(), DEFAULT_GATEWAY_URL, plugin_root)
}

pub(super) trait PluginSetupRunner {
    fn action_description(&self, action: &str) -> String {
        action.to_string()
    }

    fn snapshot(&self, _host_arg: &str) -> Result<Option<PluginSetupSnapshot>, String> {
        Ok(None)
    }

    fn restore_snapshot(&self, snapshot: &PluginSetupSnapshot) -> Result<(), String> {
        snapshot.restore()
    }

    fn refresh_gateway(&self) -> Result<(), String> {
        Ok(())
    }

    fn setup(&self, host_arg: &str, gateway_url: &str, plugin_root: &Path) -> Result<(), String>;

    fn setup_with_generation(
        &self,
        host_arg: &str,
        gateway_url: &str,
        plugin_root: &Path,
        _generation_token: Option<&str>,
    ) -> Result<(), String> {
        self.setup(host_arg, gateway_url, plugin_root)
    }

    fn uninstall(
        &self,
        host_arg: &str,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String>;

    fn doctor(&self, host_arg: &str, gateway_url: &str, plugin_root: &Path) -> Result<(), String>;

    fn doctor_with_generation(
        &self,
        host_arg: &str,
        gateway_url: &str,
        plugin_root: &Path,
        _generation_token: Option<&str>,
    ) -> Result<(), String> {
        self.doctor(host_arg, gateway_url, plugin_root)
    }

    fn doctor_json(
        &self,
        host_arg: &str,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<Value, String>;
}

pub(super) struct HostPluginSetupRunner<H: MarketplaceHost> {
    host: H,
}

impl<H: MarketplaceHost> HostPluginSetupRunner<H> {
    pub(super) const fn new(host: H) -> Self {
        Self { host }
    }
}

impl<H: MarketplaceHost> PluginSetupRunner for HostPluginSetupRunner<H> {
    fn action_description(&self, action: &str) -> String {
        self.host.setup_action_description(action)
    }

    fn snapshot(&self, _host_arg: &str) -> Result<Option<PluginSetupSnapshot>, String> {
        self.host.snapshot_setup()
    }

    fn refresh_gateway(&self) -> Result<(), String> {
        crate::agents::stop_plugin_gateway()
    }

    fn setup(&self, _host_arg: &str, gateway_url: &str, plugin_root: &Path) -> Result<(), String> {
        self.host.setup_plugin(gateway_url, plugin_root, None)
    }

    fn setup_with_generation(
        &self,
        _host_arg: &str,
        gateway_url: &str,
        plugin_root: &Path,
        generation_token: Option<&str>,
    ) -> Result<(), String> {
        self.host
            .setup_plugin(gateway_url, plugin_root, generation_token)
    }

    fn uninstall(
        &self,
        _host_arg: &str,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<(), String> {
        self.host.uninstall_plugin(gateway_url, plugin_root)
    }

    fn doctor(&self, _host_arg: &str, gateway_url: &str, plugin_root: &Path) -> Result<(), String> {
        self.host.doctor_plugin(gateway_url, plugin_root, None)
    }

    fn doctor_with_generation(
        &self,
        _host_arg: &str,
        gateway_url: &str,
        plugin_root: &Path,
        generation_token: Option<&str>,
    ) -> Result<(), String> {
        self.host
            .doctor_plugin(gateway_url, plugin_root, generation_token)
    }

    fn doctor_json(
        &self,
        _host_arg: &str,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<Value, String> {
        self.host.doctor_plugin_json(gateway_url, plugin_root)
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/plugin_install_setup_tests.rs"]
mod tests;
