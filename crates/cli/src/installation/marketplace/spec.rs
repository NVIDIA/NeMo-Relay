// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agent-owned behavior required by the shared marketplace transaction.

use std::path::Path;

use serde_json::Value;

use super::host::{CommandRunner, HostRegistrationReport};
use super::state::PluginInstallOptions;

pub(crate) enum PluginSetupSnapshot {
    Callback(Box<dyn Fn() -> Result<(), String>>),
    #[cfg(test)]
    Mock,
}

impl PluginSetupSnapshot {
    pub(crate) fn new(restore: impl Fn() -> Result<(), String> + 'static) -> Self {
        Self::Callback(Box::new(restore))
    }

    pub(crate) fn restore(&self) -> Result<(), String> {
        match self {
            Self::Callback(restore) => restore(),
            #[cfg(test)]
            Self::Mock => Ok(()),
        }
    }
}

pub(crate) trait MarketplaceHost: Copy {
    fn install_arg(self) -> &'static str;
    fn label(self) -> &'static str;
    fn executable(self) -> &'static str;
    fn validate_version_output(self, output: &str) -> Result<(), String>;
    fn version_requirement(self) -> String;
    fn marketplace_manifest_relative(self) -> &'static [&'static str];
    fn plugin_manifest_relative(self) -> &'static [&'static str];
    fn marketplace_manifest(self, marketplace: &str, plugin: &str) -> Value;
    fn plugin_manifest(self, plugin: &str) -> Value;
    fn plugin_mcp_config(self, server: Value) -> Result<Value, String>;
    fn plugin_hooks(
        self,
        relay: &Path,
        generation_fence: &Path,
        generation_token: &str,
    ) -> Result<Value, String>;
    fn plugin_registration_args(self, plugin_id: &str) -> Vec<String>;
    fn plugin_removal_args(self, plugin_name: &str, plugin_id: &str) -> Vec<String>;
    fn registration_report(
        self,
        options: &PluginInstallOptions,
        runner: &dyn CommandRunner,
    ) -> Result<HostRegistrationReport, String>;
    fn setup_may_mutate_before_success(self) -> bool;
    fn unsafe_generation_fence_error(self, problem: &str) -> String;
    fn accepts_legacy_hook_only_plugin(self) -> bool;
    fn accepts_mcp_environment_superset(self) -> bool;
    fn local_install_exists(
        self,
        marketplace_root: &Path,
        plugin_root: &Path,
        plugin_manifest: &Path,
        generation_fence: &Path,
    ) -> bool;
    fn setup_action_description(self, action: &str) -> String;
    fn snapshot_setup(self) -> Result<Option<PluginSetupSnapshot>, String>;
    fn setup_plugin(
        self,
        gateway_url: &str,
        plugin_root: &Path,
        generation_token: Option<&str>,
    ) -> Result<(), String>;
    fn uninstall_plugin(self, gateway_url: &str, plugin_root: &Path) -> Result<(), String>;
    fn doctor_plugin(
        self,
        gateway_url: &str,
        plugin_root: &Path,
        generation_token: Option<&str>,
    ) -> Result<(), String>;
    fn doctor_plugin_json(self, gateway_url: &str, plugin_root: &Path) -> Result<Value, String>;
}
