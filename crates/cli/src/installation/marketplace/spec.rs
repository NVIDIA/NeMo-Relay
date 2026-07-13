// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agent-owned behavior required by the shared marketplace transaction.

use std::path::Path;

use serde_json::Value;

use super::host::{CommandRunner, HostRegistrationReport};
use super::state::PluginInstallOptions;

pub(crate) trait MarketplaceHost: Copy {
    fn install_arg(self) -> &'static str;
    fn label(self) -> &'static str;
    fn executable(self) -> &'static str;
    fn validate_version_output(self, output: &str) -> Result<(), String>;
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
}
