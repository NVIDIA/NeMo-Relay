// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Default composition and configuration for the independent plugin host.

pub use host::{
    ConfigReport, PluginConfig, PluginError, PluginHostActivation, PluginHostConfig,
    PluginHostReport, Result,
};
use nemo_relay_plugin_host as host;
use std::path::PathBuf;

/// Make the default component implementations available to the runtime host.
pub fn ensure_builtin_plugins_registered() -> Result<()> {
    crate::initialize_runtime();
    host::ensure_builtin_plugins_registered()
}
/// Return component kinds after composing default components.
pub fn list_plugin_kinds() -> Vec<String> {
    crate::initialize_runtime();
    host::list_plugin_kinds()
}
/// Look up a component after composing default components.
pub fn lookup_plugin(
    kind: &str,
) -> Option<std::sync::Arc<dyn nemo_relay_plugin::component::Plugin>> {
    crate::initialize_runtime();
    host::lookup_plugin(kind)
}
/// Validate configured components against the default's component implementations.
pub fn validate_static_plugin_config(config: &PluginConfig) -> ConfigReport {
    crate::initialize_runtime();
    host::validate_static_plugin_config(config)
}
/// Return the host configuration schema.
pub fn plugin_config_schema() -> nemo_relay_types::Json {
    host::plugin_config_schema()
}
/// Validate host configuration with normal discovery and policy resolution.
pub fn validate(
    config: impl Into<PluginHostConfig>,
    additional_plugins_toml: Option<PathBuf>,
) -> Result<PluginHostReport> {
    crate::initialize_runtime();
    host::validate(config, additional_plugins_toml)
}
/// Validate an exact host configuration with default preflight checks.
pub fn validate_exact(config: PluginConfig) -> PluginHostReport {
    crate::initialize_runtime();
    host::validate_exact(config)
}
/// Initialize the plugin host with the default components.
pub async fn initialize(
    config: impl Into<PluginHostConfig>,
    additional_plugins_toml: Option<PathBuf>,
) -> Result<PluginHostActivation> {
    crate::initialize_runtime();
    host::initialize(config, additional_plugins_toml).await
}
/// Initialize the host with explicitly selected runtime diagnostics.
pub async fn initialize_with_diagnostics(
    config: PluginConfig,
    additional_plugins_toml: Option<PathBuf>,
    diagnostics: &nemo_relay_diagnostics::Diagnostics,
) -> Result<PluginHostActivation> {
    crate::initialize_runtime();
    host::initialize_with_diagnostics(config, additional_plugins_toml, diagnostics).await
}
/// Activate an exact configuration without discovering additional files.
pub async fn initialize_exact(config: PluginConfig) -> Result<PluginHostActivation> {
    crate::initialize_runtime();
    host::initialize_exact(config).await
}
/// Activate CLI-verified artifacts under the runtime host's lease.
pub async fn initialize_with_verified_specs<I>(
    config: PluginConfig,
    dynamic_plugins: I,
) -> Result<(PluginHostActivation, ConfigReport)>
where
    I: IntoIterator<Item = host::VerifiedDynamicPluginSpec>,
{
    crate::initialize_runtime();
    host::initialize_with_verified_specs(config, dynamic_plugins).await
}

/// Register an application plugin after reserving the default built-in kinds.
pub fn register_plugin(
    plugin: std::sync::Arc<dyn nemo_relay_plugin::component::Plugin>,
) -> Result<()> {
    ensure_builtin_plugins_registered()?;
    host::register_plugin(plugin)
}
/// Remove a component registration. Default components are restored on next lookup.
pub fn deregister_plugin(kind: &str) -> bool {
    crate::initialize_runtime();
    host::deregister_plugin(kind)
}
