// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use nemo_relay::plugin::dynamic::{PluginHostActivation, PluginHostActivationPlan};
use nemo_relay::plugin::{ConfigReport, PluginConfig, PluginError};

use crate::error::{PluginHostConfigError, Result};
use crate::lifecycle::prepare_plugin_host_activation;
use crate::resolver::{PluginFileResolveOptions, resolve_plugin_files};

/// Owns one static or dynamic plugin configuration initialized from plugin files.
///
/// An inactive handle represents successful discovery with no physical input and no caller
/// configuration. Active handles retain Relay's process-wide plugin host lease until cleared.
#[must_use = "dropping the file activation starts best-effort plugin host teardown"]
pub struct PluginFileActivation {
    host: Option<PluginHostActivation>,
    report: ConfigReport,
}

impl PluginFileActivation {
    /// Returns the configuration and runtime diagnostics produced during activation.
    pub fn report(&self) -> &ConfigReport {
        &self.report
    }

    /// Returns whether this handle owns an active process-wide plugin host.
    pub fn is_active(&self) -> bool {
        self.host
            .as_ref()
            .is_some_and(PluginHostActivation::is_active)
    }

    /// Clears configured callbacks before unloading dynamic runtimes and releasing the host.
    pub fn clear(mut self) -> Result<()> {
        if let Some(host) = self.host.take() {
            host.clear()?;
        }
        Ok(())
    }

    /// Activates an already resolved and snapshotted file-backed host plan.
    ///
    /// Embedding hosts that participate in configuration resolution before runtime startup use
    /// this path to preserve the exact snapshot resources used for bootstrap identity.
    #[doc(hidden)]
    pub async fn activate_plan(plan: PluginHostActivationPlan) -> Result<Self> {
        let (host, report) = PluginHostActivation::activate_plan(plan).await?;
        Ok(Self {
            host: Some(host),
            report,
        })
    }

    fn inactive() -> Self {
        Self {
            host: None,
            report: ConfigReport::default(),
        }
    }
}

/// Resolves and activates static components and enabled dynamic plugins from `plugins.toml`.
///
/// `plugin_config_path` replaces the ambient user-level file while project and system layers
/// continue to participate. The optional typed `config` is the highest-precedence static overlay.
/// Dynamic enablement remains controlled by each source's sibling `.dynamic-plugins.json`.
pub async fn initialize_from_plugins_toml(
    config: Option<PluginConfig>,
    plugin_config_path: Option<PathBuf>,
) -> Result<PluginFileActivation> {
    initialize_from_plugins_toml_with_options(
        config,
        PluginFileResolveOptions::from_environment(plugin_config_path),
    )
    .await
}

async fn initialize_from_plugins_toml_with_options(
    config: Option<PluginConfig>,
    options: PluginFileResolveOptions,
) -> Result<PluginFileActivation> {
    let resolved = tokio::task::spawn_blocking(move || resolve_plugin_files(config, options))
        .await
        .map_err(|error| {
            PluginHostConfigError::Relay(PluginError::Internal(format!(
                "plugin file resolution task failed: {error}"
            )))
        })??;

    if !resolved.had_input {
        return Ok(PluginFileActivation::inactive());
    }

    let plan = tokio::task::spawn_blocking(move || prepare_plugin_host_activation(resolved))
        .await
        .map_err(|error| {
            PluginHostConfigError::Relay(PluginError::Internal(format!(
                "plugin lifecycle preparation task failed: {error}"
            )))
        })??;
    PluginFileActivation::activate_plan(plan).await
}

#[cfg(test)]
#[path = "../tests/unit/activation.rs"]
mod tests;
