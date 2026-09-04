// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{LazyLock, Mutex};

use nemo_relay::plugin::dynamic::PluginHostActivation;
use nemo_relay::plugin::{ConfigReport, PluginConfig, PluginError, Result};

static TEST_PLUGIN_HOST: LazyLock<Mutex<Option<PluginHostActivation>>> =
    LazyLock::new(|| Mutex::new(None));

pub async fn test_initialize_plugin_host_exact(config: PluginConfig) -> Result<ConfigReport> {
    test_close_plugin_host()?;
    let activation = PluginHostActivation::initialize_exact(config).await?;
    let report = activation.report().config;
    *TEST_PLUGIN_HOST.lock().map_err(|error| {
        PluginError::Internal(format!("test plugin host lock poisoned: {error}"))
    })? = Some(activation);
    Ok(report)
}

pub fn test_close_plugin_host() -> Result<()> {
    let activation = TEST_PLUGIN_HOST
        .lock()
        .map_err(|error| PluginError::Internal(format!("test plugin host lock poisoned: {error}")))?
        .take();
    let Some(mut activation) = activation else {
        return Ok(());
    };
    match activation.close() {
        Ok(()) => Ok(()),
        Err(error) => {
            *TEST_PLUGIN_HOST.lock().map_err(|lock_error| {
                PluginError::Internal(format!("test plugin host lock poisoned: {lock_error}"))
            })? = Some(activation);
            Err(error)
        }
    }
}

#[allow(dead_code)]
pub fn test_plugin_host_report() -> Option<ConfigReport> {
    TEST_PLUGIN_HOST
        .lock()
        .ok()
        .and_then(|host| host.as_ref().map(|host| host.report().config))
}
