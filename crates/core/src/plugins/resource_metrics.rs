// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core activation of the crate-owned component.

use crate::plugin::{PluginError, Result, register_builtin_plugin};
pub use nemo_relay_resource_metrics::plugin::{
    RESOURCE_METRICS_PLUGIN_KIND, ResourceMetricsPlugin,
};
use std::sync::Arc;
pub mod config;
pub(crate) fn register_resource_metrics_component() -> Result<()> {
    register_builtin_plugin(Arc::new(ResourceMetricsPlugin::new(|| {
        crate::api::resource_metrics::polling_snapshot_sink()
            .map_err(|error| PluginError::RegistrationFailed(error.to_string()))
    })))
}
