// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Default plugin composition shared by SDKs and the CLI.
pub mod exporter;
pub mod plugin;

/// Install default services without activating configured plugins.
pub fn initialize_runtime() {
    nemo_relay_plugin_host::composition::install_builtin_extensions(register_components);
    nemo_relay_plugin_host::initialize_runtime();
}

fn register_components() -> nemo_relay_plugin_host::Result<()> {
    nemo_relay_adaptive::plugin_component::register_adaptive_component()?;
    nemo_relay_pii_redaction::component::register_pii_redaction_component()
}

/// On-demand resource collection through the default composition.
pub mod resource_metrics {
    pub use nemo_relay_plugin_host::resource_metrics::RESOURCE_METRICS_EVENT_NAME;

    /// Acquire a fresh resource snapshot on a Tokio runtime.
    pub async fn collect() -> nemo_relay_types::error::Result<
        nemo_relay_types::api::resource_metrics::ResourceMetricsSnapshot,
    > {
        crate::initialize_runtime();
        nemo_relay_plugin_host::resource_metrics::collect().await
    }
}
