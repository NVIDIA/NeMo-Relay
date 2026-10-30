// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Default built-in components and execution services.
use nemo_relay_runtime as rt;

/// Install shared default services without activating configured plugins.
/// Explicit services installed by an embedding application remain selected.
pub fn initialize_runtime() {
    crate::composition::install_host_services(|| crate::composition::HostServices {
        register_builtin_plugins,
        preflight,
    });
    nemo_relay_diagnostics::integration::install_resource_memory(
        nemo_relay_resource_metrics::manager::latest_memory_view,
    );
    rt::composition::install_runtime_services(|| rt::composition::RuntimeServices {
        pricing_snapshot: || nemo_relay_model_pricing::active_pricing_resolver(),
        infer_model_provider: nemo_relay_model_pricing::infer_model_provider,
        attach_estimated_cost: nemo_relay_model_pricing::attach_estimated_cost_for_provider,
    });
}

fn register_builtin_plugins() -> crate::Result<()> {
    use std::sync::Arc;
    crate::register_builtin_plugin(Arc::new(
        nemo_relay_observability::plugin::ObservabilityPlugin::new(
            crate::observability_runtime::runtime_host(),
        ),
    ))?;
    crate::register_builtin_plugin(Arc::new(nemo_relay_model_pricing::plugin::PricingPlugin))?;
    crate::register_builtin_plugin(Arc::new(
        nemo_relay_resource_metrics::plugin::ResourceMetricsPlugin::new(|| {
            rt::api::resource_metrics::polling_snapshot_sink()
                .map_err(|error| crate::PluginError::RegistrationFailed(error.to_string()))
        }),
    ))
}

fn preflight(config: &crate::PluginConfig) -> Vec<crate::ConfigDiagnostic> {
    config
        .components
        .iter()
        .filter(|component| {
            component.enabled
                && component.kind == nemo_relay_observability::plugin::OBSERVABILITY_PLUGIN_KIND
        })
        .flat_map(|component| {
            nemo_relay_observability::plugin::validate_observability_plaintext_destinations(
                &component.config,
            )
        })
        .collect()
}
