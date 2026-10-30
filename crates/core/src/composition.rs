// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Default SDK composition over the independent runtime engine.

use nemo_relay_runtime as rt;

pub(crate) fn ensure_runtime_services() {
    rt::composition::install_runtime_services(services);
}

pub(crate) fn services() -> rt::composition::RuntimeServices {
    rt::composition::RuntimeServices {
        register_builtin_plugins,
        preflight,
        pricing_snapshot: || nemo_relay_model_pricing::active_pricing_resolver(),
        infer_model_provider: nemo_relay_model_pricing::infer_model_provider,
        attach_estimated_cost: nemo_relay_model_pricing::attach_estimated_cost_for_provider,
        estimate_billed_cost: nemo_relay_model_pricing::estimate_billed_cost_for_provider,
        resource_memory: nemo_relay_resource_metrics::manager::latest_memory_view,
    }
}

fn register_builtin_plugins() -> rt::plugin::Result<()> {
    use rt::plugin::{PluginError, register_builtin_plugin};
    use std::sync::Arc;
    register_builtin_plugin(Arc::new(
        nemo_relay_observability::plugin::ObservabilityPlugin::new(
            crate::observability_runtime::core_host(),
        ),
    ))?;
    register_builtin_plugin(Arc::new(nemo_relay_model_pricing::plugin::PricingPlugin))?;
    register_builtin_plugin(Arc::new(
        nemo_relay_resource_metrics::plugin::ResourceMetricsPlugin::new(|| {
            rt::api::resource_metrics::polling_snapshot_sink()
                .map_err(|error| PluginError::RegistrationFailed(error.to_string()))
        }),
    ))
}

fn preflight(config: &rt::plugin::PluginConfig) -> Vec<rt::plugin::ConfigDiagnostic> {
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
