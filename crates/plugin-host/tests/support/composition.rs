// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Component fixtures for runtime implementation tests.

#![allow(dead_code)]

use nemo_relay_runtime as rt;

pub(crate) fn ensure_runtime_services() {
    rt::composition::install_runtime_services(services);
}

pub(crate) fn services() -> rt::composition::RuntimeServices {
    nemo_relay_plugin_host::composition::install_host_services(|| {
        nemo_relay_plugin_host::composition::HostServices {
            register_builtin_plugins,
            preflight,
        }
    });
    nemo_relay_diagnostics::integration::install_resource_memory(
        nemo_relay_resource_metrics::manager::latest_memory_view,
    );
    rt::composition::RuntimeServices {
        pricing_snapshot: || nemo_relay_model_pricing::active_pricing_resolver(),
        infer_model_provider: nemo_relay_model_pricing::infer_model_provider,
        attach_estimated_cost: nemo_relay_model_pricing::attach_estimated_cost_for_provider,
    }
}

fn register_builtin_plugins() -> nemo_relay_plugin_host::Result<()> {
    use nemo_relay_plugin_host::{PluginError, register_builtin_plugin};
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

fn preflight(
    config: &nemo_relay_plugin_host::PluginConfig,
) -> Vec<nemo_relay_plugin_host::ConfigDiagnostic> {
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
