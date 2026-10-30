// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Component services supplied by an SDK or embedding host.

use nemo_relay_types::{
    api::resource_metrics::ResourceMetricsSnapshot,
    codec::{
        pricing::CostEstimator,
        response::{AnnotatedLlmResponse, CostEstimate},
    },
};
use std::sync::{Arc, OnceLock};

/// Callback receiving a resource snapshot without owning a collection implementation.
pub type SnapshotSink = Arc<dyn Fn(&ResourceMetricsSnapshot) + Send + Sync>;

/// Composition callbacks owned by the SDK, independent from the runtime engine.
#[derive(Clone, Copy)]
pub struct RuntimeServices {
    /// Capture the current immutable pricing resolver.
    pub pricing_snapshot: fn() -> Arc<dyn CostEstimator>,
    /// Infer the provider label used for price lookup.
    pub infer_model_provider: fn(&str, Option<&str>) -> Option<String>,
    /// Attach estimates without replacing provider-reported costs.
    pub attach_estimated_cost: fn(&mut AnnotatedLlmResponse, Option<&str>),
}

struct NoPricing;
impl CostEstimator for NoPricing {
    fn estimate_cost_for_provider(
        &self,
        _: Option<&str>,
        _: &str,
        _: &nemo_relay_types::codec::response::Usage,
    ) -> Option<CostEstimate> {
        None
    }
}
impl Default for RuntimeServices {
    fn default() -> Self {
        Self {
            pricing_snapshot: || Arc::new(NoPricing),
            infer_model_provider: |provider, _| Some(provider.to_string()),
            attach_estimated_cost: |_, _| {},
        }
    }
}
static SERVICES: OnceLock<RuntimeServices> = OnceLock::new();
#[cfg(not(test))]
static UNCOMPOSED: std::sync::LazyLock<RuntimeServices> =
    std::sync::LazyLock::new(RuntimeServices::default);

/// Install composition once; subsequent SDK entry points preserve the chosen host.
pub fn install_runtime_services(
    factory: impl FnOnce() -> RuntimeServices,
) -> &'static RuntimeServices {
    SERVICES.get_or_init(factory)
}

/// Return the host services, or an uncomposed engine with no built-in components.
pub fn services() -> &'static RuntimeServices {
    #[cfg(test)]
    {
        SERVICES.get_or_init(crate::test_composition::services)
    }
    #[cfg(not(test))]
    {
        SERVICES.get().unwrap_or(&UNCOMPOSED)
    }
}

#[cfg(test)]
pub(crate) fn ensure_runtime_services() {
    let _ = services();
}
