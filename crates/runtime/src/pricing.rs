// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-supplied model pricing without a dependency on a pricing implementation.

pub use nemo_relay_types::codec::billing::{
    BilledUsage, GeminiBilling, InputSplit, SeparateUsageCounts, ToolUsePromptBilling,
    billed_usage, input_split,
};
use nemo_relay_types::codec::{
    pricing::CostEstimator,
    response::{AnnotatedLlmResponse, CostEstimate, Usage},
};
use std::sync::Arc;

/// Capture a consistent resolver for one finalization operation.
pub fn active_pricing_resolver() -> Arc<dyn CostEstimator> {
    (crate::composition::services().pricing_snapshot)()
}
/// Estimate cost through the configured provider.
pub fn estimate_cost_for_provider(
    provider: Option<&str>,
    model: &str,
    usage: &Usage,
) -> Option<CostEstimate> {
    active_pricing_resolver().estimate_cost_for_provider(provider, model, usage)
}
/// Infer the catalog provider for a normalized model.
pub fn infer_model_provider(default_provider: &str, model: Option<&str>) -> Option<String> {
    (crate::composition::services().infer_model_provider)(default_provider, model)
}
/// Preserve provider-specific billed and reported token accounting.
pub fn estimate_billed_cost_for_provider(
    provider: Option<&str>,
    model: &str,
    billed: &BilledUsage<'_>,
) -> Option<CostEstimate> {
    (crate::composition::services().estimate_billed_cost)(provider, model, billed)
}
/// Attach missing cost information using the host's pricing implementation.
pub fn attach_estimated_cost_for_provider(
    response: &mut AnnotatedLlmResponse,
    provider: Option<&str>,
) {
    (crate::composition::services().attach_estimated_cost)(response, provider);
}
