// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-supplied model pricing without a dependency on a pricing implementation.

use nemo_relay_types::codec::{pricing::CostEstimator, response::AnnotatedLlmResponse};
use std::sync::Arc;

/// Capture a consistent resolver for one finalization operation.
pub fn active_pricing_resolver() -> Arc<dyn CostEstimator> {
    (crate::composition::services().pricing_snapshot)()
}
/// Infer the catalog provider for a normalized model.
pub fn infer_model_provider(default_provider: &str, model: Option<&str>) -> Option<String> {
    (crate::composition::services().infer_model_provider)(default_provider, model)
}
/// Attach missing cost information using the host's pricing implementation.
pub fn attach_estimated_cost_for_provider(
    response: &mut AnnotatedLlmResponse,
    provider: Option<&str>,
) {
    (crate::composition::services().attach_estimated_cost)(response, provider);
}

#[cfg(test)]
#[path = "../tests/unit/pricing_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/unit/gemini_generate_content_pricing_tests.rs"]
mod gemini_generate_content_tests;

#[cfg(test)]
#[path = "../tests/unit/anthropic_pricing_tests.rs"]
mod anthropic_tests;

#[cfg(test)]
#[path = "../tests/unit/codec_pricing_parity_tests.rs"]
mod codec_parity_tests;
