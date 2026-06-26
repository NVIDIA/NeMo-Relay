// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Re-exported normalized LLM response data types and core pricing helpers.

use serde::Deserialize;

pub use nemo_relay_types::codec::response::*;

pub use super::pricing::{
    CacheReadAccounting, ModelPricing, PricingCatalog, PricingCatalogError, PricingConfig,
    PricingResolver, PricingSource, PricingSourceConfig, PricingUnit, PromptCachePricing,
    TokenPricingRates, active_pricing_resolver, attach_estimated_cost,
    attach_estimated_cost_for_provider, estimate_cost, estimate_cost_for_provider,
    estimate_cost_with_catalog, estimate_cost_with_provider, infer_model_provider,
    pricing_for_model, pricing_for_provider, reset_active_pricing_resolver,
    set_active_pricing_resolver,
};

/// Provider/framework cost object accepted by built-in response codecs.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct RawUsageCost {
    /// Normalized total cost in the supplied currency.
    pub total: Option<f64>,
    /// Uncached prompt/input token cost in the supplied currency.
    pub input: Option<f64>,
    /// Completion/output token cost in the supplied currency.
    pub output: Option<f64>,
    /// Prompt cache read cost in the supplied currency.
    pub cache_read: Option<f64>,
    /// Prompt cache write cost in the supplied currency.
    pub cache_write: Option<f64>,
    /// Optional currency override from provider data.
    pub currency: Option<String>,
    /// Optional provider provenance.
    pub pricing_provider: Option<String>,
    /// Optional model provenance.
    pub pricing_model: Option<String>,
    /// Optional as-of provenance.
    pub pricing_as_of: Option<String>,
    /// Optional source provenance.
    pub pricing_source: Option<String>,
}

pub(crate) fn provider_reported_cost(
    provider_total_cost: Option<f64>,
    cost: Option<RawUsageCost>,
) -> Option<CostEstimate> {
    let cost = cost.unwrap_or_default();
    let provider_total_uses_default_currency = provider_total_cost.is_some();
    let nested_currency_is_default = cost
        .currency
        .as_deref()
        .is_none_or(|currency| currency.eq_ignore_ascii_case("USD"));
    let keep_component_costs = !provider_total_uses_default_currency || nested_currency_is_default;
    let input = keep_component_costs.then_some(cost.input).flatten();
    let output = keep_component_costs.then_some(cost.output).flatten();
    let cache_read = keep_component_costs.then_some(cost.cache_read).flatten();
    let cache_write = keep_component_costs.then_some(cost.cache_write).flatten();
    let has_currency_native_amount = cost.total.is_some()
        || cost.input.is_some()
        || cost.output.is_some()
        || cost.cache_read.is_some()
        || cost.cache_write.is_some();
    let component_total = [input, output, cache_read, cache_write]
        .into_iter()
        .flatten()
        .sum();
    let has_component_cost =
        input.is_some() || output.is_some() || cache_read.is_some() || cache_write.is_some();
    let total = provider_total_cost
        .or(cost.total)
        .or_else(|| has_component_cost.then_some(component_total));

    if total.is_none()
        && input.is_none()
        && output.is_none()
        && cache_read.is_none()
        && cache_write.is_none()
    {
        return None;
    }

    Some(CostEstimate {
        total,
        currency: if provider_total_uses_default_currency {
            default_cost_currency()
        } else if has_currency_native_amount {
            cost.currency.unwrap_or_else(default_cost_currency)
        } else {
            default_cost_currency()
        },
        input,
        output,
        cache_read,
        cache_write,
        source: CostSource::ProviderReported,
        pricing_provider: cost.pricing_provider,
        pricing_model: cost.pricing_model,
        pricing_as_of: cost.pricing_as_of,
        pricing_source: cost.pricing_source,
    })
}

fn default_cost_currency() -> String {
    "USD".into()
}

#[cfg(test)]
#[path = "../../tests/unit/codec/response_tests.rs"]
mod tests;
