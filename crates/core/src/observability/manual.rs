// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared best-effort extraction for non-provider/manual LLM output, used as the
//! fallback by observability exporters before they project into exporter-specific
//! schemas.

use serde_json::Map;

use crate::codec::response::Usage;
use crate::json::Json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ManualCostPolicy {
    #[cfg(feature = "otel")]
    AnyCurrency,
    #[cfg(feature = "openinference")]
    UsdOnly,
    AtifUsdOnly,
}

pub(crate) fn model_name_from_manual_llm_output(output: Option<&Json>) -> Option<&str> {
    output?.as_object()?.get("model").and_then(Json::as_str)
}

pub(crate) fn usage_from_manual_llm_output(output: Option<&Json>) -> Option<Usage> {
    let object = output?.as_object()?;
    let usage = object.get("usage").and_then(Json::as_object);
    let token_usage = object.get("token_usage").and_then(Json::as_object);
    if usage.is_none() && token_usage.is_none() {
        return None;
    }

    let prompt_tokens = first_u64_from_manual_usage(
        usage,
        token_usage,
        &["prompt_tokens", "input_tokens", "inputTokens", "input"],
    );
    let completion_tokens = first_u64_from_manual_usage(
        usage,
        token_usage,
        &[
            "completion_tokens",
            "output_tokens",
            "completionTokens",
            "outputTokens",
            "output",
        ],
    );
    let reported_total_tokens = first_u64_from_manual_usage(
        usage,
        token_usage,
        &["total_tokens", "totalTokens", "total"],
    );
    // Keep the legacy ATIF cache-read alias order for conflicting manual
    // payloads while sharing the scalar reader across exporters.
    let cache_read_tokens =
        first_u64_from_manual_usage(usage, token_usage, &["cached_tokens", "cachedTokens"])
            .or_else(|| {
                first_nested_u64_from_manual_usage(
                    usage,
                    token_usage,
                    "prompt_tokens_details",
                    "cached_tokens",
                )
            })
            .or_else(|| {
                first_nested_u64_from_manual_usage(
                    usage,
                    token_usage,
                    "input_tokens_details",
                    "cached_tokens",
                )
            })
            .or_else(|| {
                first_u64_from_manual_usage(
                    usage,
                    token_usage,
                    &[
                        "cache_read_tokens",
                        "cache_read_input_tokens",
                        "cacheReadTokens",
                        "cacheReadInputTokens",
                        "cacheRead",
                    ],
                )
            });
    let cache_write_tokens = first_u64_from_manual_usage(
        usage,
        token_usage,
        &[
            "cache_write_tokens",
            "cache_creation_input_tokens",
            "cacheWriteTokens",
            "cacheCreationInputTokens",
            "cacheWrite",
        ],
    );

    if prompt_tokens.is_none()
        && completion_tokens.is_none()
        && reported_total_tokens.is_none()
        && cache_read_tokens.is_none()
        && cache_write_tokens.is_none()
    {
        return None;
    }
    let total_tokens =
        normalize_total_tokens(reported_total_tokens, prompt_tokens, completion_tokens);

    Some(Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cache_read_tokens,
        cache_write_tokens,
        cost: None,
    })
}

pub(crate) fn cost_from_manual_llm_output(
    output: Option<&Json>,
    policy: ManualCostPolicy,
) -> Option<(f64, String)> {
    let object = output?.as_object()?;
    let usage = object.get("usage").and_then(Json::as_object);
    let token_usage = object.get("token_usage").and_then(Json::as_object);
    usage
        .and_then(|usage| cost_from_manual_usage(usage, policy))
        .or_else(|| token_usage.and_then(|usage| cost_from_manual_usage(usage, policy)))
}

fn cost_from_manual_usage(
    usage: &Map<String, Json>,
    policy: ManualCostPolicy,
) -> Option<(f64, String)> {
    if let Some(total) = usage.get("cost_usd").and_then(Json::as_f64) {
        return Some((total, "USD".to_string()));
    }
    let cost = usage.get("cost")?.as_object()?;
    let currency = cost.get("currency").and_then(Json::as_str);
    if !policy.accepts_currency(currency, cost) {
        return None;
    }
    let total = cost.get("total").and_then(Json::as_f64).or_else(|| {
        let (has_component, component_total) = ["input", "output", "cache_read", "cache_write"]
            .iter()
            .filter_map(|field| cost.get(*field).and_then(Json::as_f64))
            .fold((false, 0.0), |(_, total), value| (true, total + value));
        has_component.then_some(component_total)
    })?;
    Some((total, currency.unwrap_or("USD").to_string()))
}

impl ManualCostPolicy {
    fn accepts_currency(self, currency: Option<&str>, cost: &Map<String, Json>) -> bool {
        match self {
            #[cfg(feature = "otel")]
            Self::AnyCurrency => true,
            #[cfg(feature = "openinference")]
            Self::UsdOnly => currency
                .map(|currency| currency.eq_ignore_ascii_case("USD"))
                .unwrap_or(true),
            Self::AtifUsdOnly => currency
                .map(|currency| currency.eq_ignore_ascii_case("USD"))
                .unwrap_or_else(|| {
                    let source_is_relay_normalized = cost
                        .get("source")
                        .and_then(Json::as_str)
                        .is_some_and(|source| {
                            matches!(source, "provider_reported" | "model_pricing")
                        });
                    let has_legacy_provider_total =
                        cost.get("total").and_then(Json::as_f64).is_some();
                    source_is_relay_normalized || has_legacy_provider_total
                }),
        }
    }
}

// Keep a reported total only when it is internally consistent with the component
// counts. Deriving a total from components is a provider-specific concern owned by
// the provider codecs, not this manual fallback, so an absent total stays absent.
fn normalize_total_tokens(
    total_tokens: Option<u64>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
) -> Option<u64> {
    let total_tokens = total_tokens?;
    let minimum_total = prompt_tokens
        .unwrap_or(0)
        .saturating_add(completion_tokens.unwrap_or(0));
    if minimum_total == 0 || total_tokens >= minimum_total {
        Some(total_tokens)
    } else {
        None
    }
}

fn first_u64_from_manual_usage(
    usage: Option<&Map<String, Json>>,
    token_usage: Option<&Map<String, Json>>,
    keys: &[&str],
) -> Option<u64> {
    usage
        .and_then(|value| first_u64(value, keys))
        .or_else(|| token_usage.and_then(|value| first_u64(value, keys)))
}

fn first_nested_u64_from_manual_usage(
    usage: Option<&Map<String, Json>>,
    token_usage: Option<&Map<String, Json>>,
    parent_key: &str,
    child_key: &str,
) -> Option<u64> {
    usage
        .and_then(|value| nested_u64(value, parent_key, child_key))
        .or_else(|| token_usage.and_then(|value| nested_u64(value, parent_key, child_key)))
}

fn first_u64(usage: &Map<String, Json>, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| usage.get(*key).and_then(Json::as_u64))
}

fn nested_u64(usage: &Map<String, Json>, parent_key: &str, child_key: &str) -> Option<u64> {
    usage
        .get(parent_key)
        .and_then(Json::as_object)
        .and_then(|details| details.get(child_key))
        .and_then(Json::as_u64)
}

#[cfg(test)]
#[path = "../../tests/unit/observability/manual_tests.rs"]
mod tests;
