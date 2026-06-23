// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared manual fallback readers for exporter-local LLM observability projection.

#[cfg(any(feature = "otel", feature = "openinference", test))]
use crate::codec::response::Usage;
use crate::codec::response::{CostEstimate, CostSource};
use crate::json::Json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ManualUsageFields {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub reported_total_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManualUsagePrecedence {
    /// Preserve OTel/OpenInference's existing behavior: try each preferred key
    /// across both maps before moving to weaker aliases.
    #[cfg(any(feature = "otel", feature = "openinference", test))]
    KeyOrder,
    /// Preserve ATIF's selected-map preference: search all aliases in
    /// `token_usage` before using provider-native `usage` for the same field.
    PrimaryMap,
}

#[cfg(any(feature = "otel", feature = "openinference", test))]
pub(crate) fn manual_usage_from_llm_output_with(
    output: Option<&Json>,
    normalize_total_tokens: impl FnOnce(Option<u64>, Option<u64>, Option<u64>) -> Option<u64>,
) -> Option<Usage> {
    let fields = manual_usage_fields_from_llm_output(output)?;
    Some(Usage {
        prompt_tokens: fields.prompt_tokens,
        completion_tokens: fields.completion_tokens,
        total_tokens: normalize_total_tokens(
            fields.reported_total_tokens,
            fields.prompt_tokens,
            fields.completion_tokens,
        ),
        cache_read_tokens: fields.cache_read_tokens,
        cache_write_tokens: fields.cache_write_tokens,
        cost: None,
    })
}

#[cfg(any(feature = "otel", feature = "openinference", test))]
pub(crate) fn manual_usage_fields_from_llm_output(
    output: Option<&Json>,
) -> Option<ManualUsageFields> {
    let object = output?.as_object()?;
    let usage = object.get("usage").and_then(Json::as_object);
    let token_usage = object.get("token_usage").and_then(Json::as_object);
    manual_usage_fields_from_maps(usage, token_usage, ManualUsagePrecedence::KeyOrder)
}

pub(crate) fn manual_usage_fields_from_preferred_token_usage(
    output: Option<&Json>,
) -> Option<ManualUsageFields> {
    let object = output?.as_object()?;
    let usage = object.get("usage").and_then(Json::as_object);
    let token_usage = object.get("token_usage").and_then(Json::as_object);
    manual_usage_fields_from_maps(token_usage, usage, ManualUsagePrecedence::PrimaryMap)
}

fn manual_usage_fields_from_maps(
    primary_usage: Option<&serde_json::Map<String, Json>>,
    secondary_usage: Option<&serde_json::Map<String, Json>>,
    precedence: ManualUsagePrecedence,
) -> Option<ManualUsageFields> {
    if primary_usage.is_none() && secondary_usage.is_none() {
        return None;
    }

    let prompt_tokens = first_u64_from_manual_usage(
        primary_usage,
        secondary_usage,
        &["prompt_tokens", "input_tokens", "inputTokens", "input"],
        precedence,
    );
    let completion_tokens = first_u64_from_manual_usage(
        primary_usage,
        secondary_usage,
        &[
            "completion_tokens",
            "output_tokens",
            "completionTokens",
            "outputTokens",
            "output",
        ],
        precedence,
    );
    let reported_total_tokens = first_u64_from_manual_usage(
        primary_usage,
        secondary_usage,
        &["total_tokens", "totalTokens", "total"],
        precedence,
    );
    let cache_read_tokens = first_u64_from_manual_usage(
        primary_usage,
        secondary_usage,
        &[
            "cache_read_tokens",
            "cached_tokens",
            "cache_read_input_tokens",
            "cacheReadTokens",
            "cachedTokens",
            "cacheReadInputTokens",
            "cacheRead",
        ],
        precedence,
    )
    .or_else(|| {
        first_nested_u64_from_manual_usage(
            primary_usage,
            secondary_usage,
            "input_tokens_details",
            "cached_tokens",
        )
    })
    .or_else(|| {
        first_nested_u64_from_manual_usage(
            primary_usage,
            secondary_usage,
            "prompt_tokens_details",
            "cached_tokens",
        )
    });
    let cache_write_tokens = first_u64_from_manual_usage(
        primary_usage,
        secondary_usage,
        &[
            "cache_write_tokens",
            "cache_creation_input_tokens",
            "cacheWriteTokens",
            "cacheCreationInputTokens",
            "cacheWrite",
        ],
        precedence,
    );

    if prompt_tokens.is_none()
        && completion_tokens.is_none()
        && reported_total_tokens.is_none()
        && cache_read_tokens.is_none()
        && cache_write_tokens.is_none()
    {
        return None;
    }

    Some(ManualUsageFields {
        prompt_tokens,
        completion_tokens,
        reported_total_tokens,
        cache_read_tokens,
        cache_write_tokens,
    })
}

pub(crate) fn manual_model_name_from_llm_output(output: Option<&Json>) -> Option<&str> {
    output?.as_object()?.get("model").and_then(Json::as_str)
}

#[cfg(any(feature = "otel", feature = "openinference", test))]
pub(crate) fn manual_cost_estimate_from_llm_output(output: Option<&Json>) -> Option<CostEstimate> {
    let object = output?.as_object()?;
    let usage = object.get("usage").and_then(Json::as_object);
    let token_usage = object.get("token_usage").and_then(Json::as_object);
    manual_cost_estimate_from_maps(usage, token_usage)
}

pub(crate) fn manual_cost_estimate_from_usage(
    usage: &serde_json::Map<String, Json>,
) -> Option<CostEstimate> {
    cost_estimate_from_manual_usage(usage)
}

#[cfg(any(feature = "openinference", test))]
pub(crate) fn manual_cost_total_for_currency_from_llm_output(
    output: Option<&Json>,
    currency: &str,
) -> Option<f64> {
    let object = output?.as_object()?;
    let usage = object.get("usage").and_then(Json::as_object);
    let token_usage = object.get("token_usage").and_then(Json::as_object);
    manual_cost_total_for_currency_from_maps(usage, token_usage, currency)
}

#[cfg(any(feature = "otel", feature = "openinference", test))]
fn manual_cost_estimate_from_maps(
    primary_usage: Option<&serde_json::Map<String, Json>>,
    secondary_usage: Option<&serde_json::Map<String, Json>>,
) -> Option<CostEstimate> {
    primary_usage
        .and_then(cost_estimate_from_manual_usage)
        .or_else(|| secondary_usage.and_then(cost_estimate_from_manual_usage))
}

#[cfg(any(feature = "openinference", test))]
fn manual_cost_total_for_currency_from_maps(
    primary_usage: Option<&serde_json::Map<String, Json>>,
    secondary_usage: Option<&serde_json::Map<String, Json>>,
    currency: &str,
) -> Option<f64> {
    primary_usage
        .and_then(cost_estimate_from_manual_usage)
        .and_then(|cost| cost.total_or_component_sum_for_currency(currency))
        .or_else(|| {
            secondary_usage
                .and_then(cost_estimate_from_manual_usage)
                .and_then(|cost| cost.total_or_component_sum_for_currency(currency))
        })
}

fn cost_estimate_from_manual_usage(usage: &serde_json::Map<String, Json>) -> Option<CostEstimate> {
    if let Some(total) = usage.get("cost_usd").and_then(Json::as_f64) {
        return Some(CostEstimate {
            total: Some(total),
            currency: "USD".to_string(),
            input: None,
            output: None,
            cache_read: None,
            cache_write: None,
            source: CostSource::ProviderReported,
            pricing_provider: None,
            pricing_model: None,
            pricing_as_of: None,
            pricing_source: None,
        });
    }

    let cost = usage.get("cost")?.as_object()?;
    let estimate = CostEstimate {
        total: cost.get("total").and_then(Json::as_f64),
        currency: cost
            .get("currency")
            .and_then(Json::as_str)
            .unwrap_or("USD")
            .to_string(),
        input: cost.get("input").and_then(Json::as_f64),
        output: cost.get("output").and_then(Json::as_f64),
        cache_read: cost.get("cache_read").and_then(Json::as_f64),
        cache_write: cost.get("cache_write").and_then(Json::as_f64),
        source: CostSource::ProviderReported,
        pricing_provider: None,
        pricing_model: None,
        pricing_as_of: None,
        pricing_source: None,
    };
    estimate.total_or_component_sum().map(|_| estimate)
}

fn first_u64_from_manual_usage(
    usage: Option<&serde_json::Map<String, Json>>,
    token_usage: Option<&serde_json::Map<String, Json>>,
    keys: &[&str],
    precedence: ManualUsagePrecedence,
) -> Option<u64> {
    match precedence {
        #[cfg(any(feature = "otel", feature = "openinference", test))]
        ManualUsagePrecedence::KeyOrder => keys.iter().find_map(|key| {
            usage
                .and_then(|value| value.get(*key).and_then(Json::as_u64))
                .or_else(|| token_usage.and_then(|value| value.get(*key).and_then(Json::as_u64)))
        }),
        ManualUsagePrecedence::PrimaryMap => usage
            .and_then(|value| first_u64(value, keys))
            .or_else(|| token_usage.and_then(|value| first_u64(value, keys))),
    }
}

fn first_nested_u64_from_manual_usage(
    usage: Option<&serde_json::Map<String, Json>>,
    token_usage: Option<&serde_json::Map<String, Json>>,
    parent_key: &str,
    child_key: &str,
) -> Option<u64> {
    usage
        .and_then(|value| nested_u64(value, parent_key, child_key))
        .or_else(|| token_usage.and_then(|value| nested_u64(value, parent_key, child_key)))
}

fn nested_u64(
    usage: &serde_json::Map<String, Json>,
    parent_key: &str,
    child_key: &str,
) -> Option<u64> {
    usage
        .get(parent_key)
        .and_then(Json::as_object)
        .and_then(|details| details.get(child_key))
        .and_then(Json::as_u64)
}

fn first_u64(usage: &serde_json::Map<String, Json>, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| usage.get(*key).and_then(Json::as_u64))
}

pub(crate) fn manual_reasoning_effort_from_llm_input(input: &Json) -> Option<Json> {
    input
        .as_object()?
        .get("reasoning_effort")
        .filter(|value| !value.is_null())
        .cloned()
}

pub(crate) fn manual_reasoning_content_from_llm_output(output: &Json) -> Option<String> {
    output
        .as_object()?
        .get("reasoning")
        .and_then(Json::as_str)
        .map(String::from)
}

pub(crate) fn manual_tool_call_id(tool_call: &Json) -> Option<&str> {
    tool_call
        .get("id")
        .or_else(|| tool_call.get("tool_call_id"))
        .or_else(|| tool_call.get("call_id"))
        .and_then(Json::as_str)
}

pub(crate) fn manual_tool_call_name(tool_call: &Json) -> Option<&str> {
    tool_call
        .get("function")
        .and_then(|function| function.get("name"))
        .and_then(Json::as_str)
        .or_else(|| tool_call.get("name").and_then(Json::as_str))
        .or_else(|| tool_call.get("tool_name").and_then(Json::as_str))
        .or_else(|| tool_call.get("toolName").and_then(Json::as_str))
        .or_else(|| tool_call.get("function_name").and_then(Json::as_str))
}

pub(crate) fn manual_tool_call_arguments(tool_call: &Json) -> Option<&Json> {
    tool_call
        .get("function")
        .and_then(|function| function.get("arguments"))
        .or_else(|| tool_call.get("arguments"))
        .or_else(|| tool_call.get("args"))
        .or_else(|| tool_call.get("input"))
}

#[cfg(any(feature = "openinference", test))]
pub(crate) fn manual_replay_llm_payload(input: &Json) -> Option<&Json> {
    let content = input.as_object().and_then(|object| object.get("content"))?;
    let content_object = content.as_object()?;
    is_openclaw_replay_payload(content_object).then_some(content)
}

#[cfg(any(feature = "openinference", test))]
pub(crate) fn manual_replay_llm_response(output: &Json) -> Option<&Json> {
    output
        .as_object()
        .and_then(|object| object.get("openclaw"))
        .and_then(Json::as_object)
        .map(|_| output)
}

#[cfg(any(feature = "openinference", test))]
fn is_openclaw_replay_payload(content: &serde_json::Map<String, Json>) -> bool {
    content
        .get("source")
        .and_then(Json::as_str)
        .is_some_and(|source| source.starts_with("openclaw."))
        || content.contains_key("placeholderRequest")
}

#[cfg(test)]
mod tests {
    use super::{
        manual_cost_estimate_from_llm_output, manual_cost_estimate_from_usage,
        manual_cost_total_for_currency_from_llm_output, manual_model_name_from_llm_output,
        manual_reasoning_content_from_llm_output, manual_reasoning_effort_from_llm_input,
        manual_replay_llm_payload, manual_replay_llm_response, manual_tool_call_arguments,
        manual_tool_call_id, manual_tool_call_name, manual_usage_fields_from_llm_output,
        manual_usage_fields_from_preferred_token_usage,
    };
    use serde_json::json;

    #[test]
    fn manual_usage_fields_support_aliases_and_nested_cache_details() {
        let output = json!({
            "usage": {
                "inputTokens": 11,
                "outputTokens": 7,
                "totalTokens": 18,
                "prompt_tokens_details": {"cached_tokens": 5},
                "cacheWriteTokens": 2
            }
        });

        let fields = manual_usage_fields_from_llm_output(Some(&output)).unwrap();
        assert_eq!(fields.prompt_tokens, Some(11));
        assert_eq!(fields.completion_tokens, Some(7));
        assert_eq!(fields.reported_total_tokens, Some(18));
        assert_eq!(fields.cache_read_tokens, Some(5));
        assert_eq!(fields.cache_write_tokens, Some(2));
    }

    #[test]
    fn manual_usage_fields_can_preserve_token_usage_precedence() {
        let output = json!({
            "usage": {"prompt_tokens": 1},
            "token_usage": {"inputTokens": 2}
        });

        let fields = manual_usage_fields_from_preferred_token_usage(Some(&output)).unwrap();
        assert_eq!(fields.prompt_tokens, Some(2));
    }

    #[test]
    fn manual_usage_fields_preserve_key_order_for_manual_output() {
        let output = json!({
            "usage": {"input": 1},
            "token_usage": {"prompt_tokens": 2}
        });

        let fields = manual_usage_fields_from_llm_output(Some(&output)).unwrap();
        assert_eq!(fields.prompt_tokens, Some(2));
    }

    #[test]
    fn manual_cost_estimate_preserves_currency_and_component_costs() {
        let output = json!({
            "model": "demo-model",
            "usage": {
                "cost": {
                    "currency": "EUR",
                    "input": 0.25,
                    "output": 0.5,
                    "cache_read": 0.125
                }
            }
        });

        let cost = manual_cost_estimate_from_llm_output(Some(&output)).unwrap();
        assert_eq!(
            manual_model_name_from_llm_output(Some(&output)),
            Some("demo-model")
        );
        assert_eq!(cost.currency, "EUR");
        assert_eq!(cost.total, None);
        assert_eq!(cost.input, Some(0.25));
        assert_eq!(cost.output, Some(0.5));
        assert_eq!(cost.cache_read, Some(0.125));
        assert_eq!(cost.total_or_component_sum(), Some(0.875));
    }

    #[test]
    fn manual_cost_estimate_prefers_explicit_cost_usd() {
        let output = json!({
            "usage": {
                "cost_usd": 0.005,
                "cost": {
                    "currency": "EUR",
                    "total": 9.0
                }
            }
        });

        let cost = manual_cost_estimate_from_llm_output(Some(&output)).unwrap();
        assert_eq!(cost.currency, "USD");
        assert_eq!(cost.total, Some(0.005));
        assert_eq!(cost.total_for_currency("USD"), Some(0.005));
    }

    #[test]
    fn manual_cost_estimate_falls_through_amountless_primary_cost() {
        let output = json!({
            "usage": {
                "cost": {
                    "currency": "USD"
                }
            },
            "token_usage": {
                "cost_usd": 0.005
            }
        });

        let cost = manual_cost_estimate_from_llm_output(Some(&output)).unwrap();
        assert_eq!(cost.currency, "USD");
        assert_eq!(cost.total, Some(0.005));
    }

    #[test]
    fn manual_cost_estimate_from_usage_returns_none_without_amounts() {
        let usage = json!({
            "cost": {
                "currency": "USD"
            }
        });

        let usage = usage.as_object().unwrap();
        assert!(manual_cost_estimate_from_usage(usage).is_none());
    }

    #[test]
    fn manual_cost_total_for_currency_falls_through_non_matching_currency() {
        let output = json!({
            "usage": {
                "cost": {
                    "currency": "EUR",
                    "total": 9.0
                }
            },
            "token_usage": {
                "cost_usd": 0.005
            }
        });

        assert_eq!(
            manual_cost_total_for_currency_from_llm_output(Some(&output), "USD"),
            Some(0.005)
        );
    }

    #[test]
    fn manual_reasoning_and_tool_call_readers_cover_provider_aliases() {
        let input = json!({"reasoning_effort": "high"});
        let output = json!({"reasoning": "chain summary"});
        let tool_call = json!({
            "call_id": "call_1",
            "tool_name": "read_file",
            "args": {"path": "README.md"}
        });

        assert_eq!(
            manual_reasoning_effort_from_llm_input(&input),
            Some(json!("high"))
        );
        assert_eq!(
            manual_reasoning_content_from_llm_output(&output),
            Some("chain summary".to_string())
        );
        assert_eq!(manual_tool_call_id(&tool_call), Some("call_1"));
        assert_eq!(manual_tool_call_name(&tool_call), Some("read_file"));
        assert_eq!(
            manual_tool_call_arguments(&tool_call),
            Some(&json!({"path": "README.md"}))
        );
    }

    #[test]
    fn manual_replay_payload_readers_detect_openclaw_shapes() {
        let input = json!({
            "content": {
                "source": "openclaw.llm_input",
                "messages": [{"role": "user", "content": "hi"}]
            }
        });
        let output = json!({
            "openclaw": {"model": "demo"},
            "content": "hello"
        });

        assert_eq!(
            manual_replay_llm_payload(&input).and_then(|payload| payload.get("source")),
            Some(&json!("openclaw.llm_input"))
        );
        assert_eq!(
            manual_replay_llm_response(&output).and_then(|payload| payload.get("content")),
            Some(&json!("hello"))
        );
    }
}
