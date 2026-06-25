// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the shared manual/non-provider fallback helpers, focused on
//! exporter-specific policy boundaries and shared scalar extraction.

use super::*;
use serde_json::json;

#[test]
#[cfg(feature = "otel")]
fn cost_otel_policy_emits_any_currency() {
    let output = json!({"usage": {"cost": {"total": 0.5, "currency": "EUR"}}});
    assert_eq!(
        cost_from_manual_llm_output(Some(&output), ManualCostPolicy::AnyCurrency),
        Some((0.5, "EUR".to_string()))
    );
}

#[test]
#[cfg(feature = "openinference")]
fn cost_openinference_policy_drops_non_usd() {
    let output = json!({"usage": {"cost": {"total": 0.5, "currency": "EUR"}}});
    assert_eq!(
        cost_from_manual_llm_output(Some(&output), ManualCostPolicy::UsdOnly),
        None
    );
}

#[test]
#[cfg(feature = "otel")]
fn cost_component_sum_emits_currency_for_otel() {
    let output = json!({"usage": {"cost": {"input": 0.5, "output": 0.375, "currency": "EUR"}}});
    assert_eq!(
        cost_from_manual_llm_output(Some(&output), ManualCostPolicy::AnyCurrency),
        Some((0.875, "EUR".to_string()))
    );
}

#[test]
#[cfg(feature = "openinference")]
fn cost_usd_field_passes_usd_only() {
    let output = json!({"usage": {"cost_usd": 1.25}});
    assert_eq!(
        cost_from_manual_llm_output(Some(&output), ManualCostPolicy::UsdOnly),
        Some((1.25, "USD".to_string()))
    );
}

#[test]
#[cfg(feature = "openinference")]
fn cost_absent_currency_treated_as_usd() {
    let output = json!({"usage": {"cost": {"total": 0.9}}});
    assert_eq!(
        cost_from_manual_llm_output(Some(&output), ManualCostPolicy::UsdOnly),
        Some((0.9, "USD".to_string()))
    );
}

#[test]
fn cost_atif_policy_rejects_component_only_cost_without_currency() {
    let output = json!({"usage": {"cost": {"input": 0.5, "output": 0.375}}});
    assert_eq!(
        cost_from_manual_llm_output(Some(&output), ManualCostPolicy::AtifUsdOnly),
        None
    );
}

#[test]
fn cost_atif_policy_accepts_relay_normalized_component_cost_without_currency() {
    let output = json!({
        "usage": {
            "cost": {
                "input": 0.5,
                "output": 0.375,
                "source": "provider_reported"
            }
        }
    });
    assert_eq!(
        cost_from_manual_llm_output(Some(&output), ManualCostPolicy::AtifUsdOnly),
        Some((0.875, "USD".to_string()))
    );
}

#[test]
#[cfg(feature = "openinference")]
fn cost_per_map_fallthrough_under_usd_only() {
    // A non-USD `usage` cost is skipped under usd_only; `token_usage` USD wins.
    let output = json!({
        "usage": {"cost": {"total": 0.5, "currency": "EUR"}},
        "token_usage": {"cost_usd": 0.2}
    });
    assert_eq!(
        cost_from_manual_llm_output(Some(&output), ManualCostPolicy::UsdOnly),
        Some((0.2, "USD".to_string()))
    );
}

#[test]
fn first_u64_is_map_major() {
    // `usage`'s `total` (5) wins over `token_usage`'s `total_tokens` (10):
    // all keys are tried against `usage` before `token_usage`.
    let usage = json!({"total": 5});
    let token_usage = json!({"total_tokens": 10});
    let got = first_u64_from_manual_usage(
        usage.as_object(),
        token_usage.as_object(),
        &["total_tokens", "totalTokens", "total"],
    );
    assert_eq!(got, Some(5));
}

#[test]
fn normalize_total_strict_drops_absent_and_inconsistent() {
    assert_eq!(normalize_total_tokens(None, Some(5), Some(5)), None);
    assert_eq!(normalize_total_tokens(Some(3), Some(5), Some(5)), None); // 3 < 10
    assert_eq!(normalize_total_tokens(Some(12), Some(5), Some(5)), Some(12));
    assert_eq!(normalize_total_tokens(Some(7), None, None), Some(7)); // minimum 0
}

#[test]
fn usage_extracts_aliases_and_returns_none_without_tokens() {
    let output = json!({"usage": {"inputTokens": 3, "outputTokens": 4}});
    let usage = usage_from_manual_llm_output(Some(&output)).expect("has tokens");
    assert_eq!(usage.prompt_tokens, Some(3));
    assert_eq!(usage.completion_tokens, Some(4));
    assert!(usage_from_manual_llm_output(Some(&json!({"usage": {"foo": 1}}))).is_none());
    assert!(usage_from_manual_llm_output(Some(&json!({}))).is_none());
}

#[test]
fn usage_cache_read_prefers_legacy_atif_order_for_conflicts() {
    let output = json!({
        "usage": {
            "cache_read_tokens": 99,
            "cache_read_input_tokens": 77,
            "prompt_tokens_details": {"cached_tokens": 42},
            "input_tokens_details": {"cached_tokens": 24},
            "cached_tokens": 12
        }
    });
    let usage = usage_from_manual_llm_output(Some(&output)).expect("has tokens");
    assert_eq!(usage.cache_read_tokens, Some(12));
}

#[test]
fn model_name_extraction() {
    assert_eq!(
        model_name_from_manual_llm_output(Some(&json!({"model": "m"}))),
        Some("m")
    );
    assert_eq!(model_name_from_manual_llm_output(Some(&json!({}))), None);
    assert_eq!(model_name_from_manual_llm_output(None), None);
}

#[test]
fn tool_call_fields_prefers_function_fields_before_aliases() {
    let tool_call = json!({
        "id": "call-1",
        "call_id": "call-2",
        "name": "top_level_name",
        "toolName": "camel_name",
        "function": {
            "name": "function_name",
            "arguments": {"query": "needle"}
        },
        "arguments": {"query": "fallback"}
    });

    let fields = tool_call_fields(&tool_call).unwrap();
    assert_eq!(fields.id, Some("call-1"));
    assert_eq!(fields.name, Some("function_name"));
    assert_eq!(fields.arguments, Some(&json!({"query": "needle"})));
}

#[test]
fn tool_call_fields_supports_flat_aliases() {
    let tool_call = json!({
        "call_id": "call-3",
        "function_name": "search_docs",
        "args": "{\"query\":\"docs\"}"
    });

    let fields = tool_call_fields(&tool_call).unwrap();
    assert_eq!(fields.id, Some("call-3"));
    assert_eq!(fields.name, Some("search_docs"));
    assert_eq!(fields.arguments, Some(&json!("{\"query\":\"docs\"}")));
    assert!(tool_call_fields(&json!("not an object")).is_none());
}
