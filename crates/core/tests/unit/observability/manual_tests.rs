// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the shared manual/non-provider fallback helpers, focused on
//! the points where the OpenTelemetry and OpenInference copies diverged.

use super::*;
use serde_json::json;

#[test]
fn cost_otel_policy_emits_any_currency() {
    let output = json!({"usage": {"cost": {"total": 0.5, "currency": "EUR"}}});
    assert_eq!(
        cost_from_manual_llm_output(Some(&output), false),
        Some((0.5, "EUR".to_string()))
    );
}

#[test]
fn cost_openinference_policy_drops_non_usd() {
    let output = json!({"usage": {"cost": {"total": 0.5, "currency": "EUR"}}});
    assert_eq!(cost_from_manual_llm_output(Some(&output), true), None);
}

#[test]
fn cost_component_sum_emits_currency_for_otel() {
    let output = json!({"usage": {"cost": {"input": 0.5, "output": 0.375, "currency": "EUR"}}});
    assert_eq!(
        cost_from_manual_llm_output(Some(&output), false),
        Some((0.875, "EUR".to_string()))
    );
}

#[test]
fn cost_usd_field_passes_usd_only() {
    let output = json!({"usage": {"cost_usd": 1.25}});
    assert_eq!(
        cost_from_manual_llm_output(Some(&output), true),
        Some((1.25, "USD".to_string()))
    );
}

#[test]
fn cost_absent_currency_treated_as_usd() {
    let output = json!({"usage": {"cost": {"total": 0.9}}});
    assert_eq!(
        cost_from_manual_llm_output(Some(&output), true),
        Some((0.9, "USD".to_string()))
    );
}

#[test]
fn cost_per_map_fallthrough_under_usd_only() {
    // A non-USD `usage` cost is skipped under usd_only; `token_usage` USD wins.
    let output = json!({
        "usage": {"cost": {"total": 0.5, "currency": "EUR"}},
        "token_usage": {"cost_usd": 0.2}
    });
    assert_eq!(
        cost_from_manual_llm_output(Some(&output), true),
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
fn model_name_extraction() {
    assert_eq!(
        model_name_from_manual_llm_output(Some(&json!({"model": "m"}))),
        Some("m")
    );
    assert_eq!(model_name_from_manual_llm_output(Some(&json!({}))), None);
    assert_eq!(model_name_from_manual_llm_output(None), None);
}
