// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::json;
use std::collections::HashMap;

#[test]
fn optimization_summary_emits_namespaced_openinference_attributes() {
    let summary: nemo_relay_types::codec::optimization::LlmOptimizationSummary =
        serde_json::from_value(json!({
            "schema_version":"1", "calculation_version":"1", "status":"complete",
            "baseline_model":{"model":"baseline"}, "effective_model":{"model":"effective"},
            "tokens_saved":{"prompt_tokens":12,"total_tokens":12},
            "baseline_cost":{"total":0.02,"currency":"USD","source":"model_pricing","pricing_as_of":"2026-07-08","pricing_source":"test"},
            "actual_cost":{"total":0.01,"currency":"USD","source":"model_pricing"},
            "estimated_cost_saved":0.01, "currency":"USD", "contributions":[]
        }))
        .unwrap();
    let mut attributes = Vec::new();
    push_optimization_attributes(&mut attributes, &summary);
    let attributes = attr_map(&attributes);
    assert_eq!(
        attributes["nemo_relay.llm.optimization.effective_model"],
        "effective"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.total_tokens_saved"],
        "12"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_cost"],
        "0.01"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.pricing_source"],
        "test"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_cost_currency"],
        "USD"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_cost_currency"],
        "USD"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.estimated_cost_saved_currency"],
        "USD"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_pricing_as_of"],
        "2026-07-08"
    );
}

#[test]
fn optimization_cost_attributes_preserve_independent_currency_and_provenance() {
    let summary: nemo_relay_types::codec::optimization::LlmOptimizationSummary =
        serde_json::from_value(json!({
            "schema_version":"1", "calculation_version":"1", "status":"partial",
            "limitations":["cost_currency_mismatch"],
            "tokens_saved":{},
            "baseline_cost":{"total":2.0,"currency":"EUR","source":"model_pricing","pricing_as_of":"2026-01-01"},
            "actual_cost":{"total":1.0,"currency":"GBP","source":"provider_reported","pricing_as_of":"2026-02-02","pricing_source":"provider"},
            "contributions":[]
        }))
        .unwrap();
    let mut attributes = Vec::new();
    push_optimization_attributes(&mut attributes, &summary);
    let attributes = attr_map(&attributes);

    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_cost_currency"],
        "EUR"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_cost_currency"],
        "GBP"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_pricing_source"],
        "provider"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_pricing_as_of"],
        "2026-02-02"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.pricing_source"],
        "provider"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.pricing_as_of"],
        "2026-01-01"
    );
    assert!(!attributes.contains_key("nemo_relay.llm.optimization.estimated_cost_saved"));
    assert!(!attributes.contains_key("nemo_relay.llm.optimization.estimated_cost_saved_currency"));
}

#[test]
fn complete_non_usd_optimization_costs_keep_currency_and_independent_provenance() {
    let summary: nemo_relay_types::codec::optimization::LlmOptimizationSummary =
        serde_json::from_value(json!({
            "schema_version":"1", "calculation_version":"1", "status":"complete",
            "tokens_saved":{},
            "baseline_cost":{"total":2.0,"currency":"EUR","source":"model_pricing","pricing_as_of":"2026-01-01","pricing_source":"baseline-catalog"},
            "actual_cost":{"total":1.0,"currency":"EUR","source":"provider_reported","pricing_as_of":"2026-02-02","pricing_source":"provider-meter"},
            "estimated_cost_saved":1.0, "currency":"EUR", "contributions":[]
        }))
        .unwrap();
    let mut attributes = Vec::new();
    push_optimization_attributes(&mut attributes, &summary);
    let attributes = attr_map(&attributes);

    assert_attr(
        &attributes,
        "nemo_relay.llm.optimization.baseline_cost_currency",
        "EUR",
    );
    assert_attr(
        &attributes,
        "nemo_relay.llm.optimization.actual_cost_currency",
        "EUR",
    );
    assert_attr(
        &attributes,
        "nemo_relay.llm.optimization.estimated_cost_saved_currency",
        "EUR",
    );
    assert_attr(
        &attributes,
        "nemo_relay.llm.optimization.baseline_pricing_source",
        "baseline-catalog",
    );
    assert_attr(
        &attributes,
        "nemo_relay.llm.optimization.actual_pricing_source",
        "provider-meter",
    );
    assert_attr(
        &attributes,
        "nemo_relay.llm.optimization.baseline_pricing_as_of",
        "2026-01-01",
    );
    assert_attr(
        &attributes,
        "nemo_relay.llm.optimization.actual_pricing_as_of",
        "2026-02-02",
    );
}

#[test]
fn display_text_from_tool_calls_preserves_legacy_name_precedence() {
    assert_eq!(
        display_text_from_tool_calls(&json!([
            {
                "name": "top_level",
                "toolName": "camel",
                "function": {"name": "nested"}
            },
            {
                "toolName": "camel_without_top_level",
                "function": {"name": "nested_without_top_level"}
            }
        ])),
        Some("Requested tools: top_level, camel_without_top_level".to_string())
    );
}

#[test]
fn pre_epoch_timestamps_round_trip_through_system_time() {
    let timestamp = DateTime::parse_from_rfc3339("1969-12-31T23:59:58.500000000Z")
        .unwrap()
        .with_timezone(&Utc);

    assert_eq!(
        to_system_time(timestamp),
        UNIX_EPOCH - Duration::new(1, 500_000_000)
    );
}

#[test]
fn annotated_input_projection_covers_extended_roles_and_native_text() {
    let messages = vec![
        Message::Developer {
            content: MessageContent::Text("developer".into()),
            name: None,
        },
        Message::Function {
            content: Some("legacy".into()),
            name: "legacy".into(),
        },
        Message::ToolCallItem {
            id: None,
            call_id: "call_1".into(),
            name: "lookup".into(),
            arguments: json!({}),
            extra: serde_json::Map::new(),
        },
        Message::ToolResultItem {
            id: None,
            call_id: "call_1".into(),
            output: json!("ok"),
            extra: serde_json::Map::new(),
        },
        Message::ProviderNative {
            provider: "openai_responses".into(),
            kind: "message".into(),
            value: json!({"role": "critic", "content": "native"}),
        },
        Message::ProviderNative {
            provider: "openai_responses".into(),
            kind: "reasoning".into(),
            value: json!({"type": "reasoning"}),
        },
    ];
    let mut attributes = Vec::new();
    push_annotated_input_messages(&mut attributes, &messages, 0);
    let attributes = attr_map(&attributes);
    assert_attr(
        &attributes,
        "llm.input_messages.0.message.role",
        "developer",
    );
    assert_attr(&attributes, "llm.input_messages.1.message.role", "function");
    assert_attr(
        &attributes,
        "llm.input_messages.1.message.content",
        "legacy",
    );
    assert_attr(
        &attributes,
        "llm.input_messages.2.message.role",
        "assistant",
    );
    assert_attr(&attributes, "llm.input_messages.3.message.role", "tool");
    assert_attr(&attributes, "llm.input_messages.4.message.role", "critic");
    assert_attr(
        &attributes,
        "llm.input_messages.4.message.content",
        "native",
    );
    assert_attr(
        &attributes,
        "llm.input_messages.5.message.role",
        "provider_native",
    );

    let content = MessageContent::Parts(vec![
        ContentPart::Refusal {
            refusal: "portable refusal".into(),
            extra: serde_json::Map::new(),
        },
        ContentPart::ProviderNative {
            provider: "openai_responses".into(),
            kind: "output_text".into(),
            value: json!({"text": "native text"}),
        },
        ContentPart::ProviderNative {
            provider: "openai_responses".into(),
            kind: "refusal".into(),
            value: json!({"refusal": "native refusal"}),
        },
    ]);
    assert_eq!(
        message_content_text(&content).as_deref(),
        Some("portable refusal\nnative text\nnative refusal")
    );
}

#[test]
fn annotated_input_projection_normalizes_tool_result_messages() {
    let messages = vec![
        Message::User {
            content: MessageContent::Parts(vec![
                ContentPart::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: json!("first result"),
                    is_error: None,
                    extra: serde_json::Map::new(),
                },
                ContentPart::ToolResult {
                    tool_use_id: "call_2".into(),
                    content: json!({"status": "ok"}),
                    is_error: Some(false),
                    extra: serde_json::Map::new(),
                },
            ]),
            name: None,
        },
        Message::User {
            content: MessageContent::Parts(vec![
                ContentPart::ToolResult {
                    tool_use_id: "call_3".into(),
                    content: json!("mixed result"),
                    is_error: None,
                    extra: serde_json::Map::new(),
                },
                ContentPart::Text {
                    text: "follow-up".into(),
                    extra: serde_json::Map::new(),
                },
            ]),
            name: None,
        },
        Message::Tool {
            content: MessageContent::Text("openai result".into()),
            tool_call_id: "call_4".into(),
        },
        Message::ToolResultItem {
            id: None,
            call_id: "call_5".into(),
            output: json!(["first", 2]),
            extra: serde_json::Map::new(),
        },
    ];

    let mut attributes = Vec::new();
    push_annotated_input_messages(&mut attributes, &messages, 0);
    let attributes = attr_map(&attributes);

    for (index, call_id, content) in [
        (0, "call_1", "first result"),
        (1, "call_2", r#"{"status":"ok"}"#),
        (3, "call_4", "openai result"),
        (4, "call_5", r#"["first",2]"#),
    ] {
        assert_attr(
            &attributes,
            &format!("llm.input_messages.{index}.message.role"),
            "tool",
        );
        assert_attr(
            &attributes,
            &format!("llm.input_messages.{index}.message.tool_call_id"),
            call_id,
        );
        assert_attr(
            &attributes,
            &format!("llm.input_messages.{index}.message.content"),
            content,
        );
    }
    assert_attr(&attributes, "llm.input_messages.2.message.role", "user");
    assert_attr(
        &attributes,
        "llm.input_messages.2.message.content",
        "follow-up",
    );
    assert!(!attributes.contains_key("llm.input_messages.2.message.tool_call_id"));
    assert!(!attributes.contains_key("llm.input_messages.5.message.role"));
}

fn assert_attr(attributes: &HashMap<String, String>, key: &str, value: &str) {
    assert_eq!(attributes.get(key).map(String::as_str), Some(value));
}

fn attr_map(attributes: &[KeyValue]) -> HashMap<String, String> {
    attributes
        .iter()
        .map(|attribute| {
            (
                attribute.key.as_str().to_string(),
                attribute.value.to_string(),
            )
        })
        .collect()
}
