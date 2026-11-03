// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_relay_types::{
    Json,
    codec::{request::MessageContent, traits::LlmResponseCodec},
    error::FlowError,
};
use serde_json::json;

fn provider_cost_codecs() -> Vec<(Box<dyn LlmResponseCodec>, Json)> {
    use super::super::{
        anthropic::AnthropicMessagesCodec, openai_chat::OpenAIChatCodec,
        openai_responses::OpenAIResponsesCodec,
    };

    vec![
        (
            Box::new(OpenAIChatCodec),
            json!({
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
            }),
        ),
        (
            Box::new(OpenAIResponsesCodec),
            json!({
                "status": "completed",
                "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}]}],
                "usage": {"input_tokens": 11, "output_tokens": 5, "total_tokens": 16}
            }),
        ),
        (
            Box::new(AnthropicMessagesCodec),
            json!({
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Hello"}],
                "usage": {"input_tokens": 11, "output_tokens": 5}
            }),
        ),
    ]
}

#[test]
fn test_provider_cost_wire_shapes_preserve_annotations() {
    for (codec, response) in provider_cost_codecs() {
        for (wire_cost, expected_total, expected_currency) in [
            (json!(0), 0.0, "USD"),
            (json!(0.000058), 0.000058, "USD"),
            (
                json!({"total": 0.12, "input": 0.10, "output": 0.02, "currency": "EUR"}),
                0.12,
                "EUR",
            ),
        ] {
            let mut response = response.clone();
            response["usage"]["cost"] = wire_cost;
            let decoded = codec.decode_response(&response).unwrap();
            assert_eq!(decoded.message, Some(MessageContent::Text("Hello".into())));
            assert_eq!(decoded.finish_reason, Some(FinishReason::Complete));
            let usage = decoded.usage.unwrap();
            assert_eq!(usage.prompt_tokens, Some(11));
            assert_eq!(usage.completion_tokens, Some(5));
            assert_eq!(usage.total_tokens, Some(16));
            let cost = usage.cost.unwrap();
            assert_eq!(cost.total, Some(expected_total));
            assert_eq!(cost.currency, expected_currency);
            assert_eq!(cost.source, CostSource::ProviderReported);
            if expected_currency == "EUR" {
                assert_eq!(cost.input, Some(0.10));
                assert_eq!(cost.output, Some(0.02));
            }

            response["usage"]["cost_usd"] = json!(0.42);
            let cost = codec
                .decode_response(&response)
                .unwrap()
                .usage
                .unwrap()
                .cost
                .unwrap();
            assert_eq!(cost.total, Some(0.42));
            assert_eq!(cost.currency, "USD");
        }
    }
}

#[test]
fn test_unrecognized_provider_cost_preserves_annotations() {
    for (codec, response) in provider_cost_codecs() {
        for wire_cost in [
            json!(null),
            json!("unknown"),
            json!(true),
            json!([]),
            json!({"total": "unknown"}),
            json!({"future_cost": 1}),
        ] {
            let mut response = response.clone();
            response["usage"]["cost"] = wire_cost;
            let decoded = codec.decode_response(&response).unwrap();
            assert_eq!(decoded.message, Some(MessageContent::Text("Hello".into())));
            assert_eq!(decoded.finish_reason, Some(FinishReason::Complete));
            let usage = decoded.usage.unwrap();
            assert_eq!(usage.prompt_tokens, Some(11));
            assert_eq!(usage.completion_tokens, Some(5));
            assert_eq!(usage.total_tokens, Some(16));
            assert!(usage.cost.is_none());

            response["usage"]["cost_usd"] = json!(0.42);
            let cost = codec
                .decode_response(&response)
                .unwrap()
                .usage
                .unwrap()
                .cost
                .unwrap();
            assert_eq!(cost.total, Some(0.42));
            assert_eq!(cost.currency, "USD");
            assert_eq!(cost.source, CostSource::ProviderReported);
        }
    }
}

#[test]
fn test_provider_reported_cost_sums_components_and_defaults_currency() {
    let cost = provider_reported_cost(
        None,
        Some(RawProviderUsageCost::Detailed(RawUsageCost {
            input: Some(0.12),
            output: Some(0.30),
            cache_read: Some(0.01),
            cache_write: Some(0.02),
            ..RawUsageCost::default()
        })),
    )
    .expect("component-only provider cost should be retained");

    assert_eq!(cost.total, Some(0.45));
    assert_eq!(cost.currency, "USD");
    assert_eq!(cost.source, CostSource::ProviderReported);
}

#[test]
fn test_provider_reported_cost_keeps_top_level_provider_usd_currency() {
    let cost = provider_reported_cost(
        Some(0.42),
        Some(RawProviderUsageCost::Detailed(RawUsageCost {
            currency: Some("EUR".to_string()),
            input: Some(0.10),
            output: Some(0.20),
            cache_read: Some(0.01),
            cache_write: Some(0.02),
            ..RawUsageCost::default()
        })),
    )
    .expect("top-level provider USD cost should be retained");

    assert_eq!(cost.total, Some(0.42));
    assert_eq!(cost.currency, "USD");
    assert_eq!(cost.input, None);
    assert_eq!(cost.output, None);
    assert_eq!(cost.cache_read, None);
    assert_eq!(cost.cache_write, None);
    assert_eq!(cost.total_for_currency("USD"), Some(0.42));
}

#[test]
fn test_provider_reported_cost_keeps_usd_components_with_top_level_usd_cost() {
    let cost = provider_reported_cost(
        Some(0.42),
        Some(RawProviderUsageCost::Detailed(RawUsageCost {
            currency: Some("usd".to_string()),
            input: Some(0.10),
            output: Some(0.20),
            cache_read: Some(0.01),
            cache_write: Some(0.02),
            ..RawUsageCost::default()
        })),
    )
    .expect("top-level and nested USD cost fields should be retained");

    assert_eq!(cost.total, Some(0.42));
    assert_eq!(cost.currency, "USD");
    assert_eq!(cost.input, Some(0.10));
    assert_eq!(cost.output, Some(0.20));
    assert_eq!(cost.cache_read, Some(0.01));
    assert_eq!(cost.cache_write, Some(0.02));
}

// -------------------------------------------------------------------
// FinishReason serialization
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// ResponseToolCall serialization
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// ApiSpecificResponse serialization
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// Helper: response_text()
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// Helper: has_tool_calls()
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// Helper: FinishReason::is_complete()
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// LlmResponseCodec trait: mock implementation
// -------------------------------------------------------------------

struct MockResponseCodec;

impl LlmResponseCodec for MockResponseCodec {
    fn decode_response(
        &self,
        _response: &Json,
    ) -> nemo_relay_types::error::Result<AnnotatedLlmResponse> {
        Ok(AnnotatedLlmResponse {
            id: Some("mock-id".into()),
            model: Some("mock-model".into()),
            message: Some(MessageContent::Text("mock response".into())),
            tool_calls: None,
            finish_reason: Some(FinishReason::Complete),
            usage: None,
            optimization_summary: None,
            api_specific: None,
            extra: serde_json::Map::new(),
        })
    }
}

struct FailingMockResponseCodec;

impl LlmResponseCodec for FailingMockResponseCodec {
    fn decode_response(
        &self,
        _response: &Json,
    ) -> nemo_relay_types::error::Result<AnnotatedLlmResponse> {
        Err(FlowError::Internal("decode failed".into()))
    }
}

// -------------------------------------------------------------------
// Extra field (flatten) captures unmodeled keys
// -------------------------------------------------------------------

#[test]
fn test_mock_response_codec_compiles_and_returns_ok() {
    let codec = MockResponseCodec;
    let result = codec.decode_response(&json!({}));
    assert!(result.is_ok());
    let resp = result.unwrap();
    assert_eq!(resp.id, Some("mock-id".into()));
    assert_eq!(resp.model, Some("mock-model".into()));
}

#[test]
fn test_failing_mock_codec_demonstrates_non_fatal_pattern() {
    let codec = FailingMockResponseCodec;
    let result = codec.decode_response(&json!({"choices": []}));
    assert!(result.is_err());

    // Non-fatal pattern: callers use .ok() to convert Err to None
    let annotated: Option<AnnotatedLlmResponse> =
        codec.decode_response(&json!({"choices": []})).ok();
    assert!(annotated.is_none());
}
