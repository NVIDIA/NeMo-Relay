// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::codec::request::{ContentPart, MessageContent};
use serde_json::json;

#[test]
fn test_annotated_llm_response_full_round_trip() {
    let resp = full_response();
    let json_val = serde_json::to_value(&resp).unwrap();
    let deserialized: AnnotatedLlmResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(resp, deserialized);
}

#[test]
fn test_annotated_llm_response_minimal_round_trip() {
    let resp = minimal_response();
    let json_val = serde_json::to_value(&resp).unwrap();
    // Minimal response should serialize to just `{}`
    assert_eq!(json_val, json!({}));
    let deserialized: AnnotatedLlmResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(resp, deserialized);
}

#[test]
fn test_usage_all_none_deserializes_from_empty() {
    let usage: Usage = serde_json::from_value(json!({})).unwrap();
    assert_eq!(usage, Usage::default());
    assert!(usage.prompt_tokens.is_none());
    assert!(usage.completion_tokens.is_none());
    assert!(usage.total_tokens.is_none());
    assert!(usage.cache_read_tokens.is_none());
    assert!(usage.cache_write_tokens.is_none());
}

#[test]
fn test_usage_all_populated_round_trip() {
    let usage = Usage {
        prompt_tokens: Some(100),
        completion_tokens: Some(50),
        total_tokens: Some(150),
        cache_read_tokens: Some(20),
        cache_write_tokens: Some(10),
        cache_write_1h_tokens: None,
        uncached_input_tokens: None,
        cost: None,
    };
    let json_val = serde_json::to_value(&usage).unwrap();
    let deserialized: Usage = serde_json::from_value(json_val).unwrap();
    assert_eq!(usage, deserialized);
}

#[test]
fn test_usage_ignores_unmodeled_provider_subfields() {
    let usage: Usage = serde_json::from_value(json!({
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "some_future_field": 99
    }))
    .unwrap();
    assert_eq!(usage.prompt_tokens, Some(5));
    assert_eq!(usage.completion_tokens, Some(7));
    let reserialized = serde_json::to_value(&usage).unwrap();
    assert!(reserialized.get("some_future_field").is_none());
}

#[test]
fn test_finish_reason_complete_serializes_to_complete() {
    let reason = FinishReason::Complete;
    let json_val = serde_json::to_value(&reason).unwrap();
    assert_eq!(json_val, json!("complete"));
    let deserialized: FinishReason = serde_json::from_value(json_val).unwrap();
    assert_eq!(deserialized, FinishReason::Complete);
}

#[test]
fn test_finish_reason_length_round_trip() {
    let reason = FinishReason::Length;
    let json_val = serde_json::to_value(&reason).unwrap();
    assert_eq!(json_val, json!("length"));
    let deserialized: FinishReason = serde_json::from_value(json_val).unwrap();
    assert_eq!(deserialized, FinishReason::Length);
}

#[test]
fn test_finish_reason_tool_use_round_trip() {
    let reason = FinishReason::ToolUse;
    let json_val = serde_json::to_value(&reason).unwrap();
    assert_eq!(json_val, json!("tool_use"));
    let deserialized: FinishReason = serde_json::from_value(json_val).unwrap();
    assert_eq!(deserialized, FinishReason::ToolUse);
}

#[test]
fn test_finish_reason_content_filter_round_trip() {
    let reason = FinishReason::ContentFilter;
    let json_val = serde_json::to_value(&reason).unwrap();
    assert_eq!(json_val, json!("content_filter"));
    let deserialized: FinishReason = serde_json::from_value(json_val).unwrap();
    assert_eq!(deserialized, FinishReason::ContentFilter);
}

#[test]
fn test_finish_reason_unknown_round_trip() {
    let reason = FinishReason::Unknown("custom_reason".into());
    let json_val = serde_json::to_value(&reason).unwrap();
    let deserialized: FinishReason = serde_json::from_value(json_val).unwrap();
    assert_eq!(deserialized, FinishReason::Unknown("custom_reason".into()));
}

#[test]
fn test_response_tool_call_json_arguments_round_trip() {
    let tc = ResponseToolCall {
        id: "call_abc".into(),
        name: "search".into(),
        arguments: json!({"query": "weather", "limit": 5}),
    };
    let json_val = serde_json::to_value(&tc).unwrap();
    assert_eq!(json_val["arguments"]["query"], json!("weather"));
    assert_eq!(json_val["arguments"]["limit"], json!(5));
    let deserialized: ResponseToolCall = serde_json::from_value(json_val).unwrap();
    assert_eq!(tc, deserialized);
}

#[test]
fn test_api_specific_openai_chat_round_trip() {
    let api = ApiSpecificResponse::OpenAIChat {
        logprobs: Some(json!({"content": []})),
        system_fingerprint: Some("fp_abc".into()),
        service_tier: Some("default".into()),
    };
    let json_val = serde_json::to_value(&api).unwrap();
    assert_eq!(json_val["api"], json!("openai_chat"));
    let deserialized: ApiSpecificResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(api, deserialized);
}

#[test]
fn test_api_specific_openai_responses_round_trip() {
    let api = ApiSpecificResponse::OpenAIResponses {
        output_items: Some(vec![json!({"type": "message", "content": []})]),
        status: Some("completed".into()),
        incomplete_details: None,
        previous_response_id: None,
        store: None,
        service_tier: None,
        truncation: None,
        reasoning: None,
        input_tokens_details: None,
        output_tokens_details: None,
    };
    let json_val = serde_json::to_value(&api).unwrap();
    assert_eq!(json_val["api"], json!("openai_responses"));
    let deserialized: ApiSpecificResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(api, deserialized);
}

#[test]
fn test_api_specific_anthropic_messages_round_trip() {
    let api = ApiSpecificResponse::AnthropicMessages {
        object_type: Some("message".into()),
        role: Some("assistant".into()),
        stop_reason: Some("end_turn".into()),
        stop_sequence: Some("\n\nHuman:".into()),
        service_tier: Some("default".into()),
        container: Some(json!({"id":"container_123"})),
        content_blocks: Some(vec![json!({"type": "text", "text": "Hello"})]),
    };
    let json_val = serde_json::to_value(&api).unwrap();
    assert_eq!(json_val["api"], json!("anthropic_messages"));
    let deserialized: ApiSpecificResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(api, deserialized);
}

#[test]
fn test_api_specific_gemini_generate_content_round_trip_empty_extra() {
    let api = ApiSpecificResponse::GeminiGenerateContent {
        thoughts_tokens: Some(3),
        tool_use_prompt_tokens: None,
        safety_ratings: None,
        grounding_metadata: None,
        citation_metadata: None,
        extra: serde_json::Map::new(),
    };
    let json_val = serde_json::to_value(&api).unwrap();
    assert_eq!(json_val["api"], json!("gemini_generate_content"));
    let deserialized: ApiSpecificResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(api, deserialized);
}

#[test]
fn test_api_specific_gemini_generate_content_round_trip_extra() {
    let api = ApiSpecificResponse::GeminiGenerateContent {
        thoughts_tokens: None,
        tool_use_prompt_tokens: None,
        safety_ratings: Some(json!([{"category": "HARM_CATEGORY_HATE_SPEECH"}])),
        grounding_metadata: None,
        citation_metadata: None,
        extra: serde_json::Map::from_iter([("urlContextMetadata".into(), json!({"url": "u"}))]),
    };
    let json_val = serde_json::to_value(&api).unwrap();
    assert_eq!(json_val["api"], json!("gemini_generate_content"));
    assert_eq!(json_val["urlContextMetadata"], json!({"url": "u"}));
    let deserialized: ApiSpecificResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(api, deserialized);
}

#[test]
fn test_api_specific_gemini_generate_content_extra_cannot_override_api_tag() {
    let api = ApiSpecificResponse::GeminiGenerateContent {
        thoughts_tokens: None,
        tool_use_prompt_tokens: None,
        safety_ratings: None,
        grounding_metadata: None,
        citation_metadata: None,
        extra: serde_json::Map::from_iter([
            ("api".into(), json!("not_the_tag")),
            ("futureField".into(), json!(true)),
        ]),
    };
    let json_val = serde_json::to_value(&api).unwrap();
    assert_eq!(
        json_val["api"],
        json!("gemini_generate_content"),
        "Gemini extra.api must not overwrite the enum discriminator"
    );
    assert_eq!(json_val["futureField"], json!(true));
    let deserialized: ApiSpecificResponse = serde_json::from_value(json_val).unwrap();
    let ApiSpecificResponse::GeminiGenerateContent { extra, .. } = deserialized else {
        panic!("expected Gemini generateContent metadata");
    };
    assert!(extra.get("api").is_none());
    assert_eq!(extra.get("futureField"), Some(&json!(true)));
}

#[test]
fn test_api_specific_custom_round_trip() {
    let api = ApiSpecificResponse::Custom {
        api_name: "my_custom_llm".into(),
        data: json!({"version": "2.0", "extra_field": true}),
    };
    let json_val = serde_json::to_value(&api).unwrap();
    assert_eq!(json_val["api"], json!("custom"));
    assert_eq!(json_val["api_name"], json!("my_custom_llm"));
    let deserialized: ApiSpecificResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(api, deserialized);
}

#[test]
fn test_response_text_returns_some_for_text_content() {
    let resp = AnnotatedLlmResponse {
        message: Some(MessageContent::Text("Hello!".into())),
        ..minimal_response()
    };
    assert_eq!(resp.response_text(), Some("Hello!"));
}

#[test]
fn test_response_text_returns_none_when_message_is_none() {
    let resp = minimal_response();
    assert_eq!(resp.response_text(), None);
}

#[test]
fn test_response_text_extracts_first_text_from_parts() {
    let resp = AnnotatedLlmResponse {
        message: Some(MessageContent::Parts(vec![ContentPart::Text {
            text: "Part text".into(),
            extra: serde_json::Map::new(),
        }])),
        ..minimal_response()
    };
    assert_eq!(resp.response_text(), Some("Part text"));
}

#[test]
fn test_has_tool_calls_true_when_present() {
    let resp = AnnotatedLlmResponse {
        tool_calls: Some(vec![ResponseToolCall {
            id: "tc_1".into(),
            name: "search".into(),
            arguments: json!({}),
        }]),
        ..minimal_response()
    };
    assert!(resp.has_tool_calls());
}

#[test]
fn test_has_tool_calls_false_when_none() {
    let resp = minimal_response();
    assert!(!resp.has_tool_calls());
}

#[test]
fn test_has_tool_calls_false_when_empty_vec() {
    let resp = AnnotatedLlmResponse {
        tool_calls: Some(vec![]),
        ..minimal_response()
    };
    assert!(!resp.has_tool_calls());
}

#[test]
fn test_is_complete_true_for_complete() {
    assert!(FinishReason::Complete.is_complete());
}

#[test]
fn test_is_complete_false_for_other_variants() {
    assert!(!FinishReason::Length.is_complete());
    assert!(!FinishReason::ToolUse.is_complete());
    assert!(!FinishReason::ContentFilter.is_complete());
    assert!(!FinishReason::Unknown("other".into()).is_complete());
}

#[test]
fn test_annotated_llm_response_extra_captures_unmodeled_keys() {
    let json_val = json!({
        "id": "test-123",
        "model": "gpt-4",
        "custom_field": "custom_value",
        "another_field": 42
    });
    let resp: AnnotatedLlmResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(resp.id, Some("test-123".into()));
    assert_eq!(resp.model, Some("gpt-4".into()));
    assert_eq!(resp.extra.get("custom_field"), Some(&json!("custom_value")));
    assert_eq!(resp.extra.get("another_field"), Some(&json!(42)));

    // Round-trip: extra fields should appear as top-level keys
    let serialized = serde_json::to_value(&resp).unwrap();
    assert_eq!(serialized["custom_field"], json!("custom_value"));
    assert_eq!(serialized["another_field"], json!(42));
}

fn full_response() -> AnnotatedLlmResponse {
    AnnotatedLlmResponse {
        id: Some("chatcmpl-abc123".into()),
        model: Some("gpt-4".into()),
        message: Some(MessageContent::Text("Hello, world!".into())),
        tool_calls: Some(vec![ResponseToolCall {
            id: "call_1".into(),
            name: "get_weather".into(),
            arguments: json!({"city": "NYC"}),
        }]),
        finish_reason: Some(FinishReason::Complete),
        usage: Some(Usage {
            prompt_tokens: Some(10),
            completion_tokens: Some(20),
            total_tokens: Some(30),
            cache_read_tokens: Some(5),
            cache_write_tokens: Some(3),
            cache_write_1h_tokens: None,
            uncached_input_tokens: None,
            cost: None,
        }),
        optimization_summary: None,
        api_specific: Some(ApiSpecificResponse::OpenAIChat {
            logprobs: None,
            system_fingerprint: Some("fp_abc123".into()),
            service_tier: Some("default".into()),
        }),
        extra: serde_json::Map::new(),
    }
}

fn minimal_response() -> AnnotatedLlmResponse {
    AnnotatedLlmResponse {
        id: None,
        model: None,
        message: None,
        tool_calls: None,
        finish_reason: None,
        usage: None,
        optimization_summary: None,
        api_specific: None,
        extra: serde_json::Map::new(),
    }
}

#[test]
fn test_usage_cost_round_trip_preserves_provider_reported_codec_compatibility() {
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        total_tokens: Some(1_500),
        cache_read_tokens: Some(200),
        cache_write_tokens: None,
        cache_write_1h_tokens: None,
        uncached_input_tokens: None,
        cost: Some(CostEstimate {
            total: Some(0.42),
            currency: "USD".into(),
            input: Some(0.12),
            output: Some(0.30),
            cache_read: None,
            cache_write: None,
            source: CostSource::ProviderReported,
            pricing_provider: None,
            pricing_model: None,
            pricing_as_of: None,
            pricing_source: None,
        }),
    };

    let json_val = serde_json::to_value(&usage).unwrap();

    assert_eq!(json_val["cost"]["total"], json!(0.42));
    assert_eq!(json_val["cost"]["source"], json!("provider_reported"));
    let deserialized: Usage = serde_json::from_value(json_val).unwrap();
    assert_eq!(usage, deserialized);
}
