// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for openai responses in the NeMo Flow core crate.

use super::*;
use serde_json::json;

use super::super::request::MessageContent;
use super::super::response::{ApiSpecificResponse, FinishReason};

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

fn make_request(content: Json) -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content,
    }
}

fn fixture_json(path: &str) -> Json {
    serde_json::from_str(path).expect("valid fixture json")
}

/// Full Responses API response with message, function_call, reasoning, and usage.
fn full_responses_response() -> Json {
    json!({
        "id": "resp_abc123",
        "object": "response",
        "created_at": 1746989954.0,
        "model": "gpt-4o-2024-08-06",
        "status": "completed",
        "output": [
            {
                "id": "rs_abc123",
                "type": "reasoning",
                "summary": [],
                "status": null,
                "encrypted_content": "gAAAAABo..."
            },
            {
                "id": "msg_abc123",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello!",
                        "annotations": []
                    }
                ]
            },
            {
                "type": "function_call",
                "id": "fc_abc123",
                "name": "get_weather",
                "call_id": "call_abc123",
                "arguments": "{\"city\":\"NYC\"}",
                "status": "completed"
            }
        ],
        "usage": {
            "input_tokens": 75,
            "output_tokens": 1186,
            "total_tokens": 1261,
            "input_tokens_details": { "cached_tokens": 10 },
            "output_tokens_details": { "reasoning_tokens": 1024 }
        }
    })
}

// ===================================================================
// Response decode tests
// ===================================================================

#[test]
fn test_decode_full_response() {
    let codec = OpenAIResponsesCodec;
    let resp = codec.decode_response(&full_responses_response()).unwrap();

    assert_eq!(resp.id, Some("resp_abc123".into()));
    assert_eq!(resp.model, Some("gpt-4o-2024-08-06".into()));

    // Text from output_text items
    assert_eq!(resp.message, Some(MessageContent::Text("Hello!".into())));

    // Tool calls from function_call items
    let tool_calls = resp.tool_calls.unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "call_abc123"); // call_id, NOT id
    assert_eq!(tool_calls[0].name, "get_weather");
    assert_eq!(tool_calls[0].arguments, json!({"city": "NYC"}));

    // Finish reason from status
    assert_eq!(resp.finish_reason, Some(FinishReason::Complete));

    // Usage mapping
    let usage = resp.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(75)); // input_tokens -> prompt_tokens
    assert_eq!(usage.completion_tokens, Some(1186)); // output_tokens -> completion_tokens
    assert_eq!(usage.total_tokens, Some(1261));
    assert_eq!(usage.cache_read_tokens, Some(10));
    assert_eq!(usage.cache_write_tokens, None);

    // API specific fields
    match resp.api_specific.unwrap() {
        ApiSpecificResponse::OpenAIResponses {
            output_items,
            status,
            incomplete_details,
            previous_response_id,
            store,
            service_tier,
            truncation,
            reasoning,
            input_tokens_details,
            output_tokens_details,
        } => {
            assert_eq!(status, Some("completed".into()));
            assert!(output_items.is_some());
            assert_eq!(output_items.unwrap().len(), 3);
            assert!(incomplete_details.is_none());
            assert_eq!(previous_response_id, None);
            assert_eq!(store, None);
            assert_eq!(service_tier, None);
            assert_eq!(truncation, None);
            assert_eq!(reasoning, None);
            assert_eq!(input_tokens_details, Some(json!({"cached_tokens": 10})));
            assert_eq!(
                output_tokens_details,
                Some(json!({"reasoning_tokens": 1024}))
            );
        }
        other => panic!("Expected OpenAIResponses, got {other:?}"),
    }
}

#[test]
fn test_decode_response_openai_responses_api_specific_top_level_fields() {
    let codec = OpenAIResponsesCodec;
    let response = json!({
        "id": "resp_abc123",
        "status": "completed",
        "output": [],
        "previous_response_id": "resp_prev_1",
        "store": true,
        "service_tier": "default",
        "truncation": "auto",
        "reasoning": {"effort": "high"}
    });
    let resp = codec.decode_response(&response).unwrap();
    match resp.api_specific.unwrap() {
        ApiSpecificResponse::OpenAIResponses {
            previous_response_id,
            store,
            service_tier,
            truncation,
            reasoning,
            ..
        } => {
            assert_eq!(previous_response_id.as_deref(), Some("resp_prev_1"));
            assert_eq!(store, Some(true));
            assert_eq!(service_tier.as_deref(), Some("default"));
            assert_eq!(truncation, Some(json!("auto")));
            assert_eq!(reasoning, Some(json!({"effort":"high"})));
        }
        other => panic!("Expected OpenAIResponses, got {other:?}"),
    }
}

#[test]
fn test_decode_response_status_completed() {
    let codec = OpenAIResponsesCodec;
    let response = json!({
        "status": "completed",
        "output": []
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.finish_reason, Some(FinishReason::Complete));
}

#[test]
fn test_decode_response_status_incomplete_max_output_tokens() {
    let codec = OpenAIResponsesCodec;
    let response = json!({
        "status": "incomplete",
        "output": [],
        "incomplete_details": { "reason": "max_output_tokens" }
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.finish_reason, Some(FinishReason::Length));
}

#[test]
fn test_decode_response_status_incomplete_content_filter() {
    let codec = OpenAIResponsesCodec;
    let response = json!({
        "status": "incomplete",
        "output": [],
        "incomplete_details": { "reason": "content_filter" }
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.finish_reason, Some(FinishReason::ContentFilter));
}

#[test]
fn test_decode_response_status_failed() {
    let codec = OpenAIResponsesCodec;
    let response = json!({
        "status": "failed",
        "output": []
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(
        resp.finish_reason,
        Some(FinishReason::Unknown("failed".into()))
    );
}

#[test]
fn test_decode_response_status_incomplete_no_details() {
    let codec = OpenAIResponsesCodec;
    let response = json!({
        "status": "incomplete",
        "output": []
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(
        resp.finish_reason,
        Some(FinishReason::Unknown("incomplete".into()))
    );
}

#[test]
fn test_decode_response_function_call_uses_call_id() {
    let codec = OpenAIResponsesCodec;
    let response = json!({
        "output": [{
            "type": "function_call",
            "id": "fc_should_not_be_used",
            "name": "search",
            "call_id": "call_correct_id",
            "arguments": "{}",
            "status": "completed"
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    let tc = &resp.tool_calls.unwrap()[0];
    assert_eq!(tc.id, "call_correct_id");
    assert_ne!(tc.id, "fc_should_not_be_used");
}

#[test]
fn test_decode_response_tool_call_arguments_parsed() {
    let codec = OpenAIResponsesCodec;
    let response = json!({
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "name": "search",
            "call_id": "call_1",
            "arguments": "{\"query\":\"weather\",\"limit\":5}",
            "status": "completed"
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    let tc = &resp.tool_calls.unwrap()[0];
    assert_eq!(tc.arguments, json!({"query": "weather", "limit": 5}));
    assert!(tc.arguments.is_object());
}

#[test]
fn test_decode_response_usage_mapping() {
    let codec = OpenAIResponsesCodec;
    let response = json!({
        "output": [],
        "usage": {
            "input_tokens": 75,
            "output_tokens": 1186,
            "total_tokens": 1261,
            "input_tokens_details": { "cached_tokens": 42 }
        }
    });
    let resp = codec.decode_response(&response).unwrap();
    let usage = resp.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(75));
    assert_eq!(usage.completion_tokens, Some(1186));
    assert_eq!(usage.total_tokens, Some(1261));
    assert_eq!(usage.cache_read_tokens, Some(42));
    assert_eq!(usage.cache_write_tokens, None);
}

#[test]
fn test_decode_response_multiple_output_text_items() {
    let codec = OpenAIResponsesCodec;
    let response = json!({
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    { "type": "output_text", "text": "First part." },
                    { "type": "output_text", "text": "Second part." }
                ]
            }
        ]
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(
        resp.message,
        Some(MessageContent::Text("First part.\nSecond part.".into()))
    );
}

#[test]
fn test_decode_response_only_reasoning_items() {
    let codec = OpenAIResponsesCodec;
    let response = json!({
        "output": [{
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "gAAAAABo..."
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    // No message content when there's only reasoning
    assert_eq!(resp.message, None);
    // Reasoning items captured in api_specific
    match resp.api_specific.unwrap() {
        ApiSpecificResponse::OpenAIResponses { output_items, .. } => {
            let items = output_items.unwrap();
            assert_eq!(items.len(), 1);
            assert_eq!(items[0]["type"], "reasoning");
        }
        other => panic!("Expected OpenAIResponses, got {other:?}"),
    }
}

#[test]
fn test_decode_response_extra_fields_preserved() {
    let codec = OpenAIResponsesCodec;
    let response = json!({
        "id": "resp_test",
        "object": "response",
        "created_at": 1234567890.0,
        "model": "gpt-4o",
        "status": "completed",
        "output": [],
        "custom_future_field": "preserved_value",
        "another_field": 42
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.extra.get("object"), Some(&json!("response")));
    assert_eq!(resp.extra.get("created_at"), Some(&json!(1234567890.0)));
    assert_eq!(
        resp.extra.get("custom_future_field"),
        Some(&json!("preserved_value"))
    );
    assert_eq!(resp.extra.get("another_field"), Some(&json!(42)));
}

#[test]
fn test_decode_minimal_response() {
    let codec = OpenAIResponsesCodec;
    let response = json!({});
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.id, None);
    assert_eq!(resp.model, None);
    assert_eq!(resp.message, None);
    assert!(resp.tool_calls.is_none() || resp.tool_calls.as_ref().unwrap().is_empty());
    assert_eq!(resp.usage, None);
}

#[test]
fn test_decode_invalid_json() {
    let codec = OpenAIResponsesCodec;
    let response = json!("not an object");
    let result = codec.decode_response(&response);
    assert!(result.is_err());
}

// ===================================================================
// Request decode tests
// ===================================================================

#[test]
fn test_decode_request_with_input_array() {
    let codec = OpenAIResponsesCodec;
    let mut request_json = fixture_json(include_str!(
        "../../fixtures/codec/openai_responses/strict_messages_array.json"
    ));
    request_json["tools"] = json!([{
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate math",
            "parameters": {"type": "object"}
        }
    }]);
    let request = make_request(request_json);
    let annotated = codec.decode(&request).unwrap();
    assert_eq!(annotated.model, Some("gpt-4o".into()));

    // instructions becomes system message (first)
    assert!(annotated.messages.len() >= 2);
    assert_eq!(annotated.system_prompt(), Some("Be helpful."));

    // input items become messages (after system)
    // System + 3 input items = 4 total messages
    assert_eq!(annotated.messages.len(), 4);

    // Tools present
    let tools = annotated.tools.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "calculate");
}

#[test]
fn test_decode_request_with_input_string() {
    let codec = OpenAIResponsesCodec;
    let request = make_request(json!({
        "model": "gpt-4o",
        "input": "Hello, world!"
    }));
    let annotated = codec.decode(&request).unwrap();
    assert_eq!(annotated.messages.len(), 1);
    assert_eq!(annotated.last_user_message(), Some("Hello, world!"));
}

#[test]
fn test_decode_request_max_output_tokens() {
    let codec = OpenAIResponsesCodec;
    let request = make_request(json!({
        "model": "gpt-4o",
        "input": "Hi",
        "max_output_tokens": 500
    }));
    let annotated = codec.decode(&request).unwrap();
    let params = annotated.params.unwrap();
    assert_eq!(params.max_tokens, Some(500));
}

#[test]
fn test_decode_request_extra_fields() {
    let codec = OpenAIResponsesCodec;
    let request = make_request(json!({
        "model": "gpt-4o",
        "input": "Hi",
        "store": true,
        "metadata": { "key": "value" },
        "tool_choice": "auto"
    }));
    let annotated = codec.decode(&request).unwrap();
    assert_eq!(annotated.store, Some(true));
    assert_eq!(annotated.metadata, Some(json!({"key": "value"})));
}

#[test]
fn test_decode_request_openai_controls_typed() {
    let codec = OpenAIResponsesCodec;
    let request = make_request(json!({
        "model": "gpt-4o",
        "input": "Hi",
        "store": true,
        "previous_response_id": "resp_prev",
        "truncation": "disabled",
        "reasoning": { "effort": "high" },
        "include": ["reasoning.encrypted_content"],
        "user": "u123",
        "metadata": { "k": "v" },
        "service_tier": "default",
        "parallel_tool_calls": true,
        "max_output_tokens": 777,
        "max_tool_calls": 3,
        "top_logprobs": 2,
        "stream": true
    }));
    let annotated = codec.decode(&request).unwrap();
    assert_eq!(annotated.store, Some(true));
    assert_eq!(annotated.previous_response_id.as_deref(), Some("resp_prev"));
    assert_eq!(annotated.truncation, Some(json!("disabled")));
    assert_eq!(annotated.reasoning, Some(json!({"effort":"high"})));
    assert_eq!(
        annotated.include,
        Some(json!(["reasoning.encrypted_content"]))
    );
    assert_eq!(annotated.user.as_deref(), Some("u123"));
    assert_eq!(annotated.metadata, Some(json!({"k":"v"})));
    assert_eq!(annotated.service_tier.as_deref(), Some("default"));
    assert_eq!(annotated.parallel_tool_calls, Some(true));
    assert_eq!(annotated.max_output_tokens, Some(777));
    assert_eq!(annotated.max_tool_calls, Some(3));
    assert_eq!(annotated.top_logprobs, Some(2));
    assert_eq!(annotated.stream, Some(true));
}

#[test]
fn test_decode_request_input_array_preserves_unparsed_items_in_extra() {
    let codec = OpenAIResponsesCodec;
    let request = make_request(fixture_json(include_str!(
        "../../fixtures/codec/openai_responses/mixed_input_with_function_call_output.json"
    )));
    let annotated = codec.decode(&request).unwrap();
    // strict-first behavior: no partial message extraction on mixed arrays
    assert!(annotated.messages.is_empty());
    assert_eq!(
        annotated
            .extra
            .get("_openai_responses_unparsed_input_items"),
        Some(&json!([
            { "role": "user", "content": "hello" },
            { "type": "function_call_output", "call_id": "call_1", "output": "ok" }
        ]))
    );
}

#[test]
fn test_decode_request_accepts_anthropic_hint_tool_choice() {
    let codec = OpenAIResponsesCodec;
    let request = make_request(fixture_json(include_str!(
        "../../fixtures/codec/openai_responses/anthropic_tool_choice_hint.json"
    )));
    let annotated = codec.decode(&request).unwrap();
    assert_eq!(annotated.tool_choice, Some(ToolChoice::Auto));
    assert_eq!(annotated.parallel_tool_calls, Some(false));
}

#[test]
fn test_decode_request_litellm_reasoning_input_item_preserved_and_controls_extracted() {
    let codec = OpenAIResponsesCodec;
    let request = make_request(fixture_json(include_str!(
        "../../fixtures/codec/openai_responses/litellm_reasoning_input_item.json"
    )));
    let annotated = codec.decode(&request).unwrap();
    // strict-first parse: mixed input array preserved whole in extra
    assert!(annotated.messages.is_empty());
    assert!(
        annotated
            .extra
            .get("_openai_responses_unparsed_input_items")
            .is_some()
    );
    // stable controls still extracted
    assert_eq!(annotated.store, Some(true));
    assert_eq!(annotated.parallel_tool_calls, Some(true));
    assert_eq!(annotated.truncation, Some(json!("disabled")));
    assert_eq!(annotated.reasoning, Some(json!({"effort":"minimal"})));
}

#[test]
fn test_decode_request_sglang_extensions_preserved_in_extra() {
    let codec = OpenAIResponsesCodec;
    let request = make_request(fixture_json(include_str!(
        "../../fixtures/codec/openai_responses/sglang_responses_request_with_extensions.json"
    )));
    let annotated = codec.decode(&request).unwrap();
    // core controls extracted
    assert_eq!(annotated.store, Some(true));
    assert_eq!(annotated.parallel_tool_calls, Some(true));
    assert_eq!(annotated.truncation, Some(json!("auto")));
    assert_eq!(annotated.reasoning, Some(json!({"effort":"low"})));
    assert_eq!(annotated.tool_choice, Some(ToolChoice::None));
    // sglang-specific extensions retained losslessly
    assert_eq!(
        annotated.extra.get("request_id"),
        Some(&json!("resp_custom_1"))
    );
    assert_eq!(annotated.extra.get("priority"), Some(&json!(3)));
    assert_eq!(annotated.extra.get("extra_key"), Some(&json!("tenant-a")));
    assert_eq!(annotated.extra.get("cache_salt"), Some(&json!("salt-123")));
    assert_eq!(annotated.extra.get("top_k"), Some(&json!(40)));
    assert_eq!(annotated.extra.get("min_p"), Some(&json!(0.05)));
    assert_eq!(
        annotated.extra.get("repetition_penalty"),
        Some(&json!(1.02))
    );
}

// ===================================================================
// Request encode tests
// ===================================================================

#[test]
fn test_encode_round_trip_preserves_unmodeled_fields() {
    let codec = OpenAIResponsesCodec;
    let original = make_request(json!({
        "model": "gpt-4o",
        "instructions": "Be helpful.",
        "input": [
            { "role": "user", "content": "Hello" }
        ],
        "store": true,
        "metadata": { "session": "abc" },
        "max_output_tokens": 100,
        "temperature": 0.7
    }));
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();
    let obj = encoded.content.as_object().unwrap();
    // Unmodeled fields preserved
    assert_eq!(obj.get("store"), Some(&json!(true)));
    assert_eq!(obj.get("metadata"), Some(&json!({"session": "abc"})));
}

#[test]
fn test_encode_writes_instructions_and_input() {
    let codec = OpenAIResponsesCodec;
    let original = make_request(json!({
        "model": "gpt-4o",
        "instructions": "Be concise.",
        "input": [
            { "role": "user", "content": "Hello" }
        ]
    }));
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();
    let obj = encoded.content.as_object().unwrap();
    // instructions should be present
    assert!(obj.contains_key("instructions"));
    // input should be present
    assert!(obj.contains_key("input"));
    // Should NOT contain "messages"
    assert!(!obj.contains_key("messages"));
}

#[test]
fn test_encode_writes_max_output_tokens() {
    let codec = OpenAIResponsesCodec;
    let original = make_request(json!({
        "model": "gpt-4o",
        "input": "Hi",
        "max_output_tokens": 200
    }));
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();
    let obj = encoded.content.as_object().unwrap();
    // Should use max_output_tokens, not max_tokens
    assert_eq!(obj.get("max_output_tokens"), Some(&json!(200)));
    assert!(!obj.contains_key("max_tokens"));
}

#[test]
fn test_encode_request_openai_controls_typed() {
    let codec = OpenAIResponsesCodec;
    let mut annotated = codec
        .decode(&make_request(json!({"model":"gpt-4o","input":"hello"})))
        .unwrap();
    annotated.store = Some(false);
    annotated.previous_response_id = Some("resp_1".into());
    annotated.truncation = Some(json!("auto"));
    annotated.reasoning = Some(json!({"effort":"low"}));
    annotated.include = Some(json!(["reasoning.encrypted_content"]));
    annotated.user = Some("abc".into());
    annotated.metadata = Some(json!({"x":1}));
    annotated.service_tier = Some("default".into());
    annotated.parallel_tool_calls = Some(false);
    annotated.max_output_tokens = Some(222);
    annotated.max_tool_calls = Some(5);
    annotated.top_logprobs = Some(1);
    annotated.stream = Some(true);

    let encoded = codec
        .encode(
            &annotated,
            &make_request(json!({"model":"gpt-4o","input":"hello"})),
        )
        .unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert_eq!(obj.get("store"), Some(&json!(false)));
    assert_eq!(obj.get("previous_response_id"), Some(&json!("resp_1")));
    assert_eq!(obj.get("truncation"), Some(&json!("auto")));
    assert_eq!(obj.get("reasoning"), Some(&json!({"effort":"low"})));
    assert_eq!(
        obj.get("include"),
        Some(&json!(["reasoning.encrypted_content"]))
    );
    assert_eq!(obj.get("user"), Some(&json!("abc")));
    assert_eq!(obj.get("metadata"), Some(&json!({"x":1})));
    assert_eq!(obj.get("service_tier"), Some(&json!("default")));
    assert_eq!(obj.get("parallel_tool_calls"), Some(&json!(false)));
    assert_eq!(obj.get("max_output_tokens"), Some(&json!(222)));
    assert_eq!(obj.get("max_tool_calls"), Some(&json!(5)));
    assert_eq!(obj.get("top_logprobs"), Some(&json!(1)));
    assert_eq!(obj.get("stream"), Some(&json!(true)));
}

#[test]
fn test_encode_extra_overrides_typed_controls() {
    let codec = OpenAIResponsesCodec;
    let mut annotated = codec
        .decode(&make_request(json!({"model":"gpt-4o","input":"hello"})))
        .unwrap();
    annotated.store = Some(false);
    annotated.extra.insert("store".into(), json!(true));
    let encoded = codec
        .encode(
            &annotated,
            &make_request(json!({"model":"gpt-4o","input":"hello"})),
        )
        .unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert_eq!(obj.get("store"), Some(&json!(true)));
}

#[test]
fn test_helper_and_error_paths_cover_remaining_responses_branches() {
    assert_eq!(
        parse_arguments("{not-json"),
        Json::String("{not-json".into())
    );
    assert_eq!(json_f64(f64::NAN), Json::Null);
    assert_eq!(
        map_responses_finish_reason(Some("incomplete"), Some(&json!({"reason": "new_reason"}))),
        Some(FinishReason::Unknown("new_reason".into()))
    );

    let codec = OpenAIResponsesCodec;

    match codec
        .decode(&make_request(json!("not-an-object")))
        .unwrap_err()
    {
        FlowError::Internal(message) => {
            assert!(message.contains("request content is not an object"))
        }
        other => panic!("unexpected decode error: {other}"),
    }

    match codec
        .decode(&make_request(json!({
            "input": "hello",
            "tools": "bad-tools"
        })))
        .unwrap_err()
    {
        FlowError::Internal(message) => {
            assert!(message.contains("OpenAI Responses tools decode"));
        }
        other => panic!("unexpected tools decode error: {other}"),
    }

    let annotated = AnnotatedLlmRequest {
        messages: vec![super::super::request::Message::User {
            content: MessageContent::Text("hello".into()),
            name: None,
        }],
        model: Some("gpt-4.1-mini".into()),
        params: Some(GenerationParams {
            temperature: Some(0.1),
            max_tokens: Some(32),
            top_p: Some(0.95),
            stop: None,
        }),
        tools: Some(vec![ToolDefinition {
            tool_type: "function".into(),
            function: super::super::request::FunctionDefinition {
                name: "lookup".into(),
                description: Some("Look up data".into()),
                parameters: Some(json!({"type": "object"})),
            },
        }]),
        tool_choice: Some(ToolChoice::Auto),
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: serde_json::Map::new(),
    };

    let encoded = codec
        .encode(
            &annotated,
            &make_request(json!({
                "model": "gpt-4o",
                "instructions": "drop me",
                "input": []
            })),
        )
        .unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert!(!obj.contains_key("instructions"));
    assert_eq!(obj.get("temperature"), Some(&json!(0.1)));
    assert_eq!(obj.get("top_p"), Some(&json!(0.95)));
    assert_eq!(obj.get("max_output_tokens"), Some(&json!(32)));
    assert!(obj.get("tools").unwrap().is_array());
    assert_eq!(obj.get("tool_choice"), Some(&json!("auto")));

    match codec.encode(&annotated, &make_request(json!("still-not-an-object"))) {
        Err(FlowError::Internal(message)) => {
            assert!(message.contains("original content is not an object"));
        }
        other => panic!("unexpected encode result: {other:?}"),
    }
}

// ===================================================================
// Streaming codec tests
// ===================================================================

use super::super::streaming::StreamingCodec;

#[test]
fn openai_responses_streaming_codec_uses_terminal_snapshot() {
    // Common case: response.completed carries the full final state. Streaming codec emits that
    // verbatim; per-item accumulator is unused.
    let codec = OpenAIResponsesStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "type": "response.created",
        "response": {"id": "resp_1", "model": "gpt-5.5", "status": "in_progress",
                     "output": [], "usage": null}
    }))
    .unwrap();
    collector(json!({
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "model": "gpt-5.5",
            "status": "completed",
            "output": [
                {"type": "message", "content": [
                    {"type": "output_text", "text": "Hello, world."}
                ]}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
        }
    }))
    .unwrap();

    let assembled = finalizer();
    let annotated = OpenAIResponsesCodec
        .decode_response(&assembled)
        .expect("assembled response should decode through the existing codec");
    assert_eq!(annotated.id.as_deref(), Some("resp_1"));
    assert_eq!(annotated.model.as_deref(), Some("gpt-5.5"));
    assert_eq!(annotated.finish_reason, Some(FinishReason::Complete));
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("Hello, world.".to_string()))
    );
    let usage = annotated.usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, Some(10));
    assert_eq!(usage.completion_tokens, Some(4));
}

#[test]
fn openai_responses_streaming_codec_assembles_from_output_item_done_when_terminal_lacks_output() {
    // Schema variant: terminal `response.completed` event omits `output` (or sends empty array).
    // Codec falls back to per-item accumulator populated by output_item.done.
    let codec = OpenAIResponsesStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "type": "response.created",
        "response": {"id": "resp_x", "model": "gpt-5.5", "status": "in_progress", "output": []}
    }))
    .unwrap();
    collector(json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {"type": "message", "content": [
            {"type": "output_text", "text": "Hi from item 0."}
        ]}
    }))
    .unwrap();
    collector(json!({
        "type": "response.output_item.done",
        "output_index": 1,
        "item": {
            "type": "function_call",
            "call_id": "call_42",
            "name": "lookup",
            "arguments": "{\"q\": \"weather\"}"
        }
    }))
    .unwrap();
    collector(json!({
        "type": "response.completed",
        "response": {
            "id": "resp_x",
            "model": "gpt-5.5",
            "status": "completed",
            "usage": {"input_tokens": 5, "output_tokens": 3}
        }
    }))
    .unwrap();

    let assembled = finalizer();
    let annotated = OpenAIResponsesCodec
        .decode_response(&assembled)
        .expect("assembled response should decode");
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("Hi from item 0.".to_string()))
    );
    let tool_calls = annotated.tool_calls.expect("function call extracted");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "call_42");
    assert_eq!(tool_calls[0].name, "lookup");
    assert_eq!(tool_calls[0].arguments, json!({"q": "weather"}));
}

#[test]
fn openai_responses_streaming_codec_preserves_incomplete_terminal_state() {
    // response.incomplete with `reason: max_output_tokens` should map to FinishReason::Length
    // through the existing decoder. The streaming codec must surface incomplete_details intact.
    let codec = OpenAIResponsesStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "type": "response.incomplete",
        "response": {
            "id": "resp_inc",
            "model": "gpt-5.5",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {"type": "message", "content": [
                    {"type": "output_text", "text": "partial..."}
                ]}
            ]
        }
    }))
    .unwrap();

    let assembled = finalizer();
    let annotated = OpenAIResponsesCodec
        .decode_response(&assembled)
        .expect("assembled response should decode");
    assert_eq!(annotated.finish_reason, Some(FinishReason::Length));
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("partial...".to_string()))
    );
}

#[test]
fn openai_responses_streaming_codec_ignores_per_token_deltas() {
    // output_text.delta events are intentionally not accumulated — their content is redelivered
    // in output_item.done. Codec must not double-count or insert delta-only state.
    let codec = OpenAIResponsesStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "type": "response.created",
        "response": {"id": "resp_d", "model": "gpt-5.5", "status": "in_progress", "output": []}
    }))
    .unwrap();
    collector(json!({
        "type": "response.output_text.delta",
        "output_index": 0, "content_index": 0, "delta": "Hel"
    }))
    .unwrap();
    collector(json!({
        "type": "response.output_text.delta",
        "output_index": 0, "content_index": 0, "delta": "lo"
    }))
    .unwrap();
    collector(json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {"type": "message", "content": [
            {"type": "output_text", "text": "Hello"}
        ]}
    }))
    .unwrap();
    collector(json!({
        "type": "response.completed",
        "response": {"id": "resp_d", "model": "gpt-5.5", "status": "completed", "output": []}
    }))
    .unwrap();

    let assembled = finalizer();
    let annotated = OpenAIResponsesCodec
        .decode_response(&assembled)
        .expect("assembled response should decode");
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("Hello".to_string()))
    );
}
