// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for openai chat in the NeMo Flow core crate.

use super::*;
use serde_json::json;

use super::super::request::{ContentPart, MessageContent, OpenAiImageUrl};
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

/// Full Chat Completions response with text + tool calls + usage + cached tokens.
fn full_chat_response() -> Json {
    json!({
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4o-2024-08-06",
        "service_tier": "default",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello!",
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"NYC\"}"
                    }
                }]
            },
            "finish_reason": "stop",
            "logprobs": {
                "content": [{
                    "token": "Hello",
                    "logprob": -0.317
                }]
            }
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
            "prompt_tokens_details": {
                "cached_tokens": 5
            }
        }
    })
}

// ===================================================================
// Response decode tests
// ===================================================================

#[test]
fn test_decode_full_response() {
    let codec = OpenAIChatCodec;
    let resp = codec.decode_response(&full_chat_response()).unwrap();

    assert_eq!(resp.id, Some("chatcmpl-abc123".into()));
    assert_eq!(resp.model, Some("gpt-4o-2024-08-06".into()));
    assert_eq!(resp.message, Some(MessageContent::Text("Hello!".into())));
    assert_eq!(resp.finish_reason, Some(FinishReason::Complete));

    let tool_calls = resp.tool_calls.unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "call_abc123");
    assert_eq!(tool_calls[0].name, "get_weather");
    assert_eq!(tool_calls[0].arguments, json!({"city": "NYC"}));

    let usage = resp.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(9));
    assert_eq!(usage.completion_tokens, Some(12));
    assert_eq!(usage.total_tokens, Some(21));
    assert_eq!(usage.cache_read_tokens, Some(5));
    assert_eq!(usage.cache_write_tokens, None);
}

#[test]
fn test_decode_response_cached_tokens() {
    let codec = OpenAIChatCodec;
    let response = json!({
        "id": "chatcmpl-cached",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "Hi" },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {
                "cached_tokens": 42
            }
        }
    });
    let resp = codec.decode_response(&response).unwrap();
    let usage = resp.usage.unwrap();
    assert_eq!(usage.cache_read_tokens, Some(42));
}

#[test]
fn test_decode_response_finish_reason_stop() {
    let codec = OpenAIChatCodec;
    let response = json!({
        "choices": [{
            "message": { "content": "done" },
            "finish_reason": "stop"
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.finish_reason, Some(FinishReason::Complete));
}

#[test]
fn test_decode_response_finish_reason_tool_calls() {
    let codec = OpenAIChatCodec;
    let response = json!({
        "choices": [{
            "message": { "content": null },
            "finish_reason": "tool_calls"
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.finish_reason, Some(FinishReason::ToolUse));
}

#[test]
fn test_decode_response_finish_reason_length() {
    let codec = OpenAIChatCodec;
    let response = json!({
        "choices": [{
            "message": { "content": "truncated" },
            "finish_reason": "length"
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.finish_reason, Some(FinishReason::Length));
}

#[test]
fn test_decode_response_finish_reason_content_filter() {
    let codec = OpenAIChatCodec;
    let response = json!({
        "choices": [{
            "message": { "content": "" },
            "finish_reason": "content_filter"
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.finish_reason, Some(FinishReason::ContentFilter));
}

#[test]
fn test_decode_response_finish_reason_unknown() {
    let codec = OpenAIChatCodec;
    let response = json!({
        "choices": [{
            "message": { "content": "" },
            "finish_reason": "some_new_reason"
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(
        resp.finish_reason,
        Some(FinishReason::Unknown("some_new_reason".into()))
    );
}

#[test]
fn test_decode_response_tool_call_arguments_parsed() {
    let codec = OpenAIChatCodec;
    let response = json!({
        "choices": [{
            "message": {
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": "{\"query\":\"weather\",\"limit\":5}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    let tc = &resp.tool_calls.unwrap()[0];
    assert_eq!(tc.arguments, json!({"query": "weather", "limit": 5}));
    // Arguments should be a Json object, not a Json::String
    assert!(tc.arguments.is_object());
}

#[test]
fn test_decode_response_api_specific_fields() {
    let codec = OpenAIChatCodec;
    let resp = codec.decode_response(&full_chat_response()).unwrap();
    match resp.api_specific.unwrap() {
        ApiSpecificResponse::OpenAIChat {
            logprobs,
            system_fingerprint,
            service_tier,
        } => {
            assert!(logprobs.is_some());
            assert_eq!(system_fingerprint, Some("fp_44709d6fcb".into()));
            assert_eq!(service_tier, Some("default".into()));
        }
        other => panic!("Expected OpenAIChat, got {other:?}"),
    }
}

#[test]
fn test_decode_response_extra_fields_preserved() {
    let codec = OpenAIChatCodec;
    let response = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [{
            "message": { "content": "hi" },
            "finish_reason": "stop"
        }],
        "custom_future_field": "preserved_value",
        "another_field": 42
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.extra.get("object"), Some(&json!("chat.completion")));
    assert_eq!(resp.extra.get("created"), Some(&json!(1234567890)));
    assert_eq!(
        resp.extra.get("custom_future_field"),
        Some(&json!("preserved_value"))
    );
    assert_eq!(resp.extra.get("another_field"), Some(&json!(42)));
}

#[test]
fn test_decode_minimal_response() {
    let codec = OpenAIChatCodec;
    let response = json!({});
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.id, None);
    assert_eq!(resp.model, None);
    assert_eq!(resp.message, None);
    assert_eq!(resp.tool_calls, None);
    assert_eq!(resp.finish_reason, None);
    assert_eq!(resp.usage, None);
}

#[test]
fn test_decode_invalid_json_type() {
    let codec = OpenAIChatCodec;
    // A JSON string (not an object) should fail to decode
    let response = json!("not an object");
    let result = codec.decode_response(&response);
    assert!(result.is_err());
}

// ===================================================================
// Tool call robustness: partial / missing fields (issue #6)
// ===================================================================

#[test]
fn test_decode_response_tool_call_missing_function() {
    let codec = OpenAIChatCodec;
    let response = json!({
        "choices": [{
            "message": {
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function"
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    let tool_calls = resp.tool_calls.unwrap();
    assert!(
        tool_calls.is_empty(),
        "tool call without function body should be skipped"
    );
}

#[test]
fn test_decode_response_tool_call_null_function() {
    let codec = OpenAIChatCodec;
    let response = json!({
        "choices": [{
            "message": {
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": null
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    let tool_calls = resp.tool_calls.unwrap();
    assert!(
        tool_calls.is_empty(),
        "tool call with null function should be skipped"
    );
}

#[test]
fn test_decode_response_tool_call_missing_id() {
    let codec = OpenAIChatCodec;
    let response = json!({
        "choices": [{
            "message": {
                "content": null,
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"NYC\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    let tool_calls = resp.tool_calls.unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].id, "",
        "missing id should default to empty string"
    );
    assert_eq!(tool_calls[0].name, "get_weather");
}

#[test]
fn test_decode_response_tool_call_missing_function_name() {
    let codec = OpenAIChatCodec;
    let response = json!({
        "choices": [{
            "message": {
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "arguments": "{\"city\":\"NYC\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    let tool_calls = resp.tool_calls.unwrap();
    assert!(
        tool_calls.is_empty(),
        "tool call without function name should be skipped"
    );
}

#[test]
fn test_decode_response_tool_call_missing_arguments() {
    let codec = OpenAIChatCodec;
    let response = json!({
        "choices": [{
            "message": {
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_time"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    let tool_calls = resp.tool_calls.unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].name, "get_time");
    assert_eq!(
        tool_calls[0].arguments,
        json!({}),
        "missing arguments should default to empty object"
    );
}

#[test]
fn test_decode_response_mixed_valid_and_partial_tool_calls() {
    let codec = OpenAIChatCodec;
    let response = json!({
        "choices": [{
            "message": {
                "content": null,
                "tool_calls": [
                    {
                        "id": "call_good",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"NYC\"}"
                        }
                    },
                    {
                        "id": "call_partial",
                        "type": "function"
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": "{}"
                        }
                    }
                ]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    let tool_calls = resp.tool_calls.unwrap();
    assert_eq!(
        tool_calls.len(),
        2,
        "only complete tool calls should be kept"
    );
    assert_eq!(tool_calls[0].id, "call_good");
    assert_eq!(tool_calls[0].name, "get_weather");
    assert_eq!(tool_calls[1].id, "", "missing id defaults to empty string");
    assert_eq!(tool_calls[1].name, "get_time");
}

#[test]
fn test_decode_response_tool_call_empty_tool_calls_array() {
    let codec = OpenAIChatCodec;
    let response = json!({
        "choices": [{
            "message": {
                "content": "No tools needed",
                "tool_calls": []
            },
            "finish_reason": "stop"
        }]
    });
    let resp = codec.decode_response(&response).unwrap();
    let tool_calls = resp.tool_calls.unwrap();
    assert!(tool_calls.is_empty());
    assert_eq!(
        resp.message,
        Some(MessageContent::Text("No tools needed".into()))
    );
}

// ===================================================================
// Request decode tests
// ===================================================================

#[test]
fn test_decode_request_full() {
    let codec = OpenAIChatCodec;
    let request = make_request(json!({
        "messages": [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"}
        ],
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 100,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object"}
            }
        }],
        "tool_choice": "auto"
    }));
    let annotated = codec.decode(&request).unwrap();
    assert_eq!(annotated.messages.len(), 2);
    assert_eq!(annotated.model, Some("gpt-4o".into()));

    let params = annotated.params.unwrap();
    assert_eq!(params.temperature, Some(0.7));
    assert_eq!(params.max_tokens, Some(100));

    let tools = annotated.tools.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    assert_eq!(annotated.tool_choice, Some(ToolChoice::Auto));
}

#[test]
fn test_decode_request_max_completion_tokens() {
    let codec = OpenAIChatCodec;
    let request = make_request(json!({
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "gpt-4o",
        "max_completion_tokens": 200
    }));
    let annotated = codec.decode(&request).unwrap();
    let params = annotated.params.unwrap();
    assert_eq!(params.max_tokens, Some(200));
}

#[test]
fn test_decode_request_extra_fields() {
    let codec = OpenAIChatCodec;
    let request = make_request(json!({
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "gpt-4o",
        "stream": true,
        "seed": 42,
        "response_format": {"type": "json_object"}
    }));
    let annotated = codec.decode(&request).unwrap();
    assert_eq!(annotated.stream, Some(true));
    assert_eq!(annotated.extra.get("seed"), Some(&json!(42)));
    assert_eq!(
        annotated.extra.get("response_format"),
        Some(&json!({"type": "json_object"}))
    );
}

#[test]
fn test_decode_request_openai_chat_typed_controls() {
    let codec = OpenAIChatCodec;
    let request = make_request(json!({
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "gpt-4o",
        "store": true,
        "user": "u1",
        "metadata": {"k":"v"},
        "service_tier": "default",
        "parallel_tool_calls": true,
        "top_logprobs": 2,
        "stream": true
    }));
    let annotated = codec.decode(&request).unwrap();
    assert_eq!(annotated.store, Some(true));
    assert_eq!(annotated.user.as_deref(), Some("u1"));
    assert_eq!(annotated.metadata, Some(json!({"k":"v"})));
    assert_eq!(annotated.service_tier.as_deref(), Some("default"));
    assert_eq!(annotated.parallel_tool_calls, Some(true));
    assert_eq!(annotated.top_logprobs, Some(2));
    assert_eq!(annotated.stream, Some(true));
}

#[test]
fn test_decode_request_no_messages_key() {
    let codec = OpenAIChatCodec;
    let request = make_request(json!({
        "model": "gpt-4o"
    }));
    let annotated = codec.decode(&request).unwrap();
    assert!(annotated.messages.is_empty());
}

#[test]
fn test_decode_request_multimodal_image_url_parts() {
    let codec = OpenAIChatCodec;
    let request = make_request(json!({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}}
            ]
        }],
        "model": "gpt-4o"
    }));
    let annotated = codec.decode(&request).unwrap();
    match &annotated.messages[0] {
        Message::User { content, .. } => match content {
            MessageContent::Parts(parts) => {
                assert_eq!(
                    parts,
                    &vec![
                        ContentPart::Text {
                            text: "describe this".into()
                        },
                        ContentPart::ImageUrl {
                            image_url: OpenAiImageUrl {
                                url: "https://example.com/cat.png".into(),
                                detail: Some("high".into())
                            }
                        }
                    ]
                );
            }
            _ => panic!("expected parts content"),
        },
        _ => panic!("expected user message"),
    }
}

// ===================================================================
// Request encode tests
// ===================================================================

#[test]
fn test_encode_round_trip_preserves_unmodeled_fields() {
    let codec = OpenAIChatCodec;
    let original = make_request(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-4o",
        "stream": true,
        "seed": 42,
        "temperature": 0.7
    }));
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();
    let obj = encoded.content.as_object().unwrap();
    // Unmodeled fields preserved
    assert_eq!(obj.get("stream"), Some(&json!(true)));
    assert_eq!(obj.get("seed"), Some(&json!(42)));
    // Modeled fields present
    assert!(obj.contains_key("messages"));
    assert_eq!(obj.get("model"), Some(&json!("gpt-4o")));
}

#[test]
fn test_encode_with_modified_model() {
    let codec = OpenAIChatCodec;
    let original = make_request(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-4o"
    }));
    let mut annotated = codec.decode(&original).unwrap();
    annotated.model = Some("gpt-4o-mini".into());
    let encoded = codec.encode(&annotated, &original).unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert_eq!(obj.get("model"), Some(&json!("gpt-4o-mini")));
}

#[test]
fn test_encode_writes_openai_chat_typed_controls() {
    let codec = OpenAIChatCodec;
    let mut annotated = codec
        .decode(&make_request(json!({
            "messages": [{"role":"user","content":"hi"}],
            "model": "gpt-4o"
        })))
        .unwrap();
    annotated.store = Some(false);
    annotated.user = Some("u2".into());
    annotated.metadata = Some(json!({"m":1}));
    annotated.service_tier = Some("default".into());
    annotated.parallel_tool_calls = Some(false);
    annotated.top_logprobs = Some(1);
    annotated.stream = Some(true);
    let encoded = codec
        .encode(
            &annotated,
            &make_request(json!({"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o"})),
        )
        .unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert_eq!(obj.get("store"), Some(&json!(false)));
    assert_eq!(obj.get("user"), Some(&json!("u2")));
    assert_eq!(obj.get("metadata"), Some(&json!({"m":1})));
    assert_eq!(obj.get("service_tier"), Some(&json!("default")));
    assert_eq!(obj.get("parallel_tool_calls"), Some(&json!(false)));
    assert_eq!(obj.get("top_logprobs"), Some(&json!(1)));
    assert_eq!(obj.get("stream"), Some(&json!(true)));
}

#[test]
fn test_encode_chat_extra_overrides_typed_controls() {
    let codec = OpenAIChatCodec;
    let mut annotated = codec
        .decode(&make_request(json!({
            "messages": [{"role":"user","content":"hi"}],
            "model": "gpt-4o"
        })))
        .unwrap();
    annotated.store = Some(false);
    annotated.extra.insert("store".into(), json!(true));
    let encoded = codec
        .encode(
            &annotated,
            &make_request(json!({"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o"})),
        )
        .unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert_eq!(obj.get("store"), Some(&json!(true)));
}

#[test]
fn test_encode_request_multimodal_image_url_parts() {
    let codec = OpenAIChatCodec;
    let original = make_request(json!({
        "messages": [{"role":"user","content":"hi"}],
        "model": "gpt-4o"
    }));
    let mut annotated = codec.decode(&original).unwrap();
    annotated.messages = vec![Message::User {
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "describe this".into(),
            },
            ContentPart::ImageUrl {
                image_url: OpenAiImageUrl {
                    url: "https://example.com/cat.png".into(),
                    detail: Some("low".into()),
                },
            },
        ]),
        name: None,
    }];
    let encoded = codec.encode(&annotated, &original).unwrap();
    assert_eq!(
        encoded.content["messages"][0]["content"][1]["type"],
        json!("image_url")
    );
    assert_eq!(
        encoded.content["messages"][0]["content"][1]["image_url"]["url"],
        json!("https://example.com/cat.png")
    );
}

#[test]
fn test_encode_restores_max_completion_tokens_key() {
    let codec = OpenAIChatCodec;
    let original = make_request(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-4o",
        "max_completion_tokens": 200
    }));
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();
    let obj = encoded.content.as_object().unwrap();
    // Should write back to max_completion_tokens, not max_tokens
    assert_eq!(obj.get("max_completion_tokens"), Some(&json!(200)));
    assert!(!obj.contains_key("max_tokens"));
}

#[test]
fn test_helper_and_error_paths_cover_remaining_chat_branches() {
    assert_eq!(
        parse_arguments("{not-json"),
        Json::String("{not-json".into())
    );
    assert_eq!(json_f64(f64::NAN), Json::Null);

    let codec = OpenAIChatCodec;

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
            "messages": [],
            "tools": "bad-tools"
        })))
        .unwrap_err()
    {
        FlowError::Internal(message) => assert!(message.contains("OpenAI Chat tools decode")),
        other => panic!("unexpected tools decode error: {other}"),
    }

    match codec
        .decode(&make_request(json!({
            "messages": [],
            "tool_choice": []
        })))
        .unwrap_err()
    {
        FlowError::Internal(message) => {
            assert!(message.contains("OpenAI Chat tool_choice decode"));
        }
        other => panic!("unexpected tool_choice decode error: {other}"),
    }

    let annotated = AnnotatedLlmRequest {
        messages: vec![],
        model: Some("gpt-4.1-mini".into()),
        params: Some(GenerationParams {
            temperature: Some(0.2),
            max_tokens: Some(64),
            top_p: Some(0.9),
            stop: Some(vec!["END".into()]),
        }),
        tools: Some(vec![ToolDefinition {
            tool_type: "function".into(),
            function: super::super::request::FunctionDefinition {
                name: "lookup".into(),
                description: Some("Look up data".into()),
                parameters: Some(json!({"type": "object"})),
            },
        }]),
        tool_choice: Some(ToolChoice::Required),
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
                "messages": [],
                "model": "gpt-4o"
            })),
        )
        .unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert_eq!(obj.get("temperature"), Some(&json!(0.2)));
    assert_eq!(obj.get("top_p"), Some(&json!(0.9)));
    assert_eq!(obj.get("stop"), Some(&json!(["END"])));
    assert_eq!(obj.get("max_tokens"), Some(&json!(64)));
    assert!(obj.get("tools").unwrap().is_array());
    assert_eq!(obj.get("tool_choice"), Some(&json!("required")));

    match codec.encode(&annotated, &make_request(json!("still-not-an-object"))) {
        Err(FlowError::Internal(message)) => {
            assert!(message.contains("original content is not an object"));
        }
        other => panic!("unexpected encode result: {other:?}"),
    }
}
// ===================================================================
// stream_options injection tests (for Phoenix / OpenInference usage)
// ===================================================================

/// On streaming requests (`stream: true`), the encoder forces
/// `stream_options.include_usage=true` when the caller did not provide
/// `stream_options`. Without this, OpenAI-compatible providers never emit
/// the terminal usage chunk and token counts are lost in Phoenix traces.
#[test]
fn test_encode_injects_stream_options_on_streaming_request() {
    let codec = OpenAIChatCodec;
    let annotated = AnnotatedLlmRequest {
        messages: vec![],
        model: Some("gpt-4o".into()),
        params: None,
        tools: None,
        tool_choice: None,
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
                "messages": [],
                "model": "gpt-4o",
                "stream": true,
            })),
        )
        .unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert_eq!(
        obj.get("stream_options"),
        Some(&json!({"include_usage": true})),
        "encoder must inject stream_options.include_usage on streaming requests when absent",
    );
}

/// Caller-supplied `stream_options` must be preserved verbatim. This matters
/// for callers that deliberately opt out of usage reporting or pass other
/// options (e.g., `include_usage: false`, future fields).
#[test]
fn test_encode_preserves_caller_stream_options() {
    let codec = OpenAIChatCodec;
    let annotated = AnnotatedLlmRequest {
        messages: vec![],
        model: Some("gpt-4o".into()),
        params: None,
        tools: None,
        tool_choice: None,
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
    let caller_set = json!({
        "messages": [],
        "model": "gpt-4o",
        "stream": true,
        "stream_options": { "include_usage": false }
    });
    let encoded = codec.encode(&annotated, &make_request(caller_set)).unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert_eq!(
        obj.get("stream_options"),
        Some(&json!({"include_usage": false})),
        "caller-provided stream_options must be preserved verbatim",
    );
}

/// Per the OpenAI Chat Completions spec, `stream_options` is only valid on
/// streaming requests (`stream: true`). The encoder must not inject it
/// when `stream` is false or absent, even though usage telemetry would
/// otherwise be desirable.
#[test]
fn test_encode_does_not_inject_stream_options_on_non_streaming() {
    let codec = OpenAIChatCodec;
    let annotated = AnnotatedLlmRequest {
        messages: vec![],
        model: Some("gpt-4o".into()),
        params: None,
        tools: None,
        tool_choice: None,
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
            &make_request(json!({"messages": [], "model": "gpt-4o"})),
        )
        .unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert!(
        !obj.contains_key("stream_options"),
        "stream_options must not be injected when stream key is absent",
    );

    let encoded = codec
        .encode(
            &annotated,
            &make_request(json!({
                "messages": [],
                "model": "gpt-4o",
                "stream": false,
            })),
        )
        .unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert!(
        !obj.contains_key("stream_options"),
        "stream_options must not be injected when stream: false",
    );
}

// ===================================================================
// Streaming codec tests
// ===================================================================

use super::super::streaming::StreamingCodec;

#[test]
fn openai_chat_streaming_codec_assembles_text_response() {
    let codec = OpenAIChatStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    // First chunk: top-level fields + role-only delta.
    collector(json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
    }))
    .unwrap();
    // Content deltas.
    for part in &["Hello, ", "world", "."] {
        collector(json!({
            "id": "chatcmpl-1", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": null}]
        }))
        .unwrap();
    }
    // Final chunk with finish_reason and usage (when stream_options.include_usage was set).
    collector(json!({
        "id": "chatcmpl-1", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
    }))
    .unwrap();

    let assembled = finalizer();
    // Verify the assembled object is wire-compatible with non-streaming Chat Completions and
    // round-trips through the existing decoder.
    assert_eq!(assembled["object"], json!("chat.completion"));
    let annotated = OpenAIChatCodec
        .decode_response(&assembled)
        .expect("assembled response should decode");
    assert_eq!(annotated.id.as_deref(), Some("chatcmpl-1"));
    assert_eq!(annotated.model.as_deref(), Some("gpt-4o"));
    assert_eq!(annotated.finish_reason, Some(FinishReason::Complete));
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("Hello, world.".to_string()))
    );
    let usage = annotated.usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, Some(10));
    assert_eq!(usage.completion_tokens, Some(4));
    assert_eq!(usage.total_tokens, Some(14));
}

#[test]
fn openai_chat_streaming_codec_assembles_tool_call_arguments_from_fragments() {
    let codec = OpenAIChatStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    // Initial chunk: role + tool_call header (id, type, function.name).
    collector(json!({
        "id": "chatcmpl-tc", "object": "chat.completion.chunk", "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": ""}
                }]
            },
            "finish_reason": null
        }]
    }))
    .unwrap();
    // Argument fragments arrive over multiple chunks.
    for fragment in &["{\"q", "uery\":", " \"weath", "er\"}"] {
        collector(json!({
            "id": "chatcmpl-tc", "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "function": {"arguments": fragment}
                }]},
                "finish_reason": null
            }]
        }))
        .unwrap();
    }
    collector(json!({
        "id": "chatcmpl-tc", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
    }))
    .unwrap();

    let assembled = finalizer();
    let annotated = OpenAIChatCodec
        .decode_response(&assembled)
        .expect("assembled response should decode");
    assert_eq!(annotated.finish_reason, Some(FinishReason::ToolUse));
    let tool_calls = annotated.tool_calls.expect("tool_calls present");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "call_a");
    assert_eq!(tool_calls[0].name, "lookup");
    assert_eq!(tool_calls[0].arguments, json!({"query": "weather"}));
}

#[test]
fn openai_chat_streaming_codec_emits_null_content_when_only_tool_calls_streamed() {
    // OpenAI's non-streaming wire format uses `content: null` when the assistant only emitted
    // tool calls. The streaming codec must preserve that distinction so downstream consumers
    // (or anyone manually inspecting the assembled JSON) match what a non-streaming response
    // would have shown.
    let codec = OpenAIChatStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "id": "chatcmpl-nc", "object": "chat.completion.chunk", "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "tool_calls": [{
                    "index": 0, "id": "call_x", "type": "function",
                    "function": {"name": "go", "arguments": "{}"}
                }]
            },
            "finish_reason": null
        }]
    }))
    .unwrap();
    collector(json!({
        "id": "chatcmpl-nc", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
    }))
    .unwrap();

    let assembled = finalizer();
    let message = &assembled["choices"][0]["message"];
    assert_eq!(message["content"], json!(null));
    assert!(message["tool_calls"].is_array());
}

#[test]
fn openai_chat_streaming_codec_handles_multiple_choices() {
    // OpenAI Chat Completions supports `n > 1` requesting multiple completions; each gets its
    // own choice index. Streaming codec must keep them separate.
    let codec = OpenAIChatStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "id": "chatcmpl-multi", "object": "chat.completion.chunk", "model": "gpt-4o",
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": null},
            {"index": 1, "delta": {"role": "assistant"}, "finish_reason": null}
        ]
    }))
    .unwrap();
    collector(json!({
        "id": "chatcmpl-multi", "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": "First"}, "finish_reason": null},
            {"index": 1, "delta": {"content": "Second"}, "finish_reason": null}
        ]
    }))
    .unwrap();
    collector(json!({
        "id": "chatcmpl-multi", "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"},
            {"index": 1, "delta": {}, "finish_reason": "stop"}
        ]
    }))
    .unwrap();

    let assembled = finalizer();
    let choices = assembled["choices"].as_array().expect("choices array");
    assert_eq!(choices.len(), 2);
    assert_eq!(choices[0]["index"], json!(0));
    assert_eq!(choices[0]["message"]["content"], json!("First"));
    assert_eq!(choices[1]["index"], json!(1));
    assert_eq!(choices[1]["message"]["content"], json!("Second"));
}

#[test]
fn openai_chat_streaming_codec_skips_null_usage_chunks() {
    // Some streams emit `usage: null` on every chunk and the real usage only on the final chunk.
    // Codec must not let intermediate nulls overwrite a captured usage object.
    let codec = OpenAIChatStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "id": "chatcmpl-u", "object": "chat.completion.chunk", "model": "gpt-4o", "usage": null,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
    }))
    .unwrap();
    collector(json!({
        "id": "chatcmpl-u", "object": "chat.completion.chunk", "usage": null,
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": null}]
    }))
    .unwrap();
    collector(json!({
        "id": "chatcmpl-u", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    }))
    .unwrap();

    let assembled = finalizer();
    assert_eq!(assembled["usage"]["prompt_tokens"], json!(1));
    assert_eq!(assembled["usage"]["total_tokens"], json!(2));
}
