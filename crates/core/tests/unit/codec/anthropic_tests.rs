// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for anthropic in the NeMo Flow core crate.

use super::*;
use serde_json::json;

use super::super::request::{Message, MessageContent, ToolChoice};
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

/// Full Anthropic Messages response with text, tool_use, thinking, usage, etc.
fn full_anthropic_response() -> Json {
    json!({
        "id": "msg_abc123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "content": [
            {
                "type": "thinking",
                "thinking": "Let me analyze...",
                "signature": "sig_xxx"
            },
            {
                "type": "text",
                "text": "Here is my answer."
            },
            {
                "type": "tool_use",
                "id": "toolu_abc123",
                "name": "get_weather",
                "input": { "city": "NYC" }
            },
            {
                "type": "redacted_thinking",
                "data": "gAAAAABo..."
            }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 1024,
            "output_tokens": 256,
            "cache_creation_input_tokens": 512,
            "cache_read_input_tokens": 0
        }
    })
}

// ===================================================================
// Response decode tests
// ===================================================================

#[test]
fn test_decode_full_response() {
    let codec = AnthropicMessagesCodec;
    let resp = codec.decode_response(&full_anthropic_response()).unwrap();

    assert_eq!(resp.id, Some("msg_abc123".into()));
    assert_eq!(resp.model, Some("claude-sonnet-4-20250514".into()));
    assert_eq!(
        resp.message,
        Some(MessageContent::Text("Here is my answer.".into()))
    );
    assert_eq!(resp.finish_reason, Some(FinishReason::Complete));

    let tool_calls = resp.tool_calls.unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "toolu_abc123");
    assert_eq!(tool_calls[0].name, "get_weather");
    assert_eq!(tool_calls[0].arguments, json!({"city": "NYC"}));

    let usage = resp.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(1024));
    assert_eq!(usage.completion_tokens, Some(256));
    assert_eq!(usage.total_tokens, Some(1280));
    assert_eq!(usage.cache_read_tokens, Some(0));
    assert_eq!(usage.cache_write_tokens, Some(512));
}

#[test]
fn test_decode_response_multiple_text_blocks() {
    let codec = AnthropicMessagesCodec;
    let response = json!({
        "id": "msg_multi",
        "model": "claude-sonnet-4-20250514",
        "content": [
            { "type": "text", "text": "First paragraph." },
            { "type": "text", "text": "Second paragraph." }
        ],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 10, "output_tokens": 20 }
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(
        resp.message,
        Some(MessageContent::Text(
            "First paragraph.\nSecond paragraph.".into()
        ))
    );
}

#[test]
fn test_decode_response_tool_use_input_stored_as_json() {
    let codec = AnthropicMessagesCodec;
    let response = json!({
        "id": "msg_tool",
        "model": "claude-sonnet-4-20250514",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_xyz",
                "name": "search",
                "input": { "query": "weather", "limit": 5 }
            }
        ],
        "stop_reason": "tool_use",
        "usage": { "input_tokens": 10, "output_tokens": 20 }
    });
    let resp = codec.decode_response(&response).unwrap();
    let tc = &resp.tool_calls.unwrap()[0];
    assert_eq!(tc.id, "toolu_xyz");
    assert_eq!(tc.name, "search");
    assert_eq!(tc.arguments, json!({"query": "weather", "limit": 5}));
    // Arguments should be a Json object, not a Json::String
    assert!(tc.arguments.is_object());
}

#[test]
fn test_decode_response_finish_reason_end_turn() {
    let codec = AnthropicMessagesCodec;
    let response = json!({
        "content": [{ "type": "text", "text": "done" }],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.finish_reason, Some(FinishReason::Complete));
}

#[test]
fn test_decode_response_finish_reason_max_tokens() {
    let codec = AnthropicMessagesCodec;
    let response = json!({
        "content": [{ "type": "text", "text": "truncated" }],
        "stop_reason": "max_tokens",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.finish_reason, Some(FinishReason::Length));
}

#[test]
fn test_decode_response_finish_reason_tool_use() {
    let codec = AnthropicMessagesCodec;
    let response = json!({
        "content": [{
            "type": "tool_use",
            "id": "toolu_1",
            "name": "fn",
            "input": {}
        }],
        "stop_reason": "tool_use",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.finish_reason, Some(FinishReason::ToolUse));
}

#[test]
fn test_decode_response_finish_reason_stop_sequence() {
    let codec = AnthropicMessagesCodec;
    let response = json!({
        "content": [{ "type": "text", "text": "stopped" }],
        "stop_reason": "stop_sequence",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(
        resp.finish_reason,
        Some(FinishReason::Unknown("stop_sequence".into()))
    );
}

#[test]
fn test_decode_response_usage_mapping() {
    let codec = AnthropicMessagesCodec;
    let response = json!({
        "id": "msg_usage",
        "model": "claude-sonnet-4-20250514",
        "content": [],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 25,
            "cache_creation_input_tokens": 10
        }
    });
    let resp = codec.decode_response(&response).unwrap();
    let usage = resp.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(100));
    assert_eq!(usage.completion_tokens, Some(50));
    assert_eq!(usage.total_tokens, Some(150));
    assert_eq!(usage.cache_read_tokens, Some(25));
    assert_eq!(usage.cache_write_tokens, Some(10));
}

#[test]
fn test_decode_response_thinking_blocks_in_api_specific() {
    let codec = AnthropicMessagesCodec;
    let resp = codec.decode_response(&full_anthropic_response()).unwrap();
    match resp.api_specific.unwrap() {
        ApiSpecificResponse::AnthropicMessages {
            object_type,
            role,
            stop_reason,
            content_blocks,
            stop_sequence,
            service_tier,
            container,
        } => {
            let blocks = content_blocks.unwrap();
            // Should contain ALL content blocks
            assert_eq!(blocks.len(), 4);
            // Verify thinking and redacted_thinking are present
            let types: Vec<&str> = blocks
                .iter()
                .filter_map(|b| b.get("type").and_then(|t| t.as_str()))
                .collect();
            assert!(types.contains(&"thinking"));
            assert!(types.contains(&"redacted_thinking"));
            assert!(types.contains(&"text"));
            assert!(types.contains(&"tool_use"));
            assert_eq!(object_type.as_deref(), Some("message"));
            assert_eq!(role.as_deref(), Some("assistant"));
            assert_eq!(stop_reason.as_deref(), Some("end_turn"));
            assert_eq!(stop_sequence, None);
            assert_eq!(service_tier, None);
            assert_eq!(container, None);
        }
        other => panic!("Expected AnthropicMessages, got {other:?}"),
    }
}

#[test]
fn test_decode_response_stop_sequence_value() {
    let codec = AnthropicMessagesCodec;
    let response = json!({
        "id": "msg_seq",
        "model": "claude-sonnet-4-20250514",
        "content": [{ "type": "text", "text": "stopped" }],
        "stop_reason": "stop_sequence",
        "stop_sequence": "\n\nHuman:",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let resp = codec.decode_response(&response).unwrap();
    match resp.api_specific.unwrap() {
        ApiSpecificResponse::AnthropicMessages {
            stop_sequence,
            content_blocks: _,
            object_type: _,
            role: _,
            stop_reason: _,
            service_tier: _,
            container: _,
        } => {
            assert_eq!(stop_sequence, Some("\n\nHuman:".into()));
        }
        other => panic!("Expected AnthropicMessages, got {other:?}"),
    }
}

#[test]
fn test_decode_response_extra_fields_preserved() {
    let codec = AnthropicMessagesCodec;
    let response = json!({
        "id": "msg_extra",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "content": [{ "type": "text", "text": "hi" }],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 1, "output_tokens": 1 },
        "container": { "id": "container_abc123" }
    });
    let resp = codec.decode_response(&response).unwrap();
    // type/role/container are now modeled in api_specific.
    assert!(resp.extra.get("type").is_none());
    assert!(resp.extra.get("role").is_none());
    assert!(resp.extra.get("container").is_none());
    match resp.api_specific.unwrap() {
        ApiSpecificResponse::AnthropicMessages {
            object_type,
            role,
            container,
            ..
        } => {
            assert_eq!(object_type.as_deref(), Some("message"));
            assert_eq!(role.as_deref(), Some("assistant"));
            assert_eq!(container, Some(json!({"id":"container_abc123"})));
        }
        other => panic!("Expected AnthropicMessages, got {other:?}"),
    }
}

#[test]
fn test_decode_minimal_response() {
    let codec = AnthropicMessagesCodec;
    let response = json!({
        "content": [],
        "usage": { "input_tokens": 0, "output_tokens": 0 }
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.id, None);
    assert_eq!(resp.model, None);
    assert_eq!(resp.message, None);
    assert!(resp.tool_calls.is_none() || resp.tool_calls.as_ref().is_some_and(|t| t.is_empty()));
    assert_eq!(resp.finish_reason, None);
}

#[test]
fn test_decode_invalid_json() {
    let codec = AnthropicMessagesCodec;
    let response = json!("not an object");
    let result = codec.decode_response(&response);
    assert!(result.is_err());
}

#[test]
fn test_decode_response_mcp_tool_use_not_in_tool_calls() {
    let codec = AnthropicMessagesCodec;
    let response = json!({
        "id": "msg_mcp",
        "model": "claude-sonnet-4-20250514",
        "content": [
            {
                "type": "mcp_tool_use",
                "id": "mcptoolu_abc123",
                "name": "search",
                "server_name": "my_server",
                "input": { "query": "test" }
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_abc123",
                "name": "code_execution",
                "input": { "code": "print(1+1)" }
            }
        ],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let resp = codec.decode_response(&response).unwrap();
    // mcp_tool_use and server_tool_use should NOT appear in normalized tool_calls
    assert!(resp.tool_calls.is_none() || resp.tool_calls.as_ref().is_some_and(|t| t.is_empty()));
    // But they should appear in api_specific content_blocks
    match resp.api_specific.unwrap() {
        ApiSpecificResponse::AnthropicMessages { content_blocks, .. } => {
            let blocks = content_blocks.unwrap();
            assert_eq!(blocks.len(), 2);
            let types: Vec<&str> = blocks
                .iter()
                .filter_map(|b| b.get("type").and_then(|t| t.as_str()))
                .collect();
            assert!(types.contains(&"mcp_tool_use"));
            assert!(types.contains(&"server_tool_use"));
        }
        other => panic!("Expected AnthropicMessages, got {other:?}"),
    }
}

// ===================================================================
// Request decode tests
// ===================================================================

#[test]
fn test_decode_request_full() {
    let codec = AnthropicMessagesCodec;
    let request = make_request(json!({
        "system": "Be helpful",
        "messages": [
            { "role": "user", "content": "Hello" }
        ],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "tools": [{
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": { "type": "object", "properties": { "city": { "type": "string" } } }
        }],
        "tool_choice": { "type": "auto" }
    }));
    let annotated = codec.decode(&request).unwrap();

    // System should be prepended as Message::System
    assert_eq!(annotated.messages.len(), 2);
    assert!(
        matches!(&annotated.messages[0], Message::System { content: MessageContent::Text(t), .. } if t == "Be helpful")
    );
    assert!(matches!(&annotated.messages[1], Message::User { .. }));

    assert_eq!(annotated.model, Some("claude-sonnet-4-20250514".into()));

    let params = annotated.params.unwrap();
    assert_eq!(params.max_tokens, Some(1024));

    // Tools should be normalized: input_schema -> parameters
    let tools = annotated.tools.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].tool_type, "function");
    assert_eq!(tools[0].function.name, "get_weather");
    assert_eq!(tools[0].function.description, Some("Get weather".into()));
    assert!(tools[0].function.parameters.is_some());

    assert_eq!(annotated.tool_choice, Some(ToolChoice::Auto));
}

#[test]
fn test_decode_request_system_array() {
    let codec = AnthropicMessagesCodec;
    let request = make_request(json!({
        "system": [
            { "type": "text", "text": "First instruction." },
            { "type": "text", "text": "Second instruction." }
        ],
        "messages": [
            { "role": "user", "content": "Hello" }
        ],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100
    }));
    let annotated = codec.decode(&request).unwrap();
    assert_eq!(annotated.messages.len(), 2);
    assert!(matches!(
        &annotated.messages[0],
        Message::System { content: MessageContent::Text(t), .. }
        if t == "First instruction.\nSecond instruction."
    ));
}

#[test]
fn test_decode_request_stop_sequences() {
    let codec = AnthropicMessagesCodec;
    let request = make_request(json!({
        "messages": [{ "role": "user", "content": "Hi" }],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "stop_sequences": ["\n\nHuman:", "END"]
    }));
    let annotated = codec.decode(&request).unwrap();
    let params = annotated.params.unwrap();
    assert_eq!(params.stop, Some(vec!["\n\nHuman:".into(), "END".into()]));
}

#[test]
fn test_decode_request_tool_choice_auto() {
    let codec = AnthropicMessagesCodec;
    let request = make_request(json!({
        "messages": [{ "role": "user", "content": "Hi" }],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "tool_choice": { "type": "auto" }
    }));
    let annotated = codec.decode(&request).unwrap();
    assert_eq!(annotated.tool_choice, Some(ToolChoice::Auto));
}

#[test]
fn test_decode_request_tool_choice_any() {
    let codec = AnthropicMessagesCodec;
    let request = make_request(json!({
        "messages": [{ "role": "user", "content": "Hi" }],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "tool_choice": { "type": "any" }
    }));
    let annotated = codec.decode(&request).unwrap();
    assert_eq!(annotated.tool_choice, Some(ToolChoice::Required));
}

#[test]
fn test_decode_request_tool_choice_specific() {
    let codec = AnthropicMessagesCodec;
    let request = make_request(json!({
        "messages": [{ "role": "user", "content": "Hi" }],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "tool_choice": { "type": "tool", "name": "get_weather" }
    }));
    let annotated = codec.decode(&request).unwrap();
    match annotated.tool_choice.unwrap() {
        ToolChoice::Specific(tc) => {
            assert_eq!(tc.function.name, "get_weather");
        }
        other => panic!("Expected Specific, got {other:?}"),
    }
}

#[test]
fn test_decode_request_extra_fields() {
    let codec = AnthropicMessagesCodec;
    let request = make_request(json!({
        "messages": [{ "role": "user", "content": "Hi" }],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "metadata": { "user_id": "abc" },
        "stream": true
    }));
    let annotated = codec.decode(&request).unwrap();
    assert_eq!(annotated.metadata, Some(json!({"user_id": "abc"})));
    assert_eq!(annotated.extra.get("stream"), Some(&json!(true)));
}

#[test]
fn test_decode_request_service_tier_and_parallel_tool_calls() {
    let codec = AnthropicMessagesCodec;
    let request = make_request(json!({
        "messages": [{ "role": "user", "content": "Hi" }],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "service_tier": "default",
        "tool_choice": { "type": "auto", "disable_parallel_tool_use": true }
    }));
    let annotated = codec.decode(&request).unwrap();
    assert_eq!(annotated.service_tier.as_deref(), Some("default"));
    assert_eq!(annotated.parallel_tool_calls, Some(false));
}

// ===================================================================
// Request encode tests
// ===================================================================

#[test]
fn test_encode_round_trip_preserves_unmodeled_fields() {
    let codec = AnthropicMessagesCodec;
    let original = make_request(json!({
        "system": "Be helpful",
        "messages": [{ "role": "user", "content": "Hello" }],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "metadata": { "user_id": "abc" },
        "stream": true
    }));
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();
    let obj = encoded.content.as_object().unwrap();
    // Unmodeled fields preserved
    assert_eq!(obj.get("metadata"), Some(&json!({"user_id": "abc"})));
    assert_eq!(obj.get("stream"), Some(&json!(true)));
}

#[test]
fn test_encode_writes_anthropic_modeled_controls() {
    let codec = AnthropicMessagesCodec;
    let mut annotated = codec
        .decode(&make_request(json!({
            "messages": [{ "role": "user", "content": "Hi" }],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "tool_choice": { "type": "auto" }
        })))
        .unwrap();
    annotated.metadata = Some(json!({"user_id":"abc"}));
    annotated.service_tier = Some("default".into());
    annotated.parallel_tool_calls = Some(false);
    let encoded = codec
        .encode(
            &annotated,
            &make_request(json!({
                "messages": [{ "role": "user", "content": "Hi" }],
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "tool_choice": { "type": "auto" }
            })),
        )
        .unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert_eq!(obj.get("metadata"), Some(&json!({"user_id":"abc"})));
    assert_eq!(obj.get("service_tier"), Some(&json!("default")));
    assert_eq!(
        obj.get("tool_choice")
            .and_then(|v| v.get("disable_parallel_tool_use")),
        Some(&json!(true))
    );
}

#[test]
fn test_encode_system_as_top_level() {
    let codec = AnthropicMessagesCodec;
    let original = make_request(json!({
        "system": "Original system",
        "messages": [{ "role": "user", "content": "Hello" }],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100
    }));
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();
    let obj = encoded.content.as_object().unwrap();
    // System should be a top-level field, not in messages
    assert_eq!(obj.get("system"), Some(&json!("Original system")));
    // Messages array should not contain a system role message
    let messages = obj.get("messages").unwrap().as_array().unwrap();
    for msg in messages {
        assert_ne!(msg.get("role").and_then(|r| r.as_str()), Some("system"));
    }
}

#[test]
fn test_encode_stop_sequences() {
    let codec = AnthropicMessagesCodec;
    let original = make_request(json!({
        "messages": [{ "role": "user", "content": "Hi" }],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "stop_sequences": ["\n\nHuman:"]
    }));
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();
    let obj = encoded.content.as_object().unwrap();
    // Should write stop_sequences (not stop)
    assert_eq!(obj.get("stop_sequences"), Some(&json!(["\n\nHuman:"])));
    assert!(!obj.contains_key("stop"));
}

#[test]
fn test_encode_tools_with_input_schema() {
    let codec = AnthropicMessagesCodec;
    let original = make_request(json!({
        "messages": [{ "role": "user", "content": "Hi" }],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "tools": [{
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": { "type": "object" }
        }]
    }));
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();
    let obj = encoded.content.as_object().unwrap();
    let tools = obj.get("tools").unwrap().as_array().unwrap();
    assert_eq!(tools.len(), 1);
    // Should write input_schema (not parameters), and no function wrapper
    assert!(tools[0].get("input_schema").is_some());
    assert!(!tools[0].as_object().unwrap().contains_key("parameters"));
    assert!(!tools[0].as_object().unwrap().contains_key("type"));
    assert!(!tools[0].as_object().unwrap().contains_key("function"));
}

#[test]
fn test_encode_tool_choice_anthropic_format() {
    let codec = AnthropicMessagesCodec;
    let original = make_request(json!({
        "messages": [{ "role": "user", "content": "Hi" }],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "tool_choice": { "type": "auto" }
    }));
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert_eq!(obj.get("tool_choice"), Some(&json!({"type": "auto"})));
}

#[test]
fn test_encode_max_tokens() {
    let codec = AnthropicMessagesCodec;
    let original = make_request(json!({
        "messages": [{ "role": "user", "content": "Hi" }],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 200
    }));
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();
    let obj = encoded.content.as_object().unwrap();
    // Should write max_tokens (not max_completion_tokens or max_output_tokens)
    assert_eq!(obj.get("max_tokens"), Some(&json!(200)));
}

#[test]
fn test_helper_and_error_paths_cover_remaining_anthropic_branches() {
    assert_eq!(json_f64(f64::NAN), Json::Null);
    assert_eq!(
        decode_anthropic_tool_choice(&json!({"type": "mystery"})),
        None
    );
    assert_eq!(decode_anthropic_tool_choice(&json!({"type": "tool"})), None);
    assert_eq!(
        extract_system_message(&json!([{ "type": "image", "source": "ignored" }])),
        None
    );

    let system_parts = Message::System {
        content: MessageContent::Parts(vec![
            super::super::request::ContentPart::Text {
                text: "First".into(),
            },
            super::super::request::ContentPart::Text {
                text: "Second".into(),
            },
        ]),
        name: None,
    };
    assert_eq!(
        extract_system_text(&system_parts),
        Some("First\nSecond".to_string())
    );
    assert_eq!(
        extract_system_text(&Message::User {
            content: MessageContent::Text("hi".into()),
            name: None,
        }),
        None
    );

    let codec = AnthropicMessagesCodec;

    match codec
        .decode(&make_request(json!("not-an-object")))
        .unwrap_err()
    {
        FlowError::Internal(message) => {
            assert!(message.contains("request content is not an object"))
        }
        other => panic!("unexpected decode error: {other}"),
    }

    let partial_usage = codec
        .decode_response(&json!({
            "content": [{ "type": "text", "text": "hi" }],
            "usage": { "input_tokens": 7 }
        }))
        .unwrap();
    assert_eq!(partial_usage.usage.unwrap().total_tokens, None);

    let annotated = AnnotatedLlmRequest {
        messages: vec![system_parts],
        model: Some("claude-sonnet-4-20250514".into()),
        params: Some(GenerationParams {
            temperature: Some(0.3),
            max_tokens: Some(128),
            top_p: Some(0.8),
            stop: Some(vec!["END".into()]),
        }),
        tools: Some(vec![ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "lookup".into(),
                description: None,
                parameters: None,
            },
        }]),
        tool_choice: Some(ToolChoice::None),
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
                "model": "claude-sonnet-4-20250514"
            })),
        )
        .unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert_eq!(obj.get("temperature"), Some(&json!(0.3)));
    assert_eq!(obj.get("top_p"), Some(&json!(0.8)));
    assert_eq!(obj.get("stop_sequences"), Some(&json!(["END"])));
    assert_eq!(obj.get("tool_choice"), Some(&json!({"type": "auto"})));
    assert_eq!(obj.get("system"), Some(&json!("First\nSecond")));

    let tools = obj.get("tools").unwrap().as_array().unwrap();
    assert_eq!(tools[0].get("name"), Some(&json!("lookup")));
    assert!(tools[0].get("description").is_none());
    assert!(tools[0].get("input_schema").is_none());

    match codec.encode(&annotated, &make_request(json!("still-not-an-object"))) {
        Err(FlowError::Internal(message)) => {
            assert!(message.contains("original content is not an object"));
        }
        other => panic!("unexpected encode result: {other:?}"),
    }
}

#[test]
fn test_encode_tool_choice_any_to_anthropic() {
    let codec = AnthropicMessagesCodec;
    let original = make_request(json!({
        "messages": [{ "role": "user", "content": "Hi" }],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "tool_choice": { "type": "any" }
    }));
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert_eq!(obj.get("tool_choice"), Some(&json!({"type": "any"})));
}

#[test]
fn test_encode_tool_choice_specific_to_anthropic() {
    let codec = AnthropicMessagesCodec;
    let original = make_request(json!({
        "messages": [{ "role": "user", "content": "Hi" }],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "tool_choice": { "type": "tool", "name": "my_func" }
    }));
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();
    let obj = encoded.content.as_object().unwrap();
    assert_eq!(
        obj.get("tool_choice"),
        Some(&json!({"type": "tool", "name": "my_func"}))
    );
}

// ===================================================================
// Streaming codec tests
// ===================================================================

use super::super::streaming::StreamingCodec;

#[test]
fn anthropic_streaming_codec_assembles_text_response() {
    let codec = AnthropicMessagesStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "type": "message_start",
        "message": {
            "id": "msg_01ABC",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [],
            "stop_reason": null,
            "stop_sequence": null,
            "usage": {"input_tokens": 100, "output_tokens": 0}
        }
    }))
    .unwrap();
    collector(json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""}
    }))
    .unwrap();
    collector(json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello, "}
    }))
    .unwrap();
    collector(json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "world."}
    }))
    .unwrap();
    collector(json!({"type": "content_block_stop", "index": 0})).unwrap();
    collector(json!({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": null},
        "usage": {"input_tokens": 100, "output_tokens": 5}
    }))
    .unwrap();
    collector(json!({"type": "message_stop"})).unwrap();

    let assembled = finalizer();
    // Wire-compatible with RawAnthropicResponse — feed it back through the existing decoder.
    let annotated = AnthropicMessagesCodec
        .decode_response(&assembled)
        .expect("assembled response should decode");
    assert_eq!(annotated.id.as_deref(), Some("msg_01ABC"));
    assert_eq!(
        annotated.model.as_deref(),
        Some("claude-haiku-4-5-20251001")
    );
    assert_eq!(annotated.finish_reason, Some(FinishReason::Complete));
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("Hello, world.".to_string()))
    );
    let usage = annotated.usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, Some(100));
    assert_eq!(usage.completion_tokens, Some(5));
}

#[test]
fn anthropic_streaming_codec_assembles_tool_use_input_from_partial_json() {
    let codec = AnthropicMessagesStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "type": "message_start",
        "message": {
            "id": "msg_tool",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [],
            "usage": {"input_tokens": 50, "output_tokens": 0}
        }
    }))
    .unwrap();
    collector(json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "lookup",
            "input": {}
        }
    }))
    .unwrap();
    for fragment in &["{\"q", "uery\":", " \"weath", "er\"}"] {
        collector(json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": fragment}
        }))
        .unwrap();
    }
    collector(json!({"type": "content_block_stop", "index": 0})).unwrap();
    collector(json!({
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use"},
        "usage": {"input_tokens": 50, "output_tokens": 12}
    }))
    .unwrap();

    let assembled = finalizer();
    let annotated = AnthropicMessagesCodec
        .decode_response(&assembled)
        .expect("assembled response should decode");
    assert_eq!(annotated.finish_reason, Some(FinishReason::ToolUse));
    let tool_calls = annotated.tool_calls.expect("tool_calls present");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "toolu_01");
    assert_eq!(tool_calls[0].name, "lookup");
    assert_eq!(tool_calls[0].arguments, json!({"query": "weather"}));
}

#[test]
fn anthropic_streaming_codec_preserves_unknown_block_types() {
    // Server-side tool blocks (web_search_tool_result) ship full content at content_block_start
    // and have no deltas; the codec must preserve them in the assembled content array.
    let codec = AnthropicMessagesStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "type": "message_start",
        "message": {
            "id": "msg_ws",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "usage": {"input_tokens": 1, "output_tokens": 0}
        }
    }))
    .unwrap();
    collector(json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_42",
            "content": [
                {"type": "web_search_result", "title": "First", "url": "https://a"},
                {"type": "web_search_result", "title": "Second", "url": "https://b"}
            ]
        }
    }))
    .unwrap();
    collector(json!({"type": "content_block_stop", "index": 0})).unwrap();

    let assembled = finalizer();
    let block = &assembled["content"][0];
    assert_eq!(block["type"], json!("web_search_tool_result"));
    assert_eq!(block["tool_use_id"], json!("srvtoolu_42"));
    assert_eq!(block["content"][0]["title"], json!("First"));
    assert_eq!(block["content"][1]["url"], json!("https://b"));
}

#[test]
fn anthropic_streaming_codec_attaches_citations_to_text_blocks() {
    let codec = AnthropicMessagesStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "type": "message_start",
        "message": {
            "id": "msg_c", "type": "message", "role": "assistant",
            "model": "claude-haiku-4-5-20251001", "usage": {}
        }
    }))
    .unwrap();
    collector(json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""}
    }))
    .unwrap();
    collector(json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "citations_delta", "citation": {
            "type": "web_search_result_location",
            "cited_text": "Hello",
            "url": "https://example.com",
            "title": "Source"
        }}
    }))
    .unwrap();
    collector(json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello"}
    }))
    .unwrap();

    let assembled = finalizer();
    let block = &assembled["content"][0];
    assert_eq!(block["text"], json!("Hello"));
    let citations = block["citations"].as_array().expect("citations array");
    assert_eq!(citations.len(), 1);
    assert_eq!(citations[0]["url"], json!("https://example.com"));
}

#[test]
fn anthropic_streaming_codec_keeps_partial_json_when_unparseable() {
    // Truncated stream: input_json_delta fragments don't form valid JSON. Codec must not drop
    // the tool_use block; surface the raw concatenation as a string fallback so observability
    // captures partial intent.
    let codec = AnthropicMessagesStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "type": "message_start",
        "message": {
            "id": "msg_p", "type": "message", "role": "assistant",
            "model": "claude-haiku-4-5-20251001", "usage": {}
        }
    }))
    .unwrap();
    collector(json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "toolu_p", "name": "go", "input": {}}
    }))
    .unwrap();
    collector(json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": "{\"q\": \"trun"}
    }))
    .unwrap();

    let assembled = finalizer();
    let block = &assembled["content"][0];
    assert_eq!(block["type"], json!("tool_use"));
    assert_eq!(block["id"], json!("toolu_p"));
    assert_eq!(block["input"], json!("{\"q\": \"trun"));
}
