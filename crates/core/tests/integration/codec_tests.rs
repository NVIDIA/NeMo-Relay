// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the Codec type system, LlmCodec trait, helper methods,
//! Codec registry, and 4-layer resolution precedence.

use serde_json::json;

use nemo_relay::api::llm::LlmRequest;
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::request::{
    ContentPart, FunctionCall, FunctionDefinition, GenerationParams, Message, MessageContent,
    ToolCall, ToolChoice, ToolChoiceFunction, ToolChoiceFunctionName, ToolDefinition,
};
use nemo_relay::codec::traits::LlmCodec;
use nemo_relay::error::Result;

// ---------------------------------------------------------------------------
// Mock Codec for registry and resolution tests
// ---------------------------------------------------------------------------

struct MockCodec {
    id: String,
}

impl LlmCodec for MockCodec {
    fn decode(&self, _request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        Ok(AnnotatedLlmRequest {
            instructions: None,
            api_specific: None,
            messages: vec![],
            model: Some(self.id.clone()),
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
        })
    }

    fn encode(
        &self,
        _annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> Result<LlmRequest> {
        Ok(original.clone())
    }
}

fn dummy_llm_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({}),
    }
}

// ===========================================================================
// Section 1: Type serialization tests
// ===========================================================================

#[test]
fn test_annotated_llm_request_full_roundtrip() {
    let req = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![
            Message::System {
                content: MessageContent::Text("Be helpful".into()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text("Hello".into()),
                name: Some("alice".into()),
            },
            Message::Assistant {
                content: Some(MessageContent::Text("Hi there".into())),
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "search".into(),
                        arguments: r#"{"q":"rust"}"#.into(),
                    },
                }]),
                name: None,
            },
            Message::Tool {
                content: MessageContent::Text("result".into()),
                tool_call_id: "call_1".into(),
            },
        ],
        model: Some("gpt-4".into()),
        params: Some(GenerationParams {
            temperature: Some(0.7),
            max_tokens: Some(1024),
            top_p: Some(0.9),
            stop: Some(vec!["END".into()]),
        }),
        tools: Some(vec![ToolDefinition::Function {
            function: FunctionDefinition {
                name: "search".into(),
                description: Some("Search the web".into()),
                parameters: Some(
                    json!({"type": "object", "properties": {"q": {"type": "string"}}}),
                ),
                strict: None,
                extra: serde_json::Map::new(),
            },
            extra: serde_json::Map::new(),
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
        extra: {
            let mut m = serde_json::Map::new();
            m.insert("response_format".into(), json!({"type": "json_object"}));
            m
        },
    };

    let json_val = serde_json::to_value(&req).unwrap();
    let deserialized: AnnotatedLlmRequest = serde_json::from_value(json_val).unwrap();
    assert_eq!(req, deserialized);
}

#[test]
fn test_annotated_llm_request_minimal() {
    let req = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![Message::User {
            content: MessageContent::Text("Hi".into()),
            name: None,
        }],
        model: None,
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

    let json_val = serde_json::to_value(&req).unwrap();

    // Optional fields should be absent (skip_serializing_if)
    assert!(json_val.get("model").is_none());
    assert!(json_val.get("params").is_none());
    assert!(json_val.get("tools").is_none());
    assert!(json_val.get("tool_choice").is_none());

    // Round-trip
    let deserialized: AnnotatedLlmRequest = serde_json::from_value(json_val).unwrap();
    assert_eq!(req, deserialized);
}

#[test]
fn test_message_system_roundtrip() {
    let msg = Message::System {
        content: MessageContent::Text("Be helpful".into()),
        name: None,
    };
    let json_val = serde_json::to_value(&msg).unwrap();
    assert_eq!(json_val, json!({"role": "system", "content": "Be helpful"}));
    let deserialized: Message = serde_json::from_value(json_val).unwrap();
    assert_eq!(msg, deserialized);
}

#[test]
fn test_message_user_roundtrip() {
    let msg = Message::User {
        content: MessageContent::Text("Hello".into()),
        name: None,
    };
    let json_val = serde_json::to_value(&msg).unwrap();
    assert_eq!(json_val, json!({"role": "user", "content": "Hello"}));
    let deserialized: Message = serde_json::from_value(json_val).unwrap();
    assert_eq!(msg, deserialized);
}

#[test]
fn test_message_assistant_with_tool_calls() {
    let msg = Message::Assistant {
        content: None,
        tool_calls: Some(vec![ToolCall {
            id: "call_123".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "get_weather".into(),
                arguments: r#"{"city":"NYC"}"#.into(),
            },
        }]),
        name: None,
    };
    let json_val = serde_json::to_value(&msg).unwrap();
    assert_eq!(
        json_val,
        json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"city\":\"NYC\"}"
                }
            }]
        })
    );
    let deserialized: Message = serde_json::from_value(json_val).unwrap();
    assert_eq!(msg, deserialized);
}

#[test]
fn test_message_assistant_content_only() {
    let msg = Message::Assistant {
        content: Some(MessageContent::Text("response".into())),
        tool_calls: None,
        name: None,
    };
    let json_val = serde_json::to_value(&msg).unwrap();
    assert_eq!(
        json_val,
        json!({"role": "assistant", "content": "response"})
    );
    // tool_calls should be absent
    assert!(json_val.get("tool_calls").is_none());
    let deserialized: Message = serde_json::from_value(json_val).unwrap();
    assert_eq!(msg, deserialized);
}

#[test]
fn test_message_tool_roundtrip() {
    let msg = Message::Tool {
        content: MessageContent::Text("72F, sunny".into()),
        tool_call_id: "call_123".into(),
    };
    let json_val = serde_json::to_value(&msg).unwrap();
    assert_eq!(
        json_val,
        json!({"role": "tool", "content": "72F, sunny", "tool_call_id": "call_123"})
    );
    let deserialized: Message = serde_json::from_value(json_val).unwrap();
    assert_eq!(msg, deserialized);
}

#[test]
fn test_message_content_text_serializes_as_string() {
    let content = MessageContent::Text("hello".into());
    let json_val = serde_json::to_value(&content).unwrap();
    assert_eq!(json_val, json!("hello"));
}

#[test]
fn test_message_content_parts_serializes_as_array() {
    let content = MessageContent::Parts(vec![ContentPart::Text {
        text: "hi".into(),
        extra: serde_json::Map::new(),
    }]);
    let json_val = serde_json::to_value(&content).unwrap();
    assert_eq!(json_val, json!([{"type": "text", "text": "hi"}]));
}

#[test]
fn test_tool_call_roundtrip() {
    let tc = ToolCall {
        id: "call_1".into(),
        call_type: "function".into(),
        function: FunctionCall {
            name: "search".into(),
            arguments: r#"{"q":"rust"}"#.into(),
        },
    };
    let json_val = serde_json::to_value(&tc).unwrap();
    // call_type should serialize as "type" (not "call_type")
    assert_eq!(
        json_val,
        json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": "{\"q\":\"rust\"}"}
        })
    );
    let deserialized: ToolCall = serde_json::from_value(json_val).unwrap();
    assert_eq!(tc, deserialized);
}

#[test]
fn test_tool_definition_roundtrip() {
    let td = ToolDefinition::Function {
        function: FunctionDefinition {
            name: "get_weather".into(),
            description: Some("Get current weather".into()),
            parameters: Some(json!({"type": "object", "properties": {"city": {"type": "string"}}})),
            strict: None,
            extra: serde_json::Map::new(),
        },
        extra: serde_json::Map::new(),
    };
    let json_val = serde_json::to_value(&td).unwrap();
    assert_eq!(
        json_val,
        json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        })
    );
    let deserialized: ToolDefinition = serde_json::from_value(json_val).unwrap();
    assert_eq!(td, deserialized);
}

#[test]
fn test_tool_choice_auto_serializes_as_string() {
    let tc = ToolChoice::Auto;
    let json_val = serde_json::to_value(&tc).unwrap();
    assert_eq!(json_val, json!("auto"));
}

#[test]
fn test_tool_choice_none_serializes_as_string() {
    let tc = ToolChoice::None;
    let json_val = serde_json::to_value(&tc).unwrap();
    assert_eq!(json_val, json!("none"));
}

#[test]
fn test_tool_choice_required_serializes_as_string() {
    let tc = ToolChoice::Required;
    let json_val = serde_json::to_value(&tc).unwrap();
    assert_eq!(json_val, json!("required"));
}

#[test]
fn test_tool_choice_specific_serializes_as_object() {
    let tc = ToolChoice::Specific(ToolChoiceFunction {
        choice_type: "function".into(),
        function: ToolChoiceFunctionName {
            name: "search".into(),
        },
    });
    let json_val = serde_json::to_value(&tc).unwrap();
    assert_eq!(
        json_val,
        json!({"type": "function", "function": {"name": "search"}})
    );
}

#[test]
fn test_generation_params_empty() {
    let params = GenerationParams::default();
    let json_val = serde_json::to_value(&params).unwrap();
    assert_eq!(json_val, json!({}));
}

#[test]
fn test_generation_params_partial() {
    let params = GenerationParams {
        temperature: Some(0.7),
        max_tokens: None,
        top_p: None,
        stop: None,
    };
    let json_val = serde_json::to_value(&params).unwrap();
    assert_eq!(json_val, json!({"temperature": 0.7}));
    // Other fields should not be present
    assert!(json_val.get("max_tokens").is_none());
    assert!(json_val.get("top_p").is_none());
    assert!(json_val.get("stop").is_none());
}

#[test]
fn test_extra_field_flatten() {
    let mut extra = serde_json::Map::new();
    extra.insert("response_format".into(), json!({"type": "json_object"}));

    let req = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![Message::User {
            content: MessageContent::Text("hi".into()),
            name: None,
        }],
        model: None,
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
        extra,
    };

    let json_val = serde_json::to_value(&req).unwrap();

    // response_format should appear as a top-level key (not nested under "extra")
    assert_eq!(json_val["response_format"], json!({"type": "json_object"}));
    assert!(json_val.get("extra").is_none());

    // Round-trip: deserialize and verify extra captures it
    let deserialized: AnnotatedLlmRequest = serde_json::from_value(json_val).unwrap();
    assert_eq!(
        deserialized.extra.get("response_format"),
        Some(&json!({"type": "json_object"}))
    );
}

#[test]
fn test_clone_and_partial_eq() {
    let a = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![Message::User {
            content: MessageContent::Text("hello".into()),
            name: None,
        }],
        model: Some("gpt-4".into()),
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

    let b = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![Message::User {
            content: MessageContent::Text("hello".into()),
            name: None,
        }],
        model: Some("gpt-4".into()),
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

    assert_eq!(a, b);

    let mut c = a.clone();
    assert_eq!(a, c);

    c.model = Some("claude".into());
    assert_ne!(a, c);
}

// ===========================================================================
// Section 2: Helper method tests
// ===========================================================================

#[test]
fn test_system_prompt_text() {
    let req = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![
            Message::System {
                content: MessageContent::Text("Be helpful".into()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text("Hi".into()),
                name: None,
            },
        ],
        model: None,
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
    assert_eq!(req.system_prompt(), Some("Be helpful"));
}

#[test]
fn test_system_prompt_none() {
    let req = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![Message::User {
            content: MessageContent::Text("Hi".into()),
            name: None,
        }],
        model: None,
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
    assert_eq!(req.system_prompt(), None);
}

#[test]
fn test_system_prompt_parts() {
    let req = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![Message::System {
            content: MessageContent::Parts(vec![ContentPart::Text {
                text: "Be careful".into(),
                extra: serde_json::Map::new(),
            }]),
            name: None,
        }],
        model: None,
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
    assert_eq!(req.system_prompt(), Some("Be careful"));
}

#[test]
fn test_last_user_message_basic() {
    let req = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![
            Message::User {
                content: MessageContent::Text("first".into()),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text("ok".into())),
                tool_calls: None,
                name: None,
            },
            Message::User {
                content: MessageContent::Text("last".into()),
                name: None,
            },
        ],
        model: None,
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
    assert_eq!(req.last_user_message(), Some("last"));
}

#[test]
fn test_last_user_message_none() {
    let req = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![
            Message::System {
                content: MessageContent::Text("hi".into()),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text("hello".into())),
                tool_calls: None,
                name: None,
            },
        ],
        model: None,
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
    assert_eq!(req.last_user_message(), None);
}

#[test]
fn test_has_tool_calls_true() {
    let req = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![Message::Assistant {
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "tc_1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "search".into(),
                    arguments: "{}".into(),
                },
            }]),
            name: None,
        }],
        model: None,
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
    assert!(req.has_tool_calls());
}

#[test]
fn test_has_tool_calls_false_no_assistant() {
    let req = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![Message::User {
            content: MessageContent::Text("hi".into()),
            name: None,
        }],
        model: None,
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
    assert!(!req.has_tool_calls());
}

#[test]
fn test_has_tool_calls_false_empty_calls() {
    let req = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![Message::Assistant {
            content: Some(MessageContent::Text("hello".into())),
            tool_calls: Some(vec![]),
            name: None,
        }],
        model: None,
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
    assert!(!req.has_tool_calls());
}

// ===========================================================================
// Section 3: Codec trait tests
// ===========================================================================

#[test]
fn test_mock_codec_decode_encode() {
    let codec = MockCodec {
        id: "test_codec".into(),
    };
    let req = dummy_llm_request();

    // decode should return an AnnotatedLlmRequest with model == id
    let annotated = codec.decode(&req).unwrap();
    assert_eq!(annotated.model, Some("test_codec".into()));
    assert!(annotated.messages.is_empty());

    // encode should return the original request unchanged
    let encoded = codec.encode(&annotated, &req).unwrap();
    assert_eq!(encoded.content, req.content);
}
