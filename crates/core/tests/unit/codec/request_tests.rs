// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for request in the NeMo Relay core crate.

use super::*;
use serde_json::json;

// -------------------------------------------------------------------
// AnnotatedLlmRequest serialization round-trip
// -------------------------------------------------------------------

#[test]
fn test_annotated_llm_request_round_trip() {
    let req = AnnotatedLlmRequest {
        messages: vec![Message::User {
            content: MessageContent::Text("Hello".into()),
            name: None,
        }],
        model: Some("gpt-4".into()),
        params: Some(GenerationParams {
            temperature: Some(0.7),
            max_tokens: Some(100),
            top_p: None,
            stop: None,
        }),
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
    let deserialized: AnnotatedLlmRequest = serde_json::from_value(json_val).unwrap();
    assert_eq!(req, deserialized);
}

// -------------------------------------------------------------------
// Message role serialization
// -------------------------------------------------------------------

#[test]
fn test_message_system_serialization() {
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
fn test_message_user_serialization() {
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
fn test_message_tool_serialization() {
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

// -------------------------------------------------------------------
// MessageContent serialization
// -------------------------------------------------------------------

#[test]
fn test_message_content_text_serialization() {
    let content = MessageContent::Text("hello".into());
    let json_val = serde_json::to_value(&content).unwrap();
    assert_eq!(json_val, json!("hello"));
}

#[test]
fn test_message_content_parts_serialization() {
    let content = MessageContent::Parts(vec![ContentPart::Text {
        text: "Hello world".into(),
    }]);
    let json_val = serde_json::to_value(&content).unwrap();
    assert_eq!(json_val, json!([{"type": "text", "text": "Hello world"}]));
}

// -------------------------------------------------------------------
// ToolCall serialization
// -------------------------------------------------------------------

#[test]
fn test_tool_call_serialization() {
    let tc = ToolCall {
        id: "tc_1".into(),
        call_type: "function".into(),
        function: FunctionCall {
            name: "search".into(),
            arguments: r#"{"q":"test"}"#.into(),
        },
    };
    let json_val = serde_json::to_value(&tc).unwrap();
    assert_eq!(
        json_val,
        json!({
            "id": "tc_1",
            "type": "function",
            "function": {"name": "search", "arguments": "{\"q\":\"test\"}"}
        })
    );
}

// -------------------------------------------------------------------
// ToolDefinition serialization
// -------------------------------------------------------------------

#[test]
fn test_tool_definition_serialization() {
    let td = ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "get_weather".into(),
            description: Some("Get current weather".into()),
            parameters: Some(json!({"type": "object", "properties": {"city": {"type": "string"}}})),
        },
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
}

// -------------------------------------------------------------------
// ToolChoice serialization
// -------------------------------------------------------------------

#[test]
fn test_tool_choice_auto_serialization() {
    let tc = ToolChoice::Auto;
    let json_val = serde_json::to_value(&tc).unwrap();
    assert_eq!(json_val, json!("auto"));
}

#[test]
fn test_tool_choice_none_serialization() {
    let tc = ToolChoice::None;
    let json_val = serde_json::to_value(&tc).unwrap();
    assert_eq!(json_val, json!("none"));
}

#[test]
fn test_tool_choice_required_serialization() {
    let tc = ToolChoice::Required;
    let json_val = serde_json::to_value(&tc).unwrap();
    assert_eq!(json_val, json!("required"));
}

#[test]
fn test_tool_choice_specific_serialization() {
    let tc = ToolChoice::Specific(ToolChoiceFunction {
        choice_type: "function".into(),
        function: ToolChoiceFunctionName {
            name: "my_func".into(),
        },
    });
    let json_val = serde_json::to_value(&tc).unwrap();
    assert_eq!(
        json_val,
        json!({"type": "function", "function": {"name": "my_func"}})
    );
}

// -------------------------------------------------------------------
// GenerationParams serialization
// -------------------------------------------------------------------

#[test]
fn test_generation_params_all_none_serializes_to_empty() {
    let params = GenerationParams::default();
    let json_val = serde_json::to_value(&params).unwrap();
    assert_eq!(json_val, json!({}));
}

// -------------------------------------------------------------------
// Extra / flatten field
// -------------------------------------------------------------------

#[test]
fn test_annotated_llm_request_extra_flatten() {
    let json_val = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "custom_field": "value"
    });
    let req: AnnotatedLlmRequest = serde_json::from_value(json_val).unwrap();
    assert_eq!(req.stream, Some(true));
    assert_eq!(req.extra.get("custom_field"), Some(&json!("value")));
    // Round-trip: extra fields should appear as top-level keys
    let serialized = serde_json::to_value(&req).unwrap();
    assert_eq!(serialized["stream"], json!(true));
    assert_eq!(serialized["custom_field"], json!("value"));
}

// -------------------------------------------------------------------
// Clone trait
// -------------------------------------------------------------------

#[test]
fn test_all_types_clone() {
    let req = AnnotatedLlmRequest {
        messages: vec![
            Message::System {
                content: MessageContent::Text("system".into()),
                name: None,
            },
            Message::User {
                content: MessageContent::Parts(vec![ContentPart::Text {
                    text: "user part".into(),
                }]),
                name: Some("alice".into()),
            },
        ],
        model: Some("gpt-4".into()),
        params: Some(GenerationParams {
            temperature: Some(0.5),
            max_tokens: None,
            top_p: Some(0.9),
            stop: Some(vec!["END".into()]),
        }),
        tools: Some(vec![ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "test".into(),
                description: None,
                parameters: None,
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
    let cloned = req.clone();
    assert_eq!(req, cloned);
}

// -------------------------------------------------------------------
// PartialEq trait
// -------------------------------------------------------------------

#[test]
fn test_all_types_partial_eq() {
    let msg1 = Message::User {
        content: MessageContent::Text("hello".into()),
        name: None,
    };
    let msg2 = Message::User {
        content: MessageContent::Text("hello".into()),
        name: None,
    };
    let msg3 = Message::User {
        content: MessageContent::Text("world".into()),
        name: None,
    };
    assert_eq!(msg1, msg2);
    assert_ne!(msg1, msg3);

    let tc1 = ToolChoice::Auto;
    let tc2 = ToolChoice::Auto;
    let tc3 = ToolChoice::None;
    assert_eq!(tc1, tc2);
    assert_ne!(tc1, tc3);
}

// -------------------------------------------------------------------
// Helper method: system_prompt()
// -------------------------------------------------------------------

#[test]
fn test_system_prompt_returns_text() {
    let req = AnnotatedLlmRequest {
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
fn test_system_prompt_returns_none_when_absent() {
    let req = AnnotatedLlmRequest {
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
fn test_system_prompt_from_parts() {
    let req = AnnotatedLlmRequest {
        messages: vec![Message::System {
            content: MessageContent::Parts(vec![ContentPart::Text {
                text: "Be concise".into(),
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
    assert_eq!(req.system_prompt(), Some("Be concise"));
}

// -------------------------------------------------------------------
// Helper method: last_user_message()
// -------------------------------------------------------------------

#[test]
fn test_last_user_message_returns_last() {
    let req = AnnotatedLlmRequest {
        messages: vec![
            Message::User {
                content: MessageContent::Text("first".into()),
                name: None,
            },
            Message::Assistant {
                content: None,
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
fn test_last_user_message_returns_none_when_absent() {
    let req = AnnotatedLlmRequest {
        messages: vec![Message::System {
            content: MessageContent::Text("system".into()),
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
    assert_eq!(req.last_user_message(), None);
}

#[test]
fn test_last_user_message_from_parts() {
    let req = AnnotatedLlmRequest {
        messages: vec![
            Message::Assistant {
                content: None,
                tool_calls: None,
                name: None,
            },
            Message::User {
                content: MessageContent::Parts(vec![ContentPart::Text {
                    text: "from parts".into(),
                }]),
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
    assert_eq!(req.last_user_message(), Some("from parts"));
}

// -------------------------------------------------------------------
// Helper method: has_tool_calls()
// -------------------------------------------------------------------

#[test]
fn test_has_tool_calls_true() {
    let req = AnnotatedLlmRequest {
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
fn test_has_tool_calls_false_empty_vec() {
    let req = AnnotatedLlmRequest {
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
