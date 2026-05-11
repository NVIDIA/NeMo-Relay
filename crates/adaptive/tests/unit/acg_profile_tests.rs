// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for acg profile in the NeMo Flow adaptive crate.

use nemo_flow::codec::request::{
    AnnotatedLlmRequest, ContentPart, FunctionDefinition, Message, MessageContent, ToolDefinition,
};
use serde_json::json;

use super::*;

fn request(messages: Vec<Message>, tools: Option<Vec<ToolDefinition>>) -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
        messages,
        model: Some("gpt-4o".to_string()),
        params: None,
        tools,
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
    }
}

fn sample_tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: Some("desc".to_string()),
            parameters: Some(json!({"type":"object","properties":{"a":{"type":"string"}}})),
        },
    }
}

#[test]
fn acg_profile_derivation_covers_anchor_hash_system_fallback_and_empty_tools() {
    let layered = request(
        vec![
            Message::System {
                content: MessageContent::Parts(vec![ContentPart::Text {
                    text: "System guide".to_string(),
                }]),
                name: None,
            },
            Message::User {
                content: MessageContent::Parts(vec![ContentPart::Text {
                    text: "Language guide".to_string(),
                }]),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Parts(vec![ContentPart::Text {
                    text: "Acknowledged".to_string(),
                }])),
                tool_calls: None,
                name: None,
            },
            Message::User {
                content: MessageContent::Text("Prompt body".to_string()),
                name: None,
            },
            Message::Tool {
                content: MessageContent::Text("tool result".to_string()),
                tool_call_id: "call-1".to_string(),
            },
        ],
        Some(vec![sample_tool("search")]),
    );

    let key = derive_acg_profile_key("agent-a", &layered);
    assert!(key.contains("roles=system.user.assistant.user.tool"));
    assert!(!key.contains("anchor=no-anchor"));

    let no_system = request(
        vec![Message::User {
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        Some(vec![]),
    );
    let no_system_key = derive_acg_profile_key("agent-b", &no_system);
    assert!(no_system_key.contains("system=no-system"));
    assert!(no_system_key.contains("anchor=no-anchor"));
    assert!(no_system_key.contains("tools=tools-unavailab"));
}

#[test]
fn acg_profile_helpers_cover_none_paths_and_short_hash() {
    let too_short = request(
        vec![
            Message::User {
                content: MessageContent::Text("u".to_string()),
                name: None,
            },
            Message::Assistant {
                content: None,
                tool_calls: None,
                name: None,
            },
            Message::User {
                content: MessageContent::Text("v".to_string()),
                name: None,
            },
        ],
        None,
    );
    assert!(layered_anchor_fingerprint(&too_short).is_none());
    assert_eq!(system_prompt_fingerprint(&too_short), "no-system");
    assert_eq!(tool_schema_fingerprint(None), "no-tools");
    assert_eq!(short_hash("short"), "short");
    assert_eq!(message_role_tag(&too_short.messages[0]), "user");
}
