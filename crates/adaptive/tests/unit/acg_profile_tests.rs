// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for acg profile in the NeMo Relay adaptive crate.

use nemo_relay::codec::request::{
    AnnotatedLlmRequest, ContentPart, FunctionDefinition, Message, MessageContent, OpenAiImageUrl,
    ToolDefinition,
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

fn scaffolded_request(task: &str) -> AnnotatedLlmRequest {
    request(
        vec![
            Message::System {
                content: MessageContent::Text("Follow policy".to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text(task.to_string()),
                name: None,
            },
        ],
        Some(vec![sample_tool("search")]),
    )
}

#[test]
fn acg_learning_key_groups_variable_tasks_by_stable_scaffold() {
    let first = scaffolded_request("find alpha");
    let second = scaffolded_request("find beta");

    assert_eq!(
        derive_acg_learning_key("agent-a", &first),
        derive_acg_learning_key("agent-a", &second)
    );
}

#[test]
fn acg_learning_key_separates_scaffold_and_output_contract_changes() {
    let baseline = scaffolded_request("find alpha");

    let mut changed_system = baseline.clone();
    changed_system.messages[0] = Message::System {
        content: MessageContent::Text("Follow a different policy".to_string()),
        name: None,
    };

    let mut changed_tool = baseline.clone();
    changed_tool.tools = Some(vec![sample_tool("lookup")]);

    let mut contract_a = baseline.clone();
    contract_a.extra.insert(
        "response_format".to_string(),
        json!({"type":"json_schema","json_schema":{"type":"object","properties":{"a":{"type":"string"}}}}),
    );
    let mut contract_b = baseline.clone();
    contract_b.extra.insert(
        "response_format".to_string(),
        json!({"json_schema":{"properties":{"b":{"type":"string"}},"type":"object"},"type":"json_schema"}),
    );

    let baseline_key = derive_acg_learning_key("agent-a", &baseline);
    assert_ne!(
        baseline_key,
        derive_acg_learning_key("agent-a", &changed_system)
    );
    assert_ne!(
        baseline_key,
        derive_acg_learning_key("agent-a", &changed_tool)
    );
    assert_ne!(
        derive_acg_learning_key("agent-a", &contract_a),
        derive_acg_learning_key("agent-a", &contract_b)
    );

    let mut null_contract = baseline.clone();
    null_contract
        .extra
        .insert("response_format".to_string(), serde_json::Value::Null);
    assert_eq!(
        baseline_key,
        derive_acg_learning_key("agent-a", &null_contract)
    );
}

#[test]
fn acg_learning_key_keeps_task_seed_without_a_stable_scaffold() {
    let first = request(
        vec![Message::User {
            content: MessageContent::Text("alpha".to_string()),
            name: None,
        }],
        None,
    );
    let second = request(
        vec![Message::User {
            content: MessageContent::Text("beta".to_string()),
            name: None,
        }],
        None,
    );

    assert_ne!(
        derive_acg_learning_key("agent-a", &first),
        derive_acg_learning_key("agent-a", &second)
    );
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

#[test]
fn acg_profile_image_parts_contribute_stable_fingerprint_signal() {
    let with_image_a = request(
        vec![Message::User {
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: OpenAiImageUrl {
                    url: "https://example.com/a.png".to_string(),
                    detail: Some("high".to_string()),
                },
            }]),
            name: None,
        }],
        None,
    );
    let with_image_b = request(
        vec![Message::User {
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: OpenAiImageUrl {
                    url: "https://example.com/b.png".to_string(),
                    detail: Some("high".to_string()),
                },
            }]),
            name: None,
        }],
        None,
    );

    assert_ne!(
        learning_seed_fingerprint(&with_image_a),
        learning_seed_fingerprint(&with_image_b)
    );
}

#[test]
fn acg_profile_fingerprints_cover_alternate_role_sequences() {
    let late_user = request(
        vec![
            Message::System {
                content: MessageContent::Text("system".into()),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text("assistant".into())),
                tool_calls: None,
                name: None,
            },
            Message::User {
                content: MessageContent::Text("user".into()),
                name: None,
            },
            Message::System {
                content: MessageContent::Text("tail".into()),
                name: None,
            },
        ],
        None,
    );
    assert!(layered_anchor_fingerprint(&late_user).is_none());

    let wrong_assistant = request(
        vec![
            Message::User {
                content: MessageContent::Text("first".into()),
                name: None,
            },
            Message::System {
                content: MessageContent::Text("not assistant".into()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text("second".into()),
                name: None,
            },
            Message::Tool {
                content: MessageContent::Text("tool".into()),
                tool_call_id: "call".into(),
            },
        ],
        None,
    );
    assert!(layered_anchor_fingerprint(&wrong_assistant).is_none());

    let wrong_followup = request(
        vec![
            Message::User {
                content: MessageContent::Text("first".into()),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text("answer".into())),
                tool_calls: None,
                name: None,
            },
            Message::Tool {
                content: MessageContent::Text("tool".into()),
                tool_call_id: "call".into(),
            },
            Message::System {
                content: MessageContent::Text("tail".into()),
                name: None,
            },
        ],
        None,
    );
    assert!(layered_anchor_fingerprint(&wrong_followup).is_none());

    for (message, prefix) in [
        (
            Message::Assistant {
                content: Some(MessageContent::Text("answer".into())),
                tool_calls: None,
                name: None,
            },
            "assistant:",
        ),
        (
            Message::Assistant {
                content: None,
                tool_calls: None,
                name: None,
            },
            "assistant:no-content",
        ),
        (
            Message::Tool {
                content: MessageContent::Text("result".into()),
                tool_call_id: "call".into(),
            },
            "tool:",
        ),
    ] {
        assert!(
            learning_seed_fingerprint(&request(vec![message], None)).starts_with(prefix),
            "expected seed prefix {prefix}"
        );
    }

    let system_only = request(
        vec![Message::System {
            content: MessageContent::Text("system".into()),
            name: None,
        }],
        None,
    );
    assert_eq!(learning_seed_fingerprint(&system_only), "no-seed");
}
