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
        instructions: None,
        api_specific: None,
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
    ToolDefinition::Function {
        function: FunctionDefinition {
            name: name.to_string(),
            description: Some("desc".to_string()),
            parameters: Some(json!({"type":"object","properties":{"a":{"type":"string"}}})),
            strict: None,
            extra: serde_json::Map::new(),
        },
        extra: serde_json::Map::new(),
    }
}

#[test]
fn acg_profile_derivation_covers_anchor_hash_system_fallback_and_empty_tools() {
    let layered = request(
        vec![
            Message::System {
                content: MessageContent::Parts(vec![ContentPart::Text {
                    text: "System guide".to_string(),
                    extra: serde_json::Map::new(),
                }]),
                name: None,
            },
            Message::User {
                content: MessageContent::Parts(vec![ContentPart::Text {
                    text: "Language guide".to_string(),
                    extra: serde_json::Map::new(),
                }]),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Parts(vec![ContentPart::Text {
                    text: "Acknowledged".to_string(),
                    extra: serde_json::Map::new(),
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
fn system_fingerprint_preserves_sources_roles_and_content_boundaries() {
    let mut instructions_and_system = request(
        vec![Message::System {
            content: MessageContent::Text("b".into()),
            name: None,
        }],
        None,
    );
    instructions_and_system.instructions = Some(MessageContent::Text("a".into()));
    let joined_system = request(
        vec![Message::System {
            content: MessageContent::Text("a\nb".into()),
            name: None,
        }],
        None,
    );
    let developer = request(
        vec![Message::Developer {
            content: MessageContent::Text("a\nb".into()),
            name: None,
        }],
        None,
    );
    let split_parts = request(
        vec![Message::System {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "a".into(),
                    extra: serde_json::Map::new(),
                },
                ContentPart::Text {
                    text: "b".into(),
                    extra: serde_json::Map::new(),
                },
            ]),
            name: None,
        }],
        None,
    );

    assert_ne!(
        system_prompt_fingerprint(&instructions_and_system),
        system_prompt_fingerprint(&joined_system)
    );
    assert_ne!(
        system_prompt_fingerprint(&joined_system),
        system_prompt_fingerprint(&developer)
    );
    assert_ne!(
        system_prompt_fingerprint(&joined_system),
        system_prompt_fingerprint(&split_parts)
    );
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
                extra: serde_json::Map::new(),
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
                extra: serde_json::Map::new(),
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

#[test]
fn acg_profile_covers_extended_roles_and_native_content() {
    let all_roles = request(
        vec![
            Message::Developer {
                content: MessageContent::Text("developer".into()),
                name: None,
            },
            Message::Function {
                content: Some("legacy result".into()),
                name: "legacy".into(),
            },
            Message::ToolCallItem {
                id: None,
                call_id: "call_1".into(),
                name: "lookup".into(),
                arguments: json!({"q": "x"}),
                extra: serde_json::Map::new(),
            },
            Message::ToolResultItem {
                id: None,
                call_id: "call_1".into(),
                output: json!({"ok": true}),
                extra: serde_json::Map::new(),
            },
            Message::ProviderNative {
                provider: "openai_responses".into(),
                kind: "reasoning".into(),
                value: json!({"type": "reasoning"}),
            },
        ],
        None,
    );
    let key = derive_acg_profile_key("agent-extended", &all_roles);
    assert!(key.contains("roles=developer.function.tool_call.tool_result.provider_native"));

    for (message, prefix) in [
        (
            Message::Function {
                content: None,
                name: "legacy".into(),
            },
            "function:",
        ),
        (
            Message::ToolCallItem {
                id: None,
                call_id: "call_1".into(),
                name: "lookup".into(),
                arguments: json!({"q": "x"}),
                extra: serde_json::Map::new(),
            },
            "tool-call:",
        ),
        (
            Message::ToolResultItem {
                id: None,
                call_id: "call_1".into(),
                output: json!({"ok": true}),
                extra: serde_json::Map::new(),
            },
            "tool-result:",
        ),
        (
            Message::ProviderNative {
                provider: "openai_responses".into(),
                kind: "reasoning".into(),
                value: json!({"type": "reasoning"}),
            },
            "native:",
        ),
    ] {
        assert!(learning_seed_fingerprint(&request(vec![message], None)).starts_with(prefix));
    }

    let seed_for_part = |part| {
        learning_seed_fingerprint(&request(
            vec![Message::User {
                content: MessageContent::Parts(vec![part]),
                name: None,
            }],
            None,
        ))
    };
    assert_ne!(
        seed_for_part(ContentPart::Refusal {
            refusal: "no-a".into(),
            extra: serde_json::Map::new(),
        }),
        seed_for_part(ContentPart::Refusal {
            refusal: "no-b".into(),
            extra: serde_json::Map::new(),
        })
    );
    assert_ne!(
        seed_for_part(ContentPart::Audio {
            audio: json!({"data": "audio-a"}),
            extra: serde_json::Map::new(),
        }),
        seed_for_part(ContentPart::Audio {
            audio: json!({"data": "audio-b"}),
            extra: serde_json::Map::new(),
        })
    );
    assert_ne!(
        seed_for_part(ContentPart::ProviderNative {
            provider: "openai_responses".into(),
            kind: "future".into(),
            value: json!({"type": "future", "value": "a"}),
        }),
        seed_for_part(ContentPart::ProviderNative {
            provider: "openai_responses".into(),
            kind: "future".into(),
            value: json!({"type": "future", "value": "b"}),
        })
    );
}

#[test]
fn learning_seed_fingerprint_includes_variant_discriminator_fields() {
    let seed_for_message = |message| learning_seed_fingerprint(&request(vec![message], None));
    assert_ne!(
        seed_for_message(Message::Function {
            content: Some("same".into()),
            name: "one".into(),
        }),
        seed_for_message(Message::Function {
            content: Some("same".into()),
            name: "two".into(),
        })
    );
    assert_ne!(
        seed_for_message(Message::ToolCallItem {
            id: None,
            call_id: "call".into(),
            name: "one".into(),
            arguments: json!({"same": true}),
            extra: serde_json::Map::new(),
        }),
        seed_for_message(Message::ToolCallItem {
            id: None,
            call_id: "call".into(),
            name: "two".into(),
            arguments: json!({"same": true}),
            extra: serde_json::Map::new(),
        })
    );
    assert_ne!(
        seed_for_message(Message::ProviderNative {
            provider: "openai_responses".into(),
            kind: "reasoning".into(),
            value: json!({"same": true}),
        }),
        seed_for_message(Message::ProviderNative {
            provider: "anthropic_messages".into(),
            kind: "thinking".into(),
            value: json!({"same": true}),
        })
    );
}
