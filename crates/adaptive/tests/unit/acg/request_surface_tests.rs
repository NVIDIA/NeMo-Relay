// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for request surface in the NeMo Relay adaptive crate.

use chrono::Utc;
use nemo_relay::api::llm::LlmRequest;
use serde_json::json;
use uuid::Uuid;

use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel, SpanId,
};
use crate::acg::request_surfaces::RequestSurface;
use crate::acg::request_surfaces::RequestSurfaceApplier;
use crate::acg::request_surfaces::anthropic_messages::AnthropicMessages;
use crate::acg::request_surfaces::openai_chat::OpenAIChat;
use crate::acg::request_surfaces::openai_responses::OpenAIResponses;
use crate::acg::translation::{
    AnthropicCacheTtl, AnthropicHintDirective, HintPlan, HintTarget, OpenAIHintDirective,
};
use crate::acg::types::SharingScope;

fn prompt_ir_for_request_surface() -> PromptIR {
    PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: vec![
            PromptBlock {
                span_id: SpanId("system-0".to_string()),
                sequence_index: 0,
                role: PromptRole::System,
                content: "You are helpful.".to_string(),
                content_type: BlockContentType::Text,
                provenance: ProvenanceLabel::System,
                sensitivity: SensitivityLabel::Public,
                token_metadata: None,
            },
            PromptBlock {
                span_id: SpanId("tool-1".to_string()),
                sequence_index: 1,
                role: PromptRole::System,
                content: "{\"function\":{\"name\":\"search\"}}".to_string(),
                content_type: BlockContentType::ToolSchema,
                provenance: ProvenanceLabel::System,
                sensitivity: SensitivityLabel::Public,
                token_metadata: None,
            },
            PromptBlock {
                span_id: SpanId("user-2".to_string()),
                sequence_index: 2,
                role: PromptRole::User,
                content: "{\"type\":\"tool_result\",\"data\":{\"z\":1,\"a\":2}}".to_string(),
                content_type: BlockContentType::StructuredOutput,
                provenance: ProvenanceLabel::User,
                sensitivity: SensitivityLabel::Public,
                token_metadata: None,
            },
            PromptBlock {
                span_id: SpanId("user-3".to_string()),
                sequence_index: 3,
                role: PromptRole::User,
                content: "{\"type\":\"tool_result\",\"data\":{\"y\":1,\"b\":2}}".to_string(),
                content_type: BlockContentType::StructuredOutput,
                provenance: ProvenanceLabel::User,
                sensitivity: SensitivityLabel::Public,
                token_metadata: None,
            },
        ],
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    }
}

#[test]
fn test_request_surface_anthropic_messages_maps_semantic_targets_to_system_messages_and_tools() {
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "claude-sonnet-4",
            "system": "You are helpful.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"}
                    ]
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "parameters": {
                            "z": 1,
                            "a": 2
                        },
                        "name": "search"
                    }
                }
            ]
        }),
    };
    let prompt_ir = prompt_ir_for_request_surface();
    let mut plan = HintPlan::new("anthropic");
    plan.push(AnthropicHintDirective::CanonicalizeToolSchemas);
    plan.push(AnthropicHintDirective::CacheBreakpoint {
        target: HintTarget::span(SpanId("system-0".to_string())),
        scope: SharingScope::Session,
    });
    plan.push(AnthropicHintDirective::CacheBreakpoint {
        target: HintTarget::span(SpanId("tool-1".to_string())),
        scope: SharingScope::Session,
    });
    plan.push(AnthropicHintDirective::CacheBreakpoint {
        target: HintTarget::span(SpanId("user-2".to_string())),
        scope: SharingScope::Session,
    });
    plan.push(AnthropicHintDirective::ApplyTtl {
        ttl: AnthropicCacheTtl::OneHour,
    });

    let translated = AnthropicMessages
        .apply(&request, &prompt_ir, &plan)
        .unwrap();

    assert!(translated.content["system"].is_array());
    assert_eq!(
        translated.content["system"][0]["cache_control"]["ttl"],
        json!("1h")
    );
    assert_eq!(
        translated.content["messages"][0]["content"][1]["cache_control"]["ttl"],
        json!("1h")
    );
    assert_eq!(
        translated.content["tools"][0]["cache_control"]["ttl"],
        json!("1h")
    );
}

#[test]
fn test_request_surface_openai_chat_applies_stable_prefix_to_messages_and_tools() {
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "tool_result", "data": {"z": 1, "a": 2}}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Later"},
                        {"type": "tool_result", "data": {"y": 1, "b": 2}}
                    ]
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "parameters": {
                            "z": 1,
                            "a": 2
                        },
                        "name": "search"
                    }
                }
            ]
        }),
    };
    let prompt_ir = prompt_ir_for_request_surface();
    let mut plan = HintPlan::new("openai");
    plan.push(OpenAIHintDirective::CanonicalizeToolSchemas);
    let target = HintTarget::stable_prefix(3, Some(SpanId("user-2".to_string())));
    plan.push(OpenAIHintDirective::CanonicalizeStablePrefix {
        target: target.clone(),
    });

    let translated = OpenAIChat.apply(&request, &prompt_ir, &plan).unwrap();
    let indices = super::target_block_indices(&prompt_ir, &target);

    assert!(translated.content.get("messages").is_some());
    assert!(translated.content.get("tools").is_some());
    assert_eq!(indices, vec![0, 1, 2]);
    assert_eq!(super::prompt_ir_tool_index(&prompt_ir, 1), 0);
    assert_eq!(super::prompt_ir_message_index(&prompt_ir, 2, true), 1);
    assert_eq!(super::prompt_ir_message_index(&prompt_ir, 3, true), 2);
}

#[test]
fn test_request_surface_openai_responses_applies_instructions_input_and_tools() {
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4.1",
            "instructions": "You are helpful.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "tool_result", "data": {"z": 1, "a": 2}}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Later"},
                        {"type": "tool_result", "data": {"y": 1, "b": 2}}
                    ]
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "parameters": {
                            "z": 1,
                            "a": 2
                        },
                        "name": "search"
                    }
                }
            ]
        }),
    };
    let prompt_ir = prompt_ir_for_request_surface();
    let mut plan = HintPlan::new("openai");
    plan.push(OpenAIHintDirective::CanonicalizeToolSchemas);
    let target = HintTarget::stable_prefix(3, Some(SpanId("user-2".to_string())));
    plan.push(OpenAIHintDirective::CanonicalizeStablePrefix {
        target: target.clone(),
    });

    let translated = OpenAIResponses.apply(&request, &prompt_ir, &plan).unwrap();
    let indices = super::target_block_indices(&prompt_ir, &target);

    assert_eq!(
        translated.content["instructions"],
        json!("You are helpful.")
    );
    assert!(translated.content.get("input").is_some());
    assert!(translated.content.get("tools").is_some());
    assert_eq!(indices, vec![0, 1, 2]);
    assert_eq!(super::prompt_ir_tool_index(&prompt_ir, 1), 0);
    assert_eq!(super::prompt_ir_message_index(&prompt_ir, 2, false), 0);
    assert_eq!(super::prompt_ir_message_index(&prompt_ir, 3, false), 1);
}

#[test]
fn test_request_surface_resolution_and_passthrough_support_cover_matrix() {
    let chat_request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }),
    };
    let responses_request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4.1",
            "instructions": "helpful",
            "input": []
        }),
    };
    let anthropic_request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "claude-sonnet-4",
            "system": "helpful",
            "messages": []
        }),
    };
    let invalid_request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"model": "unknown"}),
    };

    assert!(RequestSurface::OpenAIChat.supports_provider("passthrough"));
    assert!(!RequestSurface::OpenAIChat.supports_provider("unknown"));
    assert_eq!(
        super::resolve_request_surface_from_request(&chat_request, None).unwrap(),
        RequestSurface::OpenAIChat
    );
    assert_eq!(
        super::resolve_request_surface_from_request(&responses_request, None).unwrap(),
        RequestSurface::OpenAIResponses
    );
    assert_eq!(
        super::resolve_request_surface_from_request(&anthropic_request, None).unwrap(),
        RequestSurface::AnthropicMessages
    );
    assert!(matches!(
        super::resolve_request_surface_from_request(&invalid_request, None),
        Err(crate::acg::AcgError::Internal(message))
            if message.contains("unable to resolve request surface")
    ));
    // `resolve_request_surface` is the plugin-facade path, not the live translate path.
    assert_eq!(
        super::resolve_request_surface("anthropic", &chat_request).unwrap(),
        RequestSurface::AnthropicMessages
    );
    assert!(matches!(
        super::resolve_request_surface("anthropic", &responses_request),
        Err(crate::acg::AcgError::Internal(message))
            if message.contains("does not support resolved request surface")
    ));
}

#[test]
fn test_request_surface_helpers_cover_apply_and_target_resolution_edges() {
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "tool_result", "data": {"z": 1, "a": 2}}
                    ]
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "parameters": {"z": 1, "a": 2},
                        "name": "search"
                    }
                }
            ]
        }),
    };
    let prompt_ir = prompt_ir_for_request_surface();
    let mut plan = HintPlan::new("openai");
    plan.push(OpenAIHintDirective::CanonicalizeToolSchemas);
    plan.push(OpenAIHintDirective::CanonicalizeStablePrefix {
        target: HintTarget::Span {
            span_id: SpanId("user-2".to_string()),
        },
    });

    let translated = super::apply_request_surface("openai", &request, &prompt_ir, &plan).unwrap();
    assert_eq!(
        translated.content["tools"][0]["function"]["parameters"],
        json!({"a": 2, "z": 1}),
    );
    assert_eq!(
        super::target_block_indices(
            &prompt_ir,
            &HintTarget::Span {
                span_id: SpanId("user-2".to_string()),
            },
        ),
        vec![2]
    );
}

#[test]
fn test_request_surface_canonicalization_helpers_handle_missing_fields_and_duplicates() {
    let prompt_ir = prompt_ir_for_request_surface();
    let target = HintTarget::stable_prefix(4, Some(SpanId("user-3".to_string())));
    let mut message_without_content = json!({"role": "user"});
    let mut scalar_content = json!("plain-text");
    let mut request_without_messages = json!({"tools": []});
    let mut request_with_duplicate_target = json!({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "data": {"z": 1, "a": 2}},
                    {"type": "tool_result", "data": {"y": 1, "b": 2}}
                ]
            }
        ]
    });

    assert!(!super::canonicalize_message_content_blocks(
        &mut message_without_content
    ));
    assert!(!super::canonicalize_content_blocks(&mut scalar_content));

    super::canonicalize_target_messages(
        &mut request_without_messages,
        &prompt_ir,
        &target,
        false,
        "messages",
    );
    super::canonicalize_target_messages(
        &mut request_with_duplicate_target,
        &prompt_ir,
        &target,
        false,
        "messages",
    );

    assert_eq!(
        request_with_duplicate_target["messages"][0]["content"][0],
        json!({"data": {"a": 2, "z": 1}, "type": "tool_result"})
    );
}
