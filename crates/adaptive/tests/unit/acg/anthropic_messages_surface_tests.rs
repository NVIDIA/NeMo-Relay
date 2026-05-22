// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for anthropic messages surface in the NeMo Relay adaptive crate.

use serde_json::json;

use super::*;
use chrono::Utc;
use nemo_relay::api::llm::LlmRequest;
use uuid::Uuid;

use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel, SpanId,
};
use crate::acg::translation::{
    AnthropicCacheTtl, AnthropicHintDirective, HintPlan, HintTarget, OpenAIHintDirective,
};
use crate::acg::types::SharingScope;

fn sample_prompt_ir() -> PromptIR {
    PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: vec![PromptBlock {
            span_id: SpanId("system-0".to_string()),
            sequence_index: 0,
            role: PromptRole::System,
            content: "hello".to_string(),
            content_type: BlockContentType::Text,
            provenance: ProvenanceLabel::System,
            sensitivity: SensitivityLabel::Public,
            token_metadata: None,
        }],
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    }
}

#[test]
fn anthropic_messages_helpers_cover_false_and_object_paths() {
    let mut no_system = json!({});
    assert!(!inject_cache_control_system(&mut no_system));

    let mut array_system = json!({
        "system": [{"type": "text", "text": "hello"}]
    });
    assert!(inject_cache_control_system(&mut array_system));
    assert!(array_system["system"][0]["cache_control"].is_object());

    let mut no_messages = json!({});
    assert!(!inject_cache_control_message(&mut no_messages, 0));

    let mut string_message = json!({
        "messages": [{"role": "user", "content": "hello"}]
    });
    assert!(inject_cache_control_message(&mut string_message, 0));
    assert!(string_message["messages"][0]["content"][0]["cache_control"].is_object());

    let mut array_message = json!({
        "messages": [{"role": "user", "content": [{"type":"text","text":"hello"}]}]
    });
    assert!(inject_cache_control_message(&mut array_message, 0));
    assert!(array_message["messages"][0]["content"][0]["cache_control"].is_object());

    let mut object_message = json!({
        "messages": [{"role": "user"}]
    });
    assert!(inject_cache_control_message(&mut object_message, 0));
    assert!(object_message["messages"][0]["cache_control"].is_object());

    let mut no_tools = json!({});
    assert!(!inject_cache_control_tool(&mut no_tools, 0));
}

#[test]
fn anthropic_messages_ttl_helper_handles_scalars_arrays_and_objects() {
    let mut value = json!({
        "cache_control": {"type": "ephemeral"},
        "children": [
            {"cache_control": {"type": "ephemeral"}},
            "text"
        ]
    });

    apply_ttl_to_value(&mut value, "1h");
    assert_eq!(value["cache_control"]["ttl"], "1h");
    assert_eq!(value["children"][0]["cache_control"]["ttl"], "1h");
}

#[test]
fn anthropic_messages_apply_ignores_unresolved_breakpoints_and_openai_directives() {
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "claude-sonnet-4",
            "system": "hello",
            "messages": [{"role": "user", "content": "world"}],
            "tools": [{"name": "search"}]
        }),
    };
    let mut plan = HintPlan::new("anthropic");
    plan.push(AnthropicHintDirective::CacheBreakpoint {
        target: HintTarget::span(SpanId("missing-span".to_string())),
        scope: SharingScope::Session,
    });
    plan.push(OpenAIHintDirective::CanonicalizeToolSchemas);

    let translated = AnthropicMessages
        .apply(&request, &sample_prompt_ir(), &plan)
        .unwrap();

    assert_eq!(translated.content, request.content);
}

#[test]
fn anthropic_messages_ttl_application_walks_system_messages_and_tools() {
    let mut content = json!({
        "system": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}],
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "world", "cache_control": {"type": "ephemeral"}}],
            "cache_control": {"type": "ephemeral"}
        }],
        "tools": [{
            "name": "search",
            "cache_control": {"type": "ephemeral"}
        }]
    });

    apply_ttl_to_cache_controls(&mut content, "1h");

    assert_eq!(content["system"][0]["cache_control"]["ttl"], "1h");
    assert_eq!(content["messages"][0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        content["messages"][0]["content"][0]["cache_control"]["ttl"],
        "1h"
    );
    assert_eq!(content["tools"][0]["cache_control"]["ttl"], "1h");
}

#[test]
fn anthropic_messages_apply_ttl_directive_updates_existing_cache_controls() {
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "claude-sonnet-4",
            "system": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}],
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "world", "cache_control": {"type": "ephemeral"}}]
            }],
            "tools": [{
                "name": "search",
                "cache_control": {"type": "ephemeral"}
            }]
        }),
    };
    let mut plan = HintPlan::new("anthropic");
    plan.push(AnthropicHintDirective::ApplyTtl {
        ttl: AnthropicCacheTtl::OneHour,
    });

    let translated = AnthropicMessages
        .apply(&request, &sample_prompt_ir(), &plan)
        .unwrap();

    assert_eq!(
        translated.content["system"][0]["cache_control"]["ttl"],
        "1h"
    );
    assert_eq!(
        translated.content["messages"][0]["content"][0]["cache_control"]["ttl"],
        "1h"
    );
    assert_eq!(translated.content["tools"][0]["cache_control"]["ttl"], "1h");
}
