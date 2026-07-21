// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for ir builder in the NeMo Relay adaptive crate.

use nemo_relay::codec::request::{
    AnnotatedLlmRequest, ContentPart, FunctionCall, FunctionDefinition, Message, MessageContent,
    ToolCall, ToolDefinition,
};

use super::super::ir_builder::build_prompt_ir;
use crate::acg::prompt_ir::{BlockContentType, PromptRole, ProvenanceLabel};

fn sample_tool_definition(name: &str) -> ToolDefinition {
    ToolDefinition::Function {
        function: FunctionDefinition {
            name: name.to_string(),
            description: Some(format!("describe {name}")),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            })),
            strict: None,
            extra: serde_json::Map::new(),
        },
        extra: serde_json::Map::new(),
    }
}

fn sample_tool_call(name: &str) -> ToolCall {
    ToolCall {
        id: format!("call-{name}"),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: "{\"query\":\"weather\"}".to_string(),
        },
    }
}

#[test]
fn build_prompt_ir_inserts_tools_before_first_non_system_message_and_preserves_all_message_kinds() {
    let request = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![
            Message::System {
                content: MessageContent::Text("You are helpful.".to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "Hello".to_string(),
                        extra: serde_json::Map::new(),
                    },
                    ContentPart::Text {
                        text: "World".to_string(),
                        extra: serde_json::Map::new(),
                    },
                ]),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text("Calling search".to_string())),
                tool_calls: Some(vec![sample_tool_call("search")]),
                name: None,
            },
            Message::Tool {
                content: MessageContent::Text("{\"result\":true}".to_string()),
                tool_call_id: "call-search".to_string(),
            },
        ],
        model: Some("gpt-4o".to_string()),
        params: None,
        tools: Some(vec![sample_tool_definition("search")]),
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

    let prompt_ir = build_prompt_ir(&request).unwrap();

    assert_eq!(prompt_ir.blocks.len(), 6);
    assert_eq!(prompt_ir.blocks[0].role, PromptRole::System);
    assert_eq!(prompt_ir.blocks[0].provenance, ProvenanceLabel::System);
    assert_eq!(
        prompt_ir.blocks[1].content_type,
        BlockContentType::ToolSchema
    );
    assert_eq!(prompt_ir.blocks[2].role, PromptRole::User);
    assert_eq!(prompt_ir.blocks[2].content, "Hello\nWorld");
    assert_eq!(prompt_ir.blocks[3].role, PromptRole::Assistant);
    assert_eq!(prompt_ir.blocks[4].role, PromptRole::Assistant);
    assert_eq!(
        prompt_ir.blocks[5].content_type,
        BlockContentType::ToolResult
    );
    assert_eq!(prompt_ir.blocks[5].role, PromptRole::Tool);
    assert!(prompt_ir.tool_schema_hashes.is_some());
    assert!(prompt_ir.source_request_hash.is_some());
}

#[test]
fn build_prompt_ir_appends_tool_blocks_when_request_contains_only_system_messages() {
    let request = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![Message::System {
            content: MessageContent::Text("System only".to_string()),
            name: None,
        }],
        model: Some("gpt-4o".to_string()),
        params: None,
        tools: Some(vec![
            sample_tool_definition("search"),
            sample_tool_definition("lookup"),
        ]),
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

    let prompt_ir = build_prompt_ir(&request).unwrap();

    assert_eq!(prompt_ir.blocks.len(), 3);
    assert_eq!(prompt_ir.blocks[0].content_type, BlockContentType::Text);
    assert_eq!(
        prompt_ir.blocks[1].content_type,
        BlockContentType::ToolSchema
    );
    assert_eq!(
        prompt_ir.blocks[2].content_type,
        BlockContentType::ToolSchema
    );
    assert_eq!(prompt_ir.blocks[2].sequence_index, 2);
}

#[test]
fn build_prompt_ir_omits_tool_schema_hashes_when_no_tools_are_present() {
    let request = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![Message::User {
            content: MessageContent::Text("No tools".to_string()),
            name: None,
        }],
        model: Some("gpt-4o".to_string()),
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

    let prompt_ir = build_prompt_ir(&request).unwrap();

    assert_eq!(prompt_ir.blocks.len(), 1);
    assert!(prompt_ir.tool_schema_hashes.is_none());
    assert_eq!(prompt_ir.blocks[0].span_id.0, "user-0");
}

#[test]
fn build_prompt_ir_covers_extended_request_messages_and_content_parts() {
    let request = AnnotatedLlmRequest {
        instructions: Some(MessageContent::Text("Top-level instructions".into())),
        messages: vec![
            Message::Developer {
                content: MessageContent::Text("Developer guide".into()),
                name: Some("developer".into()),
            },
            Message::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Refusal {
                        refusal: "refusal".into(),
                        extra: serde_json::Map::new(),
                    },
                    ContentPart::Image {
                        image: serde_json::json!({"file_id": "image_1"}),
                        extra: serde_json::Map::new(),
                    },
                    ContentPart::Audio {
                        audio: serde_json::json!({"data": "audio"}),
                        extra: serde_json::Map::new(),
                    },
                    ContentPart::File {
                        file: serde_json::json!({"file_id": "file_1"}),
                        extra: serde_json::Map::new(),
                    },
                    ContentPart::ToolUse {
                        id: "call_1".into(),
                        name: "lookup".into(),
                        input: serde_json::json!({"q": "x"}),
                        extra: serde_json::Map::new(),
                    },
                    ContentPart::ToolResult {
                        tool_use_id: "call_1".into(),
                        content: serde_json::json!("ok"),
                        is_error: Some(false),
                        extra: serde_json::Map::new(),
                    },
                    ContentPart::ProviderNative {
                        provider: "openai_responses".into(),
                        kind: "future".into(),
                        value: serde_json::json!({"type": "future"}),
                    },
                ]),
                name: None,
            },
            Message::Function {
                content: None,
                name: "legacy".into(),
            },
            Message::ToolCallItem {
                id: Some("fc_1".into()),
                call_id: "call_1".into(),
                name: "lookup".into(),
                arguments: serde_json::json!({"q": "x"}),
                extra: serde_json::Map::new(),
            },
            Message::ToolResultItem {
                id: Some("fco_1".into()),
                call_id: "call_1".into(),
                output: serde_json::json!({"ok": true}),
                extra: serde_json::Map::new(),
            },
            Message::ProviderNative {
                provider: "openai_responses".into(),
                kind: "reasoning".into(),
                value: serde_json::json!({"type": "reasoning", "summary": []}),
            },
        ],
        tools: Some(vec![ToolDefinition::ProviderNative {
            provider: "openai_responses".into(),
            kind: "web_search_preview".into(),
            value: serde_json::json!({"type": "web_search_preview"}),
        }]),
        model: Some("gpt-5".into()),
        ..AnnotatedLlmRequest::default()
    };

    let prompt_ir = build_prompt_ir(&request).unwrap();
    assert_eq!(prompt_ir.blocks[0].span_id.0, "system-0-instructions");
    assert_eq!(prompt_ir.blocks[1].span_id.0, "system-1-developer");
    assert_eq!(prompt_ir.blocks[2].span_id.0, "system-2-web_search_preview");
    assert!(
        prompt_ir
            .blocks
            .iter()
            .any(|block| block.content_type == BlockContentType::ToolSchema)
    );
    assert!(
        prompt_ir
            .blocks
            .iter()
            .any(|block| block.span_id.0.ends_with("reasoning"))
    );
    assert_eq!(
        prompt_ir.tool_schema_hashes.as_ref().unwrap()[0].tool_name,
        "web_search_preview"
    );
}
