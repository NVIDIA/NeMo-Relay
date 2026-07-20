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
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: Some(format!("describe {name}")),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            })),
        },
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
        messages: vec![
            Message::System {
                content: MessageContent::Text("You are helpful.".to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "Hello".to_string(),
                    },
                    ContentPart::Text {
                        text: "World".to_string(),
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
fn build_prompt_ir_canonicalizes_structured_output_before_user_content() {
    let mut request = AnnotatedLlmRequest {
        messages: vec![Message::User {
            content: MessageContent::Text("variable task".to_string()),
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
    request.extra.insert(
        "response_format".to_string(),
        serde_json::json!({"type":"json_schema","json_schema":{"required":["answer"],"type":"object"}}),
    );

    let first = build_prompt_ir(&request).unwrap();
    request.extra.insert(
        "response_format".to_string(),
        serde_json::json!({"json_schema":{"type":"object","required":["answer"]},"type":"json_schema"}),
    );
    let reordered = build_prompt_ir(&request).unwrap();

    assert_eq!(
        first.structured_output_schema_id,
        reordered.structured_output_schema_id
    );
    assert!(first.structured_output_schema_id.is_some());
    assert_eq!(
        first.blocks[0].content_type,
        BlockContentType::StructuredOutput
    );
    assert_eq!(first.blocks[1].role, PromptRole::User);

    request
        .extra
        .insert("response_format".to_string(), serde_json::Value::Null);
    let null_contract = build_prompt_ir(&request).unwrap();
    assert!(null_contract.structured_output_schema_id.is_none());
    assert!(
        null_contract
            .blocks
            .iter()
            .all(|block| block.content_type != BlockContentType::StructuredOutput)
    );
}
