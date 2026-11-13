// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for ir builder in the NeMo Relay adaptive crate.

use nemo_relay_types::codec::request::{
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
fn prompt_ir_rendering_preserves_mixed_content_boundaries() {
    let content: MessageContent = serde_json::from_value(serde_json::json!([
        {"type": "text", "text": "first"},
        {"type": "refusal", "refusal": ""},
        {"type": "image_url", "image_url": {"url": "image-a"}},
        {"type": "tool_use", "id": "c", "name": "lookup", "input": {"q": "x"}}
    ]))
    .unwrap();
    let ir = build_prompt_ir(&AnnotatedLlmRequest {
        messages: vec![Message::User {
            content,
            name: None,
        }],
        ..Default::default()
    })
    .unwrap();
    assert_eq!(ir.blocks.len(), 1);
    let content: serde_json::Value = serde_json::from_str(&ir.blocks[0].content).unwrap();
    assert_eq!(content.as_array().unwrap().len(), 4);
    assert_eq!(
        content[0],
        serde_json::json!({"type": "text", "text": "first"})
    );
    assert_eq!(
        content[1],
        serde_json::json!({"type": "refusal", "refusal": ""})
    );
    assert_eq!(content[2]["type"], "image_url");
    assert!(
        content[2]["fingerprint"]
            .as_str()
            .unwrap()
            .starts_with("sha256:")
    );
    assert_eq!(content[3]["input"], serde_json::json!({"q": "x"}));
    assert_eq!(ir.blocks[0].span_id.0, "user-0");
}

#[test]
fn adaptive_content_image_changes_stop_false_stable_prefixes() {
    let request = |url: &str| AnnotatedLlmRequest {
        messages: vec![
            Message::System {
                content: MessageContent::Text("stable system".into()),
                name: None,
            },
            Message::User {
                content: serde_json::from_value(serde_json::json!([
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": url}}
                ]))
                .unwrap(),
                name: None,
            },
        ],
        ..Default::default()
    };
    let first = build_prompt_ir(&request("private-image-a")).unwrap();
    let second = build_prompt_ir(&request("private-image-b")).unwrap();
    assert_eq!(first.blocks.len(), second.blocks.len());
    assert_eq!(first.blocks[1].span_id, second.blocks[1].span_id);
    assert_ne!(first.blocks[1].content, second.blocks[1].content);
    assert!(!first.blocks[1].content.contains("private-image-a"));
    assert_eq!(
        serde_json::to_value(first.blocks[1].content_type).unwrap(),
        "opaque"
    );
    let stability = crate::acg::stability::analyze_stability(
        &[first, second],
        &crate::acg::stability::StabilityThresholds::default(),
    );
    assert_eq!(stability.stable_prefix_length, 1);
}

#[test]
fn adaptive_content_structured_values_preserve_whitespace_and_key_order_independence() {
    let request = |arguments: serde_json::Value| AnnotatedLlmRequest {
        messages: vec![Message::ToolCallItem {
            id: None,
            call_id: "c".into(),
            name: "lookup".into(),
            arguments,
            extra: Default::default(),
        }],
        ..Default::default()
    };
    let first = build_prompt_ir(&request(
        serde_json::from_str(r#"{"city":"New  York","days":2}"#).unwrap(),
    ))
    .unwrap();
    let different =
        build_prompt_ir(&request(serde_json::json!({"city": "New York", "days": 2}))).unwrap();
    let reordered = build_prompt_ir(&request(
        serde_json::from_str(r#"{"days":2,"city":"New  York"}"#).unwrap(),
    ))
    .unwrap();
    assert_ne!(first.blocks[0].content, different.blocks[0].content);
    assert_eq!(first.blocks[0].content, reordered.blocks[0].content);
    let value: serde_json::Value = serde_json::from_str(&first.blocks[0].content).unwrap();
    assert_eq!(value["arguments"]["city"], "New  York");
    assert_eq!(
        serde_json::to_value(first.blocks[0].content_type).unwrap(),
        "structured_input"
    );
}

#[test]
fn adaptive_content_stability_distinguishes_typed_structure_from_literal_prose() {
    let mut request = AnnotatedLlmRequest {
        messages: vec![Message::User {
            content: serde_json::from_value(serde_json::json!([
                {"type": "tool_use", "id": "c", "name": "lookup", "input": {"q": "same"}}
            ]))
            .unwrap(),
            name: None,
        }],
        ..Default::default()
    };
    let structured = build_prompt_ir(&request).unwrap();
    request.messages = vec![Message::User {
        content: MessageContent::Text(structured.blocks[0].content.clone()),
        name: None,
    }];
    let literal = build_prompt_ir(&request).unwrap();
    assert_eq!(structured.blocks[0].content, literal.blocks[0].content);
    let stability = crate::acg::stability::analyze_stability(
        &[structured, literal],
        &crate::acg::stability::StabilityThresholds::default(),
    );
    assert_eq!(stability.stable_prefix_length, 0);
    assert_eq!(stability.scores[0].score, 0.5);
}

#[test]
fn tool_result_stability_distinguishes_typed_structure_from_literal_text() {
    for message in [
        serde_json::json!({"role": "tool", "tool_call_id": "c", "content": [{"type": "refusal", "refusal": "no"}]}),
        serde_json::json!({"role": "tool", "tool_call_id": "c", "content": [{"type": "tool_use", "id": "nested", "name": "lookup", "input": {"q": "same"}}]}),
        serde_json::json!({"role": "tool", "tool_call_id": "c", "content": [{"type": "tool_result", "tool_use_id": "nested", "content": "same"}]}),
        serde_json::json!({"role": "tool_result", "call_id": "c", "output": {"result": "same"}}),
    ] {
        let request = |message| AnnotatedLlmRequest {
            messages: vec![message],
            ..Default::default()
        };
        let structured =
            build_prompt_ir(&request(serde_json::from_value(message).unwrap())).unwrap();
        let literal = build_prompt_ir(&request(Message::Tool {
            tool_call_id: "c".into(),
            content: MessageContent::Text(structured.blocks[0].content.clone()),
        }))
        .unwrap();
        assert_eq!(structured.blocks[0].content, literal.blocks[0].content);
        assert_eq!(structured.blocks[0].span_id, literal.blocks[0].span_id);
        assert_eq!(structured.blocks[0].role, PromptRole::Tool);
        assert_eq!(structured.blocks[0].provenance, ProvenanceLabel::Tool);
        assert_ne!(
            crate::acg::stability::block_key(&structured.blocks[0]),
            crate::acg::stability::block_key(&literal.blocks[0]),
        );
        assert_eq!(
            structured.blocks[0].content_type,
            BlockContentType::StructuredInput
        );
        assert_eq!(literal.blocks[0].content_type, BlockContentType::ToolResult);
        let stability = crate::acg::stability::analyze_stability(
            &[structured, literal],
            &crate::acg::stability::StabilityThresholds::default(),
        );
        assert_eq!(stability.stable_prefix_length, 0);
        assert_eq!(stability.scores[0].score, 0.5);
    }
}

#[test]
fn adaptive_content_tool_payload_whitespace_is_not_prose_normalized() {
    let request = AnnotatedLlmRequest {
        messages: vec![
            Message::Tool {
                tool_call_id: "c".into(),
                content: MessageContent::Text("{\"city\":\"New  York\"}".into()),
            },
            Message::Function {
                name: "lookup".into(),
                content: Some("{\"city\":\"New  York\"}".into()),
            },
            Message::Assistant {
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "c".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "lookup".into(),
                        arguments: "{\"city\":\"New  York\"}".into(),
                    },
                }]),
                name: None,
            },
        ],
        ..Default::default()
    };
    let ir = build_prompt_ir(&request).unwrap();
    assert_eq!(ir.blocks.len(), 3);
    assert!(
        ir.blocks
            .iter()
            .all(|block| block.content.contains("New  York"))
    );
    assert!(
        ir.blocks
            .iter()
            .all(|block| block.content_type != BlockContentType::Text)
    );
}

#[test]
fn adaptive_content_media_and_native_values_are_typed_opaque_fingerprints() {
    for part in [
        serde_json::json!({"type": "image", "image": {"data": "private-payload"}}),
        serde_json::json!({"type": "audio", "audio": {"data": "private-payload"}}),
        serde_json::json!({"type": "file", "file": {"file_id": "private-payload"}}),
        serde_json::json!({"type": "provider_native", "provider": "custom", "kind": "output_text", "value": {"text": "private-payload"}}),
    ] {
        let ir = build_prompt_ir(&AnnotatedLlmRequest {
            messages: vec![Message::User {
                content: serde_json::from_value(serde_json::json!([
                    {"type": "text", "text": "inspect"}, part
                ]))
                .unwrap(),
                name: None,
            }],
            ..Default::default()
        })
        .unwrap();
        let value: serde_json::Value = serde_json::from_str(&ir.blocks[0].content).unwrap();
        assert_eq!(value[1]["type"], part["type"]);
        assert!(
            value[1]["fingerprint"]
                .as_str()
                .unwrap()
                .starts_with("sha256:")
        );
        assert!(!ir.blocks[0].content.contains("private-payload"));
        assert_eq!(ir.blocks[0].content_type, BlockContentType::Opaque);
        assert!(ir.blocks[0].token_metadata.is_none());
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

#[test]
fn build_prompt_ir_canonicalizes_structured_output_before_user_content() {
    let mut request = AnnotatedLlmRequest {
        messages: vec![Message::User {
            content: MessageContent::Text("variable task".to_string()),
            name: None,
        }],
        model: Some("gpt-4o".to_string()),
        ..AnnotatedLlmRequest::default()
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
