// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for prompt ir in the NeMo Relay adaptive crate.

use super::*;
use chrono::Utc;
use std::collections::HashMap;
use uuid::Uuid;

fn assert_send_sync<T: Send + Sync>() {}

fn sample_prompt_block(
    index: u32,
    role: PromptRole,
    content_type: BlockContentType,
) -> PromptBlock {
    PromptBlock {
        span_id: SpanId(format!("span-{index}")),
        sequence_index: index,
        role,
        content: format!("content for block {index}"),
        content_type,
        provenance: ProvenanceLabel::Developer,
        sensitivity: SensitivityLabel::Public,
        token_metadata: None,
    }
}

// -------------------------------------------------------------------
// Send + Sync compile-time assertions
// -------------------------------------------------------------------

#[test]
fn test_prompt_ir_types_are_send_sync() {
    assert_send_sync::<SpanId>();
    assert_send_sync::<ProvenanceLabel>();
    assert_send_sync::<SensitivityLabel>();
    assert_send_sync::<PromptRole>();
    assert_send_sync::<BlockContentType>();
    assert_send_sync::<TokenizationMetadata>();
    assert_send_sync::<PromptBlock>();
    assert_send_sync::<ToolSchemaHash>();
    assert_send_sync::<PromptIR>();
}

// -------------------------------------------------------------------
// SpanId equality and hashing
// -------------------------------------------------------------------

#[test]
fn test_span_id_equality_and_hashing() {
    let a = SpanId("span-1".to_string());
    let b = SpanId("span-1".to_string());
    let c = SpanId("span-2".to_string());

    assert_eq!(a, b);
    assert_ne!(a, c);

    // Can be used as HashMap key
    let mut map = HashMap::new();
    map.insert(a.clone(), 42);
    assert_eq!(map.get(&b), Some(&42));
    assert_eq!(map.get(&c), None);
}

// -------------------------------------------------------------------
// ProvenanceLabel serde round-trip
// -------------------------------------------------------------------

#[test]
fn test_provenance_label_serde() {
    for label in [
        ProvenanceLabel::System,
        ProvenanceLabel::Developer,
        ProvenanceLabel::User,
        ProvenanceLabel::Tool,
        ProvenanceLabel::Retrieval,
        ProvenanceLabel::Memory,
    ] {
        let json = serde_json::to_string(&label).unwrap();
        let restored: ProvenanceLabel = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, label);
    }

    // Verify snake_case naming
    assert_eq!(
        serde_json::to_string(&ProvenanceLabel::System).unwrap(),
        "\"system\""
    );
    assert_eq!(
        serde_json::to_string(&ProvenanceLabel::Developer).unwrap(),
        "\"developer\""
    );
    assert_eq!(
        serde_json::to_string(&ProvenanceLabel::User).unwrap(),
        "\"user\""
    );
    assert_eq!(
        serde_json::to_string(&ProvenanceLabel::Tool).unwrap(),
        "\"tool\""
    );
    assert_eq!(
        serde_json::to_string(&ProvenanceLabel::Retrieval).unwrap(),
        "\"retrieval\""
    );
    assert_eq!(
        serde_json::to_string(&ProvenanceLabel::Memory).unwrap(),
        "\"memory\""
    );
}

// -------------------------------------------------------------------
// SensitivityLabel serde round-trip and default
// -------------------------------------------------------------------

#[test]
fn test_sensitivity_label_serde() {
    for label in [
        SensitivityLabel::Public,
        SensitivityLabel::Private,
        SensitivityLabel::Restricted,
    ] {
        let json = serde_json::to_string(&label).unwrap();
        let restored: SensitivityLabel = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, label);
    }
}

#[test]
fn test_sensitivity_label_default_is_public() {
    assert_eq!(SensitivityLabel::default(), SensitivityLabel::Public);
}

// -------------------------------------------------------------------
// PromptRole serde round-trip
// -------------------------------------------------------------------

#[test]
fn test_prompt_role_serde() {
    for role in [
        PromptRole::System,
        PromptRole::User,
        PromptRole::Assistant,
        PromptRole::Tool,
    ] {
        let json = serde_json::to_string(&role).unwrap();
        let restored: PromptRole = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, role);
    }

    assert_eq!(
        serde_json::to_string(&PromptRole::System).unwrap(),
        "\"system\""
    );
    assert_eq!(
        serde_json::to_string(&PromptRole::User).unwrap(),
        "\"user\""
    );
    assert_eq!(
        serde_json::to_string(&PromptRole::Assistant).unwrap(),
        "\"assistant\""
    );
    assert_eq!(
        serde_json::to_string(&PromptRole::Tool).unwrap(),
        "\"tool\""
    );
}

// -------------------------------------------------------------------
// BlockContentType serde round-trip
// -------------------------------------------------------------------

#[test]
fn test_block_content_type_serde() {
    for ct in [
        BlockContentType::Text,
        BlockContentType::ToolSchema,
        BlockContentType::ToolResult,
        BlockContentType::StructuredOutput,
        BlockContentType::Image,
    ] {
        let json = serde_json::to_string(&ct).unwrap();
        let restored: BlockContentType = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, ct);
    }

    assert_eq!(
        serde_json::to_string(&BlockContentType::Text).unwrap(),
        "\"text\""
    );
    assert_eq!(
        serde_json::to_string(&BlockContentType::ToolSchema).unwrap(),
        "\"tool_schema\""
    );
    assert_eq!(
        serde_json::to_string(&BlockContentType::ToolResult).unwrap(),
        "\"tool_result\""
    );
    assert_eq!(
        serde_json::to_string(&BlockContentType::StructuredOutput).unwrap(),
        "\"structured_output\""
    );
    assert_eq!(
        serde_json::to_string(&BlockContentType::Image).unwrap(),
        "\"image\""
    );
}

// -------------------------------------------------------------------
// TokenizationMetadata serde round-trip
// -------------------------------------------------------------------

#[test]
fn test_tokenization_metadata_serde() {
    let meta = TokenizationMetadata {
        model_family: "claude".to_string(),
        token_count: 1234,
    };
    let json = serde_json::to_string(&meta).unwrap();
    let restored: TokenizationMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, meta);
}

// -------------------------------------------------------------------
// PromptBlock serde round-trip
// -------------------------------------------------------------------

#[test]
fn test_prompt_block_serde_with_all_fields() {
    let block = PromptBlock {
        span_id: SpanId("span-42".to_string()),
        sequence_index: 3,
        role: PromptRole::User,
        content: "What is the weather?".to_string(),
        content_type: BlockContentType::Text,
        provenance: ProvenanceLabel::User,
        sensitivity: SensitivityLabel::Private,
        token_metadata: Some(TokenizationMetadata {
            model_family: "gpt".to_string(),
            token_count: 5,
        }),
    };
    let json = serde_json::to_string(&block).unwrap();
    let restored: PromptBlock = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, block);
}

#[test]
fn test_prompt_block_serde_without_optional_fields() {
    let block = PromptBlock {
        span_id: SpanId("span-1".to_string()),
        sequence_index: 0,
        role: PromptRole::System,
        content: "You are a helpful assistant.".to_string(),
        content_type: BlockContentType::Text,
        provenance: ProvenanceLabel::System,
        sensitivity: SensitivityLabel::Public,
        token_metadata: None,
    };
    let json = serde_json::to_string(&block).unwrap();
    // token_metadata should be absent from JSON when None
    assert!(!json.contains("token_metadata"));
    let restored: PromptBlock = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, block);
}

// -------------------------------------------------------------------
// ToolSchemaHash serde round-trip and equality
// -------------------------------------------------------------------

#[test]
fn test_tool_schema_hash_serde_and_equality() {
    let hash_a = ToolSchemaHash {
        tool_name: "search".to_string(),
        schema_hash: "sha256:abc123".to_string(),
    };
    let hash_b = ToolSchemaHash {
        tool_name: "search".to_string(),
        schema_hash: "sha256:abc123".to_string(),
    };
    let hash_c = ToolSchemaHash {
        tool_name: "calculator".to_string(),
        schema_hash: "sha256:def456".to_string(),
    };

    assert_eq!(hash_a, hash_b);
    assert_ne!(hash_a, hash_c);

    let json = serde_json::to_string(&hash_a).unwrap();
    let restored: ToolSchemaHash = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, hash_a);
}

// -------------------------------------------------------------------
// PromptIR serde round-trip
// -------------------------------------------------------------------

#[test]
fn test_prompt_ir_serde_full() {
    let ir = PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: vec![
            sample_prompt_block(0, PromptRole::System, BlockContentType::Text),
            sample_prompt_block(1, PromptRole::User, BlockContentType::Text),
            PromptBlock {
                span_id: SpanId("span-2".to_string()),
                sequence_index: 2,
                role: PromptRole::Tool,
                content: r#"{"result": "42"}"#.to_string(),
                content_type: BlockContentType::ToolResult,
                provenance: ProvenanceLabel::Tool,
                sensitivity: SensitivityLabel::Restricted,
                token_metadata: Some(TokenizationMetadata {
                    model_family: "claude".to_string(),
                    token_count: 10,
                }),
            },
        ],
        tool_schema_hashes: Some(vec![ToolSchemaHash {
            tool_name: "search".to_string(),
            schema_hash: "sha256:abc".to_string(),
        }]),
        structured_output_schema_id: Some("output-schema-v1".to_string()),
        source_request_hash: Some("sha256:request-hash".to_string()),
        created_at: Utc::now(),
    };

    let json = serde_json::to_string(&ir).unwrap();
    let restored: PromptIR = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, ir);
}

#[test]
fn test_prompt_ir_serde_without_optional_fields() {
    let ir = PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: vec![sample_prompt_block(
            0,
            PromptRole::System,
            BlockContentType::Text,
        )],
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    };

    let json = serde_json::to_string(&ir).unwrap();
    // Optional fields should be absent
    assert!(!json.contains("tool_schema_hashes"));
    assert!(!json.contains("structured_output_schema_id"));
    assert!(!json.contains("source_request_hash"));

    let restored: PromptIR = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, ir);
}

#[test]
fn test_prompt_ir_empty_blocks() {
    let ir = PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: vec![],
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    };

    let json = serde_json::to_string(&ir).unwrap();
    let restored: PromptIR = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, ir);
    assert!(restored.blocks.is_empty());
}
