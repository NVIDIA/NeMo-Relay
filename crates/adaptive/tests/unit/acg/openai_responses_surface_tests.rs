// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for openai responses surface in the NeMo Relay adaptive crate.

use chrono::Utc;
use serde_json::json;

use super::*;

use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel, SpanId,
};
use crate::acg::translation::HintTarget;

fn prompt_ir() -> PromptIR {
    PromptIR {
        ir_id: uuid::Uuid::new_v4(),
        blocks: vec![
            PromptBlock {
                span_id: SpanId("system-0".to_string()),
                sequence_index: 0,
                role: PromptRole::System,
                content: "system".to_string(),
                content_type: BlockContentType::Text,
                provenance: ProvenanceLabel::System,
                sensitivity: SensitivityLabel::Public,
                token_metadata: None,
            },
            PromptBlock {
                span_id: SpanId("tool-1".to_string()),
                sequence_index: 1,
                role: PromptRole::System,
                content: "tool".to_string(),
                content_type: BlockContentType::ToolSchema,
                provenance: ProvenanceLabel::System,
                sensitivity: SensitivityLabel::Public,
                token_metadata: None,
            },
            PromptBlock {
                span_id: SpanId("user-2".to_string()),
                sequence_index: 2,
                role: PromptRole::User,
                content: "user".to_string(),
                content_type: BlockContentType::Text,
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
fn openai_responses_stable_prefix_helper_handles_missing_input_and_skips_system_and_tools() {
    let prompt_ir = prompt_ir();
    let target = HintTarget::stable_prefix(3, Some(SpanId("user-2".to_string())));

    let mut missing_input = json!({});
    canonicalize_responses_stable_prefix(&mut missing_input, &prompt_ir, &target);

    let mut content = json!({
        "input": [
            {"role": "user", "content": [{"type":"tool_result","data":{"z":1,"a":2}}]}
        ]
    });
    canonicalize_responses_stable_prefix(&mut content, &prompt_ir, &target);

    assert_eq!(
        content["input"][0]["content"][0],
        json!({"data":{"a":2,"z":1},"type":"tool_result"})
    );
}
