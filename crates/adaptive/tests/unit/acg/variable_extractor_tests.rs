// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for variable extractor in the NeMo Relay adaptive crate.

use regex::Regex;

use super::*;

use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptRole, ProvenanceLabel, SensitivityLabel,
};

#[test]
fn variable_extractor_default_patterns_cover_expected_categories_and_debug_output() {
    let patterns = default_variable_patterns();
    let names = patterns
        .iter()
        .map(|pattern| pattern.name.as_str())
        .collect::<Vec<_>>();

    assert!(names.contains(&"iso8601_timestamp"));
    assert!(names.contains(&"uuid"));
    assert!(names.contains(&"request_id"));
    assert!(names.contains(&"date_string"));
    assert!(names.contains(&"unix_timestamp"));
    assert_eq!(patterns[0].category, VariableCategory::Timestamp);
    assert!(format!("{:?}", patterns[0]).contains("iso8601_timestamp"));
}

#[test]
fn variable_extractor_returns_none_when_no_patterns_match() {
    assert!(
        extract_variables(
            "stable prompt content",
            &SpanId("span-0".to_string()),
            &default_variable_patterns(),
        )
        .is_none()
    );
}

#[test]
fn variable_extractor_prefers_longer_overlapping_matches_and_extracts_blocks() {
    let patterns = vec![
        VariablePattern {
            name: "long".to_string(),
            regex: Regex::new(r"req_abc12345").unwrap(),
            category: VariableCategory::RequestId,
        },
        VariablePattern {
            name: "short".to_string(),
            regex: Regex::new(r"abc123").unwrap(),
            category: VariableCategory::Custom("short".into()),
        },
    ];
    let span_id = SpanId("span-1".to_string());
    let extraction = extract_variables("trace req_abc12345 done", &span_id, &patterns).unwrap();

    assert_eq!(extraction.template_content, "trace {{long}} done");
    assert_eq!(extraction.variables.len(), 1);
    assert_eq!(extraction.variables[0].pattern_name, "long");
    assert_eq!(extraction.variables[0].original_value, "req_abc12345");

    let blocks = vec![
        PromptBlock {
            span_id: span_id.clone(),
            sequence_index: 0,
            role: PromptRole::System,
            content: "2026-04-17T12:34:56Z".to_string(),
            content_type: BlockContentType::Text,
            provenance: ProvenanceLabel::System,
            sensitivity: SensitivityLabel::Public,
            token_metadata: None,
        },
        PromptBlock {
            span_id: SpanId("span-2".to_string()),
            sequence_index: 1,
            role: PromptRole::User,
            content: "no variables here".to_string(),
            content_type: BlockContentType::Text,
            provenance: ProvenanceLabel::User,
            sensitivity: SensitivityLabel::Public,
            token_metadata: None,
        },
    ];

    let results = extract_variables_from_blocks(&blocks, &default_variable_patterns());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].span_id, span_id);
}
