// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for stability internal in the NeMo Relay adaptive crate.

use chrono::Utc;

use super::*;

use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptRole, ProvenanceLabel, SensitivityLabel,
};

fn prompt(blocks: Vec<PromptBlock>) -> PromptIR {
    PromptIR {
        ir_id: uuid::Uuid::new_v4(),
        blocks,
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    }
}

fn block(span_id: &str, sequence_index: u32, content: &str) -> PromptBlock {
    PromptBlock {
        span_id: SpanId(span_id.to_string()),
        sequence_index,
        role: PromptRole::System,
        content: content.to_string(),
        content_type: BlockContentType::Text,
        provenance: ProvenanceLabel::System,
        sensitivity: SensitivityLabel::Public,
        token_metadata: None,
    }
}

#[test]
fn stability_internal_handles_empty_inputs_variable_scores_and_zero_confidence_threshold() {
    let thresholds = StabilityThresholds::default();
    let empty = analyze_stability(&[], &thresholds);
    assert_eq!(empty.total_observations, 0);
    assert_eq!(empty.stable_prefix_length, 0);
    assert!(empty.scores.is_empty());

    let observations = vec![
        prompt(vec![block("span-0", 0, "A"), block("span-1", 1, "X")]),
        prompt(vec![block("span-0", 0, "A")]),
        prompt(vec![block("span-0", 0, "B"), block("span-1", 1, "Y")]),
    ];
    let result = analyze_stability(&observations, &thresholds);
    assert_eq!(result.scores.len(), 2);
    assert!(
        result
            .scores
            .iter()
            .any(|score| score.classification == StabilityClass::Variable)
    );

    let zero_threshold = StabilityThresholds {
        min_observations_for_full_confidence: 0,
        ..StabilityThresholds::default()
    };
    assert_eq!(stability_confidence(1, &zero_threshold), 1.0);
    assert_eq!(
        classify_stability(0.1, &thresholds),
        StabilityClass::Variable
    );
}

#[test]
fn stability_internal_effective_score_handles_zero_present_count() {
    let observations = SpanObservations {
        hash_counts: std::collections::HashMap::new(),
        present_count: 0,
        first_seen_sequence_index: 0,
    };

    assert_eq!(effective_stability_score(&observations, 3), 0.0);
}

#[test]
fn stability_internal_prefix_fingerprint_ties_choose_lexicographically_smallest_hash() {
    let mut counts = std::collections::HashMap::new();
    counts.insert("sha256:bbbb".to_string(), 2);
    counts.insert("sha256:aaaa".to_string(), 2);
    counts.insert("sha256:cccc".to_string(), 1);

    assert_eq!(
        select_dominant_prefix_fingerprint(counts).as_deref(),
        Some("sha256:aaaa")
    );
}

#[test]
fn stability_internal_equal_sequence_indexes_sort_by_span_id() {
    let mut indexed_scores = vec![
        (
            1,
            BlockStabilityScore {
                span_id: SpanId("span-b".to_string()),
                classification: StabilityClass::Stable,
                score: 1.0,
                confidence: 1.0,
                observation_count: 3,
            },
        ),
        (
            1,
            BlockStabilityScore {
                span_id: SpanId("span-a".to_string()),
                classification: StabilityClass::Stable,
                score: 1.0,
                confidence: 1.0,
                observation_count: 3,
            },
        ),
        (
            0,
            BlockStabilityScore {
                span_id: SpanId("span-0".to_string()),
                classification: StabilityClass::Stable,
                score: 1.0,
                confidence: 1.0,
                observation_count: 3,
            },
        ),
    ];

    sort_indexed_scores(&mut indexed_scores);

    let ordered_span_ids = indexed_scores
        .iter()
        .map(|(_, score)| score.span_id.0.as_str())
        .collect::<Vec<_>>();
    assert_eq!(ordered_span_ids, vec!["span-0", "span-a", "span-b"]);
}
