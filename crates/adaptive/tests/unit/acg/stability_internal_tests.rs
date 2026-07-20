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
fn stability_internal_fingerprints_the_exact_dominant_prefix() {
    let first = prompt(vec![
        block("system-0", 0, "stable policy"),
        block("user-1", 1, "task alpha"),
    ]);
    let second = prompt(vec![
        block("system-0", 0, "stable policy"),
        block("user-1", 1, "task beta"),
    ]);
    let changed = prompt(vec![
        block("system-0", 0, "changed policy"),
        block("user-1", 1, "task gamma"),
    ]);

    let result = analyze_stability(&[first.clone(), second], &StabilityThresholds::default());

    assert_eq!(result.stable_prefix_length, 1);
    assert_eq!(
        result.stable_prefix_fingerprint,
        prompt_prefix_fingerprint(&first, 1)
    );
    assert_ne!(
        result.stable_prefix_fingerprint,
        prompt_prefix_fingerprint(&changed, 1)
    );
    assert_eq!(prompt_prefix_fingerprint(&first, 0), None);
}

#[test]
fn stability_internal_deserializes_persisted_state_without_a_prefix_fingerprint() {
    let restored: StabilityAnalysisResult = serde_json::from_value(serde_json::json!({
        "scores": [],
        "stable_prefix_length": 0,
        "total_observations": 4
    }))
    .unwrap();

    assert_eq!(restored.stable_prefix_fingerprint, None);
}
