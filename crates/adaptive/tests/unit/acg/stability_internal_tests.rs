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

#[test]
fn profile_fingerprint_requires_source_hash_beyond_the_scaffold() {
    let mut observation = prompt(vec![
        block("system-0", 0, "stable policy"),
        block("user-1", 1, "stable task"),
    ]);
    observation.blocks[1].role = PromptRole::User;

    assert_eq!(
        profile_prefix_fingerprint(&observation, 2, "learning-key"),
        None
    );
}

fn scored(span_id: &str, classification: StabilityClass) -> BlockStabilityScore {
    BlockStabilityScore {
        span_id: SpanId(span_id.to_string()),
        classification,
        score: 1.0,
        confidence: 1.0,
        observation_count: 1,
    }
}

/// Two spans differing only in role or tool suffix share a sequence index, so
/// the index alone is not a total order over the span set. `canonical_scores`
/// must therefore map any permutation of the same span set onto one sequence.
#[test]
fn stability_internal_canonical_scores_are_invariant_under_input_permutation() {
    let tied = vec![
        (1_u32, scored("assistant-1-search", StabilityClass::Stable)),
        (1, scored("assistant-1-fetch", StabilityClass::Variable)),
        (1, scored("assistant-1-write", StabilityClass::SemiStable)),
        // Shares the index and the rank of `assistant-1-search`, so only the
        // span id separates them.
        (1, scored("assistant-1-annotate", StabilityClass::Stable)),
        (0, scored("system-0", StabilityClass::Stable)),
        (2, scored("tool-2-search", StabilityClass::Stable)),
    ];

    // Every rotation is a distinct HashMap iteration order over the same set.
    let expected = canonical_scores(tied.clone());
    for rotation in 1..tied.len() {
        let mut permuted = tied.clone();
        permuted.rotate_left(rotation);
        assert_eq!(canonical_scores(permuted), expected, "rotation {rotation}");
    }
}

/// At a contested sequence index the least stable span sorts first, so the
/// stable prefix stops before the position rather than extending across it.
#[test]
fn stability_internal_canonical_scores_put_the_least_stable_span_first() {
    let contested = vec![
        (0_u32, scored("system-0", StabilityClass::Stable)),
        (1, scored("assistant-1-search", StabilityClass::Stable)),
        (1, scored("assistant-1-fetch", StabilityClass::Variable)),
    ];

    let ordered = canonical_scores(contested);
    assert_eq!(ordered[1].span_id, SpanId("assistant-1-fetch".to_string()));
}

/// The observation window is a multiset: the analysis aggregates it with `min`,
/// summation, and multiset counts, all symmetric, so reordering the window must
/// not move the stable prefix or its fingerprint.
#[test]
fn stability_internal_analysis_is_invariant_under_observation_permutation() {
    let thresholds = StabilityThresholds::default();
    let turn = |span: &str, content: &str| {
        prompt(vec![
            block("system-0", 0, "Follow policy"),
            block(span, 1, content),
        ])
    };
    let observations = vec![
        turn("assistant-1-search", "search(alpha)"),
        turn("assistant-1-search", "search(alpha)"),
        turn("assistant-1-fetch", "fetch(beta)"),
        turn("assistant-1-search", "search(gamma)"),
    ];

    let expected = analyze_stability(&observations, &thresholds);
    for rotation in 1..observations.len() {
        let mut permuted = observations.clone();
        permuted.rotate_left(rotation);
        assert_eq!(
            analyze_stability(&permuted, &thresholds),
            expected,
            "rotation {rotation}"
        );
    }
}

/// 19 of 20 observations share an exact two-block prefix, so a provider cache
/// would hit on it for those 19. The per-span rule discards it: the minority
/// span at index 1 is Variable and stops the leading run.
#[test]
fn stability_internal_keeps_a_prefix_shared_by_the_dominant_mass() {
    let thresholds = StabilityThresholds::default();
    let turn = |span: &str, content: &str| {
        prompt(vec![
            block("system-0", 0, "Follow policy"),
            block(span, 1, content),
        ])
    };
    let observations: Vec<_> = (0..20)
        .map(|i| {
            if i == 7 {
                turn("assistant-1-fetch", "fetch(beta)")
            } else {
                turn("assistant-1-search", "search(alpha)")
            }
        })
        .collect();

    assert_eq!(
        analyze_stability(&observations, &thresholds).stable_prefix_length,
        2
    );
}
