// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for economics internal in the NeMo Relay adaptive crate.

use std::collections::HashSet;

use chrono::Utc;
use uuid::Uuid;

use super::*;

use crate::acg::capability::{CacheEconomics, ModelFamilyCapabilities};
use crate::acg::profile::{BlockStabilityScore, StabilityClass};
use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel, SpanId,
    TokenizationMetadata,
};
use crate::acg::stability::StabilityAnalysisResult;

fn pricing() -> CacheEconomics {
    CacheEconomics {
        write_short_multiplier: 1.25,
        write_long_multiplier: Some(2.0),
        read_multiplier: 0.1,
    }
}

fn model_capabilities() -> ModelFamilyCapabilities {
    ModelFamilyCapabilities {
        model_family: "claude-sonnet-4-20250514".to_string(),
        supported_features: HashSet::new(),
        max_cache_breakpoints: Some(3),
        min_cacheable_tokens: Some(400),
        cache_economics: Some(pricing()),
    }
}

fn block(
    index: usize,
    role: PromptRole,
    content_type: BlockContentType,
    provenance: ProvenanceLabel,
    tokens: u32,
) -> PromptBlock {
    PromptBlock {
        span_id: SpanId(format!("block-{index}")),
        sequence_index: index as u32,
        role,
        content: "x".repeat((tokens as usize) * 4),
        content_type,
        provenance,
        sensitivity: SensitivityLabel::Public,
        token_metadata: Some(TokenizationMetadata {
            model_family: "claude".to_string(),
            token_count: tokens,
        }),
    }
}

fn prompt_ir(blocks: Vec<PromptBlock>) -> PromptIR {
    PromptIR {
        ir_id: Uuid::new_v4(),
        blocks,
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    }
}

fn score(
    index: usize,
    classification: StabilityClass,
    value: f64,
    confidence: f64,
) -> BlockStabilityScore {
    BlockStabilityScore {
        span_id: SpanId(format!("block-{index}")),
        classification,
        score: value,
        confidence,
        observation_count: 6,
    }
}

#[test]
fn economics_internal_skip_reasons_include_every_gate() {
    assert_eq!(
        breakpoint_skip_reasons(0.0, 0, 0, u32::MAX),
        vec![
            "reuse_horizon_non_positive",
            "max_breakpoints_zero",
            "stable_prefix_empty",
            "min_cacheable_tokens_unavailable",
        ]
    );
}

#[test]
fn economics_internal_build_prefix_stats_stops_at_first_non_stable_block() {
    let prompt_ir = prompt_ir(vec![
        block(
            0,
            PromptRole::System,
            BlockContentType::Text,
            ProvenanceLabel::System,
            200,
        ),
        block(
            1,
            PromptRole::User,
            BlockContentType::Text,
            ProvenanceLabel::User,
            300,
        ),
    ]);
    let stability = StabilityAnalysisResult {
        scores: vec![
            score(0, StabilityClass::Stable, 0.8, 0.5),
            score(1, StabilityClass::Variable, 0.4, 0.9),
        ],
        stable_prefix_length: 2,
        total_observations: 6,
        converged: false,
    };

    let stats = build_prefix_stats(
        &prompt_ir,
        &stability,
        2,
        400,
        5.0,
        &model_capabilities(),
        6,
    );

    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].cumulative_tokens, 200);
    assert!((stats[0].weakest_prefix_signal - 0.4).abs() < 1e-9);
}

#[test]
fn economics_internal_classify_candidate_boundary_covers_semantic_variants() {
    let prompt_ir = prompt_ir(vec![
        block(
            0,
            PromptRole::System,
            BlockContentType::Text,
            ProvenanceLabel::System,
            200,
        ),
        block(
            1,
            PromptRole::User,
            BlockContentType::Text,
            ProvenanceLabel::User,
            200,
        ),
        block(
            2,
            PromptRole::Assistant,
            BlockContentType::StructuredOutput,
            ProvenanceLabel::Developer,
            200,
        ),
        block(
            3,
            PromptRole::Assistant,
            BlockContentType::Text,
            ProvenanceLabel::Retrieval,
            200,
        ),
        block(
            4,
            PromptRole::System,
            BlockContentType::ToolSchema,
            ProvenanceLabel::System,
            200,
        ),
        block(
            5,
            PromptRole::System,
            BlockContentType::ToolSchema,
            ProvenanceLabel::System,
            200,
        ),
        block(
            6,
            PromptRole::Assistant,
            BlockContentType::Text,
            ProvenanceLabel::Developer,
            200,
        ),
    ]);

    assert_eq!(
        classify_candidate_boundary(&prompt_ir, 0, 7, false),
        Some(CandidateKind::System)
    );
    assert_eq!(
        classify_candidate_boundary(&prompt_ir, 1, 7, false),
        Some(CandidateKind::User)
    );
    assert_eq!(
        classify_candidate_boundary(&prompt_ir, 2, 7, false),
        Some(CandidateKind::Structured)
    );
    assert_eq!(
        classify_candidate_boundary(&prompt_ir, 3, 7, false),
        Some(CandidateKind::Retrieval)
    );
    assert_eq!(classify_candidate_boundary(&prompt_ir, 4, 7, false), None);
    assert_eq!(
        classify_candidate_boundary(&prompt_ir, 4, 7, true),
        Some(CandidateKind::ToolCluster)
    );
    assert_eq!(
        classify_candidate_boundary(&prompt_ir, 6, 7, false),
        Some(CandidateKind::Generic)
    );
}

#[test]
fn economics_internal_collect_candidates_rejects_non_semantic_internal_boundaries() {
    let prompt_ir = prompt_ir(vec![
        block(
            0,
            PromptRole::System,
            BlockContentType::Text,
            ProvenanceLabel::System,
            150,
        ),
        block(
            1,
            PromptRole::System,
            BlockContentType::ToolSchema,
            ProvenanceLabel::System,
            150,
        ),
        block(
            2,
            PromptRole::System,
            BlockContentType::ToolSchema,
            ProvenanceLabel::System,
            150,
        ),
        block(
            3,
            PromptRole::User,
            BlockContentType::Text,
            ProvenanceLabel::User,
            150,
        ),
    ]);
    let prefix_stats = vec![
        PrefixStats {
            cumulative_tokens: 150,
            weakest_prefix_signal: 1.0,
        },
        PrefixStats {
            cumulative_tokens: 300,
            weakest_prefix_signal: 0.9,
        },
        PrefixStats {
            cumulative_tokens: 450,
            weakest_prefix_signal: 0.8,
        },
        PrefixStats {
            cumulative_tokens: 600,
            weakest_prefix_signal: 0.7,
        },
    ];

    let candidates = collect_breakpoint_candidates(
        &prompt_ir,
        &prefix_stats,
        4,
        5.0,
        100,
        &model_capabilities(),
    );

    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.stable_prefix_end == 1)
    );
    assert!(
        !candidates
            .iter()
            .any(|candidate| candidate.stable_prefix_end == 2)
    );
    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.stable_prefix_end == 3)
    );
    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.stable_prefix_end == 4)
    );
}

#[test]
fn economics_internal_append_selected_breakpoints_skips_zero_and_non_profitable_segments() {
    let candidates = vec![
        BreakpointCandidate {
            stable_prefix_end: 1,
            cumulative_tokens: 100,
            weakest_prefix_signal: 1.0,
            expected_reads: 1.0,
            kind: CandidateKind::System,
        },
        BreakpointCandidate {
            stable_prefix_end: 2,
            cumulative_tokens: 100,
            weakest_prefix_signal: 1.0,
            expected_reads: 10.0,
            kind: CandidateKind::ToolCluster,
        },
        BreakpointCandidate {
            stable_prefix_end: 3,
            cumulative_tokens: 150,
            weakest_prefix_signal: 1.0,
            expected_reads: 0.0,
            kind: CandidateKind::User,
        },
    ];
    let mut plan = build_economics_plan(&pricing(), 100, 5.0);

    append_selected_breakpoints(
        &mut plan,
        &[0, 1, 2],
        &candidates,
        &pricing(),
        &model_capabilities(),
        5.0,
    );

    assert_eq!(plan.planned_breakpoints.len(), 1);
    assert_eq!(plan.planned_breakpoints[0].stable_prefix_end, 1);
    assert_eq!(plan.planned_breakpoints[0].marginal_tokens, 100);
}

#[test]
fn economics_internal_candidate_selection_covers_empty_paths_extensions_and_tiebreakers() {
    let pricing = pricing();
    let candidates = vec![
        BreakpointCandidate {
            stable_prefix_end: 1,
            cumulative_tokens: 100,
            weakest_prefix_signal: 1.0,
            expected_reads: 1.0,
            kind: CandidateKind::System,
        },
        BreakpointCandidate {
            stable_prefix_end: 2,
            cumulative_tokens: 100,
            weakest_prefix_signal: 1.0,
            expected_reads: 1.0,
            kind: CandidateKind::ToolCluster,
        },
        BreakpointCandidate {
            stable_prefix_end: 3,
            cumulative_tokens: 200,
            weakest_prefix_signal: 1.0,
            expected_reads: 0.0,
            kind: CandidateKind::User,
        },
        BreakpointCandidate {
            stable_prefix_end: 4,
            cumulative_tokens: 220,
            weakest_prefix_signal: 1.0,
            expected_reads: 1.0,
            kind: CandidateKind::Retrieval,
        },
    ];

    assert!(select_breakpoint_candidates(&[], 1, &pricing).is_none());
    assert!(select_breakpoint_candidates(&candidates, 0, &pricing).is_none());

    let mut dp = vec![vec![None; 3]; candidates.len()];
    seed_candidate_selection_state(&mut dp, 0, &candidates[0], &pricing);
    assert!(candidate_extension_proposal(&dp, &candidates, 1, 2, 0, &pricing).is_none());
    assert!(candidate_extension_proposal(&dp, &candidates, 2, 2, 0, &pricing).is_none());
    let proposal = candidate_extension_proposal(&dp, &candidates, 3, 2, 0, &pricing)
        .expect("positive segment should produce a proposal");
    assert!(proposal.total_value > 0.0);

    assert!(is_better_candidate_state(
        Some(CandidateSelectionState {
            total_value: 5.0,
            previous_candidate_index: Some(0),
        }),
        CandidateSelectionState {
            total_value: 5.0,
            previous_candidate_index: Some(1),
        },
        &candidates,
        1,
    ));

    assert!(is_better_terminal_state(
        None,
        (
            0,
            1,
            CandidateSelectionState {
                total_value: 1.0,
                previous_candidate_index: None,
            },
        ),
        &candidates,
    ));
    assert!(is_better_terminal_state(
        Some((
            0,
            2,
            CandidateSelectionState {
                total_value: 3.0,
                previous_candidate_index: Some(0),
            },
        )),
        (
            1,
            1,
            CandidateSelectionState {
                total_value: 3.0,
                previous_candidate_index: None,
            },
        ),
        &candidates,
    ));
    assert!(is_better_terminal_state(
        Some((
            0,
            1,
            CandidateSelectionState {
                total_value: 3.0,
                previous_candidate_index: None,
            },
        )),
        (
            3,
            1,
            CandidateSelectionState {
                total_value: 3.0,
                previous_candidate_index: None,
            },
        ),
        &candidates,
    ));
    assert!(is_better_terminal_state(
        Some((
            1,
            1,
            CandidateSelectionState {
                total_value: 3.0,
                previous_candidate_index: None,
            },
        )),
        (
            2,
            1,
            CandidateSelectionState {
                total_value: 3.0,
                previous_candidate_index: None,
            },
        ),
        &[
            BreakpointCandidate {
                stable_prefix_end: 1,
                cumulative_tokens: 100,
                weakest_prefix_signal: 1.0,
                expected_reads: 1.0,
                kind: CandidateKind::User,
            },
            BreakpointCandidate {
                stable_prefix_end: 2,
                cumulative_tokens: 200,
                weakest_prefix_signal: 1.0,
                expected_reads: 1.0,
                kind: CandidateKind::User,
            },
            BreakpointCandidate {
                stable_prefix_end: 3,
                cumulative_tokens: 300,
                weakest_prefix_signal: 1.0,
                expected_reads: 1.0,
                kind: CandidateKind::User,
            },
        ],
    ));
}
