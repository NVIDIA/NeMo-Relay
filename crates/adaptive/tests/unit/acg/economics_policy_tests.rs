// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for economics policy in the NeMo Relay adaptive crate.

use std::collections::HashSet;

use chrono::Utc;
use uuid::Uuid;

use crate::acg::capability::{CacheEconomics, ModelFamilyCapabilities};
use crate::acg::economics::plan_breakpoints;
use crate::acg::profile::{BlockStabilityScore, StabilityClass};
use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel, SpanId,
    TokenizationMetadata,
};
use crate::acg::stability::StabilityAnalysisResult;
use crate::acg::types::SharingScope;

fn model_capabilities(
    max_cache_breakpoints: u32,
    min_cacheable_tokens: u32,
) -> ModelFamilyCapabilities {
    ModelFamilyCapabilities {
        model_family: "claude-sonnet-4-20250514".to_string(),
        supported_features: HashSet::new(),
        max_cache_breakpoints: Some(max_cache_breakpoints),
        min_cacheable_tokens: Some(min_cacheable_tokens),
        cache_economics: Some(CacheEconomics {
            write_short_multiplier: 1.25,
            write_long_multiplier: Some(2.0),
            read_multiplier: 0.1,
        }),
    }
}

fn prompt_ir_with_token_counts(token_counts: &[u32]) -> PromptIR {
    PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: token_counts
            .iter()
            .enumerate()
            .map(|(index, token_count)| PromptBlock {
                span_id: SpanId(format!("block-{index}")),
                sequence_index: index as u32,
                role: if index == 0 {
                    PromptRole::System
                } else {
                    PromptRole::User
                },
                content: "x".repeat((*token_count as usize) * 4),
                content_type: BlockContentType::Text,
                provenance: if index == 0 {
                    ProvenanceLabel::System
                } else {
                    ProvenanceLabel::User
                },
                sensitivity: SensitivityLabel::Public,
                token_metadata: Some(TokenizationMetadata {
                    model_family: "claude".to_string(),
                    token_count: *token_count,
                }),
            })
            .collect(),
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    }
}

fn stability_result(scores: &[(f64, f64)], observation_count: u32) -> StabilityAnalysisResult {
    StabilityAnalysisResult {
        scores: scores
            .iter()
            .enumerate()
            .map(|(index, (score, confidence))| BlockStabilityScore {
                span_id: SpanId(format!("block-{index}")),
                classification: StabilityClass::Stable,
                score: *score,
                confidence: *confidence,
                observation_count,
            })
            .collect(),
        stable_prefix_length: scores.len(),
        stable_prefix_fingerprint: None,
        total_observations: observation_count,
    }
}

#[test]
fn economics_policy_returns_no_breakpoints_when_expected_savings_are_non_positive() {
    let prompt_ir = prompt_ir_with_token_counts(&[1800]);
    let stability = stability_result(&[(1.0, 1.0)], 1);

    let plan = plan_breakpoints(&prompt_ir, &stability, 1, &model_capabilities(4, 1024));

    assert!(
        plan.planned_breakpoints.is_empty(),
        "planner must fail open when it cannot prove positive net savings"
    );
}

#[test]
fn economics_policy_returns_no_breakpoints_without_provider_economics() {
    let prompt_ir = prompt_ir_with_token_counts(&[1800]);
    let stability = stability_result(&[(1.0, 1.0)], 3);
    let mut capabilities = model_capabilities(4, 1024);
    capabilities.cache_economics = None;

    let plan = plan_breakpoints(&prompt_ir, &stability, 3, &capabilities);

    assert!(
        plan.planned_breakpoints.is_empty(),
        "planner must fail open when the provider/plugin does not supply cache economics"
    );
}

#[test]
fn economics_policy_returns_exactly_one_breakpoint_when_only_the_earliest_boundary_pays_back() {
    let prompt_ir = prompt_ir_with_token_counts(&[1600, 1200]);
    let stability = stability_result(&[(1.0, 1.0), (0.95, 0.05)], 4);

    let plan = plan_breakpoints(&prompt_ir, &stability, 4, &model_capabilities(4, 1024));

    assert_eq!(plan.planned_breakpoints.len(), 1);
    assert_eq!(plan.planned_breakpoints[0].stable_prefix_end, 1);
    assert!(plan.planned_breakpoints[0].expected_net_savings > 0.0);
}

#[test]
fn economics_policy_keeps_all_planned_breakpoints_session_scoped() {
    let prompt_ir = prompt_ir_with_token_counts(&[1400, 1500, 1600]);
    let stability = stability_result(&[(1.0, 1.0), (1.0, 0.8), (0.99, 0.7)], 5);

    let plan = plan_breakpoints(&prompt_ir, &stability, 5, &model_capabilities(4, 1024));

    assert!(!plan.planned_breakpoints.is_empty());
    assert!(
        plan.planned_breakpoints
            .iter()
            .all(|breakpoint| breakpoint.scope == SharingScope::Session)
    );
}
