// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for multi breakpoint in the NeMo Relay adaptive crate.

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

fn layered_prompt_ir(token_counts: &[u32]) -> PromptIR {
    PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: token_counts
            .iter()
            .enumerate()
            .map(|(index, token_count)| PromptBlock {
                span_id: SpanId(format!("layer-{index}")),
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

fn layered_stability(observation_count: u32, layers: usize) -> StabilityAnalysisResult {
    StabilityAnalysisResult {
        scores: (0..layers)
            .map(|index| BlockStabilityScore {
                span_id: SpanId(format!("layer-{index}")),
                classification: StabilityClass::Stable,
                score: 0.99,
                confidence: 0.9 - ((index as f64) * 0.05),
                observation_count,
            })
            .collect(),
        stable_prefix_length: layers,
        total_observations: observation_count,
        converged: false,
    }
}

#[test]
fn multi_breakpoint_planner_returns_ordered_profitable_boundaries_up_to_model_cap() {
    let prompt_ir = layered_prompt_ir(&[1300, 1300, 1300, 1300, 1300]);
    let stability = layered_stability(6, 5);

    let plan = plan_breakpoints(&prompt_ir, &stability, 6, &model_capabilities(3, 1024));

    assert_eq!(plan.planned_breakpoints.len(), 3);
    let selected_prefix_ends = plan
        .planned_breakpoints
        .iter()
        .map(|breakpoint| breakpoint.stable_prefix_end)
        .collect::<Vec<_>>();
    assert_eq!(selected_prefix_ends, vec![2, 4, 5]);
    assert!(
        selected_prefix_ends
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    );
    assert!(
        plan.planned_breakpoints
            .iter()
            .all(|breakpoint| breakpoint.expected_net_savings > 0.0)
    );
}
