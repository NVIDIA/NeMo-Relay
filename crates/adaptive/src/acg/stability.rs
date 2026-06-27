// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stability analysis for prompt blocks across multiple observations.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::acg::canonicalize::sha256_hex;
use crate::acg::profile::{BlockStabilityScore, StabilityClass};
use crate::acg::prompt_ir::{BlockContentType, PromptBlock, PromptIR, PromptRole, SpanId};

/// Thresholds controlling prompt-block stability classification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StabilityThresholds {
    /// Minimum effective score required for a block to be classified as stable.
    pub stable_threshold: f64,
    /// Minimum effective score required for a block to be classified as semi-stable.
    pub semi_stable_threshold: f64,
    /// Observation count required to reach full confidence.
    pub min_observations_for_full_confidence: u32,
}

impl Default for StabilityThresholds {
    fn default() -> Self {
        Self {
            stable_threshold: 0.95,
            semi_stable_threshold: 0.50,
            min_observations_for_full_confidence: 20,
        }
    }
}

/// Result of analyzing prompt stability across a set of observations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StabilityAnalysisResult {
    /// Stability score for each distinct prompt span.
    pub scores: Vec<BlockStabilityScore>,
    /// Number of leading blocks that were classified as stable.
    pub stable_prefix_length: usize,
    /// Fingerprint of the dominant observed stable prefix content.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub stable_prefix_fingerprint: Option<String>,
    /// Total number of observations included in the analysis.
    pub total_observations: u32,
    /// Whether topological convergence has been declared for this profile.
    #[serde(default)]
    pub converged: bool,
}

struct SpanObservations {
    hash_counts: HashMap<String, u32>,
    present_count: u32,
    first_seen_sequence_index: u32,
}

/// Analyze prompt-block stability across multiple observations.
///
/// The analysis computes one stability score per span, ordered by the first
/// sequence index at which that span appeared, and derives the length of the
/// stable prefix at the start of the prompt.
///
/// # Parameters
/// - `observations`: Prompt observations to compare.
/// - `thresholds`: Thresholds used for stability classification and confidence.
///
/// # Returns
/// A [`StabilityAnalysisResult`] summarizing span-level stability.
pub fn analyze_stability(
    observations: &[PromptIR],
    thresholds: &StabilityThresholds,
) -> StabilityAnalysisResult {
    if observations.is_empty() {
        return StabilityAnalysisResult {
            scores: Vec::new(),
            stable_prefix_length: 0,
            stable_prefix_fingerprint: None,
            total_observations: 0,
            converged: false,
        };
    }

    let total_observations = observations.len() as u32;
    let mut span_map: HashMap<SpanId, SpanObservations> = HashMap::new();

    for observation in observations {
        record_observation(observation, &mut span_map);
    }

    let mut indexed_scores: Vec<(u32, BlockStabilityScore)> = span_map
        .into_iter()
        .map(|(span_id, obs)| build_stability_score(span_id, obs, total_observations, thresholds))
        .collect();

    sort_indexed_scores(&mut indexed_scores);
    let scores: Vec<BlockStabilityScore> =
        indexed_scores.into_iter().map(|(_, score)| score).collect();
    let stable_prefix_length = find_stable_prefix_length(&scores);
    let stable_prefix_fingerprint =
        dominant_stable_prefix_fingerprint(observations, stable_prefix_length);

    StabilityAnalysisResult {
        scores,
        stable_prefix_length,
        stable_prefix_fingerprint,
        total_observations,
        converged: false,
    }
}

pub(crate) fn prompt_prefix_fingerprint(
    observation: &PromptIR,
    prefix_length: usize,
) -> Option<String> {
    if prefix_length == 0 || observation.blocks.len() < prefix_length {
        return None;
    }

    let prefix = observation
        .blocks
        .iter()
        .take(prefix_length)
        .map(block_fingerprint_part)
        .collect::<Vec<_>>()
        .join("\n");
    Some(sha256_hex(&prefix))
}

fn dominant_stable_prefix_fingerprint(
    observations: &[PromptIR],
    stable_prefix_length: usize,
) -> Option<String> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    for observation in observations {
        let Some(fingerprint) = prompt_prefix_fingerprint(observation, stable_prefix_length) else {
            continue;
        };
        *counts.entry(fingerprint).or_insert(0) += 1;
    }

    select_dominant_prefix_fingerprint(counts)
}

fn select_dominant_prefix_fingerprint(counts: HashMap<String, u32>) -> Option<String> {
    counts
        .into_iter()
        .fold(None, |best, candidate| match best {
            None => Some(candidate),
            Some((best_fingerprint, best_count)) => {
                let (candidate_fingerprint, candidate_count) = candidate;
                if candidate_count > best_count
                    || (candidate_count == best_count && candidate_fingerprint < best_fingerprint)
                {
                    Some((candidate_fingerprint, candidate_count))
                } else {
                    Some((best_fingerprint, best_count))
                }
            }
        })
        .map(|(fingerprint, _)| fingerprint)
}

fn sort_indexed_scores(indexed_scores: &mut [(u32, BlockStabilityScore)]) {
    indexed_scores.sort_by(|(left_index, left_score), (right_index, right_score)| {
        left_index
            .cmp(right_index)
            .then_with(|| left_score.span_id.0.cmp(&right_score.span_id.0))
    });
}

fn block_fingerprint_part(block: &PromptBlock) -> String {
    format!(
        "{}|{}|{}|{}",
        block.span_id.0,
        prompt_role_tag(block.role),
        content_type_tag(block.content_type),
        sha256_hex(&block.content)
    )
}

fn prompt_role_tag(role: PromptRole) -> &'static str {
    match role {
        PromptRole::System => "system",
        PromptRole::User => "user",
        PromptRole::Assistant => "assistant",
        PromptRole::Tool => "tool",
    }
}

fn content_type_tag(content_type: BlockContentType) -> &'static str {
    match content_type {
        BlockContentType::Text => "text",
        BlockContentType::ToolSchema => "tool_schema",
        BlockContentType::ToolResult => "tool_result",
        BlockContentType::StructuredOutput => "structured_output",
        BlockContentType::Image => "image",
    }
}

fn record_observation(observation: &PromptIR, span_map: &mut HashMap<SpanId, SpanObservations>) {
    let mut seen_in_observation: HashSet<SpanId> = HashSet::new();

    for block in &observation.blocks {
        record_block_observation(block, span_map);
        seen_in_observation.insert(block.span_id.clone());
    }

    increment_present_counts(span_map, &seen_in_observation);
}

fn record_block_observation(
    block: &crate::acg::prompt_ir::PromptBlock,
    span_map: &mut HashMap<SpanId, SpanObservations>,
) {
    let hash = sha256_hex(&block.content);
    let entry = span_map
        .entry(block.span_id.clone())
        .or_insert_with(|| SpanObservations {
            hash_counts: HashMap::new(),
            present_count: 0,
            first_seen_sequence_index: block.sequence_index,
        });

    *entry.hash_counts.entry(hash).or_insert(0) += 1;
    entry.first_seen_sequence_index = entry.first_seen_sequence_index.min(block.sequence_index);
}

fn increment_present_counts(
    span_map: &mut HashMap<SpanId, SpanObservations>,
    seen_in_observation: &HashSet<SpanId>,
) {
    for span_id in seen_in_observation {
        if let Some(entry) = span_map.get_mut(span_id) {
            entry.present_count += 1;
        }
    }
}

fn build_stability_score(
    span_id: SpanId,
    observations: SpanObservations,
    total_observations: u32,
    thresholds: &StabilityThresholds,
) -> (u32, BlockStabilityScore) {
    let effective_score = effective_stability_score(&observations, total_observations);
    let classification = classify_stability(effective_score, thresholds);
    let confidence = stability_confidence(observations.present_count, thresholds);

    (
        observations.first_seen_sequence_index,
        BlockStabilityScore {
            span_id,
            classification,
            score: effective_score,
            confidence,
            observation_count: observations.present_count,
        },
    )
}

fn effective_stability_score(observations: &SpanObservations, total_observations: u32) -> f64 {
    let max_hash_count = observations
        .hash_counts
        .values()
        .max()
        .copied()
        .unwrap_or(0);
    let presence_rate = observations.present_count as f64 / total_observations as f64;
    let dominant_fraction = if observations.present_count == 0 {
        0.0
    } else {
        max_hash_count as f64 / observations.present_count as f64
    };

    presence_rate * dominant_fraction
}

fn classify_stability(effective_score: f64, thresholds: &StabilityThresholds) -> StabilityClass {
    if effective_score >= thresholds.stable_threshold {
        StabilityClass::Stable
    } else if effective_score >= thresholds.semi_stable_threshold {
        StabilityClass::SemiStable
    } else {
        StabilityClass::Variable
    }
}

fn stability_confidence(present_count: u32, thresholds: &StabilityThresholds) -> f64 {
    if thresholds.min_observations_for_full_confidence == 0 {
        return 1.0;
    }

    (present_count as f64 / thresholds.min_observations_for_full_confidence as f64).min(1.0)
}

/// Count the number of leading scores classified as stable.
///
/// # Parameters
/// - `scores`: Span-level stability scores in prompt order.
///
/// # Returns
/// The number of consecutive leading entries whose classification is
/// [`StabilityClass::Stable`].
pub fn find_stable_prefix_length(scores: &[BlockStabilityScore]) -> usize {
    scores
        .iter()
        .take_while(|score| score.classification == StabilityClass::Stable)
        .count()
}

#[cfg(test)]
#[path = "../../tests/unit/acg/stability_internal_tests.rs"]
mod tests;
