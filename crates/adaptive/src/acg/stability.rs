// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stability analysis for prompt blocks across multiple observations.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::acg::canonicalize::sha256_hex;
use crate::acg::profile::{BlockStabilityScore, StabilityClass};
use crate::acg::prompt_ir::{PromptIR, SpanId};

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
    /// Fingerprint of the dominant stable prefix. Generic analysis hashes the
    /// prompt IR; ACG profile persistence additionally binds it to the learning
    /// key and, beyond the leading scaffold, the full source request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stable_prefix_fingerprint: Option<String>,
    /// Total number of observations included in the analysis.
    pub total_observations: u32,
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

    indexed_scores.sort_by_key(|(idx, _)| *idx);
    let scores: Vec<BlockStabilityScore> =
        indexed_scores.into_iter().map(|(_, score)| score).collect();
    let stable_prefix_length = find_stable_prefix_length(&scores);
    let stable_prefix_fingerprint = observations
        .iter()
        .filter_map(|observation| prompt_prefix_fingerprint(observation, stable_prefix_length))
        .fold(HashMap::new(), |mut counts, fingerprint| {
            *counts.entry(fingerprint).or_insert(0_u32) += 1;
            counts
        })
        .into_iter()
        .max_by(|(left_hash, left_count), (right_hash, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| right_hash.cmp(left_hash))
        })
        .map(|(fingerprint, _)| fingerprint);

    StabilityAnalysisResult {
        scores,
        stable_prefix_length,
        stable_prefix_fingerprint,
        total_observations,
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
        .map(|block| {
            serde_json::to_string(&(
                &block.span_id,
                block.role,
                block.content_type,
                &block.content,
            ))
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .ok()?
        .join("\n");
    Some(sha256_hex(&prefix))
}

pub(crate) fn profile_prefix_fingerprint(
    observation: &PromptIR,
    prefix_length: usize,
    learning_key: &str,
) -> Option<String> {
    let prefix_fingerprint = prompt_prefix_fingerprint(observation, prefix_length)?;
    let scaffold_length = observation
        .blocks
        .iter()
        .take_while(|block| {
            block.role == crate::acg::prompt_ir::PromptRole::System
                || matches!(
                    block.content_type,
                    crate::acg::prompt_ir::BlockContentType::ToolSchema
                        | crate::acg::prompt_ir::BlockContentType::StructuredOutput
                )
        })
        .count();
    // Conservatively bind deeper prefixes to the complete request because the
    // normalized IR does not retain a lossless provider-prefix representation.
    let request_fingerprint = (prefix_length > scaffold_length)
        .then_some(observation.source_request_hash.as_deref())
        .flatten();

    Some(sha256_hex(
        &[
            learning_key,
            &prefix_fingerprint,
            request_fingerprint.unwrap_or("stable-scaffold"),
        ]
        .join("\n"),
    ))
}

pub(crate) fn dominant_profile_prefix_fingerprint(
    observations: &[PromptIR],
    prefix_length: usize,
    learning_key: &str,
) -> Option<String> {
    observations
        .iter()
        .filter_map(|observation| {
            profile_prefix_fingerprint(observation, prefix_length, learning_key)
        })
        .fold(HashMap::new(), |mut counts, fingerprint| {
            *counts.entry(fingerprint).or_insert(0_u32) += 1;
            counts
        })
        .into_iter()
        .max_by(|(left_hash, left_count), (right_hash, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| right_hash.cmp(left_hash))
        })
        .map(|(fingerprint, _)| fingerprint)
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
