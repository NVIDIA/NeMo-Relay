// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stability analysis for prompt blocks across multiple observations.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::acg::canonicalize::sha256_hex;
use crate::acg::profile::{BlockStabilityScore, StabilityClass};
use crate::acg::prompt_ir::{PromptBlock, PromptIR, SpanId};

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
/// The result must not depend on the per-process seed `s` that Rust draws for
/// [`HashMap`] iteration, because a profile persisted by one process is read
/// back by another. Over `N` processes producing outcomes with multiplicities
/// `c_1..c_k`, the cross-process agreement rate is
///
/// ```text
/// A = sum_i c_i * (c_i - 1) / (N * (N - 1))
/// ```
///
/// the probability that two independent processes agree, and equivalently the
/// probability that a persisted profile satisfies the reuse gate elsewhere.
/// `A = 1` exactly when the analysis is seed-independent, which is what
/// [`canonical_scores`] and [`dominant_prefix_length`] each establish.
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

    let indexed_scores: Vec<(u32, BlockStabilityScore)> = span_map
        .into_iter()
        .map(|(span_id, obs)| build_stability_score(span_id, obs, total_observations, thresholds))
        .collect();

    let scores = canonical_scores(indexed_scores);
    let stable_prefix_length = dominant_prefix_length(observations, thresholds.stable_threshold);
    let stable_prefix_fingerprint =
        dominant_fingerprint(observations.iter().filter_map(|observation| {
            prompt_prefix_fingerprint(observation, stable_prefix_length)
        }));

    StabilityAnalysisResult {
        scores,
        stable_prefix_length,
        stable_prefix_fingerprint,
        total_observations,
    }
}

/// Fingerprint the leading `prefix_length` blocks of a single observation.
///
/// The digest covers each block's span id, role, content type, and normalized
/// content, so any reordering or edit inside the prefix changes the result.
///
/// # Parameters
/// - `observation`: Normalized prompt IR to fingerprint.
/// - `prefix_length`: Number of leading blocks to include.
///
/// # Returns
/// The prefix digest, or [`None`] when `prefix_length` is zero or exceeds the
/// number of blocks in the observation.
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
        .map(block_key)
        .collect::<Option<Vec<_>>>()?
        .join("\n");
    Some(sha256_hex(&prefix))
}

/// Identify a block by span, role, content type, and normalized content.
///
/// The prefix length rule and the prefix fingerprint must agree on when two
/// blocks are the same block, so both read this one definition.
fn block_key(block: &PromptBlock) -> Option<String> {
    serde_json::to_string(&(
        &block.span_id,
        block.role,
        block.content_type,
        &block.content,
    ))
    .ok()
}

/// Length of the longest exact block prefix shared by a dominant share of the
/// observation window.
///
/// Under `d(x, y) = 2^-lcp(x, y)` the window is ultrametric: closed balls are
/// the sets agreeing on a prefix, and every point of a ball is a center, so a
/// ball is fixed by its members rather than by traversal order. Descent also
/// requires a strict majority, which two disjoint children cannot both hold, so
/// the dominant child is unique where it exists and no tie-break is reachable.
/// Where none dominates, the prefix stops.
///
/// # Parameters
/// - `observations`: Prompt observations to compare.
/// - `threshold`: Share of the window that must share the prefix.
///
/// # Returns
/// The number of leading blocks shared by the dominant share of the window.
fn dominant_prefix_length(observations: &[PromptIR], threshold: f64) -> usize {
    let total = observations.len();
    let required = ((threshold * total as f64).ceil() as usize).clamp(total / 2 + 1, total.max(1));
    let mut members: Vec<&PromptIR> = observations.iter().collect();

    for depth in 0.. {
        let mut children: HashMap<String, Vec<&PromptIR>> = HashMap::new();
        for observation in &members {
            // An observation that ends here joins no child, so its mass stops
            // contributing and the descent terminates at the longest window.
            if let Some(key) = observation.blocks.get(depth).and_then(block_key) {
                children.entry(key).or_default().push(observation);
            }
        }
        let Some(dominant) = children.into_values().find(|child| child.len() >= required) else {
            return depth;
        };
        members = dominant;
    }

    unreachable!("descent returns at the first depth where no child dominates")
}

/// Fingerprint an observation's stable prefix as stored against a learning key.
///
/// The digest binds the prefix to its learning key so one profile's stable
/// scaffold can never satisfy the reuse gate for another. A prefix that stops
/// inside the leading scaffold (system, tool-schema, and structured-output
/// blocks) is bound only to that scaffold and therefore stays reusable across
/// turns. A prefix that extends past the scaffold is bound to the complete
/// source request instead, because the normalized IR does not retain a lossless
/// provider-prefix representation and a looser binding could admit a request
/// whose real provider prefix differs.
///
/// # Parameters
/// - `observation`: Normalized prompt IR to fingerprint.
/// - `prefix_length`: Stable prefix length reported by stability analysis.
/// - `learning_key`: Learning key the observation is bucketed under.
///
/// # Returns
/// The bound digest, or [`None`] when the prefix cannot be fingerprinted or
/// when a beyond-scaffold prefix has no recorded source request hash.
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
    let request_fingerprint = if prefix_length > scaffold_length {
        observation.source_request_hash.as_deref()?
    } else {
        "stable-scaffold"
    };

    Some(sha256_hex(
        &[learning_key, &prefix_fingerprint, request_fingerprint].join("\n"),
    ))
}

/// Pick the most frequent profile prefix fingerprint across an observation window.
///
/// Ties break on the lexicographically smallest digest so the same window always
/// yields the same fingerprint regardless of iteration order.
///
/// # Parameters
/// - `observations`: Observation window persisted for the learning key.
/// - `prefix_length`: Stable prefix length reported by stability analysis.
/// - `learning_key`: Learning key the window is bucketed under.
///
/// # Returns
/// The dominant digest, or [`None`] when no observation can be fingerprinted.
pub(crate) fn dominant_profile_prefix_fingerprint(
    observations: &[PromptIR],
    prefix_length: usize,
    learning_key: &str,
) -> Option<String> {
    dominant_fingerprint(observations.iter().filter_map(|observation| {
        profile_prefix_fingerprint(observation, prefix_length, learning_key)
    }))
}

/// Pick the most frequent digest, breaking ties on the smallest digest.
///
/// Counting into a [`HashMap`] loses the input order, so the winner is selected
/// by a total order over `(count, digest)` rather than by iteration position.
/// Digests are unique map keys, so that order has exactly one maximum.
fn dominant_fingerprint(fingerprints: impl Iterator<Item = String>) -> Option<String> {
    fingerprints
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

/// Rank a classification so the least stable span sorts first.
fn stability_rank(classification: StabilityClass) -> u8 {
    match classification {
        StabilityClass::Variable => 0,
        StabilityClass::SemiStable => 1,
        StabilityClass::Stable => 2,
    }
}

/// Order span scores into the one canonical sequence for a given span set.
///
/// The sequence index alone does not order the set: span ids carry the role and
/// the tool suffix, so `assistant-3-search` and `assistant-3-fetch` share index
/// 3, and the index is a minimum across observations. Tied spans would keep the
/// per-process order [`HashMap`] iteration produced. Extending the key to
/// `(index, stability rank, span id)` makes it injective, because span ids are
/// unique within one analysis, and a total order admits one sorted sequence.
///
/// Prefix economics walks this vector and stops pricing at the first span that
/// is not stable, so ranking the least stable first ends the priced prefix at a
/// contested position rather than pricing across it.
///
/// # Parameters
/// - `indexed`: Span scores paired with the sequence index each span was first
///   seen at, in [`HashMap`] iteration order.
///
/// # Returns
/// The scores in prompt order.
fn canonical_scores(mut indexed: Vec<(u32, BlockStabilityScore)>) -> Vec<BlockStabilityScore> {
    indexed.sort_by(|(left_index, left), (right_index, right)| {
        left_index
            .cmp(right_index)
            .then_with(|| {
                stability_rank(left.classification).cmp(&stability_rank(right.classification))
            })
            .then_with(|| left.span_id.0.cmp(&right.span_id.0))
    });
    indexed.into_iter().map(|(_, score)| score).collect()
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

#[cfg(test)]
#[path = "../../tests/unit/acg/stability_internal_tests.rs"]
mod tests;
