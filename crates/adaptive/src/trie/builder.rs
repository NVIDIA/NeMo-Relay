// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prediction trie builder with incremental accumulator merge.
//!
//! Ports the core algorithm from NAT's `trie_builder.py`: extract LLM call
//! contexts from run records, compute 4-signal sensitivity scores with
//! min-max normalization, update streaming accumulators at every trie node
//! along the path, and build the final [`PredictionTrieNode`] tree.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::accumulator::{AccumulatorState, NodeAccumulators, RunningStats};
use super::data_models::{LlmCallPrediction, PredictionTrieNode};
use crate::types::records::{CallKind, CallRecord, RunRecord};

/// Configuration for auto-sensitivity scoring.
///
/// Weights and scale match NAT defaults from trie_builder.py lines 41-48.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityConfig {
    /// Integer scale for quantized sensitivity (1..=scale).
    pub sensitivity_scale: u32,
    /// Weight for the critical-path signal.
    pub w_critical: f64,
    /// Weight for the fan-out signal.
    pub w_fanout: f64,
    /// Weight for the U-shaped position signal.
    pub w_position: f64,
    /// Weight for the parallel-penalty signal.
    pub w_parallel: f64,
}

impl Default for SensitivityConfig {
    fn default() -> Self {
        Self {
            sensitivity_scale: 5,
            w_critical: 0.5,
            w_fanout: 0.3,
            w_position: 0.2,
            w_parallel: 0.0,
        }
    }
}

/// Internal context for a single LLM call extracted from a [`RunRecord`].
#[derive(Debug, Clone)]
pub(crate) struct LlmCallContext {
    pub path: Vec<String>,
    pub call_index: u32,
    pub remaining_calls: u32,
    pub time_to_next_ms: Option<f64>,
    pub output_tokens: u32,
    pub call_duration_s: f64,
    pub workflow_duration_s: f64,
    pub parallel_slack_ratio: f64,
    pub sensitivity_score: f64,
    pub span_start_time: f64,
    pub span_end_time: f64,
}

/// Builds a [`PredictionTrieNode`] tree from [`RunRecord`]s via incremental
/// accumulator merge.
///
/// # Usage
///
/// ```ignore
/// let mut builder = PredictionTrieBuilder::new(Some(SensitivityConfig::default()));
/// builder.add_run(&run1);
/// builder.add_run(&run2);
/// let trie = builder.build();
/// ```
pub struct PredictionTrieBuilder {
    accumulators: AccumulatorState,
    sensitivity_config: Option<SensitivityConfig>,
}

impl PredictionTrieBuilder {
    /// Creates a new builder with optional sensitivity scoring.
    pub fn new(sensitivity_config: Option<SensitivityConfig>) -> Self {
        Self {
            accumulators: AccumulatorState::default(),
            sensitivity_config,
        }
    }

    /// Creates a builder seeded with pre-existing accumulators.
    ///
    /// Used by the learner pipeline to resume incremental learning
    /// from a stored [`AccumulatorState`].
    pub fn with_accumulators(
        accumulators: AccumulatorState,
        sensitivity_config: Option<SensitivityConfig>,
    ) -> Self {
        Self {
            accumulators,
            sensitivity_config,
        }
    }

    /// Processes a single [`RunRecord`] and updates accumulators.
    ///
    /// Extracts LLM call contexts, optionally computes sensitivity scores,
    /// and updates accumulators at every node along each call's path.
    pub fn add_run(&mut self, run: &RunRecord) {
        let mut contexts = extract_llm_contexts(run);
        if let Some(ref config) = self.sensitivity_config {
            compute_sensitivity_scores(&mut contexts, config);
        }
        for ctx in &contexts {
            self.update_accumulators(ctx);
        }
    }

    /// Constructs the prediction trie from accumulated data.
    ///
    /// Iterates all accumulated nodes, navigates/creates the trie path,
    /// and populates predictions from the accumulators.
    pub fn build(&self) -> PredictionTrieNode {
        let mut root = PredictionTrieNode::new("root");

        for (path_key, node_accs) in &self.accumulators.nodes {
            let node = get_or_create_node(&mut root, path_key);
            populate_node_predictions(node, node_accs, &self.sensitivity_config);
        }

        root
    }

    /// Returns a reference to the underlying accumulator state.
    pub fn accumulators(&self) -> &AccumulatorState {
        &self.accumulators
    }

    /// Updates accumulators at root + each ancestor + leaf for a given context.
    fn update_accumulators(&mut self, ctx: &LlmCallContext) {
        let has_sensitivity = self.sensitivity_config.is_some();

        // Update root node (key = "")
        let root_accs = self.accumulators.nodes.entry(String::new()).or_default();
        add_to_accumulators(root_accs, ctx, has_sensitivity);

        // Update each node along the path
        for i in 0..ctx.path.len() {
            let path_key = ctx.path[..=i].join("/");
            let node_accs = self.accumulators.nodes.entry(path_key).or_default();
            add_to_accumulators(node_accs, ctx, has_sensitivity);
        }
    }
}

/// Extracts [`LlmCallContext`]s from a [`RunRecord`].
///
/// Port of NAT's `_extract_llm_contexts` adapted for `RunRecord`/`CallRecord`.
/// Only completed LLM calls (with `ended_at`) are extracted.
fn extract_llm_contexts(run: &RunRecord) -> Vec<LlmCallContext> {
    // Compute workflow duration
    let workflow_duration_s = if let Some(end) = run.ended_at {
        (end - run.started_at).num_milliseconds() as f64 / 1000.0
    } else {
        // Fall back to last call ended_at
        run.calls
            .iter()
            .filter_map(|c| c.ended_at)
            .max()
            .map(|end| (end - run.started_at).num_milliseconds() as f64 / 1000.0)
            .unwrap_or(0.0)
    };

    // Collect completed LLM calls with their original indices
    let llm_calls: Vec<(usize, &CallRecord)> = run
        .calls
        .iter()
        .enumerate()
        .filter(|(_, c)| c.kind == CallKind::Llm && c.ended_at.is_some())
        .collect();

    let total_llm = llm_calls.len();

    // Track call_index per parent key (for Phase 4, parent = call name)
    let mut call_counts: HashMap<String, u32> = HashMap::new();

    let mut contexts = Vec::with_capacity(total_llm);

    for (llm_pos, (orig_idx, call)) in llm_calls.iter().enumerate() {
        let ended_at = call.ended_at.expect("filtered to completed calls");

        // Path: Phase 4 simplification -- single-element vec with call name
        let path = vec![call.name.clone()];

        // Call index per parent
        let counter = call_counts.entry(call.name.clone()).or_insert(0);
        *counter += 1;
        let call_index = *counter;

        // Remaining calls
        let remaining_calls = (total_llm - llm_pos - 1) as u32;

        // Time to next LLM start: scan forward in ALL calls to find next LLM start
        let time_to_next_ms = run
            .calls
            .iter()
            .skip(orig_idx + 1)
            .find(|c| c.kind == CallKind::Llm)
            .map(|next_llm| {
                next_llm
                    .started_at
                    .signed_duration_since(ended_at)
                    .num_milliseconds() as f64
            });

        // Output tokens
        let output_tokens = call.output_tokens.unwrap_or(0);

        // Call duration
        let call_duration_s = (ended_at - call.started_at).num_milliseconds() as f64 / 1000.0;

        // Span timestamps
        let span_start_time = call.started_at.timestamp() as f64;
        let span_end_time = ended_at.timestamp() as f64;

        contexts.push(LlmCallContext {
            path,
            call_index,
            remaining_calls,
            time_to_next_ms,
            output_tokens,
            call_duration_s,
            workflow_duration_s,
            parallel_slack_ratio: 0.0,
            sensitivity_score: 0.0,
            span_start_time,
            span_end_time,
        });
    }

    contexts
}

/// Computes composite sensitivity scores for each call in a trace.
///
/// Direct port of NAT trie_builder.py lines 186-272: four weighted signals
/// (critical path, fan-out, position, parallel penalty) with min-max
/// normalization across the trace.
fn compute_sensitivity_scores(contexts: &mut [LlmCallContext], config: &SensitivityConfig) {
    if contexts.is_empty() {
        return;
    }

    // Compute logical positions that collapse parallel siblings
    let logical_positions = compute_logical_positions(contexts);
    let num_logical_steps = logical_positions
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(1);

    let max_logical_remaining = if num_logical_steps > 0 {
        num_logical_steps - 1
    } else {
        0
    };

    // Count group sizes per logical position
    let mut group_sizes: HashMap<usize, usize> = HashMap::new();
    for &lpos in &logical_positions {
        *group_sizes.entry(lpos).or_insert(0) += 1;
    }

    let mut raw_scores: Vec<f64> = Vec::with_capacity(contexts.len());

    for (i, ctx) in contexts.iter().enumerate() {
        let lpos = logical_positions[i];

        // Signal 1: Critical path weight
        let critical_path_weight = if ctx.workflow_duration_s > 0.0 {
            (ctx.call_duration_s / ctx.workflow_duration_s).min(1.0)
        } else {
            1.0
        };

        // Signal 2: Fan-out score (based on logical remaining steps)
        let logical_remaining = max_logical_remaining.saturating_sub(lpos);
        let fanout_score = if max_logical_remaining > 0 {
            logical_remaining as f64 / max_logical_remaining as f64
        } else {
            0.0
        };

        // Signal 3: Position score (U-shaped, based on logical position)
        let position_score = if num_logical_steps > 1 {
            let normalized_pos = lpos as f64 / (num_logical_steps - 1) as f64;
            (1.0 - normalized_pos).max(normalized_pos)
        } else {
            1.0
        };

        // Parallel penalty
        let mut parallel_penalty = ctx.parallel_slack_ratio;
        let gs = group_sizes.get(&lpos).copied().unwrap_or(1);
        if gs > 1 {
            let group_penalty = (gs - 1) as f64 / gs as f64;
            parallel_penalty = (parallel_penalty + group_penalty) / 2.0;
        }

        let score = config.w_critical * critical_path_weight
            + config.w_fanout * fanout_score
            + config.w_position * position_score
            - config.w_parallel * parallel_penalty;

        raw_scores.push(score);
    }

    // Min-max normalize
    let min_score = raw_scores.iter().copied().fold(f64::INFINITY, f64::min);
    let max_score = raw_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let score_range = max_score - min_score;

    for (ctx, &raw) in contexts.iter_mut().zip(raw_scores.iter()) {
        if score_range > 0.0 {
            ctx.sensitivity_score = (raw - min_score) / score_range;
        } else {
            ctx.sensitivity_score = 0.5;
        }
    }
}

/// Assigns logical positions to calls, collapsing parallel siblings.
///
/// Uses standard interval-merging: contexts sorted by span start time,
/// overlapping intervals get the same group index. Direct port of NAT's
/// `_compute_logical_positions`.
fn compute_logical_positions(contexts: &[LlmCallContext]) -> Vec<usize> {
    if contexts.is_empty() {
        return vec![];
    }

    let n = contexts.len();

    // Sort indices by span_start_time
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        contexts[a]
            .span_start_time
            .partial_cmp(&contexts[b].span_start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut group_assignments = vec![0usize; n];
    let mut current_group = 0usize;
    let mut group_max_end = contexts[sorted_indices[0]].span_end_time;

    group_assignments[sorted_indices[0]] = current_group;

    for &idx in &sorted_indices[1..] {
        if contexts[idx].span_start_time < group_max_end {
            // Overlaps with current group
            group_assignments[idx] = current_group;
            group_max_end = group_max_end.max(contexts[idx].span_end_time);
        } else {
            // New sequential step
            current_group += 1;
            group_assignments[idx] = current_group;
            group_max_end = contexts[idx].span_end_time;
        }
    }

    group_assignments
}

/// Adds context data to a node's accumulators.
///
/// Updates both per-call-index and aggregated (all_*) accumulators.
fn add_to_accumulators(accs: &mut NodeAccumulators, ctx: &LlmCallContext, has_sensitivity: bool) {
    // By call index
    accs.remaining_calls
        .entry(ctx.call_index)
        .or_default()
        .add_sample(ctx.remaining_calls as f64);
    accs.output_tokens
        .entry(ctx.call_index)
        .or_default()
        .add_sample(ctx.output_tokens as f64);
    if let Some(ttm) = ctx.time_to_next_ms {
        accs.interarrival_ms
            .entry(ctx.call_index)
            .or_default()
            .add_sample(ttm);
    }

    // Aggregated across all indices
    accs.all_remaining_calls
        .add_sample(ctx.remaining_calls as f64);
    accs.all_output_tokens.add_sample(ctx.output_tokens as f64);
    if let Some(ttm) = ctx.time_to_next_ms {
        accs.all_interarrival_ms.add_sample(ttm);
    }

    // Sensitivity accumulators
    if has_sensitivity {
        accs.sensitivity
            .entry(ctx.call_index)
            .or_default()
            .add_sample(ctx.sensitivity_score);
        accs.all_sensitivity.add_sample(ctx.sensitivity_score);
    }
}

/// Navigates from root through path segments (split by "/"), creating nodes as needed.
fn get_or_create_node<'a>(
    root: &'a mut PredictionTrieNode,
    path_key: &str,
) -> &'a mut PredictionTrieNode {
    if path_key.is_empty() {
        return root;
    }

    let mut current = root;
    for name in path_key.split('/') {
        current = current
            .children
            .entry(name.to_string())
            .or_insert_with(|| PredictionTrieNode::new(name));
    }
    current
}

/// Populates a trie node's predictions from its accumulators.
fn populate_node_predictions(
    node: &mut PredictionTrieNode,
    accs: &NodeAccumulators,
    sensitivity_config: &Option<SensitivityConfig>,
) {
    // Collect all call indices from all per-index maps
    let mut all_indices: std::collections::HashSet<u32> = std::collections::HashSet::new();
    all_indices.extend(accs.remaining_calls.keys());
    all_indices.extend(accs.interarrival_ms.keys());
    all_indices.extend(accs.output_tokens.keys());

    let scale = sensitivity_config.as_ref().map(|c| c.sensitivity_scale);

    for idx in all_indices {
        let remaining = accs
            .remaining_calls
            .get(&idx)
            .map(|s| s.compute_metrics())
            .unwrap_or_default();
        let interarrival = accs
            .interarrival_ms
            .get(&idx)
            .map(|s| s.compute_metrics())
            .unwrap_or_default();
        let output_tok = accs
            .output_tokens
            .get(&idx)
            .map(|s| s.compute_metrics())
            .unwrap_or_default();
        let sensitivity = match (scale, accs.sensitivity.get(&idx)) {
            (Some(s), Some(acc)) => score_to_sensitivity(acc, s),
            _ => None,
        };

        node.predictions_by_call_index.insert(
            idx,
            LlmCallPrediction {
                remaining_calls: remaining,
                interarrival_ms: interarrival,
                output_tokens: output_tok,
                latency_sensitivity: sensitivity,
            },
        );
    }

    // Aggregated predictions
    if accs.all_remaining_calls.has_samples() {
        let sensitivity = match scale {
            Some(s) if accs.all_sensitivity.has_samples() => {
                score_to_sensitivity(&accs.all_sensitivity, s)
            }
            _ => None,
        };

        node.predictions_any_index = Some(LlmCallPrediction {
            remaining_calls: accs.all_remaining_calls.compute_metrics(),
            interarrival_ms: accs.all_interarrival_ms.compute_metrics(),
            output_tokens: accs.all_output_tokens.compute_metrics(),
            latency_sensitivity: sensitivity,
        });
    }
}

/// Converts accumulated sensitivity scores to a clamped integer on [1, scale].
///
/// Returns `None` if the accumulator has no samples.
fn score_to_sensitivity(acc: &RunningStats, scale: u32) -> Option<u32> {
    if !acc.has_samples() {
        return None;
    }
    let mean_score = acc.compute_metrics().mean;
    let raw = (mean_score * (scale as f64 - 1.0)).round() as i64 + 1;
    Some(raw.clamp(1, scale as i64) as u32)
}

#[cfg(test)]
#[path = "../../tests/unit/trie/builder_tests.rs"]
mod tests;
