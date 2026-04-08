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
use crate::types::{CallKind, RunRecord};

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
    let llm_calls: Vec<(usize, &crate::types::CallRecord)> = run
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
        let ended_at = call.ended_at.unwrap();

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
            .map(|next_llm| (next_llm.started_at - ended_at).num_milliseconds() as f64);

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
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use uuid::Uuid;

    use crate::types::{CallKind, CallRecord, RunRecord};

    /// Helper: create a RunRecord with `llm_count` LLM calls and `tool_count` tool calls
    /// interleaved. Each call is 1s long with a 100ms gap between calls.
    fn make_test_run(llm_count: usize, tool_count: usize) -> RunRecord {
        let base = Utc::now();
        let mut calls = Vec::new();
        let mut offset_ms: i64 = 0;

        let total = llm_count + tool_count;
        let mut llm_placed = 0;
        let mut tool_placed = 0;

        for _ in 0..total {
            // Alternate: place LLM first, then tool, etc.
            let (kind, name, tokens) = if llm_placed < llm_count
                && (tool_placed >= tool_count || llm_placed <= tool_placed)
            {
                llm_placed += 1;
                // Give some calls output_tokens and others None
                let tokens = if llm_placed % 2 == 0 {
                    Some(100 * llm_placed as u32)
                } else {
                    None
                };
                (CallKind::Llm, "gpt-4".to_string(), tokens)
            } else {
                tool_placed += 1;
                (CallKind::Tool, "search".to_string(), None)
            };

            let start = base + Duration::milliseconds(offset_ms);
            let end = start + Duration::seconds(1);
            calls.push(CallRecord {
                kind,
                name,
                started_at: start,
                ended_at: Some(end),
                metadata_snapshot: None,
                output_tokens: tokens,
            });
            offset_ms += 1100; // 1s call + 100ms gap
        }

        let run_end = calls.last().map(|c| c.ended_at.unwrap()).unwrap_or(base);
        RunRecord {
            id: Uuid::new_v4(),
            agent_id: "test-agent".to_string(),
            calls,
            started_at: base,
            ended_at: Some(run_end),
        }
    }

    // -----------------------------------------------------------------------
    // SensitivityConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sensitivity_config_default() {
        let cfg = SensitivityConfig::default();
        assert_eq!(cfg.sensitivity_scale, 5);
        assert!((cfg.w_critical - 0.5).abs() < f64::EPSILON);
        assert!((cfg.w_fanout - 0.3).abs() < f64::EPSILON);
        assert!((cfg.w_position - 0.2).abs() < f64::EPSILON);
        assert!((cfg.w_parallel - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sensitivity_config_serde_roundtrip() {
        let cfg = SensitivityConfig::default();
        let json = serde_json::to_value(&cfg).unwrap();
        let restored: SensitivityConfig = serde_json::from_value(json).unwrap();
        assert_eq!(restored.sensitivity_scale, cfg.sensitivity_scale);
        assert!((restored.w_critical - cfg.w_critical).abs() < f64::EPSILON);
        assert!((restored.w_fanout - cfg.w_fanout).abs() < f64::EPSILON);
        assert!((restored.w_position - cfg.w_position).abs() < f64::EPSILON);
        assert!((restored.w_parallel - cfg.w_parallel).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // extract_llm_contexts tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_llm_contexts_count() {
        let run = make_test_run(3, 2);
        let contexts = extract_llm_contexts(&run);
        assert_eq!(
            contexts.len(),
            3,
            "Should extract exactly 3 LlmCallContexts from 3 LLM + 2 tool calls"
        );
    }

    #[test]
    fn test_extract_llm_contexts_remaining_calls() {
        let run = make_test_run(3, 0);
        let contexts = extract_llm_contexts(&run);
        assert_eq!(contexts[0].remaining_calls, 2);
        assert_eq!(contexts[1].remaining_calls, 1);
        assert_eq!(contexts[2].remaining_calls, 0);
    }

    #[test]
    fn test_extract_llm_contexts_interarrival_ms() {
        // With 3 LLM calls each 1s apart with 100ms gap, interarrival should be ~100ms
        let run = make_test_run(3, 0);
        let contexts = extract_llm_contexts(&run);
        // First call should have time_to_next (gap to next LLM start)
        assert!(contexts[0].time_to_next_ms.is_some());
        let ttm = contexts[0].time_to_next_ms.unwrap();
        assert!(
            (ttm - 100.0).abs() < 1.0,
            "time_to_next_ms should be ~100ms, got {ttm}"
        );
        // Last call should have None (no next LLM)
        assert!(contexts[2].time_to_next_ms.is_none());
    }

    #[test]
    fn test_extract_llm_contexts_output_tokens() {
        let run = make_test_run(3, 0);
        let contexts = extract_llm_contexts(&run);
        // First LLM call (llm_placed=1, odd) has None -> unwrap_or(0) -> 0
        assert_eq!(contexts[0].output_tokens, 0);
        // Second LLM call (llm_placed=2, even) has Some(200) -> 200
        assert_eq!(contexts[1].output_tokens, 200);
        // Third LLM call (llm_placed=3, odd) has None -> 0
        assert_eq!(contexts[2].output_tokens, 0);
    }

    #[test]
    fn test_extract_llm_contexts_path_single_element() {
        let run = make_test_run(1, 0);
        let contexts = extract_llm_contexts(&run);
        assert_eq!(
            contexts[0].path,
            vec!["gpt-4"],
            "Phase 4 simplification: path is single-element vec with call name"
        );
    }

    #[test]
    fn test_extract_llm_contexts_call_duration() {
        let run = make_test_run(1, 0);
        let contexts = extract_llm_contexts(&run);
        assert!(
            (contexts[0].call_duration_s - 1.0).abs() < 0.01,
            "Each call is 1 second, got {}",
            contexts[0].call_duration_s
        );
    }

    #[test]
    fn test_extract_llm_contexts_workflow_duration() {
        let run = make_test_run(3, 0);
        let contexts = extract_llm_contexts(&run);
        // 3 calls: [0..1s], [1.1..2.1s], [2.2..3.2s]
        // workflow_duration = run.ended_at - run.started_at = 3.2s
        let wd = contexts[0].workflow_duration_s;
        assert!(
            (wd - 3.2).abs() < 0.1,
            "Workflow duration should be ~3.2s, got {wd}"
        );
    }

    // -----------------------------------------------------------------------
    // compute_sensitivity_scores tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sensitivity_scores_u_curve() {
        // With default weights (w_critical=0.5, w_fanout=0.3, w_position=0.2),
        // 3 sequential equal-duration calls produce:
        //   - call 0: highest (high fanout + high position)
        //   - call 1: middle
        //   - call 2: lowest (zero fanout outweighs position U)
        //
        // To see the position U-curve dominate, use position-only weights.
        let config = SensitivityConfig {
            sensitivity_scale: 5,
            w_critical: 0.0,
            w_fanout: 0.0,
            w_position: 1.0,
            w_parallel: 0.0,
        };
        let run = make_test_run(3, 0);
        let mut contexts = extract_llm_contexts(&run);
        compute_sensitivity_scores(&mut contexts, &config);

        let first = contexts[0].sensitivity_score;
        let mid = contexts[1].sensitivity_score;
        let last = contexts[2].sensitivity_score;
        assert!(
            first > mid && last > mid,
            "U-curve (position only): first ({first}) and last ({last}) should be > middle ({mid})"
        );

        // Also verify default weights produce monotonically decreasing scores
        // (fanout dominates, first has highest remaining calls)
        let config_default = SensitivityConfig::default();
        let mut contexts2 = extract_llm_contexts(&run);
        compute_sensitivity_scores(&mut contexts2, &config_default);
        let s0 = contexts2[0].sensitivity_score;
        let s1 = contexts2[1].sensitivity_score;
        let s2 = contexts2[2].sensitivity_score;
        assert!(
            s0 > s1 && s1 > s2,
            "Default weights: scores should decrease ({s0} > {s1} > {s2})"
        );
    }

    #[test]
    fn test_sensitivity_scores_single_call() {
        let run = make_test_run(1, 0);
        let mut contexts = extract_llm_contexts(&run);
        let config = SensitivityConfig::default();
        compute_sensitivity_scores(&mut contexts, &config);
        assert!(
            (contexts[0].sensitivity_score - 0.5).abs() < f64::EPSILON,
            "Single call should get 0.5, got {}",
            contexts[0].sensitivity_score
        );
    }

    // -----------------------------------------------------------------------
    // compute_logical_positions tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_logical_positions_overlapping() {
        // Two overlapping spans should be in the same group
        let contexts = vec![
            LlmCallContext {
                path: vec!["a".into()],
                call_index: 1,
                remaining_calls: 1,
                time_to_next_ms: None,
                output_tokens: 0,
                call_duration_s: 1.0,
                workflow_duration_s: 2.0,
                parallel_slack_ratio: 0.0,
                sensitivity_score: 0.0,
                span_start_time: 0.0,
                span_end_time: 2.0,
            },
            LlmCallContext {
                path: vec!["b".into()],
                call_index: 1,
                remaining_calls: 0,
                time_to_next_ms: None,
                output_tokens: 0,
                call_duration_s: 1.0,
                workflow_duration_s: 2.0,
                parallel_slack_ratio: 0.0,
                sensitivity_score: 0.0,
                span_start_time: 1.0,
                span_end_time: 3.0,
            },
        ];
        let positions = compute_logical_positions(&contexts);
        assert_eq!(
            positions[0], positions[1],
            "Overlapping spans should share logical position"
        );
    }

    #[test]
    fn test_logical_positions_sequential() {
        let contexts = vec![
            LlmCallContext {
                path: vec!["a".into()],
                call_index: 1,
                remaining_calls: 1,
                time_to_next_ms: None,
                output_tokens: 0,
                call_duration_s: 1.0,
                workflow_duration_s: 4.0,
                parallel_slack_ratio: 0.0,
                sensitivity_score: 0.0,
                span_start_time: 0.0,
                span_end_time: 1.0,
            },
            LlmCallContext {
                path: vec!["b".into()],
                call_index: 1,
                remaining_calls: 0,
                time_to_next_ms: None,
                output_tokens: 0,
                call_duration_s: 1.0,
                workflow_duration_s: 4.0,
                parallel_slack_ratio: 0.0,
                sensitivity_score: 0.0,
                span_start_time: 2.0,
                span_end_time: 3.0,
            },
        ];
        let positions = compute_logical_positions(&contexts);
        assert_ne!(
            positions[0], positions[1],
            "Non-overlapping spans should have different logical positions"
        );
    }

    // -----------------------------------------------------------------------
    // score_to_sensitivity tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_score_to_sensitivity_zero() {
        let mut acc = RunningStats::new();
        acc.add_sample(0.0);
        assert_eq!(score_to_sensitivity(&acc, 5), Some(1));
    }

    #[test]
    fn test_score_to_sensitivity_one() {
        let mut acc = RunningStats::new();
        acc.add_sample(1.0);
        assert_eq!(score_to_sensitivity(&acc, 5), Some(5));
    }

    #[test]
    fn test_score_to_sensitivity_half() {
        let mut acc = RunningStats::new();
        acc.add_sample(0.5);
        assert_eq!(score_to_sensitivity(&acc, 5), Some(3));
    }

    #[test]
    fn test_score_to_sensitivity_no_samples() {
        let acc = RunningStats::new();
        assert_eq!(score_to_sensitivity(&acc, 5), None);
    }

    // -----------------------------------------------------------------------
    // PredictionTrieBuilder integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_run_updates_accumulators() {
        let mut builder = PredictionTrieBuilder::new(Some(SensitivityConfig::default()));
        let run = make_test_run(3, 2);
        builder.add_run(&run);

        // Root and path nodes should have accumulators
        let accs = builder.accumulators();
        assert!(
            accs.nodes.contains_key(""),
            "Root node accumulators should exist (key='')"
        );
        assert!(
            accs.nodes.contains_key("gpt-4"),
            "Path node 'gpt-4' accumulators should exist"
        );
    }

    #[test]
    fn test_build_produces_trie() {
        let mut builder = PredictionTrieBuilder::new(Some(SensitivityConfig::default()));
        let run = make_test_run(3, 0);
        builder.add_run(&run);
        let trie = builder.build();

        assert_eq!(trie.name, "root");
        assert!(
            trie.children.contains_key("gpt-4"),
            "Trie should have a 'gpt-4' child node"
        );
    }

    #[test]
    fn test_build_empty_produces_empty_root() {
        let builder = PredictionTrieBuilder::new(None);
        let trie = builder.build();
        assert_eq!(trie.name, "root");
        assert!(trie.children.is_empty());
        assert!(trie.predictions_by_call_index.is_empty());
        assert!(trie.predictions_any_index.is_none());
    }

    #[test]
    fn test_two_runs_merge_accumulators() {
        let mut builder = PredictionTrieBuilder::new(Some(SensitivityConfig::default()));
        let run1 = make_test_run(2, 0);
        let run2 = make_test_run(2, 0);
        builder.add_run(&run1);
        builder.add_run(&run2);

        let accs = builder.accumulators();
        let root_accs = &accs.nodes[""];
        // Each run has 2 LLM calls -> root should have 4 samples in all_remaining_calls
        assert_eq!(
            root_accs.all_remaining_calls.count, 4,
            "Two runs of 2 LLM calls should give 4 samples at root"
        );
    }

    #[test]
    fn test_build_predictions_any_index() {
        let mut builder = PredictionTrieBuilder::new(None);
        let run = make_test_run(2, 0);
        builder.add_run(&run);
        let trie = builder.build();

        // Root should have predictions_any_index since aggregated accumulators have data
        assert!(
            trie.predictions_any_index.is_some(),
            "Root should have predictions_any_index"
        );
    }

    #[test]
    fn test_build_latency_sensitivity_with_config() {
        let mut builder = PredictionTrieBuilder::new(Some(SensitivityConfig::default()));
        let run = make_test_run(3, 0);
        builder.add_run(&run);
        let trie = builder.build();

        // With sensitivity config, predictions should have latency_sensitivity
        let root_any = trie.predictions_any_index.as_ref().unwrap();
        assert!(
            root_any.latency_sensitivity.is_some(),
            "With sensitivity config, predictions should have latency_sensitivity"
        );
    }

    #[test]
    fn test_build_latency_sensitivity_without_config() {
        let mut builder = PredictionTrieBuilder::new(None);
        let run = make_test_run(3, 0);
        builder.add_run(&run);
        let trie = builder.build();

        // Without sensitivity config, all latency_sensitivity should be None
        let root_any = trie.predictions_any_index.as_ref().unwrap();
        assert!(
            root_any.latency_sensitivity.is_none(),
            "Without sensitivity config, latency_sensitivity should be None"
        );

        // Also check per-index predictions
        for pred in trie.predictions_by_call_index.values() {
            assert!(
                pred.latency_sensitivity.is_none(),
                "Without config, per-index latency_sensitivity should be None"
            );
        }
    }

    // -----------------------------------------------------------------------
    // with_accumulators tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_with_accumulators_empty_same_as_new() {
        let config = Some(SensitivityConfig::default());
        let builder_new = PredictionTrieBuilder::new(config.clone());
        let builder_seeded =
            PredictionTrieBuilder::with_accumulators(AccumulatorState::default(), config);

        let trie_new = builder_new.build();
        let trie_seeded = builder_seeded.build();

        // Both should produce empty root tries
        assert_eq!(trie_new.name, "root");
        assert_eq!(trie_seeded.name, "root");
        assert!(trie_new.children.is_empty());
        assert!(trie_seeded.children.is_empty());
        assert!(trie_new.predictions_by_call_index.is_empty());
        assert!(trie_seeded.predictions_by_call_index.is_empty());
        assert!(trie_new.predictions_any_index.is_none());
        assert!(trie_seeded.predictions_any_index.is_none());
    }

    #[test]
    fn test_with_accumulators_pre_seeded() {
        let config = Some(SensitivityConfig::default());
        let run1 = make_test_run(3, 1);
        let run2 = make_test_run(2, 0);

        // Phase 1: Build accumulators from run1
        let mut builder1 = PredictionTrieBuilder::new(config.clone());
        builder1.add_run(&run1);
        let accs_after_run1 = builder1.accumulators().clone();
        let run1_root_count = accs_after_run1.nodes[""].all_remaining_calls.count;

        // Phase 2: Seed a new builder with those accumulators, add run2
        let mut builder2 = PredictionTrieBuilder::with_accumulators(accs_after_run1, config);
        builder2.add_run(&run2);
        let accs_after_both = builder2.accumulators();

        // run1 has 3 LLM calls, run2 has 2 LLM calls -> root should have 5 samples total
        let total_count = accs_after_both.nodes[""].all_remaining_calls.count;
        assert_eq!(
            total_count,
            run1_root_count + 2,
            "Seeded builder should have run1 samples ({run1_root_count}) + run2 samples (2) = {} total, got {total_count}",
            run1_root_count + 2
        );
    }

    #[test]
    fn test_with_accumulators_getter_returns_seeded_state() {
        use super::super::accumulator::NodeAccumulators;

        let mut state = AccumulatorState::default();
        state
            .nodes
            .insert("known_key".to_string(), NodeAccumulators::default());

        let builder =
            PredictionTrieBuilder::with_accumulators(state, Some(SensitivityConfig::default()));
        let accs = builder.accumulators();

        assert!(
            accs.nodes.contains_key("known_key"),
            "accumulators() should return the seeded state containing 'known_key'"
        );
    }
}
