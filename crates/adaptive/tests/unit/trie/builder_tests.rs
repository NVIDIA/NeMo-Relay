// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use chrono::{Duration, Utc};
use uuid::Uuid;

use crate::types::records::{CallKind, CallRecord, RunRecord};

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
        let (kind, name, tokens) =
            if llm_placed < llm_count && (tool_placed >= tool_count || llm_placed <= tool_placed) {
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
            prompt_tokens: None,
            total_tokens: None,
            model_name: None,
            tool_call_count: None,
        });
        offset_ms += 1100; // 1s call + 100ms gap
    }

    let run_end = calls.last().map(|c| c.ended_at.unwrap()).unwrap_or(base);
    RunRecord {
        id: Uuid::now_v7(),
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
