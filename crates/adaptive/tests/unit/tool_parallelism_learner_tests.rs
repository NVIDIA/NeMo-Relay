// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for tool parallelism learner in the NeMo Relay adaptive crate.

use std::sync::{Arc, RwLock};

use chrono::{Duration, Utc};
use serde_json::json;
use uuid::Uuid;

use super::*;
use crate::config::DriftConfig;
use crate::storage::memory::InMemoryBackend;
use crate::storage::traits::StorageBackend;
use crate::types::cache::HotCache;
use crate::types::records::CallRecord;

fn make_hot_cache() -> Arc<RwLock<HotCache>> {
    Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }))
}

fn make_tool_call(
    name: &str,
    base: chrono::DateTime<Utc>,
    start_offset_ms: i64,
    end_offset_ms: Option<i64>,
) -> CallRecord {
    CallRecord {
        kind: CallKind::Tool,
        name: name.to_string(),
        started_at: base + Duration::milliseconds(start_offset_ms),
        ended_at: end_offset_ms.map(|offset| base + Duration::milliseconds(offset)),
        metadata_snapshot: None,
        output_tokens: None,
        prompt_tokens: None,
        total_tokens: None,
        model_name: None,
        tool_call_count: None,
        annotated_request: None,
        annotated_response: None,
    }
}

fn make_llm_call(name: &str, base: chrono::DateTime<Utc>) -> CallRecord {
    CallRecord {
        kind: CallKind::Llm,
        name: name.to_string(),
        started_at: base,
        ended_at: Some(base + Duration::milliseconds(10)),
        metadata_snapshot: None,
        output_tokens: None,
        prompt_tokens: None,
        total_tokens: None,
        model_name: None,
        tool_call_count: None,
        annotated_request: None,
        annotated_response: None,
    }
}

fn make_run(agent_id: &str, calls: Vec<CallRecord>) -> RunRecord {
    let started_at = calls
        .first()
        .map(|call| call.started_at)
        .unwrap_or_else(Utc::now);
    let ended_at = calls.last().and_then(|call| call.ended_at);
    RunRecord {
        id: Uuid::new_v4(),
        agent_id: agent_id.to_string(),
        calls,
        started_at,
        ended_at,
    }
}

fn make_existing_plan(agent_id: &str) -> ExecutionPlan {
    ExecutionPlan {
        agent_id: agent_id.to_string(),
        parallel_groups: vec![ParallelGroup {
            group_id: "fanout:existing".to_string(),
            tool_names: vec!["existing".to_string()],
        }],
        metadata_template: MetadataEnvelope {
            run_id: Uuid::new_v4(),
            agent_id: agent_id.to_string(),
            parallel_hints: vec![ParallelHint {
                tool_name: "existing".to_string(),
                group_id: "fanout:existing".to_string(),
                explicit: false,
            }],
            extensions: json!({}),
        },
    }
}

#[test]
fn derive_observed_cohorts_ignores_non_tool_incomplete_and_non_overlapping_calls() {
    let base = Utc::now();
    let run = make_run(
        "agent-a",
        vec![
            make_llm_call("planner", base),
            make_tool_call("search", base, 0, Some(50)),
            make_tool_call("fetch", base, 10, Some(60)),
            make_tool_call("later", base, 100, Some(120)),
            make_tool_call("incomplete", base, 5, None),
        ],
    );

    let cohorts = derive_observed_cohorts(&run);

    assert_eq!(
        cohorts,
        vec![vec!["fetch".to_string(), "search".to_string()]]
    );
}

#[test]
fn derive_observed_cohorts_deduplicates_sorted_cohorts() {
    let base = Utc::now();
    let run = make_run(
        "agent-a",
        vec![
            make_tool_call("search", base, 0, Some(100)),
            make_tool_call("fetch", base, 10, Some(80)),
            make_tool_call("fetch", base, 20, Some(70)),
        ],
    );

    let cohorts = derive_observed_cohorts(&run);

    assert_eq!(cohorts.len(), 2);
    assert!(cohorts.contains(&vec!["fetch".to_string(), "search".to_string()]));
    assert!(cohorts.contains(&vec![
        "fetch".to_string(),
        "fetch".to_string(),
        "search".to_string(),
    ]));
}

#[test]
fn merge_observed_cohorts_preserves_existing_groups_and_deduplicates_hints_per_group() {
    let run_id = Uuid::new_v4();
    let mut plan = make_existing_plan("agent-b");
    let observed = vec![
        vec!["search".to_string(), "search".to_string()],
        vec!["fetch".to_string(), "search".to_string()],
    ];

    merge_observed_cohorts(&mut plan, &observed, run_id);

    assert_eq!(plan.parallel_groups.len(), 3);
    assert_eq!(plan.metadata_template.agent_id, "agent-b");
    assert_eq!(plan.metadata_template.run_id, run_id);
    assert!(
        plan.parallel_groups
            .iter()
            .any(|group| group.tool_names == vec!["search".to_string(), "search".to_string()])
    );
    assert!(
        plan.parallel_groups
            .iter()
            .any(|group| group.tool_names == vec!["fetch".to_string(), "search".to_string()])
    );
    assert_eq!(
        plan.metadata_template
            .parallel_hints
            .iter()
            .filter(|hint| hint.tool_name == "search")
            .count(),
        2,
    );
}

#[test]
fn build_parallel_group_hashes_joined_tool_names() {
    let group = build_parallel_group(&["fetch".to_string(), "search".to_string()]);

    assert_eq!(
        group.tool_names,
        vec!["fetch".to_string(), "search".to_string()]
    );
    assert!(group.group_id.starts_with("fanout:"));
    assert_eq!(group.group_id.len(), "fanout:".len() + 12);
}

#[test]
fn empty_execution_plan_starts_with_blank_metadata_and_groups() {
    let run_id = Uuid::new_v4();

    let plan = empty_execution_plan("agent-c", run_id);

    assert_eq!(plan.agent_id, "agent-c");
    assert!(plan.parallel_groups.is_empty());
    assert_eq!(plan.metadata_template.run_id, run_id);
    assert_eq!(plan.metadata_template.agent_id, "agent-c");
    assert!(plan.metadata_template.parallel_hints.is_empty());
    assert_eq!(plan.metadata_template.extensions, json!({}));
}

#[tokio::test]
async fn process_run_persists_parallel_groups_and_updates_hot_cache() {
    let backend = InMemoryBackend::new();
    let hot_cache = make_hot_cache();
    let learner = ToolParallelismLearner::new("agent-overlap");
    let base = Utc::now();
    let run = make_run(
        "agent-overlap",
        vec![
            make_tool_call("search", base, 0, Some(90)),
            make_tool_call("search", base, 10, Some(100)),
        ],
    );

    learner
        .process_run(&run, &backend, &hot_cache)
        .await
        .unwrap();

    let plan = backend
        .load_plan("agent-overlap")
        .await
        .unwrap()
        .expect("fanout should create a persisted plan");
    assert_eq!(plan.parallel_groups.len(), 1);
    assert_eq!(
        plan.parallel_groups[0].tool_names,
        vec!["search".to_string(), "search".to_string()],
    );

    let cache_plan = hot_cache.read().unwrap().plan.clone().unwrap();
    assert_eq!(cache_plan.parallel_groups.len(), 1);
}

#[tokio::test]
async fn process_run_leaves_state_unchanged_when_no_cohorts_are_observed() {
    let backend = InMemoryBackend::new();
    let hot_cache = make_hot_cache();
    let learner = ToolParallelismLearner::new("agent-serial");
    let base = Utc::now();
    let run = make_run(
        "agent-serial",
        vec![
            make_tool_call("search", base, 0, Some(20)),
            make_tool_call("fetch", base, 40, Some(60)),
        ],
    );

    learner
        .process_run(&run, &backend, &hot_cache)
        .await
        .unwrap();

    assert!(backend.load_plan("agent-serial").await.unwrap().is_none());
    assert!(hot_cache.read().unwrap().plan.is_none());
}

#[tokio::test]
async fn process_run_merges_new_cohorts_into_existing_plan() {
    let backend = InMemoryBackend::new();
    let hot_cache = make_hot_cache();
    let learner = ToolParallelismLearner::new("agent-merge");
    let base = Utc::now();
    let run = make_run(
        "agent-merge",
        vec![
            make_tool_call("search", base, 0, Some(90)),
            make_tool_call("fetch", base, 10, Some(100)),
        ],
    );

    backend
        .store_plan(&make_existing_plan("agent-merge"))
        .unwrap();
    learner
        .process_run(&run, &backend, &hot_cache)
        .await
        .unwrap();

    let plan = backend.load_plan("agent-merge").await.unwrap().unwrap();
    assert_eq!(plan.parallel_groups.len(), 2);
    assert!(
        plan.parallel_groups
            .iter()
            .any(|group| group.group_id == "fanout:existing")
    );
    assert!(
        plan.parallel_groups
            .iter()
            .any(|group| group.tool_names == vec!["fetch".to_string(), "search".to_string()])
    );
}

#[tokio::test]
async fn process_run_invalidates_existing_plan_when_tool_cohort_topology_drifts() {
    let backend = InMemoryBackend::new();
    let hot_cache = make_hot_cache();
    let learner = ToolParallelismLearner::new_with_drift(
        "agent-drift",
        Some(DriftConfig {
            enabled: true,
            threshold: 0.01,
        }),
    );
    let base = Utc::now();
    let first_run = make_run(
        "agent-drift",
        vec![
            make_tool_call("search", base, 0, Some(90)),
            make_tool_call("fetch", base, 10, Some(100)),
        ],
    );
    let drifted_run = make_run(
        "agent-drift",
        vec![
            make_tool_call("compile", base, 0, Some(120)),
            make_tool_call("test", base, 10, Some(130)),
            make_tool_call("lint", base, 20, Some(140)),
        ],
    );

    backend
        .store_plan(&make_existing_plan("agent-drift"))
        .unwrap();
    learner
        .process_run(&first_run, &backend, &hot_cache)
        .await
        .unwrap();
    learner
        .process_run(&drifted_run, &backend, &hot_cache)
        .await
        .unwrap();

    let plan = backend.load_plan("agent-drift").await.unwrap().unwrap();
    assert!(
        !plan
            .parallel_groups
            .iter()
            .any(|group| group.group_id == "fanout:existing"),
        "drifted cohort topology should invalidate stale plan groups"
    );
    assert!(plan.parallel_groups.iter().any(|group| {
        group.tool_names
            == vec![
                "compile".to_string(),
                "lint".to_string(),
                "test".to_string(),
            ]
    }));
}

#[tokio::test]
async fn process_run_reports_hot_cache_lock_poisoning() {
    let backend = InMemoryBackend::new();
    let hot_cache = make_hot_cache();
    let poisoned = hot_cache.clone();
    let learner = ToolParallelismLearner::new("agent-poisoned");
    let base = Utc::now();
    let run = make_run(
        "agent-poisoned",
        vec![
            make_tool_call("search", base, 0, Some(90)),
            make_tool_call("fetch", base, 10, Some(100)),
        ],
    );

    let _ = std::panic::catch_unwind(move || {
        let _guard = poisoned.write().unwrap();
        panic!("poison hot cache");
    });

    let error = learner
        .process_run(&run, &backend, &hot_cache)
        .await
        .unwrap_err();

    assert!(
        matches!(error, AdaptiveError::Internal(message) if message.contains("hot cache lock poisoned"))
    );
}
