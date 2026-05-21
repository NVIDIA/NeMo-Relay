// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for tool parallelism plan in the NeMo Relay adaptive crate.

use std::sync::{Arc, RwLock};

use chrono::{Duration, Utc};
use serde_json::json;
use uuid::Uuid;

use nemo_relay_adaptive::{
    InMemoryBackend, StorageBackend, StorageBackendDyn, ToolParallelismLearner,
};
use nemo_relay_adaptive::learner::traits::Learner;
use nemo_relay_adaptive::types::cache::HotCache;
use nemo_relay_adaptive::types::metadata::{MetadataEnvelope, ParallelHint};
use nemo_relay_adaptive::types::plan::{ExecutionPlan, ParallelGroup};
use nemo_relay_adaptive::types::records::{CallKind, CallRecord, RunRecord};

fn make_hot_cache() -> Arc<RwLock<HotCache>> {
    Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
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
    end_offset_ms: i64,
) -> CallRecord {
    CallRecord {
        kind: CallKind::Tool,
        name: name.to_string(),
        started_at: base + Duration::milliseconds(start_offset_ms),
        ended_at: Some(base + Duration::milliseconds(end_offset_ms)),
        metadata_snapshot: None,
        output_tokens: None,
        prompt_tokens: None,
        total_tokens: None,
        model_name: None,
        tool_call_count: None,
    annotated_request: None,
    annotated_response: None,
        annotated_request: None,
        annotated_response: None,
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
            tool_names: vec!["existing".to_string(), "existing".to_string()],
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

#[tokio::test]
async fn tool_parallelism_plan_persists_overlapping_groups_and_preserves_duplicates() {
    let backend = InMemoryBackend::new();
    let hot_cache = make_hot_cache();
    let learner = ToolParallelismLearner::new("agent-overlap");
    let base = Utc::now();
    let run = make_run(
        "agent-overlap",
        vec![
            make_tool_call("search", base, 0, 90),
            make_tool_call("search", base, 10, 100),
            make_tool_call("finalize", base, 200, 240),
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
        .expect("overlapping tool fan-out should persist a plan");
    assert_eq!(plan.parallel_groups.len(), 1);
    assert_eq!(
        plan.parallel_groups[0].tool_names,
        vec!["search".to_string(), "search".to_string()],
    );
    assert_eq!(plan.metadata_template.parallel_hints.len(), 1);
    assert_eq!(plan.metadata_template.parallel_hints[0].tool_name, "search");
    assert_eq!(
        plan.metadata_template.parallel_hints[0].group_id,
        plan.parallel_groups[0].group_id,
    );
}

#[tokio::test]
async fn tool_parallelism_plan_preserves_existing_plan_when_run_has_no_fanout() {
    let backend = InMemoryBackend::new();
    let hot_cache = make_hot_cache();
    let learner = ToolParallelismLearner::new("agent-no-fanout");
    let existing = make_existing_plan("agent-no-fanout");
    let base = Utc::now();
    let run = make_run(
        "agent-no-fanout",
        vec![
            make_tool_call("search", base, 0, 20),
            make_tool_call("search", base, 40, 60),
        ],
    );

    backend.store_plan(&existing).unwrap();

    learner
        .process_run(&run, &backend, &hot_cache)
        .await
        .unwrap();

    let stored = backend
        .load_plan("agent-no-fanout")
        .await
        .unwrap()
        .expect("existing plan should remain stored");
    assert_eq!(stored.parallel_groups.len(), 1);
    assert_eq!(stored.parallel_groups[0].group_id, "fanout:existing");
    assert_eq!(stored.metadata_template.parallel_hints.len(), 1);
    assert_eq!(
        stored.metadata_template.parallel_hints[0].tool_name,
        "existing"
    );
}

#[tokio::test]
async fn tool_parallelism_plan_refreshes_hot_cache_after_store() {
    let backend = InMemoryBackend::new();
    let hot_cache = make_hot_cache();
    let learner = ToolParallelismLearner::new("agent-hot-cache");
    let base = Utc::now();
    let run = make_run(
        "agent-hot-cache",
        vec![
            make_tool_call("search", base, 0, 80),
            make_tool_call("search", base, 5, 85),
        ],
    );

    learner
        .process_run(&run, &backend, &hot_cache)
        .await
        .unwrap();

    let guard = hot_cache.read().unwrap();
    let plan = guard
        .plan
        .as_ref()
        .expect("hot_cache.plan should be refreshed immediately after store");
    assert_eq!(plan.parallel_groups.len(), 1);
    assert_eq!(
        plan.parallel_groups[0].tool_names,
        vec!["search".to_string(), "search".to_string()],
    );
}
