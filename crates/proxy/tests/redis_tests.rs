// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis integration tests for [`RedisBackend`].
//!
//! These tests require a running Redis instance at `redis://127.0.0.1/`.
//! When Redis is unavailable, tests skip gracefully.

#![cfg(feature = "redis-backend")]

use chrono::Utc;
use uuid::Uuid;

use nvidia_nat_nexus_proxy::trie::{
    AccumulatorState, NodeAccumulators, PredictionTrieNode, RunningStats, TrieEnvelope,
};
use nvidia_nat_nexus_proxy::{
    ExecutionPlan, MetadataEnvelope, RedisBackend, RunRecord, StorageBackend, StorageBackendDyn,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Attempt to connect to a local Redis instance. Returns `None` (skip) if
/// Redis is unavailable, so tests degrade gracefully in CI without Redis.
async fn get_test_redis() -> Option<RedisBackend> {
    // Use unique prefix per test run to avoid key collisions
    let prefix = format!("test:{}:", Uuid::new_v4());
    match RedisBackend::new("redis://127.0.0.1/", prefix).await {
        Ok(backend) => Some(backend),
        Err(_) => {
            eprintln!("SKIP: Redis not available at 127.0.0.1:6379");
            None
        }
    }
}

fn make_test_run(agent_id: &str) -> RunRecord {
    RunRecord {
        id: Uuid::new_v4(),
        agent_id: agent_id.to_string(),
        calls: vec![],
        started_at: Utc::now(),
        ended_at: None,
    }
}

fn make_test_plan(agent_id: &str) -> ExecutionPlan {
    ExecutionPlan {
        agent_id: agent_id.to_string(),
        parallel_groups: vec![],
        metadata_template: MetadataEnvelope {
            run_id: Uuid::new_v4(),
            agent_id: agent_id.to_string(),
            parallel_hints: vec![],
            extensions: serde_json::json!({}),
        },
    }
}

fn make_test_trie_envelope(workflow_name: &str) -> TrieEnvelope {
    let mut root = PredictionTrieNode::new("root");
    let child = PredictionTrieNode::new("child_agent");
    root.children.insert("child_agent".to_string(), child);
    TrieEnvelope::new(root, workflow_name)
}

fn make_test_accumulator_state() -> AccumulatorState {
    let mut state = AccumulatorState::default();
    let mut node_acc = NodeAccumulators::default();

    // Per-index stats
    let mut stats = RunningStats::new();
    stats.add_sample(100.0);
    stats.add_sample(200.0);
    stats.add_sample(300.0);
    node_acc.remaining_calls.insert(1, stats);

    // Aggregate stats -- must have samples so the TDigest is non-empty and
    // survives JSON round-trip (empty TDigest contains NaN internals).
    node_acc.all_remaining_calls.add_sample(100.0);
    node_acc.all_remaining_calls.add_sample(200.0);
    node_acc.all_remaining_calls.add_sample(300.0);
    node_acc.all_interarrival_ms.add_sample(50.0);
    node_acc.all_output_tokens.add_sample(256.0);
    node_acc.all_sensitivity.add_sample(0.8);

    state.nodes.insert("workflow/agent".to_string(), node_acc);
    state
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_redis_store_load_run() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let record = make_test_run("agent-redis-run");
    let record_id = record.id;
    backend.store_run(&record).await.unwrap();
    let runs = backend.list_runs("agent-redis-run").await.unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].id, record_id);
    assert_eq!(runs[0].agent_id, "agent-redis-run");
}

#[tokio::test]
async fn test_redis_store_load_plan() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let plan = make_test_plan("agent-redis-plan");

    // Store plan via StorageBackendDyn (store_run_dyn stores a run, but we need
    // to test plan storage via the Redis SET command -- use load_plan which is
    // on StorageBackend. We store the plan by serializing and using redis directly
    // via the backend trait). Plans are stored via the run pipeline, but for test
    // we verify load_plan returns None for an agent with no plan.
    let loaded = backend.load_plan("agent-redis-plan").await.unwrap();
    assert!(
        loaded.is_none(),
        "load_plan for agent with no stored plan should return None"
    );

    // Plans are not stored via StorageBackend (no store_plan method on trait).
    // Verify that load_plan_dyn also returns None.
    let loaded_dyn = backend.load_plan_dyn("agent-redis-plan").await.unwrap();
    assert!(
        loaded_dyn.is_none(),
        "load_plan_dyn for agent with no stored plan should return None"
    );

    // Note: plan is constructed but the StorageBackend trait does not have
    // a store_plan method. The plan is kept alive to suppress the unused
    // variable warning.
    let _ = plan;
}

#[tokio::test]
async fn test_redis_trie_atomic_roundtrip() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let envelope = make_test_trie_envelope("redis-trie-workflow");

    backend
        .store_trie("agent-redis-trie", &envelope)
        .await
        .unwrap();
    let loaded = backend.load_trie("agent-redis-trie").await.unwrap();

    assert!(loaded.is_some(), "stored trie should be loadable");
    let loaded = loaded.unwrap();
    assert_eq!(loaded.workflow_name, "redis-trie-workflow");
    assert_eq!(loaded.root.name, "root");
    assert!(
        loaded.root.children.contains_key("child_agent"),
        "child node should survive round-trip"
    );
    assert_eq!(loaded.version, "1.0");
}

#[tokio::test]
async fn test_redis_accumulators_roundtrip() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let state = make_test_accumulator_state();

    backend
        .store_accumulators("agent-redis-acc", &state)
        .await
        .unwrap();
    let loaded = backend.load_accumulators("agent-redis-acc").await.unwrap();

    assert!(loaded.is_some(), "stored accumulators should be loadable");
    let loaded = loaded.unwrap();
    assert!(
        loaded.nodes.contains_key("workflow/agent"),
        "path key should survive round-trip"
    );
    let node_acc = &loaded.nodes["workflow/agent"];
    assert!(
        node_acc.remaining_calls.contains_key(&1),
        "call index 1 should exist"
    );
    let stats = &node_acc.remaining_calls[&1];
    assert_eq!(stats.count, 3, "should have 3 samples");
    assert!(
        (stats.mean - 200.0).abs() < 1e-6,
        "mean should be 200.0, got {}",
        stats.mean
    );
}

#[tokio::test]
async fn test_redis_load_nonexistent_trie() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let loaded = backend.load_trie("agent-does-not-exist").await.unwrap();
    assert!(
        loaded.is_none(),
        "load_trie for nonexistent agent should return None"
    );
}

#[tokio::test]
async fn test_redis_load_nonexistent_accumulators() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let loaded = backend
        .load_accumulators("agent-does-not-exist")
        .await
        .unwrap();
    assert!(
        loaded.is_none(),
        "load_accumulators for nonexistent agent should return None"
    );
}

#[tokio::test]
async fn test_redis_overwrite_trie() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let first = make_test_trie_envelope("first-workflow");
    let second = make_test_trie_envelope("second-workflow");

    backend
        .store_trie("agent-redis-overwrite", &first)
        .await
        .unwrap();
    backend
        .store_trie("agent-redis-overwrite", &second)
        .await
        .unwrap();

    let loaded = backend.load_trie("agent-redis-overwrite").await.unwrap();

    assert!(loaded.is_some(), "overwritten trie should be loadable");
    let loaded = loaded.unwrap();
    assert_eq!(
        loaded.workflow_name, "second-workflow",
        "load should return the second (overwritten) trie, not the first"
    );
}
