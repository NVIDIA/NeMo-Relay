// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use chrono::Utc;
use uuid::Uuid;

use crate::storage::traits::{StorageBackend, StorageBackendDyn};
use crate::trie::accumulator::AccumulatorState;
use crate::trie::data_models::PredictionTrieNode;
use crate::trie::serialization::TrieEnvelope;
use crate::types::metadata::MetadataEnvelope;
use crate::types::plan::ExecutionPlan;
use crate::types::records::RunRecord;

async fn get_test_redis() -> Option<RedisBackend> {
    let prefix = format!("unit:{}:", Uuid::now_v7());
    match RedisBackend::new("redis://127.0.0.1/", prefix).await {
        Ok(backend) => Some(backend),
        Err(_) => {
            eprintln!("SKIP: Redis not available at 127.0.0.1:6379");
            None
        }
    }
}

fn sample_run(agent_id: &str) -> RunRecord {
    RunRecord {
        id: Uuid::now_v7(),
        agent_id: agent_id.to_string(),
        calls: vec![],
        started_at: Utc::now(),
        ended_at: Some(Utc::now()),
    }
}

fn sample_plan(agent_id: &str) -> ExecutionPlan {
    ExecutionPlan {
        agent_id: agent_id.to_string(),
        parallel_groups: vec![],
        metadata_template: MetadataEnvelope {
            run_id: Uuid::now_v7(),
            agent_id: agent_id.to_string(),
            parallel_hints: vec![],
            extensions: serde_json::json!({}),
        },
    }
}

fn sample_trie(agent_id: &str) -> TrieEnvelope {
    TrieEnvelope::new(PredictionTrieNode::new("root"), agent_id)
}

#[tokio::test(flavor = "current_thread")]
async fn redis_backend_new_rejects_invalid_urls_before_connecting() {
    let err = RedisBackend::new("not-a-redis-url", "prefix:")
        .await
        .err()
        .expect("expected invalid redis url to fail");

    match err {
        AdaptiveError::Storage(message) => {
            assert!(message.contains("redis client"));
        }
        other => panic!("unexpected redis constructor error: {other}"),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn redis_backend_new_builds_prefixed_keys() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let run_id = Uuid::now_v7();

    assert_eq!(
        backend.key("plan", "agent-a"),
        format!("{}plan:agent-a", backend.key_prefix)
    );
    assert_eq!(
        backend.run_key("agent-a", &run_id),
        format!("{}runs:agent-a:{run_id}", backend.key_prefix)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn redis_backend_store_run_and_dyn_wrappers_round_trip_multiple_records() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let first = sample_run("agent-runs");
    let second = sample_run("agent-runs");

    backend.store_run(&first).await.unwrap();
    backend.store_run_dyn(&second).await.unwrap();

    let runs = backend.list_runs("agent-runs").await.unwrap();
    let dyn_runs = backend.list_runs_dyn("agent-runs").await.unwrap();

    assert_eq!(runs.len(), 2);
    assert_eq!(dyn_runs.len(), 2);
    assert_eq!(runs[0].agent_id, "agent-runs");
    assert_eq!(dyn_runs[1].id, second.id);
}

#[tokio::test(flavor = "current_thread")]
async fn redis_backend_list_runs_skips_index_entries_without_records() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let mut conn = backend.conn.clone();
    let index_key = backend.key("runs_index", "agent-missing");

    redis::cmd("RPUSH")
        .arg(&index_key)
        .arg("missing-run-id")
        .query_async::<()>(&mut conn)
        .await
        .unwrap();

    let runs = backend.list_runs("agent-missing").await.unwrap();
    assert!(runs.is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn redis_backend_load_plan_supports_none_success_and_invalid_json() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    assert!(backend.load_plan("agent-plan").await.unwrap().is_none());
    assert!(backend.load_plan_dyn("agent-plan").await.unwrap().is_none());

    let plan = sample_plan("agent-plan");
    let key = backend.key("plan", "agent-plan");
    let mut conn = backend.conn.clone();
    let json = serde_json::to_string(&plan).unwrap();
    redis::cmd("SET")
        .arg(&key)
        .arg(&json)
        .query_async::<()>(&mut conn)
        .await
        .unwrap();
    let loaded = backend.load_plan("agent-plan").await.unwrap().unwrap();
    assert_eq!(loaded.agent_id, "agent-plan");

    redis::cmd("SET")
        .arg(&key)
        .arg("{not-json")
        .query_async::<()>(&mut conn)
        .await
        .unwrap();
    let err = backend.load_plan("agent-plan").await.unwrap_err();
    assert!(matches!(err, AdaptiveError::Serialization(_)));
}

#[tokio::test(flavor = "current_thread")]
async fn redis_backend_trie_and_accumulators_round_trip_and_report_invalid_json() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let trie = sample_trie("agent-state");
    let accumulators = AccumulatorState::default();

    backend.store_trie("agent-state", &trie).await.unwrap();
    backend
        .store_accumulators("agent-state", &accumulators)
        .await
        .unwrap();

    let loaded_trie = backend.load_trie("agent-state").await.unwrap().unwrap();
    let loaded_accumulators = backend
        .load_accumulators("agent-state")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(loaded_trie.workflow_name, "agent-state");
    assert!(loaded_accumulators.nodes.is_empty());

    let mut conn = backend.conn.clone();
    let trie_key = backend.key("trie", "agent-state");
    redis::cmd("SET")
        .arg(&trie_key)
        .arg("{bad-json")
        .query_async::<()>(&mut conn)
        .await
        .unwrap();
    let trie_err = backend.load_trie("agent-state").await.unwrap_err();
    assert!(matches!(trie_err, AdaptiveError::Serialization(_)));

    let accum_key = backend.key("accumulators", "agent-state");
    redis::cmd("SET")
        .arg(&accum_key)
        .arg("{bad-json")
        .query_async::<()>(&mut conn)
        .await
        .unwrap();
    let accum_err = backend.load_accumulators("agent-state").await.unwrap_err();
    assert!(matches!(accum_err, AdaptiveError::Serialization(_)));
}

#[tokio::test(flavor = "current_thread")]
async fn redis_backend_list_runs_reports_invalid_json_records() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let mut conn = backend.conn.clone();
    let index_key = backend.key("runs_index", "agent-invalid");
    let run_key = backend.run_key("agent-invalid", &Uuid::nil());

    redis::cmd("SET")
        .arg(&run_key)
        .arg("{broken-json")
        .query_async::<()>(&mut conn)
        .await
        .unwrap();
    redis::cmd("RPUSH")
        .arg(&index_key)
        .arg(Uuid::nil().to_string())
        .query_async::<()>(&mut conn)
        .await
        .unwrap();

    let err = backend.list_runs("agent-invalid").await.unwrap_err();
    assert!(matches!(err, AdaptiveError::Serialization(_)));
}
