// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use chrono::Utc;
use serde_json::json;
use uuid::Uuid;

use crate::storage::erased::AnyBackend;
use crate::storage::memory::InMemoryBackend;
use crate::storage::traits::{StorageBackend, StorageBackendDyn};
use crate::trie::accumulator::AccumulatorState;
use crate::trie::data_models::PredictionTrieNode;
use crate::trie::serialization::TrieEnvelope;
use crate::types::metadata::MetadataEnvelope;
use crate::types::plan::ExecutionPlan;
use crate::types::records::{CallKind, CallRecord, RunRecord};

fn sample_run(agent_id: &str) -> RunRecord {
    let now = Utc::now();
    RunRecord {
        id: Uuid::now_v7(),
        agent_id: agent_id.to_string(),
        calls: vec![CallRecord {
            kind: CallKind::Llm,
            name: "planner".to_string(),
            started_at: now,
            ended_at: Some(now),
            metadata_snapshot: None,
            output_tokens: Some(64),
            prompt_tokens: Some(16),
            total_tokens: Some(80),
            model_name: Some("gpt-test".to_string()),
            tool_call_count: Some(1),
        }],
        started_at: now,
        ended_at: Some(now),
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
            extensions: json!({"mode": "test"}),
        },
    }
}

struct DefaultStorePlanBackend;

impl StorageBackendDyn for DefaultStorePlanBackend {
    fn store_run_dyn<'a>(
        &'a self,
        _record: &'a RunRecord,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::error::Result<()>> + Send + 'a>>
    {
        Box::pin(async { Ok(()) })
    }

    fn load_plan_dyn<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = crate::error::Result<Option<ExecutionPlan>>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async { Ok(None) })
    }

    fn list_runs_dyn<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = crate::error::Result<Vec<RunRecord>>> + Send + 'a>,
    > {
        Box::pin(async { Ok(vec![]) })
    }

    fn store_trie<'a>(
        &'a self,
        _agent_id: &'a str,
        _envelope: &'a TrieEnvelope,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::error::Result<()>> + Send + 'a>>
    {
        Box::pin(async { Ok(()) })
    }

    fn load_trie<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = crate::error::Result<Option<TrieEnvelope>>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async { Ok(None) })
    }

    fn store_accumulators<'a>(
        &'a self,
        _agent_id: &'a str,
        _state: &'a AccumulatorState,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::error::Result<()>> + Send + 'a>>
    {
        Box::pin(async { Ok(()) })
    }

    fn load_accumulators<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = crate::error::Result<Option<AccumulatorState>>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async { Ok(None) })
    }
}

#[tokio::test(flavor = "current_thread")]
async fn in_memory_backend_round_trips_runs_plan_trie_and_accumulators() {
    let backend = InMemoryBackend::new();
    let run = sample_run("agent-a");
    let plan = sample_plan("agent-a");
    let envelope = TrieEnvelope::new(PredictionTrieNode::new("root"), "agent-a");
    let accumulators = AccumulatorState::default();

    backend.store_run(&run).await.unwrap();
    backend.store_plan(&plan).unwrap();
    backend.store_trie("agent-a", &envelope).await.unwrap();
    backend
        .store_accumulators("agent-a", &accumulators)
        .await
        .unwrap();

    let runs = backend.list_runs("agent-a").await.unwrap();
    let loaded_plan = backend.load_plan("agent-a").await.unwrap().unwrap();
    let loaded_trie = backend.load_trie("agent-a").await.unwrap().unwrap();
    let loaded_accumulators = backend.load_accumulators("agent-a").await.unwrap().unwrap();

    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].agent_id, "agent-a");
    assert_eq!(loaded_plan.agent_id, "agent-a");
    assert_eq!(loaded_trie.workflow_name, "agent-a");
    assert!(loaded_accumulators.nodes.is_empty());
    assert!(backend.list_runs("missing").await.unwrap().is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn erased_backend_alias_exposes_dynamic_storage_operations() {
    let backend: AnyBackend = Box::<InMemoryBackend>::default();
    let run = sample_run("agent-b");

    backend.store_run_dyn(&run).await.unwrap();
    let runs = backend.list_runs_dyn("agent-b").await.unwrap();

    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].calls[0].name, "planner");
    assert!(backend.load_plan_dyn("agent-b").await.unwrap().is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn storage_backend_dyn_default_store_plan_is_noop() {
    let backend: AnyBackend = Box::new(DefaultStorePlanBackend);
    let plan = sample_plan("agent-c");

    backend.store_plan(&plan).unwrap();

    assert!(backend.load_plan_dyn("agent-c").await.unwrap().is_none());
    assert!(backend.list_runs_dyn("agent-c").await.unwrap().is_empty());
}
