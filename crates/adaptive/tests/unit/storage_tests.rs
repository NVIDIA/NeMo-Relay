// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for storage in the NeMo Relay adaptive crate.

use chrono::Utc;
use serde_json::json;
use uuid::Uuid;

use crate::acg::profile::{BlockStabilityScore, StabilityClass};
use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel, SpanId,
};
use crate::acg::stability::StabilityAnalysisResult;
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
            annotated_request: None,
            annotated_response: None,
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

fn sample_prompt_ir(agent_id: &str) -> PromptIR {
    PromptIR {
        ir_id: Uuid::now_v7(),
        blocks: vec![PromptBlock {
            span_id: SpanId(format!("{agent_id}-system-0")),
            sequence_index: 0,
            role: PromptRole::System,
            content: "You are helpful.".to_string(),
            content_type: BlockContentType::Text,
            provenance: ProvenanceLabel::System,
            sensitivity: SensitivityLabel::Public,
            token_metadata: None,
        }],
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    }
}

fn sample_stability(agent_id: &str) -> StabilityAnalysisResult {
    StabilityAnalysisResult {
        scores: vec![BlockStabilityScore {
            span_id: SpanId(format!("{agent_id}-system-0")),
            classification: StabilityClass::Stable,
            score: 1.0,
            confidence: 1.0,
            observation_count: 3,
        }],
        stable_prefix_length: 1,
        total_observations: 3,
        converged: false,
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

#[tokio::test(flavor = "current_thread")]
async fn in_memory_backend_round_trips_observations_and_stability() {
    let backend = InMemoryBackend::new();
    let observations = vec![sample_prompt_ir("agent-d"), sample_prompt_ir("agent-d-2")];
    let stability = sample_stability("agent-d");

    backend
        .store_observations("agent-d", &observations)
        .await
        .unwrap();
    backend
        .store_stability("agent-d", &stability)
        .await
        .unwrap();

    let loaded_observations = backend.load_observations("agent-d").await.unwrap().unwrap();
    let loaded_stability = backend.load_stability("agent-d").await.unwrap().unwrap();

    assert_eq!(loaded_observations.len(), 2);
    assert_eq!(
        loaded_observations[0].blocks[0].span_id.0,
        "agent-d-system-0"
    );
    assert_eq!(loaded_stability.stable_prefix_length, 1);
    assert_eq!(loaded_stability.total_observations, 3);
    assert!(!loaded_stability.converged);
}

#[tokio::test(flavor = "current_thread")]
async fn storage_backend_dyn_default_observation_and_stability_methods_are_noops() {
    let backend: AnyBackend = Box::new(DefaultStorePlanBackend);
    let observations = vec![sample_prompt_ir("agent-e")];
    let stability = sample_stability("agent-e");

    backend
        .store_observations("agent-e", &observations)
        .await
        .unwrap();
    backend
        .store_stability("agent-e", &stability)
        .await
        .unwrap();

    assert!(
        backend
            .load_observations("agent-e")
            .await
            .unwrap()
            .is_none()
    );
    assert!(backend.load_stability("agent-e").await.unwrap().is_none());
}
