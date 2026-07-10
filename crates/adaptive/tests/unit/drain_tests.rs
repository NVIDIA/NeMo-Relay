// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for drain in the NeMo Relay adaptive crate.

use super::*;
use crate::storage::memory::InMemoryBackend;
use crate::storage::traits::{StorageBackend, StorageBackendDyn};
use crate::trie::accumulator::AccumulatorState;
use crate::trie::serialization::TrieEnvelope;
use crate::types::cache::HotCache;
use crate::types::metadata::MetadataEnvelope;
use crate::types::plan::{ExecutionPlan, ParallelGroup};
use crate::types::records::RunRecord;
use nemo_relay::api::event::{
    BaseEvent, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
};
use nemo_relay::api::scope::ScopeType;
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use uuid::{Uuid, Version};

#[derive(Clone, Copy)]
enum EventType {
    Start,
    End,
}

/// Helper to construct a minimal test [`Event`] with caller-controlled ancestry.
fn make_event(
    event_type: EventType,
    scope_type: Option<ScopeType>,
    name: Option<&str>,
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
) -> Event {
    let event_name = name.unwrap_or("event");
    match (event_type, scope_type) {
        (EventType::Start, Some(ScopeType::Tool)) => Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .parent_uuid_opt(parent_uuid)
                .uuid(uuid)
                .name(event_name)
                .build(),
            ScopeCategory::Start,
            Vec::new(),
            EventCategory::tool(),
            None,
        )),
        (EventType::End, Some(ScopeType::Tool)) => Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .parent_uuid_opt(parent_uuid)
                .uuid(uuid)
                .name(event_name)
                .build(),
            ScopeCategory::End,
            Vec::new(),
            EventCategory::tool(),
            None,
        )),
        (EventType::Start, Some(ScopeType::Llm)) => Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .parent_uuid_opt(parent_uuid)
                .uuid(uuid)
                .name(event_name)
                .build(),
            ScopeCategory::Start,
            Vec::new(),
            EventCategory::llm(),
            None,
        )),
        (EventType::End, Some(ScopeType::Llm)) => Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .parent_uuid_opt(parent_uuid)
                .uuid(uuid)
                .name(event_name)
                .build(),
            ScopeCategory::End,
            Vec::new(),
            EventCategory::llm(),
            None,
        )),
        (EventType::Start, Some(scope_type)) => Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .parent_uuid_opt(parent_uuid)
                .uuid(uuid)
                .name(event_name)
                .build(),
            ScopeCategory::Start,
            Vec::new(),
            EventCategory::from(scope_type),
            None,
        )),
        (EventType::End, Some(scope_type)) => Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .parent_uuid_opt(parent_uuid)
                .uuid(uuid)
                .name(event_name)
                .build(),
            ScopeCategory::End,
            Vec::new(),
            EventCategory::from(scope_type),
            None,
        )),
        (_, None) => Event::Mark(MarkEvent::new(
            BaseEvent::builder()
                .parent_uuid_opt(parent_uuid)
                .uuid(uuid)
                .name(event_name)
                .build(),
            None,
            None,
        )),
    }
}

/// Helper: make an Agent Start event whose own uuid acts as the inferred root.
fn make_agent_start() -> Event {
    let uuid = Uuid::now_v7();
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder().uuid(uuid).name("my-agent").build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::agent(),
        None,
    ))
}

/// Helper: make an Agent End event for a given root event UUID.
fn make_agent_end(root_uuid: Uuid) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(root_uuid)
            .name("my-agent")
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::agent(),
        None,
    ))
}

fn make_test_plan(agent_id: &str) -> ExecutionPlan {
    ExecutionPlan {
        agent_id: agent_id.to_string(),
        parallel_groups: vec![ParallelGroup {
            group_id: "pg-1".to_string(),
            tool_names: vec!["search".to_string()],
        }],
        metadata_template: MetadataEnvelope {
            run_id: Uuid::now_v7(),
            agent_id: agent_id.to_string(),
            parallel_hints: vec![],
            extensions: json!({}),
        },
    }
}

struct StoreFailBackend;

impl StorageBackendDyn for StoreFailBackend {
    fn store_run_dyn<'a>(
        &'a self,
        _record: &'a RunRecord,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<()>> + Send + 'a>> {
        Box::pin(async { Err(crate::error::AdaptiveError::Storage("store failed".into())) })
    }

    fn load_plan_dyn<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<Option<ExecutionPlan>>> + Send + 'a>>
    {
        Box::pin(async { Ok(None) })
    }

    fn list_runs_dyn<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<Vec<RunRecord>>> + Send + 'a>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn store_trie<'a>(
        &'a self,
        _agent_id: &'a str,
        _envelope: &'a TrieEnvelope,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn load_trie<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<Option<TrieEnvelope>>> + Send + 'a>> {
        Box::pin(async { Ok(None) })
    }

    fn store_accumulators<'a>(
        &'a self,
        _agent_id: &'a str,
        _state: &'a AccumulatorState,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn load_accumulators<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<Option<AccumulatorState>>> + Send + 'a>>
    {
        Box::pin(async { Ok(None) })
    }
}

struct LoadPlanFailBackend {
    inner: InMemoryBackend,
}

impl StorageBackendDyn for LoadPlanFailBackend {
    fn store_run_dyn<'a>(
        &'a self,
        record: &'a RunRecord,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<()>> + Send + 'a>> {
        Box::pin(self.inner.store_run(record))
    }

    fn load_plan_dyn<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<Option<ExecutionPlan>>> + Send + 'a>>
    {
        Box::pin(async {
            Err(crate::error::AdaptiveError::Storage(
                "load_plan failed".into(),
            ))
        })
    }

    fn list_runs_dyn<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<Vec<RunRecord>>> + Send + 'a>> {
        Box::pin(self.inner.list_runs(agent_id))
    }

    fn store_trie<'a>(
        &'a self,
        agent_id: &'a str,
        envelope: &'a TrieEnvelope,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<()>> + Send + 'a>> {
        self.inner.store_trie(agent_id, envelope)
    }

    fn load_trie<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<Option<TrieEnvelope>>> + Send + 'a>> {
        self.inner.load_trie(agent_id)
    }

    fn store_accumulators<'a>(
        &'a self,
        agent_id: &'a str,
        state: &'a AccumulatorState,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<()>> + Send + 'a>> {
        self.inner.store_accumulators(agent_id, state)
    }

    fn load_accumulators<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<Option<AccumulatorState>>> + Send + 'a>>
    {
        self.inner.load_accumulators(agent_id)
    }
}

struct FailingLearner;

impl Learner for FailingLearner {
    fn process_run<'a>(
        &'a self,
        _run: &'a RunRecord,
        _backend: &'a dyn StorageBackendDyn,
        _hot_cache: &'a Arc<RwLock<HotCache>>,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<()>> + Send + 'a>> {
        Box::pin(async {
            Err(crate::error::AdaptiveError::Internal(
                "learner failed".to_string(),
            ))
        })
    }
}

// -----------------------------------------------------------------------
// RunAccumulator tests
// -----------------------------------------------------------------------

#[test]
fn test_accumulator_new_is_empty() {
    let acc = RunAccumulator::new("agent-1".to_string());
    assert_eq!(acc.open_run_count(), 0);
}

#[test]
fn test_accumulator_start_run() {
    let mut acc = RunAccumulator::new("agent-1".to_string());
    let event = make_agent_start();
    let result = acc.process_event(&event);
    assert!(result.is_none(), "Start should not return a completed run");
    assert_eq!(acc.open_run_count(), 1);
}

#[test]
fn test_accumulator_end_run_returns_record() {
    let mut acc = RunAccumulator::new("agent-1".to_string());

    let start = make_agent_start();
    let root_uuid = start.uuid();
    acc.process_event(&start);

    let end = make_agent_end(root_uuid);
    let result = acc.process_event(&end);

    assert!(result.is_some(), "End should return a completed run");
    let run = result.unwrap();
    assert_eq!(run.agent_id, "agent-1");
    assert!(run.ended_at.is_some());
    assert_eq!(run.id.get_version(), Some(Version::SortRand));
    assert_eq!(acc.open_run_count(), 0);
}

#[test]
fn test_accumulator_collects_calls() {
    let mut acc = RunAccumulator::new("agent-1".to_string());

    let start = make_agent_start();
    let root_uuid = start.uuid();
    acc.process_event(&start);

    // Tool Start + Tool End
    let tool_uuid = Uuid::now_v7();
    let tool_start = make_event(
        EventType::Start,
        Some(ScopeType::Tool),
        Some("search"),
        tool_uuid,
        Some(root_uuid),
    );
    acc.process_event(&tool_start);

    let tool_end = make_event(
        EventType::End,
        Some(ScopeType::Tool),
        Some("search"),
        tool_uuid,
        Some(root_uuid),
    );
    acc.process_event(&tool_end);

    // LLM Start + LLM End
    let llm_uuid = Uuid::now_v7();
    let llm_start = make_event(
        EventType::Start,
        Some(ScopeType::Llm),
        Some("gpt-4"),
        llm_uuid,
        Some(root_uuid),
    );
    acc.process_event(&llm_start);

    let llm_end = make_event(
        EventType::End,
        Some(ScopeType::Llm),
        Some("gpt-4"),
        llm_uuid,
        Some(root_uuid),
    );
    acc.process_event(&llm_end);

    // Agent End
    let end = make_agent_end(root_uuid);
    let result = acc.process_event(&end);

    let run = result.expect("should return completed run");
    assert_eq!(run.calls.len(), 2, "should have 2 call records");
    assert!(
        run.calls[0].ended_at.is_some(),
        "tool call should have ended_at"
    );
    assert!(
        run.calls[1].ended_at.is_some(),
        "llm call should have ended_at"
    );
}

#[test]
fn test_accumulator_tracks_non_agent_scope_roots_and_cleans_them_up() {
    let mut acc = RunAccumulator::new("agent-1".to_string());

    let agent_start = make_agent_start();
    let root_uuid = agent_start.uuid();
    acc.process_event(&agent_start);

    let function_uuid = Uuid::now_v7();
    let function_start = make_event(
        EventType::Start,
        Some(ScopeType::Function),
        Some("helper"),
        function_uuid,
        Some(root_uuid),
    );
    acc.process_event(&function_start);
    assert_eq!(acc.event_roots.get(&function_uuid), Some(&root_uuid));

    let function_end = make_event(
        EventType::End,
        Some(ScopeType::Function),
        Some("helper"),
        function_uuid,
        Some(root_uuid),
    );
    acc.process_event(&function_end);
    assert!(!acc.event_roots.contains_key(&function_uuid));
}

#[test]
fn test_accumulator_llm_end_clears_event_root_mapping() {
    let mut acc = RunAccumulator::new("agent-1".to_string());

    let agent_start = make_agent_start();
    let root_uuid = agent_start.uuid();
    acc.process_event(&agent_start);

    let llm_uuid = Uuid::now_v7();
    let llm_start = make_event(
        EventType::Start,
        Some(ScopeType::Llm),
        Some("gpt-4"),
        llm_uuid,
        Some(root_uuid),
    );
    acc.process_event(&llm_start);
    assert_eq!(acc.event_roots.get(&llm_uuid), Some(&root_uuid));

    let llm_end = make_event(
        EventType::End,
        Some(ScopeType::Llm),
        Some("gpt-4"),
        llm_uuid,
        Some(root_uuid),
    );
    acc.process_event(&llm_end);
    assert!(!acc.event_roots.contains_key(&llm_uuid));
}

#[test]
fn test_accumulator_tracks_calls_nested_under_non_agent_scope() {
    let mut acc = RunAccumulator::new("agent-1".to_string());

    let agent_start = make_agent_start();
    let root_uuid = agent_start.uuid();
    acc.process_event(&agent_start);

    let function_uuid = Uuid::now_v7();
    let function_start = make_event(
        EventType::Start,
        Some(ScopeType::Function),
        Some("helper"),
        function_uuid,
        Some(root_uuid),
    );
    acc.process_event(&function_start);

    let tool_uuid = Uuid::now_v7();
    let tool_start = make_event(
        EventType::Start,
        Some(ScopeType::Tool),
        Some("search"),
        tool_uuid,
        Some(function_uuid),
    );
    acc.process_event(&tool_start);

    let tool_end = make_event(
        EventType::End,
        Some(ScopeType::Tool),
        Some("search"),
        tool_uuid,
        Some(function_uuid),
    );
    acc.process_event(&tool_end);

    let function_end = make_event(
        EventType::End,
        Some(ScopeType::Function),
        Some("helper"),
        function_uuid,
        Some(root_uuid),
    );
    acc.process_event(&function_end);

    let run = acc
        .process_event(&make_agent_end(root_uuid))
        .expect("agent end should return completed run");
    assert_eq!(run.calls.len(), 1, "nested tool call should be tracked");
    assert_eq!(run.calls[0].name, "search");
    assert!(run.calls[0].ended_at.is_some());
}

#[test]
fn test_accumulator_tracks_llm_calls_nested_under_non_agent_scope() {
    let mut acc = RunAccumulator::new("agent-1".to_string());

    let agent_start = make_agent_start();
    let root_uuid = agent_start.uuid();
    acc.process_event(&agent_start);

    let function_uuid = Uuid::now_v7();
    let function_start = make_event(
        EventType::Start,
        Some(ScopeType::Function),
        Some("helper"),
        function_uuid,
        Some(root_uuid),
    );
    acc.process_event(&function_start);

    let llm_uuid = Uuid::now_v7();
    let llm_start = make_event(
        EventType::Start,
        Some(ScopeType::Llm),
        Some("gpt-4"),
        llm_uuid,
        Some(function_uuid),
    );
    acc.process_event(&llm_start);

    let llm_end = make_event(
        EventType::End,
        Some(ScopeType::Llm),
        Some("gpt-4"),
        llm_uuid,
        Some(function_uuid),
    );
    acc.process_event(&llm_end);

    let function_end = make_event(
        EventType::End,
        Some(ScopeType::Function),
        Some("helper"),
        function_uuid,
        Some(root_uuid),
    );
    acc.process_event(&function_end);

    let run = acc
        .process_event(&make_agent_end(root_uuid))
        .expect("agent end should return completed run");
    assert_eq!(run.calls.len(), 1, "nested llm call should be tracked");
    assert_eq!(run.calls[0].name, "gpt-4");
    assert!(run.calls[0].ended_at.is_some());
}

#[test]
fn test_accumulator_orphaned_end_returns_none() {
    let mut acc = RunAccumulator::new("agent-1".to_string());
    let end = make_agent_end(Uuid::now_v7());
    let result = acc.process_event(&end);
    assert!(
        result.is_none(),
        "Orphaned end event should not return a run"
    );
}

// -----------------------------------------------------------------------
// drain_task tests
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_drain_task_exits_on_channel_close() {
    let concrete = Arc::new(InMemoryBackend::new());
    let backend: Arc<dyn StorageBackendDyn + Send + Sync> = concrete.clone();
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    let handle = tokio::spawn(drain_task(
        rx,
        Arc::clone(&backend),
        Arc::clone(&hot_cache),
        "agent-1".to_string(),
        vec![],
    ));

    // Drop sender -- channel closes
    drop(tx);

    // drain_task should exit cleanly
    let result = tokio::time::timeout(Duration::from_secs(2), handle).await;
    assert!(
        result.is_ok(),
        "drain_task should exit promptly after channel close"
    );
    result.unwrap().expect("drain_task should not panic");
}

#[tokio::test]
async fn test_drain_task_stores_completed_run() {
    let concrete = Arc::new(InMemoryBackend::new());
    let backend: Arc<dyn StorageBackendDyn + Send + Sync> = concrete.clone();
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    let handle = tokio::spawn(drain_task(
        rx,
        Arc::clone(&backend),
        Arc::clone(&hot_cache),
        "agent-1".to_string(),
        vec![],
    ));

    // Send Agent Start
    let start = make_agent_start();
    let root_uuid = start.uuid();
    tx.send(start).expect("send should succeed");

    // Send Agent End
    let end = make_agent_end(root_uuid);
    tx.send(end).expect("send should succeed");

    // Give drain time to process
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Drop sender to allow drain to exit
    drop(tx);
    let _ = handle.await;

    // Verify the run was stored (use concrete handle for StorageBackend methods)
    let runs = concrete.list_runs("agent-1").await.unwrap();
    assert_eq!(runs.len(), 1, "should have stored 1 run");
    assert_eq!(runs[0].agent_id, "agent-1");
    assert!(runs[0].ended_at.is_some());
}

#[tokio::test]
async fn test_drain_task_updates_hot_cache() {
    let concrete = Arc::new(InMemoryBackend::new());
    let backend: Arc<dyn StorageBackendDyn + Send + Sync> = concrete.clone();

    // Pre-seed a plan in the backend
    let plan = make_test_plan("agent-1");
    concrete.store_plan(&plan).unwrap();

    let hot_cache: Arc<RwLock<HotCache>> = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    let handle = tokio::spawn(drain_task(
        rx,
        Arc::clone(&backend),
        Arc::clone(&hot_cache),
        "agent-1".to_string(),
        vec![],
    ));

    // Send Agent Start + End to trigger a store + cache refresh
    let start = make_agent_start();
    let root_uuid = start.uuid();
    tx.send(start).expect("send should succeed");
    tx.send(make_agent_end(root_uuid))
        .expect("send should succeed");

    // Give drain time to process
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Drop sender to allow drain to exit
    drop(tx);
    let _ = handle.await;

    // Verify hot cache was updated with the plan
    let guard = hot_cache.read().unwrap();
    assert!(guard.plan.is_some(), "hot cache should contain a plan");
    let cached_plan = guard.plan.as_ref().unwrap();
    assert_eq!(cached_plan.agent_id, "agent-1");
    assert_eq!(cached_plan.parallel_groups.len(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn test_drain_task_continues_when_store_run_fails() {
    let backend: Arc<dyn StorageBackendDyn + Send + Sync> = Arc::new(StoreFailBackend);
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    let handle = tokio::spawn(drain_task(
        rx,
        backend,
        hot_cache.clone(),
        "agent-drain".to_string(),
        vec![],
    ));

    let start = make_agent_start();
    let end = make_agent_end(start.uuid());
    tx.send(start).unwrap();
    tx.send(end).unwrap();
    drop(tx);

    tokio::time::timeout(Duration::from_secs(2), handle)
        .await
        .expect("drain task should exit")
        .unwrap();
    assert!(hot_cache.read().unwrap().plan.is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn test_drain_task_continues_when_learner_and_plan_refresh_fail() {
    let backend_impl = Arc::new(LoadPlanFailBackend {
        inner: InMemoryBackend::new(),
    });
    let backend: Arc<dyn StorageBackendDyn + Send + Sync> = backend_impl.clone();
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    let learners: Vec<Box<dyn Learner>> = vec![Box::new(FailingLearner)];
    let handle = tokio::spawn(drain_task(
        rx,
        backend,
        hot_cache.clone(),
        "agent-drain".to_string(),
        learners,
    ));

    let start = make_agent_start();
    let end = make_agent_end(start.uuid());
    tx.send(start).unwrap();
    tx.send(end).unwrap();
    drop(tx);

    tokio::time::timeout(Duration::from_secs(2), handle)
        .await
        .expect("drain task should exit")
        .unwrap();

    assert_eq!(
        backend_impl
            .inner
            .list_runs("agent-drain")
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(hot_cache.read().unwrap().plan.is_none());
}

// -----------------------------------------------------------------------
// Annotated response extraction tests
// -----------------------------------------------------------------------

/// Helper: create an LlmEnd event with an annotated_response.
fn make_llm_end_with_annotated(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    annotated: nemo_relay::codec::response::AnnotatedLlmResponse,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(uuid)
            .name(name)
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::llm(),
        Some(
            nemo_relay::api::event::CategoryProfile::builder()
                .annotated_response(std::sync::Arc::new(annotated))
                .build(),
        ),
    ))
}

#[test]
fn test_accumulator_extracts_annotated_response() {
    use nemo_relay::codec::response::{AnnotatedLlmResponse, ResponseToolCall, Usage};

    let mut acc = RunAccumulator::new("agent-1".to_string());

    // Agent Start
    let agent_start = make_agent_start();
    let root_uuid = agent_start.uuid();
    acc.process_event(&agent_start);

    // LLM Start
    let llm_uuid = Uuid::now_v7();
    let llm_start = make_event(
        EventType::Start,
        Some(ScopeType::Llm),
        Some("gpt-4o"),
        llm_uuid,
        Some(root_uuid),
    );
    acc.process_event(&llm_start);

    // LLM End with full annotated response
    let annotated = AnnotatedLlmResponse {
        id: Some("chatcmpl-123".into()),
        model: Some("gpt-4o".into()),
        message: None,
        tool_calls: Some(vec![
            ResponseToolCall {
                id: "call_1".into(),
                name: "search".into(),
                arguments: serde_json::json!({"q": "test"}),
            },
            ResponseToolCall {
                id: "call_2".into(),
                name: "fetch".into(),
                arguments: serde_json::json!({"url": "http://example.com"}),
            },
        ]),
        finish_reason: None,
        usage: Some(Usage {
            prompt_tokens: Some(50),
            completion_tokens: Some(100),
            total_tokens: Some(150),
            cache_read_tokens: None,
            cache_write_tokens: None,
            cost: None,
        }),
        api_specific: None,
        optimization_summary: None,
        extra: serde_json::Map::new(),
    };

    let llm_end = make_llm_end_with_annotated(llm_uuid, Some(root_uuid), "gpt-4o", annotated);
    acc.process_event(&llm_end);

    // Agent End
    let run = acc
        .process_event(&make_agent_end(root_uuid))
        .expect("should return completed run");

    assert_eq!(run.calls.len(), 1);
    let call = &run.calls[0];
    assert_eq!(
        call.output_tokens,
        Some(100),
        "output_tokens from completion_tokens"
    );
    assert_eq!(call.prompt_tokens, Some(50), "prompt_tokens from usage");
    assert_eq!(call.total_tokens, Some(150), "total_tokens from usage");
    assert_eq!(
        call.model_name.as_deref(),
        Some("gpt-4o"),
        "model_name from annotated"
    );
    assert_eq!(
        call.tool_call_count,
        Some(2),
        "tool_call_count from tool_calls vec"
    );
}

#[test]
fn test_accumulator_llm_end_no_annotated_response() {
    let mut acc = RunAccumulator::new("agent-1".to_string());

    let agent_start = make_agent_start();
    let root_uuid = agent_start.uuid();
    acc.process_event(&agent_start);

    // LLM Start + LLM End without annotated (use existing make_event helper)
    let llm_uuid = Uuid::now_v7();
    let llm_start = make_event(
        EventType::Start,
        Some(ScopeType::Llm),
        Some("gpt-4"),
        llm_uuid,
        Some(root_uuid),
    );
    acc.process_event(&llm_start);

    let llm_end = make_event(
        EventType::End,
        Some(ScopeType::Llm),
        Some("gpt-4"),
        llm_uuid,
        Some(root_uuid),
    );
    acc.process_event(&llm_end);

    let run = acc
        .process_event(&make_agent_end(root_uuid))
        .expect("should return completed run");

    assert_eq!(run.calls.len(), 1);
    let call = &run.calls[0];
    assert!(
        call.output_tokens.is_none(),
        "output_tokens should be None without annotated"
    );
    assert!(
        call.prompt_tokens.is_none(),
        "prompt_tokens should be None without annotated"
    );
    assert!(
        call.total_tokens.is_none(),
        "total_tokens should be None without annotated"
    );
    assert!(
        call.model_name.is_none(),
        "model_name should be None without annotated"
    );
    assert!(
        call.tool_call_count.is_none(),
        "tool_call_count should be None without annotated"
    );
}

#[test]
fn test_accumulator_annotated_response_partial_data() {
    use nemo_relay::codec::response::AnnotatedLlmResponse;

    let mut acc = RunAccumulator::new("agent-1".to_string());

    let agent_start = make_agent_start();
    let root_uuid = agent_start.uuid();
    acc.process_event(&agent_start);

    let llm_uuid = Uuid::now_v7();
    let llm_start = make_event(
        EventType::Start,
        Some(ScopeType::Llm),
        Some("gpt-4o-mini"),
        llm_uuid,
        Some(root_uuid),
    );
    acc.process_event(&llm_start);

    // Annotated with model but no usage and no tool_calls
    let annotated = AnnotatedLlmResponse {
        id: None,
        model: Some("gpt-4o-mini".into()),
        message: None,
        tool_calls: None,
        finish_reason: None,
        usage: None,
        api_specific: None,
        optimization_summary: None,
        extra: serde_json::Map::new(),
    };

    let llm_end = make_llm_end_with_annotated(llm_uuid, Some(root_uuid), "gpt-4o-mini", annotated);
    acc.process_event(&llm_end);

    let run = acc
        .process_event(&make_agent_end(root_uuid))
        .expect("should return completed run");

    assert_eq!(run.calls.len(), 1);
    let call = &run.calls[0];
    assert_eq!(
        call.model_name.as_deref(),
        Some("gpt-4o-mini"),
        "model_name should be set"
    );
    assert!(
        call.prompt_tokens.is_none(),
        "prompt_tokens should be None when usage is None"
    );
    assert!(
        call.output_tokens.is_none(),
        "output_tokens should be None when usage is None"
    );
    assert!(
        call.total_tokens.is_none(),
        "total_tokens should be None when usage is None"
    );
    assert!(
        call.tool_call_count.is_none(),
        "tool_call_count should be None when tool_calls is None"
    );
}
