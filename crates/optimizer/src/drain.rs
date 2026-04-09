// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Background drain task for async telemetry processing.
//!
//! The drain task receives Nexus [`Event`]s from an unbounded mpsc channel,
//! accumulates them into [`RunRecord`]s via [`RunAccumulator`], and stores
//! completed runs through the [`StorageBackend`]. After each stored run it
//! refreshes the hot cache so intercepts can use up-to-date metadata.
//!
//! This module is the async half of the subscriber/drain pair. The subscriber
//! (see [`crate::subscriber`]) sends events non-blockingly; the drain task
//! processes them at its own pace without touching the execution hot path.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use uuid::Uuid;

use nvidia_nat_nexus_core::{Event, ScopeType};

use crate::learner::Learner;
use crate::storage::StorageBackendDyn;
use crate::subscriber::{event_to_call_record, is_run_boundary};
use crate::types::{HotCache, RunRecord};

/// Tracks in-flight agent runs. Event ancestry is used to infer each event's run root.
/// When the Agent scope End event arrives, the run is finalized and returned.
pub(crate) struct RunAccumulator {
    agent_id: String,
    open_runs: HashMap<Uuid, RunRecord>,
    event_roots: HashMap<Uuid, Uuid>,
}

impl RunAccumulator {
    /// Creates a new empty accumulator for the given agent.
    pub(crate) fn new(agent_id: String) -> Self {
        Self {
            agent_id,
            open_runs: HashMap::new(),
            event_roots: HashMap::new(),
        }
    }

    /// Returns the number of currently open (in-flight) runs.
    #[cfg(test)]
    pub(crate) fn open_run_count(&self) -> usize {
        self.open_runs.len()
    }

    /// Processes a single event and returns a completed [`RunRecord`] if a run
    /// has just ended.
    ///
    /// - Agent Start events create a new open run keyed by their own UUID.
    /// - Agent End events finalize and return the run.
    /// - Nested scope Start/End events inherit and clear root ownership.
    /// - LLM/Tool Start events create [`CallRecord`] entries in the open run.
    /// - LLM/Tool End events set `ended_at` on the matching call record.
    /// - All other events are ignored.
    pub(crate) fn process_event(&mut self, event: &Event) -> Option<RunRecord> {
        if is_run_boundary(event) {
            if matches!(event, Event::ScopeStart(_)) {
                let root_uuid = event.uuid();
                self.event_roots.insert(root_uuid, root_uuid);
                let run = RunRecord {
                    id: Uuid::new_v4(),
                    agent_id: self.agent_id.clone(),
                    calls: vec![],
                    started_at: *event.timestamp(),
                    ended_at: None,
                };
                self.open_runs.insert(root_uuid, run);
                return None;
            }

            if matches!(event, Event::ScopeEnd(_)) {
                let root_uuid = self
                    .event_roots
                    .remove(&event.uuid())
                    .unwrap_or_else(|| event.uuid());
                if let Some(mut run) = self.open_runs.remove(&root_uuid) {
                    run.ended_at = Some(*event.timestamp());
                    return Some(run);
                }
                // Orphaned end event -- no matching start
                return None;
            }
        }

        match event {
            Event::ScopeStart(inner) => {
                if inner.scope_type != ScopeType::Agent {
                    let root_uuid = self.infer_root_uuid(event)?;
                    self.event_roots.insert(event.uuid(), root_uuid);
                }
                None
            }
            Event::ScopeEnd(inner) => {
                if inner.scope_type != ScopeType::Agent {
                    self.event_roots.remove(&event.uuid());
                }
                None
            }
            Event::ToolStart(_) | Event::LLMStart(_) => {
                let root_uuid = self.infer_root_uuid(event)?;
                self.event_roots.insert(event.uuid(), root_uuid);
                if let Some(record) = event_to_call_record(event) {
                    if let Some(run) = self.open_runs.get_mut(&root_uuid) {
                        run.calls.push(record);
                    }
                }
                None
            }
            Event::ToolEnd(_) | Event::LLMEnd(_) => {
                let root_uuid = self.infer_root_uuid(event)?;
                if let Some(run) = self.open_runs.get_mut(&root_uuid) {
                    let event_name = event.name();
                    // Find the last matching call record (same name, not yet ended)
                    if let Some(call) = run
                        .calls
                        .iter_mut()
                        .rev()
                        .find(|c| c.name == event_name && c.ended_at.is_none())
                    {
                        call.ended_at = Some(*event.timestamp());

                        // Extract structured telemetry from annotated response
                        if let Event::LLMEnd(ref inner) = event {
                            if let Some(ref annotated) = inner.annotated_response {
                                if let Some(ref usage) = annotated.usage {
                                    call.output_tokens = usage.completion_tokens.map(|t| t as u32);
                                    call.prompt_tokens = usage.prompt_tokens.map(|t| t as u32);
                                    call.total_tokens = usage.total_tokens.map(|t| t as u32);
                                }
                                call.model_name = annotated.model.clone();
                                call.tool_call_count =
                                    annotated.tool_calls.as_ref().map(|tc| tc.len() as u32);
                            }
                        }
                    }
                }
                self.event_roots.remove(&event.uuid());
                None
            }
            Event::Mark(_) => None,
        }
    }

    fn infer_root_uuid(&self, event: &Event) -> Option<Uuid> {
        self.event_roots.get(&event.uuid()).copied().or_else(|| {
            event
                .parent_uuid()
                .and_then(|parent_uuid| self.event_roots.get(&parent_uuid).copied())
        })
    }
}

/// Background task that drains events from the telemetry channel, accumulates
/// them into [`RunRecord`]s, stores completed runs, and refreshes the hot cache.
///
/// Exits cleanly when the channel sender is dropped (optimizer shutting down).
pub(crate) async fn drain_task(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    backend: Arc<dyn StorageBackendDyn + Send + Sync>,
    hot_cache: Arc<RwLock<HotCache>>,
    agent_id: String,
    learners: Vec<Box<dyn Learner>>,
) {
    let mut accumulator = RunAccumulator::new(agent_id.clone());

    while let Some(event) = rx.recv().await {
        if let Some(completed_run) = accumulator.process_event(&event) {
            // Store the completed run (async, not on hot path)
            if let Err(e) = backend.store_run_dyn(&completed_run).await {
                // Log error but continue -- don't crash the drain
                eprintln!("nexus-optimizer drain: store_run failed: {e}");
                continue;
            }

            // Invoke learner pipeline (each learner updates hot_cache internally)
            for learner in &learners {
                if let Err(e) = learner
                    .process_run(&completed_run, backend.as_ref(), &hot_cache)
                    .await
                {
                    eprintln!("nexus-optimizer drain: learner failed: {e}");
                    // Continue -- one learner failure doesn't stop others
                }
            }

            // Update hot cache with latest plan from backend
            match backend.load_plan_dyn(&agent_id).await {
                Ok(plan) => {
                    if let Ok(mut guard) = hot_cache.write() {
                        guard.plan = plan;
                    }
                }
                Err(e) => {
                    eprintln!("nexus-optimizer drain: load_plan failed: {e}");
                }
            }
        }
    }
    // Channel closed -- sender dropped, optimizer shutting down. Exit cleanly.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{InMemoryBackend, StorageBackend, StorageBackendDyn};
    use crate::types::{ExecutionPlan, HotCache, MetadataEnvelope, ParallelGroup};
    use nvidia_nat_nexus_core::{Event, ScopeType};
    use serde_json::json;
    use std::time::Duration;
    use uuid::Uuid;

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
            (EventType::Start, Some(ScopeType::Tool)) => Event::tool_start(
                parent_uuid,
                uuid,
                event_name,
                None,
                None,
                nvidia_nat_nexus_core::ToolAttributes::empty(),
                None,
                None,
            ),
            (EventType::End, Some(ScopeType::Tool)) => Event::tool_end(
                parent_uuid,
                uuid,
                event_name,
                None,
                None,
                nvidia_nat_nexus_core::ToolAttributes::empty(),
                None,
                None,
            ),
            (EventType::Start, Some(ScopeType::Llm)) => Event::llm_start(
                parent_uuid,
                uuid,
                event_name,
                None,
                None,
                nvidia_nat_nexus_core::LLMAttributes::empty(),
                None,
                None,
                None,
            ),
            (EventType::End, Some(ScopeType::Llm)) => Event::llm_end(
                parent_uuid,
                uuid,
                event_name,
                None,
                None,
                nvidia_nat_nexus_core::LLMAttributes::empty(),
                None,
                None,
                None,
            ),
            (EventType::Start, Some(scope_type)) => Event::scope_start(
                parent_uuid,
                uuid,
                event_name,
                None,
                None,
                nvidia_nat_nexus_core::ScopeAttributes::empty(),
                scope_type,
            ),
            (EventType::End, Some(scope_type)) => Event::scope_end(
                parent_uuid,
                uuid,
                event_name,
                None,
                None,
                nvidia_nat_nexus_core::ScopeAttributes::empty(),
                scope_type,
            ),
            (_, None) => Event::mark(parent_uuid, uuid, event_name, None, None),
        }
    }

    /// Helper: make an Agent Start event whose own uuid acts as the inferred root.
    fn make_agent_start() -> Event {
        let uuid = Uuid::new_v4();
        Event::scope_start(
            None,
            uuid,
            "my-agent",
            None,
            None,
            nvidia_nat_nexus_core::ScopeAttributes::empty(),
            ScopeType::Agent,
        )
    }

    /// Helper: make an Agent End event for a given root event UUID.
    fn make_agent_end(root_uuid: Uuid) -> Event {
        Event::scope_end(
            None,
            root_uuid,
            "my-agent",
            None,
            None,
            nvidia_nat_nexus_core::ScopeAttributes::empty(),
            ScopeType::Agent,
        )
    }

    fn make_test_plan(agent_id: &str) -> ExecutionPlan {
        ExecutionPlan {
            agent_id: agent_id.to_string(),
            parallel_groups: vec![ParallelGroup {
                group_id: "pg-1".to_string(),
                tool_names: vec!["search".to_string()],
            }],
            metadata_template: MetadataEnvelope {
                run_id: Uuid::new_v4(),
                agent_id: agent_id.to_string(),
                parallel_hints: vec![],
                extensions: json!({}),
            },
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
        assert_eq!(acc.open_run_count(), 0);
    }

    #[test]
    fn test_accumulator_collects_calls() {
        let mut acc = RunAccumulator::new("agent-1".to_string());

        let start = make_agent_start();
        let root_uuid = start.uuid();
        acc.process_event(&start);

        // Tool Start + Tool End
        let tool_uuid = Uuid::new_v4();
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
        let llm_uuid = Uuid::new_v4();
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
    fn test_accumulator_tracks_calls_nested_under_non_agent_scope() {
        let mut acc = RunAccumulator::new("agent-1".to_string());

        let agent_start = make_agent_start();
        let root_uuid = agent_start.uuid();
        acc.process_event(&agent_start);

        let function_uuid = Uuid::new_v4();
        let function_start = make_event(
            EventType::Start,
            Some(ScopeType::Function),
            Some("helper"),
            function_uuid,
            Some(root_uuid),
        );
        acc.process_event(&function_start);

        let tool_uuid = Uuid::new_v4();
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

        let function_uuid = Uuid::new_v4();
        let function_start = make_event(
            EventType::Start,
            Some(ScopeType::Function),
            Some("helper"),
            function_uuid,
            Some(root_uuid),
        );
        acc.process_event(&function_start);

        let llm_uuid = Uuid::new_v4();
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
        let end = make_agent_end(Uuid::new_v4());
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

    // -----------------------------------------------------------------------
    // Annotated response extraction tests
    // -----------------------------------------------------------------------

    /// Helper: create an LLMEnd event with an annotated_response.
    fn make_llm_end_with_annotated(
        uuid: Uuid,
        parent_uuid: Option<Uuid>,
        name: &str,
        annotated: nvidia_nat_nexus_core::AnnotatedLLMResponse,
    ) -> Event {
        Event::llm_end(
            parent_uuid,
            uuid,
            name,
            None,
            None,
            nvidia_nat_nexus_core::LLMAttributes::empty(),
            None,
            None,
            Some(std::sync::Arc::new(annotated)),
        )
    }

    #[test]
    fn test_accumulator_extracts_annotated_response() {
        use nvidia_nat_nexus_core::{AnnotatedLLMResponse, ResponseToolCall, Usage};

        let mut acc = RunAccumulator::new("agent-1".to_string());

        // Agent Start
        let agent_start = make_agent_start();
        let root_uuid = agent_start.uuid();
        acc.process_event(&agent_start);

        // LLM Start
        let llm_uuid = Uuid::new_v4();
        let llm_start = make_event(
            EventType::Start,
            Some(ScopeType::Llm),
            Some("gpt-4o"),
            llm_uuid,
            Some(root_uuid),
        );
        acc.process_event(&llm_start);

        // LLM End with full annotated response
        let annotated = AnnotatedLLMResponse {
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
            }),
            api_specific: None,
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
        let llm_uuid = Uuid::new_v4();
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
        use nvidia_nat_nexus_core::AnnotatedLLMResponse;

        let mut acc = RunAccumulator::new("agent-1".to_string());

        let agent_start = make_agent_start();
        let root_uuid = agent_start.uuid();
        acc.process_event(&agent_start);

        let llm_uuid = Uuid::new_v4();
        let llm_start = make_event(
            EventType::Start,
            Some(ScopeType::Llm),
            Some("gpt-4o-mini"),
            llm_uuid,
            Some(root_uuid),
        );
        acc.process_event(&llm_start);

        // Annotated with model but no usage and no tool_calls
        let annotated = AnnotatedLLMResponse {
            id: None,
            model: Some("gpt-4o-mini".into()),
            message: None,
            tool_calls: None,
            finish_reason: None,
            usage: None,
            api_specific: None,
            extra: serde_json::Map::new(),
        };

        let llm_end =
            make_llm_end_with_annotated(llm_uuid, Some(root_uuid), "gpt-4o-mini", annotated);
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
}
