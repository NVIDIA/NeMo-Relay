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

use nvidia_nat_nexus_core::{Event, EventType, ScopeType};

use crate::learner::Learner;
use crate::storage::StorageBackendDyn;
use crate::subscriber::{event_to_call_record, is_run_boundary};
use crate::types::{HotCache, RunRecord};

/// Tracks in-flight agent runs. Events are grouped by `root_uuid`.
/// When the Agent scope End event arrives, the run is finalized and returned.
pub(crate) struct RunAccumulator {
    agent_id: String,
    open_runs: HashMap<Uuid, RunRecord>,
}

impl RunAccumulator {
    /// Creates a new empty accumulator for the given agent.
    pub(crate) fn new(agent_id: String) -> Self {
        Self {
            agent_id,
            open_runs: HashMap::new(),
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
    /// - Agent Start events create a new open run keyed by `root_uuid`.
    /// - Agent End events finalize and return the run.
    /// - LLM/Tool Start events create [`CallRecord`] entries in the open run.
    /// - LLM/Tool End events set `ended_at` on the matching call record.
    /// - All other events are ignored.
    pub(crate) fn process_event(&mut self, event: &Event) -> Option<RunRecord> {
        if is_run_boundary(event) {
            let root_uuid = event.root_uuid.unwrap_or(event.uuid);

            if event.event_type == EventType::Start {
                let run = RunRecord {
                    id: Uuid::new_v4(),
                    agent_id: self.agent_id.clone(),
                    calls: vec![],
                    started_at: event.timestamp,
                    ended_at: None,
                };
                self.open_runs.insert(root_uuid, run);
                return None;
            }

            if event.event_type == EventType::End {
                if let Some(mut run) = self.open_runs.remove(&root_uuid) {
                    run.ended_at = Some(event.timestamp);
                    return Some(run);
                }
                // Orphaned end event -- no matching start
                return None;
            }
        }

        // LLM/Tool events: accumulate into the open run
        let root_uuid = event.root_uuid?;

        if event.event_type == EventType::Start {
            if let Some(record) = event_to_call_record(event) {
                if let Some(run) = self.open_runs.get_mut(&root_uuid) {
                    run.calls.push(record);
                }
            }
        } else if event.event_type == EventType::End
            && matches!(
                event.scope_type,
                Some(ScopeType::Llm) | Some(ScopeType::Tool)
            )
        {
            if let Some(run) = self.open_runs.get_mut(&root_uuid) {
                let event_name = event.name.as_deref().unwrap_or("");
                // Find the last matching call record (same name, not yet ended)
                if let Some(call) = run
                    .calls
                    .iter_mut()
                    .rev()
                    .find(|c| c.name == event_name && c.ended_at.is_none())
                {
                    call.ended_at = Some(event.timestamp);
                }
            }
        }

        None
    }
}

/// Background task that drains events from the telemetry channel, accumulates
/// them into [`RunRecord`]s, stores completed runs, and refreshes the hot cache.
///
/// Exits cleanly when the channel sender is dropped (proxy shutting down).
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
            // Store the completed run (WIRE-04: async, not on hot path)
            if let Err(e) = backend.store_run_dyn(&completed_run).await {
                // Log error but continue -- don't crash the drain
                eprintln!("nexus-proxy drain: store_run failed: {e}");
                continue;
            }

            // Invoke learner pipeline (each learner updates hot_cache internally)
            for learner in &learners {
                if let Err(e) = learner
                    .process_run(&completed_run, backend.as_ref(), &hot_cache)
                    .await
                {
                    eprintln!("nexus-proxy drain: learner failed: {e}");
                    // Continue -- one learner failure doesn't stop others
                }
            }

            // Update hot cache with latest plan from backend (WIRE-05)
            match backend.load_plan_dyn(&agent_id).await {
                Ok(plan) => {
                    if let Ok(mut guard) = hot_cache.write() {
                        guard.plan = plan;
                    }
                }
                Err(e) => {
                    eprintln!("nexus-proxy drain: load_plan failed: {e}");
                }
            }
        }
    }
    // Channel closed -- sender dropped, proxy shutting down. Exit cleanly.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{InMemoryBackend, StorageBackend, StorageBackendDyn};
    use crate::types::{ExecutionPlan, HotCache, MetadataEnvelope, ParallelGroup};
    use chrono::Utc;
    use nvidia_nat_nexus_core::{Event, EventType, ScopeType};
    use serde_json::json;
    use std::time::Duration;
    use uuid::Uuid;

    /// Helper to construct a minimal test [`Event`] with `root_uuid` parameter.
    fn make_event(
        event_type: EventType,
        scope_type: Option<ScopeType>,
        name: Option<&str>,
        root_uuid: Option<Uuid>,
    ) -> Event {
        Event {
            parent_uuid: None,
            uuid: Uuid::new_v4(),
            timestamp: Utc::now(),
            name: name.map(|s| s.to_string()),
            data: None,
            metadata: None,
            attributes: None,
            event_type,
            scope_type,
            input: None,
            output: None,
            model_name: None,
            tool_call_id: None,
            root_uuid,
        }
    }

    /// Helper: make an Agent Start event whose own uuid acts as root_uuid.
    fn make_agent_start() -> Event {
        let uuid = Uuid::new_v4();
        Event {
            parent_uuid: None,
            uuid,
            timestamp: Utc::now(),
            name: Some("my-agent".to_string()),
            data: None,
            metadata: None,
            attributes: None,
            event_type: EventType::Start,
            scope_type: Some(ScopeType::Agent),
            input: None,
            output: None,
            model_name: None,
            tool_call_id: None,
            root_uuid: None, // root scope's own uuid IS the root_uuid
        }
    }

    /// Helper: make an Agent End event for a given root_uuid.
    fn make_agent_end(root_uuid: Uuid) -> Event {
        Event {
            parent_uuid: None,
            uuid: Uuid::new_v4(),
            timestamp: Utc::now(),
            name: Some("my-agent".to_string()),
            data: None,
            metadata: None,
            attributes: None,
            event_type: EventType::End,
            scope_type: Some(ScopeType::Agent),
            input: None,
            output: None,
            model_name: None,
            tool_call_id: None,
            root_uuid: Some(root_uuid),
        }
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
        let root_uuid = start.uuid; // root scope uuid is root_uuid
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
        let root_uuid = start.uuid;
        acc.process_event(&start);

        // Tool Start + Tool End
        let tool_start = make_event(
            EventType::Start,
            Some(ScopeType::Tool),
            Some("search"),
            Some(root_uuid),
        );
        acc.process_event(&tool_start);

        let tool_end = make_event(
            EventType::End,
            Some(ScopeType::Tool),
            Some("search"),
            Some(root_uuid),
        );
        acc.process_event(&tool_end);

        // LLM Start + LLM End
        let llm_start = make_event(
            EventType::Start,
            Some(ScopeType::Llm),
            Some("gpt-4"),
            Some(root_uuid),
        );
        acc.process_event(&llm_start);

        let llm_end = make_event(
            EventType::End,
            Some(ScopeType::Llm),
            Some("gpt-4"),
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
        let root_uuid = start.uuid;
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
        let root_uuid = start.uuid;
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
}
