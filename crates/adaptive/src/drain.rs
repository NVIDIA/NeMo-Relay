// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Background drain task for async telemetry processing.
//!
//! The drain task receives NeMo Flow [`Event`]s from an unbounded mpsc channel,
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

use nemo_flow::types::event::Event;
use nemo_flow::types::scope::ScopeType;

use crate::learner::traits::Learner;
use crate::storage::traits::StorageBackendDyn;
use crate::subscriber::{event_to_call_record, is_run_boundary};
use crate::types::cache::HotCache;
use crate::types::records::RunRecord;

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
                    id: Uuid::now_v7(),
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
                if let Some(record) = event_to_call_record(event)
                    && let Some(run) = self.open_runs.get_mut(&root_uuid)
                {
                    run.calls.push(record);
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
                        if let Event::LLMEnd(inner) = event
                            && let Some(ref annotated) = inner.annotated_response
                        {
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
/// Exits cleanly when the channel sender is dropped (adaptive shutting down).
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
            let store_result = backend.store_run_dyn(&completed_run).await;
            if let Err(e) = store_result {
                // Log error but continue -- don't crash the drain
                eprintln!("nemo-flow-adaptive drain: store_run failed: {e}");
                continue;
            }

            // Invoke learner pipeline (each learner updates hot_cache internally)
            for learner in &learners {
                let learner_result = learner
                    .process_run(&completed_run, backend.as_ref(), &hot_cache)
                    .await;
                if let Err(e) = learner_result {
                    eprintln!("nemo-flow-adaptive drain: learner failed: {e}");
                    // Continue -- one learner failure doesn't stop others
                }
            }

            // Update hot cache with latest plan from backend
            let plan_result = backend.load_plan_dyn(&agent_id).await;
            match plan_result {
                Ok(plan) => {
                    let hot_cache_write = hot_cache.write();
                    if let Ok(mut guard) = hot_cache_write {
                        guard.plan = plan;
                    }
                }
                Err(e) => {
                    eprintln!("nemo-flow-adaptive drain: load_plan failed: {e}");
                }
            }
        }
    }
    // Channel closed -- sender dropped, adaptive shutting down. Exit cleanly.
}

#[cfg(test)]
#[path = "../tests/unit/drain_tests.rs"]
mod tests;
