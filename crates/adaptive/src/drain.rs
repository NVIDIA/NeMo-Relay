// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Background drain task for async telemetry processing.

use std::collections::HashMap;
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicUsize, Ordering},
};

use nemo_relay::api::event::{Event, ScopeCategory};
use nemo_relay::api::scope::ScopeType;
use uuid::Uuid;

use crate::learner::traits::Learner;
use crate::storage::traits::StorageBackendDyn;
use crate::subscriber::{event_to_call_record, is_run_boundary};
use crate::types::cache::HotCache;
use crate::types::records::{CallRecord, RunRecord};

pub(crate) struct RunAccumulator {
    agent_id: String,
    open_runs: HashMap<Uuid, RunRecord>,
    event_roots: HashMap<Uuid, Uuid>,
}

impl RunAccumulator {
    pub(crate) fn new(agent_id: String) -> Self {
        Self {
            agent_id,
            open_runs: HashMap::new(),
            event_roots: HashMap::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn open_run_count(&self) -> usize {
        self.open_runs.len()
    }

    pub(crate) fn process_event(&mut self, event: &Event) -> Option<RunRecord> {
        if let Some(boundary_result) = self.process_run_boundary(event) {
            return boundary_result;
        }

        match (event.scope_category(), event.scope_type()) {
            (Some(ScopeCategory::Start), Some(ScopeType::Tool | ScopeType::Llm)) => {
                self.track_call_start(event)?;
                None
            }
            (Some(ScopeCategory::End), Some(ScopeType::Tool | ScopeType::Llm)) => {
                self.track_call_end(event)?;
                None
            }
            (Some(ScopeCategory::Start), Some(scope_type)) => {
                self.track_nested_scope_start(event, scope_type)?;
                None
            }
            (Some(ScopeCategory::End), Some(scope_type)) => {
                self.track_nested_scope_end(event, scope_type);
                None
            }
            _ => None,
        }
    }

    fn process_run_boundary(&mut self, event: &Event) -> Option<Option<RunRecord>> {
        if !is_run_boundary(event) {
            return None;
        }

        if event.scope_category() == Some(ScopeCategory::Start) {
            self.start_run(event);
            return Some(None);
        }

        Some(self.finish_run(event))
    }

    fn start_run(&mut self, event: &Event) {
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
    }

    fn finish_run(&mut self, event: &Event) -> Option<RunRecord> {
        let root_uuid = self
            .event_roots
            .remove(&event.uuid())
            .unwrap_or_else(|| event.uuid());
        let mut run = self.open_runs.remove(&root_uuid)?;
        run.ended_at = Some(*event.timestamp());
        Some(run)
    }

    fn track_nested_scope_start(&mut self, event: &Event, scope_type: ScopeType) -> Option<()> {
        if scope_type != ScopeType::Agent {
            let root_uuid = self.infer_root_uuid(event)?;
            self.event_roots.insert(event.uuid(), root_uuid);
        }
        Some(())
    }

    fn track_nested_scope_end(&mut self, event: &Event, scope_type: ScopeType) {
        if scope_type != ScopeType::Agent {
            self.event_roots.remove(&event.uuid());
        }
    }

    fn track_call_start(&mut self, event: &Event) -> Option<()> {
        let root_uuid = self.infer_root_uuid(event)?;
        self.event_roots.insert(event.uuid(), root_uuid);
        if let Some(record) = event_to_call_record(event)
            && let Some(run) = self.open_runs.get_mut(&root_uuid)
        {
            run.calls.push(record);
        }
        Some(())
    }

    fn track_call_end(&mut self, event: &Event) -> Option<()> {
        let root_uuid = self.infer_root_uuid(event)?;
        if let Some(run) = self.open_runs.get_mut(&root_uuid)
            && let Some(call) = find_open_call(run, event.name())
        {
            call.ended_at = Some(*event.timestamp());
            apply_llm_end_metadata(call, event);
        }
        self.event_roots.remove(&event.uuid());
        Some(())
    }

    fn infer_root_uuid(&self, event: &Event) -> Option<Uuid> {
        self.event_roots.get(&event.uuid()).copied().or_else(|| {
            event
                .parent_uuid()
                .and_then(|parent_uuid| self.event_roots.get(&parent_uuid).copied())
        })
    }
}

fn find_open_call<'a>(run: &'a mut RunRecord, event_name: &str) -> Option<&'a mut CallRecord> {
    run.calls
        .iter_mut()
        .rev()
        .find(|call| call.name == event_name && call.ended_at.is_none())
}

fn apply_llm_end_metadata(call: &mut CallRecord, event: &Event) {
    if event.category().map(|category| category.as_str()) != Some("llm") {
        return;
    }
    call.annotated_response = event.annotated_response().cloned();
    let Some(ref annotated) = call.annotated_response else {
        return;
    };

    if let Some(ref usage) = annotated.usage {
        call.output_tokens = usage.completion_tokens.map(|tokens| tokens as u32);
        call.prompt_tokens = usage.prompt_tokens.map(|tokens| tokens as u32);
        call.total_tokens = usage.total_tokens.map(|tokens| tokens as u32);
    }
    call.model_name = annotated.model.clone();
    call.tool_call_count = annotated
        .tool_calls
        .as_ref()
        .map(|calls| calls.len() as u32);
}

async fn store_run(
    backend: &Arc<dyn StorageBackendDyn + Send + Sync>,
    completed_run: &RunRecord,
) -> bool {
    if backend.store_run_dyn(completed_run).await.is_err() {
        log::warn!(
            target: "nemo_relay.runtime",
            event = "adaptive_run_storage_failed";
            "Adaptive runtime could not persist a completed run"
        );
        return false;
    }
    true
}

async fn run_learners(
    learners: &[Box<dyn Learner>],
    completed_run: &RunRecord,
    backend: &Arc<dyn StorageBackendDyn + Send + Sync>,
    hot_cache: &Arc<RwLock<HotCache>>,
) {
    for learner in learners {
        if learner
            .process_run(completed_run, backend.as_ref(), hot_cache)
            .await
            .is_err()
        {
            log::warn!(
                target: "nemo_relay.runtime",
                event = "adaptive_learner_failed";
                "Adaptive runtime learner failed while processing a completed run"
            );
        }
    }
}

async fn refresh_hot_cache_plan(
    backend: &Arc<dyn StorageBackendDyn + Send + Sync>,
    hot_cache: &Arc<RwLock<HotCache>>,
    agent_id: &str,
) {
    match backend.load_plan_dyn(agent_id).await {
        Ok(plan) => {
            if let Ok(mut guard) = hot_cache.write() {
                guard.plan = plan;
            }
        }
        Err(_) => log::warn!(
            target: "nemo_relay.runtime",
            event = "adaptive_plan_refresh_failed";
            "Adaptive runtime could not refresh the cached plan"
        ),
    }
}

/// Background task that drains events from the telemetry channel, accumulates
/// them into [`RunRecord`]s, stores completed runs, and refreshes the hot cache.
///
/// Exits cleanly when the channel sender is dropped (adaptive shutting down).
///
/// Convenience wrapper around [`drain_task_with_counter`] used by tests. The
/// adaptive runtime spawns `drain_task_with_counter` directly so it can observe
/// the in-flight event counter.
#[allow(dead_code)]
pub(crate) async fn drain_task(
    rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    backend: Arc<dyn StorageBackendDyn + Send + Sync>,
    hot_cache: Arc<RwLock<HotCache>>,
    agent_id: String,
    learners: Vec<Box<dyn Learner>>,
) {
    drain_task_with_counter(
        rx,
        backend,
        hot_cache,
        Arc::new(AtomicUsize::new(0)),
        agent_id,
        learners,
    )
    .await;
}

pub(crate) async fn drain_task_with_counter(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    backend: Arc<dyn StorageBackendDyn + Send + Sync>,
    hot_cache: Arc<RwLock<HotCache>>,
    pending_events: Arc<AtomicUsize>,
    agent_id: String,
    learners: Vec<Box<dyn Learner>>,
) {
    let mut accumulator = RunAccumulator::new(agent_id.clone());

    while let Some(event) = rx.recv().await {
        if let Some(completed_run) = accumulator.process_event(&event) {
            if !store_run(&backend, &completed_run).await {
                pending_events.fetch_sub(1, Ordering::SeqCst);
                continue;
            }

            run_learners(&learners, &completed_run, &backend, &hot_cache).await;
            refresh_hot_cache_plan(&backend, &hot_cache, &agent_id).await;
        }
        pending_events.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
#[path = "../tests/unit/drain_tests.rs"]
mod tests;
