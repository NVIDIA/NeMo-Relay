// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Idle-session sweeping and shutdown closure.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nemo_relay_runtime::api::runtime::TASK_SCOPE_STACK;
use tokio::sync::{Mutex, OwnedMutexGuard};

use crate::agents::shared::alignment::SessionAlignmentState;
use crate::error::CliError;

use super::{Session, SessionGates, session_gate};

pub(super) const AGENT_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
// Hook-driven tools do not send heartbeats while running. Allow long builds and tests to
// finish, while retaining a finite fail-safe for children whose completion hooks never arrive.
pub(super) const CODEX_TOOL_IDLE_TIMEOUT: Duration = Duration::from_secs(60 * 60);
pub(super) const AGENT_IDLE_SWEEP_INTERVAL: Duration = Duration::from_secs(5);

pub(super) async fn close_sessions_for_shutdown(
    sessions: &mut [Session],
    reason: &str,
) -> Result<(), CliError> {
    let mut first_error = None;
    for session in sessions {
        if let Err(error) = session.close_for_shutdown(reason).await
            && first_error.is_none()
        {
            first_error = Some(error);
        }
    }
    first_error.map_or(Ok(()), Err)
}

pub(super) async fn close_idle_sessions_from_parts(
    inner: &Arc<Mutex<HashMap<String, Session>>>,
    session_gates: &SessionGates,
    authenticated_owners: &Arc<Mutex<HashMap<String, String>>>,
    alignment: &Arc<Mutex<SessionAlignmentState>>,
    now: Instant,
    timeout: Duration,
    reason: &str,
) -> Result<usize, CliError> {
    // Acquisitions clone under this same map lock. A map-only owner therefore has no
    // holder or waiter, and a later acquisition can safely create a replacement gate.
    // Run even when there are no idle sessions so finished sessions leave no stale gates.
    session_gates
        .lock()
        .await
        .retain(|_, gate| Arc::strong_count(gate) > 1);
    let candidate_ids = idle_session_ids(inner, now, timeout).await;
    if candidate_ids.is_empty() {
        return Ok(0);
    }
    let (closed_turns, released_owner_ids, first_error) = close_idle_turns(
        inner,
        session_gates,
        alignment,
        candidate_ids,
        now,
        timeout,
        reason,
    )
    .await;
    release_closed_owner_ids(
        inner,
        session_gates,
        authenticated_owners,
        &released_owner_ids,
    )
    .await;
    first_error.map_or(Ok(closed_turns), Err)
}

pub(super) async fn release_closed_owner_ids(
    inner: &Arc<Mutex<HashMap<String, Session>>>,
    session_gates: &SessionGates,
    authenticated_owners: &Arc<Mutex<HashMap<String, String>>>,
    released_owner_ids: &HashSet<String>,
) {
    for session_id in released_owner_ids {
        // The gate protects this presence check and deletion from a hook recreating the same
        // session while it is applying middleware.
        let gate = session_gate(session_gates, session_id).await;
        let _gate = gate.lock().await;
        if !inner.lock().await.contains_key(session_id) {
            authenticated_owners.lock().await.remove(session_id);
        }
    }
}

/// Select sessions needing whole-session or stale-child idle cleanup.
async fn idle_session_ids(
    inner: &Arc<Mutex<HashMap<String, Session>>>,
    now: Instant,
    timeout: Duration,
) -> Vec<String> {
    inner
        .lock()
        .await
        .iter()
        .filter_map(|(session_id, session)| {
            session
                .needs_idle_cleanup(now, timeout)
                .then_some(session_id.clone())
        })
        .collect()
}

// Finish route cleanup on cancellation while retaining the parent gate, as hook cleanup does.
struct IdleAliasCleanup {
    alignment: Arc<Mutex<SessionAlignmentState>>,
    session_id: String,
    subagent_ids: Vec<String>,
    session_gate: Option<OwnedMutexGuard<()>>,
}

impl IdleAliasCleanup {
    async fn commit(mut self) {
        if self.subagent_ids.is_empty() {
            return;
        }
        let mut alignment = self.alignment.lock().await;
        for identifier in &self.subagent_ids {
            alignment.clear_for_ended_subagent(&self.session_id, identifier);
        }
        self.subagent_ids.clear();
    }
}

impl Drop for IdleAliasCleanup {
    fn drop(&mut self) {
        if self.subagent_ids.is_empty() {
            return;
        }
        let alignment = Arc::clone(&self.alignment);
        let session_id = self.session_id.clone();
        let subagent_ids = std::mem::take(&mut self.subagent_ids);
        let session_gate = self.session_gate.take();
        drop(tokio::spawn(async move {
            let mut alignment = alignment.lock().await;
            for identifier in subagent_ids {
                alignment.clear_for_ended_subagent(&session_id, &identifier);
            }
            drop(alignment);
            drop(session_gate);
        }));
    }
}

type ClosedIdleTurns = (usize, HashSet<String>, Option<CliError>);

/// Close expired children independently, then close sessions whose remaining work is idle.
async fn close_idle_turns(
    inner: &Arc<Mutex<HashMap<String, Session>>>,
    session_gates: &SessionGates,
    alignment: &Arc<Mutex<SessionAlignmentState>>,
    candidate_ids: Vec<String>,
    now: Instant,
    timeout: Duration,
    reason: &str,
) -> ClosedIdleTurns {
    let mut closed_turns = 0;
    let mut released_owner_ids = HashSet::new();
    let mut first_error = None;
    for session_id in candidate_ids {
        let gate = session_gate(session_gates, &session_id).await;
        let session_gate = gate.lock_owned().await;
        let Some(mut session) = ({
            let mut sessions = inner.lock().await;
            sessions
                .get(&session_id)
                .is_some_and(|session| session.needs_idle_cleanup(now, timeout))
                .then(|| sessions.remove(&session_id))
                .flatten()
        }) else {
            continue;
        };
        let retiring_ids = session.subagent_stack.clone();
        let stack = session.scope_stack.clone();
        match TASK_SCOPE_STACK
            .scope(stack, async {
                let (_, mut delivery) = session.close_idle_subagents(now, timeout).await?;
                if session.is_idle_for(now, timeout) {
                    let (_, idle_delivery) = session.close_idle_scopes_for_reason(reason).await?;
                    delivery = idle_delivery.or(delivery);
                }
                Ok::<_, CliError>(delivery)
            })
            .await
        {
            Ok(subscriber_delivery) => {
                closed_turns += 1;
                if let Some(subscriber_delivery) = subscriber_delivery
                    && let Err(error) = subscriber_delivery.wait().await
                    && first_error.is_none()
                {
                    first_error = Some(error.into());
                }
            }
            Err(error) => {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        let closed_subagent_ids = retiring_ids
            .into_iter()
            .filter(|identifier| {
                session.completed_subagents.contains(identifier)
                    && !session.subagents.contains_key(identifier)
                    && !session.retiring_subagents.contains_key(identifier)
            })
            .collect();
        // Restore before the fallible/cancellable alignment wait. Cleanup retains the gate.
        if !session.is_empty() {
            inner.lock().await.insert(session_id.clone(), session);
        } else {
            released_owner_ids.insert(session_id.clone());
        }
        IdleAliasCleanup {
            alignment: Arc::clone(alignment),
            session_id,
            subagent_ids: closed_subagent_ids,
            session_gate: Some(session_gate),
        }
        .commit()
        .await;
    }
    (closed_turns, released_owner_ids, first_error)
}
