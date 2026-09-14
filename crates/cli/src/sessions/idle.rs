// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Idle-session sweeping and shutdown closure.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nemo_relay::api::runtime::TASK_SCOPE_STACK;
use tokio::sync::Mutex;

use crate::agents::shared::alignment::SessionAlignmentState;
use crate::error::CliError;

use super::{Session, SessionGates, session_gate};

pub(super) const AGENT_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
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
    let candidate_ids = idle_session_ids(inner, now, timeout).await;
    if candidate_ids.is_empty() {
        return Ok(0);
    }
    let (closed_turns, closed_subagents, released_owner_ids, first_error) =
        close_idle_turns(inner, session_gates, candidate_ids, now, timeout, reason).await;
    let cleanup_sessions = closed_subagents
        .iter()
        .map(|(session_id, _)| session_id.clone())
        .collect();
    clear_closed_subagents(alignment, closed_subagents, &cleanup_sessions).await;
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
                .is_idle_for(now, timeout)
                .then_some(session_id.clone())
        })
        .collect()
}

type ClosedIdleTurns = (
    usize,
    Vec<(String, String)>,
    HashSet<String>,
    Option<CliError>,
);

async fn close_idle_turns(
    inner: &Arc<Mutex<HashMap<String, Session>>>,
    session_gates: &SessionGates,
    candidate_ids: Vec<String>,
    now: Instant,
    timeout: Duration,
    reason: &str,
) -> ClosedIdleTurns {
    let mut closed_turns = 0;
    let mut closed_subagents = Vec::new();
    let mut released_owner_ids = HashSet::new();
    let mut first_error = None;
    for session_id in candidate_ids {
        let gate = session_gate(session_gates, &session_id).await;
        let _gate = gate.lock().await;
        let Some(mut session) = ({
            let mut sessions = inner.lock().await;
            sessions
                .get(&session_id)
                .is_some_and(|session| session.is_idle_for(now, timeout))
                .then(|| sessions.remove(&session_id))
                .flatten()
        }) else {
            continue;
        };
        let stack = session.scope_stack.clone();
        match TASK_SCOPE_STACK
            .scope(stack, async {
                session.close_idle_scopes_for_reason(reason).await
            })
            .await
        {
            Ok((subagent_ids, subscriber_delivery)) => {
                closed_turns += 1;
                closed_subagents.extend(
                    subagent_ids
                        .into_iter()
                        .map(|subagent_id| (session_id.clone(), subagent_id)),
                );
                if let Some(subscriber_delivery) = subscriber_delivery
                    && let Err(error) = subscriber_delivery.wait().await
                    && first_error.is_none()
                {
                    first_error = Some(error.into());
                }
            }
            Err(error) if first_error.is_none() => first_error = Some(error),
            Err(_) => {}
        }
        if !session.is_empty() {
            inner.lock().await.insert(session_id, session);
        } else {
            released_owner_ids.insert(session_id);
        }
    }
    (
        closed_turns,
        closed_subagents,
        released_owner_ids,
        first_error,
    )
}

async fn clear_closed_subagents(
    alignment: &Arc<Mutex<SessionAlignmentState>>,
    closed_subagents: Vec<(String, String)>,
    cleanup_sessions: &HashSet<String>,
) {
    if closed_subagents.is_empty() {
        return;
    }
    let mut alignment_state = alignment.lock().await;
    for (session_id, subagent_id) in closed_subagents {
        if cleanup_sessions.contains(&session_id) {
            alignment_state.clear_for_ended_subagent(&session_id, &subagent_id);
        }
    }
}
