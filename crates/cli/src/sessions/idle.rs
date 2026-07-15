// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Idle-session sweeping and shutdown closure.

use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nemo_relay::api::runtime::TASK_SCOPE_STACK;
use tokio::sync::Mutex;

use crate::agents::shared::alignment::SessionAlignmentState;
use crate::error::CliError;

use super::Session;

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
    alignment: &Arc<Mutex<SessionAlignmentState>>,
    now: Instant,
    timeout: Duration,
    reason: &str,
) -> Result<usize, CliError> {
    let idle_sessions = take_idle_sessions(inner, now, timeout).await;
    if idle_sessions.is_empty() {
        return Ok(0);
    }
    let (closed_turns, closed_subagents, retained_sessions, first_error) =
        close_idle_turns(idle_sessions, reason).await;
    let cleanup_sessions =
        restore_retained_sessions(inner, retained_sessions, &closed_subagents).await;
    clear_closed_subagents(alignment, closed_subagents, &cleanup_sessions).await;
    first_error.map_or(Ok(closed_turns), Err)
}

async fn take_idle_sessions(
    inner: &Arc<Mutex<HashMap<String, Session>>>,
    now: Instant,
    timeout: Duration,
) -> Vec<(String, Session)> {
    let mut sessions = inner.lock().await;
    let ids = sessions
        .iter()
        .filter_map(|(session_id, session)| {
            session
                .is_idle_for(now, timeout)
                .then_some(session_id.clone())
        })
        .collect::<Vec<_>>();
    ids.into_iter()
        .filter_map(|session_id| {
            sessions
                .remove(&session_id)
                .map(|session| (session_id, session))
        })
        .collect()
}

type ClosedIdleTurns = (
    usize,
    Vec<(String, String)>,
    Vec<(String, Session)>,
    Option<CliError>,
);

async fn close_idle_turns(idle_sessions: Vec<(String, Session)>, reason: &str) -> ClosedIdleTurns {
    let mut closed_turns = 0;
    let mut closed_subagents = Vec::new();
    let mut retained_sessions = Vec::new();
    let mut first_error = None;
    for (session_id, mut session) in idle_sessions {
        let stack = session.scope_stack.clone();
        match TASK_SCOPE_STACK
            .scope(stack, async { session.close_turn_for_reason(reason).await })
            .await
        {
            Ok(subagent_ids) => {
                closed_turns += 1;
                closed_subagents.extend(
                    subagent_ids
                        .into_iter()
                        .map(|subagent_id| (session_id.clone(), subagent_id)),
                );
            }
            Err(error) if first_error.is_none() => first_error = Some(error),
            Err(_) => {}
        }
        if !session.is_empty() {
            retained_sessions.push((session_id, session));
        }
    }
    (
        closed_turns,
        closed_subagents,
        retained_sessions,
        first_error,
    )
}

async fn restore_retained_sessions(
    inner: &Arc<Mutex<HashMap<String, Session>>>,
    retained_sessions: Vec<(String, Session)>,
    closed_subagents: &[(String, String)],
) -> HashSet<String> {
    let mut cleanup_sessions = HashSet::new();
    let mut sessions = inner.lock().await;
    for (session_id, session) in retained_sessions {
        if let Entry::Vacant(entry) = sessions.entry(session_id.clone()) {
            entry.insert(session);
            cleanup_sessions.insert(session_id);
        }
    }
    for (session_id, _) in closed_subagents {
        if !sessions.contains_key(session_id) {
            cleanup_sessions.insert(session_id.clone());
        }
    }
    cleanup_sessions
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
