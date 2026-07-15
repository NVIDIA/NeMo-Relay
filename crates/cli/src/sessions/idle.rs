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
    let mut idle_sessions = Vec::new();
    {
        let mut sessions = inner.lock().await;
        let ids = sessions
            .iter()
            .filter_map(|(session_id, session)| {
                session
                    .is_idle_for(now, timeout)
                    .then_some(session_id.clone())
            })
            .collect::<Vec<_>>();
        for session_id in ids {
            if let Some(session) = sessions.remove(&session_id) {
                idle_sessions.push((session_id, session));
            }
        }
    }
    if idle_sessions.is_empty() {
        return Ok(0);
    }
    let mut closed_turns = 0;
    let mut closed_subagents = Vec::new();
    let mut retained_sessions = Vec::new();
    let mut first_error = None;
    for (session_id, mut session) in idle_sessions {
        let stack = session.scope_stack.clone();
        let result = TASK_SCOPE_STACK
            .scope(stack, async { session.close_turn_for_reason(reason).await })
            .await;
        match result {
            Ok(subagent_ids) => {
                closed_turns += 1;
                for subagent_id in subagent_ids {
                    closed_subagents.push((session_id.clone(), subagent_id));
                }
            }
            Err(error) if first_error.is_none() => first_error = Some(error),
            Err(_) => {}
        }
        if !session.is_empty() {
            retained_sessions.push((session_id, session));
        }
    }
    let mut alignment_cleanup_sessions = HashSet::new();
    {
        let mut sessions = inner.lock().await;
        for (session_id, session) in retained_sessions {
            if let Entry::Vacant(entry) = sessions.entry(session_id.clone()) {
                entry.insert(session);
                alignment_cleanup_sessions.insert(session_id);
            }
        }
        for (session_id, _) in &closed_subagents {
            if !sessions.contains_key(session_id) {
                alignment_cleanup_sessions.insert(session_id.clone());
            }
        }
    }
    if !closed_subagents.is_empty() {
        let mut alignment_state = alignment.lock().await;
        for (session_id, subagent_id) in closed_subagents {
            if alignment_cleanup_sessions.contains(&session_id) {
                alignment_state.clear_for_ended_subagent(&session_id, &subagent_id);
            }
        }
    }
    first_error.map_or(Ok(closed_turns), Err)
}
