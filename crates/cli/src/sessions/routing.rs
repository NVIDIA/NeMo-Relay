// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Child-session aliasing and lifecycle-event routing.

use std::collections::HashMap;

use serde_json::Value;

use crate::agents::shared::alignment::{
    self, PendingSubagentStart, SessionAlias, SessionAlignmentState, merge_metadata,
};
use crate::configuration::SessionConfig;
use crate::error::CliError;
use crate::events::{AgentKind, NormalizedEvent, SessionEvent};

use super::{LlmGatewayStart, Session};

pub(super) fn apply_start_alias(start: &mut LlmGatewayStart, alias: &SessionAlias) {
    start.session_id = Some(alias.parent_session_id.clone());
    start.subagent_id = Some(alias.subagent_id.clone());
    start.metadata = merge_metadata(start.metadata.clone(), alias.metadata());
}

pub(super) async fn queue_or_promote_child_start(
    event: &mut NormalizedEvent,
    sessions: &mut HashMap<String, Session>,
    alignment_state: &mut SessionAlignmentState,
    config: SessionConfig,
) -> Result<bool, CliError> {
    let Some((child_session_id, pending)) = alignment::pending_subagent_start(event).await else {
        return Ok(false);
    };
    if sessions
        .get(&child_session_id)
        .is_some_and(|session| !session.can_reparent_as_subagent_alias())
    {
        return Ok(false);
    }
    if sessions.contains_key(pending.parent_session_id()) {
        alignment_state.remove_pending(&child_session_id);
        promote_pending_subagent(sessions, alignment_state, child_session_id, pending, config)
            .await?;
    } else {
        sessions.remove(&child_session_id);
        alignment_state.insert_pending(child_session_id, pending);
    }
    Ok(true)
}

pub(super) async fn apply_event_to_session(
    sessions: &mut HashMap<String, Session>,
    session_id: &str,
    event: NormalizedEvent,
    event_kind: AgentKind,
    config: SessionConfig,
    is_agent_started: bool,
) -> Result<bool, CliError> {
    let session = sessions
        .entry(session_id.to_string())
        .or_insert_with(|| Session::new(session_id.to_string(), event_kind, config));
    if is_agent_started
        && session.agent_kind == AgentKind::Gateway
        && event_kind != AgentKind::Gateway
    {
        session.agent_kind = event_kind;
    }
    session.apply(event).await?;
    Ok(session.is_empty())
}

pub(super) async fn promote_pending_subagents_for_parent(
    sessions: &mut HashMap<String, Session>,
    alignment_state: &mut SessionAlignmentState,
    parent_session_id: &str,
    config: SessionConfig,
) -> Result<(), CliError> {
    for (child_session_id, pending) in alignment_state.pending_for_parent(parent_session_id) {
        promote_pending_subagent(
            sessions,
            alignment_state,
            child_session_id,
            pending,
            config.clone(),
        )
        .await?;
    }
    Ok(())
}

pub(super) async fn promote_pending_subagent(
    sessions: &mut HashMap<String, Session>,
    alignment_state: &mut SessionAlignmentState,
    child_session_id: String,
    pending: PendingSubagentStart,
    config: SessionConfig,
) -> Result<Option<SessionAlias>, CliError> {
    if sessions
        .get(&child_session_id)
        .is_some_and(|session| !session.can_reparent_as_subagent_alias())
    {
        return Ok(None);
    }
    sessions.remove(&child_session_id);
    let parent_session_id = pending.parent_session_id().to_string();
    let parent_session = sessions
        .entry(parent_session_id.clone())
        .or_insert_with(|| {
            Session::new(parent_session_id.clone(), pending.event.agent_kind, config)
        });
    if !parent_session.session_started && parent_session.agent_scope.is_none() {
        parent_session
            .apply(NormalizedEvent::AgentStarted(SessionEvent {
                session_id: parent_session_id,
                agent_kind: pending.event.agent_kind,
                event_name: "implicit_parent_for_aligned_subagent".into(),
                payload: Value::Null,
                metadata: Value::Null,
            }))
            .await?;
    }
    parent_session
        .apply(NormalizedEvent::SubagentStarted(
            pending.subagent_start_event(),
        ))
        .await?;
    let alias = pending.alias_for_child_session(child_session_id.clone());
    alignment_state.insert_alias(child_session_id, alias.clone());
    Ok(Some(alias))
}

pub(super) fn route_event_for_session(
    event: NormalizedEvent,
    sessions: &mut HashMap<String, Session>,
    alignment_state: &mut SessionAlignmentState,
) -> Option<(NormalizedEvent, String, bool)> {
    let mut event = alignment_state.route_event(event);
    let explicit_subagent_alias = alignment::explicit_subagent_alias(&mut event);
    let session_id = event.session_id().to_string();
    let is_agent_started = matches!(&event, NormalizedEvent::AgentStarted(_));

    if event.is_terminal() && !sessions.contains_key(&session_id) {
        return None;
    }
    if !apply_explicit_subagent_alias(
        &mut event,
        sessions,
        alignment_state,
        explicit_subagent_alias,
    ) {
        return None;
    }
    Some((event, session_id, is_agent_started))
}

fn apply_explicit_subagent_alias(
    event: &mut NormalizedEvent,
    sessions: &mut HashMap<String, Session>,
    alignment_state: &mut SessionAlignmentState,
    explicit_subagent_alias: Option<(String, SessionAlias)>,
) -> bool {
    let Some((child_session_id, alias)) = explicit_subagent_alias else {
        alignment_state.align_explicit_subagent_end(event);
        return true;
    };
    if sessions
        .get(&child_session_id)
        .is_some_and(|session| !session.can_reparent_as_subagent_alias())
    {
        return false;
    }
    sessions.remove(&child_session_id);
    alignment_state.insert_alias(child_session_id, alias);
    alignment_state.align_explicit_subagent_end(event);
    true
}
