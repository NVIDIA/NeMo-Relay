// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Child-session aliasing and lifecycle-event routing.

use std::collections::HashMap;

use nemo_relay::api::runtime::SubscriberDelivery;
use serde_json::Value;

use crate::agents::shared::alignment::{
    self, PendingSubagentStart, SessionAlias, SessionAlignmentState, merge_metadata,
};
use crate::configuration::SessionConfig;
use crate::error::CliError;
use crate::events::{AgentKind, NormalizedEvent, SessionEvent};

use super::{LlmGatewayStart, Session, ToolArgumentTransform};

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
    authenticated_owners: Option<&HashMap<String, String>>,
    authenticated_owner: Option<&str>,
) -> Result<bool, CliError> {
    let Some((child_session_id, mut pending)) = alignment::pending_subagent_start(event).await
    else {
        return Ok(false);
    };
    pending.set_authenticated_owner(authenticated_owner.map(ToOwned::to_owned));
    if sessions
        .get(&child_session_id)
        .is_some_and(|session| !session.can_reparent_as_subagent_alias())
    {
        return Ok(false);
    }
    if sessions.contains_key(pending.parent_session_id()) {
        if !parent_owner_matches(
            authenticated_owners,
            pending.parent_session_id(),
            pending.authenticated_owner(),
        ) {
            return Err(CliError::Unauthorized(format!(
                "Relay hook client does not own session '{}'",
                pending.parent_session_id()
            )));
        }
        alignment_state.remove_pending(&child_session_id);
        promote_pending_subagent(
            sessions,
            alignment_state,
            child_session_id,
            pending,
            config,
            authenticated_owners,
        )
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
) -> Result<
    (
        bool,
        Option<SubscriberDelivery>,
        Option<ToolArgumentTransform>,
    ),
    CliError,
> {
    let session = sessions
        .entry(session_id.to_string())
        .or_insert_with(|| Session::new(session_id.to_string(), event_kind, config));
    if is_agent_started
        && session.agent_kind == AgentKind::Gateway
        && event_kind != AgentKind::Gateway
    {
        session.agent_kind = event_kind;
    }
    let subscriber_delivery = session.apply(event).await?;
    let tool_argument_transform = session.take_tool_argument_transform();
    Ok((
        session.is_empty(),
        subscriber_delivery,
        tool_argument_transform,
    ))
}

pub(super) async fn promote_pending_subagents_for_parent(
    sessions: &mut HashMap<String, Session>,
    alignment_state: &mut SessionAlignmentState,
    parent_session_id: &str,
    config: SessionConfig,
    authenticated_owners: Option<&HashMap<String, String>>,
) -> Result<(), CliError> {
    for (child_session_id, pending) in alignment_state.pending_for_parent(parent_session_id) {
        if !parent_owner_matches(
            authenticated_owners,
            parent_session_id,
            pending.authenticated_owner(),
        ) {
            continue;
        }
        promote_pending_subagent(
            sessions,
            alignment_state,
            child_session_id,
            pending,
            config.clone(),
            authenticated_owners,
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
    authenticated_owners: Option<&HashMap<String, String>>,
) -> Result<Option<SessionAlias>, CliError> {
    if sessions
        .get(&child_session_id)
        .is_some_and(|session| !session.can_reparent_as_subagent_alias())
    {
        return Ok(None);
    }
    sessions.remove(&child_session_id);
    let parent_session_id = pending.parent_session_id().to_string();
    if !parent_owner_matches(
        authenticated_owners,
        &parent_session_id,
        pending.authenticated_owner(),
    ) {
        return Ok(None);
    }
    let parent_session = sessions
        .entry(parent_session_id.clone())
        .or_insert_with(|| {
            Session::new(parent_session_id.clone(), pending.event.agent_kind, config)
        });
    if !parent_session.session_started && parent_session.agent_scope.is_none() {
        let _ = parent_session
            .apply(NormalizedEvent::AgentStarted(SessionEvent {
                session_id: parent_session_id,
                agent_kind: pending.event.agent_kind,
                event_name: "implicit_parent_for_aligned_subagent".into(),
                payload: Value::Null,
                metadata: Value::Null,
            }))
            .await?;
    }
    let _ = parent_session
        .apply(NormalizedEvent::SubagentStarted(
            pending.subagent_start_event(),
        ))
        .await?;
    let mut alias = pending.alias_for_child_session(child_session_id.clone());
    alias.set_authenticated_owner(pending.authenticated_owner().map(ToOwned::to_owned));
    alignment_state.insert_alias(child_session_id, alias.clone());
    Ok(Some(alias))
}

fn parent_owner_matches(
    authenticated_owners: Option<&HashMap<String, String>>,
    parent_session_id: &str,
    pending_owner: Option<&str>,
) -> bool {
    match (authenticated_owners, pending_owner) {
        (Some(owners), Some(owner)) => owners
            .get(parent_session_id)
            .is_some_and(|existing| existing == owner),
        _ => true,
    }
}

pub(super) fn route_event_for_session(
    event: NormalizedEvent,
    sessions: &mut HashMap<String, Session>,
    alignment_state: &mut SessionAlignmentState,
) -> Option<(NormalizedEvent, String, bool)> {
    let event = alignment_state.route_event(event);
    let session_id = event.session_id().to_string();
    let is_agent_started = matches!(&event, NormalizedEvent::AgentStarted(_));

    if event.is_terminal() && !sessions.contains_key(&session_id) {
        return None;
    }
    Some((event, session_id, is_agent_started))
}
