// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Child-session aliasing and lifecycle-event routing.

use std::collections::HashMap;
use std::sync::Arc;

use nemo_relay::api::runtime::SubscriberDelivery;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::agents::shared::alignment::{
    PendingSubagentStart, SessionAlias, SessionAlignmentState, merge_metadata,
};
use crate::configuration::SessionConfig;
use crate::error::CliError;
use crate::events::{AgentKind, NormalizedEvent, SessionEvent};

use super::{
    AuthenticatedOwners, AuthenticatedReservations, LlmGatewayStart, Session, SessionActivity,
    SessionGates, ToolArgumentTransform, session_gate,
};

#[derive(Clone, Copy)]
pub(super) struct AuthenticatedRouting<'a> {
    pub(super) owner: Option<&'a str>,
    owners: Option<&'a AuthenticatedOwners>,
    reservations: Option<&'a AuthenticatedReservations>,
}

impl<'a> AuthenticatedRouting<'a> {
    pub(super) fn new(
        owner: Option<&'a str>,
        owners: &'a AuthenticatedOwners,
        reservations: &'a AuthenticatedReservations,
    ) -> Self {
        Self {
            owner,
            owners: owner.map(|_| owners),
            reservations: owner.map(|_| reservations),
        }
    }
}

pub(super) struct SessionEventApplier<'a> {
    sessions: &'a Arc<Mutex<HashMap<String, Session>>>,
    gates: &'a SessionGates,
    activity: &'a SessionActivity,
    config: SessionConfig,
}

impl<'a> SessionEventApplier<'a> {
    pub(super) fn new(
        sessions: &'a Arc<Mutex<HashMap<String, Session>>>,
        gates: &'a SessionGates,
        activity: &'a SessionActivity,
        config: SessionConfig,
    ) -> Self {
        Self {
            sessions,
            gates,
            activity,
            config,
        }
    }

    pub(super) async fn apply(
        &self,
        session_id: &str,
        event: NormalizedEvent,
        event_kind: AgentKind,
        is_agent_started: bool,
    ) -> Result<
        Option<(
            bool,
            Option<SubscriberDelivery>,
            Option<ToolArgumentTransform>,
        )>,
        CliError,
    > {
        let _activity = self.activity.begin();
        let gate = session_gate(self.gates, session_id).await;
        let _gate = gate.lock().await;
        if self.activity.is_closing() {
            return Ok(None);
        }
        let session = {
            let mut sessions = self.sessions.lock().await;
            sessions.remove(session_id)
        };
        if session.is_none() && event.is_terminal() {
            return Ok(None);
        }
        let mut session = session.unwrap_or_else(|| {
            Session::new(session_id.to_string(), event_kind, self.config.clone())
        });
        if is_agent_started
            && session.agent_kind == AgentKind::Gateway
            && event_kind != AgentKind::Gateway
        {
            session.agent_kind = event_kind;
        }
        match session.apply(event).await {
            Ok(subscriber_delivery) => {
                let is_empty = session.is_empty();
                let tool_argument_transform = session.take_tool_argument_transform();
                if !is_empty {
                    self.sessions
                        .lock()
                        .await
                        .insert(session_id.to_string(), session);
                }
                Ok(Some((
                    is_empty,
                    subscriber_delivery,
                    tool_argument_transform,
                )))
            }
            Err(error) => {
                self.sessions
                    .lock()
                    .await
                    .insert(session_id.to_string(), session);
                Err(error)
            }
        }
    }
}

pub(super) fn apply_start_alias(start: &mut LlmGatewayStart, alias: &SessionAlias) {
    start.session_id = Some(alias.parent_session_id.clone());
    start.subagent_id = Some(alias.subagent_id.clone());
    start.metadata = merge_metadata(start.metadata.clone(), alias.metadata());
}

pub(super) async fn queue_or_promote_child_start(
    pending_child: Option<(String, PendingSubagentStart)>,
    sessions: &mut HashMap<String, Session>,
    alignment_state: &mut SessionAlignmentState,
    config: SessionConfig,
    authenticated: AuthenticatedRouting<'_>,
) -> Result<bool, CliError> {
    let Some((child_session_id, mut pending)) = pending_child else {
        return Ok(false);
    };
    pending.set_authenticated_owner(authenticated.owner.map(ToOwned::to_owned));
    if sessions
        .get(&child_session_id)
        .is_some_and(|session| !session.can_reparent_as_subagent_alias())
    {
        return Ok(false);
    }
    if sessions.contains_key(pending.parent_session_id()) {
        if !parent_owner_matches(
            authenticated,
            pending.parent_session_id(),
            pending.authenticated_owner(),
        )
        .await
        {
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
            authenticated,
        )
        .await?;
    } else {
        sessions.remove(&child_session_id);
        alignment_state.insert_pending(child_session_id, pending);
    }
    Ok(true)
}

pub(super) async fn promote_pending_subagents_for_parent(
    sessions: &mut HashMap<String, Session>,
    alignment_state: &mut SessionAlignmentState,
    parent_session_id: &str,
    config: SessionConfig,
    authenticated: AuthenticatedRouting<'_>,
) -> Result<(), CliError> {
    for (child_session_id, pending) in alignment_state.pending_for_parent(parent_session_id) {
        if !parent_owner_matches(
            authenticated,
            parent_session_id,
            pending.authenticated_owner(),
        )
        .await
        {
            continue;
        }
        promote_pending_subagent(
            sessions,
            alignment_state,
            child_session_id,
            pending,
            config.clone(),
            authenticated,
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
    authenticated: AuthenticatedRouting<'_>,
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
        authenticated,
        &parent_session_id,
        pending.authenticated_owner(),
    )
    .await
    {
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

async fn parent_owner_matches(
    authenticated: AuthenticatedRouting<'_>,
    parent_session_id: &str,
    pending_owner: Option<&str>,
) -> bool {
    match (
        authenticated.owners,
        authenticated.reservations,
        pending_owner,
    ) {
        (Some(owners), Some(reservations), Some(owner)) => {
            super::owner_matches(owners, reservations, parent_session_id, owner).await
        }
        _ => true,
    }
}

pub(super) fn route_event_for_session(
    event: NormalizedEvent,
    alignment_state: &mut SessionAlignmentState,
) -> Option<(NormalizedEvent, String, bool)> {
    let event = alignment_state.route_event(event);
    let session_id = event.session_id().to_string();
    let is_agent_started = matches!(&event, NormalizedEvent::AgentStarted(_));

    Some((event, session_id, is_agent_started))
}
