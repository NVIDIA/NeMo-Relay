// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoint-local session policy state shared by OpenTelemetry signals.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use opentelemetry::trace::TraceId;
use regex::Regex;
use uuid::Uuid;

use crate::api::event::{Event, ScopeCategory};
use crate::json::Json;

use super::relay_trace_id;

/// Compiled policy and mutable state for one OpenTelemetry destination.
#[derive(Debug)]
pub(crate) struct EndpointSessionFilter {
    session_metadata_key: String,
    tool_name_patterns: Vec<Regex>,
    state: Mutex<EndpointSessionFilterState>,
}

#[derive(Debug, Default)]
struct EndpointSessionFilterState {
    blocked_sessions: HashSet<String>,
    blocked_trace_ids: HashSet<TraceId>,
    scope_sessions: HashMap<Uuid, String>,
    block_unattributed_events: bool,
}

impl EndpointSessionFilter {
    pub(crate) fn new(session_metadata_key: String, tool_name_patterns: Vec<Regex>) -> Self {
        Self {
            session_metadata_key,
            tool_name_patterns,
            state: Mutex::new(EndpointSessionFilterState::default()),
        }
    }

    /// Observe an event before any signal subscriber processes it.
    ///
    /// Observation updates endpoint policy state but never suppresses subscriber
    /// delivery. Trace and log subscribers must still consume scope lifecycle
    /// events so their active state remains balanced.
    pub(crate) fn observe(&self, event: &Event) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let session = self.resolve_session(&state, event);
        let trace_id = event_trace_id(event);
        let is_matching_tool = event.scope_category() == Some(ScopeCategory::Start)
            && event
                .category()
                .is_some_and(|category| category.as_str() == "tool")
            && self
                .tool_name_patterns
                .iter()
                .any(|pattern| pattern.is_match(event.name()));

        if is_matching_tool {
            state.block_unattributed_events = true;
            if let Some(session) = &session {
                state.blocked_sessions.insert(session.clone());
            }
            state.blocked_trace_ids.insert(trace_id);
        } else if session
            .as_ref()
            .is_some_and(|session| state.blocked_sessions.contains(session))
            || session.is_none() && state.block_unattributed_events
        {
            state.blocked_trace_ids.insert(trace_id);
        }

        match event.scope_category() {
            Some(ScopeCategory::Start) => {
                if let Some(session) = session {
                    state.scope_sessions.insert(event.uuid(), session);
                }
            }
            Some(ScopeCategory::End) => {
                state.scope_sessions.remove(&event.uuid());
            }
            None => {}
        }
    }

    /// Whether an observed Relay event must not emit a log record.
    pub(crate) fn blocks_event(&self, event: &Event) -> bool {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.blocked_trace_ids.contains(&event_trace_id(event)) {
            return true;
        }
        match self.resolve_session(&state, event) {
            Some(session) => state.blocked_sessions.contains(&session),
            None => state.block_unattributed_events,
        }
    }

    /// Whether completed telemetry for this Relay trace must not be exported.
    pub(crate) fn blocks_trace(&self, trace_id: TraceId) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .blocked_trace_ids
            .contains(&trace_id)
    }

    fn resolve_session(&self, state: &EndpointSessionFilterState, event: &Event) -> Option<String> {
        event
            .metadata()
            .and_then(Json::as_object)
            .and_then(|metadata| metadata.get(&self.session_metadata_key))
            .and_then(json_session_value)
            .or_else(|| {
                event
                    .parent_uuid()
                    .and_then(|parent| state.scope_sessions.get(&parent).cloned())
            })
            .or_else(|| state.scope_sessions.get(&event.uuid()).cloned())
    }
}

fn event_trace_id(event: &Event) -> TraceId {
    relay_trace_id(event.propagation_root_uuid().unwrap_or(event.uuid()))
}

fn json_session_value(value: &Json) -> Option<String> {
    value
        .as_str()
        .map(str::to_owned)
        .filter(|value| !value.is_empty())
}
