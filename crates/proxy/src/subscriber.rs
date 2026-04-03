// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event subscriber factory and event-to-record mapping helpers.
//!
//! This module provides [`create_subscriber`], which produces an
//! [`EventSubscriberFn`] callback that clones incoming Nexus events into a
//! `tokio::sync::mpsc` channel for async processing by the drain task.
//!
//! It also provides helper functions for mapping raw [`Event`] values to
//! proxy-specific types:
//!
//! - [`event_to_call_record`] — converts LLM/Tool start events into
//!   [`CallRecord`] entries.
//! - [`is_run_boundary`] — detects Agent scope start/end events that
//!   delineate run boundaries.

use nvidia_nat_nexus_core::{Event, EventSubscriberFn, EventType, ScopeType};

use crate::types::{CallKind, CallRecord};

/// Creates an [`EventSubscriberFn`] that forwards cloned events through the
/// given unbounded channel sender.
///
/// # Hot-path safety
///
/// The returned closure runs **synchronously** under the global context read
/// lock (see `NatNexusContextState::emit_event`). It MUST NOT:
///
/// - Perform I/O
/// - Acquire write locks on the global context
/// - Call Nexus API functions
/// - Panic
///
/// The only work done is `event.clone()` followed by
/// [`UnboundedSender::send`](tokio::sync::mpsc::UnboundedSender::send), which
/// never blocks. If the receiver has been dropped (proxy is shutting down),
/// `send` returns `Err` and the event is silently discarded via `let _ = ...`.
pub(crate) fn create_subscriber(
    tx: tokio::sync::mpsc::UnboundedSender<Event>,
) -> EventSubscriberFn {
    Box::new(move |event: &Event| {
        // CRITICAL: This runs under the global context read lock.
        // MUST NOT: do I/O, acquire write locks, call Nexus APIs, or panic.
        // ONLY: clone + send. UnboundedSender::send() never blocks.
        let _ = tx.send(event.clone());
    })
}

/// Maps a Nexus Start event to a partial [`CallRecord`] (with `ended_at = None`).
///
/// Returns `None` for:
/// - Events that are not [`EventType::Start`]
/// - Events whose [`ScopeType`] is not [`ScopeType::Llm`] or [`ScopeType::Tool`]
///
/// Agent scope events are intentionally excluded — they represent run
/// boundaries, not individual call records. Use [`is_run_boundary`] instead.
pub(crate) fn event_to_call_record(event: &Event) -> Option<CallRecord> {
    if event.event_type != EventType::Start {
        return None;
    }
    let kind = match event.scope_type {
        Some(ScopeType::Llm) => CallKind::Llm,
        Some(ScopeType::Tool) => CallKind::Tool,
        _ => return None,
    };
    Some(CallRecord {
        kind,
        name: event.name.clone().unwrap_or_default(),
        started_at: event.timestamp,
        ended_at: None,
        metadata_snapshot: None,
        output_tokens: None,
    })
}

/// Returns `true` if this event represents a root scope lifecycle boundary.
///
/// A run starts with [`EventType::Start`] + [`ScopeType::Agent`] at the root.
/// A run ends with [`EventType::End`] + [`ScopeType::Agent`] at the root.
///
/// Non-agent events (Tool, LLM, Function, etc.) are never run boundaries.
pub(crate) fn is_run_boundary(event: &Event) -> bool {
    matches!(event.scope_type, Some(ScopeType::Agent))
        && matches!(event.event_type, EventType::Start | EventType::End)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nvidia_nat_nexus_core::{Event, EventType, ScopeType};
    use uuid::Uuid;

    /// Helper to construct a minimal test [`Event`] with only the fields
    /// relevant to subscriber/mapping logic populated.
    fn make_test_event(
        event_type: EventType,
        scope_type: Option<ScopeType>,
        name: Option<&str>,
    ) -> Event {
        Event {
            parent_uuid: None,
            uuid: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            name: name.map(|s| s.to_string()),
            data: None,
            metadata: None,
            attributes: None,
            event_type,
            scope_type,
            input: None,
            output: None,
            model_name: None,
            tool_call_id: None,
            root_uuid: None,
        }
    }

    // -----------------------------------------------------------------------
    // create_subscriber tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_subscriber_sends_event() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let subscriber = create_subscriber(tx);

        let event = make_test_event(EventType::Start, Some(ScopeType::Llm), Some("gpt-4"));
        subscriber(&event);

        let received = rx.try_recv().expect("should receive event");
        assert_eq!(received.uuid, event.uuid);
        assert_eq!(received.name, Some("gpt-4".to_string()));
    }

    #[test]
    fn test_subscriber_survives_dropped_receiver() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let subscriber = create_subscriber(tx);

        // Drop the receiver — subscriber must not panic
        drop(rx);

        let event = make_test_event(EventType::Start, Some(ScopeType::Tool), Some("search"));
        subscriber(&event); // Must not panic
    }

    // -----------------------------------------------------------------------
    // event_to_call_record tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_event_to_call_record_llm_start() {
        let event = make_test_event(EventType::Start, Some(ScopeType::Llm), Some("gpt-4"));
        let record = event_to_call_record(&event).expect("should produce CallRecord for LLM start");

        assert_eq!(record.kind, CallKind::Llm);
        assert_eq!(record.name, "gpt-4");
        assert!(record.ended_at.is_none());
        assert!(record.metadata_snapshot.is_none());
    }

    #[test]
    fn test_event_to_call_record_tool_start() {
        let event = make_test_event(EventType::Start, Some(ScopeType::Tool), Some("search"));
        let record =
            event_to_call_record(&event).expect("should produce CallRecord for Tool start");

        assert_eq!(record.kind, CallKind::Tool);
        assert_eq!(record.name, "search");
        assert!(record.ended_at.is_none());
    }

    #[test]
    fn test_event_to_call_record_end_event_returns_none() {
        let event = make_test_event(EventType::End, Some(ScopeType::Llm), Some("gpt-4"));
        assert!(
            event_to_call_record(&event).is_none(),
            "End events should not produce CallRecords"
        );
    }

    #[test]
    fn test_event_to_call_record_agent_scope_returns_none() {
        let event = make_test_event(EventType::Start, Some(ScopeType::Agent), Some("my-agent"));
        assert!(
            event_to_call_record(&event).is_none(),
            "Agent scope events are run boundaries, not call records"
        );
    }

    #[test]
    fn test_event_to_call_record_no_name_defaults_to_empty() {
        let event = make_test_event(EventType::Start, Some(ScopeType::Tool), None);
        let record = event_to_call_record(&event).expect("should produce CallRecord");
        assert_eq!(record.name, "");
    }

    // -----------------------------------------------------------------------
    // is_run_boundary tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_run_boundary_agent_start() {
        let event = make_test_event(EventType::Start, Some(ScopeType::Agent), Some("agent-1"));
        assert!(
            is_run_boundary(&event),
            "Agent Start should be a run boundary"
        );
    }

    #[test]
    fn test_is_run_boundary_agent_end() {
        let event = make_test_event(EventType::End, Some(ScopeType::Agent), Some("agent-1"));
        assert!(
            is_run_boundary(&event),
            "Agent End should be a run boundary"
        );
    }

    #[test]
    fn test_is_run_boundary_tool_start() {
        let event = make_test_event(EventType::Start, Some(ScopeType::Tool), Some("search"));
        assert!(
            !is_run_boundary(&event),
            "Tool Start should NOT be a run boundary"
        );
    }

    #[test]
    fn test_is_run_boundary_agent_mark() {
        let event = make_test_event(EventType::Mark, Some(ScopeType::Agent), Some("agent-1"));
        assert!(
            !is_run_boundary(&event),
            "Agent Mark should NOT be a run boundary"
        );
    }
}
