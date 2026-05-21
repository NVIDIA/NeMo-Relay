// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for subscriber in the NeMo Relay adaptive crate.

use super::*;
use nemo_relay::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
};
use nemo_relay::api::scope::ScopeType;
use nemo_relay::codec::response::{AnnotatedLlmResponse, FinishReason};
use std::sync::Arc;

#[derive(Clone, Copy)]
enum EventType {
    Start,
    End,
    Mark,
}

/// Helper to construct a minimal test [`Event`] with only the fields
/// relevant to subscriber/mapping logic populated.
fn make_test_event(
    event_type: EventType,
    scope_type: Option<ScopeType>,
    name: Option<&str>,
) -> Event {
    let event_name = name.unwrap_or("");
    match (event_type, scope_type) {
        (EventType::Start, Some(scope_type)) => Event::Scope(ScopeEvent::new(
            BaseEvent::builder().name(event_name).build(),
            ScopeCategory::Start,
            Vec::new(),
            EventCategory::from(scope_type),
            None,
        )),
        (EventType::End, Some(scope_type)) => Event::Scope(ScopeEvent::new(
            BaseEvent::builder().name(event_name).build(),
            ScopeCategory::End,
            Vec::new(),
            EventCategory::from(scope_type),
            None,
        )),
        (EventType::Mark, _) | (_, None) => Event::Mark(MarkEvent::new(
            BaseEvent::builder().name(event_name).build(),
            None,
            None,
        )),
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
    assert_eq!(received.uuid(), event.uuid());
    assert_eq!(received.name(), "gpt-4");
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
    let record = event_to_call_record(&event).expect("should produce CallRecord for Tool start");

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
fn test_event_to_call_record_llm_end_with_annotated_response_stays_observability_only() {
    let event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("gpt-4")
            .data(serde_json::json!({"response": "ok"}))
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::llm(),
        Some(
            CategoryProfile::builder()
                .model_name("gpt-4")
                .annotated_response(Arc::new(AnnotatedLlmResponse {
                    id: Some("resp-1".to_string()),
                    model: Some("gpt-4".to_string()),
                    message: None,
                    tool_calls: None,
                    finish_reason: Some(FinishReason::Complete),
                    usage: None,
                    api_specific: None,
                    extra: serde_json::Map::new(),
                }))
                .build(),
        ),
    ));

    assert!(
        event_to_call_record(&event).is_none(),
        "annotated_response belongs to LLM end observability, not request/start call records",
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
