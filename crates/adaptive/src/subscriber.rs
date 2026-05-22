// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event subscriber factory and event-to-record mapping helpers.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use nemo_relay::api::event::{Event, ScopeCategory};
use nemo_relay::api::runtime::EventSubscriberFn;
use nemo_relay::api::scope::ScopeType;

use crate::types::records::{CallKind, CallRecord};

#[cfg(test)]
pub(crate) fn create_subscriber(
    tx: tokio::sync::mpsc::UnboundedSender<Event>,
) -> EventSubscriberFn {
    create_subscriber_with_counter(tx, Arc::new(AtomicUsize::new(0)))
}

pub(crate) fn create_subscriber_with_counter(
    tx: tokio::sync::mpsc::UnboundedSender<Event>,
    pending_events: Arc<AtomicUsize>,
) -> EventSubscriberFn {
    std::sync::Arc::new(move |event: &Event| {
        pending_events.fetch_add(1, Ordering::SeqCst);
        if tx.send(event.clone()).is_err() {
            pending_events.fetch_sub(1, Ordering::SeqCst);
        }
    })
}

pub(crate) fn event_to_call_record(event: &Event) -> Option<CallRecord> {
    if event.scope_category() != Some(ScopeCategory::Start) {
        return None;
    }
    let (kind, annotated_request) = match event.category().map(|category| category.as_str()) {
        Some("llm") => (CallKind::Llm, event.annotated_request().cloned()),
        Some("tool") => (CallKind::Tool, None),
        _ => return None,
    };
    Some(CallRecord {
        kind,
        name: event.name().to_string(),
        started_at: *event.timestamp(),
        ended_at: None,
        metadata_snapshot: None,
        output_tokens: None,
        prompt_tokens: None,
        total_tokens: None,
        model_name: None,
        tool_call_count: None,
        annotated_request,
        annotated_response: None,
    })
}

pub(crate) fn is_run_boundary(event: &Event) -> bool {
    event.scope_type() == Some(ScopeType::Agent)
        && matches!(
            event.scope_category(),
            Some(ScopeCategory::Start | ScopeCategory::End)
        )
}

#[cfg(test)]
#[path = "../tests/coverage/subscriber_tests.rs"]
mod tests;
