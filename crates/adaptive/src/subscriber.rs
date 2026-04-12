// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event subscriber factory and event-to-record mapping helpers.
//!
//! This module provides [`create_subscriber`], which produces an
//! [`EventSubscriberFn`] callback that clones incoming NeMo Flow events into a
//! `tokio::sync::mpsc` channel for async processing by the drain task.
//!
//! It also provides helper functions for mapping raw [`Event`] values to
//! adaptive-specific types:
//!
//! - [`event_to_call_record`] — converts LLM/Tool start events into
//!   [`CallRecord`] entries.
//! - [`is_run_boundary`] — detects Agent scope start/end events that
//!   delineate run boundaries.

use nemo_flow::context::callbacks::EventSubscriberFn;
use nemo_flow::types::event::Event;
use nemo_flow::types::scope::ScopeType;

use crate::types::records::{CallKind, CallRecord};

/// Creates an [`EventSubscriberFn`] that forwards cloned events through the
/// given unbounded channel sender.
///
/// # Hot-path safety
///
/// The returned closure runs **synchronously** on the request path after NeMo Flow
/// releases its runtime locks. It MUST NOT:
///
/// - Perform I/O
/// - Acquire write locks on the global context
/// - Call NeMo Flow API functions
/// - Panic
///
/// The only work done is `event.clone()` followed by
/// [`UnboundedSender::send`](tokio::sync::mpsc::UnboundedSender::send), which
/// never blocks. If the receiver has been dropped (adaptive is shutting down),
/// `send` returns `Err` and the event is silently discarded via `let _ = ...`.
pub(crate) fn create_subscriber(
    tx: tokio::sync::mpsc::UnboundedSender<Event>,
) -> EventSubscriberFn {
    std::sync::Arc::new(move |event: &Event| {
        // CRITICAL: This runs synchronously on the call path, so it must stay non-blocking.
        // MUST NOT: do I/O, acquire write locks, call NeMo Flow APIs, or panic.
        // ONLY: clone + send. UnboundedSender::send() never blocks.
        let _ = tx.send(event.clone());
    })
}

/// Maps a NeMo Flow Start event to a partial [`CallRecord`] (with `ended_at = None`).
///
/// Returns `None` for:
/// - Events that are not start variants
/// - Events whose [`ScopeType`] is not [`ScopeType::Llm`] or [`ScopeType::Tool`]
///
/// Agent scope events are intentionally excluded — they represent run
/// boundaries, not individual call records. Use [`is_run_boundary`] instead.
pub(crate) fn event_to_call_record(event: &Event) -> Option<CallRecord> {
    let kind = match event {
        Event::LLMStart(_) => CallKind::Llm,
        Event::ToolStart(_) => CallKind::Tool,
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
    })
}

/// Returns `true` if this event represents a root scope lifecycle boundary.
///
/// A run starts with an agent scope start event.
/// A run ends with an agent scope end event.
///
/// Non-agent events (Tool, LLM, Function, etc.) are never run boundaries.
pub(crate) fn is_run_boundary(event: &Event) -> bool {
    matches!(event, Event::ScopeStart(inner) if inner.scope_type == ScopeType::Agent)
        || matches!(event, Event::ScopeEnd(inner) if inner.scope_type == ScopeType::Agent)
}

#[cfg(test)]
#[path = "../tests/coverage/subscriber_tests.rs"]
mod tests;
