// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for native subscriber dispatch behavior.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use nemo_relay::api::event::Event;
use nemo_relay::api::registry::{
    deregister_mark_sanitize_guardrail, deregister_scope_sanitize_end_guardrail,
    deregister_tool_sanitize_request_guardrail, register_mark_sanitize_guardrail,
    register_scope_sanitize_end_guardrail, register_tool_sanitize_request_guardrail,
};
use nemo_relay::api::runtime::{
    NemoRelayContextState, create_scope_stack, current_scope_stack, global_context,
    set_thread_scope_stack,
};
use nemo_relay::api::scope::{
    EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeType, event, pop_scope, push_scope,
};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::api::tool::{ToolCallEndParams, ToolCallParams, tool_call, tool_call_end};
use nemo_relay::error::FlowError;
use serde_json::json;

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_global() {
    let _ = spdlog::init_log_crate_proxy();
    log::set_max_level(log::LevelFilter::Info);
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoRelayContextState::new();
}

fn setup_isolated_thread() {
    let stack = create_scope_stack();
    set_thread_scope_stack(stack);
}

fn emit_mark(name: &str) {
    event(EmitMarkEventParams::builder().name(name).build()).unwrap();
}

#[test]
fn dispatch_event_returns_while_subscriber_is_blocked() {
    let _lock = TEST_MUTEX.lock().unwrap();
    flush_subscribers().unwrap();
    reset_global();
    setup_isolated_thread();

    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (returned_tx, returned_rx) = mpsc::channel();
    let release_rx = Arc::new(Mutex::new(release_rx));
    register_subscriber(
        "blocking-subscriber",
        Arc::new(move |_event| {
            let _ = started_tx.send(());
            let _ = release_rx.lock().unwrap().recv();
        }),
    )
    .unwrap();

    let event_thread = std::thread::spawn(move || {
        emit_mark("nonblocking");
        returned_tx.send(()).unwrap();
    });

    started_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("subscriber should start on dispatcher thread");
    let returned = returned_rx.recv_timeout(Duration::from_secs(1));
    release_tx.send(()).unwrap();
    event_thread.join().unwrap();
    flush_subscribers().unwrap();
    deregister_subscriber("blocking-subscriber").unwrap();

    returned.expect("event emission should return while subscriber callback waits");
}

#[test]
fn dispatcher_preserves_event_order() {
    let _lock = TEST_MUTEX.lock().unwrap();
    flush_subscribers().unwrap();
    reset_global();
    setup_isolated_thread();

    let observed = Arc::new(Mutex::new(Vec::new()));
    let observed_events = Arc::clone(&observed);
    register_subscriber(
        "ordered-subscriber",
        Arc::new(move |event| {
            observed_events
                .lock()
                .unwrap()
                .push(event.name().to_string());
        }),
    )
    .unwrap();

    emit_mark("one");
    emit_mark("two");
    flush_subscribers().unwrap();
    deregister_subscriber("ordered-subscriber").unwrap();

    assert_eq!(observed.lock().unwrap().as_slice(), ["one", "two"]);
}

#[test]
fn nested_event_sanitizer_publication_precedes_already_queued_events() {
    let _lock = TEST_MUTEX.lock().unwrap();
    flush_subscribers().unwrap();
    reset_global();
    setup_isolated_thread();

    let observed = Arc::new(Mutex::new(Vec::new()));
    let observed_events = Arc::clone(&observed);
    register_subscriber(
        "nested-event-order-subscriber",
        Arc::new(move |event| {
            observed_events
                .lock()
                .unwrap()
                .push(event.name().to_string());
        }),
    )
    .unwrap();

    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Arc::new(Mutex::new(release_rx));
    register_mark_sanitize_guardrail(
        "nested-event-order-sanitizer",
        0,
        Arc::new(move |event, fields| {
            let release_rx = Arc::clone(&release_rx);
            let started_tx = started_tx.clone();
            Box::pin(async move {
                if event.name() == "outer-event" {
                    started_tx.send(()).unwrap();
                    release_rx.lock().unwrap().recv().unwrap();
                    tokio::spawn(async {
                        emit_mark("nested-event");
                    })
                    .await
                    .unwrap();
                }
                Ok(fields)
            })
        }),
    )
    .unwrap();

    emit_mark("outer-event");
    started_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("outer sanitizer should start");
    emit_mark("later-event");
    release_tx.send(()).unwrap();
    flush_subscribers().unwrap();

    assert_eq!(
        observed.lock().unwrap().as_slice(),
        ["outer-event", "nested-event", "later-event"]
    );
    deregister_mark_sanitize_guardrail("nested-event-order-sanitizer").unwrap();
    deregister_subscriber("nested-event-order-subscriber").unwrap();
}

#[test]
fn nested_request_sanitizer_publication_precedes_manual_end_event() {
    let _lock = TEST_MUTEX.lock().unwrap();
    flush_subscribers().unwrap();
    reset_global();
    setup_isolated_thread();

    let observed = Arc::new(Mutex::new(Vec::new()));
    let observed_events = Arc::clone(&observed);
    register_subscriber(
        "nested-transform-order-subscriber",
        Arc::new(move |event| {
            observed_events
                .lock()
                .unwrap()
                .push(event.name().to_string());
        }),
    )
    .unwrap();

    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Arc::new(Mutex::new(release_rx));
    register_tool_sanitize_request_guardrail(
        "nested-transform-order-sanitizer",
        0,
        Arc::new(move |name, args| {
            let release_rx = Arc::clone(&release_rx);
            let started_tx = started_tx.clone();
            Box::pin(async move {
                if name == "manual-ordered-tool" {
                    started_tx.send(()).unwrap();
                    release_rx.lock().unwrap().recv().unwrap();
                    emit_mark("nested-transform-event");
                }
                Ok(args)
            })
        }),
    )
    .unwrap();

    let handle = tool_call(
        ToolCallParams::builder()
            .name("manual-ordered-tool")
            .args(json!({"input": true}))
            .build(),
    )
    .unwrap();
    started_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("request sanitizer should start");
    tool_call_end(
        ToolCallEndParams::builder()
            .handle(&handle)
            .execution_result(json!({"output": true}).into())
            .build(),
    )
    .unwrap();
    release_tx.send(()).unwrap();
    flush_subscribers().unwrap();

    assert_eq!(
        observed.lock().unwrap().as_slice(),
        [
            "manual-ordered-tool",
            "nested-transform-event",
            "manual-ordered-tool"
        ]
    );
    deregister_tool_sanitize_request_guardrail("nested-transform-order-sanitizer").unwrap();
    deregister_subscriber("nested-transform-order-subscriber").unwrap();
}

#[test]
fn queued_sanitizer_keeps_the_emission_time_scope_after_pop() {
    let _lock = TEST_MUTEX.lock().unwrap();
    flush_subscribers().unwrap();
    reset_global();
    setup_isolated_thread();

    let scope = push_scope(
        PushScopeParams::builder()
            .name("emission-scope")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    let expected_uuid = scope.uuid;
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Arc::new(Mutex::new(release_rx));
    let observed = Arc::new(Mutex::new(None));
    let observed_scope = Arc::clone(&observed);
    register_subscriber("scope-snapshot-subscriber", Arc::new(|_| {})).unwrap();
    register_mark_sanitize_guardrail(
        "scope-snapshot-sanitizer",
        10,
        Arc::new(move |_, fields| {
            let release_rx = Arc::clone(&release_rx);
            let observed_scope = Arc::clone(&observed_scope);
            let started_tx = started_tx.clone();
            Box::pin(async move {
                started_tx.send(()).unwrap();
                release_rx.lock().unwrap().recv().unwrap();
                *observed_scope.lock().unwrap() =
                    Some(current_scope_stack().read().unwrap().top().uuid);
                Ok(fields)
            })
        }),
    )
    .unwrap();

    emit_mark("scope-snapshot");
    started_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("sanitizer should suspend on the dispatcher");
    pop_scope(PopScopeParams::builder().handle_uuid(&scope.uuid).build()).unwrap();
    release_tx.send(()).unwrap();
    flush_subscribers().unwrap();

    assert_eq!(*observed.lock().unwrap(), Some(expected_uuid));
    deregister_mark_sanitize_guardrail("scope-snapshot-sanitizer").unwrap();
    deregister_subscriber("scope-snapshot-subscriber").unwrap();
}

#[test]
fn scope_end_sanitizer_keeps_the_ending_scope_across_await() {
    let _lock = TEST_MUTEX.lock().unwrap();
    flush_subscribers().unwrap();
    reset_global();
    setup_isolated_thread();

    let scope = push_scope(
        PushScopeParams::builder()
            .name("ending-scope")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    let expected_uuid = scope.uuid;
    let observed = Arc::new(Mutex::new(None));
    let observed_scope = Arc::clone(&observed);
    register_subscriber("scope-end-context-subscriber", Arc::new(|_| {})).unwrap();
    register_scope_sanitize_end_guardrail(
        "scope-end-context-sanitizer",
        10,
        Arc::new(move |_, fields| {
            let observed_scope = Arc::clone(&observed_scope);
            Box::pin(async move {
                tokio::task::yield_now().await;
                *observed_scope.lock().unwrap() =
                    Some(current_scope_stack().read().unwrap().top().uuid);
                Ok(fields)
            })
        }),
    )
    .unwrap();

    pop_scope(PopScopeParams::builder().handle_uuid(&scope.uuid).build()).unwrap();
    flush_subscribers().unwrap();

    assert_eq!(*observed.lock().unwrap(), Some(expected_uuid));
    deregister_scope_sanitize_end_guardrail("scope-end-context-sanitizer").unwrap();
    deregister_subscriber("scope-end-context-subscriber").unwrap();
}

#[test]
fn dispatcher_continues_after_subscriber_panic() {
    let _lock = TEST_MUTEX.lock().unwrap();
    flush_subscribers().unwrap();
    reset_global();
    setup_isolated_thread();

    let observed = Arc::new(Mutex::new(Vec::new()));
    let observed_events = Arc::clone(&observed);
    register_subscriber(
        "panic-isolated-subscriber",
        Arc::new(move |event| {
            if event.name() == "panic-isolated" {
                panic!("subscriber failed");
            }
            observed_events
                .lock()
                .unwrap()
                .push(event.name().to_string());
        }),
    )
    .unwrap();

    emit_mark("panic-isolated");
    emit_mark("after-panic");
    flush_subscribers().unwrap();
    deregister_subscriber("panic-isolated-subscriber").unwrap();

    assert_eq!(observed.lock().unwrap().as_slice(), ["after-panic"]);
}

#[test]
fn dispatcher_clears_observability_fields_when_an_async_sanitizer_fails() {
    let _lock = TEST_MUTEX.lock().unwrap();
    flush_subscribers().unwrap();
    reset_global();
    setup_isolated_thread();

    let observed = Arc::new(Mutex::new(Vec::new()));
    let observed_events = Arc::clone(&observed);
    register_subscriber(
        "fail-closed-sanitizer-subscriber",
        Arc::new(move |event| observed_events.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_mark_sanitize_guardrail(
        "fail-closed-mark-sanitizer",
        10,
        Arc::new(|_, _| {
            Box::pin(async {
                Err(FlowError::Internal(
                    "intentional event-sanitizer failure".to_string(),
                ))
            })
        }),
    )
    .unwrap();

    event(
        EmitMarkEventParams::builder()
            .name("unsanitized-fallback")
            .data(json!({"original_data": true}))
            .metadata(json!({"original_metadata": true}))
            .build(),
    )
    .unwrap();
    flush_subscribers().unwrap();

    let observed = observed.lock().unwrap();
    assert_eq!(observed.len(), 1);
    assert_eq!(observed[0].name(), "unsanitized-fallback");
    assert_eq!(observed[0].sanitize_fields().data, None);
    assert_eq!(observed[0].sanitize_fields().metadata, None);
    drop(observed);
    deregister_mark_sanitize_guardrail("fail-closed-mark-sanitizer").unwrap();
    deregister_subscriber("fail-closed-sanitizer-subscriber").unwrap();
}

#[test]
fn mark_emission_skips_sanitizers_without_subscribers() {
    let _lock = TEST_MUTEX.lock().unwrap();
    flush_subscribers().unwrap();
    reset_global();
    setup_isolated_thread();

    let sanitizer_called = Arc::new(AtomicBool::new(false));
    let called = Arc::clone(&sanitizer_called);
    register_mark_sanitize_guardrail(
        "unused-mark-sanitizer",
        10,
        Arc::new(move |_, fields| {
            called.store(true, Ordering::Release);
            Box::pin(async move { Ok(fields) })
        }),
    )
    .unwrap();

    emit_mark("no-subscribers");
    flush_subscribers().unwrap();
    deregister_mark_sanitize_guardrail("unused-mark-sanitizer").unwrap();

    assert!(!sanitizer_called.load(Ordering::Acquire));
}

#[test]
fn dispatcher_clears_observability_fields_when_an_async_sanitizer_panics() {
    let _lock = TEST_MUTEX.lock().unwrap();
    flush_subscribers().unwrap();
    reset_global();
    setup_isolated_thread();

    let observed = Arc::new(Mutex::new(Vec::<Event>::new()));
    let observed_events = Arc::clone(&observed);
    register_subscriber(
        "panic-sanitizer-subscriber",
        Arc::new(move |event| observed_events.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_mark_sanitize_guardrail(
        "successful-mark-sanitizer",
        0,
        Arc::new(|_, mut fields| {
            Box::pin(async move {
                fields.data = Some(json!({"redacted": true}));
                Ok(fields)
            })
        }),
    )
    .unwrap();
    register_mark_sanitize_guardrail(
        "panic-mark-sanitizer",
        10,
        Arc::new(|_, _| Box::pin(async { panic!("intentional event-sanitizer panic") })),
    )
    .unwrap();

    emit_mark("panic-fallback");
    flush_subscribers().unwrap();

    let events = observed.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].name(), "panic-fallback");
    assert_eq!(events[0].sanitize_fields().data, None);
    drop(events);
    deregister_mark_sanitize_guardrail("successful-mark-sanitizer").unwrap();
    deregister_mark_sanitize_guardrail("panic-mark-sanitizer").unwrap();
    deregister_subscriber("panic-sanitizer-subscriber").unwrap();
}
