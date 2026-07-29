// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use super::EventSubscriberFn;
use super::native::{
    DispatcherMessage, dispatcher_sender, enqueue_dispatch_message, flush_subscribers,
    register_async_publication, sanitize_event_snapshot, set_sanitizer_runtime_failure_for_test,
    spawn_background_publication,
};
use crate::api::registry::RegistryRecord;
use crate::api::runtime::EventSanitizeFn;
use crate::api::runtime::scope_stack::current_scope_stack;
use std::sync::{Arc, Mutex, mpsc};

#[test]
fn flush_waits_for_active_but_not_later_publication_barriers() {
    let _lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    flush_subscribers().unwrap();
    let first = register_async_publication().expect("first publication barrier");
    let sender = dispatcher_sender().expect("dispatcher sender");
    let delivered = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let subscriber: EventSubscriberFn = {
        let delivered = delivered.clone();
        std::sync::Arc::new(move |event| {
            delivered
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(event.name().to_string());
        })
    };
    let queued_event = serde_json::from_value(serde_json::json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": "019c1df6-4a57-7000-8000-000000000001",
        "timestamp": "2026-07-28T00:00:00Z",
        "name": "queued-before-flush"
    }))
    .expect("valid event");
    sender
        .send(DispatcherMessage::Deliver {
            event: Box::new(queued_event),
            transform: None,
            sanitizers: Vec::new(),
            subscribers: vec![subscriber.clone()],
            scope_stack: current_scope_stack(),
            publication_context: None,
        })
        .unwrap();
    let (flush_tx, flush_rx) = mpsc::channel();
    sender
        .send(DispatcherMessage::Flush { done: flush_tx })
        .unwrap();
    let later = register_async_publication().expect("later publication barrier");

    assert!(
        flush_rx
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err(),
        "flush must wait for an active publication barrier"
    );
    let deferred_event = serde_json::from_value(serde_json::json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": "019c1df6-4a57-7000-8000-000000000002",
        "timestamp": "2026-07-28T00:00:00Z",
        "name": "deferred-at-barrier"
    }))
    .expect("valid event");
    first
        .sender
        .send(vec![DispatcherMessage::Deliver {
            event: Box::new(deferred_event),
            transform: None,
            sanitizers: Vec::new(),
            subscribers: vec![subscriber],
            scope_stack: current_scope_stack(),
            publication_context: None,
        }])
        .unwrap();
    flush_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("flush queued before the later barrier must complete");
    assert_eq!(
        *delivered.lock().unwrap_or_else(|error| error.into_inner()),
        ["deferred-at-barrier", "queued-before-flush"],
        "the barrier must publish deferred work at its reserved FIFO position"
    );
    later.sender.send(Vec::new()).unwrap();
    flush_subscribers().unwrap();
}

#[test]
fn flush_does_not_wait_for_later_delivery() {
    let _lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    flush_subscribers().unwrap();
    let barrier = register_async_publication().expect("publication barrier");
    let sender = dispatcher_sender().expect("dispatcher sender");
    let (flush_tx, flush_rx) = mpsc::channel();
    sender
        .send(DispatcherMessage::Flush { done: flush_tx })
        .unwrap();

    let (release_tx, release_rx) = tokio::sync::oneshot::channel();
    let event: crate::api::event::Event = serde_json::from_value(serde_json::json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": "019c1df6-4a57-7000-8000-000000000003",
        "timestamp": "2026-07-28T00:00:00Z",
        "name": "queued-after-flush"
    }))
    .expect("valid event");
    sender
        .send(DispatcherMessage::Deliver {
            event: Box::new(event),
            transform: Some(Box::new(move |event| {
                Box::pin(async move {
                    let _ = release_rx.await;
                    event
                })
            })),
            sanitizers: Vec::new(),
            subscribers: Vec::new(),
            scope_stack: current_scope_stack(),
            publication_context: None,
        })
        .unwrap();
    barrier.sender.send(Vec::new()).unwrap();

    let flush_result = flush_rx.recv_timeout(std::time::Duration::from_millis(100));
    let _ = release_tx.send(());
    flush_subscribers().unwrap();
    assert!(
        flush_result.is_ok(),
        "a delivery queued after a flush must not delay that flush"
    );
}

#[test]
fn nested_publication_barrier_precedes_already_queued_delivery() {
    let _lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    flush_subscribers().unwrap();
    let sender = dispatcher_sender().expect("dispatcher sender");
    let delivered = Arc::new(Mutex::new(Vec::new()));
    let subscriber: EventSubscriberFn = {
        let delivered = Arc::clone(&delivered);
        Arc::new(move |event| {
            delivered
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(event.name().to_string());
        })
    };
    let event = |uuid: &str, name: &str| {
        serde_json::from_value(serde_json::json!({
            "kind": "mark",
            "atof_version": "0.1",
            "uuid": uuid,
            "timestamp": "2026-07-28T00:00:00Z",
            "name": name
        }))
        .expect("valid event")
    };
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let nested_subscriber = subscriber.clone();
    let nested_scope_stack = current_scope_stack();
    sender
        .send(DispatcherMessage::Deliver {
            event: Box::new(event("019c1df6-4a57-7000-8000-000000000004", "outer")),
            transform: Some(Box::new(move |event| {
                Box::pin(async move {
                    started_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    assert!(enqueue_dispatch_message(DispatcherMessage::Deliver {
                        event: Box::new(
                            serde_json::from_value(serde_json::json!({
                                "kind": "mark",
                                "atof_version": "0.1",
                                "uuid": "019c1df6-4a57-7000-8000-000000000005",
                                "timestamp": "2026-07-28T00:00:00Z",
                                "name": "nested-start"
                            }))
                            .expect("valid event"),
                        ),
                        transform: None,
                        sanitizers: Vec::new(),
                        subscribers: vec![nested_subscriber.clone()],
                        scope_stack: nested_scope_stack.clone(),
                        publication_context: None,
                    }));
                    let publication =
                        register_async_publication().expect("nested publication barrier");
                    publication
                        .sender
                        .send(vec![DispatcherMessage::Deliver {
                            event: Box::new(
                                serde_json::from_value(serde_json::json!({
                                    "kind": "mark",
                                    "atof_version": "0.1",
                                    "uuid": "019c1df6-4a57-7000-8000-000000000006",
                                    "timestamp": "2026-07-28T00:00:00Z",
                                    "name": "nested-end"
                                }))
                                .expect("valid event"),
                            ),
                            transform: None,
                            sanitizers: Vec::new(),
                            subscribers: vec![nested_subscriber],
                            scope_stack: nested_scope_stack,
                            publication_context: None,
                        }])
                        .unwrap();
                    event
                })
            })),
            sanitizers: Vec::new(),
            subscribers: vec![subscriber.clone()],
            scope_stack: current_scope_stack(),
            publication_context: None,
        })
        .unwrap();
    started_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("outer transform should start");
    sender
        .send(DispatcherMessage::Deliver {
            event: Box::new(event("019c1df6-4a57-7000-8000-000000000007", "later")),
            transform: None,
            sanitizers: Vec::new(),
            subscribers: vec![subscriber],
            scope_stack: current_scope_stack(),
            publication_context: None,
        })
        .unwrap();
    release_tx.send(()).unwrap();
    flush_subscribers().unwrap();
    assert_eq!(
        *delivered.lock().unwrap_or_else(|error| error.into_inner()),
        ["outer", "nested-start", "nested-end", "later"]
    );
}

#[test]
fn detached_publications_share_one_background_executor_thread() {
    let _lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = tokio::sync::watch::channel(false);
    for _ in 0..32 {
        let started_tx = started_tx.clone();
        let mut release_rx = release_rx.clone();
        assert!(spawn_background_publication(async move {
            started_tx.send(std::thread::current().id()).unwrap();
            while !*release_rx.borrow() {
                release_rx.changed().await.unwrap();
            }
        }));
    }
    drop(started_tx);
    let threads = (0..32)
        .map(|_| {
            started_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("background publication should start")
        })
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(
        threads.len(),
        1,
        "detached publications must not allocate one OS thread per future"
    );
    release_tx.send(true).unwrap();
}

#[test]
fn sanitizer_runtime_failure_preserves_untransformed_event_snapshot() {
    let _lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let event: crate::api::event::Event = serde_json::from_value(serde_json::json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": "019c1df6-4a57-7000-8000-000000000008",
        "timestamp": "2026-07-28T00:00:00Z",
        "name": "fail-open-runtime"
    }))
    .expect("valid event");
    let sanitizer: EventSanitizeFn = Arc::new(|_, _| {
        Box::pin(async {
            panic!("the unavailable sanitizer runtime must not invoke middleware");
        })
    });

    set_sanitizer_runtime_failure_for_test(Some("injected runtime failure"));
    let (published, nested) = sanitize_event_snapshot(
        event.clone(),
        None,
        vec![RegistryRecord::new("unreachable", 0, sanitizer)],
        None,
    );
    set_sanitizer_runtime_failure_for_test(None);

    assert_eq!(published, Some(event));
    assert!(nested.is_empty());
}
