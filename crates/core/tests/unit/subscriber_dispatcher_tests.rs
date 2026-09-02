// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use super::native::{
    DispatcherLoopState, DispatcherMessage, PendingFlush, PublicationLineage, PublicationPermit,
    dispatch_sanitized_event_with_delivery, dispatcher_sender, enqueue_dispatch_message,
    flush_queued_subscribers, flush_subscribers, prepare_for_fork, register_async_publication,
    register_pending_publication, resume_after_fork_parent, sanitize_event_snapshot,
    set_sanitizer_runtime_failure_for_test, spawn_background_publication,
};
use super::{EventSubscriberFn, SubscriberDelivery, publication_context, with_publication_context};
use crate::api::event::{BaseEvent, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent};
use crate::api::registry::RegistryRecord;
use crate::api::runtime::scope_stack::current_scope_stack;
use crate::api::runtime::{EventMetadataInjectorFn, EventSanitizeFn};
use crate::api::scope::ScopeType;
use crate::error::FlowError;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use uuid::Uuid;

#[test]
fn publication_context_and_completed_delivery_restore_the_calling_thread() {
    assert!(publication_context::<String>().is_none());
    let observed = with_publication_context(Some(Arc::new("binding".to_string())), || {
        publication_context::<String>().map(|value| value.as_str().to_string())
    });
    assert_eq!(observed.as_deref(), Some("binding"));
    assert!(publication_context::<String>().is_none());
    futures::executor::block_on(SubscriberDelivery::completed().wait()).unwrap();
}

#[test]
fn subscriber_dispatcher_parent_fork_hooks_validate_balanced_calls() {
    let _lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    prepare_for_fork();
    assert!(std::panic::catch_unwind(prepare_for_fork).is_err());
    resume_after_fork_parent();
    assert!(std::panic::catch_unwind(resume_after_fork_parent).is_err());
}

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
            injectors: Vec::new(),
            sanitizers: Vec::new(),
            subscribers: vec![subscriber.clone()],
            scope_stack: current_scope_stack(),
            publication_context: None,
            lineage: None,
            completion: None,
        })
        .unwrap();
    let (flush_tx, flush_rx) = mpsc::channel();
    sender
        .send(DispatcherMessage::Flush {
            done: flush_tx,
            include_pending: true,
        })
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
            injectors: Vec::new(),
            sanitizers: Vec::new(),
            subscribers: vec![subscriber],
            scope_stack: current_scope_stack(),
            publication_context: None,
            lineage: None,
            completion: None,
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
fn pending_publication_defers_flush_without_blocking_unrelated_delivery() {
    let _lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    flush_subscribers().unwrap();
    let pending = register_pending_publication().expect("pending publication");
    let (delivered_tx, delivered_rx) = mpsc::channel();
    let subscriber: EventSubscriberFn = Arc::new(move |_event| {
        delivered_tx.send(()).unwrap();
    });
    let event = serde_json::from_value(serde_json::json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": "019c1df6-4a57-7000-8000-000000000004",
        "timestamp": "2026-07-28T00:00:00Z",
        "name": "unrelated-while-pending"
    }))
    .expect("valid event");
    enqueue_dispatch_message(DispatcherMessage::Deliver {
        event: Box::new(event),
        transform: None,
        injectors: Vec::new(),
        sanitizers: Vec::new(),
        subscribers: vec![subscriber],
        scope_stack: current_scope_stack(),
        publication_context: None,
        lineage: None,
        completion: None,
    });
    delivered_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("pending publication must not block unrelated delivery");
    flush_queued_subscribers().expect("queued-only flush must ignore managed work");

    let (flush_tx, flush_rx) = mpsc::channel();
    dispatcher_sender()
        .expect("dispatcher sender")
        .send(DispatcherMessage::Flush {
            done: flush_tx,
            include_pending: true,
        })
        .unwrap();
    assert!(
        matches!(
            flush_rx.recv_timeout(std::time::Duration::from_millis(50)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ),
        "flush must wait for work registered before it"
    );

    drop(pending);
    flush_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("flush must complete after pending work");
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
        .send(DispatcherMessage::Flush {
            done: flush_tx,
            include_pending: true,
        })
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
            injectors: Vec::new(),
            sanitizers: Vec::new(),
            subscribers: Vec::new(),
            scope_stack: current_scope_stack(),
            publication_context: None,
            lineage: None,
            completion: None,
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
fn subscriber_delivery_receipt_waits_for_its_event() {
    let _lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    flush_subscribers().unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Arc::new(Mutex::new(release_rx));
    let subscriber: EventSubscriberFn = Arc::new(move |_event| {
        started_tx.send(()).unwrap();
        release_rx
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .recv()
            .unwrap();
    });
    let event = serde_json::from_value(serde_json::json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": "019c1df6-4a57-7000-8000-000000000017",
        "timestamp": "2026-07-28T00:00:00Z",
        "name": "tracked-delivery"
    }))
    .expect("valid event");
    let delivery = dispatch_sanitized_event_with_delivery(
        event,
        Vec::new(),
        &[subscriber],
        current_scope_stack(),
    )
    .unwrap();
    started_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("tracked subscriber should start");

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    let mut wait = Box::pin(delivery.wait());
    assert!(
        runtime
            .block_on(async {
                tokio::time::timeout(std::time::Duration::from_millis(50), wait.as_mut()).await
            })
            .is_err(),
        "delivery receipt must remain pending while its subscriber is active"
    );
    release_tx.send(()).unwrap();
    runtime
        .block_on(async {
            tokio::time::timeout(std::time::Duration::from_secs(1), wait.as_mut()).await
        })
        .expect("delivery receipt should complete after subscriber delivery")
        .unwrap();
    flush_subscribers().unwrap();
}

#[test]
fn subscriber_delivery_receipt_does_not_capture_later_events() {
    let _lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    flush_subscribers().unwrap();
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
    let delivery = dispatch_sanitized_event_with_delivery(
        event("019c1df6-4a57-7000-8000-000000000018", "tracked"),
        Vec::new(),
        &[Arc::new(|_event| {})],
        current_scope_stack(),
    )
    .unwrap();

    let (later_started_tx, later_started_rx) = mpsc::channel();
    let (release_later_tx, release_later_rx) = mpsc::channel();
    enqueue_dispatch_message(DispatcherMessage::Deliver {
        event: Box::new(event(
            "019c1df6-4a57-7000-8000-000000000019",
            "later-blocked",
        )),
        transform: Some(Box::new(move |event| {
            Box::pin(async move {
                later_started_tx.send(()).unwrap();
                release_later_rx.recv().unwrap();
                event
            })
        })),
        injectors: Vec::new(),
        sanitizers: Vec::new(),
        subscribers: Vec::new(),
        scope_stack: current_scope_stack(),
        publication_context: None,
        lineage: None,
        completion: None,
    });
    later_started_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("later delivery should block the dispatcher");

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    let result = runtime.block_on(async {
        tokio::time::timeout(std::time::Duration::from_millis(100), delivery.wait()).await
    });
    release_later_tx.send(()).unwrap();
    flush_subscribers().unwrap();
    result
        .expect("tracked delivery must not wait for a later queued event")
        .unwrap();
}

#[test]
fn pending_flushes_do_not_acknowledge_out_of_order() {
    let first_lineage = Arc::new(PublicationLineage::default());
    let second_lineage = Arc::new(PublicationLineage::default());
    let first_permit = PublicationPermit::new(Arc::clone(&first_lineage));
    let second_permit = PublicationPermit::new(Arc::clone(&second_lineage));
    let (first_tx, first_rx) = mpsc::channel();
    let (second_tx, second_rx) = mpsc::channel();
    let mut state = DispatcherLoopState {
        active_lineages: Vec::new(),
        pending_flushes: vec![
            PendingFlush {
                done: first_tx,
                lineages: vec![first_lineage],
            },
            PendingFlush {
                done: second_tx,
                lineages: vec![second_lineage],
            },
        ],
    };

    drop(second_permit);
    state.complete_ready_flushes();
    assert!(matches!(
        first_rx.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    assert!(
        matches!(second_rx.try_recv(), Err(mpsc::TryRecvError::Empty)),
        "a ready later flush must not overtake an earlier pending flush"
    );

    drop(first_permit);
    state.complete_ready_flushes();
    first_rx.recv().expect("first flush should complete first");
    second_rx
        .recv()
        .expect("second flush should complete afterward");
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
                        injectors: Vec::new(),
                        sanitizers: Vec::new(),
                        subscribers: vec![nested_subscriber.clone()],
                        scope_stack: nested_scope_stack.clone(),
                        publication_context: None,
                        lineage: None,
                        completion: None,
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
                            injectors: Vec::new(),
                            sanitizers: Vec::new(),
                            subscribers: vec![nested_subscriber],
                            scope_stack: nested_scope_stack,
                            publication_context: None,
                            lineage: None,
                            completion: None,
                        }])
                        .unwrap();
                    event
                })
            })),
            injectors: Vec::new(),
            sanitizers: Vec::new(),
            subscribers: vec![subscriber.clone()],
            scope_stack: current_scope_stack(),
            publication_context: None,
            lineage: None,
            completion: None,
        })
        .unwrap();
    started_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("outer transform should start");
    sender
        .send(DispatcherMessage::Deliver {
            event: Box::new(event("019c1df6-4a57-7000-8000-000000000007", "later")),
            transform: None,
            injectors: Vec::new(),
            sanitizers: Vec::new(),
            subscribers: vec![subscriber],
            scope_stack: current_scope_stack(),
            publication_context: None,
            lineage: None,
            completion: None,
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
fn flush_waits_for_transitive_subscriber_publications_without_reordering() {
    let _lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    flush_subscribers().unwrap();
    let sender = dispatcher_sender().expect("dispatcher sender");
    let delivered = Arc::new(Mutex::new(Vec::new()));
    let (outer_started_tx, outer_started_rx) = mpsc::channel();
    let (release_outer_tx, release_outer_rx) = mpsc::channel();
    let release_outer_rx = Arc::new(Mutex::new(release_outer_rx));
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

    let grandchild_subscriber: EventSubscriberFn = {
        let delivered = Arc::clone(&delivered);
        Arc::new(move |event| {
            delivered
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(event.name().to_string());
        })
    };
    let child_subscriber: EventSubscriberFn = {
        let delivered = Arc::clone(&delivered);
        let grandchild_subscriber = grandchild_subscriber.clone();
        Arc::new(move |event| {
            delivered
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(event.name().to_string());
            assert!(enqueue_dispatch_message(DispatcherMessage::Deliver {
                event: Box::new(
                    serde_json::from_value(serde_json::json!({
                        "kind": "mark",
                        "atof_version": "0.1",
                        "uuid": "019c1df6-4a57-7000-8000-000000000010",
                        "timestamp": "2026-07-28T00:00:00Z",
                        "name": "grandchild"
                    }))
                    .expect("valid event"),
                ),
                transform: None,
                injectors: Vec::new(),
                sanitizers: Vec::new(),
                subscribers: vec![grandchild_subscriber.clone()],
                scope_stack: current_scope_stack(),
                publication_context: None,
                lineage: None,
                completion: None,
            }));
        })
    };
    let outer_subscriber: EventSubscriberFn = {
        let delivered = Arc::clone(&delivered);
        let child_subscriber = child_subscriber.clone();
        let release_outer_rx = Arc::clone(&release_outer_rx);
        Arc::new(move |event| {
            delivered
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(event.name().to_string());
            outer_started_tx
                .send(())
                .expect("outer subscriber start receiver was dropped");
            release_outer_rx
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("outer subscriber was not released within 2 seconds");
            assert!(enqueue_dispatch_message(DispatcherMessage::Deliver {
                event: Box::new(
                    serde_json::from_value(serde_json::json!({
                        "kind": "mark",
                        "atof_version": "0.1",
                        "uuid": "019c1df6-4a57-7000-8000-000000000009",
                        "timestamp": "2026-07-28T00:00:00Z",
                        "name": "child"
                    }))
                    .expect("valid event"),
                ),
                transform: None,
                injectors: Vec::new(),
                sanitizers: Vec::new(),
                subscribers: vec![child_subscriber.clone()],
                scope_stack: current_scope_stack(),
                publication_context: None,
                lineage: None,
                completion: None,
            }));
        })
    };
    let later_subscriber: EventSubscriberFn = {
        let delivered = Arc::clone(&delivered);
        Arc::new(move |event| {
            delivered
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(event.name().to_string());
        })
    };

    sender
        .send(DispatcherMessage::Deliver {
            event: Box::new(event("019c1df6-4a57-7000-8000-000000000008", "outer")),
            transform: None,
            injectors: Vec::new(),
            sanitizers: Vec::new(),
            subscribers: vec![outer_subscriber],
            scope_stack: current_scope_stack(),
            publication_context: None,
            lineage: None,
            completion: None,
        })
        .unwrap();
    outer_started_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("outer subscriber did not start within 2 seconds");
    sender
        .send(DispatcherMessage::Deliver {
            event: Box::new(event("019c1df6-4a57-7000-8000-000000000011", "later")),
            transform: None,
            injectors: Vec::new(),
            sanitizers: Vec::new(),
            subscribers: vec![later_subscriber],
            scope_stack: current_scope_stack(),
            publication_context: None,
            lineage: None,
            completion: None,
        })
        .unwrap();
    release_outer_tx
        .send(())
        .expect("outer subscriber release receiver was dropped");

    flush_subscribers().unwrap();
    assert_eq!(
        *delivered.lock().unwrap_or_else(|error| error.into_inner()),
        ["outer", "later", "child", "grandchild"],
        "subscriber publications retain FIFO position and one flush waits for all descendants"
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
fn event_metadata_injection_applies_to_every_canonical_event_record() {
    let _lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let invocations = Arc::new(AtomicUsize::new(0));
    let injector: EventMetadataInjectorFn = {
        let invocations = Arc::clone(&invocations);
        Arc::new(move |event| {
            invocations.fetch_add(1, Ordering::SeqCst);
            let marker = match event.as_ref() {
                Event::Scope(scope) => {
                    assert_eq!(event.category(), Some(&scope.category));
                    match scope.scope_category {
                        ScopeCategory::Start => "scope_start",
                        ScopeCategory::End => "scope_end",
                    }
                }
                Event::Mark(_) => {
                    assert_eq!(event.category(), None);
                    "mark"
                }
            };
            assert!(!event.name().is_empty());
            assert_eq!(event.data(), Some(&serde_json::json!({"readable": true})));
            assert_eq!(
                event
                    .metadata()
                    .and_then(|metadata| metadata.get("nv.test.existing")),
                Some(&serde_json::json!("original"))
            );
            Box::pin(async move {
                Ok(BTreeMap::from([(
                    "nv.test.injector.global".to_string(),
                    serde_json::json!(marker),
                )]))
            })
        })
    };
    let injectors = vec![RegistryRecord::new("all-events", 0, injector)];
    let scope_types = [
        ScopeType::Agent,
        ScopeType::Function,
        ScopeType::Tool,
        ScopeType::Llm,
        ScopeType::Retriever,
        ScopeType::Embedder,
        ScopeType::Reranker,
        ScopeType::Guardrail,
        ScopeType::Evaluator,
        ScopeType::Custom,
        ScopeType::Unknown,
    ];

    for scope_type in scope_types {
        for (phase, expected_marker) in [
            (ScopeCategory::Start, "scope_start"),
            (ScopeCategory::End, "scope_end"),
        ] {
            let event = Event::Scope(ScopeEvent::new(
                BaseEvent::builder()
                    .name(format!("{}-{expected_marker}", scope_type.as_str()))
                    .data(serde_json::json!({"readable": true}))
                    .metadata(serde_json::json!({"nv.test.existing": "original"}))
                    .build(),
                phase,
                Vec::new(),
                EventCategory::from(scope_type),
                None,
            ));
            let (published, nested) =
                sanitize_event_snapshot(event, None, injectors.clone(), Vec::new(), None);
            let published = published.expect("metadata injection must preserve the Scope Event");

            assert_eq!(published.kind(), "scope");
            assert_eq!(published.scope_category(), Some(phase));
            assert_eq!(
                published.category().map(EventCategory::as_str),
                Some(scope_type.as_str())
            );
            assert_eq!(
                published
                    .metadata()
                    .and_then(|metadata| metadata.get("nv.test.injector.global")),
                Some(&serde_json::json!(expected_marker))
            );
            assert_eq!(
                published
                    .metadata()
                    .and_then(|metadata| metadata.get("nv.test.existing")),
                Some(&serde_json::json!("original"))
            );
            assert!(nested.is_empty());
        }
    }

    let mark = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("mark")
            .data(serde_json::json!({"readable": true}))
            .metadata(serde_json::json!({"nv.test.existing": "original"}))
            .build(),
        None,
        None,
    ));
    let (published, nested) = sanitize_event_snapshot(mark, None, injectors, Vec::new(), None);
    let published = published.expect("metadata injection must preserve the Mark Event");

    assert_eq!(published.kind(), "mark");
    assert_eq!(
        published
            .metadata()
            .and_then(|metadata| metadata.get("nv.test.injector.global")),
        Some(&serde_json::json!("mark"))
    );
    assert_eq!(
        published
            .metadata()
            .and_then(|metadata| metadata.get("nv.test.existing")),
        Some(&serde_json::json!("original"))
    );
    assert!(nested.is_empty());
    assert_eq!(invocations.load(Ordering::SeqCst), 23);
}

#[test]
fn failing_event_metadata_injector_preserves_scope_and_mark_events() {
    let injector: EventMetadataInjectorFn = Arc::new(|_| {
        Box::pin(async { Err(FlowError::Internal("expected injector failure".into())) })
    });
    let injectors = vec![RegistryRecord::new("failure", 0, injector)];
    let events = [
        Event::Scope(ScopeEvent::new(
            BaseEvent::builder().name("scope").build(),
            ScopeCategory::Start,
            Vec::new(),
            EventCategory::agent(),
            None,
        )),
        Event::Mark(MarkEvent::new(
            BaseEvent::builder().name("mark").build(),
            None,
            None,
        )),
    ];

    for event in events {
        let (published, nested) =
            sanitize_event_snapshot(event.clone(), None, injectors.clone(), Vec::new(), None);

        assert_eq!(published, Some(event));
        assert!(nested.is_empty());
    }
}
#[test]
fn sanitizer_runtime_failure_clears_untransformed_event_fields() {
    let _lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let event: crate::api::event::Event = serde_json::from_value(serde_json::json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": "019c1df6-4a57-7000-8000-000000000008",
        "timestamp": "2026-07-28T00:00:00Z",
        "name": "fail-closed-runtime",
        "data": {"secret": true},
        "metadata": {"secret": true}
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
        Vec::new(),
        vec![RegistryRecord::new("unreachable", 0, sanitizer)],
        None,
    );
    set_sanitizer_runtime_failure_for_test(None);

    let published = published.expect("sanitizer runtime failure still publishes the event shell");
    assert_eq!(published.name(), event.name());
    assert_eq!(published.data(), None);
    assert_eq!(published.metadata(), None);
    assert!(nested.is_empty());
}

#[test]
fn synchronous_transform_panics_drop_only_the_current_event() {
    let event: crate::api::event::Event = serde_json::from_value(serde_json::json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": "019c1df6-4a57-7000-8000-000000000015",
        "timestamp": "2026-07-28T00:00:00Z",
        "name": "synchronous-transform-panic"
    }))
    .expect("valid event");
    let transform: super::EventTransformFn =
        Box::new(|_| panic!("transform panicked before returning its future"));

    let (published, nested) =
        sanitize_event_snapshot(event, Some(transform), Vec::new(), Vec::new(), None);

    assert!(published.is_none());
    assert!(nested.is_empty());
}

#[test]
fn transforms_preserve_the_captured_propagation_root() {
    let root_uuid = Uuid::now_v7();
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("transformed-propagation-root")
            .propagation_root_uuid(root_uuid)
            .build(),
        None,
        None,
    ));
    let transform: super::EventTransformFn = Box::new(|_| {
        Box::pin(async {
            Event::Mark(MarkEvent::new(
                BaseEvent::builder().name("replacement-event").build(),
                None,
                None,
            ))
        })
    });

    let (published, nested) =
        sanitize_event_snapshot(event, Some(transform), Vec::new(), Vec::new(), None);

    assert_eq!(published.unwrap().propagation_root_uuid(), Some(root_uuid));
    assert!(nested.is_empty());
}

#[test]
fn detached_blocking_sanitizer_work_does_not_stall_publication() {
    let event: crate::api::event::Event = serde_json::from_value(serde_json::json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": "019c1df6-4a57-7000-8000-000000000016",
        "timestamp": "2026-07-28T00:00:00Z",
        "name": "detached-blocking-work"
    }))
    .expect("valid event");
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Arc::new(Mutex::new(Some(release_rx)));
    let sanitizer: EventSanitizeFn = Arc::new(move |_, fields| {
        let release_rx = release_rx.lock().unwrap().take().unwrap();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                let _ = release_rx.recv();
            });
            Ok(fields)
        })
    });

    let (completed_tx, completed_rx) = mpsc::channel();
    let publication = std::thread::spawn(move || {
        let result = sanitize_event_snapshot(
            event,
            None,
            Vec::new(),
            vec![RegistryRecord::new("detached-blocking", 0, sanitizer)],
            None,
        );
        completed_tx.send(result).unwrap();
    });
    let result = completed_rx.recv_timeout(std::time::Duration::from_secs(1));
    let _ = release_tx.send(());
    publication.join().unwrap();

    let (published, nested) =
        result.expect("detached blocking work must not stall the subscriber dispatcher");
    assert!(published.is_some());
    assert!(nested.is_empty());
}

#[test]
fn sanitizer_spawned_tasks_inherit_the_binding_publication_context() {
    let _lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let event: crate::api::event::Event = serde_json::from_value(serde_json::json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": "019c1df6-4a57-7000-8000-000000000012",
        "timestamp": "2026-07-28T00:00:00Z",
        "name": "spawned-publication-context"
    }))
    .expect("valid event");
    let (observed_tx, observed_rx) = mpsc::channel();
    let sanitizer: EventSanitizeFn = Arc::new(move |_, fields| {
        let observed_tx = observed_tx.clone();
        Box::pin(async move {
            tokio::spawn(async move {
                observed_tx
                    .send(publication_context::<String>().map(|value| value.as_str().to_string()))
                    .unwrap();
            })
            .await
            .unwrap();
            Ok(fields)
        })
    });

    let (published, nested) = sanitize_event_snapshot(
        event.clone(),
        None,
        Vec::new(),
        vec![RegistryRecord::new("spawned-context", 0, sanitizer)],
        Some(Arc::new("binding-context".to_string())),
    );

    assert_eq!(published, Some(event));
    assert!(nested.is_empty());
    assert_eq!(
        observed_rx.recv().unwrap().as_deref(),
        Some("binding-context")
    );
}

#[test]
fn detached_sanitizer_tasks_cannot_inherit_a_later_publication_context() {
    let _lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let first_event: crate::api::event::Event = serde_json::from_value(serde_json::json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": "019c1df6-4a57-7000-8000-000000000013",
        "timestamp": "2026-07-28T00:00:00Z",
        "name": "detached-publication-context"
    }))
    .expect("valid event");
    let second_event: crate::api::event::Event = serde_json::from_value(serde_json::json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": "019c1df6-4a57-7000-8000-000000000014",
        "timestamp": "2026-07-28T00:00:00Z",
        "name": "later-publication-context"
    }))
    .expect("valid event");
    let (release_tx, release_rx) = tokio::sync::oneshot::channel();
    let release_rx = Arc::new(Mutex::new(Some(release_rx)));
    let (observed_tx, observed_rx) = mpsc::channel();
    let first_sanitizer: EventSanitizeFn = Arc::new(move |_, fields| {
        let release_rx = release_rx.lock().unwrap().take().unwrap();
        let observed_tx = observed_tx.clone();
        Box::pin(async move {
            tokio::spawn(async move {
                let _ = release_rx.await;
                observed_tx
                    .send(publication_context::<String>().map(|value| value.as_str().to_string()))
                    .unwrap();
                let (done, _ignored) = mpsc::channel();
                enqueue_dispatch_message(DispatcherMessage::Flush {
                    done,
                    include_pending: true,
                });
            });
            Ok(fields)
        })
    });

    let (published, nested) = sanitize_event_snapshot(
        first_event.clone(),
        None,
        Vec::new(),
        vec![RegistryRecord::new("detached-context", 0, first_sanitizer)],
        Some(Arc::new("first-context".to_string())),
    );
    assert_eq!(published, Some(first_event));
    assert!(nested.is_empty());

    let release_tx = Arc::new(Mutex::new(Some(release_tx)));
    let second_sanitizer: EventSanitizeFn = Arc::new(move |_, fields| {
        let release_tx = release_tx.lock().unwrap().take().unwrap();
        Box::pin(async move {
            let _ = release_tx.send(());
            tokio::task::yield_now().await;
            tokio::task::yield_now().await;
            Ok(fields)
        })
    });
    let (published, nested) = sanitize_event_snapshot(
        second_event.clone(),
        None,
        Vec::new(),
        vec![RegistryRecord::new("later-context", 0, second_sanitizer)],
        Some(Arc::new("second-context".to_string())),
    );

    assert_eq!(published, Some(second_event));
    assert!(nested.is_empty());
    assert!(matches!(
        observed_rx.try_recv(),
        Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected)
    ));
}
