// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for native subscriber dispatch behavior.

use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use nemo_relay::api::runtime::{
    NemoRelayContextState, create_scope_stack, global_context, set_thread_scope_stack,
};
use nemo_relay::api::scope::{EmitMarkEventParams, event};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};

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
