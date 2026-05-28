// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Asynchronous subscriber delivery for native targets.

use crate::api::event::Event;
use crate::api::runtime::EventSubscriberFn;
use crate::error::Result;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::cell::Cell;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::OnceLock;
    use std::sync::mpsc::{self, Sender};

    use super::*;
    use crate::api::runtime::scope_stack::{
        ScopeStackHandle, capture_thread_scope_stack, current_scope_stack,
        restore_thread_scope_stack, set_thread_scope_stack,
    };
    use crate::error::FlowError;

    enum DispatcherMessage {
        Deliver {
            event: Box<Event>,
            subscribers: Vec<EventSubscriberFn>,
            scope_stack: ScopeStackHandle,
        },
        Flush {
            done: Sender<()>,
        },
    }

    static DISPATCHER: OnceLock<std::result::Result<Sender<DispatcherMessage>, String>> =
        OnceLock::new();

    thread_local! {
        static IN_DISPATCHER: Cell<bool> = const { Cell::new(false) };
    }

    pub(super) fn dispatch_event(event: &Event, subscribers: &[EventSubscriberFn]) {
        if subscribers.is_empty() {
            return;
        }
        let message = DispatcherMessage::Deliver {
            event: Box::new(event.clone()),
            subscribers: subscribers.to_vec(),
            scope_stack: current_scope_stack(),
        };
        match dispatcher_sender() {
            Ok(sender) => {
                if let Err(error) = sender.send(message) {
                    eprintln!("nemo_relay: failed to queue subscriber event: {error}");
                }
            }
            Err(error) => {
                eprintln!("nemo_relay: failed to start subscriber dispatcher: {error}");
            }
        }
    }

    pub(super) fn flush_subscribers() -> Result<()> {
        if IN_DISPATCHER.with(Cell::get) {
            return Ok(());
        }
        let Some(sender_result) = DISPATCHER.get() else {
            return Ok(());
        };
        let sender = sender_result
            .as_ref()
            .map_err(|error| FlowError::Internal(error.clone()))?;
        let (done_tx, done_rx) = mpsc::channel();
        sender
            .send(DispatcherMessage::Flush { done: done_tx })
            .map_err(|error| {
                FlowError::Internal(format!("failed to queue subscriber flush: {error}"))
            })?;
        done_rx
            .recv()
            .map_err(|error| FlowError::Internal(format!("subscriber flush failed: {error}")))?;
        Ok(())
    }

    fn dispatcher_sender() -> std::result::Result<Sender<DispatcherMessage>, String> {
        DISPATCHER
            .get_or_init(|| {
                let (tx, rx) = mpsc::channel::<DispatcherMessage>();
                std::thread::Builder::new()
                    .name("nemo-relay-subscriber-dispatcher".into())
                    .spawn(move || {
                        while let Ok(message) = rx.recv() {
                            match message {
                                DispatcherMessage::Deliver {
                                    event,
                                    subscribers,
                                    scope_stack,
                                } => {
                                    let previous_scope_stack = capture_thread_scope_stack();
                                    set_thread_scope_stack(scope_stack);
                                    IN_DISPATCHER.with(|flag| flag.set(true));
                                    for subscriber in subscribers {
                                        if catch_unwind(AssertUnwindSafe(|| subscriber(&event)))
                                            .is_err()
                                        {
                                            eprintln!(
                                                "nemo_relay: event subscriber callback panicked"
                                            );
                                        }
                                    }
                                    IN_DISPATCHER.with(|flag| flag.set(false));
                                    restore_thread_scope_stack(previous_scope_stack);
                                }
                                DispatcherMessage::Flush { done } => {
                                    let _ = done.send(());
                                }
                            }
                        }
                    })
                    .map(|_| tx)
                    .map_err(|error| error.to_string())
            })
            .clone()
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;

    pub(super) fn dispatch_event(event: &Event, subscribers: &[EventSubscriberFn]) {
        for subscriber in subscribers {
            subscriber(event);
        }
    }

    pub(super) fn flush_subscribers() -> Result<()> {
        Ok(())
    }
}

/// Queue an event for subscriber delivery.
pub(crate) fn dispatch_event(event: &Event, subscribers: &[EventSubscriberFn]) {
    #[cfg(not(target_arch = "wasm32"))]
    native::dispatch_event(event, subscribers);
    #[cfg(target_arch = "wasm32")]
    wasm::dispatch_event(event, subscribers);
}

/// Wait for all queued subscriber callbacks submitted before this call.
pub fn flush_subscribers() -> Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        native::flush_subscribers()
    }
    #[cfg(target_arch = "wasm32")]
    {
        wasm::flush_subscribers()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::sync::{Arc, Mutex, OnceLock, mpsc};
    use std::time::Duration;

    use serde_json::json;

    use super::*;
    use crate::api::event::{BaseEvent, Event, MarkEvent};

    static TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

    fn test_guard() -> std::sync::MutexGuard<'static, ()> {
        TEST_MUTEX.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn mark(name: &str) -> Event {
        Event::Mark(MarkEvent::new(
            BaseEvent::builder().name(name).data(json!({})).build(),
            None,
            None,
        ))
    }

    #[test]
    fn dispatch_event_returns_while_subscriber_is_blocked() {
        let _guard = test_guard();
        flush_subscribers().unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Arc::new(Mutex::new(release_rx));
        let subscriber: EventSubscriberFn = Arc::new(move |_event| {
            let _ = started_tx.send(());
            let _ = release_rx.lock().unwrap().recv();
        });

        dispatch_event(&mark("nonblocking"), &[subscriber]);

        started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("subscriber should start on dispatcher thread");
        release_tx.send(()).unwrap();
        flush_subscribers().unwrap();
    }

    #[test]
    fn dispatcher_preserves_event_and_subscriber_order() {
        let _guard = test_guard();
        flush_subscribers().unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let first_observed = Arc::clone(&observed);
        let second_observed = Arc::clone(&observed);
        let first: EventSubscriberFn = Arc::new(move |event| {
            first_observed
                .lock()
                .unwrap()
                .push(format!("first:{}", event.name()));
        });
        let second: EventSubscriberFn = Arc::new(move |event| {
            second_observed
                .lock()
                .unwrap()
                .push(format!("second:{}", event.name()));
        });
        let subscribers = [first, second];

        dispatch_event(&mark("one"), &subscribers);
        dispatch_event(&mark("two"), &subscribers);
        flush_subscribers().unwrap();

        assert_eq!(
            observed.lock().unwrap().as_slice(),
            ["first:one", "second:one", "first:two", "second:two"]
        );
    }

    #[test]
    fn dispatcher_continues_after_subscriber_panic() {
        let _guard = test_guard();
        flush_subscribers().unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let observed_after_panic = Arc::clone(&observed);
        let panic_subscriber: EventSubscriberFn = Arc::new(|_event| panic!("subscriber failed"));
        let capture_subscriber: EventSubscriberFn = Arc::new(move |event| {
            observed_after_panic
                .lock()
                .unwrap()
                .push(event.name().to_string());
        });

        dispatch_event(
            &mark("panic-isolated"),
            &[panic_subscriber, capture_subscriber],
        );
        flush_subscribers().unwrap();

        assert_eq!(observed.lock().unwrap().as_slice(), ["panic-isolated"]);
    }
}
