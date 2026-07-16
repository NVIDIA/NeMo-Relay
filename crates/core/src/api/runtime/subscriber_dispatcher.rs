// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Asynchronous subscriber delivery for native targets.

use crate::api::event::Event;
use crate::api::runtime::EventSubscriberFn;
use crate::error::Result;

mod native {
    use std::cell::Cell;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::OnceLock;
    use std::sync::mpsc::{self, Receiver, Sender};

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

    pub(super) fn dispatch_event(event: &Event, subscribers: &[EventSubscriberFn]) -> bool {
        if subscribers.is_empty() {
            return true;
        }
        let message = DispatcherMessage::Deliver {
            event: Box::new(event.clone()),
            subscribers: subscribers.to_vec(),
            scope_stack: current_scope_stack(),
        };
        match dispatcher_sender() {
            Ok(sender) => {
                if sender.send(message).is_err() {
                    log::warn!(
                        target: "nemo_relay.runtime",
                        event = "subscriber_event_dropped",
                        reason = "dispatcher_disconnected";
                        "Subscriber event was dropped because the dispatcher stopped"
                    );
                    false
                } else {
                    true
                }
            }
            Err(_error) => {
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "subscriber_dispatcher_failed",
                    error_kind = "initialization";
                    "Subscriber dispatcher failed to start"
                );
                false
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
        DISPATCHER.get_or_init(start_dispatcher).clone()
    }

    fn start_dispatcher() -> std::result::Result<Sender<DispatcherMessage>, String> {
        let (tx, rx) = mpsc::channel::<DispatcherMessage>();
        let sender = std::thread::Builder::new()
            .name("nemo-relay-subscriber-dispatcher".into())
            .spawn(move || run_dispatcher(rx))
            .map(|_| tx)
            .map_err(|error| error.to_string());
        if sender.is_ok() {
            log::info!(
                target: "nemo_relay.runtime",
                event = "subscriber_dispatcher_started";
                "Subscriber dispatcher started"
            );
        }
        sender
    }

    fn run_dispatcher(rx: Receiver<DispatcherMessage>) {
        while let Ok(message) = rx.recv() {
            match message {
                DispatcherMessage::Flush { done } => {
                    let pending_flushes = drain_pending_messages(&rx);
                    let _ = done.send(());
                    for pending in pending_flushes {
                        let _ = pending.send(());
                    }
                }
                message => handle_message(message),
            }
        }
    }

    fn drain_pending_messages(rx: &Receiver<DispatcherMessage>) -> Vec<Sender<()>> {
        let mut pending_flushes = Vec::new();
        while let Ok(message) = rx.try_recv() {
            match message {
                DispatcherMessage::Flush { done } => pending_flushes.push(done),
                message => handle_message(message),
            }
        }
        pending_flushes
    }

    fn handle_message(message: DispatcherMessage) {
        match message {
            DispatcherMessage::Deliver {
                event,
                subscribers,
                scope_stack,
            } => deliver_event(event, subscribers, scope_stack),
            DispatcherMessage::Flush { done } => {
                let _ = done.send(());
            }
        }
    }

    fn deliver_event(
        event: Box<Event>,
        subscribers: Vec<EventSubscriberFn>,
        scope_stack: ScopeStackHandle,
    ) {
        let previous_scope_stack = capture_thread_scope_stack();
        set_thread_scope_stack(scope_stack);
        IN_DISPATCHER.with(|flag| flag.set(true));
        for subscriber in subscribers {
            if catch_unwind(AssertUnwindSafe(|| subscriber(&event))).is_err() {
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "subscriber_callback_panicked";
                    "Event subscriber callback panicked"
                );
            }
        }
        IN_DISPATCHER.with(|flag| flag.set(false));
        restore_thread_scope_stack(previous_scope_stack);
    }
}

/// Queue an event for subscriber delivery.
pub(crate) fn dispatch_event(event: &Event, subscribers: &[EventSubscriberFn]) -> bool {
    native::dispatch_event(event, subscribers)
}

/// Wait for all queued subscriber callbacks submitted before this call.
pub fn flush_subscribers() -> Result<()> {
    native::flush_subscribers()
}
