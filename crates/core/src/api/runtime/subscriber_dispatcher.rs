// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Asynchronous subscriber delivery for native targets.

use crate::api::event::Event;
use crate::api::registry::Guardrail;
use crate::api::runtime::{
    EventSanitizeFn, EventSubscriberFn, NemoRelayContextState, ScopeStackHandle,
};
use crate::error::Result;
use std::any::Any;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Binding-owned context captured when an event is emitted.
///
/// The dispatcher treats this value as opaque. Bindings can use it to carry
/// task-local state from synchronous emission into queued middleware.
pub type PublicationContext = Arc<dyn Any + Send + Sync>;

thread_local! {
    static THREAD_PUBLICATION_CONTEXT: RefCell<Option<PublicationContext>> = const { RefCell::new(None) };
}
tokio::task_local! {
    static TASK_PUBLICATION_CONTEXT: Option<PublicationContext>;
}

struct ThreadPublicationContextGuard(Option<PublicationContext>);

impl Drop for ThreadPublicationContextGuard {
    fn drop(&mut self) {
        THREAD_PUBLICATION_CONTEXT.with(|current| {
            current.replace(self.0.take());
        });
    }
}

fn current_publication_context() -> Option<PublicationContext> {
    TASK_PUBLICATION_CONTEXT
        .try_with(Clone::clone)
        .ok()
        .flatten()
        .or_else(|| THREAD_PUBLICATION_CONTEXT.with(|current| current.borrow().clone()))
}

/// Capture the current opaque binding publication context for a spawned task.
#[doc(hidden)]
pub fn capture_publication_context() -> Option<PublicationContext> {
    current_publication_context()
}

/// Run synchronous event emission with an opaque binding context snapshot.
#[doc(hidden)]
pub fn with_publication_context<T>(
    context: Option<PublicationContext>,
    f: impl FnOnce() -> T,
) -> T {
    let previous = THREAD_PUBLICATION_CONTEXT.with(|current| current.replace(context));
    let _guard = ThreadPublicationContextGuard(previous);
    f()
}

/// Run asynchronous event emission with an opaque binding context snapshot.
#[doc(hidden)]
pub async fn with_task_publication_context<F: Future>(
    context: Option<PublicationContext>,
    future: F,
) -> F::Output {
    TASK_PUBLICATION_CONTEXT.scope(context, future).await
}

/// Return a typed binding context while queued middleware is running.
#[doc(hidden)]
pub fn publication_context<T: Any + Send + Sync>() -> Option<Arc<T>> {
    current_publication_context()?.downcast().ok()
}

pub(crate) type EventTransformFn = Box<
    dyn FnOnce(Event) -> Pin<Box<dyn Future<Output = Event> + Send + 'static>> + Send + 'static,
>;

mod native {
    use std::cell::{Cell, RefCell};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
    use std::sync::mpsc::{self, Receiver, Sender};

    use super::*;
    #[cfg(test)]
    use crate::api::runtime::scope_stack::current_scope_stack;
    use crate::api::runtime::scope_stack::{
        ScopeStackHandle, capture_thread_scope_stack, restore_thread_scope_stack,
        set_thread_scope_stack, snapshot_scope_stack,
    };
    use crate::error::FlowError;

    enum DispatcherMessage {
        Deliver {
            event: Box<Event>,
            transform: Option<EventTransformFn>,
            sanitizers: Vec<Guardrail<EventSanitizeFn>>,
            subscribers: Vec<EventSubscriberFn>,
            scope_stack: ScopeStackHandle,
            publication_context: Option<PublicationContext>,
        },
        Flush {
            done: Sender<()>,
        },
        Barrier {
            publications: Receiver<Vec<DispatcherMessage>>,
        },
    }

    type DispatcherState = Option<std::result::Result<Sender<DispatcherMessage>, String>>;
    type SanitizerRuntimeState = Option<std::result::Result<tokio::runtime::Runtime, String>>;
    type BackgroundPublication = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;
    type BackgroundPublicationState = Option<
        std::result::Result<tokio::sync::mpsc::UnboundedSender<BackgroundPublication>, String>,
    >;

    struct ProcessState {
        dispatcher: Mutex<DispatcherState>,
        sanitizer_runtime: Mutex<SanitizerRuntimeState>,
        background_publications: Mutex<BackgroundPublicationState>,
        dispatcher_failure_logged: AtomicBool,
        sanitizer_runtime_failure_logged: AtomicBool,
        background_publication_failure_logged: AtomicBool,
    }

    impl ProcessState {
        fn new() -> Self {
            Self {
                dispatcher: Mutex::new(None),
                sanitizer_runtime: Mutex::new(None),
                background_publications: Mutex::new(None),
                dispatcher_failure_logged: AtomicBool::new(false),
                sanitizer_runtime_failure_logged: AtomicBool::new(false),
                background_publication_failure_logged: AtomicBool::new(false),
            }
        }
    }

    // Process states are intentionally never reclaimed after becoming active.
    // A forked child cannot safely drop the inherited state because another
    // vanished parent thread may have held one of its mutexes at fork time.
    static PROCESS_STATE: AtomicPtr<ProcessState> = AtomicPtr::new(std::ptr::null_mut());
    thread_local! {
        static IN_DISPATCHER: Cell<bool> = const { Cell::new(false) };
        static PREPARED_FORK_STATE: Cell<*mut ProcessState> = const { Cell::new(std::ptr::null_mut()) };
    }
    tokio::task_local! {
        static ASYNC_PUBLICATION_MESSAGES: RefCell<Option<Vec<DispatcherMessage>>>;
    }

    struct DispatchGuard;

    pub(crate) struct AsyncPublication {
        sender: Sender<Vec<DispatcherMessage>>,
    }

    fn process_state() -> &'static ProcessState {
        let mut state = PROCESS_STATE.load(Ordering::Acquire);
        if state.is_null() {
            let fresh = Box::into_raw(Box::new(ProcessState::new()));
            match PROCESS_STATE.compare_exchange(
                std::ptr::null_mut(),
                fresh,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => state = fresh,
                Err(existing) => {
                    state = existing;
                    unsafe { drop(Box::from_raw(fresh)) };
                }
            }
        }
        unsafe { &*state }
    }

    impl DispatchGuard {
        fn enter() -> Self {
            IN_DISPATCHER.with(|flag| flag.set(true));
            Self
        }
    }

    impl Drop for DispatchGuard {
        fn drop(&mut self) {
            IN_DISPATCHER.with(|flag| flag.set(false));
        }
    }

    fn immutable_scope_stack(scope_stack: &ScopeStackHandle) -> Option<ScopeStackHandle> {
        match snapshot_scope_stack(scope_stack) {
            Ok(scope_stack) => Some(scope_stack),
            Err(error) => {
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "subscriber_scope_snapshot_failed";
                    "Queued publication could not snapshot its emitting scope stack: {error}"
                );
                None
            }
        }
    }

    #[cfg(test)]
    pub(super) fn block_on_sanitizer_future<F: Future>(
        future: F,
    ) -> std::result::Result<F::Output, String> {
        let mut runtime = process_state()
            .sanitizer_runtime
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let runtime = runtime.get_or_insert_with(build_sanitizer_runtime);
        runtime
            .as_ref()
            .map(|runtime| runtime.block_on(future))
            .map_err(Clone::clone)
    }

    fn build_sanitizer_runtime() -> std::result::Result<tokio::runtime::Runtime, String> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| error.to_string())
    }

    fn start_background_publication_executor()
    -> std::result::Result<tokio::sync::mpsc::UnboundedSender<BackgroundPublication>, String> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| error.to_string())?;
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        std::thread::Builder::new()
            .name("nemo-relay-background-publication".into())
            .spawn(move || {
                runtime.block_on(async move {
                    while let Some(publication) = receiver.recv().await {
                        tokio::spawn(publication);
                    }
                });
            })
            .map_err(|error| error.to_string())?;
        Ok(sender)
    }

    pub(super) fn spawn_background_publication<F>(future: F) -> bool
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let state = process_state();
        let sender = {
            let mut executor = state
                .background_publications
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            executor
                .get_or_insert_with(start_background_publication_executor)
                .clone()
        };
        match sender {
            Ok(sender) if sender.send(Box::pin(future)).is_ok() => true,
            Ok(_) => {
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "background_publication_executor_stopped";
                    "Background publication executor stopped before accepting stream finalization"
                );
                false
            }
            Err(error)
                if !state
                    .background_publication_failure_logged
                    .swap(true, Ordering::AcqRel) =>
            {
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "background_publication_executor_failed";
                    "Background publication executor failed to start: {error}"
                );
                false
            }
            Err(_) => false,
        }
    }

    #[cfg(test)]
    pub(super) fn dispatch_event(event: &Event, subscribers: &[EventSubscriberFn]) -> bool {
        if subscribers.is_empty() {
            return true;
        }
        let Some(scope_stack) = immutable_scope_stack(&current_scope_stack()) else {
            return false;
        };
        let message = DispatcherMessage::Deliver {
            event: Box::new(event.clone()),
            transform: None,
            sanitizers: Vec::new(),
            subscribers: subscribers.to_vec(),
            scope_stack,
            publication_context: current_publication_context(),
        };
        send_dispatch_message(message)
    }

    pub(super) fn dispatch_sanitized_event(
        event: Event,
        sanitizers: Vec<Guardrail<EventSanitizeFn>>,
        subscribers: &[EventSubscriberFn],
        scope_stack: ScopeStackHandle,
    ) -> bool {
        if subscribers.is_empty() {
            return true;
        }
        let Some(scope_stack) = immutable_scope_stack(&scope_stack) else {
            return false;
        };
        let message = DispatcherMessage::Deliver {
            event: Box::new(event),
            transform: None,
            sanitizers,
            subscribers: subscribers.to_vec(),
            scope_stack,
            publication_context: current_publication_context(),
        };
        enqueue_dispatch_message(message)
    }

    pub(super) fn dispatch_reserved_sanitized_event(
        event: Event,
        sanitizers: Vec<Guardrail<EventSanitizeFn>>,
        subscribers: &[EventSubscriberFn],
        scope_stack: ScopeStackHandle,
    ) -> bool {
        if subscribers.is_empty() {
            return true;
        }
        let Some(scope_stack) = immutable_scope_stack(&scope_stack) else {
            return false;
        };
        let message = DispatcherMessage::Deliver {
            event: Box::new(event),
            transform: None,
            sanitizers,
            subscribers: subscribers.to_vec(),
            scope_stack,
            publication_context: current_publication_context(),
        };
        enqueue_dispatch_message(message)
    }

    pub(super) fn dispatch_transformed_event(
        event: Event,
        transform: EventTransformFn,
        sanitizers: Vec<Guardrail<EventSanitizeFn>>,
        subscribers: &[EventSubscriberFn],
        scope_stack: ScopeStackHandle,
    ) -> bool {
        let Some(scope_stack) = immutable_scope_stack(&scope_stack) else {
            return false;
        };
        let message = DispatcherMessage::Deliver {
            event: Box::new(event),
            transform: Some(transform),
            sanitizers,
            subscribers: subscribers.to_vec(),
            scope_stack,
            publication_context: current_publication_context(),
        };
        enqueue_dispatch_message(message)
    }

    /// Reserve a FIFO position for publications produced by an async task.
    /// A later flush waits for the task and drains its buffered publications
    /// at the reserved position before acknowledging the flush.
    pub(super) fn register_async_publication() -> Option<AsyncPublication> {
        let sender = dispatcher_sender().ok()?;
        let (publication_tx, publication_rx) = mpsc::channel();
        sender
            .send(DispatcherMessage::Barrier {
                publications: publication_rx,
            })
            .ok()
            .map(|_| AsyncPublication {
                sender: publication_tx,
            })
    }

    pub(super) fn flush_subscribers() -> Result<()> {
        if in_dispatcher_callback() {
            return Ok(());
        }
        let sender = {
            let dispatcher = process_state()
                .dispatcher
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let Some(sender_result) = dispatcher.as_ref() else {
                return Ok(());
            };
            sender_result
                .as_ref()
                .map_err(|error| FlowError::Internal(error.clone()))?
                .clone()
        };
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

    pub(super) fn in_dispatcher_callback() -> bool {
        IN_DISPATCHER.with(Cell::get) || ASYNC_PUBLICATION_MESSAGES.try_with(|_| ()).is_ok()
    }

    pub(super) async fn with_async_publication_context<F: Future>(
        publication: Option<AsyncPublication>,
        future: F,
    ) -> F::Output {
        if ASYNC_PUBLICATION_MESSAGES.try_with(|_| ()).is_ok() {
            future.await
        } else {
            let (output, publications) = ASYNC_PUBLICATION_MESSAGES
                .scope(
                    RefCell::new(publication.as_ref().map(|_| Vec::new())),
                    async {
                        let output = future.await;
                        let publications = ASYNC_PUBLICATION_MESSAGES
                            .with(|messages| messages.borrow_mut().take());
                        (output, publications)
                    },
                )
                .await;
            if let (Some(publication), Some(publications)) = (publication, publications) {
                let _ = publication.sender.send(publications);
            }
            output
        }
    }

    fn dispatcher_sender() -> std::result::Result<Sender<DispatcherMessage>, String> {
        let mut dispatcher = process_state()
            .dispatcher
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        dispatcher.get_or_insert_with(start_dispatcher).clone()
    }

    fn send_dispatch_message(message: DispatcherMessage) -> bool {
        match dispatcher_sender() {
            Ok(sender) if sender.send(message).is_ok() => true,
            Ok(_) => {
                log::warn!(
                    target: "nemo_relay.runtime",
                    event = "subscriber_event_dropped",
                    reason = "dispatcher_disconnected";
                    "Subscriber event was dropped because the dispatcher stopped"
                );
                false
            }
            Err(error)
                if !process_state()
                    .dispatcher_failure_logged
                    .swap(true, Ordering::AcqRel) =>
            {
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "subscriber_dispatcher_failed";
                    "Subscriber dispatcher failed to start: {error}"
                );
                false
            }
            Err(_) => false,
        }
    }

    fn enqueue_dispatch_message(message: DispatcherMessage) -> bool {
        let mut message = Some(message);
        let buffered = ASYNC_PUBLICATION_MESSAGES
            .try_with(|messages| {
                let mut messages = messages.borrow_mut();
                match messages.as_mut() {
                    Some(messages) => {
                        messages.push(message.take().expect("message is buffered once"));
                        true
                    }
                    None => false,
                }
            })
            .unwrap_or(false);
        buffered || send_dispatch_message(message.expect("unbuffered message remains available"))
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
                    let _ = done.send(());
                }
                DispatcherMessage::Barrier { publications } => {
                    if let Ok(publications) = publications.recv() {
                        for publication in publications {
                            handle_message(publication);
                        }
                    }
                }
                message => handle_message(message),
            }
        }
    }

    fn handle_message(message: DispatcherMessage) {
        match message {
            DispatcherMessage::Deliver {
                event,
                transform,
                sanitizers,
                subscribers,
                scope_stack,
                publication_context,
            } => deliver_event(
                event,
                transform,
                sanitizers,
                subscribers,
                scope_stack,
                publication_context,
            ),
            DispatcherMessage::Flush { done } => {
                let _ = done.send(());
            }
            DispatcherMessage::Barrier { publications } => {
                if let Ok(publications) = publications.recv() {
                    for publication in publications {
                        handle_message(publication);
                    }
                }
            }
        }
    }

    fn deliver_event(
        event: Box<Event>,
        transform: Option<EventTransformFn>,
        sanitizers: Vec<Guardrail<EventSanitizeFn>>,
        subscribers: Vec<EventSubscriberFn>,
        scope_stack: ScopeStackHandle,
        publication_context: Option<PublicationContext>,
    ) {
        let previous_scope_stack = capture_thread_scope_stack();
        set_thread_scope_stack(scope_stack);
        let _dispatch_guard = DispatchGuard::enter();
        let Some(event) =
            sanitize_event_snapshot(*event, transform, sanitizers, publication_context)
        else {
            restore_thread_scope_stack(previous_scope_stack);
            return;
        };
        for subscriber in subscribers {
            if catch_unwind(AssertUnwindSafe(|| subscriber(&event))).is_err() {
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "subscriber_callback_panicked";
                    "Event subscriber callback panicked"
                );
            }
        }
        restore_thread_scope_stack(previous_scope_stack);
    }

    /// Apply a transform and sanitizers on the dispatcher thread. A transform
    /// failure drops the event because it may be responsible for inserting the
    /// sanitized payload. A sanitizer failure retains the transformed snapshot
    /// and continues publication (fail open).
    fn sanitize_event_snapshot(
        event: Event,
        transform: Option<EventTransformFn>,
        sanitizers: Vec<Guardrail<EventSanitizeFn>>,
        publication_context: Option<PublicationContext>,
    ) -> Option<Event> {
        let state = process_state();
        let mut runtime = state
            .sanitizer_runtime
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let runtime = runtime.get_or_insert_with(build_sanitizer_runtime);
        let runtime = match runtime.as_ref() {
            Ok(runtime) => runtime,
            Err(error) => {
                if !state
                    .sanitizer_runtime_failure_logged
                    .swap(true, Ordering::AcqRel)
                {
                    log::error!(
                        target: "nemo_relay.runtime",
                        event = "event_sanitizer_runtime_failed";
                        "Event sanitizer runtime failed; dropping events: {error}"
                    );
                }
                return None;
            }
        };
        let transform_context = publication_context.clone();
        let transformed = match catch_unwind(AssertUnwindSafe(|| {
            runtime.block_on(
                TASK_PUBLICATION_CONTEXT.scope(transform_context, async move {
                    match transform {
                        Some(transform) => transform(event).await,
                        None => event,
                    }
                }),
            )
        })) {
            Ok(event) => event,
            Err(_) => {
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "event_transform_panicked";
                    "Event transform panicked; dropping the event"
                );
                return None;
            }
        };
        if sanitizers.is_empty() {
            return Some(transformed);
        }
        let fallback = transformed.clone();
        match catch_unwind(AssertUnwindSafe(|| {
            runtime.block_on(TASK_PUBLICATION_CONTEXT.scope(
                publication_context,
                NemoRelayContextState::event_sanitize_snapshot_chain(transformed, &sanitizers),
            ))
        })) {
            Ok(event) => Some(event),
            Err(_) => {
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "event_sanitizer_panicked";
                    "Event sanitizer panicked; preserving the last valid event snapshot"
                );
                Some(fallback)
            }
        }
    }

    pub(super) fn prepare_for_fork() {
        // Allocate the child's fresh state before fork. Do not lock active
        // dispatcher state here: a pending Python sanitizer may require the
        // forking event-loop thread to make progress.
        PREPARED_FORK_STATE.with(|prepared| {
            assert!(
                prepared.get().is_null(),
                "subscriber fork preparation is nested"
            );
            prepared.set(Box::into_raw(Box::new(ProcessState::new())));
        });
    }

    pub(super) fn resume_after_fork_parent() {
        PREPARED_FORK_STATE.with(|prepared| {
            let state = prepared.replace(std::ptr::null_mut());
            assert!(
                !state.is_null(),
                "subscriber fork parent hook ran without preparation"
            );
            unsafe { drop(Box::from_raw(state)) };
        });
    }

    pub(super) fn reset_after_fork_child() {
        PREPARED_FORK_STATE.with(|prepared| {
            let state = prepared.replace(std::ptr::null_mut());
            assert!(
                !state.is_null(),
                "subscriber fork child hook ran without preparation"
            );
            PROCESS_STATE.store(state, Ordering::Release);
        });
    }

    #[cfg(test)]
    mod tests {
        use super::*;

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
            let event = serde_json::from_value(serde_json::json!({
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
    }
}

#[cfg(test)]
pub(crate) fn block_on_sanitizer_future<F: Future>(
    future: F,
) -> std::result::Result<F::Output, String> {
    native::block_on_sanitizer_future(future)
}

/// Queue an event for subscriber delivery.
#[cfg(test)]
pub(crate) fn dispatch_event(event: &Event, subscribers: &[EventSubscriberFn]) -> bool {
    native::dispatch_event(event, subscribers)
}

/// Queue a snapshot for serial event sanitization followed by subscriber
/// delivery. Used by synchronous scope and mark APIs.
pub(crate) fn dispatch_sanitized_event(
    event: Event,
    sanitizers: Vec<Guardrail<EventSanitizeFn>>,
    subscribers: &[EventSubscriberFn],
    scope_stack: ScopeStackHandle,
) -> bool {
    native::dispatch_sanitized_event(event, sanitizers, subscribers, scope_stack)
}

/// Publish a stream-finalization event at its reserved FIFO position.
pub(crate) fn dispatch_reserved_sanitized_event(
    event: Event,
    sanitizers: Vec<Guardrail<EventSanitizeFn>>,
    subscribers: &[EventSubscriberFn],
    scope_stack: ScopeStackHandle,
) -> bool {
    native::dispatch_reserved_sanitized_event(event, sanitizers, subscribers, scope_stack)
}

/// Queue a snapshot for a middleware-specific asynchronous transformation,
/// followed by event sanitization and subscriber delivery.
pub(crate) fn dispatch_transformed_event(
    event: Event,
    transform: EventTransformFn,
    sanitizers: Vec<Guardrail<EventSanitizeFn>>,
    subscribers: &[EventSubscriberFn],
    scope_stack: ScopeStackHandle,
) -> bool {
    native::dispatch_transformed_event(event, transform, sanitizers, subscribers, scope_stack)
}

/// Register a FIFO barrier for async work that will queue a subscriber event.
///
/// Dropping the returned publication handle releases the barrier, so error
/// paths cannot leave the dispatcher blocked.
pub(crate) fn register_async_publication() -> Option<native::AsyncPublication> {
    native::register_async_publication()
}

/// Run asynchronous middleware as part of an already-registered publication,
/// buffering the finalization publications explicitly assigned to its reserved
/// FIFO position.
///
/// Re-entrant subscriber flushes are no-ops in this context because the
/// publication's FIFO barrier cannot complete until the middleware returns.
pub(crate) async fn with_async_publication_context<F: Future>(
    publication: Option<native::AsyncPublication>,
    future: F,
) -> F::Output {
    native::with_async_publication_context(publication, future).await
}

/// Schedule detached stream-finalization publication on the process-local
/// executor. The executor uses one shared OS thread and is reset after fork.
pub(crate) fn spawn_background_publication<F>(future: F) -> bool
where
    F: Future<Output = ()> + Send + 'static,
{
    native::spawn_background_publication(future)
}

/// Wait for all queued subscriber callbacks submitted before this call.
pub fn flush_subscribers() -> Result<()> {
    native::flush_subscribers()
}

/// Acquire process-local dispatcher resources before a Unix `fork`.
#[doc(hidden)]
pub fn prepare_for_fork() {
    native::prepare_for_fork();
}

/// Release process-local dispatcher resources in the parent after a Unix `fork`.
#[doc(hidden)]
pub fn resume_after_fork_parent() {
    native::resume_after_fork_parent();
}

/// Reset and release inherited dispatcher resources in the child after a Unix `fork`.
#[doc(hidden)]
pub fn reset_after_fork_child() {
    native::reset_after_fork_child();
}

/// Return whether the current callback was invoked by queued event publication.
///
/// Bindings use this to make re-entrant flush operations non-blocking while
/// the serial dispatcher is awaiting middleware on another language runtime.
#[doc(hidden)]
#[must_use]
pub fn in_dispatcher_callback() -> bool {
    native::in_dispatcher_callback()
}
