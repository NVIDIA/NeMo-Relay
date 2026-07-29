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

    pub(super) enum DispatcherMessage {
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

    /// Opaque routing handle for publications emitted on foreign callback threads.
    #[derive(Clone)]
    pub struct PublicationBuffer {
        messages: Arc<Mutex<Option<Vec<DispatcherMessage>>>>,
    }

    impl PublicationBuffer {
        fn new(messages: Option<Vec<DispatcherMessage>>) -> Self {
            Self {
                messages: Arc::new(Mutex::new(messages)),
            }
        }

        fn enabled() -> Self {
            Self::new(Some(Vec::new()))
        }

        fn push(&self, message: DispatcherMessage) -> std::result::Result<(), DispatcherMessage> {
            let mut messages = self
                .messages
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            match messages.as_mut() {
                Some(messages) => {
                    messages.push(message);
                    Ok(())
                }
                None => Err(message),
            }
        }

        fn take(&self) -> Vec<DispatcherMessage> {
            self.messages
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take()
                .unwrap_or_default()
        }

        fn is_active(&self) -> bool {
            self.messages
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_some()
        }
    }

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
        static THREAD_PUBLICATION_BUFFER: RefCell<Option<PublicationBuffer>> = const { RefCell::new(None) };
    }
    tokio::task_local! {
        static ASYNC_PUBLICATION_BUFFER: PublicationBuffer;
    }

    struct DispatchGuard;
    struct ThreadPublicationBufferGuard(Option<PublicationBuffer>);

    pub(crate) struct AsyncPublication {
        pub(super) sender: Sender<Vec<DispatcherMessage>>,
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

    impl Drop for ThreadPublicationBufferGuard {
        fn drop(&mut self) {
            THREAD_PUBLICATION_BUFFER.with(|current| {
                current.replace(self.0.take());
            });
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
        let (publication_tx, publication_rx) = mpsc::channel();
        enqueue_dispatch_message(DispatcherMessage::Barrier {
            publications: publication_rx,
        })
        .then_some(AsyncPublication {
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
        IN_DISPATCHER.with(Cell::get)
            || ASYNC_PUBLICATION_BUFFER.try_with(|_| ()).is_ok()
            || THREAD_PUBLICATION_BUFFER.with(|buffer| {
                buffer
                    .borrow()
                    .as_ref()
                    .is_some_and(PublicationBuffer::is_active)
            })
    }

    pub(super) fn capture_nested_publication_buffer() -> Option<PublicationBuffer> {
        ASYNC_PUBLICATION_BUFFER
            .try_with(Clone::clone)
            .ok()
            .filter(PublicationBuffer::is_active)
            .or_else(|| {
                THREAD_PUBLICATION_BUFFER
                    .with(|buffer| buffer.borrow().clone())
                    .filter(PublicationBuffer::is_active)
            })
    }

    pub(super) fn with_nested_publication_buffer<T>(
        buffer: Option<PublicationBuffer>,
        f: impl FnOnce() -> T,
    ) -> T {
        let previous = THREAD_PUBLICATION_BUFFER.with(|current| current.replace(buffer));
        let _guard = ThreadPublicationBufferGuard(previous);
        f()
    }

    pub(super) fn sync_thread_publication_buffer(buffer: Option<PublicationBuffer>) {
        THREAD_PUBLICATION_BUFFER.with(|current| {
            current.replace(buffer);
        });
    }

    pub(super) async fn with_task_nested_publication_buffer<F: Future>(
        buffer: Option<PublicationBuffer>,
        future: F,
    ) -> F::Output {
        match buffer {
            Some(buffer) => ASYNC_PUBLICATION_BUFFER.scope(buffer, future).await,
            None => future.await,
        }
    }

    pub(super) async fn with_async_publication_context<F: Future>(
        publication: Option<AsyncPublication>,
        future: F,
    ) -> F::Output {
        if ASYNC_PUBLICATION_BUFFER.try_with(|_| ()).is_ok() {
            future.await
        } else {
            let buffer = PublicationBuffer::new(publication.as_ref().map(|_| Vec::new()));
            let output = ASYNC_PUBLICATION_BUFFER.scope(buffer.clone(), future).await;
            if let Some(publication) = publication {
                let _ = publication.sender.send(buffer.take());
            }
            output
        }
    }

    pub(super) fn dispatcher_sender() -> std::result::Result<Sender<DispatcherMessage>, String> {
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

    pub(super) fn enqueue_dispatch_message(message: DispatcherMessage) -> bool {
        let message = if let Ok(buffer) = ASYNC_PUBLICATION_BUFFER.try_with(Clone::clone) {
            match buffer.push(message) {
                Ok(()) => return true,
                Err(message) => message,
            }
        } else {
            message
        };
        let message = if let Some(buffer) =
            THREAD_PUBLICATION_BUFFER.with(|buffer| buffer.borrow().clone())
        {
            match buffer.push(message) {
                Ok(()) => return true,
                Err(message) => message,
            }
        } else {
            message
        };
        send_dispatch_message(message)
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
        let (event, nested_publications) =
            sanitize_event_snapshot(*event, transform, sanitizers, publication_context);
        if let Some(event) = event {
            for subscriber in subscribers {
                if catch_unwind(AssertUnwindSafe(|| subscriber(&event))).is_err() {
                    log::error!(
                        target: "nemo_relay.runtime",
                        event = "subscriber_callback_panicked";
                        "Event subscriber callback panicked"
                    );
                }
            }
        }
        restore_thread_scope_stack(previous_scope_stack);
        // Publications emitted while transforming or sanitizing this event
        // are causally nested within it. Drain them before the dispatcher
        // consumes messages that callers may already have queued afterward.
        for publication in nested_publications {
            handle_message(publication);
        }
    }

    fn run_with_nested_publication_buffer<F: Future>(
        runtime: &tokio::runtime::Runtime,
        publication_context: Option<PublicationContext>,
        future: F,
    ) -> (std::thread::Result<F::Output>, Vec<DispatcherMessage>) {
        let buffer = PublicationBuffer::enabled();
        let output = catch_unwind(AssertUnwindSafe(|| {
            runtime.block_on(ASYNC_PUBLICATION_BUFFER.scope(
                buffer.clone(),
                TASK_PUBLICATION_CONTEXT.scope(publication_context, future),
            ))
        }));
        (output, buffer.take())
    }

    /// Apply a transform and sanitizers on the dispatcher thread. A transform
    /// failure drops the event because it may be responsible for inserting the
    /// sanitized payload. A sanitizer failure retains the transformed snapshot
    /// and continues publication (fail open).
    pub(super) fn sanitize_event_snapshot(
        event: Event,
        transform: Option<EventTransformFn>,
        sanitizers: Vec<Guardrail<EventSanitizeFn>>,
        publication_context: Option<PublicationContext>,
    ) -> (Option<Event>, Vec<DispatcherMessage>) {
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
                        "Event sanitizer runtime failed: {error}"
                    );
                }
                if transform.is_some() {
                    log::error!(
                        target: "nemo_relay.runtime",
                        event = "event_transform_runtime_unavailable";
                        "Dropping an event because its required asynchronous transform could not run"
                    );
                    return (None, Vec::new());
                }
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "event_sanitizer_fail_open";
                    "Publishing the original event snapshot because event sanitizers could not run"
                );
                return (Some(event), Vec::new());
            }
        };
        let transform_context = publication_context.clone();
        let (transformed, mut nested_publications) =
            run_with_nested_publication_buffer(runtime, transform_context, async move {
                match transform {
                    Some(transform) => transform(event).await,
                    None => event,
                }
            });
        let transformed = match transformed {
            Ok(event) => event,
            Err(_) => {
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "event_transform_panicked";
                    "Event transform panicked; dropping the event"
                );
                return (None, nested_publications);
            }
        };
        if sanitizers.is_empty() {
            return (Some(transformed), nested_publications);
        }
        let fallback = transformed.clone();
        let (sanitized, sanitizer_publications) = run_with_nested_publication_buffer(
            runtime,
            publication_context,
            NemoRelayContextState::event_sanitize_snapshot_chain(transformed, &sanitizers),
        );
        nested_publications.extend(sanitizer_publications);
        let event = match sanitized {
            Ok(event) => Some(event),
            Err(_) => {
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "event_sanitizer_panicked";
                    "Event sanitizer panicked; preserving the last valid event snapshot"
                );
                Some(fallback)
            }
        };
        (event, nested_publications)
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
    pub(super) fn set_sanitizer_runtime_failure_for_test(error: Option<&str>) {
        let state = process_state();
        let mut runtime = state
            .sanitizer_runtime
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *runtime = error.map(|error| Err(error.to_string()));
        state
            .sanitizer_runtime_failure_logged
            .store(false, Ordering::Release);
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/subscriber_dispatcher_tests.rs"]
mod tests;

#[doc(hidden)]
pub use native::PublicationBuffer;

/// Capture the active nested-publication buffer for a foreign callback thread.
#[doc(hidden)]
pub fn capture_nested_publication_buffer() -> Option<PublicationBuffer> {
    native::capture_nested_publication_buffer()
}

/// Route synchronous publications on a foreign callback thread into the
/// dispatcher invocation that scheduled the callback.
#[doc(hidden)]
pub fn with_nested_publication_buffer<T>(
    buffer: Option<PublicationBuffer>,
    f: impl FnOnce() -> T,
) -> T {
    native::with_nested_publication_buffer(buffer, f)
}

/// Route publications from a foreign async callback task into the dispatcher
/// invocation that scheduled the callback.
#[doc(hidden)]
pub async fn with_task_nested_publication_buffer<F: Future>(
    buffer: Option<PublicationBuffer>,
    future: F,
) -> F::Output {
    native::with_task_nested_publication_buffer(buffer, future).await
}

/// Synchronize a foreign runtime's current callback publication buffer into
/// Relay's thread-local fallback.
#[doc(hidden)]
pub fn sync_thread_publication_buffer(buffer: Option<PublicationBuffer>) {
    native::sync_thread_publication_buffer(buffer);
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
