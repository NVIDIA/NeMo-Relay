// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Asynchronous subscriber delivery for native targets.

use crate::api::event::{Event, EventSanitizeFields};
use crate::api::registry::{EventMetadataInjector, Guardrail};
use crate::api::runtime::{
    EventSanitizeFn, EventSubscriberFn, NemoRelayContextState, ScopeStackHandle,
};
use crate::api::shared::snapshot_event_metadata_injectors;
use crate::error::{FlowError, Result};
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

/// Completion receipt for one queued subscriber delivery.
///
/// Unlike [`flush_subscribers`], this receipt waits only for sanitizer and
/// subscriber processing of the event that created it. Events queued later are
/// not part of the wait.
#[doc(hidden)]
pub struct SubscriberDelivery {
    completion: tokio::sync::oneshot::Receiver<()>,
}

impl SubscriberDelivery {
    fn completed() -> Self {
        let (completion_tx, completion) = tokio::sync::oneshot::channel();
        let _ = completion_tx.send(());
        Self { completion }
    }

    /// Wait until this event's subscriber delivery is complete.
    ///
    /// Do not call this from a subscriber, event-sanitizer, guardrail, or
    /// intercept callback. The dispatcher signals completion on its own
    /// thread, so waiting there creates a wait cycle.
    pub async fn wait(self) -> Result<()> {
        self.completion.await.map_err(|error| {
            FlowError::Internal(format!(
                "subscriber delivery completion channel closed: {error}"
            ))
        })
    }
}

mod native {
    use std::cell::{Cell, RefCell};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::sync::{Arc, Weak};

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
            injectors: Vec<EventMetadataInjector>,
            sanitizers: Vec<Guardrail<EventSanitizeFn>>,
            subscribers: Vec<EventSubscriberFn>,
            scope_stack: ScopeStackHandle,
            publication_context: Option<PublicationContext>,
            lineage: Option<PublicationPermit>,
            completion: Option<tokio::sync::oneshot::Sender<()>>,
        },
        Flush {
            done: Sender<()>,
            include_pending: bool,
        },
        RegisterPending {
            lineage: Arc<PublicationLineage>,
        },
        CompletePending {
            permit: PublicationPermit,
        },
        Barrier {
            publications: Receiver<Vec<DispatcherMessage>>,
            lineage: Option<PublicationPermit>,
        },
    }

    #[derive(Default)]
    pub(super) struct PublicationLineage {
        outstanding: AtomicUsize,
        pending_terminal: bool,
    }

    pub(super) struct PublicationPermit(Arc<PublicationLineage>);

    impl PublicationPermit {
        pub(super) fn new(lineage: Arc<PublicationLineage>) -> Self {
            lineage.outstanding.fetch_add(1, Ordering::AcqRel);
            Self(lineage)
        }

        fn lineage(&self) -> Arc<PublicationLineage> {
            Arc::clone(&self.0)
        }
    }

    impl Drop for PublicationPermit {
        fn drop(&mut self) {
            self.0.outstanding.fetch_sub(1, Ordering::AcqRel);
        }
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
        lineage: Option<Arc<PublicationLineage>>,
    }

    impl PublicationBuffer {
        fn new(messages: Option<Vec<DispatcherMessage>>) -> Self {
            Self {
                messages: Arc::new(Mutex::new(messages)),
                lineage: current_publication_lineage(),
            }
        }

        fn enabled() -> Self {
            Self::new(Some(Vec::new()))
        }

        fn push(
            &self,
            message: DispatcherMessage,
        ) -> std::result::Result<(), Box<DispatcherMessage>> {
            let mut messages = self
                .messages
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            match messages.as_mut() {
                Some(messages) => {
                    messages.push(message);
                    Ok(())
                }
                None => Err(Box::new(message)),
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
        static THREAD_PUBLICATION_LINEAGE: RefCell<Option<Arc<PublicationLineage>>> = const { RefCell::new(None) };
    }
    tokio::task_local! {
        static ASYNC_PUBLICATION_BUFFER: PublicationBuffer;
    }

    struct DispatchGuard;
    struct ThreadPublicationBufferGuard(Option<PublicationBuffer>);
    struct ThreadPublicationLineageGuard(Option<Arc<PublicationLineage>>);

    pub(crate) struct AsyncPublication {
        pub(super) sender: Sender<Vec<DispatcherMessage>>,
    }

    pub(crate) struct PendingPublication {
        sender: Sender<DispatcherMessage>,
        permit: Option<PublicationPermit>,
    }

    impl Drop for PendingPublication {
        fn drop(&mut self) {
            let Some(permit) = self.permit.take() else {
                return;
            };
            let _ = self
                .sender
                .send(DispatcherMessage::CompletePending { permit });
        }
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

    impl Drop for ThreadPublicationLineageGuard {
        fn drop(&mut self) {
            THREAD_PUBLICATION_LINEAGE.with(|current| {
                current.replace(self.0.take());
            });
        }
    }

    fn current_publication_lineage() -> Option<Arc<PublicationLineage>> {
        ASYNC_PUBLICATION_BUFFER
            .try_with(|buffer| buffer.lineage.clone())
            .ok()
            .flatten()
            .or_else(|| {
                THREAD_PUBLICATION_BUFFER.with(|buffer| {
                    buffer
                        .borrow()
                        .as_ref()
                        .and_then(|buffer| buffer.lineage.clone())
                })
            })
            .or_else(|| THREAD_PUBLICATION_LINEAGE.with(|lineage| lineage.borrow().clone()))
    }

    fn with_publication_lineage<T>(lineage: Arc<PublicationLineage>, f: impl FnOnce() -> T) -> T {
        let previous = THREAD_PUBLICATION_LINEAGE.with(|current| current.replace(Some(lineage)));
        let _guard = ThreadPublicationLineageGuard(previous);
        f()
    }

    fn attach_publication_lineage(
        message: &mut DispatcherMessage,
        inherited: Option<&Arc<PublicationLineage>>,
    ) -> Arc<PublicationLineage> {
        let current = inherited
            .cloned()
            .or_else(current_publication_lineage)
            .unwrap_or_default();
        match message {
            DispatcherMessage::Deliver { lineage, .. }
            | DispatcherMessage::Barrier { lineage, .. } => {
                if let Some(permit) = lineage {
                    permit.lineage()
                } else {
                    *lineage = Some(PublicationPermit::new(Arc::clone(&current)));
                    current
                }
            }
            DispatcherMessage::Flush { .. }
            | DispatcherMessage::RegisterPending { .. }
            | DispatcherMessage::CompletePending { .. } => current,
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

    fn build_sanitizer_invocation_runtime() -> std::result::Result<tokio::runtime::Runtime, String>
    {
        let runtime = process_state()
            .sanitizer_runtime
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(Err(error)) = runtime.as_ref() {
            return Err(error.clone());
        }
        drop(runtime);
        build_sanitizer_runtime()
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
        let mut event = event.clone();
        event.set_propagation_root_uuid(
            scope_stack
                .read()
                .ok()
                .and_then(|stack| stack.event_propagation_root_uuid()),
        );
        let message = DispatcherMessage::Deliver {
            event: Box::new(event),
            transform: None,
            injectors: snapshot_event_metadata_injectors(&scope_stack),
            sanitizers: Vec::new(),
            subscribers: subscribers.to_vec(),
            scope_stack,
            publication_context: current_publication_context(),
            lineage: None,
            completion: None,
        };
        send_dispatch_message(message)
    }

    pub(super) fn dispatch_sanitized_event(
        mut event: Event,
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
        event.set_propagation_root_uuid(
            scope_stack
                .read()
                .ok()
                .and_then(|stack| stack.event_propagation_root_uuid()),
        );
        let injectors = snapshot_event_metadata_injectors(&scope_stack);
        let message = DispatcherMessage::Deliver {
            event: Box::new(event),
            transform: None,
            injectors,
            sanitizers,
            subscribers: subscribers.to_vec(),
            scope_stack,
            publication_context: current_publication_context(),
            lineage: None,
            completion: None,
        };
        enqueue_dispatch_message(message)
    }

    pub(super) fn dispatch_sanitized_event_with_delivery(
        mut event: Event,
        sanitizers: Vec<Guardrail<EventSanitizeFn>>,
        subscribers: &[EventSubscriberFn],
        scope_stack: ScopeStackHandle,
    ) -> Result<SubscriberDelivery> {
        if subscribers.is_empty() {
            return Ok(SubscriberDelivery::completed());
        }
        let Some(scope_stack) = immutable_scope_stack(&scope_stack) else {
            return Err(FlowError::Internal(
                "failed to snapshot scope stack for subscriber delivery".into(),
            ));
        };
        event.set_propagation_root_uuid(
            scope_stack
                .read()
                .ok()
                .and_then(|stack| stack.event_propagation_root_uuid()),
        );
        let injectors = snapshot_event_metadata_injectors(&scope_stack);
        let (completion_tx, completion) = tokio::sync::oneshot::channel();
        let message = DispatcherMessage::Deliver {
            event: Box::new(event),
            transform: None,
            injectors,
            sanitizers,
            subscribers: subscribers.to_vec(),
            scope_stack,
            publication_context: current_publication_context(),
            lineage: None,
            completion: Some(completion_tx),
        };
        if !enqueue_dispatch_message(message) {
            return Err(FlowError::Internal(
                "failed to queue tracked subscriber delivery".into(),
            ));
        }
        Ok(SubscriberDelivery { completion })
    }

    pub(super) fn dispatch_reserved_sanitized_event(
        mut event: Event,
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
        event.set_propagation_root_uuid(
            scope_stack
                .read()
                .ok()
                .and_then(|stack| stack.event_propagation_root_uuid()),
        );
        let injectors = snapshot_event_metadata_injectors(&scope_stack);
        let message = DispatcherMessage::Deliver {
            event: Box::new(event),
            transform: None,
            injectors,
            sanitizers,
            subscribers: subscribers.to_vec(),
            scope_stack,
            publication_context: current_publication_context(),
            lineage: None,
            completion: None,
        };
        enqueue_dispatch_message(message)
    }

    pub(super) fn dispatch_transformed_event(
        mut event: Event,
        transform: EventTransformFn,
        sanitizers: Vec<Guardrail<EventSanitizeFn>>,
        subscribers: &[EventSubscriberFn],
        scope_stack: ScopeStackHandle,
    ) -> bool {
        let Some(scope_stack) = immutable_scope_stack(&scope_stack) else {
            return false;
        };
        event.set_propagation_root_uuid(
            scope_stack
                .read()
                .ok()
                .and_then(|stack| stack.event_propagation_root_uuid()),
        );
        let injectors = snapshot_event_metadata_injectors(&scope_stack);
        let message = DispatcherMessage::Deliver {
            event: Box::new(event),
            transform: Some(transform),
            injectors,
            sanitizers,
            subscribers: subscribers.to_vec(),
            scope_stack,
            publication_context: current_publication_context(),
            lineage: None,
            completion: None,
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
            lineage: None,
        })
        .then_some(AsyncPublication {
            sender: publication_tx,
        })
    }

    /// Track asynchronous work without blocking unrelated dispatcher delivery.
    ///
    /// A flush observed after this registration waits until the returned handle
    /// is dropped. Dropping the handle queues completion behind any terminal
    /// publications sent by that work.
    pub(super) fn register_pending_publication() -> Option<PendingPublication> {
        let sender = dispatcher_sender().ok()?;
        let lineage = Arc::new(PublicationLineage {
            outstanding: AtomicUsize::new(0),
            pending_terminal: true,
        });
        let permit = PublicationPermit::new(Arc::clone(&lineage));
        sender
            .send(DispatcherMessage::RegisterPending { lineage })
            .ok()?;
        Some(PendingPublication {
            sender,
            permit: Some(permit),
        })
    }

    fn flush_subscribers_inner(include_pending: bool, started: Option<Sender<()>>) -> Result<()> {
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
            .send(DispatcherMessage::Flush {
                done: done_tx,
                include_pending,
            })
            .map_err(|error| {
                FlowError::Internal(format!("failed to queue subscriber flush: {error}"))
            })?;
        if let Some(started) = started {
            let _ = started.send(());
        }
        done_rx
            .recv()
            .map_err(|error| FlowError::Internal(format!("subscriber flush failed: {error}")))?;
        Ok(())
    }

    pub(super) fn flush_subscribers() -> Result<()> {
        flush_subscribers_inner(true, None)
    }

    pub(super) fn flush_subscribers_with_started_signal(started: Sender<()>) -> Result<()> {
        flush_subscribers_inner(true, Some(started))
    }

    pub(super) fn flush_queued_subscribers() -> Result<()> {
        flush_subscribers_inner(false, None)
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

    fn send_dispatch_message(mut message: DispatcherMessage) -> bool {
        attach_publication_lineage(&mut message, None);
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

    pub(super) fn enqueue_dispatch_message(mut message: DispatcherMessage) -> bool {
        attach_publication_lineage(&mut message, None);
        let message = if let Ok(buffer) = ASYNC_PUBLICATION_BUFFER.try_with(Clone::clone) {
            match buffer.push(message) {
                Ok(()) => return true,
                Err(message) => *message,
            }
        } else {
            message
        };
        let message = if let Some(buffer) =
            THREAD_PUBLICATION_BUFFER.with(|buffer| buffer.borrow().clone())
        {
            match buffer.push(message) {
                Ok(()) => return true,
                Err(message) => *message,
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

    pub(super) struct PendingFlush {
        pub(super) done: Sender<()>,
        pub(super) lineages: Vec<Arc<PublicationLineage>>,
    }

    #[derive(Default)]
    pub(super) struct DispatcherLoopState {
        pub(super) active_lineages: Vec<Weak<PublicationLineage>>,
        pub(super) pending_flushes: Vec<PendingFlush>,
    }

    impl DispatcherLoopState {
        fn register(&mut self, lineage: &Arc<PublicationLineage>) {
            if !self.active_lineages.iter().any(|active| {
                active
                    .upgrade()
                    .is_some_and(|active| Arc::ptr_eq(&active, lineage))
            }) {
                self.active_lineages.push(Arc::downgrade(lineage));
            }
        }

        fn defer_or_complete_flush(&mut self, done: Sender<()>, include_pending: bool) {
            let lineages = self
                .active_lineages
                .iter()
                .filter_map(Weak::upgrade)
                .filter(|lineage| include_pending || !lineage.pending_terminal)
                .filter(|lineage| lineage.outstanding.load(Ordering::Acquire) > 0)
                .collect::<Vec<_>>();
            if lineages.is_empty() {
                let _ = done.send(());
            } else {
                self.pending_flushes.push(PendingFlush { done, lineages });
            }
        }

        pub(super) fn complete_ready_flushes(&mut self) {
            self.active_lineages.retain(|lineage| {
                lineage
                    .upgrade()
                    .is_some_and(|lineage| lineage.outstanding.load(Ordering::Acquire) > 0)
            });
            let ready = self
                .pending_flushes
                .iter()
                .take_while(|flush| {
                    flush
                        .lineages
                        .iter()
                        .all(|lineage| lineage.outstanding.load(Ordering::Acquire) == 0)
                })
                .count();
            for flush in self.pending_flushes.drain(..ready) {
                let _ = flush.done.send(());
            }
        }
    }

    fn run_dispatcher(rx: Receiver<DispatcherMessage>) {
        let mut state = DispatcherLoopState::default();
        while let Ok(message) = rx.recv() {
            handle_message(message, &mut state, None);
            state.complete_ready_flushes();
        }
    }

    fn handle_message(
        mut message: DispatcherMessage,
        state: &mut DispatcherLoopState,
        inherited: Option<&Arc<PublicationLineage>>,
    ) {
        let lineage = match message {
            DispatcherMessage::Flush {
                done,
                include_pending,
            } => {
                state.defer_or_complete_flush(done, include_pending);
                return;
            }
            DispatcherMessage::RegisterPending { lineage } => {
                state.register(&lineage);
                return;
            }
            DispatcherMessage::CompletePending { permit } => {
                state.register(&permit.lineage());
                drop(permit);
                return;
            }
            _ => attach_publication_lineage(&mut message, inherited),
        };
        state.register(&lineage);
        match message {
            DispatcherMessage::Deliver {
                event,
                transform,
                injectors,
                sanitizers,
                subscribers,
                scope_stack,
                publication_context,
                lineage: permit,
                completion,
            } => {
                let nested_publications = with_publication_lineage(Arc::clone(&lineage), || {
                    deliver_event(
                        event,
                        transform,
                        injectors,
                        sanitizers,
                        subscribers,
                        scope_stack,
                        publication_context,
                    )
                });
                drop(permit);
                for publication in nested_publications {
                    handle_message(publication, state, Some(&lineage));
                }
                if let Some(completion) = completion {
                    let _ = completion.send(());
                }
            }
            DispatcherMessage::Barrier {
                publications,
                lineage: permit,
            } => {
                if let Ok(publications) = publications.recv() {
                    for publication in publications {
                        handle_message(publication, state, Some(&lineage));
                    }
                }
                drop(permit);
            }
            DispatcherMessage::Flush { .. }
            | DispatcherMessage::RegisterPending { .. }
            | DispatcherMessage::CompletePending { .. } => unreachable!(),
        }
        state.complete_ready_flushes();
    }

    fn deliver_event(
        event: Box<Event>,
        transform: Option<EventTransformFn>,
        injectors: Vec<EventMetadataInjector>,
        sanitizers: Vec<Guardrail<EventSanitizeFn>>,
        subscribers: Vec<EventSubscriberFn>,
        scope_stack: ScopeStackHandle,
        publication_context: Option<PublicationContext>,
    ) -> Vec<DispatcherMessage> {
        let previous_scope_stack = capture_thread_scope_stack();
        set_thread_scope_stack(scope_stack);
        let _dispatch_guard = DispatchGuard::enter();
        let (event, nested_publications) = sanitize_event_snapshot(
            *event,
            transform,
            injectors,
            sanitizers,
            publication_context,
        );
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
        nested_publications
    }

    fn run_with_nested_publication_buffer<F: Future>(
        runtime: &tokio::runtime::Runtime,
        publication_context: Option<PublicationContext>,
        future: F,
    ) -> (std::thread::Result<F::Output>, Vec<DispatcherMessage>) {
        let buffer = PublicationBuffer::enabled();
        let output = catch_unwind(AssertUnwindSafe(|| {
            with_nested_publication_buffer(Some(buffer.clone()), || {
                with_publication_context(publication_context.clone(), || {
                    runtime.block_on(ASYNC_PUBLICATION_BUFFER.scope(
                        buffer.clone(),
                        TASK_PUBLICATION_CONTEXT.scope(publication_context, future),
                    ))
                })
            })
        }));
        (output, buffer.take())
    }

    fn inject_event_metadata_snapshot(
        event: Event,
        injectors: Vec<EventMetadataInjector>,
        publication_context: Option<PublicationContext>,
    ) -> (Event, Vec<DispatcherMessage>) {
        if injectors.is_empty() {
            return (event, Vec::new());
        }

        let state = process_state();
        let runtime = match build_sanitizer_invocation_runtime() {
            Ok(runtime) => runtime,
            Err(error) => {
                if !state
                    .sanitizer_runtime_failure_logged
                    .swap(true, Ordering::AcqRel)
                {
                    log::error!(
                        target: "nemo_relay.runtime",
                        event = "event_middleware_runtime_failed";
                        "Event middleware runtime failed: {error}"
                    );
                }
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "event_metadata_injector_runtime_failed";
                    "Event metadata injectors could not run; continuing without injection"
                );
                return (event, Vec::new());
            }
        };

        let fallback = event.clone();
        let (injected, nested_publications) = run_with_nested_publication_buffer(
            &runtime,
            publication_context,
            NemoRelayContextState::event_metadata_injection_snapshot_chain(event, &injectors),
        );
        runtime.shutdown_background();
        match injected {
            Ok(event) => (event, nested_publications),
            Err(_) => {
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "event_metadata_injector_chain_panicked";
                    "Event metadata injector chain panicked; continuing without injection"
                );
                (fallback, nested_publications)
            }
        }
    }

    /// Apply a transform, metadata injectors, and sanitizers on the dispatcher
    /// thread. A transform failure drops the event because it may be responsible
    /// for inserting the sanitized payload. Injector failures preserve the
    /// original Event. A sanitizer failure clears mutable observability fields.
    pub(super) fn sanitize_event_snapshot(
        event: Event,
        transform: Option<EventTransformFn>,
        injectors: Vec<EventMetadataInjector>,
        sanitizers: Vec<Guardrail<EventSanitizeFn>>,
        publication_context: Option<PublicationContext>,
    ) -> (Option<Event>, Vec<DispatcherMessage>) {
        let state = process_state();
        let propagation_root_uuid = event.propagation_root_uuid();
        let (mut transformed, mut nested_publications) = match transform {
            Some(transform) => {
                let runtime = match build_sanitizer_invocation_runtime() {
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
                        log::error!(
                            target: "nemo_relay.runtime",
                            event = "event_transform_runtime_unavailable";
                            "Dropping an event because its required asynchronous transform could not run"
                        );
                        return (None, Vec::new());
                    }
                };
                let (transformed, nested_publications) = run_with_nested_publication_buffer(
                    &runtime,
                    publication_context.clone(),
                    async move { transform(event).await },
                );
                runtime.shutdown_background();
                match transformed {
                    Ok(event) => (event, nested_publications),
                    Err(_) => {
                        log::error!(
                            target: "nemo_relay.runtime",
                            event = "event_transform_panicked";
                            "Event transform panicked; dropping the event"
                        );
                        return (None, nested_publications);
                    }
                }
            }
            None => (event, Vec::new()),
        };
        transformed.set_propagation_root_uuid(propagation_root_uuid);
        let (injected, injector_publications) =
            inject_event_metadata_snapshot(transformed, injectors, publication_context.clone());
        nested_publications.extend(injector_publications);
        if sanitizers.is_empty() {
            return (Some(injected), nested_publications);
        }
        let runtime = match build_sanitizer_invocation_runtime() {
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
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "event_sanitizer_runtime_failed";
                    "Event sanitizers could not run; clearing observability fields before publication"
                );
                let mut cleared = injected;
                cleared.apply_sanitize_fields(EventSanitizeFields::default());
                return (Some(cleared), nested_publications);
            }
        };
        let fallback = injected.clone();
        let (sanitized, sanitizer_publications) = run_with_nested_publication_buffer(
            &runtime,
            publication_context,
            NemoRelayContextState::event_sanitize_snapshot_chain(injected, &sanitizers),
        );
        runtime.shutdown_background();
        nested_publications.extend(sanitizer_publications);
        let event = match sanitized {
            Ok(event) => Some(event),
            Err(_) => {
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "event_sanitizer_panicked";
                    "Event sanitizer panicked; clearing observability fields"
                );
                let mut cleared = fallback;
                cleared.apply_sanitize_fields(EventSanitizeFields::default());
                Some(cleared)
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

pub(crate) fn dispatch_sanitized_event_with_delivery(
    event: Event,
    sanitizers: Vec<Guardrail<EventSanitizeFn>>,
    subscribers: &[EventSubscriberFn],
    scope_stack: ScopeStackHandle,
) -> Result<SubscriberDelivery> {
    native::dispatch_sanitized_event_with_delivery(event, sanitizers, subscribers, scope_stack)
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

pub(crate) use native::PendingPublication;

/// Register pending asynchronous work that must finish before a later
/// subscriber flush can complete.
///
/// Unlike an async publication barrier, this does not block unrelated
/// dispatcher delivery. Dropping the returned handle releases pending flushes
/// after previously queued terminal publications have been delivered.
pub(crate) fn register_pending_publication() -> Option<native::PendingPublication> {
    native::register_pending_publication()
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

/// Wait for all queued subscriber callbacks and managed terminal publications
/// registered before this call, including publications emitted transitively by
/// those callbacks.
pub fn flush_subscribers() -> Result<()> {
    native::flush_subscribers()
}

/// Wait for subscriber completion and signal once the flush request has been
/// queued, immediately before waiting for the dispatcher to acknowledge it.
#[doc(hidden)]
pub fn flush_subscribers_with_started_signal(started: std::sync::mpsc::Sender<()>) -> Result<()> {
    native::flush_subscribers_with_started_signal(started)
}

/// Wait only for subscriber publications already queued on the dispatcher.
///
/// Plugin teardown uses this legacy barrier so it can detach registries while
/// managed callbacks from an earlier snapshot remain in flight.
pub(crate) fn flush_queued_subscribers() -> Result<()> {
    native::flush_queued_subscribers()
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
