// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming LLM response support for the Node.js NAPI bindings.
//!
//! Provides the `LlmStream` type, an async iterator that yields response chunks
//! from a streaming LLM call. A core stream is polled on the JavaScript
//! thread, and a thread without enough native stack falls back to a Tokio
//! forwarding task.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, TryLockError, Weak};
use std::task::{Context, Poll, Wake, Waker};

use napi::Env;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use nemo_relay::api::runtime::LlmJsonStream;
use nemo_relay::api::runtime::subscriber_dispatcher::PublicationBuffer;
use nemo_relay::error::{FlowError, Result as FlowResult};
use serde_json::Value as Json;
use tokio_stream::{Stream, StreamExt};

use crate::callable::{DirectStreamCallbacks, FinalizerError};
use crate::callback_factory;
use crate::convert::{JsArg, closure_function, detach, js_value};
use crate::environment::{
    Liveness, PendingPromise, Rejection, Settler, create_promise, spawn_promise,
};
use crate::error::{Error, Result};
use crate::promise_call::{
    StreamCancellation, with_publication_callback_context, with_publication_callback_context_sync,
};

type CloseFuture = Pin<Box<dyn Future<Output = FlowResult<()>> + Send>>;
type CloseResult = Arc<Mutex<Option<std::result::Result<(), CloseFailure>>>>;

struct CloseErrors {
    first: Arc<Error>,
    rest: Vec<(&'static str, Arc<Error>)>,
}

#[derive(Clone)]
struct CloseFailure(Arc<CloseErrors>);

impl CloseFailure {
    fn into_js(self, env: &Env) -> napi::Result<JsArg> {
        let CloseErrors { first, rest } = &*self.0;
        if rest.is_empty() {
            return js_value(env, first.to_napi());
        }
        let message = self.to_string();
        let errors = std::iter::once(first)
            .chain(rest.iter().map(|(_, error)| error))
            .map(|error| error.to_napi())
            .collect();
        callback_factory::aggregate_error(env, errors, message)
    }

    fn throw_into(self, env: &Env) -> napi::Error {
        if self.0.rest.is_empty() {
            return crate::error::throw(env, self.0.first.to_napi());
        }
        let message = self.to_string();
        match self.into_js(env).and_then(|value| env.throw(value)) {
            Ok(()) => napi::Error::new(Status::PendingException, message),
            Err(failure) => failure,
        }
    }
}

impl std::fmt::Display for CloseFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.first)?;
        for (label, error) in &self.0.rest {
            write!(f, "; {label}: {error}")?;
        }
        Ok(())
    }
}

impl From<Error> for CloseFailure {
    fn from(error: Error) -> Self {
        Self(Arc::new(CloseErrors {
            first: Arc::new(error),
            rest: Vec::new(),
        }))
    }
}

impl Rejection for CloseFailure {
    fn settler(self) -> Settler {
        Box::new(move |env| self.into_js(env))
    }
}

fn settle_with(pending: PendingPromise, value: Json) {
    pending.resolve_with(Box::new(move |env| js_value(env, value)));
}

enum WakeState {
    Idle,
    Armed(PendingPromise),
    Fired,
}

/// Wakes a pending poll by resolving the promise it awaits, from the thread that fires it.
///
/// `null` tells a poll loop to poll again. An object is the outcome of a
/// stream released while a poll waited.
struct WakeSlot {
    state: Mutex<WakeState>,
}

impl WakeSlot {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(WakeState::Idle),
        })
    }

    fn fire(&self) {
        self.settle(|pending| settle_with(pending, Json::Null));
    }

    fn release(&self, outcome: std::result::Result<(), CloseFailure>) {
        self.settle(|pending| match outcome {
            Ok(()) => settle_with(pending, Json::Bool(true)),
            Err(failure) => pending.reject_with(failure.settler()),
        });
    }

    fn settle(&self, settle: impl FnOnce(PendingPromise)) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let WakeState::Armed(pending) = std::mem::replace(&mut *state, WakeState::Fired) {
            settle(pending);
        }
    }

    // A slot is armed after its poll. Resolve at once if a wake already happened.
    fn arm(&self, pending: PendingPromise) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if matches!(*state, WakeState::Fired) {
            settle_with(pending, Json::Null);
        } else {
            *state = WakeState::Armed(pending);
        }
    }
}

impl Wake for WakeSlot {
    fn wake(self: Arc<Self>) {
        self.fire();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.fire();
    }
}

struct Forwarded {
    receiver: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<FlowResult<Json>>>,
    requests: tokio::sync::mpsc::Sender<()>,
    producers: StreamCancellation,
    cancel: tokio::sync::watch::Sender<bool>,
    closed: tokio::sync::watch::Receiver<Option<std::result::Result<(), CloseFailure>>>,
    close_result: tokio::sync::Mutex<Option<std::result::Result<(), CloseFailure>>>,
}

impl Forwarded {
    fn spawn(
        stream: LlmJsonStream,
        context_id: Option<String>,
        publication_buffer: Option<PublicationBuffer>,
        producers: StreamCancellation,
    ) -> Arc<Self> {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let (requests, request_rx) = tokio::sync::mpsc::channel(1);
        let (cancel, cancel_rx) = tokio::sync::watch::channel(false);
        let (closed, closed_rx) = tokio::sync::watch::channel(None);
        within_runtime_if_available(|| {
            tokio::spawn(with_publication_callback_context(
                context_id,
                publication_buffer,
                forward_stream_to_channel(stream, tx, request_rx, cancel_rx, closed),
            ))
        });
        Arc::new(Self {
            receiver: tokio::sync::Mutex::new(rx),
            requests,
            producers,
            cancel,
            closed: closed_rx,
            close_result: tokio::sync::Mutex::new(None),
        })
    }

    async fn next(&self) -> FlowResult<Option<Json>> {
        let mut receiver = self.receiver.lock().await;
        // A finished forwarder rejects a request. `recv` then reports a closed channel.
        let _ = self.requests.send(()).await;
        match receiver.recv().await {
            None => Ok(None),
            Some(Ok(value)) => Ok(Some(value)),
            Some(Err(error)) => Err(error),
        }
    }

    async fn close(
        &self,
        finalizer_error: &FinalizerError,
    ) -> std::result::Result<(), CloseFailure> {
        let mut close_result = self.close_result.lock().await;
        if let Some(result) = close_result.as_ref() {
            return result.clone();
        }
        self.cancel.send_replace(true);
        self.producers.cancel();
        let result = self
            .closed
            .clone()
            .wait_for(Option::is_some)
            .await
            .map_err(|_| {
                Error::generic_failure("stream close task ended before releasing the producer")
            })?
            .clone()
            .expect("close state checked above");
        let mut receiver = self.receiver.lock().await;
        receiver.close();
        let mut undelivered_error = None;
        while let Ok(item) = receiver.try_recv() {
            if let Err(error) = item {
                undelivered_error.get_or_insert(error);
            }
        }
        let finalizer_error = finalizer_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        let result = combine_close_errors([
            (
                "stream failed",
                undelivered_error.map(|error| CloseFailure::from(Error::from(error))),
            ),
            ("stream cleanup also failed", result.err()),
            (
                "stream finalizer also failed",
                finalizer_error.map(CloseFailure::from),
            ),
        ]);
        *close_result = Some(result.clone());
        result
    }
}

async fn forward_stream_to_channel(
    mut stream: LlmJsonStream,
    tx: tokio::sync::mpsc::Sender<FlowResult<Json>>,
    mut requests: tokio::sync::mpsc::Receiver<()>,
    mut cancel: tokio::sync::watch::Receiver<bool>,
    closed: tokio::sync::watch::Sender<Option<std::result::Result<(), CloseFailure>>>,
) {
    let mut undelivered_error = None;
    // One pull per `next()` keeps a forwarded stream as lazy as a pumped one. Reserve before
    // pulling so a cancel does not consume a chunk or error that goes undelivered.
    loop {
        let requested = tokio::select! {
            biased;
            _ = cancel.changed() => break,
            request = requests.recv() => request.is_some(),
        };
        if !requested {
            break;
        }
        let permit = tokio::select! {
            biased;
            _ = cancel.changed() => break,
            permit = tx.reserve() => match permit {
                Ok(permit) => permit,
                Err(_) => break,
            },
        };
        let item = tokio::select! {
            biased;
            _ = cancel.changed() => break,
            item = stream.next() => item,
        };
        let Some(item) = item else {
            break;
        };
        // Cancellation can arrive while `stream.next()` is being polled, after
        // the biased select checked it. Keep a consumed error for close instead
        // of delivering it to the pending read after cancellation.
        if *cancel.borrow() {
            undelivered_error = item.err();
            break;
        }
        permit.send(item);
    }
    let result = finish_forwarded_close(undelivered_error, stream.close().await);
    closed.send_replace(Some(result));
}

fn finish_forwarded_close(
    undelivered_error: Option<FlowError>,
    cleanup: FlowResult<()>,
) -> std::result::Result<(), CloseFailure> {
    match undelivered_error {
        Some(error) => {
            if let Err(cleanup) = cleanup {
                log::warn!(
                    target: "nemo_relay.runtime",
                    event = "stream_cleanup_failed";
                    "LLM stream cleanup failed after the stream failed: {cleanup}"
                );
            }
            Err(CloseFailure::from(Error::from(error)))
        }
        None => cleanup.map_err(|error| CloseFailure::from(Error::from(error))),
    }
}

/// Close and drop a core stream on a worker so its callbacks are not released on the
/// JavaScript thread.
fn spawn_close(
    stream: LlmJsonStream,
    context_id: Option<String>,
    publication_buffer: Option<PublicationBuffer>,
) -> CloseFuture {
    let closed = tokio::spawn(with_publication_callback_context(
        context_id,
        publication_buffer,
        async move {
            let mut stream = stream;
            stream.close().await
        },
    ));
    Box::pin(async move {
        closed.await.unwrap_or_else(|error| {
            Err(FlowError::Internal(format!(
                "stream close task failed: {error}"
            )))
        })
    })
}

const REQUIRED_HEADROOM: usize = 512 * 1024;

static FORCE_FORWARDING: AtomicBool = AtomicBool::new(false);

pub(crate) fn set_force_forwarding(enabled: bool) {
    FORCE_FORWARDING.store(enabled, Ordering::Release);
}

/// Whether a caller lacks the native stack that polling on this thread needs.
///
/// `stacker` reads a thread's bounds as the standard library does and caches
/// them per thread. A thread whose bounds are unknown forwards, and so does a
/// musl main thread, which reports only the stack it has touched. Checked once
/// per stream at its first call. Later polls run from microtasks and stay
/// within that depth.
fn forwarding_required() -> bool {
    FORCE_FORWARDING.load(Ordering::Acquire)
        || stacker::remaining_stack().is_none_or(|remaining| remaining < REQUIRED_HEADROOM)
}

enum Mode {
    Unpolled(LlmJsonStream),
    Pump {
        stream: Option<LlmJsonStream>,
        closing: Option<CloseFuture>,
    },
    Forwarded(Arc<Forwarded>),
    Released,
}

struct StreamState {
    mode: Mode,
    releasing: bool,
    next_wake: Option<Arc<WakeSlot>>,
    close_wake: Option<Arc<WakeSlot>>,
    producers: StreamCancellation,
    context_id: Option<String>,
    publication_buffer: Option<PublicationBuffer>,
}

fn finish_close(
    outcome: FlowResult<()>,
    finalizer_error: &FinalizerError,
) -> std::result::Result<(), CloseFailure> {
    let finalizer_error = finalizer_error
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    combine_close_errors([
        (
            "stream cleanup also failed",
            outcome
                .err()
                .map(|error| CloseFailure::from(Error::from(error))),
        ),
        (
            "stream finalizer also failed",
            finalizer_error.map(CloseFailure::from),
        ),
    ])
}

impl StreamState {
    fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<FlowResult<Option<Json>>> {
        let Mode::Pump {
            stream: Some(stream),
            closing: None,
        } = &mut self.mode
        else {
            return Poll::Ready(Ok(None));
        };
        let ended = match Pin::new(stream).poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => return Poll::Ready(Ok(Some(chunk))),
            Poll::Ready(Some(Err(error))) => Err(error),
            Poll::Ready(None) => Ok(None),
            Poll::Pending => return Poll::Pending,
        };
        if let Mode::Pump { stream, closing } = &mut self.mode
            && let Some(core) = stream.take()
        {
            *closing = Some(spawn_close(
                core,
                self.context_id.clone(),
                self.publication_buffer.clone(),
            ));
        }
        Poll::Ready(ended)
    }

    fn poll_close(
        &mut self,
        cx: &mut Context<'_>,
        finalizer_error: &FinalizerError,
        close_result: &CloseResult,
    ) -> Poll<std::result::Result<(), CloseFailure>> {
        if let Some(result) = close_result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
        {
            return Poll::Ready(result);
        }
        // A release task owns the close now. Stay pending until it settles this poll
        // with its outcome.
        if self.releasing {
            return Poll::Pending;
        }
        let Mode::Pump { stream, closing } = &mut self.mode else {
            return Poll::Ready(Ok(()));
        };
        if closing.is_none() {
            let Some(taken) = stream.take() else {
                return Poll::Ready(Ok(()));
            };
            self.producers.cancel();
            // Wake a parked `next()` so it re-polls and observes the close.
            if let Some(wake) = self.next_wake.take() {
                wake.fire();
            }
            *closing = Some(spawn_close(
                taken,
                self.context_id.clone(),
                self.publication_buffer.clone(),
            ));
        }
        let Some(future) = closing.as_mut() else {
            return Poll::Ready(Ok(()));
        };
        match future.as_mut().poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(outcome) => {
                *closing = None;
                let result = finish_close(outcome, finalizer_error);
                *close_result
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(result.clone());
                Poll::Ready(result)
            }
        }
    }
}

enum Path {
    Pump,
    Forwarded(Arc<Forwarded>),
}

/// An async iterator over chunks from a streaming LLM response.
///
/// Obtained from `llmStreamCallExecute()`. Call `next()` repeatedly to consume
/// response chunks. Returns `null` when the stream is fully consumed.
#[napi]
pub struct LlmStream {
    state: Weak<Mutex<StreamState>>,
    poll: OnceLock<FunctionRef<JsArg, JsArg>>,
    released: tokio::sync::watch::Sender<bool>,
    finalizer_error: FinalizerError,
    close_result: CloseResult,
    context_id: Option<String>,
    publication_buffer: Option<PublicationBuffer>,
}

impl LlmStream {
    pub(crate) fn new(
        stream: LlmJsonStream,
        finalizer_error: FinalizerError,
        producers: StreamCancellation,
        context_id: Option<String>,
        publication_buffer: Option<PublicationBuffer>,
    ) -> Self {
        let state = Arc::new(Mutex::new(StreamState {
            mode: Mode::Unpolled(stream),
            releasing: false,
            next_wake: None,
            close_wake: None,
            producers,
            context_id: context_id.clone(),
            publication_buffer: publication_buffer.clone(),
        }));
        let close_context_id = context_id.clone();
        let close_publication_buffer = publication_buffer.clone();
        let close_result: CloseResult = Arc::default();
        let (released, mut release) = tokio::sync::watch::channel(false);
        let handle = Arc::downgrade(&state);
        let owner_close_result = Arc::clone(&close_result);
        let owner_finalizer_error = finalizer_error.clone();
        // This task owns the core stream so it closes on a worker rather than in a
        // handle finalizer on the JavaScript thread.
        tokio::spawn(async move {
            let _ = release.wait_for(|released| *released).await;
            let (mode, producers) = {
                let mut state = state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                state.releasing = true;
                (
                    std::mem::replace(&mut state.mode, Mode::Released),
                    state.producers.clone(),
                )
            };
            producers.cancel();
            // The close below waits on a producer that may ignore its abort. Settle a
            // parked next() first so a consumer is not held with it.
            let next_wake = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .next_wake
                .take();
            if let Some(wake) = next_wake {
                wake.settle(|pending| settle_with(pending, serde_json::json!({ "ended": true })));
            }
            // Close a stream rather than drop it so a pending collector settles before END.
            let outcome = match mode {
                Mode::Unpolled(stream)
                | Mode::Pump {
                    stream: Some(stream),
                    ..
                } => Some(
                    with_publication_callback_context(
                        close_context_id,
                        close_publication_buffer,
                        async move {
                            let mut stream = stream;
                            stream.close().await
                        },
                    )
                    .await,
                ),
                Mode::Pump {
                    stream: None,
                    closing: Some(closing),
                } => Some(closing.await),
                Mode::Pump { .. } | Mode::Released => None,
                Mode::Forwarded(forwarded) => {
                    forwarded.cancel.send_replace(true);
                    None
                }
            };
            if let Some(outcome) = outcome {
                owner_close_result
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .get_or_insert_with(|| finish_close(outcome, &owner_finalizer_error));
            }
            let close_wake = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .close_wake
                .take();
            if let Some(wake) = close_wake {
                let outcome = match owner_close_result
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone()
                {
                    Some(result) => result,
                    None => Ok(()),
                };
                wake.release(outcome);
            }
        });
        Self {
            state: handle,
            poll: OnceLock::new(),
            released,
            finalizer_error,
            close_result,
            context_id,
            publication_buffer,
        }
    }

    fn path(&self) -> Path {
        // A collector or finalizer calling in from a poll finds the lock held. Report the
        // pump path so a call queues behind that poll.
        let Some(state) = self.state.upgrade() else {
            return Path::Pump;
        };
        let mut state = match state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => return Path::Pump,
        };
        if matches!(state.mode, Mode::Unpolled(_)) {
            let Mode::Unpolled(stream) = std::mem::replace(&mut state.mode, Mode::Released) else {
                unreachable!("mode was checked above");
            };
            state.mode = if forwarding_required() {
                Mode::Forwarded(Forwarded::spawn(
                    stream,
                    self.context_id.clone(),
                    self.publication_buffer.clone(),
                    state.producers.clone(),
                ))
            } else {
                Mode::Pump {
                    stream: Some(stream),
                    closing: None,
                }
            };
        }
        match &state.mode {
            Mode::Forwarded(forwarded) => Path::Forwarded(Arc::clone(forwarded)),
            _ => Path::Pump,
        }
    }

    fn poll_function<'env>(&self, env: &'env Env) -> Result<Function<'env, JsArg, JsArg>> {
        if self.poll.get().is_none() {
            let _ = self.poll.set(self.create_poll_function(env)?.create_ref()?);
        }
        Ok(self
            .poll
            .get()
            .expect("poll function was just stored")
            .borrow_back(env)?)
    }

    fn create_poll_function<'env>(&self, env: &'env Env) -> Result<Function<'env, JsArg, JsArg>> {
        let state = Weak::clone(&self.state);
        let finalizer_error = self.finalizer_error.clone();
        let close_result = Arc::clone(&self.close_result);
        let context_id = self.context_id.clone();
        let publication_buffer = self.publication_buffer.clone();
        let function = closure_function(env, "__nemo_relay_stream_poll", move |ctx| {
            let close: bool = ctx.get(0)?;
            poll_step(
                ctx.env,
                &state,
                &finalizer_error,
                &close_result,
                context_id.clone(),
                publication_buffer.clone(),
                close,
            )
        })?;
        Ok(function)
    }
}

#[napi]
impl LlmStream {
    /// Retrieve the next chunk from the stream.
    ///
    /// Returns the next JSON chunk, or `null` when the stream is exhausted.
    /// Throws if the underlying stream encountered an error.
    #[napi(ts_return_type = "Promise<Json | null>", catch_unwind)]
    pub fn next<'env>(&self, env: &'env Env, this: This<'env>) -> Result<Unknown<'env>> {
        let promise = match self.path() {
            Path::Forwarded(forwarded) => spawn_promise::<Json, _, _, _, _>(
                env,
                async move { forwarded.next().await.map_err(Error::from) },
                js_value,
            )
            .and_then(|promise| Ok(callback_factory::pin_while(env, &this.object, &promise)?)),
            Path::Pump => self
                .poll_function(env)
                .and_then(|poll| Ok(callback_factory::stream_next(env, &poll, &this.object)?)),
        };
        rejection_for_failure(env, promise)
    }

    /// Stop the producer and wait for its cleanup to complete.
    ///
    /// Throws a stream error that `next()` has not yet thrown, then a cleanup
    /// error, then a finalizer failure. A cleanup failure after a stream error
    /// is logged. Further calls return the same result.
    #[napi(ts_return_type = "Promise<void>", catch_unwind)]
    pub fn close<'env>(&self, env: &'env Env, this: This<'env>) -> Result<Unknown<'env>> {
        let promise = match self.path() {
            Path::Forwarded(forwarded) => {
                let finalizer_error = self.finalizer_error.clone();
                spawn_promise::<(), _, _, _, _>(
                    env,
                    async move { forwarded.close(&finalizer_error).await },
                    |env, ()| js_value(env, ()),
                )
                .and_then(|promise| Ok(callback_factory::pin_while(env, &this.object, &promise)?))
            }
            Path::Pump => self
                .poll_function(env)
                .and_then(|poll| Ok(callback_factory::stream_close(env, &poll, &this.object)?)),
        };
        rejection_for_failure(env, promise)
    }
}

fn rejection_for_failure<'env>(
    env: &'env Env,
    promise: Result<Unknown<'env>>,
) -> Result<Unknown<'env>> {
    match promise {
        Ok(promise) => Ok(promise),
        Err(error) => Ok(PromiseRaw::<Json>::reject(env, error)?.to_unknown()),
    }
}

impl Drop for LlmStream {
    fn drop(&mut self) {
        self.released.send_replace(true);
    }
}

/// Poll a core stream once on the JavaScript thread. A returned promise means poll again
/// once it settles.
fn poll_step(
    env: &Env,
    state: &Weak<Mutex<StreamState>>,
    finalizer_error: &FinalizerError,
    close_result: &CloseResult,
    context_id: Option<String>,
    publication_buffer: Option<PublicationBuffer>,
    close: bool,
) -> Result<JsArg> {
    let Some(state) = state.upgrade() else {
        if !close {
            return Ok(js_value(env, Null)?);
        }
        let result = close_result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        return match result {
            Some(Err(failure)) => Err(Error::Napi(failure.throw_into(env))),
            _ => Ok(js_value(env, ())?),
        };
    };
    // A collector or finalizer may call back in from inside this poll. Resolve at once so
    // a call retries after this poll returns.
    let mut state = match state.try_lock() {
        Ok(state) => state,
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => {
            return Ok(detach(env, &PromiseRaw::<Json>::resolve(env, Json::Null)?));
        }
    };
    let slot = WakeSlot::new();
    let waker = Waker::from(Arc::clone(&slot));
    let mut cx = Context::from_waker(&waker);
    let _direct = DirectStreamCallbacks::enter();
    let polled = within_runtime_if_available(|| {
        with_publication_callback_context_sync(context_id, publication_buffer, || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if close {
                    state
                        .poll_close(&mut cx, finalizer_error, close_result)
                        .map(|result| result.map(|()| None))
                } else {
                    state.poll_next(&mut cx).map(|result| {
                        result.map_err(|error| CloseFailure::from(Error::from(error)))
                    })
                }
            }))
        })
    });
    match polled {
        Err(panic) => Err(Error::generic_failure(format!(
            "stream poll panicked: {}",
            panic_message(panic.as_ref())
        ))),
        Ok(Poll::Pending) => {
            let (pending, promise) = create_promise::<Json>(env, Liveness::KeepLoopAlive)?;
            slot.arm(pending);
            if close {
                state.close_wake = Some(slot);
            } else {
                state.next_wake = Some(slot);
            }
            Ok(detach(env, &promise))
        }
        Ok(Poll::Ready(result)) => {
            if close {
                state.close_wake = None;
            } else {
                state.next_wake = None;
            }
            Ok(match result {
                Ok(Some(chunk)) => js_value(env, chunk)?,
                Ok(None) if close => js_value(env, ())?,
                Ok(None) => js_value(env, Null)?,
                Err(failure) => return Err(Error::Napi(failure.throw_into(env))),
            })
        }
    }
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}

fn combine_close_errors(
    failures: impl IntoIterator<Item = (&'static str, Option<CloseFailure>)>,
) -> std::result::Result<(), CloseFailure> {
    let mut errors = failures
        .into_iter()
        .filter_map(|(label, failure)| failure.map(|failure| (label, failure)))
        .flat_map(|(label, failure)| {
            let CloseErrors { first, rest } = &*failure.0;
            std::iter::once((label, Arc::clone(first)))
                .chain(
                    rest.iter()
                        .map(|(label, error)| (*label, Arc::clone(error))),
                )
                .collect::<Vec<_>>()
        });
    let Some((_, first)) = errors.next() else {
        return Ok(());
    };
    Err(CloseFailure(Arc::new(CloseErrors {
        first,
        rest: errors.collect(),
    })))
}

#[cfg(test)]
#[path = "../tests/rust/stream_tests.rs"]
mod tests;
