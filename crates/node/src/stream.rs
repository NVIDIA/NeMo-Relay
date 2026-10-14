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

use napi::bindgen_prelude::*;
use napi::{Env, JsDeferred, JsFunction, JsObject, JsUnknown, NapiRaw, NapiValue};
use napi_derive::napi;
use nemo_relay::api::runtime::LlmJsonStream;
use nemo_relay::api::runtime::subscriber_dispatcher::PublicationBuffer;
use nemo_relay::error::{FlowError, Result as FlowResult};
use serde_json::Value as Json;
use tokio_stream::{Stream, StreamExt};

use crate::api::PersistentJsFunction;
use crate::callable::{DirectStreamCallbacks, FinalizerError, json_to_js_unknown};
use crate::callback_factory;
use crate::convert::to_napi_err;
use crate::promise_call::{
    StreamCancellation, env_teardown_flag, with_publication_callback_context,
    with_publication_callback_context_sync,
};

type CloseFuture = Pin<Box<dyn Future<Output = FlowResult<()>> + Send>>;
type CloseResult = Arc<Mutex<Option<std::result::Result<(), String>>>>;
type WakeResolver = Box<dyn FnOnce(Env) -> Result<Json> + Send>;

fn wake_value(value: Json) -> WakeResolver {
    Box::new(move |_| Ok(value))
}

enum WakeState {
    Idle,
    Armed(JsDeferred<Json, WakeResolver>),
    Fired,
}

/// Wakes a pending poll by resolving the promise it awaits, from the thread that fires it.
///
/// `null` tells a poll loop to poll again. An object is the outcome of a
/// stream released while a poll waited.
struct WakeSlot {
    state: Mutex<WakeState>,
    teardown: Arc<AtomicBool>,
}

impl WakeSlot {
    fn new(teardown: Arc<AtomicBool>) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(WakeState::Idle),
            teardown,
        })
    }

    fn fire(&self) {
        self.settle(Json::Null);
    }

    fn settle(&self, value: Json) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // A worker finalizer can wake a poll after Node has closed its promise's threadsafe
        // function. Skip a resolve once the environment is tearing down.
        if let WakeState::Armed(deferred) = std::mem::replace(&mut *state, WakeState::Fired)
            && !self.teardown.load(Ordering::Acquire)
        {
            deferred.resolve(wake_value(value));
        }
    }

    // A slot is armed after its poll. Resolve at once if a wake already happened.
    fn arm(&self, deferred: JsDeferred<Json, WakeResolver>) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if matches!(*state, WakeState::Fired) {
            deferred.resolve(wake_value(Json::Null));
        } else {
            *state = WakeState::Armed(deferred);
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
    closed: tokio::sync::watch::Receiver<Option<std::result::Result<(), String>>>,
    close_result: tokio::sync::Mutex<Option<std::result::Result<(), String>>>,
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

    async fn close(&self, finalizer_error: &FinalizerError) -> std::result::Result<(), String> {
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
            .map_err(|_| "stream close task ended before releasing the producer".to_string())?
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
                undelivered_error.map(|error| error.to_string()),
            ),
            ("stream cleanup also failed", result.err()),
            ("stream finalizer also failed", finalizer_error),
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
    closed: tokio::sync::watch::Sender<Option<std::result::Result<(), String>>>,
) {
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
        permit.send(item);
    }
    closed.send_replace(Some(
        stream.close().await.map_err(|error| error.to_string()),
    ));
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
    codec_references: Vec<Arc<PersistentJsFunction>>,
    context_id: Option<String>,
    publication_buffer: Option<PublicationBuffer>,
}

fn finish_close(
    outcome: FlowResult<()>,
    finalizer_error: &FinalizerError,
) -> std::result::Result<(), String> {
    let finalizer_error = finalizer_error
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    combine_close_errors([
        (
            "stream cleanup also failed",
            outcome.err().map(|error| error.to_string()),
        ),
        ("stream finalizer also failed", finalizer_error),
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
    ) -> Poll<std::result::Result<(), String>> {
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

/// Reference to a poll function, deleted by its finalizer while the environment is live.
struct PollFunction(napi::sys::napi_ref);

// SAFETY: the reference is created, called, and deleted on the JavaScript thread only.
unsafe impl Send for PollFunction {}
// SAFETY: as for `Send`, the handle is never used from another thread.
unsafe impl Sync for PollFunction {}

/// An async iterator over chunks from a streaming LLM response.
///
/// Obtained from `llmStreamCallExecute()`. Call `next()` repeatedly to consume
/// response chunks. Returns `null` when the stream is fully consumed.
#[napi(custom_finalize)]
pub struct LlmStream {
    state: Weak<Mutex<StreamState>>,
    poll: OnceLock<PollFunction>,
    released: tokio::sync::watch::Sender<bool>,
    finalizer_error: FinalizerError,
    close_result: CloseResult,
    context_id: Option<String>,
    publication_buffer: Option<PublicationBuffer>,
}

impl LlmStream {
    pub(crate) fn new(
        stream: LlmJsonStream,
        codec_references: Vec<Arc<PersistentJsFunction>>,
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
            codec_references,
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
                wake.settle(serde_json::json!({ "ended": true }));
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
            let (close_wake, codec_references) = {
                let mut state = state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                (
                    state.close_wake.take(),
                    std::mem::take(&mut state.codec_references),
                )
            };
            if let Some(wake) = close_wake {
                let error = owner_close_result
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone()
                    .and_then(|result| result.err());
                wake.settle(serde_json::json!({ "closed": true, "error": error }));
            }
            drop(codec_references);
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

    fn poll_function(&self, env: &Env) -> Result<JsFunction> {
        if self.poll.get().is_none() {
            let function = self.create_poll_function(env)?;
            let mut reference = std::ptr::null_mut();
            // SAFETY: `function` is a live handle created in `env`, and `reference`
            // is writable storage for the new reference.
            let status = unsafe {
                napi::sys::napi_create_reference(env.raw(), function.raw(), 1, &mut reference)
            };
            if status != napi::sys::Status::napi_ok {
                return Err(napi::Error::from_reason(format!(
                    "failed to reference the stream poll function: {:?}",
                    napi::Status::from(status)
                )));
            }
            let _ = self.poll.set(PollFunction(reference));
        }
        let poll = self.poll.get().expect("poll function was just stored");
        let mut value = std::ptr::null_mut();
        // SAFETY: `poll.0` is a live reference created in `env`, and `value` is writable storage.
        let status = unsafe { napi::sys::napi_get_reference_value(env.raw(), poll.0, &mut value) };
        if status != napi::sys::Status::napi_ok {
            return Err(napi::Error::from_reason(format!(
                "failed to borrow the stream poll function: {:?}",
                napi::Status::from(status)
            )));
        }
        // SAFETY: `value` resolved from this handle's own reference, so it is a live function.
        Ok(unsafe { JsFunction::from_raw_unchecked(env.raw(), value) })
    }

    fn create_poll_function(&self, env: &Env) -> Result<JsFunction> {
        let state = Weak::clone(&self.state);
        let finalizer_error = self.finalizer_error.clone();
        let close_result = Arc::clone(&self.close_result);
        let context_id = self.context_id.clone();
        let publication_buffer = self.publication_buffer.clone();
        let function =
            env.create_function_from_closure("__nemo_relay_stream_poll", move |ctx| {
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

impl ObjectFinalize for LlmStream {
    fn finalize(mut self, env: Env) -> Result<()> {
        if let Some(poll) = self.poll.take() {
            // SAFETY: the reference was created in this environment and is deleted exactly once.
            let _ = unsafe { napi::sys::napi_delete_reference(env.raw(), poll.0) };
        }
        Ok(())
    }
}

#[napi]
impl LlmStream {
    /// Retrieve the next chunk from the stream.
    ///
    /// Returns the next JSON chunk, or `null` when the stream is exhausted.
    /// Throws if the underlying stream encountered an error.
    #[napi(ts_return_type = "Promise<Json | null>")]
    pub fn next(&self, env: Env) -> Result<JsObject> {
        let promise = match self.path() {
            Path::Forwarded(forwarded) => env.execute_tokio_future(
                async move { forwarded.next().await.map_err(to_napi_err) },
                |_env, value| Ok(value),
            ),
            Path::Pump => self
                .poll_function(&env)
                .and_then(|poll| callback_factory::stream_next(&env, &poll)),
        };
        rejection_for_failure(&env, promise)
    }

    /// Stop the producer and wait for its cleanup to complete.
    ///
    /// Throws a stream error that `next()` has not yet thrown, then a cleanup
    /// error, then a finalizer failure. A cleanup failure after a stream error
    /// is logged. Further calls return the same result.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn close(&self, env: Env) -> Result<JsObject> {
        let promise = match self.path() {
            Path::Forwarded(forwarded) => {
                let finalizer_error = self.finalizer_error.clone();
                env.execute_tokio_future(
                    async move {
                        forwarded
                            .close(&finalizer_error)
                            .await
                            .map_err(napi::Error::from_reason)
                    },
                    |_env, ()| Ok(()),
                )
            }
            Path::Pump => self
                .poll_function(&env)
                .and_then(|poll| callback_factory::stream_close(&env, &poll)),
        };
        rejection_for_failure(&env, promise)
    }
}

fn rejection_for_failure(env: &Env, promise: Result<JsObject>) -> Result<JsObject> {
    match promise {
        Ok(promise) => Ok(promise),
        Err(error) => {
            let (deferred, promise) = env.create_deferred::<Json, WakeResolver>()?;
            deferred.reject(error);
            Ok(promise)
        }
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
) -> Result<JsUnknown> {
    let Some(state) = state.upgrade() else {
        if !close {
            return env.get_null().map(|value| value.into_unknown());
        }
        let result = close_result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        return match result {
            Some(Err(error)) => Err(napi::Error::from_reason(error)),
            _ => env.get_undefined().map(|value| value.into_unknown()),
        };
    };
    // A collector or finalizer may call back in from inside this poll. Resolve at once so
    // a call retries after this poll returns.
    let mut state = match state.try_lock() {
        Ok(state) => state,
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => {
            let (deferred, promise) = env.create_deferred::<Json, WakeResolver>()?;
            deferred.resolve(wake_value(Json::Null));
            return Ok(promise.into_unknown());
        }
    };
    let slot = WakeSlot::new(env_teardown_flag(env)?);
    let waker = Waker::from(Arc::clone(&slot));
    let mut cx = Context::from_waker(&waker);
    let _direct = DirectStreamCallbacks::enter();
    let polled = within_runtime_if_available(|| {
        with_publication_callback_context_sync(context_id, publication_buffer, || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if close {
                    state
                        .poll_close(&mut cx, finalizer_error, close_result)
                        .map(|result| result.map(|()| None).map_err(napi::Error::from_reason))
                } else {
                    state
                        .poll_next(&mut cx)
                        .map(|result| result.map_err(to_napi_err))
                }
            }))
        })
    });
    match polled {
        Err(panic) => Err(napi::Error::from_reason(format!(
            "stream poll panicked: {}",
            panic_message(panic.as_ref())
        ))),
        Ok(Poll::Pending) => {
            let (deferred, promise) = env.create_deferred::<Json, WakeResolver>()?;
            slot.arm(deferred);
            if close {
                state.close_wake = Some(slot);
            } else {
                state.next_wake = Some(slot);
            }
            Ok(promise.into_unknown())
        }
        Ok(Poll::Ready(result)) => {
            if close {
                state.close_wake = None;
            } else {
                state.next_wake = None;
            }
            match result? {
                Some(chunk) => json_to_js_unknown(env, chunk),
                None if close => env.get_undefined().map(|value| value.into_unknown()),
                None => env.get_null().map(|value| value.into_unknown()),
            }
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
    errors: impl IntoIterator<Item = (&'static str, Option<String>)>,
) -> std::result::Result<(), String> {
    let mut errors = errors
        .into_iter()
        .filter_map(|(label, error)| error.map(|error| (label, error)));
    let Some((_, first)) = errors.next() else {
        return Ok(());
    };
    Err(errors.fold(first, |acc, (label, next)| {
        format!("{acc}; {label}: {next}")
    }))
}
