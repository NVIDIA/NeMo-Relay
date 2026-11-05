// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, TryLockError, Weak};
use std::task::{Context, Poll, Wake, Waker};

use napi::Env;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use nemo_relay_runtime::api::runtime::LlmJsonStream;
use nemo_relay_types::error::{FlowError, Result as FlowResult};
use serde_json::Value as Json;
use tokio_stream::{Stream, StreamExt};

use crate::callable::{DirectStreamCallbacks, FinalizerError};
use crate::callback_factory;
use crate::convert::{
    BoxFuture, JsArg, JsBuilder, closure_function, detach, js_value, lock, settled,
};
use crate::environment::{Liveness, PendingPromise, create_promise, spawn_promise};
use crate::error::{Error, Result};
use crate::promise_call::{PublicationContext, StreamCancellation};

type CloseFuture = BoxFuture<FlowResult<()>>;
type CloseOutcome = std::result::Result<(), CloseFailure>;
type CloseResult = Arc<Mutex<Option<CloseOutcome>>>;

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

impl From<FlowError> for CloseFailure {
    fn from(error: FlowError) -> Self {
        Error::from(error).into()
    }
}

impl From<CloseFailure> for JsBuilder {
    fn from(failure: CloseFailure) -> Self {
        Box::new(move |env| failure.into_js(env))
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

    fn release(&self, outcome: CloseOutcome) {
        self.settle(|pending| match outcome {
            Ok(()) => settle_with(pending, Json::Bool(true)),
            Err(failure) => pending.reject_with(failure.into()),
        });
    }

    fn settle(&self, settle: impl FnOnce(PendingPromise)) {
        let mut state = lock(&self.state);
        if let WakeState::Armed(pending) = std::mem::replace(&mut *state, WakeState::Fired) {
            settle(pending);
        }
    }

    fn arm(&self, pending: PendingPromise) {
        let mut state = lock(&self.state);
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
    closed: tokio::sync::watch::Receiver<Option<CloseOutcome>>,
    close_result: tokio::sync::Mutex<Option<CloseOutcome>>,
}

impl Forwarded {
    fn spawn(
        stream: LlmJsonStream,
        publication: PublicationContext,
        producers: StreamCancellation,
    ) -> Arc<Self> {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let (requests, request_rx) = tokio::sync::mpsc::channel(1);
        let (cancel, cancel_rx) = tokio::sync::watch::channel(false);
        let (closed, closed_rx) = tokio::sync::watch::channel(None);
        within_runtime_if_available(|| {
            tokio::spawn(publication.run(forward_stream_to_channel(
                stream, tx, request_rx, cancel_rx, closed,
            )))
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
        let _ = self.requests.send(()).await;
        receiver.recv().await.transpose()
    }

    async fn close(&self, finalizer_error: &FinalizerError) -> CloseOutcome {
        let mut close_result = self.close_result.lock().await;
        if let Some(result) = close_result.as_ref() {
            return result.clone();
        }
        self.cancel.send_replace(true);
        self.producers.cancel();
        let result = settled(self.closed.clone()).await.ok_or_else(|| {
            Error::generic_failure("stream close task ended before releasing the producer")
        })?;
        let mut receiver = self.receiver.lock().await;
        receiver.close();
        let mut undelivered_error = None;
        while let Ok(item) = receiver.try_recv() {
            if let Err(error) = item {
                undelivered_error.get_or_insert(error);
            }
        }
        let result = finish_close(undelivered_error, result, finalizer_error);
        *close_result = Some(result.clone());
        result
    }
}

async fn forward_stream_to_channel(
    mut stream: LlmJsonStream,
    tx: tokio::sync::mpsc::Sender<FlowResult<Json>>,
    mut requests: tokio::sync::mpsc::Receiver<()>,
    mut cancel: tokio::sync::watch::Receiver<bool>,
    closed: tokio::sync::watch::Sender<Option<CloseOutcome>>,
) {
    let mut undelivered_error = None;
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
) -> CloseOutcome {
    match undelivered_error {
        Some(error) => {
            if let Err(cleanup) = cleanup {
                log::warn!(
                    target: "nemo_relay.runtime",
                    event = "stream_cleanup_failed";
                    "LLM stream cleanup failed after the stream failed: {cleanup}"
                );
            }
            Err(CloseFailure::from(error))
        }
        None => cleanup.map_err(CloseFailure::from),
    }
}

fn spawn_close(stream: LlmJsonStream, publication: PublicationContext) -> CloseFuture {
    let closed = tokio::spawn(publication.run(async move {
        let mut stream = stream;
        stream.close().await
    }));
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
    publication: PublicationContext,
}

fn finish_close(
    stream_error: Option<FlowError>,
    cleanup: CloseOutcome,
    finalizer_error: &FinalizerError,
) -> CloseOutcome {
    let finalizer_error = lock(finalizer_error).take();
    combine_close_errors([
        ("stream failed", stream_error.map(CloseFailure::from)),
        ("stream cleanup also failed", cleanup.err()),
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
            *closing = Some(spawn_close(core, self.publication.clone()));
        }
        Poll::Ready(ended)
    }

    fn poll_close(
        &mut self,
        cx: &mut Context<'_>,
        finalizer_error: &FinalizerError,
        close_result: &CloseResult,
    ) -> Poll<CloseOutcome> {
        if let Some(result) = lock(close_result).clone() {
            return Poll::Ready(result);
        }
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
            if let Some(wake) = self.next_wake.take() {
                wake.fire();
            }
            *closing = Some(spawn_close(taken, self.publication.clone()));
        }
        let Some(future) = closing.as_mut() else {
            return Poll::Ready(Ok(()));
        };
        match future.as_mut().poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(outcome) => {
                *closing = None;
                let result =
                    finish_close(None, outcome.map_err(CloseFailure::from), finalizer_error);
                *lock(close_result) = Some(result.clone());
                Poll::Ready(result)
            }
        }
    }
}

enum Path {
    Pump,
    Forwarded(Arc<Forwarded>),
}

struct Shared {
    state: Weak<Mutex<StreamState>>,
    finalizer_error: FinalizerError,
    close_result: CloseResult,
    publication: PublicationContext,
}

/// An async iterator over chunks from a streaming LLM response.
///
/// Obtained from `llmStreamCallExecute()`. Call `next()` repeatedly to consume
/// response chunks. Returns `null` when the stream is fully consumed.
#[napi]
pub struct LlmStream {
    shared: Arc<Shared>,
    poll: OnceLock<FunctionRef<JsArg, JsArg>>,
    released: tokio::sync::watch::Sender<bool>,
}

impl LlmStream {
    pub(crate) fn new(
        stream: LlmJsonStream,
        finalizer_error: FinalizerError,
        producers: StreamCancellation,
        publication: PublicationContext,
    ) -> Self {
        let state = Arc::new(Mutex::new(StreamState {
            mode: Mode::Unpolled(stream),
            releasing: false,
            next_wake: None,
            close_wake: None,
            producers,
            publication: publication.clone(),
        }));
        let shared = Arc::new(Shared {
            state: Arc::downgrade(&state),
            finalizer_error,
            close_result: Arc::default(),
            publication,
        });
        let (released, mut release) = tokio::sync::watch::channel(false);
        let owner = Arc::clone(&shared);
        tokio::spawn(async move {
            let _ = release.wait_for(|released| *released).await;
            let (mode, producers) = {
                let mut state = lock(&state);
                state.releasing = true;
                (
                    std::mem::replace(&mut state.mode, Mode::Released),
                    state.producers.clone(),
                )
            };
            producers.cancel();
            let next_wake = lock(&state).next_wake.take();
            if let Some(wake) = next_wake {
                wake.settle(|pending| settle_with(pending, serde_json::json!({ "ended": true })));
            }
            let outcome = match mode {
                Mode::Unpolled(stream)
                | Mode::Pump {
                    stream: Some(stream),
                    ..
                } => Some(spawn_close(stream, owner.publication.clone()).await),
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
                lock(&owner.close_result).get_or_insert_with(|| {
                    finish_close(
                        None,
                        outcome.map_err(CloseFailure::from),
                        &owner.finalizer_error,
                    )
                });
            }
            let close_wake = lock(&state).close_wake.take();
            if let Some(wake) = close_wake {
                let outcome = lock(&owner.close_result).clone().unwrap_or(Ok(()));
                wake.release(outcome);
            }
        });
        Self {
            shared,
            poll: OnceLock::new(),
            released,
        }
    }

    fn path(&self) -> Path {
        let Some(state) = self.shared.state.upgrade() else {
            return Path::Pump;
        };
        let mut state = match state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => return Path::Pump,
        };
        state.mode = match std::mem::replace(&mut state.mode, Mode::Released) {
            Mode::Unpolled(stream) if forwarding_required() => Mode::Forwarded(Forwarded::spawn(
                stream,
                self.shared.publication.clone(),
                state.producers.clone(),
            )),
            Mode::Unpolled(stream) => Mode::Pump {
                stream: Some(stream),
                closing: None,
            },
            polled => polled,
        };
        match &state.mode {
            Mode::Forwarded(forwarded) => Path::Forwarded(Arc::clone(forwarded)),
            _ => Path::Pump,
        }
    }

    fn drive<'env, T, E, F>(
        &self,
        env: &'env Env,
        this: This<'env>,
        close: bool,
        forwarded: impl FnOnce(Arc<Forwarded>) -> F,
        resolve: fn(&Env, T) -> napi::Result<JsArg>,
    ) -> Result<Unknown<'env>>
    where
        T: Send + 'static,
        E: Into<JsBuilder> + Send + 'static,
        F: Future<Output = std::result::Result<T, E>> + Send + 'static,
    {
        let promise = match self.path() {
            Path::Forwarded(stream) => {
                let promise: PromiseRaw<'_, ()> = spawn_promise(env, forwarded(stream), resolve)?;
                Ok(callback_factory::pin_while(env, &this.object, &promise)?)
            }
            Path::Pump => self.poll_function(env).and_then(|poll| {
                Ok(callback_factory::stream_step(
                    env,
                    close,
                    &poll,
                    &this.object,
                )?)
            }),
        };
        match promise {
            Ok(promise) => Ok(promise),
            Err(error) => Ok(PromiseRaw::<Json>::reject(env, error)?.to_unknown()),
        }
    }

    fn poll_function<'env>(&self, env: &'env Env) -> Result<Function<'env, JsArg, JsArg>> {
        let poll = match self.poll.get() {
            Some(poll) => poll,
            None => {
                let shared = Arc::clone(&self.shared);
                let poll = closure_function(env, "__nemo_relay_stream_poll", move |ctx| {
                    let close: bool = ctx.get(0)?;
                    poll_step(ctx.env, &shared, close)
                })?
                .create_ref()?;
                self.poll.get_or_init(|| poll)
            }
        };
        Ok(poll.borrow_back(env)?)
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
        self.drive(
            env,
            this,
            false,
            |forwarded| async move { forwarded.next().await.map_err(Error::from) },
            js_value,
        )
    }

    /// Stop the producer and wait for its cleanup to complete.
    ///
    /// Throws a stream error that `next()` has not yet thrown, then a cleanup
    /// error, then a finalizer failure. A cleanup failure after a stream error
    /// is logged. Further calls return the same result.
    #[napi(ts_return_type = "Promise<void>", catch_unwind)]
    pub fn close<'env>(&self, env: &'env Env, this: This<'env>) -> Result<Unknown<'env>> {
        let finalizer_error = self.shared.finalizer_error.clone();
        self.drive(
            env,
            this,
            true,
            |forwarded| async move { forwarded.close(&finalizer_error).await },
            |env, ()| js_value(env, ()),
        )
    }
}

impl Drop for LlmStream {
    fn drop(&mut self) {
        self.released.send_replace(true);
    }
}

fn poll_step(env: &Env, shared: &Shared, close: bool) -> Result<JsArg> {
    let Some(state) = shared.state.upgrade() else {
        if !close {
            return Ok(js_value(env, Null)?);
        }
        return match lock(&shared.close_result).clone() {
            Some(Err(failure)) => Err(Error::Napi(failure.throw_into(env))),
            _ => Ok(js_value(env, ())?),
        };
    };
    let mut state = lock(&state);
    let slot = WakeSlot::new();
    let waker = Waker::from(Arc::clone(&slot));
    let mut cx = Context::from_waker(&waker);
    let _direct = DirectStreamCallbacks::enter();
    let polled = within_runtime_if_available(|| {
        shared.publication.clone().run_sync(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if close {
                    state
                        .poll_close(&mut cx, &shared.finalizer_error, &shared.close_result)
                        .map(|result| result.map(|()| None))
                } else {
                    state
                        .poll_next(&mut cx)
                        .map(|result| result.map_err(CloseFailure::from))
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
) -> CloseOutcome {
    let mut errors = Vec::new();
    for (label, failure) in failures {
        let Some(failure) = failure else {
            continue;
        };
        let CloseErrors { first, rest } = &*failure.0;
        errors.push((label, Arc::clone(first)));
        errors.extend(
            rest.iter()
                .map(|(label, error)| (*label, Arc::clone(error))),
        );
    }
    let mut errors = errors.into_iter();
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
