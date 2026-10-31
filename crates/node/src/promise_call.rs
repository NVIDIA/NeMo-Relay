// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Promise-aware JS function calling for NeMo Relay NAPI bindings.
//!
//! This module wraps JS middleware callbacks so Rust can call them from any thread
//! and await either synchronous return values or Promise-returning callbacks.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use napi::Env;
use napi::bindgen_prelude::{
    External, FnArgs, FromNapiRef, Function, FunctionCallContext, JsValue, Unknown,
    within_runtime_if_available,
};
use napi::threadsafe_function::{ThreadsafeCallContext, ThreadsafeFunctionCallMode};
use serde_json::Value as Json;

use nemo_relay::api::runtime::subscriber_dispatcher::{
    PublicationBuffer, with_nested_publication_buffer, with_task_nested_publication_buffer,
};
use nemo_relay::api::runtime::{LlmStreamInner, MiddlewareContinuationContext};
use nemo_relay::error::{FlowError, Result as FlowResult};

use crate::callback_factory::{self, CallbackScope, PromiseSettlers};
use crate::convert::{HandledTsfn, JsArg, abort_tsfn, closure_function, detach, js_value};
use crate::environment::{Liveness, create_promise, spawn_promise};
use crate::error::{Error, Result};
use tokio_stream::Stream;

tokio::task_local! {
    static PUBLICATION_CALLBACK_CONTEXT_ID: Option<String>;
}

pub(crate) async fn with_publication_callback_context<F: Future>(
    context_id: Option<String>,
    publication_buffer: Option<PublicationBuffer>,
    future: F,
) -> F::Output {
    with_task_nested_publication_buffer(
        publication_buffer,
        PUBLICATION_CALLBACK_CONTEXT_ID.scope(context_id, future),
    )
    .await
}

pub(crate) fn with_publication_callback_context_sync<T>(
    context_id: Option<String>,
    publication_buffer: Option<PublicationBuffer>,
    f: impl FnOnce() -> T,
) -> T {
    with_nested_publication_buffer(publication_buffer, || {
        PUBLICATION_CALLBACK_CONTEXT_ID.sync_scope(context_id, f)
    })
}

pub(crate) fn publication_callback_context_id() -> Option<String> {
    PUBLICATION_CALLBACK_CONTEXT_ID
        .try_with(Clone::clone)
        .unwrap_or(None)
}

pub type JsonNextFn =
    Arc<dyn Fn(Json) -> Pin<Box<dyn Future<Output = FlowResult<Json>> + Send>> + Send + Sync>;
pub type JsonStreamNextFn = Arc<
    dyn Fn(
            Json,
        ) -> Pin<
            Box<dyn Future<Output = FlowResult<nemo_relay::api::runtime::LlmJsonStream>> + Send>,
        > + Send
        + Sync,
>;

#[derive(Clone)]
enum NextFn {
    Json(JsonNextFn),
    Stream(JsonStreamNextFn),
}

struct NextState {
    next: NextFn,
    continuation_context: MiddlewareContinuationContext,
    publication_context_id: Option<String>,
}

/// Builds the first JS callback argument on the Node main thread.
///
/// Some callback arguments, such as `#[napi]` class instances, cannot cross the
/// threadsafe-function boundary as plain JSON. This builder runs inside the
/// threadsafe-function call (on the JS thread), so it can materialize those
/// values directly instead of serializing them.
pub type Arg0Builder = Box<dyn FnOnce(&Env) -> napi::Result<JsArg> + Send>;

/// The first argument passed to the wrapped JS callback.
enum PrimaryArg {
    /// A plain JSON value converted on the JS thread.
    Json(Json),
    /// A value materialized on the JS thread by a builder closure.
    Build(Arg0Builder),
}

struct InvokeArgs {
    call_id: u64,
    arg0: PrimaryArg,
    spread: bool,
    next: Option<NextState>,
    publication: bool,
    publication_context_id: Option<String>,
    scope: CallbackScope,
    stream_result: bool,
}

struct CallArgs {
    invoke: InvokeArgs,
    completion: CallCompletion,
}

enum PromiseCall {
    Invoke(Box<CallArgs>),
    Abort(u64),
}

/// `(abort, callId, arg0, spread, next, publication, publicationContextId, scope, streamResult)`.
type PromiseArgs = FnArgs<(
    JsArg,
    JsArg,
    JsArg,
    JsArg,
    JsArg,
    JsArg,
    JsArg,
    JsArg,
    JsArg,
)>;

type PromiseTsfn = HandledTsfn<PromiseCall, PromiseArgs>;

#[derive(Clone, Copy)]
struct CallMode {
    spread: bool,
    publication: bool,
}

impl CallMode {
    const DIRECT: Self = Self {
        spread: false,
        publication: false,
    };
    const SPREAD: Self = Self {
        spread: true,
        publication: false,
    };
    const SPREAD_PUBLICATION: Self = Self {
        spread: true,
        publication: true,
    };
}

struct JsonCompletion {
    sender: tokio::sync::oneshot::Sender<FlowResult<Json>>,
    finished: Arc<AtomicBool>,
}

impl JsonCompletion {
    fn settle(self, result: FlowResult<Json>) {
        self.finished.store(true, Ordering::Release);
        let _ = self.sender.send(result);
    }
}

struct StreamCompletion {
    ready: Mutex<Option<tokio::sync::oneshot::Sender<FlowResult<()>>>>,
    chunks: Mutex<Option<tokio::sync::mpsc::Sender<FlowResult<Json>>>>,
    terminal_error: Mutex<Option<FlowError>>,
    closed: tokio::sync::watch::Sender<Option<FlowResult<()>>>,
    finished: Arc<AtomicBool>,
    parked: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl StreamCompletion {
    fn ready(&self) {
        if let Some(ready) = self.ready.lock().unwrap().take() {
            let _ = ready.send(Ok(()));
        }
    }

    fn finish(&self, result: FlowResult<()>) {
        self.finished.store(true, Ordering::Release);
        self.chunks.lock().unwrap().take();
        self.closed.send_replace(Some(result));
    }

    fn fail(&self, error: FlowError) {
        match self.ready.lock().unwrap().take() {
            Some(ready) => {
                let _ = ready.send(Err(error.clone()));
            }
            None => *self.terminal_error.lock().unwrap() = Some(error.clone()),
        }
        self.finish(Err(error));
    }
}

enum CallCompletion {
    Json(JsonCompletion),
    Stream(Arc<StreamCompletion>),
}

impl CallCompletion {
    fn resolve(self, value: impl FnOnce() -> FlowResult<Json>) -> Option<Self> {
        match self {
            Self::Json(completion) => {
                completion.settle(value());
                None
            }
            Self::Stream(stream) => {
                stream.ready();
                Some(Self::Stream(stream))
            }
        }
    }

    fn fail(self, error: FlowError) {
        match self {
            Self::Json(completion) => completion.settle(Err(error)),
            Self::Stream(stream) => stream.fail(error),
        }
    }
}

#[derive(Default)]
struct CallRegistry {
    next_id: AtomicU64,
    calls: Mutex<HashMap<u64, CallCompletion>>,
}

impl CallRegistry {
    fn allocate(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn insert(&self, call_id: u64, completion: CallCompletion) {
        self.calls.lock().unwrap().insert(call_id, completion);
    }

    fn resolve(&self, call_id: u64, value: impl FnOnce() -> FlowResult<Json>) {
        let mut calls = self.calls.lock().unwrap();
        if let Some(completion) = calls.remove(&call_id)
            && let Some(kept) = completion.resolve(value)
        {
            calls.insert(call_id, kept);
        }
    }

    fn fail(&self, call_id: u64, error: FlowError) {
        if let Some(completion) = self.calls.lock().unwrap().remove(&call_id) {
            completion.fail(error);
        }
    }

    fn stream(&self, call_id: u64) -> Option<Arc<StreamCompletion>> {
        match self.calls.lock().unwrap().get(&call_id) {
            Some(CallCompletion::Stream(stream)) => Some(Arc::clone(stream)),
            Some(CallCompletion::Json(_)) | None => None,
        }
    }

    fn end_stream(&self, call_id: u64) {
        let mut calls = self.calls.lock().unwrap();
        if let Entry::Occupied(entry) = calls.entry(call_id)
            && let CallCompletion::Stream(stream) = entry.get()
        {
            stream.finish(Ok(()));
            entry.remove();
        }
    }
}

#[derive(Clone)]
struct CallCancellation {
    call_id: u64,
    requested: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
    tsfn: Arc<PromiseTsfn>,
}

impl CallCancellation {
    fn new(call_id: u64, tsfn: Arc<PromiseTsfn>, finished: Arc<AtomicBool>) -> Self {
        Self {
            call_id,
            requested: Arc::default(),
            finished,
            tsfn,
        }
    }

    fn cancel(&self) {
        if self.finished.load(Ordering::Acquire) || self.requested.swap(true, Ordering::AcqRel) {
            return;
        }
        let _ = self.tsfn.call(
            Ok(PromiseCall::Abort(self.call_id)),
            ThreadsafeFunctionCallMode::NonBlocking,
        );
    }
}

struct CallCancellationGuard(Option<CallCancellation>);

impl CallCancellationGuard {
    fn new(cancellation: CallCancellation) -> Self {
        Self(Some(cancellation))
    }

    fn disarm(&mut self) {
        self.0.take();
    }
}

impl Drop for CallCancellationGuard {
    fn drop(&mut self) {
        if let Some(cancellation) = self.0.take() {
            cancellation.cancel();
        }
    }
}

/// Provider aborts shared with the stream handle so `close()` can fire them while
/// a collector blocks the forwarding task.
#[derive(Clone, Default)]
pub(crate) struct StreamCancellation {
    state: Arc<Mutex<StreamCancellationState>>,
}

#[derive(Default)]
struct StreamCancellationState {
    requested: bool,
    calls: Vec<CallCancellation>,
}

impl StreamCancellation {
    fn attach(&self, call: &CallCancellation) {
        let mut state = self.state.lock().unwrap();
        if state.requested {
            drop(state);
            call.cancel();
            return;
        }
        state.calls.push(call.clone());
    }

    pub(crate) fn cancel(&self) {
        let mut state = self.state.lock().unwrap();
        state.requested = true;
        for call in &state.calls {
            call.cancel();
        }
    }
}

struct JsCallbackStream {
    receiver: tokio_stream::wrappers::ReceiverStream<FlowResult<Json>>,
    completion: Arc<StreamCompletion>,
    closed: tokio::sync::watch::Receiver<Option<FlowResult<()>>>,
    cancellation: CallCancellation,
}

impl Stream for JsCallbackStream {
    type Item = FlowResult<Json>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.receiver).poll_next(cx) {
            Poll::Ready(None) => Poll::Ready(
                self.completion
                    .terminal_error
                    .lock()
                    .unwrap()
                    .take()
                    .map(Err),
            ),
            poll => poll,
        }
    }
}

impl LlmStreamInner for JsCallbackStream {
    fn terminalize(self: Pin<&mut Self>) {
        let this = self.get_mut();
        this.cancellation.cancel();
        this.receiver.close();
    }

    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = FlowResult<()>> + Send + '_>> {
        let this = self.get_mut();
        this.cancellation.cancel();
        this.receiver.close();
        let mut closed = this.closed.clone();
        let completion = this.completion.clone();
        Box::pin(async move {
            let cleanup_ok = closed
                .wait_for(Option::is_some)
                .await
                .map_err(|_| FlowError::Internal("JavaScript stream cleanup ended early".into()))?
                .as_ref()
                .is_some_and(|result| result.is_ok());
            let parked = completion.parked.lock().unwrap().take();
            if let Some(parked) = parked {
                let _ = parked.await;
            }
            if cleanup_ok {
                return Ok(());
            }
            // `None` means `poll_next` already delivered the error through `next()`.
            match completion.terminal_error.lock().unwrap().take() {
                Some(error) => Err(error),
                None => Ok(()),
            }
        })
    }
}

impl Drop for JsCallbackStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

fn closed_tsfn_error() -> FlowError {
    FlowError::Internal("PromiseAwareFn threadsafe function closed".into())
}

fn queue_status_result(status: napi::Status) -> FlowResult<()> {
    if status == napi::Status::Ok {
        Ok(())
    } else {
        Err(FlowError::Internal(format!(
            "failed to queue threadsafe function call: {status:?}",
        )))
    }
}

fn call_id_argument(ctx: &FunctionCallContext) -> napi::Result<u64> {
    ctx.get::<f64>(0).map(|call_id| call_id as u64)
}

fn call_settlers(env: &Env, calls: &Arc<CallRegistry>) -> napi::Result<PromiseSettlers> {
    let resolve_calls = Arc::clone(calls);
    let resolve = closure_function(env, "__nemo_relay_resolve", move |ctx| {
        resolve_calls.resolve(call_id_argument(&ctx)?, || {
            ctx.get::<Json>(1).map_err(|error| {
                FlowError::Internal(format!(
                    "JavaScript callback result could not be converted to JSON: {error}"
                ))
            })
        });
        Ok(())
    })?;
    let reject_calls = Arc::clone(calls);
    let reject = closure_function(env, "__nemo_relay_reject", move |ctx| {
        let call_id = call_id_argument(&ctx)?;
        let exception = ctx.get::<Unknown>(1)?;
        let message = ctx.get::<String>(2)?;
        let exception_type = ctx.get::<String>(3)?;
        reject_calls.fail(
            call_id,
            FlowError::CallbackException {
                message,
                exception_type,
                source: Some(Arc::new(napi::Error::from_unknown_without_coercion(
                    exception,
                ))),
            },
        );
        Ok(())
    })?;
    let push_calls = Arc::clone(calls);
    let stream_push = closure_function(env, "__nemo_relay_stream_push", move |ctx| {
        let call_id = call_id_argument(&ctx)?;
        let chunk = ctx.get::<Json>(1).map_err(|error| {
            Error::invalid_argument(format!("stream chunk is not JSON-compatible: {error}"))
        })?;
        push_stream_chunk(ctx.env, &push_calls, call_id, chunk)
    })?;
    let end_calls = Arc::clone(calls);
    let stream_end = closure_function(env, "__nemo_relay_stream_end", move |ctx| {
        end_calls.end_stream(call_id_argument(&ctx)?);
        Ok(())
    })?;
    let next = closure_function(env, "__nemo_relay_next", continue_call)?;
    Ok(FnArgs::from((
        detach(env, &resolve),
        detach(env, &reject),
        detach(env, &stream_push),
        detach(env, &stream_end),
        detach(env, &next),
    )))
}

fn push_stream_chunk(env: &Env, calls: &CallRegistry, call_id: u64, chunk: Json) -> Result<JsArg> {
    let Some(stream) = calls.stream(call_id) else {
        return Ok(js_value(env, ())?);
    };
    let Some(chunks) = stream.chunks.lock().unwrap().as_ref().cloned() else {
        return Ok(js_value(env, ())?);
    };
    match chunks.try_send(Ok(chunk)) {
        Ok(()) | Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => Ok(js_value(env, ())?),
        Err(tokio::sync::mpsc::error::TrySendError::Full(chunk)) => {
            let (pending, promise) = create_promise::<()>(env, Liveness::LetLoopExit)?;
            let task = within_runtime_if_available(|| {
                tokio::spawn(async move {
                    let _ = chunks.send(chunk).await;
                    pending.resolve_with(Box::new(|env| js_value(env, ())));
                })
            });
            *stream.parked.lock().unwrap() = Some(task);
            Ok(detach(env, &promise))
        }
    }
}

fn continue_call(ctx: FunctionCallContext) -> Result<JsArg> {
    let state: Unknown = ctx.get(0)?;
    // SAFETY: the argument belongs to this call, and the external was created by
    // `build_invoke_args` in this environment.
    let state = unsafe { External::<NextState>::from_napi_ref(ctx.env.raw(), state.raw()) }?;
    let arg = ctx.get::<Json>(1).unwrap_or(Json::Null);
    let scope = callback_factory::callback_scope_from(ctx.env, ctx.get::<Unknown>(2)?)?;
    let continuation = match &scope {
        Some(scope) => state
            .continuation_context
            .isolated_with_scope_stack(&scope.scope_stack),
        None => state.continuation_context.isolated(),
    }
    .map_err(Error::from)?;
    let publication_context_id = state.publication_context_id.clone();
    let promise = match &state.next {
        NextFn::Json(next) => {
            let next = next.clone();
            spawn_promise::<Json, _, _, _, _>(
                ctx.env,
                async move {
                    with_publication_callback_context(publication_context_id, None, async move {
                        continuation
                            .invoke(move || async move { next(arg).await.map_err(Error::from) })
                            .await
                    })
                    .await
                },
                js_value,
            )
        }
        NextFn::Stream(next) => {
            let next = next.clone();
            let stream_context_id = publication_context_id.clone();
            spawn_promise::<Json, _, Error, _, _>(
                ctx.env,
                async move {
                    let stream = with_publication_callback_context(
                        publication_context_id,
                        None,
                        async move {
                            continuation
                                .invoke(move || async move { next(arg).await.map_err(Error::from) })
                                .await
                        },
                    )
                    .await?;
                    Ok(crate::api::llm_stream_from_rust_stream(
                        stream,
                        stream_context_id,
                    ))
                },
                js_value,
            )
        }
    }?;
    Ok(detach(ctx.env, &promise))
}

fn build_promise_args(
    ctx: ThreadsafeCallContext<PromiseCall>,
    calls: &CallRegistry,
) -> napi::Result<PromiseArgs> {
    let env = &ctx.env;
    let undefined = js_value(env, ())?;
    match ctx.value {
        PromiseCall::Abort(call_id) => Ok(FnArgs::from((
            js_value(env, true)?,
            js_value(env, call_id as f64)?,
            undefined,
            undefined,
            undefined,
            undefined,
            undefined,
            undefined,
            undefined,
        ))),
        PromiseCall::Invoke(call) => {
            let CallArgs { invoke, completion } = *call;
            let call_id = invoke.call_id;
            match build_invoke_args(env, invoke, undefined) {
                Ok(args) => {
                    calls.insert(call_id, completion);
                    Ok(args)
                }
                Err(error) => {
                    completion.fail(FlowError::Internal(format!(
                        "failed to build JavaScript middleware callback arguments: {error}"
                    )));
                    Err(error)
                }
            }
        }
    }
}

fn build_invoke_args(env: &Env, call: InvokeArgs, undefined: JsArg) -> napi::Result<PromiseArgs> {
    let arg0 = match call.arg0 {
        PrimaryArg::Json(value) => js_value(env, value)?,
        PrimaryArg::Build(build) => build(env)?,
    };
    let next = match call.next {
        Some(state) => js_value(env, External::new(state))?,
        None => undefined,
    };
    let publication_context_id = match call.publication_context_id {
        Some(context_id) => js_value(env, context_id)?,
        None => undefined,
    };
    Ok(FnArgs::from((
        js_value(env, false)?,
        js_value(env, call.call_id as f64)?,
        arg0,
        js_value(env, call.spread)?,
        next,
        js_value(env, call.publication)?,
        publication_context_id,
        call.scope.into_js(env)?,
        js_value(env, call.stream_result)?,
    )))
}

/// A wrapper around a JS function that can be called from any thread and
/// transparently handles both synchronous and Promise return values.
pub struct PromiseAwareFn {
    tsfn: Mutex<Option<Arc<PromiseTsfn>>>,
    calls: Arc<CallRegistry>,
}

impl PromiseAwareFn {
    /// Create a new `PromiseAwareFn` wrapping the given JS function.
    ///
    /// Must be called on the JS main thread (i.e., in a sync `#[napi]` function).
    pub fn new(env: &Env, func: &Function<JsArg, JsArg>) -> napi::Result<Self> {
        let calls = Arc::new(CallRegistry::default());
        let settlers = call_settlers(env, &calls)?;
        let wrapper: Function<PromiseArgs, ()> =
            callback_factory::wrap_promise_callback(env, func, settlers)?;
        let registry = Arc::clone(&calls);
        // The callback must not keep the Node event loop alive.
        let tsfn = wrapper
            .build_threadsafe_function::<PromiseCall>()
            .callee_handled::<true>()
            .weak::<true>()
            .build_callback(move |ctx: ThreadsafeCallContext<PromiseCall>| {
                build_promise_args(ctx, &registry)
            })?;
        Ok(Self {
            tsfn: Mutex::new(Some(Arc::new(tsfn))),
            calls,
        })
    }

    /// Call the JS function with the given args and await the result.
    pub async fn call(&self, args: Json) -> FlowResult<Json> {
        self.call_inner(PrimaryArg::Json(args), CallMode::DIRECT, None)
            .await
    }

    /// Call a JavaScript callback with several JSON arguments.
    ///
    /// This retains the normal callback shape for middleware such as tool
    /// guardrails, whose public contract is `(name, payload)` rather than a
    /// single envelope object.
    pub async fn call_spread(&self, args: Vec<Json>) -> FlowResult<Json> {
        self.call_inner(PrimaryArg::Json(Json::Array(args)), CallMode::SPREAD, None)
            .await
    }

    /// Call a spread callback from queued event publication.
    pub async fn call_spread_for_publication(&self, args: Vec<Json>) -> FlowResult<Json> {
        self.call_inner(
            PrimaryArg::Json(Json::Array(args)),
            CallMode::SPREAD_PUBLICATION,
            None,
        )
        .await
    }

    /// Call the JS function with a builder-constructed first argument and await
    /// the result.
    ///
    /// The builder runs on the Node main thread, so it can construct values that
    /// cannot cross the threadsafe-function boundary as plain JSON, such as a
    /// `#[napi]` class instance.
    pub async fn call_with_arg0(&self, build_arg0: Arg0Builder) -> FlowResult<Json> {
        self.call_inner(PrimaryArg::Build(build_arg0), CallMode::DIRECT, None)
            .await
    }

    /// Call a JavaScript callback with builder-constructed spread arguments.
    pub async fn call_spread_with_arg0(&self, build_arg0: Arg0Builder) -> FlowResult<Json> {
        self.call_inner(PrimaryArg::Build(build_arg0), CallMode::SPREAD, None)
            .await
    }

    /// Call a spread callback from queued event publication.
    pub async fn call_spread_with_arg0_for_publication(
        &self,
        build_arg0: Arg0Builder,
    ) -> FlowResult<Json> {
        self.call_inner(
            PrimaryArg::Build(build_arg0),
            CallMode::SPREAD_PUBLICATION,
            None,
        )
        .await
    }

    /// Call the JS function with a middleware-style `next(arg)` callback that
    /// resolves to a JSON result.
    pub async fn call_with_json_next(&self, args: Json, next: JsonNextFn) -> FlowResult<Json> {
        self.call_inner(
            PrimaryArg::Json(args),
            CallMode::DIRECT,
            Some(NextFn::Json(next)),
        )
        .await
    }

    /// Call a spread JavaScript callback with builder-constructed arguments
    /// followed by a middleware-style `next(arg)` callback.
    pub async fn call_spread_with_arg0_and_json_next(
        &self,
        build_arg0: Arg0Builder,
        next: JsonNextFn,
    ) -> FlowResult<Json> {
        self.call_inner(
            PrimaryArg::Build(build_arg0),
            CallMode::SPREAD,
            Some(NextFn::Json(next)),
        )
        .await
    }

    /// Call a spread JavaScript callback with builder-constructed arguments
    /// followed by a middleware-style streaming `next(arg)` callback.
    pub async fn call_spread_with_arg0_and_stream_next(
        &self,
        build_arg0: Arg0Builder,
        next: JsonStreamNextFn,
    ) -> FlowResult<nemo_relay::api::runtime::LlmJsonStream> {
        self.call_with_stream_next_inner(PrimaryArg::Build(build_arg0), true, Some(next), None)
            .await
    }

    /// Call the JS function and expose the `AsyncIterable` it returns as a lazy stream.
    pub async fn call_stream(
        &self,
        args: Json,
        stream_cancellation: &StreamCancellation,
    ) -> FlowResult<nemo_relay::api::runtime::LlmJsonStream> {
        self.call_with_stream_next_inner(
            PrimaryArg::Json(args),
            false,
            None,
            Some(stream_cancellation),
        )
        .await
    }

    fn tsfn(&self) -> FlowResult<Arc<PromiseTsfn>> {
        self.tsfn
            .lock()
            .unwrap()
            .as_ref()
            .cloned()
            .ok_or_else(closed_tsfn_error)
    }

    fn next_state(&self, next: Option<NextFn>) -> Option<NextState> {
        next.map(|next| NextState {
            next,
            continuation_context: MiddlewareContinuationContext::capture(),
            publication_context_id: publication_callback_context_id(),
        })
    }

    async fn call_with_stream_next_inner(
        &self,
        arg0: PrimaryArg,
        spread: bool,
        next: Option<JsonStreamNextFn>,
        stream_cancellation: Option<&StreamCancellation>,
    ) -> FlowResult<nemo_relay::api::runtime::LlmJsonStream> {
        let tsfn = self.tsfn()?;
        let call_id = self.calls.allocate();
        let (ready_sender, ready_receiver) = tokio::sync::oneshot::channel();
        let (chunk_sender, chunk_receiver) =
            tokio::sync::mpsc::channel(crate::api::LLM_STREAM_BRIDGE_CAPACITY);
        let (closed, closed_rx) = tokio::sync::watch::channel(None);
        let completion = Arc::new(StreamCompletion {
            ready: Mutex::new(Some(ready_sender)),
            chunks: Mutex::new(Some(chunk_sender)),
            terminal_error: Mutex::new(None),
            closed,
            finished: Arc::default(),
            parked: Mutex::new(None),
        });
        let cancellation =
            CallCancellation::new(call_id, tsfn.clone(), Arc::clone(&completion.finished));
        if let Some(stream_cancellation) = stream_cancellation {
            stream_cancellation.attach(&cancellation);
        }
        let mut cancellation_guard = CallCancellationGuard::new(cancellation.clone());
        let next = self.next_state(next.map(NextFn::Stream));
        let status = tsfn.call(
            Ok(PromiseCall::Invoke(Box::new(CallArgs {
                invoke: InvokeArgs {
                    call_id,
                    arg0,
                    spread,
                    next,
                    publication: false,
                    publication_context_id: publication_callback_context_id(),
                    scope: CallbackScope::capture(),
                    stream_result: true,
                },
                completion: CallCompletion::Stream(Arc::clone(&completion)),
            }))),
            ThreadsafeFunctionCallMode::NonBlocking,
        );
        queue_status_result(status)?;
        ready_receiver
            .await
            .map_err(|e| FlowError::Internal(e.to_string()))??;
        cancellation_guard.disarm();
        Ok(nemo_relay::api::runtime::LlmJsonStream::from_closeable(
            JsCallbackStream {
                receiver: tokio_stream::wrappers::ReceiverStream::new(chunk_receiver),
                completion,
                closed: closed_rx,
                cancellation,
            },
        ))
    }

    /// Release the underlying threadsafe function so it does not outlive its registration.
    pub fn close(&self) -> napi::Result<()> {
        match self.tsfn.lock().unwrap().take() {
            Some(tsfn) => abort_tsfn(&tsfn),
            None => Ok(()),
        }
    }

    async fn call_inner(
        &self,
        arg0: PrimaryArg,
        mode: CallMode,
        next: Option<NextFn>,
    ) -> FlowResult<Json> {
        let tsfn = self.tsfn()?;
        let call_id = self.calls.allocate();
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let finished = Arc::new(AtomicBool::new(false));
        let mut cancellation_guard = CallCancellationGuard::new(CallCancellation::new(
            call_id,
            tsfn.clone(),
            Arc::clone(&finished),
        ));
        let next = self.next_state(next);
        let status = tsfn.call(
            Ok(PromiseCall::Invoke(Box::new(CallArgs {
                invoke: InvokeArgs {
                    call_id,
                    arg0,
                    spread: mode.spread,
                    next,
                    publication: mode.publication,
                    publication_context_id: publication_callback_context_id(),
                    // Queued tool and LLM sanitizers use the publication context to avoid
                    // waiting on their own publication.
                    scope: CallbackScope::capture(),
                    stream_result: false,
                },
                completion: CallCompletion::Json(JsonCompletion { sender, finished }),
            }))),
            ThreadsafeFunctionCallMode::NonBlocking,
        );
        queue_status_result(status)?;

        let result = receiver
            .await
            .map_err(|e| FlowError::Internal(e.to_string()))?;
        cancellation_guard.disarm();
        result
    }
}

#[cfg(test)]
#[path = "../tests/rust/promise_call_tests.rs"]
mod tests;
