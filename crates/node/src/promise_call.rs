// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use napi::Env;
use napi::bindgen_prelude::{
    External, FnArgs, FromNapiRef, Function, FunctionCallContext, JsValue, PromiseRaw, Unknown,
    within_runtime_if_available,
};
use napi::threadsafe_function::{ThreadsafeCallContext, ThreadsafeFunctionCallMode};
use serde_json::Value as Json;

use nemo_relay_runtime::api::runtime::subscriber_dispatcher::{
    PublicationBuffer, with_nested_publication_buffer, with_task_nested_publication_buffer,
};
use nemo_relay_runtime::api::runtime::{
    LlmJsonStream, LlmStreamInner, MiddlewareContinuationContext,
};
use nemo_relay_types::error::{FlowError, Result as FlowResult};

use crate::callback_factory::{self, CallbackScope, Settlers};
use crate::convert::{
    BoxFuture, HandledTsfn, JsArg, JsBuilder, abort_tsfn, closure_function, detach, js_value, lock,
    report_queue_status, settled,
};
use crate::environment::{
    ENV_SHUTTING_DOWN, Liveness, create_promise, spawn_promise, teardown_flag,
};
use crate::error::{Error, Result};
use tokio_stream::Stream;

tokio::task_local! {
    static PUBLICATION_CALLBACK_CONTEXT_ID: Option<String>;
}

#[derive(Clone, Default)]
pub(crate) struct PublicationContext {
    pub(crate) context_id: Option<String>,
    pub(crate) publication_buffer: Option<PublicationBuffer>,
}

impl PublicationContext {
    pub(crate) async fn run<F: Future>(self, future: F) -> F::Output {
        with_task_nested_publication_buffer(
            self.publication_buffer,
            PUBLICATION_CALLBACK_CONTEXT_ID.scope(self.context_id, future),
        )
        .await
    }

    pub(crate) fn run_sync<T>(self, f: impl FnOnce() -> T) -> T {
        with_nested_publication_buffer(self.publication_buffer, || {
            PUBLICATION_CALLBACK_CONTEXT_ID.sync_scope(self.context_id, f)
        })
    }
}

pub(crate) fn publication_callback_context_id() -> Option<String> {
    PUBLICATION_CALLBACK_CONTEXT_ID
        .try_with(Clone::clone)
        .unwrap_or(None)
}

pub(crate) type JsonNextFn = Arc<dyn Fn(Json) -> BoxFuture<FlowResult<Json>> + Send + Sync>;
pub(crate) type JsonStreamNextFn =
    Arc<dyn Fn(Json) -> BoxFuture<FlowResult<LlmJsonStream>> + Send + Sync>;

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

enum PrimaryArg {
    Json(Json),
    Build(JsBuilder),
}

#[must_use]
pub(crate) struct Invocation {
    arg0: PrimaryArg,
    spread: bool,
    publication: bool,
}

impl Invocation {
    pub(crate) fn single(arg: Json) -> Self {
        Self {
            arg0: PrimaryArg::Json(arg),
            spread: false,
            publication: false,
        }
    }

    pub(crate) fn spread(args: Vec<Json>) -> Self {
        Self {
            arg0: PrimaryArg::Json(Json::Array(args)),
            spread: true,
            publication: false,
        }
    }

    pub(crate) fn built(build: impl FnOnce(&Env) -> napi::Result<JsArg> + Send + 'static) -> Self {
        Self {
            arg0: PrimaryArg::Build(Box::new(build)),
            spread: false,
            publication: false,
        }
    }

    pub(crate) fn spread_built(
        build: impl FnOnce(&Env) -> napi::Result<JsArg> + Send + 'static,
    ) -> Self {
        Self {
            arg0: PrimaryArg::Build(Box::new(build)),
            spread: true,
            publication: false,
        }
    }

    pub(crate) fn publication(self, publication: bool) -> Self {
        Self {
            publication,
            ..self
        }
    }
}

struct InvokeArgs {
    call_id: u64,
    invocation: Invocation,
    next: Option<NextState>,
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
        let ready = lock(&self.ready).take();
        if let Some(ready) = ready {
            let _ = ready.send(Ok(()));
        }
    }

    fn finish(&self, result: FlowResult<()>) {
        self.finished.store(true, Ordering::Release);
        lock(&self.chunks).take();
        self.closed.send_replace(Some(result));
    }

    fn fail(&self, error: FlowError) {
        let ready = lock(&self.ready).take();
        match ready {
            Some(ready) => {
                let _ = ready.send(Err(error.clone()));
            }
            None => *lock(&self.terminal_error) = Some(error.clone()),
        }
        self.finish(Err(error));
    }
}

enum CallCompletion {
    Json(JsonCompletion),
    Stream(Arc<StreamCompletion>),
}

impl CallCompletion {
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
    json: Mutex<HashMap<u64, JsonCompletion>>,
    streams: Mutex<HashMap<u64, Arc<StreamCompletion>>>,
}

impl CallRegistry {
    fn allocate(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn insert(&self, call_id: u64, completion: CallCompletion) {
        match completion {
            CallCompletion::Json(completion) => {
                lock(&self.json).insert(call_id, completion);
            }
            CallCompletion::Stream(stream) => {
                lock(&self.streams).insert(call_id, stream);
            }
        }
    }

    fn resolve(&self, call_id: u64, value: impl FnOnce() -> FlowResult<Json>) {
        let json = lock(&self.json).remove(&call_id);
        if let Some(completion) = json {
            completion.settle(value());
            return;
        }
        let stream = self.stream(call_id);
        if let Some(stream) = stream {
            stream.ready();
        }
    }

    fn fail(&self, call_id: u64, error: FlowError) {
        let json = lock(&self.json).remove(&call_id);
        if let Some(completion) = json {
            completion.settle(Err(error));
            return;
        }
        let stream = lock(&self.streams).remove(&call_id);
        if let Some(stream) = stream {
            stream.fail(error);
        }
    }

    fn stream(&self, call_id: u64) -> Option<Arc<StreamCompletion>> {
        lock(&self.streams).get(&call_id).cloned()
    }

    fn end_stream(&self, call_id: u64) {
        let stream = lock(&self.streams).remove(&call_id);
        if let Some(stream) = stream {
            stream.finish(Ok(()));
        }
    }

    fn cancel(&self, call_id: u64) {
        lock(&self.json).remove(&call_id);
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
        let status = self.tsfn.call(
            Ok(PromiseCall::Abort(self.call_id)),
            ThreadsafeFunctionCallMode::NonBlocking,
        );
        report_queue_status(status, "a JavaScript callback abort");
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
        let mut state = lock(&self.state);
        if state.requested {
            drop(state);
            call.cancel();
            return;
        }
        state.calls.push(call.clone());
    }

    pub(crate) fn cancel(&self) {
        let mut state = lock(&self.state);
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
            Poll::Ready(None) => Poll::Ready(lock(&self.completion.terminal_error).take().map(Err)),
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
        let closed = this.closed.clone();
        let completion = this.completion.clone();
        Box::pin(async move {
            let cleanup_ok = settled(closed)
                .await
                .ok_or_else(|| FlowError::Internal("JavaScript stream cleanup ended early".into()))?
                .is_ok();
            let parked = lock(&completion.parked).take();
            if let Some(parked) = parked {
                let _ = parked.await;
            }
            if cleanup_ok {
                return Ok(());
            }
            lock(&completion.terminal_error).take().map_or(Ok(()), Err)
        })
    }
}

impl Drop for JsCallbackStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
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

fn call_settlers(env: &Env, calls: &Arc<CallRegistry>) -> napi::Result<Settlers> {
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
    Ok(Settlers {
        resolve: detach(env, &resolve),
        reject: detach(env, &reject),
        stream_push: detach(env, &stream_push),
        stream_end: detach(env, &stream_end),
        next: detach(env, &next),
    })
}

fn push_stream_chunk(env: &Env, calls: &CallRegistry, call_id: u64, chunk: Json) -> Result<JsArg> {
    let Some(stream) = calls.stream(call_id) else {
        return Ok(js_value(env, ())?);
    };
    let chunks = lock(&stream.chunks).clone();
    let Some(chunks) = chunks else {
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
            *lock(&stream.parked) = Some(task);
            Ok(detach(env, &promise))
        }
    }
}

fn continue_call(ctx: FunctionCallContext) -> Result<JsArg> {
    let state: Unknown = ctx.get(0)?;
    // SAFETY: the argument belongs to this call, and the external was created by
    // `build_invoke_args` in this environment.
    let state = unsafe { External::<NextState>::from_napi_ref(ctx.env.raw(), state.raw()) }?;
    let arg = ctx.get::<Json>(1)?;
    let scope = callback_factory::callback_scope_from(ctx.env, ctx.get::<Unknown>(2)?)?;
    let continuation = match &scope {
        Some(scope) => state
            .continuation_context
            .isolated_with_scope_stack(&scope.scope_stack),
        None => state.continuation_context.isolated(),
    }
    .map_err(Error::from)?;
    let publication = PublicationContext {
        context_id: state.publication_context_id.clone(),
        publication_buffer: None,
    };
    let promise: PromiseRaw<'_, Json> = match &state.next {
        NextFn::Json(next) => {
            let next = next.clone();
            spawn_promise(
                ctx.env,
                publication.run(async move {
                    continuation
                        .invoke(move || async move { next(arg).await.map_err(Error::from) })
                        .await
                }),
                js_value,
            )
        }
        NextFn::Stream(next) => {
            let next = next.clone();
            let stream_publication = publication.clone();
            spawn_promise(
                ctx.env,
                async move {
                    let stream = publication
                        .run(async move {
                            continuation
                                .invoke(move || async move { next(arg).await.map_err(Error::from) })
                                .await
                        })
                        .await?;
                    Ok::<_, Error>(crate::api::llm_stream_from_rust_stream(
                        stream,
                        stream_publication,
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
    teardown: &AtomicBool,
) -> napi::Result<PromiseArgs> {
    let env = &ctx.env;
    let undefined = js_value(env, ())?;
    match ctx.value {
        PromiseCall::Abort(call_id) => {
            calls.cancel(call_id);
            Ok(FnArgs::from((
                js_value(env, true)?,
                js_value(env, call_id as f64)?,
                undefined,
                undefined,
                undefined,
                undefined,
                undefined,
                undefined,
                undefined,
            )))
        }
        PromiseCall::Invoke(call) => {
            let CallArgs { invoke, completion } = *call;
            if teardown.load(Ordering::Acquire) {
                completion.fail(FlowError::Internal(format!(
                    "JS middleware callback skipped: {ENV_SHUTTING_DOWN}"
                )));
                return Err(napi::Error::from_reason(ENV_SHUTTING_DOWN));
            }
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
    let arg0 = match call.invocation.arg0 {
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
        js_value(env, call.invocation.spread)?,
        next,
        js_value(env, call.invocation.publication)?,
        publication_context_id,
        call.scope.into_js(env)?,
        js_value(env, call.stream_result)?,
    )))
}

pub(crate) struct PromiseAwareFn {
    tsfn: Arc<PromiseTsfn>,
    calls: Arc<CallRegistry>,
    teardown: Arc<AtomicBool>,
}

impl PromiseAwareFn {
    pub(crate) fn new(env: &Env, func: &Function<JsArg, JsArg>) -> napi::Result<Self> {
        let calls = Arc::new(CallRegistry::default());
        let settlers = call_settlers(env, &calls)?;
        let wrapper: Function<PromiseArgs, ()> =
            callback_factory::wrap_promise_callback(env, func, settlers)?;
        let registry = Arc::clone(&calls);
        let teardown = teardown_flag(env)?;
        let dispatch_teardown = Arc::clone(&teardown);
        let tsfn = callback_factory::handled_tsfn(
            env,
            &wrapper,
            move |ctx: ThreadsafeCallContext<PromiseCall>| {
                build_promise_args(ctx, &registry, &dispatch_teardown)
            },
        )?;
        Ok(Self {
            tsfn: Arc::new(tsfn),
            calls,
            teardown,
        })
    }

    pub(crate) async fn call(&self, invocation: Invocation) -> FlowResult<Json> {
        self.call_inner(invocation, None).await
    }

    pub(crate) async fn call_with_next(
        &self,
        invocation: Invocation,
        next: JsonNextFn,
    ) -> FlowResult<Json> {
        self.call_inner(invocation, Some(NextFn::Json(next))).await
    }

    pub(crate) async fn call_stream(
        &self,
        invocation: Invocation,
        cancellation: &StreamCancellation,
    ) -> FlowResult<LlmJsonStream> {
        self.call_stream_inner(invocation, None, Some(cancellation))
            .await
    }

    pub(crate) async fn call_stream_with_next(
        &self,
        invocation: Invocation,
        next: JsonStreamNextFn,
    ) -> FlowResult<LlmJsonStream> {
        self.call_stream_inner(invocation, Some(next), None).await
    }

    pub(crate) fn abort_tsfn(&self) -> napi::Result<()> {
        abort_tsfn(&self.tsfn)
    }

    fn queue(
        &self,
        invocation: Invocation,
        next: Option<NextFn>,
        completion: CallCompletion,
    ) -> FlowResult<(u64, Arc<PromiseTsfn>)> {
        if self.teardown.load(Ordering::Acquire) {
            let skipped = || {
                FlowError::Internal(format!(
                    "JS middleware callback skipped: {ENV_SHUTTING_DOWN}"
                ))
            };
            completion.fail(skipped());
            return Err(skipped());
        }
        let tsfn = Arc::clone(&self.tsfn);
        let call_id = self.calls.allocate();
        let publication_context_id = publication_callback_context_id();
        let next = next.map(|next| NextState {
            next,
            continuation_context: MiddlewareContinuationContext::capture(),
            publication_context_id: publication_context_id.clone(),
        });
        let status = tsfn.call(
            Ok(PromiseCall::Invoke(Box::new(CallArgs {
                invoke: InvokeArgs {
                    call_id,
                    invocation,
                    next,
                    publication_context_id,
                    scope: CallbackScope::capture(),
                    stream_result: matches!(completion, CallCompletion::Stream(_)),
                },
                completion,
            }))),
            ThreadsafeFunctionCallMode::NonBlocking,
        );
        queue_status_result(status)?;
        Ok((call_id, tsfn))
    }

    async fn call_stream_inner(
        &self,
        invocation: Invocation,
        next: Option<JsonStreamNextFn>,
        stream_cancellation: Option<&StreamCancellation>,
    ) -> FlowResult<LlmJsonStream> {
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
        let (call_id, tsfn) = self.queue(
            invocation,
            next.map(NextFn::Stream),
            CallCompletion::Stream(Arc::clone(&completion)),
        )?;
        let cancellation = CallCancellation::new(call_id, tsfn, Arc::clone(&completion.finished));
        if let Some(stream_cancellation) = stream_cancellation {
            stream_cancellation.attach(&cancellation);
        }
        let mut cancellation_guard = CallCancellationGuard::new(cancellation.clone());
        ready_receiver
            .await
            .map_err(|e| FlowError::Internal(e.to_string()))??;
        cancellation_guard.disarm();
        Ok(LlmJsonStream::from_closeable(JsCallbackStream {
            receiver: tokio_stream::wrappers::ReceiverStream::new(chunk_receiver),
            completion,
            closed: closed_rx,
            cancellation,
        }))
    }

    async fn call_inner(&self, invocation: Invocation, next: Option<NextFn>) -> FlowResult<Json> {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let finished = Arc::new(AtomicBool::new(false));
        let (call_id, tsfn) = self.queue(
            invocation,
            next,
            CallCompletion::Json(JsonCompletion {
                sender,
                finished: Arc::clone(&finished),
            }),
        )?;
        let mut cancellation_guard =
            CallCancellationGuard::new(CallCancellation::new(call_id, tsfn, finished));

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
