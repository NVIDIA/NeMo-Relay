// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Promise-aware JS function calling for NeMo Relay NAPI bindings.
//!
//! This module wraps JS middleware callbacks so Rust can call them from any thread
//! and await either synchronous return values or Promise-returning callbacks.
//!
//! The previous implementation used a raw `napi_threadsafe_function` with a custom
//! `call_js_cb`. That path was prone to lifecycle issues under `node --test`.
//! This implementation keeps the same surface API but delegates the underlying
//! TSFN lifecycle to `napi-rs`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use napi::bindgen_prelude::ToNapiValue;
use napi::threadsafe_function::{ThreadSafeCallContext, ThreadsafeFunction};
use napi::{Env, JsFunction, JsUnknown, NapiRaw, NapiValue};
use serde_json::Value as Json;

use nemo_relay::api::runtime::subscriber_dispatcher::{
    PublicationBuffer, capture_nested_publication_buffer, with_task_nested_publication_buffer,
};
use nemo_relay::api::runtime::{
    LlmStreamInner, MiddlewareContinuationContext, ScopeStackHandle, capture_propagation_context,
    current_scope_stack,
};
use nemo_relay::error::{FlowError, Result as FlowResult};

use crate::callback_factory;
use crate::types::ScopeStack;
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

fn publication_callback_context_id() -> Option<String> {
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

/// Builds the first JS callback argument on the Node main thread.
///
/// Some callback arguments, such as `#[napi]` class instances, cannot cross the
/// threadsafe-function boundary as plain JSON. This builder runs inside the
/// threadsafe-function call (on the JS thread), so it can materialize those
/// values directly instead of serializing them.
pub type Arg0Builder = Box<dyn FnOnce(&Env) -> napi::Result<JsUnknown> + Send>;

/// The first argument passed to the wrapped JS callback.
enum PrimaryArg {
    /// A plain JSON value converted on the JS thread.
    Json(Json),
    /// A value materialized on the JS thread by a builder closure.
    Build(Arg0Builder),
}

struct CallArgs {
    arg0: PrimaryArg,
    spread: bool,
    next: Option<NextFn>,
    publication: bool,
    publication_context_id: Option<String>,
    /// Scope stack captured when Relay invokes the middleware.
    scope_stack: Option<ScopeStackHandle>,
    propagation_parent_uuid: String,
    publication_buffer: Option<PublicationBuffer>,
    stream_result: bool,
    continuation_context: Option<MiddlewareContinuationContext>,
    cancellation: CallCancellation,
    completion: CallCompletion,
}

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

enum CallCompletionSender {
    Json(tokio::sync::oneshot::Sender<FlowResult<Json>>),
    Stream(Arc<StreamCompletion>),
}

struct StreamCompletion {
    ready: std::sync::Mutex<Option<tokio::sync::oneshot::Sender<FlowResult<()>>>>,
    chunks: std::sync::Mutex<Option<tokio::sync::mpsc::Sender<FlowResult<Json>>>>,
    terminal_error: std::sync::Mutex<Option<FlowError>>,
    closed: tokio::sync::watch::Sender<Option<FlowResult<()>>>,
}

impl StreamCompletion {
    fn finish(&self, result: FlowResult<()>) {
        self.chunks.lock().unwrap().take();
        self.closed.send_replace(Some(result));
    }
}

#[derive(Clone)]
struct CallCompletion {
    sender: Arc<std::sync::Mutex<Option<CallCompletionSender>>>,
}

impl CallCompletion {
    fn json(sender: tokio::sync::oneshot::Sender<FlowResult<Json>>) -> Self {
        Self {
            sender: Arc::new(std::sync::Mutex::new(Some(CallCompletionSender::Json(
                sender,
            )))),
        }
    }

    fn stream(
        ready: tokio::sync::oneshot::Sender<FlowResult<()>>,
        chunks: tokio::sync::mpsc::Sender<FlowResult<Json>>,
    ) -> (
        Self,
        Arc<StreamCompletion>,
        tokio::sync::watch::Receiver<Option<FlowResult<()>>>,
    ) {
        let (closed, closed_rx) = tokio::sync::watch::channel(None);
        let stream = Arc::new(StreamCompletion {
            ready: std::sync::Mutex::new(Some(ready)),
            chunks: std::sync::Mutex::new(Some(chunks)),
            terminal_error: std::sync::Mutex::new(None),
            closed,
        });
        (
            Self {
                sender: Arc::new(std::sync::Mutex::new(Some(CallCompletionSender::Stream(
                    stream.clone(),
                )))),
            },
            stream,
            closed_rx,
        )
    }

    fn send_json(&self, value: FlowResult<Json>) {
        if let Some(CallCompletionSender::Json(sender)) = self.sender.lock().unwrap().take() {
            let _ = sender.send(value);
        }
    }

    fn send_error(&self, error: FlowError) {
        if let Some(sender) = self.sender.lock().unwrap().take() {
            match sender {
                CallCompletionSender::Json(sender) => {
                    let _ = sender.send(Err(error));
                }
                CallCompletionSender::Stream(sender) => {
                    if let Some(ready) = sender.ready.lock().unwrap().take() {
                        let _ = ready.send(Err(error.clone()));
                    } else {
                        *sender.terminal_error.lock().unwrap() = Some(error.clone());
                    }
                    sender.finish(Err(error));
                }
            }
        }
    }
}

#[derive(Clone, Default)]
struct CallCancellation {
    requested: Arc<AtomicBool>,
    abort: Arc<std::sync::Mutex<Option<ThreadsafeFunction<()>>>>,
}

impl CallCancellation {
    fn register(&self, env: &Env, abort: &JsFunction) -> napi::Result<()> {
        let mut abort = abort.create_threadsafe_function(0, |_ctx: ThreadSafeCallContext<()>| {
            Ok(Vec::<JsUnknown>::new())
        })?;
        abort.unref(env)?;
        let abort = {
            let mut registered = self.abort.lock().unwrap();
            *registered = Some(abort);
            registered.as_ref().cloned()
        };
        if self.requested.load(Ordering::Acquire) {
            Self::call_abort(abort);
        }
        Ok(())
    }

    fn cancel(&self) {
        self.requested.store(true, Ordering::Release);
        let abort = self.abort.lock().unwrap().as_ref().cloned();
        Self::call_abort(abort);
    }

    fn call_abort(abort: Option<ThreadsafeFunction<()>>) {
        if let Some(abort) = abort {
            let _ = abort.call(
                Ok(()),
                napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
            );
        }
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
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = FlowResult<()>> + Send + '_>> {
        let this = self.get_mut();
        this.cancellation.cancel();
        this.receiver.close();
        let mut closed = this.closed.clone();
        Box::pin(async move {
            while closed.borrow().is_none() {
                closed.changed().await.map_err(|_| {
                    FlowError::Internal("JavaScript stream cleanup ended early".into())
                })?;
            }
            closed.borrow().clone().expect("close state checked above")
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

fn json_to_unknown(env: &Env, value: Json) -> napi::Result<JsUnknown> {
    let raw = unsafe { <Json as ToNapiValue>::to_napi_value(env.raw(), value) }?;
    Ok(unsafe { JsUnknown::from_raw_unchecked(env.raw(), raw) })
}

fn function_to_unknown(env: &Env, value: &JsFunction) -> JsUnknown {
    unsafe { JsUnknown::from_raw_unchecked(env.raw(), value.raw()) }
}

fn undefined_to_unknown(env: &Env) -> napi::Result<JsUnknown> {
    let value = env.get_undefined()?;
    Ok(unsafe { JsUnknown::from_raw_unchecked(env.raw(), value.raw()) })
}

fn build_next_unknown(
    env: &Env,
    next: NextFn,
    continuation_context: MiddlewareContinuationContext,
    publication_context_id: Option<String>,
) -> napi::Result<JsUnknown> {
    let next_fn = match next {
        NextFn::Json(next) => {
            env.create_function_from_closure("__nemo_relay_next", move |ctx| {
                let arg = ctx.get::<Json>(0).unwrap_or(Json::Null);
                let next = next.clone();
                let scope_stack = match ctx.get::<&ScopeStack>(1) {
                    Ok(scope_stack) => Some(scope_stack.inner.clone()),
                    Err(_) => callback_factory::callback_scope_stack(ctx.env)?
                        .map(|(scope_stack, _)| scope_stack),
                };
                let continuation_context = match scope_stack {
                    Some(scope_stack) => {
                        continuation_context.isolated_with_scope_stack(&scope_stack)
                    }
                    None => continuation_context.isolated(),
                }
                .map_err(|error| napi::Error::from_reason(error.to_string()))?;
                let publication_context_id = publication_context_id.clone();
                ctx.env.execute_tokio_future(
                    async move {
                        with_publication_callback_context(
                            publication_context_id,
                            None,
                            async move {
                                continuation_context
                                    .invoke(move || async move {
                                        next(arg).await.map_err(|error| {
                                            napi::Error::from_reason(error.to_string())
                                        })
                                    })
                                    .await
                            },
                        )
                        .await
                    },
                    |_env, value| Ok(value),
                )
            })?
        }
        NextFn::Stream(next) => {
            env.create_function_from_closure("__nemo_relay_next", move |ctx| {
                let arg = ctx.get::<Json>(0).unwrap_or(Json::Null);
                let next = next.clone();
                let scope_stack = match ctx.get::<&ScopeStack>(1) {
                    Ok(scope_stack) => Some(scope_stack.inner.clone()),
                    Err(_) => callback_factory::callback_scope_stack(ctx.env)?
                        .map(|(scope_stack, _)| scope_stack),
                };
                let continuation_context = match scope_stack {
                    Some(scope_stack) => {
                        continuation_context.isolated_with_scope_stack(&scope_stack)
                    }
                    None => continuation_context.isolated(),
                }
                .map_err(|error| napi::Error::from_reason(error.to_string()))?;
                let publication_context_id = publication_context_id.clone();
                ctx.env.execute_tokio_future(
                    async move {
                        let stream = with_publication_callback_context(
                            publication_context_id,
                            None,
                            async move {
                                continuation_context
                                    .invoke(move || async move {
                                        next(arg).await.map_err(|error| {
                                            napi::Error::from_reason(error.to_string())
                                        })
                                    })
                                    .await
                            },
                        )
                        .await?;
                        Ok(crate::api::llm_stream_from_rust_stream(stream))
                    },
                    |_env, value| Ok(value),
                )
            })?
        }
    };

    Ok(function_to_unknown(env, &next_fn))
}

fn build_completion_unknowns(
    env: &Env,
    completion: CallCompletion,
) -> napi::Result<(JsUnknown, JsUnknown, JsUnknown, JsUnknown)> {
    let stream_completion = completion
        .sender
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|sender| match sender {
            CallCompletionSender::Stream(stream) => Some(stream.clone()),
            CallCompletionSender::Json(_) => None,
        });
    let resolve_completion = completion.clone();
    let resolve_stream = stream_completion.clone();
    let resolve = env.create_function_from_closure("__nemo_relay_resolve", move |ctx| {
        if let Some(stream) = &resolve_stream {
            if let Some(ready) = stream.ready.lock().unwrap().take() {
                let _ = ready.send(Ok(()));
            }
            return ctx.env.get_undefined();
        }
        let sender = resolve_completion.sender.lock().unwrap().take();
        if let Some(CallCompletionSender::Json(sender)) = sender {
            let value = ctx.get::<Json>(0).map_err(|error| {
                FlowError::Internal(format!(
                    "JavaScript callback result could not be converted to JSON: {error}"
                ))
            });
            let _ = sender.send(value);
        }
        ctx.env.get_undefined()
    })?;

    let reject = env.create_function_from_closure("__nemo_relay_reject", move |ctx| {
        // Do not invoke arbitrary `error.message` getters here. A throwing
        // getter used to escape this callback and abort the N-API call rather
        // than settling the middleware future as a rejection.
        let message = ctx
            .get::<String>(0)
            .unwrap_or_else(|_| "unknown error".to_string());
        let exception_type = ctx.get::<String>(1).unwrap_or_else(|_| "Error".to_string());
        completion.send_error(FlowError::CallbackException {
            message,
            exception_type,
        });
        ctx.env.get_undefined()
    })?;

    let push_stream = stream_completion.clone();
    let push = env.create_function_from_closure("__nemo_relay_stream_push", move |ctx| {
        let chunk = ctx.get::<Json>(0).map_err(|error| {
            napi::Error::from_reason(format!("stream chunk is not JSON-compatible: {error}"))
        })?;
        let chunks = push_stream
            .as_ref()
            .and_then(|stream| stream.chunks.lock().unwrap().as_ref().cloned());
        ctx.env.execute_tokio_future(
            async move {
                if let Some(chunks) = chunks {
                    let _ = chunks.send(Ok(chunk)).await;
                }
                Ok(())
            },
            |_env, ()| Ok(()),
        )
    })?;

    let end = env.create_function_from_closure("__nemo_relay_stream_end", move |ctx| {
        if let Some(stream) = &stream_completion {
            stream.finish(Ok(()));
        }
        ctx.env.get_undefined()
    })?;

    Ok((
        function_to_unknown(env, &resolve),
        function_to_unknown(env, &reject),
        function_to_unknown(env, &push),
        function_to_unknown(env, &end),
    ))
}

fn build_abort_registration_unknown(
    env: &Env,
    cancellation: CallCancellation,
) -> napi::Result<JsUnknown> {
    let register = env.create_function_from_closure("__nemo_relay_register_abort", move |ctx| {
        let abort = ctx.get::<JsFunction>(0)?;
        cancellation.register(ctx.env, &abort)?;
        ctx.env.get_undefined()
    })?;
    Ok(function_to_unknown(env, &register))
}

/// A wrapper around a JS function that can be called from any thread and
/// transparently handles both synchronous and Promise return values.
pub struct PromiseAwareFn {
    tsfn: std::sync::Mutex<Option<ThreadsafeFunction<CallArgs>>>,
}

impl PromiseAwareFn {
    /// Create a new `PromiseAwareFn` wrapping the given JS function.
    ///
    /// Must be called on the JS main thread (i.e., in a sync `#[napi]` function).
    pub fn new(env: &Env, func: &JsFunction) -> napi::Result<Self> {
        let wrapper = callback_factory::wrap_promise_callback(env, func)?;
        Self::from_wrapper(env, &wrapper)
    }

    fn from_wrapper(env: &Env, wrapper: &JsFunction) -> napi::Result<Self> {
        let mut tsfn =
            env.create_threadsafe_function(wrapper, 0, |ctx: ThreadSafeCallContext<CallArgs>| {
                let completion = ctx.value.completion.clone();
                let result = (|| {
                    let next = match ctx.value.next {
                        Some(next) => {
                            let continuation_context = ctx
                                .value
                                .continuation_context
                                .clone()
                                .ok_or_else(|| {
                                napi::Error::from_reason(
                                    "middleware next callback is missing its captured Relay context",
                                )
                            })?;
                            build_next_unknown(
                                &ctx.env,
                                next,
                                continuation_context,
                                ctx.value.publication_context_id.clone(),
                            )?
                        }
                        None => undefined_to_unknown(&ctx.env)?,
                    };
                    let arg0 = match ctx.value.arg0 {
                        PrimaryArg::Json(value) => json_to_unknown(&ctx.env, value)?,
                        PrimaryArg::Build(build) => build(&ctx.env)?,
                    };
                    let spread = unsafe {
                        JsUnknown::from_raw_unchecked(
                            ctx.env.raw(),
                            ctx.env.get_boolean(ctx.value.spread)?.raw(),
                        )
                    };
                    let publication = unsafe {
                        JsUnknown::from_raw_unchecked(
                            ctx.env.raw(),
                            ctx.env.get_boolean(ctx.value.publication)?.raw(),
                        )
                    };
                    let publication_context_id = match ctx.value.publication_context_id {
                        Some(context_id) => json_to_unknown(&ctx.env, Json::String(context_id))?,
                        None => undefined_to_unknown(&ctx.env)?,
                    };
                    let scope_stack = match ctx.value.scope_stack {
                        Some(scope_stack) => {
                            let scope_stack = ScopeStack {
                                inner: scope_stack,
                                publication_buffer: ctx.value.publication_buffer,
                            }
                            .into_instance(ctx.env)?;
                            unsafe {
                                JsUnknown::from_raw_unchecked(ctx.env.raw(), scope_stack.raw())
                            }
                        }
                        None => undefined_to_unknown(&ctx.env)?,
                    };
                    let propagation_parent_uuid = json_to_unknown(
                        &ctx.env,
                        Json::String(ctx.value.propagation_parent_uuid),
                    )?;
                    let (resolve, reject, stream_push, stream_end) =
                        build_completion_unknowns(&ctx.env, ctx.value.completion)?;
                    let register_abort =
                        build_abort_registration_unknown(&ctx.env, ctx.value.cancellation)?;
                    let stream_result = unsafe {
                        JsUnknown::from_raw_unchecked(
                            ctx.env.raw(),
                            ctx.env.get_boolean(ctx.value.stream_result)?.raw(),
                        )
                    };
                    Ok(vec![
                        arg0,
                        spread,
                        next,
                        resolve,
                        reject,
                        publication,
                        publication_context_id,
                        scope_stack,
                        propagation_parent_uuid,
                        register_abort,
                        stream_result,
                        stream_push,
                        stream_end,
                    ])
                })();
                if let Err(error) = &result {
                    completion.send_error(FlowError::Internal(format!(
                        "failed to build JavaScript middleware callback arguments: {error}"
                    )));
                }
                result
            })?;

        // The callback should not keep the Node event loop alive on its own.
        tsfn.unref(env)?;

        Ok(Self {
            tsfn: std::sync::Mutex::new(Some(tsfn)),
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

    /// Call the JS function with a middleware-style `next(arg)` callback that
    /// resolves to a lazy downstream stream.
    pub async fn call_with_stream_next(
        &self,
        args: Json,
        next: JsonStreamNextFn,
    ) -> FlowResult<nemo_relay::api::runtime::LlmJsonStream> {
        let (ready_sender, ready_receiver) = tokio::sync::oneshot::channel();
        let (chunk_sender, chunk_receiver) =
            tokio::sync::mpsc::channel(crate::api::LLM_STREAM_BRIDGE_CAPACITY);
        let (completion, stream_completion, closed) =
            CallCompletion::stream(ready_sender, chunk_sender);
        let cancellation = CallCancellation::default();
        let mut cancellation_guard = CallCancellationGuard::new(cancellation.clone());
        let continuation_context = MiddlewareContinuationContext::capture();
        let propagation_parent_uuid = capture_propagation_context()?.parent_uuid.to_string();
        let tsfn = self
            .tsfn
            .lock()
            .unwrap()
            .as_ref()
            .cloned()
            .ok_or_else(closed_tsfn_error)?;
        let status = tsfn.call(
            Ok(CallArgs {
                arg0: PrimaryArg::Json(args),
                spread: false,
                next: Some(NextFn::Stream(next)),
                publication: false,
                publication_context_id: publication_callback_context_id(),
                scope_stack: Some(current_scope_stack()),
                propagation_parent_uuid,
                publication_buffer: capture_nested_publication_buffer(),
                stream_result: true,
                continuation_context: Some(continuation_context),
                cancellation: cancellation.clone(),
                completion,
            }),
            napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
        );
        queue_status_result(status)?;
        ready_receiver
            .await
            .map_err(|e| FlowError::Internal(e.to_string()))??;
        cancellation_guard.disarm();
        Ok(nemo_relay::api::runtime::LlmJsonStream::from_closeable(
            JsCallbackStream {
                receiver: tokio_stream::wrappers::ReceiverStream::new(chunk_receiver),
                completion: stream_completion,
                closed,
                cancellation,
            },
        ))
    }

    /// Release the underlying threadsafe function so it does not outlive its registration.
    pub fn close(&self) {
        if let Some(tsfn) = self.tsfn.lock().unwrap().take() {
            let _ = tsfn.abort();
        }
    }

    async fn call_inner(
        &self,
        arg0: PrimaryArg,
        mode: CallMode,
        next: Option<NextFn>,
    ) -> FlowResult<Json> {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let cancellation = CallCancellation::default();
        let mut cancellation_guard = CallCancellationGuard::new(cancellation.clone());
        let continuation_context = next
            .as_ref()
            .map(|_| MiddlewareContinuationContext::capture());
        let propagation_parent_uuid = capture_propagation_context()?.parent_uuid.to_string();
        let tsfn = self
            .tsfn
            .lock()
            .unwrap()
            .as_ref()
            .cloned()
            .ok_or_else(closed_tsfn_error)?;
        let status = tsfn.call(
            Ok(CallArgs {
                arg0,
                spread: mode.spread,
                next,
                publication: mode.publication,
                publication_context_id: publication_callback_context_id(),
                // Scope identity applies to every middleware callback.
                // Publication context also lets queued tool/LLM observability
                // sanitizers avoid waiting on their own publication.
                scope_stack: Some(current_scope_stack()),
                propagation_parent_uuid,
                publication_buffer: capture_nested_publication_buffer(),
                stream_result: false,
                continuation_context,
                cancellation,
                completion: CallCompletion::json(sender),
            }),
            napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
        );
        queue_status_result(status)?;

        let result = receiver
            .await
            .map_err(|e| FlowError::Internal(e.to_string()))?;
        cancellation_guard.disarm();
        result
    }
}

impl Drop for PromiseAwareFn {
    fn drop(&mut self) {
        if let Some(tsfn) = self.tsfn.get_mut().unwrap().take() {
            let _ = tsfn.abort();
        }
    }
}

#[cfg(test)]
#[path = "../tests/rust/promise_call_tests.rs"]
mod tests;
