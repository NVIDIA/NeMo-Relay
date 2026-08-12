// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Future-based typed middleware adapters for native ABI v4.

use std::ffi::c_void;
use std::future::Future;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::{FutureExt, Stream};
use serde::Deserialize;
use serde_json::{Map, Value as Json};
use tokio::runtime::{Handle, Runtime};
use tokio_util::task::TaskTracker;

use super::*;

const CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(10);
const CANCELLATION_POLL_MAX_INTERVAL: Duration = Duration::from_millis(160);

/// Configuration for the executor owned by one exported native plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeExecutorConfig {
    /// Number of Tokio worker threads dedicated to the plugin.
    pub worker_threads: usize,
}

impl Default for NativeExecutorConfig {
    fn default() -> Self {
        Self { worker_threads: 2 }
    }
}

impl NativeExecutorConfig {
    /// Applies an optional component-local executor override.
    ///
    /// Relay passes `[plugins.dynamic.config.executor]` as the `executor`
    /// object in `plugin_config`. The only supported setting is the positive
    /// integer `worker_threads`.
    pub fn with_component_config(mut self, plugin_config: &Map<String, Json>) -> Result<Self> {
        let Some(executor) = plugin_config.get("executor") else {
            return self.validate();
        };
        let executor = executor
            .as_object()
            .ok_or_else(|| "executor configuration must be an object".to_string())?;
        if let Some(worker_threads) = executor.get("worker_threads") {
            let worker_threads = worker_threads
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| "executor.worker_threads must be a positive integer".to_string())?;
            self.worker_threads = worker_threads;
        }
        self.validate()
    }

    fn validate(self) -> Result<Self> {
        if self.worker_threads == 0 {
            Err("executor.worker_threads must be greater than zero".into())
        } else {
            Ok(self)
        }
    }
}

pub(crate) struct NativeExecutor {
    config: NativeExecutorConfig,
    thread_name: String,
    runtime: Mutex<Option<Runtime>>,
    tracker: TaskTracker,
    accepting: AtomicBool,
}

impl NativeExecutor {
    pub(crate) fn new(config: NativeExecutorConfig, plugin_kind: &str) -> Arc<Self> {
        Arc::new(Self {
            config,
            thread_name: format!("nemo-relay-plugin-{plugin_kind}"),
            runtime: Mutex::new(None),
            tracker: TaskTracker::new(),
            accepting: AtomicBool::new(true),
        })
    }

    fn ensure_started(&self) -> Result<Handle> {
        if !self.accepting.load(Ordering::Acquire) {
            return Err("native plugin executor is shutting down".into());
        }
        let mut runtime = self
            .runtime
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if runtime.is_none() {
            let built = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(self.config.worker_threads)
                .thread_name(self.thread_name.clone())
                .enable_all()
                .build()
                .map_err(|error| format!("failed to start native plugin executor: {error}"))?;
            *runtime = Some(built);
        }
        Ok(runtime
            .as_ref()
            .expect("runtime was initialized")
            .handle()
            .clone())
    }

    fn spawn(&self, future: impl Future<Output = ()> + Send + 'static) -> Result<()> {
        let handle = self.ensure_started()?;
        self.tracker.spawn_on(future, &handle);
        Ok(())
    }
}

impl Drop for NativeExecutor {
    fn drop(&mut self) {
        self.accepting.store(false, Ordering::Release);
        self.tracker.close();
        let runtime = self
            .runtime
            .get_mut()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(runtime) = runtime {
            let tracker = self.tracker.clone();
            // Callback deregistration can run from a Relay Tokio worker. Drain
            // and destroy this separate runtime on an OS thread so teardown
            // never starts or drops a runtime from within another runtime. The
            // join keeps callback state, including the native-library guard,
            // alive until every accepted middleware task has finished.
            std::thread::Builder::new()
                .name(format!("{}-shutdown", self.thread_name))
                .spawn(move || runtime.block_on(tracker.wait()))
                .expect("native plugin executor shutdown thread should start")
                .join()
                .expect("native plugin executor shutdown thread should not panic");
        }
    }
}

#[derive(Clone, Copy)]
struct HostV4(NemoRelayNativeHostApiV4);

unsafe impl Send for HostV4 {}
unsafe impl Sync for HostV4 {}

struct Completion {
    host: HostV4,
    raw: *const NemoRelayNativeAsyncCompletion,
}

unsafe impl Send for Completion {}
unsafe impl Sync for Completion {}

impl Completion {
    fn resolve<T: Serialize>(&self, value: &T) -> Result<()> {
        let value = HostString::from_json(&self.host.0.v3.v1, value)
            .ok_or_else(|| "failed to serialize native async middleware result".to_string())?;
        let status =
            unsafe { (self.host.0.v3.async_completion_resolve_json)(self.raw, value.as_ptr()) };
        status_result(status, "resolve native async middleware completion")
    }

    fn reject(&self, message: &str) {
        if let Some(message) = HostString::new(&self.host.0.v3.v1, message) {
            unsafe {
                (self.host.0.v3.async_completion_reject)(self.raw, message.as_ptr());
            }
        }
    }

    fn is_cancelled(&self) -> bool {
        unsafe { (self.host.0.v3.async_completion_is_cancelled)(self.raw) }
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        unsafe { (self.host.0.v3.async_completion_release)(self.raw) };
    }
}

#[derive(Clone, Copy)]
struct CompletionRef {
    host: HostV4,
    raw: *const NemoRelayNativeAsyncCompletion,
}

unsafe impl Send for CompletionRef {}

impl CompletionRef {
    fn request_context(
        self,
        codec: LlmCodecIdentity,
    ) -> Result<LlmSanitizeRequestContext<'static>> {
        let resolved = if matches!(codec, LlmCodecIdentity::None) {
            None
        } else {
            let status = unsafe { (self.host.0.async_completion_retain)(self.raw) };
            status_result(status, "retain native async completion capability")?;
            Some(LlmSanitizeRequestCodec {
                async_host: self.host.0,
                completion: self.raw,
                completion_release: self.host.0.v3.async_completion_release,
                _lifetime: PhantomData,
            })
        };
        Ok(LlmSanitizeRequestContext { codec, resolved })
    }

    fn response_context(
        self,
        codec: LlmCodecIdentity,
    ) -> Result<LlmSanitizeResponseContext<'static>> {
        let resolved = if matches!(codec, LlmCodecIdentity::None) {
            None
        } else {
            let status = unsafe { (self.host.0.async_completion_retain)(self.raw) };
            status_result(status, "retain native async completion capability")?;
            Some(LlmSanitizeResponseCodec {
                async_host: self.host.0,
                completion: self.raw,
                completion_release: self.host.0.v3.async_completion_release,
                _lifetime: PhantomData,
            })
        };
        Ok(LlmSanitizeResponseContext { codec, resolved })
    }
}

struct NextInner {
    host: HostV4,
    raw: *const NemoRelayNativeAsyncNext,
}

unsafe impl Send for NextInner {}
unsafe impl Sync for NextInner {}

impl Drop for NextInner {
    fn drop(&mut self) {
        unsafe { (self.host.0.v3.async_next_release)(self.raw) };
    }
}

/// Cloneable asynchronous tool execution continuation.
#[derive(Clone)]
pub struct ToolNext(Arc<NextInner>);

impl ToolNext {
    /// Continues the tool chain with replacement arguments.
    pub async fn call(&self, args: Json) -> Result<Json> {
        invoke_unary_next(&self.0, &args).await
    }
}

/// Cloneable asynchronous LLM execution continuation.
#[derive(Clone)]
pub struct LlmNext(Arc<NextInner>);

impl LlmNext {
    /// Continues the LLM chain with a replacement request.
    pub async fn call(&self, request: LlmRequest) -> Result<Json> {
        invoke_unary_next(&self.0, &request).await
    }
}

/// Asynchronous JSON stream returned by a native typed stream interceptor.
pub type LlmJsonAsyncStream = Pin<Box<dyn Stream<Item = Result<Json>> + Send>>;

/// Cloneable asynchronous LLM stream continuation.
#[derive(Clone)]
pub struct LlmStreamNext(Arc<NextInner>);

impl LlmStreamNext {
    /// Opens an independent pull-based downstream stream.
    pub async fn call(&self, request: LlmRequest) -> Result<LlmJsonAsyncStream> {
        let request = HostString::from_json(&self.0.host.0.v3.v1, &request)
            .ok_or_else(|| "failed to serialize LLM stream request".to_string())?;
        let (sender, receiver) = futures::channel::oneshot::channel();
        let callback_state = Box::into_raw(Box::new(OpenState {
            sender,
            host: self.0.host,
        }));
        let status = unsafe {
            (self.0.host.0.async_next_open_llm_stream)(
                self.0.raw,
                request.as_ptr(),
                open_stream_callback,
                callback_state.cast(),
            )
        };
        if status != NemoRelayStatus::Ok {
            drop(unsafe { Box::from_raw(callback_state) });
            return Err(status_message(
                &self.0.host.0.v3.v1,
                status,
                "open LLM stream",
            ));
        }
        let mut opened = receiver
            .await
            .map_err(|_| "LLM stream open callback was dropped".to_string())??;
        let raw = opened.take();
        Ok(Box::pin(PullStream {
            host: self.0.host,
            raw,
            pending: None,
            finished: false,
        }))
    }
}

struct OpenedStream {
    host: HostV4,
    raw: *const NemoRelayNativeLlmAsyncStream,
}
unsafe impl Send for OpenedStream {}

impl OpenedStream {
    fn take(&mut self) -> *const NemoRelayNativeLlmAsyncStream {
        std::mem::replace(&mut self.raw, ptr::null())
    }
}

impl Drop for OpenedStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                (self.host.0.async_llm_stream_cancel)(self.raw);
                (self.host.0.async_llm_stream_release)(self.raw);
            }
        }
    }
}
type OpenSender = futures::channel::oneshot::Sender<Result<OpenedStream>>;
struct OpenState {
    sender: OpenSender,
    host: HostV4,
}

unsafe extern "C" fn open_stream_callback(
    user_data: *mut c_void,
    stream: *const NemoRelayNativeLlmAsyncStream,
    error: *const NemoRelayNativeString,
) {
    let state = unsafe { Box::from_raw(user_data.cast::<OpenState>()) };
    let result = if !stream.is_null() {
        Ok(OpenedStream {
            host: state.host,
            raw: stream,
        })
    } else if !error.is_null() {
        Err(read_host_string(&state.host.0.v3.v1, error)
            .unwrap_or_else(|_| "failed to open LLM stream".into()))
    } else {
        Err("host returned neither an LLM stream nor an error".into())
    };
    let _ = state.sender.send(result);
}

struct PullStream {
    host: HostV4,
    raw: *const NemoRelayNativeLlmAsyncStream,
    pending: Option<futures::channel::oneshot::Receiver<Result<Option<Json>>>>,
    finished: bool,
}

unsafe impl Send for PullStream {}

impl Stream for PullStream {
    type Item = Result<Json>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }
        if self.pending.is_none() {
            let (sender, receiver) = futures::channel::oneshot::channel();
            let callback_state = Box::into_raw(Box::new(PullState {
                sender,
                host: self.host,
            }));
            let status = unsafe {
                (self.host.0.async_llm_stream_pull)(
                    self.raw,
                    pull_stream_callback,
                    callback_state.cast(),
                )
            };
            if status != NemoRelayStatus::Ok {
                drop(unsafe { Box::from_raw(callback_state) });
                self.finished = true;
                return Poll::Ready(Some(Err(status_message(
                    &self.host.0.v3.v1,
                    status,
                    "pull LLM stream",
                ))));
            }
            self.pending = Some(receiver);
        }
        let pending = self
            .pending
            .as_mut()
            .expect("pull receiver was initialized");
        match Pin::new(pending).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                self.pending = None;
                match result {
                    Ok(Ok(Some(chunk))) => Poll::Ready(Some(Ok(chunk))),
                    Ok(Ok(None)) => {
                        self.finished = true;
                        Poll::Ready(None)
                    }
                    Ok(Err(error)) => {
                        self.finished = true;
                        Poll::Ready(Some(Err(error)))
                    }
                    Err(_) => {
                        self.finished = true;
                        Poll::Ready(Some(Err("LLM stream pull callback was dropped".into())))
                    }
                }
            }
        }
    }
}

impl Drop for PullStream {
    fn drop(&mut self) {
        if !self.finished {
            unsafe { (self.host.0.async_llm_stream_cancel)(self.raw) };
        }
        unsafe { (self.host.0.async_llm_stream_release)(self.raw) };
    }
}

type PullSender = futures::channel::oneshot::Sender<Result<Option<Json>>>;
struct PullState {
    sender: PullSender,
    host: HostV4,
}

unsafe extern "C" fn pull_stream_callback(
    user_data: *mut c_void,
    chunk_json: *const NemoRelayNativeString,
    error: *const NemoRelayNativeString,
    done: bool,
) {
    let state = unsafe { Box::from_raw(user_data.cast::<PullState>()) };
    let result = if !error.is_null() {
        Err(read_host_string(&state.host.0.v3.v1, error)
            .unwrap_or_else(|_| "LLM stream pull failed".into()))
    } else if done {
        Ok(None)
    } else if !chunk_json.is_null() {
        read_json_value(&state.host.0.v3.v1, chunk_json, "LLM stream chunk")
            .map_err(|status| format!("invalid LLM stream chunk: {status:?}"))
            .map(Some)
    } else {
        Err("host returned an invalid LLM stream pull result".into())
    };
    let _ = state.sender.send(result);
}

async fn invoke_unary_next<T: Serialize>(next: &NextInner, value: &T) -> Result<Json> {
    let value = HostString::from_json(&next.host.0.v3.v1, value)
        .ok_or_else(|| "failed to serialize native continuation input".to_string())?;
    let (sender, receiver) = futures::channel::oneshot::channel();
    let callback_state = Box::into_raw(Box::new(UnaryState {
        sender,
        host: next.host,
    }));
    let status = unsafe {
        (next.host.0.v3.async_next_invoke_result)(
            next.raw,
            value.as_ptr(),
            unary_next_callback,
            callback_state.cast(),
        )
    };
    if status != NemoRelayStatus::Ok {
        drop(unsafe { Box::from_raw(callback_state) });
        return Err(status_message(
            &next.host.0.v3.v1,
            status,
            "invoke native continuation",
        ));
    }
    receiver
        .await
        .map_err(|_| "native continuation callback was dropped".to_string())?
}

type UnarySender = futures::channel::oneshot::Sender<Result<Json>>;
struct UnaryState {
    sender: UnarySender,
    host: HostV4,
}

unsafe extern "C" fn unary_next_callback(
    user_data: *mut c_void,
    value_json: *const NemoRelayNativeString,
    error: *const NemoRelayNativeString,
) {
    let state = unsafe { Box::from_raw(user_data.cast::<UnaryState>()) };
    let result = if !error.is_null() {
        Err(read_host_string(&state.host.0.v3.v1, error)
            .unwrap_or_else(|_| "native continuation failed".into()))
    } else {
        read_json_value(
            &state.host.0.v3.v1,
            value_json,
            "native continuation result",
        )
        .map_err(|status| format!("invalid native continuation result: {status:?}"))
    };
    let _ = state.sender.send(result);
}

type UnaryFuture = Pin<Box<dyn Future<Output = Result<Json>> + Send>>;
type UnaryAdapter =
    dyn Fn(Json, Option<Arc<NextInner>>, CompletionRef) -> UnaryFuture + Send + Sync;

struct UnaryCallbackState {
    host: HostV4,
    executor: Arc<NativeExecutor>,
    adapter: Box<UnaryAdapter>,
}

unsafe extern "C" fn drop_unary_callback(user_data: *mut c_void) {
    if !user_data.is_null() {
        drop(unsafe { Box::from_raw(user_data.cast::<UnaryCallbackState>()) });
    }
}

unsafe extern "C" fn unary_trampoline(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    next: *const NemoRelayNativeAsyncNext,
    completion: *const NemoRelayNativeAsyncCompletion,
) -> u32 {
    let state = unsafe { &*user_data.cast::<UnaryCallbackState>() };
    let completion = Completion {
        host: state.host,
        raw: completion,
    };
    let completion_ref = CompletionRef {
        host: state.host,
        raw: completion.raw,
    };
    let next = (!next.is_null()).then(|| {
        Arc::new(NextInner {
            host: state.host,
            raw: next,
        })
    });
    let invocation = read_json_value(&state.host.0.v3.v1, invocation_json, "async invocation")
        .map_err(|status| format!("invalid async invocation: {status:?}"));
    let binding = ScopePollBinding::capture(state.host.0.v3.v1);
    let future = catch_unwind(AssertUnwindSafe(|| match invocation {
        Ok(invocation) => (state.adapter)(invocation, next, completion_ref),
        Err(error) => Box::pin(async move { Err(error) }) as UnaryFuture,
    }));
    if let Err(error) = state.executor.ensure_started() {
        completion.reject(&error);
        set_last_error(&state.host.0.v3.v1, &error);
        return NemoRelayNativeAsyncCallbackState::Pending as u32;
    }
    let task = match future {
        Ok(future) => drive_unary(future, binding, completion),
        Err(_) => drive_unary(
            Box::pin(async move { Err("typed native middleware callback panicked".into()) }),
            binding,
            completion,
        ),
    };
    if let Err(error) = state.executor.spawn(async move {
        let _ = task.await;
    }) {
        // A stopped executor cannot retain plugin code. The callback-owned
        // handles have already been reclaimed by dropping `task`.
        set_last_error(&state.host.0.v3.v1, &error);
    }
    NemoRelayNativeAsyncCallbackState::Pending as u32
}

fn drive_unary(
    future: UnaryFuture,
    binding: Result<ScopePollBinding>,
    completion: Completion,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
    Box::pin(async move {
        let future: UnaryFuture = match binding {
            Ok(binding) => Box::pin(ScopedFuture::new(future, binding)),
            Err(error) => {
                completion.reject(&error);
                return;
            }
        };
        let result = tokio::select! {
            result = AssertUnwindSafe(future).catch_unwind() => {
                result.unwrap_or_else(|_| Err("typed native middleware future panicked".into()))
            }
            () = wait_for_completion_cancellation(&completion) => return,
        };
        match result {
            Ok(value) => {
                if let Err(error) = completion.resolve(&value) {
                    completion.reject(&error);
                }
            }
            Err(error) => completion.reject(&error),
        }
    })
}

async fn wait_for_completion_cancellation(completion: &Completion) {
    let mut delay = CANCELLATION_POLL_INTERVAL;
    loop {
        tokio::time::sleep(delay).await;
        if completion.is_cancelled() {
            return;
        }
        delay = delay.saturating_mul(2).min(CANCELLATION_POLL_MAX_INTERVAL);
    }
}

struct ScopePollBinding {
    host: NemoRelayNativeHostApiV1,
    captured: *mut NemoRelayNativeScopeStackBinding,
}

unsafe impl Send for ScopePollBinding {}

impl ScopePollBinding {
    fn capture(host: NemoRelayNativeHostApiV1) -> Result<Self> {
        let mut captured = ptr::null_mut();
        let status = unsafe { (host.scope_stack_capture_thread)(&mut captured) };
        if status == NemoRelayStatus::Ok && !captured.is_null() {
            Ok(Self { host, captured })
        } else {
            Err(format!(
                "failed to capture native callback scope stack: {status:?}"
            ))
        }
    }

    fn enter(&mut self) -> Result<*mut NemoRelayNativeScopeStackBinding> {
        let mut previous = ptr::null_mut();
        let status = unsafe { (self.host.scope_stack_capture_thread)(&mut previous) };
        if status != NemoRelayStatus::Ok || previous.is_null() {
            return Err(format!(
                "failed to capture executor scope stack: {status:?}"
            ));
        }
        let captured = std::mem::replace(&mut self.captured, ptr::null_mut());
        let status = unsafe { (self.host.scope_stack_restore_thread)(captured) };
        if status != NemoRelayStatus::Ok {
            unsafe { (self.host.scope_stack_binding_free)(previous) };
            return Err(format!(
                "failed to install callback scope stack: {status:?}"
            ));
        }
        Ok(previous)
    }

    fn exit(&mut self, previous: *mut NemoRelayNativeScopeStackBinding) -> Result<()> {
        let status = unsafe { (self.host.scope_stack_capture_thread)(&mut self.captured) };
        if status != NemoRelayStatus::Ok || self.captured.is_null() {
            let restore_status = unsafe { (self.host.scope_stack_restore_thread)(previous) };
            if restore_status != NemoRelayStatus::Ok {
                unsafe { (self.host.scope_stack_binding_free)(previous) };
                return Err(format!(
                    "failed to recapture callback scope stack: {status:?}; failed to restore executor scope stack: {restore_status:?}"
                ));
            }
            return Err(format!(
                "failed to recapture callback scope stack: {status:?}"
            ));
        }
        let status = unsafe { (self.host.scope_stack_restore_thread)(previous) };
        status_result(status, "restore executor scope stack")
    }
}

struct ScopePollRestore<'a> {
    binding: &'a mut ScopePollBinding,
    previous: Option<*mut NemoRelayNativeScopeStackBinding>,
}

impl<'a> ScopePollRestore<'a> {
    fn new(
        binding: &'a mut ScopePollBinding,
        previous: *mut NemoRelayNativeScopeStackBinding,
    ) -> Self {
        Self {
            binding,
            previous: Some(previous),
        }
    }

    fn restore(&mut self) -> Result<()> {
        let previous = self.previous.take().expect("scope binding restored once");
        self.binding.exit(previous)
    }
}

impl Drop for ScopePollRestore<'_> {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            // Do not let a panic from the wrapped future leave its scope stack
            // on an SDK executor worker. Drop cannot report a restoration
            // failure, but `exit` still restores the previous binding first.
            let _ = self.binding.exit(previous);
        }
    }
}

impl Drop for ScopePollBinding {
    fn drop(&mut self) {
        if !self.captured.is_null() {
            unsafe { (self.host.scope_stack_binding_free)(self.captured) };
        }
    }
}

struct ScopedFuture<F> {
    future: F,
    binding: ScopePollBinding,
}

impl<F> ScopedFuture<F> {
    fn new(future: F, binding: ScopePollBinding) -> Self {
        Self { future, binding }
    }
}

impl<F: Future> Future for ScopedFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: neither field moves while `self` is pinned.
        let this = unsafe { self.get_unchecked_mut() };
        let previous = this
            .binding
            .enter()
            .unwrap_or_else(|error| panic!("{error}"));
        let mut restore = ScopePollRestore::new(&mut this.binding, previous);
        let result = unsafe { Pin::new_unchecked(&mut this.future) }.poll(cx);
        restore.restore().unwrap_or_else(|error| panic!("{error}"));
        result
    }
}

struct ScopedStream<S> {
    stream: S,
    binding: ScopePollBinding,
}

impl<S> ScopedStream<S> {
    fn new(stream: S, binding: ScopePollBinding) -> Self {
        Self { stream, binding }
    }
}

impl<S: Stream> Stream for ScopedStream<S> {
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // SAFETY: neither field moves while `self` is pinned.
        let this = unsafe { self.get_unchecked_mut() };
        let previous = this
            .binding
            .enter()
            .unwrap_or_else(|error| panic!("{error}"));
        let mut restore = ScopePollRestore::new(&mut this.binding, previous);
        let result = unsafe { Pin::new_unchecked(&mut this.stream) }.poll_next(cx);
        restore.restore().unwrap_or_else(|error| panic!("{error}"));
        result
    }
}

#[derive(Deserialize)]
struct NameValueInvocation {
    name: String,
    value: Json,
}

#[derive(Deserialize)]
struct RequestInvocation {
    request: LlmRequest,
}

#[derive(Deserialize)]
struct NameRequestInvocation {
    name: String,
    request: LlmRequest,
}

#[derive(Deserialize)]
struct LlmRequestInterceptInvocation {
    name: String,
    request: LlmRequest,
    annotated: Option<AnnotatedLlmRequest>,
}

#[derive(Deserialize)]
struct EventInvocation {
    event: Event,
    fields: EventSanitizeFields,
}

type StreamFuture = Pin<Box<dyn Future<Output = Result<LlmJsonAsyncStream>> + Send>>;
type StreamAdapter = dyn Fn(Json, LlmStreamNext) -> StreamFuture + Send + Sync;

struct StreamCallbackState {
    host: HostV4,
    executor: Arc<NativeExecutor>,
    adapter: Box<StreamAdapter>,
}

unsafe extern "C" fn drop_stream_callback(user_data: *mut c_void) {
    if !user_data.is_null() {
        drop(unsafe { Box::from_raw(user_data.cast::<StreamCallbackState>()) });
    }
}

struct OutputStream {
    host: HostV4,
    raw: *const NemoRelayNativeAsyncStream,
}

unsafe impl Send for OutputStream {}
unsafe impl Sync for OutputStream {}

impl OutputStream {
    fn cancelled(&self) -> bool {
        unsafe { (self.host.0.v3.async_stream_is_cancelled)(self.raw) }
    }

    async fn push(&self, value: &Json) -> Result<()> {
        let value = HostString::from_json(&self.host.0.v3.v1, value)
            .ok_or_else(|| "failed to serialize native stream chunk".to_string())?;
        loop {
            if self.cancelled() {
                return Err("native stream consumer cancelled".into());
            }
            let status =
                unsafe { (self.host.0.v3.async_stream_push_json)(self.raw, value.as_ptr()) };
            match status {
                NemoRelayStatus::Ok => return Ok(()),
                NemoRelayStatus::Backpressured => {
                    tokio::time::sleep(CANCELLATION_POLL_INTERVAL).await;
                }
                status => return Err(format!("push native stream chunk failed: {status:?}")),
            }
        }
    }

    fn finish(&self) -> Result<()> {
        status_result(
            unsafe { (self.host.0.v3.async_stream_finish)(self.raw) },
            "finish native stream",
        )
    }

    async fn reject(&self, error: &str) {
        if let Some(error) = HostString::new(&self.host.0.v3.v1, error) {
            loop {
                if self.cancelled() {
                    break;
                }
                let status =
                    unsafe { (self.host.0.v3.async_stream_reject)(self.raw, error.as_ptr()) };
                match status {
                    NemoRelayStatus::Backpressured => {
                        tokio::time::sleep(CANCELLATION_POLL_INTERVAL).await;
                    }
                    _ => break,
                }
            }
        }
    }

    fn reject_once(&self, error: &str) {
        if let Some(error) = HostString::new(&self.host.0.v3.v1, error) {
            unsafe {
                (self.host.0.v3.async_stream_reject)(self.raw, error.as_ptr());
            }
        }
    }
}

impl Drop for OutputStream {
    fn drop(&mut self) {
        unsafe { (self.host.0.v3.async_stream_release)(self.raw) };
    }
}

unsafe extern "C" fn stream_trampoline(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    next: *const NemoRelayNativeAsyncNext,
    stream: *const NemoRelayNativeAsyncStream,
) -> u32 {
    let state = unsafe { &*user_data.cast::<StreamCallbackState>() };
    let output = OutputStream {
        host: state.host,
        raw: stream,
    };
    if next.is_null() {
        output.reject_once("native stream middleware requires a continuation");
        return NemoRelayNativeAsyncCallbackState::Pending as u32;
    }
    let next = LlmStreamNext(Arc::new(NextInner {
        host: state.host,
        raw: next,
    }));
    let invocation = read_json_value(&state.host.0.v3.v1, invocation_json, "stream invocation")
        .map_err(|status| format!("invalid native stream invocation: {status:?}"));
    let bindings = ScopePollBinding::capture(state.host.0.v3.v1).and_then(|future| {
        ScopePollBinding::capture(state.host.0.v3.v1).map(|stream| (future, stream))
    });
    let future = catch_unwind(AssertUnwindSafe(|| match invocation {
        Ok(invocation) => (state.adapter)(invocation, next),
        Err(error) => Box::pin(async move { Err(error) }) as StreamFuture,
    }));
    let future = future.unwrap_or_else(|_| {
        Box::pin(async move { Err("typed native stream callback panicked".into()) })
    });
    if let Err(error) = state.executor.ensure_started() {
        output.reject_once(&error);
        set_last_error(&state.host.0.v3.v1, &error);
        return NemoRelayNativeAsyncCallbackState::Pending as u32;
    }
    let task = async move {
        let (future_binding, stream_binding) = match bindings {
            Ok(bindings) => bindings,
            Err(error) => {
                output.reject(&error).await;
                return;
            }
        };
        let future: StreamFuture = Box::pin(ScopedFuture::new(future, future_binding));
        let stream = tokio::select! {
            result = AssertUnwindSafe(future).catch_unwind() => match result {
                Ok(result) => result,
                Err(_) => Err("typed native stream future panicked".into()),
            },
            () = wait_for_stream_cancellation(&output) => return,
        };
        let mut stream = match stream {
            Ok(stream) => ScopedStream::new(stream, stream_binding),
            Err(error) => {
                output.reject(&error).await;
                return;
            }
        };
        loop {
            let item = tokio::select! {
                result = AssertUnwindSafe(futures::StreamExt::next(&mut stream)).catch_unwind() => {
                    match result {
                        Ok(item) => item,
                        Err(_) => {
                            output.reject("typed native stream panicked while polling").await;
                            return;
                        }
                    }
                },
                () = wait_for_stream_cancellation(&output) => return,
            };
            let Some(item) = item else {
                break;
            };
            match item {
                Ok(chunk) => {
                    if let Err(error) = output.push(&chunk).await {
                        if !output.cancelled() {
                            output.reject(&error).await;
                        }
                        return;
                    }
                }
                Err(error) => {
                    output.reject(&error).await;
                    return;
                }
            }
        }
        if !output.cancelled() {
            let _ = output.finish();
        }
    };
    if let Err(error) = state.executor.spawn(task) {
        set_last_error(&state.host.0.v3.v1, &error);
    }
    NemoRelayNativeAsyncCallbackState::Pending as u32
}

async fn wait_for_stream_cancellation(output: &OutputStream) {
    let mut delay = CANCELLATION_POLL_INTERVAL;
    loop {
        tokio::time::sleep(delay).await;
        if output.cancelled() {
            return;
        }
        delay = delay.saturating_mul(2).min(CANCELLATION_POLL_MAX_INTERVAL);
    }
}

#[derive(Deserialize)]
struct CodecInvocation<T> {
    #[serde(flatten)]
    payload: T,
    context: CodecIdentityInvocation,
}

#[derive(Deserialize)]
struct CodecIdentityInvocation {
    codec_kind: String,
    codec_id: Option<String>,
}

impl CodecIdentityInvocation {
    fn identity(self) -> Result<LlmCodecIdentity> {
        match (self.codec_kind.as_str(), self.codec_id) {
            ("none", _) => Ok(LlmCodecIdentity::None),
            ("opaque", _) => Ok(LlmCodecIdentity::Opaque),
            ("builtin", Some(id)) => match id.as_str() {
                "openai_chat" => Ok(LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)),
                "openai_responses" => {
                    Ok(LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiResponses))
                }
                "anthropic_messages" => Ok(LlmCodecIdentity::BuiltIn(
                    BuiltinLlmCodec::AnthropicMessages,
                )),
                "oci_genai" => Ok(LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OCIGenAI)),
                "gemini_generate_content" => Ok(LlmCodecIdentity::BuiltIn(
                    BuiltinLlmCodec::GeminiGenerateContent,
                )),
                _ => Err(format!("unknown built-in LLM codec: {id}")),
            },
            ("runtime", Some(id)) => Ok(LlmCodecIdentity::Runtime(id)),
            (kind, _) => Err(format!("invalid LLM codec context: {kind}")),
        }
    }
}

impl PluginContext<'_> {
    fn host_v4(&self) -> Result<HostV4> {
        if self.host.abi_version < NEMO_RELAY_NATIVE_ABI_VERSION_TYPED_ASYNC
            || self.host.struct_size < std::mem::size_of::<NemoRelayNativeHostApiV4>()
        {
            return Err("typed async native middleware requires Relay ABI v4".into());
        }
        Ok(HostV4(unsafe {
            *(self.host as *const _ as *const NemoRelayNativeHostApiV4)
        }))
    }

    fn register_unary_adapter(
        &mut self,
        kind: NemoRelayNativeAsyncMiddlewareKind,
        name: &str,
        priority: i32,
        break_chain: bool,
        adapter: Box<UnaryAdapter>,
    ) -> Result<()> {
        let state = Box::into_raw(Box::new(UnaryCallbackState {
            host: self.host_v4()?,
            executor: Arc::clone(&self.executor),
            adapter,
        }));
        let status = unsafe {
            self.register_async_middleware_raw(
                kind,
                name,
                priority,
                break_chain,
                unary_trampoline,
                state.cast(),
                Some(drop_unary_callback),
            )
        };
        if status == NemoRelayStatus::Ok {
            Ok(())
        } else {
            Err(status_message(
                self.host,
                status,
                registration_operation(kind),
            ))
        }
    }

    fn register_event_adapter<F, Fut>(
        &mut self,
        kind: NemoRelayNativeAsyncMiddlewareKind,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(Arc<Event>, EventSanitizeFields) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<EventSanitizeFields>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.register_unary_adapter(
            kind,
            name,
            priority,
            false,
            Box::new(move |value, _, _| {
                let callback = Arc::clone(&callback);
                Box::pin(async move {
                    let invocation: EventInvocation = serde_json::from_value(value)
                        .map_err(|error| format!("invalid event sanitizer invocation: {error}"))?;
                    serde_json::to_value(
                        callback(Arc::new(invocation.event), invocation.fields).await?,
                    )
                    .map_err(|error| error.to_string())
                })
            }),
        )
    }

    /// Registers an asynchronous mark-event sanitizer.
    pub fn register_mark_sanitize_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(Arc<Event>, EventSanitizeFields) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<EventSanitizeFields>> + Send + 'static,
    {
        self.register_event_adapter(
            NemoRelayNativeAsyncMiddlewareKind::MarkSanitize,
            name,
            priority,
            callback,
        )
    }

    /// Registers an asynchronous scope-start sanitizer.
    pub fn register_scope_sanitize_start_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(Arc<Event>, EventSanitizeFields) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<EventSanitizeFields>> + Send + 'static,
    {
        self.register_event_adapter(
            NemoRelayNativeAsyncMiddlewareKind::ScopeSanitizeStart,
            name,
            priority,
            callback,
        )
    }

    /// Registers an asynchronous scope-end sanitizer.
    pub fn register_scope_sanitize_end_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(Arc<Event>, EventSanitizeFields) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<EventSanitizeFields>> + Send + 'static,
    {
        self.register_event_adapter(
            NemoRelayNativeAsyncMiddlewareKind::ScopeSanitizeEnd,
            name,
            priority,
            callback,
        )
    }

    fn register_tool_json_adapter<F, Fut>(
        &mut self,
        kind: NemoRelayNativeAsyncMiddlewareKind,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, Json) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Json>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.register_unary_adapter(
            kind,
            name,
            priority,
            break_chain,
            Box::new(move |value, _, _| {
                let callback = Arc::clone(&callback);
                Box::pin(async move {
                    let invocation: NameValueInvocation =
                        serde_json::from_value(value).map_err(|error| error.to_string())?;
                    callback(invocation.name, invocation.value).await
                })
            }),
        )
    }

    /// Registers an asynchronous tool request sanitizer.
    pub fn register_tool_sanitize_request_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, Json) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Json>> + Send + 'static,
    {
        self.register_tool_json_adapter(
            NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeRequest,
            name,
            priority,
            false,
            callback,
        )
    }

    /// Registers an asynchronous tool response sanitizer.
    pub fn register_tool_sanitize_response_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, Json) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Json>> + Send + 'static,
    {
        self.register_tool_json_adapter(
            NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeResponse,
            name,
            priority,
            false,
            callback,
        )
    }

    /// Registers an asynchronous tool conditional-execution guardrail.
    pub fn register_tool_conditional_execution_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, Json) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<String>>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.register_unary_adapter(
            NemoRelayNativeAsyncMiddlewareKind::ToolConditionalExecution,
            name,
            priority,
            false,
            Box::new(move |value, _, _| {
                let callback = Arc::clone(&callback);
                Box::pin(async move {
                    let invocation: NameValueInvocation =
                        serde_json::from_value(value).map_err(|error| error.to_string())?;
                    serde_json::to_value(callback(invocation.name, invocation.value).await?)
                        .map_err(|error| error.to_string())
                })
            }),
        )
    }

    /// Registers an asynchronous tool request intercept.
    pub fn register_tool_request_intercept<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, Json) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Json>> + Send + 'static,
    {
        self.register_tool_json_adapter(
            NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept,
            name,
            priority,
            break_chain,
            callback,
        )
    }

    /// Registers an asynchronous tool execution intercept.
    pub fn register_tool_execution_intercept<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, Json, ToolNext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolExecutionInterceptOutcome>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.register_unary_adapter(
            NemoRelayNativeAsyncMiddlewareKind::ToolExecutionIntercept,
            name,
            priority,
            false,
            Box::new(move |value, next, _| {
                let callback = Arc::clone(&callback);
                Box::pin(async move {
                    let invocation: NameValueInvocation =
                        serde_json::from_value(value).map_err(|error| error.to_string())?;
                    let next = ToolNext(
                        next.ok_or_else(|| "tool execution continuation was null".to_string())?,
                    );
                    serde_json::to_value(callback(invocation.name, invocation.value, next).await?)
                        .map_err(|error| error.to_string())
                })
            }),
        )
    }

    /// Registers an asynchronous LLM request sanitizer.
    pub fn register_llm_sanitize_request_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(LlmRequest, LlmSanitizeRequestContext<'static>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<LlmRequest>>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.register_unary_adapter(
            NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeRequest,
            name,
            priority,
            false,
            Box::new(move |value, _, completion| {
                let callback = Arc::clone(&callback);
                Box::pin(async move {
                    #[derive(Deserialize)]
                    struct Payload {
                        request: LlmRequest,
                    }
                    let invocation: CodecInvocation<Payload> =
                        serde_json::from_value(value).map_err(|error| error.to_string())?;
                    let codec = invocation.context.identity()?;
                    let context = completion.request_context(codec)?;
                    serde_json::to_value(callback(invocation.payload.request, context).await?)
                        .map_err(|error| error.to_string())
                })
            }),
        )
    }

    /// Registers an asynchronous LLM response sanitizer.
    pub fn register_llm_sanitize_response_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(Json, LlmSanitizeResponseContext<'static>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<Json>>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.register_unary_adapter(
            NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeResponse,
            name,
            priority,
            false,
            Box::new(move |value, _, completion| {
                let callback = Arc::clone(&callback);
                Box::pin(async move {
                    #[derive(Deserialize)]
                    struct Payload {
                        response: Json,
                    }
                    let invocation: CodecInvocation<Payload> =
                        serde_json::from_value(value).map_err(|error| error.to_string())?;
                    let codec = invocation.context.identity()?;
                    let context = completion.response_context(codec)?;
                    serde_json::to_value(callback(invocation.payload.response, context).await?)
                        .map_err(|error| error.to_string())
                })
            }),
        )
    }

    /// Registers an asynchronous LLM conditional-execution guardrail.
    pub fn register_llm_conditional_execution_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(LlmRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<String>>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.register_unary_adapter(
            NemoRelayNativeAsyncMiddlewareKind::LlmConditionalExecution,
            name,
            priority,
            false,
            Box::new(move |value, _, _| {
                let callback = Arc::clone(&callback);
                Box::pin(async move {
                    let invocation: RequestInvocation =
                        serde_json::from_value(value).map_err(|error| error.to_string())?;
                    serde_json::to_value(callback(invocation.request).await?)
                        .map_err(|error| error.to_string())
                })
            }),
        )
    }

    /// Registers an asynchronous LLM request intercept.
    pub fn register_llm_request_intercept<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, LlmRequest, Option<AnnotatedLlmRequest>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<LlmRequestInterceptOutcome>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.register_unary_adapter(
            NemoRelayNativeAsyncMiddlewareKind::LlmRequestIntercept,
            name,
            priority,
            break_chain,
            Box::new(move |value, _, _| {
                let callback = Arc::clone(&callback);
                Box::pin(async move {
                    let invocation: LlmRequestInterceptInvocation =
                        serde_json::from_value(value).map_err(|error| error.to_string())?;
                    serde_json::to_value(
                        callback(invocation.name, invocation.request, invocation.annotated).await?,
                    )
                    .map_err(|error| error.to_string())
                })
            }),
        )
    }

    /// Registers an asynchronous LLM execution intercept.
    pub fn register_llm_execution_intercept<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, LlmRequest, LlmNext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Json>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.register_unary_adapter(
            NemoRelayNativeAsyncMiddlewareKind::LlmExecutionIntercept,
            name,
            priority,
            false,
            Box::new(move |value, next, _| {
                let callback = Arc::clone(&callback);
                Box::pin(async move {
                    let invocation: NameRequestInvocation =
                        serde_json::from_value(value).map_err(|error| error.to_string())?;
                    let next = LlmNext(
                        next.ok_or_else(|| "LLM execution continuation was null".to_string())?,
                    );
                    callback(invocation.name, invocation.request, next).await
                })
            }),
        )
    }

    /// Registers an asynchronous LLM stream execution intercept.
    pub fn register_llm_stream_execution_intercept<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, LlmRequest, LlmStreamNext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<LlmJsonAsyncStream>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        let state = Box::into_raw(Box::new(StreamCallbackState {
            host: self.host_v4()?,
            executor: Arc::clone(&self.executor),
            adapter: Box::new(move |value, next| {
                let callback = Arc::clone(&callback);
                Box::pin(async move {
                    let invocation: NameRequestInvocation =
                        serde_json::from_value(value).map_err(|error| error.to_string())?;
                    callback(invocation.name, invocation.request, next).await
                })
            }),
        }));
        let status = unsafe {
            self.register_async_stream_middleware_raw(
                name,
                priority,
                stream_trampoline,
                state.cast(),
                Some(drop_stream_callback),
            )
        };
        if status == NemoRelayStatus::Ok {
            Ok(())
        } else {
            Err(status_message(
                self.host,
                status,
                "register typed async stream middleware",
            ))
        }
    }
}

fn status_result(status: NemoRelayStatus, operation: &str) -> Result<()> {
    if status == NemoRelayStatus::Ok {
        Ok(())
    } else {
        Err(format!("{operation} failed: {status:?}"))
    }
}

fn registration_operation(kind: NemoRelayNativeAsyncMiddlewareKind) -> &'static str {
    match kind {
        NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeRequest => "tool request sanitizer",
        NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeResponse => "tool response sanitizer",
        NemoRelayNativeAsyncMiddlewareKind::ToolConditionalExecution => {
            "tool conditional guardrail"
        }
        NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept => "tool request intercept",
        NemoRelayNativeAsyncMiddlewareKind::ToolExecutionIntercept => "tool execution intercept",
        NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeRequest => "LLM request sanitizer",
        NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeResponse => "LLM response sanitizer",
        NemoRelayNativeAsyncMiddlewareKind::LlmConditionalExecution => "LLM conditional guardrail",
        NemoRelayNativeAsyncMiddlewareKind::LlmRequestIntercept => "LLM request intercept",
        NemoRelayNativeAsyncMiddlewareKind::LlmExecutionIntercept => "LLM execution intercept",
        NemoRelayNativeAsyncMiddlewareKind::LlmStreamExecutionIntercept => {
            "LLM stream execution intercept"
        }
        NemoRelayNativeAsyncMiddlewareKind::MarkSanitize => "mark sanitizer",
        NemoRelayNativeAsyncMiddlewareKind::ScopeSanitizeStart => "scope start sanitizer",
        NemoRelayNativeAsyncMiddlewareKind::ScopeSanitizeEnd => "scope end sanitizer",
    }
}

fn status_message(
    host: &NemoRelayNativeHostApiV1,
    status: NemoRelayStatus,
    operation: &str,
) -> String {
    let _ = host;
    format!("{operation} failed: {status:?}")
}
