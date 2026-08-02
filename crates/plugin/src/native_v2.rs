// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Safe Rust facade for native API v2 LLM continuations.

use std::ffi::c_void;
use std::future::Future;
use std::pin::Pin;
use std::ptr;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::channel::oneshot;
use futures::future::FutureExt;
use futures::task::{ArcWake, waker_ref};
use futures::{Stream, StreamExt};
use serde::Deserialize;

use super::{
    HostString, Json, LlmContinuationFailureV2, LlmContinuationInvocationV2,
    LlmContinuationOutcomeV2, LlmContinuationStreamEventV2, LlmNonHttpFailureKindV2,
    LlmNonHttpFailureV2, LlmRequest, NemoRelayNativeAsyncCallbackState,
    NemoRelayNativeAsyncCompletion, NemoRelayNativeAsyncMiddlewareKind, NemoRelayNativeAsyncNext,
    NemoRelayNativeAsyncStream, NemoRelayNativeAsyncTaskV2, NemoRelayNativeHostApiV1,
    NemoRelayNativeHostApiV4, NemoRelayNativeLlmStreamV2, NemoRelayNativeString, NemoRelayStatus,
    PluginContext, Result, read_json_value, read_required_host_string, set_last_error,
};

const MAX_SDK_ERROR_BYTES: usize = 4 * 1024;

/// Asynchronous JSON stream returned by a safe native API v2 stream callback.
pub type LlmJsonAsyncStreamV2 = Pin<Box<dyn Stream<Item = Result<Json>> + Send>>;

/// Result of a safe native API v2 streaming execution callback.
pub enum LlmStreamExecutionOutcomeV2 {
    /// Relay emits the plugin-produced stream.
    Stream(LlmJsonAsyncStreamV2),
    /// Relay forwards the ordinary downstream continuation itself.
    ///
    /// Provider events stay inside Relay and are copied into the caller stream
    /// with the host's bounded backpressure path.
    Passthrough(LlmRequest),
}

/// Cloneable targeted buffered LLM continuation.
///
/// Clones may be called repeatedly or concurrently. The underlying C handle
/// is released exactly once after the final clone and all in-flight calls are
/// dropped.
#[derive(Clone)]
pub struct LlmContinuationV2 {
    inner: Arc<ContinuationInner>,
}

/// Cloneable targeted streaming LLM continuation.
///
/// Clones may open provider streams repeatedly or concurrently. Each returned
/// provider stream has independent pull and cancellation state.
#[derive(Clone)]
pub struct LlmStreamContinuationV2 {
    inner: Arc<StreamContinuationInner>,
}

/// Provider stream returned by [`LlmStreamContinuationV2::open_stream`].
///
/// Dropping an unfinished stream cancels provider production. At most one host
/// pull is outstanding for a stream at any time.
pub struct LlmProviderStreamV2 {
    host: NemoRelayNativeHostApiV4,
    raw: *const NemoRelayNativeLlmStreamV2,
    pending: Option<oneshot::Receiver<ProviderItem>>,
    finished: bool,
    terminal: bool,
}

struct ProviderItem {
    value: std::result::Result<Option<Json>, LlmContinuationFailureV2>,
    terminal: bool,
}

struct ContinuationInner {
    host: NemoRelayNativeHostApiV4,
    next: *const NemoRelayNativeAsyncNext,
}

enum HostTaskCancellation {
    Completion(*const NemoRelayNativeAsyncCompletion),
    Stream(*const NemoRelayNativeAsyncStream),
}

#[derive(Clone, Copy)]
struct CompletionHandle(*const NemoRelayNativeAsyncCompletion);

#[derive(Clone, Copy)]
struct OutputHandle(*const NemoRelayNativeAsyncStream);

struct TaskHostString {
    host: NemoRelayNativeHostApiV1,
    raw: usize,
}

// The host owns these opaque handles and documents their operations as
// thread-safe for a pending callback's lifetime.
unsafe impl Send for CompletionHandle {}
unsafe impl Sync for CompletionHandle {}
unsafe impl Send for OutputHandle {}
unsafe impl Sync for OutputHandle {}

impl TaskHostString {
    fn new(host: &NemoRelayNativeHostApiV1, value: &str) -> Option<Self> {
        let mut raw = ptr::null_mut();
        let status = unsafe { (host.string_new)(value.as_ptr(), value.len(), &mut raw) };
        if status != NemoRelayStatus::Ok || raw.is_null() {
            return None;
        }
        Some(Self {
            host: *host,
            raw: raw as usize,
        })
    }

    fn from_json(host: &NemoRelayNativeHostApiV1, value: &Json) -> Option<Self> {
        Self::new(host, &serde_json::to_string(value).ok()?)
    }

    fn as_ptr(&self) -> *const NemoRelayNativeString {
        self.raw as *const NemoRelayNativeString
    }
}

impl Drop for TaskHostString {
    fn drop(&mut self) {
        unsafe { (self.host.string_free)(self.raw as *mut NemoRelayNativeString) };
    }
}

impl CompletionHandle {
    fn as_ptr(&self) -> *const NemoRelayNativeAsyncCompletion {
        self.0
    }
}

impl OutputHandle {
    fn as_ptr(&self) -> *const NemoRelayNativeAsyncStream {
        self.0
    }
}

struct HostFutureTask {
    host: NemoRelayNativeHostApiV4,
    cancellation: HostTaskCancellation,
    completion_to_release: Option<*const NemoRelayNativeAsyncCompletion>,
    future: Option<Pin<Box<dyn Future<Output = ()> + Send>>>,
    waker: Option<Arc<HostTaskWaker>>,
}

struct CompletionReleaseGuard {
    host: NemoRelayNativeHostApiV4,
    completion: *const NemoRelayNativeAsyncCompletion,
}

struct HostTaskWaker {
    host: NemoRelayNativeHostApiV4,
    raw: *const NemoRelayNativeAsyncTaskV2,
}

// Relay serializes task polling and documents retained task handles as safe to
// wake from any thread.
unsafe impl Send for HostFutureTask {}
unsafe impl Send for HostTaskWaker {}
unsafe impl Sync for HostTaskWaker {}

impl HostFutureTask {
    fn is_cancelled(&self) -> bool {
        unsafe {
            match self.cancellation {
                HostTaskCancellation::Completion(completion) => {
                    (self.host.v3.async_completion_is_cancelled)(completion)
                }
                HostTaskCancellation::Stream(stream) => {
                    (self.host.v3.async_stream_is_cancelled)(stream)
                }
            }
        }
    }
}

impl Drop for HostFutureTask {
    fn drop(&mut self) {
        // Keep the callback-owned completion (and therefore the host's plugin
        // library guard) alive until every plugin-owned destructor has run.
        // User futures and streams may panic from Drop, so catch those panics
        // long enough for the completion release guard to run exactly once,
        // then resume unwinding for the outer FFI panic fence to report it.
        let completion_release =
            self.completion_to_release
                .take()
                .map(|completion| CompletionReleaseGuard {
                    host: self.host,
                    completion,
                });
        let future_panic =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(self.future.take())))
                .err();
        let waker_panic =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(self.waker.take())))
                .err();
        drop(completion_release);
        if let Some(payload) = future_panic.or(waker_panic) {
            std::panic::resume_unwind(payload);
        }
    }
}

impl Drop for CompletionReleaseGuard {
    fn drop(&mut self) {
        unsafe { (self.host.v3.async_completion_release)(self.completion) };
    }
}

impl HostTaskWaker {
    unsafe fn new(
        host: NemoRelayNativeHostApiV4,
        raw: *const NemoRelayNativeAsyncTaskV2,
    ) -> Option<Arc<Self>> {
        if raw.is_null() {
            return None;
        }
        unsafe { (host.async_task_retain_v2)(raw) };
        Some(Arc::new(Self { host, raw }))
    }
}

impl ArcWake for HostTaskWaker {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        let status = unsafe { (arc_self.host.async_task_wake_v2)(arc_self.raw) };
        if status != NemoRelayStatus::Ok {
            set_last_error(
                &arc_self.host.v3.v1,
                &format!("native API v2 host task wake failed: {status:?}"),
            );
        }
    }
}

impl Drop for HostTaskWaker {
    fn drop(&mut self) {
        unsafe { (self.host.async_task_release_v2)(self.raw) };
    }
}

unsafe extern "C" fn poll_host_future_task(
    user_data: *mut c_void,
    task: *const NemoRelayNativeAsyncTaskV2,
) -> u32 {
    if user_data.is_null() || task.is_null() {
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    }
    let state = unsafe { &mut *user_data.cast::<HostFutureTask>() };
    if state.is_cancelled() {
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    }
    if state.waker.is_none() {
        let waker = match unsafe { HostTaskWaker::new(state.host, task) } {
            Some(waker) => waker,
            None => {
                set_last_error(&state.host.v3.v1, "native API v2 host task was null");
                return NemoRelayNativeAsyncCallbackState::Complete as u32;
            }
        };
        state.waker = Some(waker);
    }
    let waker = waker_ref(
        state
            .waker
            .as_ref()
            .expect("host task waker was initialized"),
    );
    let mut context = Context::from_waker(&waker);
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state
            .future
            .as_mut()
            .expect("host task future was initialized")
            .as_mut()
            .poll(&mut context)
    })) {
        Ok(Poll::Ready(())) => NemoRelayNativeAsyncCallbackState::Complete as u32,
        Ok(Poll::Pending) => NemoRelayNativeAsyncCallbackState::Pending as u32,
        Err(_) => {
            set_last_error(&state.host.v3.v1, "native API v2 host task panicked");
            NemoRelayNativeAsyncCallbackState::Complete as u32
        }
    }
}

unsafe extern "C" fn drop_host_future_task(user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    let state = unsafe { Box::from_raw(user_data.cast::<HostFutureTask>()) };
    let host = state.host.v3.v1;
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(state))).is_err() {
        set_last_error(&host, "native API v2 host task state drop panicked");
    }
}

// Host tables are immutable, and Relay documents retained continuation
// handles as thread-safe for repeated and concurrent invocation.
unsafe impl Send for ContinuationInner {}
unsafe impl Sync for ContinuationInner {}

impl Drop for ContinuationInner {
    fn drop(&mut self) {
        unsafe { (self.host.v3.async_next_release)(self.next) };
    }
}

struct StreamContinuationInner {
    continuation: Arc<ContinuationInner>,
    output: *const NemoRelayNativeAsyncStream,
}

// The output handle is owned exclusively by this shared RAII state. Host
// operations synchronize settlement and cancellation.
unsafe impl Send for StreamContinuationInner {}
unsafe impl Sync for StreamContinuationInner {}

impl Drop for StreamContinuationInner {
    fn drop(&mut self) {
        unsafe { (self.continuation.host.v3.async_stream_release)(self.output) };
    }
}

impl LlmContinuationV2 {
    unsafe fn from_raw(
        host: NemoRelayNativeHostApiV4,
        next: *const NemoRelayNativeAsyncNext,
    ) -> std::result::Result<Self, NemoRelayStatus> {
        if next.is_null() {
            return Err(NemoRelayStatus::NullPointer);
        }
        Ok(Self {
            inner: Arc::new(ContinuationInner { host, next }),
        })
    }

    /// Dispatches one explicitly targeted LLM continuation.
    pub async fn call(
        &self,
        invocation: LlmContinuationInvocationV2,
    ) -> std::result::Result<Json, LlmContinuationFailureV2> {
        let (sender, receiver) = oneshot::channel();
        let state = Box::new(TargetedResultCallback {
            host: self.inner.host.v3.v1,
            sender,
        });
        let state = Box::into_raw(state).cast::<c_void>();
        let status = match HostString::from_json(&self.inner.host.v3.v1, &invocation) {
            Some(invocation) => unsafe {
                (self.inner.host.async_llm_next_invoke_result_v2)(
                    self.inner.next,
                    invocation.as_ptr(),
                    targeted_result_callback,
                    state,
                )
            },
            None => {
                unsafe { drop(Box::from_raw(state.cast::<TargetedResultCallback>())) };
                return Err(internal_failure(
                    "failed to serialize targeted LLM continuation invocation",
                ));
            }
        };
        if status != NemoRelayStatus::Ok {
            unsafe { drop(Box::from_raw(state.cast::<TargetedResultCallback>())) };
            return Err(status_failure("targeted LLM continuation", status));
        }
        receiver.await.unwrap_or_else(|_| {
            Err(internal_failure(
                "targeted LLM continuation callback closed without an outcome",
            ))
        })
    }

    /// Invokes the ordinary buffered downstream continuation.
    pub async fn call_passthrough(&self, request: LlmRequest) -> Result<Json> {
        let (sender, receiver) = oneshot::channel();
        let state = Box::new(PassthroughResultCallback {
            host: self.inner.host.v3.v1,
            sender,
        });
        let state = Box::into_raw(state).cast::<c_void>();
        let status = match HostString::from_json(&self.inner.host.v3.v1, &request) {
            Some(request) => unsafe {
                (self.inner.host.v3.async_next_invoke_result)(
                    self.inner.next,
                    request.as_ptr(),
                    passthrough_result_callback,
                    state,
                )
            },
            None => {
                unsafe { drop(Box::from_raw(state.cast::<PassthroughResultCallback>())) };
                return Err("failed to serialize LLM pass-through request".into());
            }
        };
        if status != NemoRelayStatus::Ok {
            unsafe { drop(Box::from_raw(state.cast::<PassthroughResultCallback>())) };
            return Err(format!("LLM pass-through continuation failed: {status:?}"));
        }
        receiver.await.unwrap_or_else(|_| {
            Err("LLM pass-through continuation callback closed without a result".into())
        })
    }
}

impl LlmStreamContinuationV2 {
    unsafe fn from_raw(
        host: NemoRelayNativeHostApiV4,
        next: *const NemoRelayNativeAsyncNext,
        output: *const NemoRelayNativeAsyncStream,
    ) -> std::result::Result<Self, NemoRelayStatus> {
        if next.is_null() || output.is_null() {
            return Err(NemoRelayStatus::NullPointer);
        }
        let continuation = unsafe { LlmContinuationV2::from_raw(host, next) }?;
        Ok(Self {
            inner: Arc::new(StreamContinuationInner {
                continuation: continuation.inner,
                output,
            }),
        })
    }

    /// Opens one explicitly targeted provider stream.
    pub async fn open_stream(
        &self,
        invocation: LlmContinuationInvocationV2,
    ) -> std::result::Result<LlmProviderStreamV2, LlmContinuationFailureV2> {
        let host = self.inner.continuation.host;
        let (sender, receiver) = oneshot::channel();
        let state = Box::new(StreamOpenCallback { host, sender });
        let state = Box::into_raw(state).cast::<c_void>();
        let status = match HostString::from_json(&host.v3.v1, &invocation) {
            Some(invocation) => unsafe {
                (host.async_llm_next_open_stream_v2)(
                    self.inner.continuation.next,
                    invocation.as_ptr(),
                    self.inner.output,
                    stream_open_callback,
                    state,
                )
            },
            None => {
                unsafe { drop(Box::from_raw(state.cast::<StreamOpenCallback>())) };
                return Err(internal_failure(
                    "failed to serialize targeted LLM stream invocation",
                ));
            }
        };
        if status != NemoRelayStatus::Ok {
            unsafe { drop(Box::from_raw(state.cast::<StreamOpenCallback>())) };
            return Err(status_failure("targeted LLM stream setup", status));
        }
        let raw = receiver.await.unwrap_or_else(|_| {
            Err(internal_failure(
                "targeted LLM stream setup callback closed without an outcome",
            ))
        })? as *const NemoRelayNativeLlmStreamV2;
        if raw.is_null() {
            return Err(internal_failure(
                "targeted LLM stream setup returned a null stream",
            ));
        }
        Ok(LlmProviderStreamV2 {
            host,
            raw,
            pending: None,
            finished: false,
            terminal: false,
        })
    }

    async fn forward_passthrough(&self, request: LlmRequest) -> Result<()> {
        let host = self.inner.continuation.host;
        let (sender, receiver) = oneshot::channel();
        let state = Box::new(ForwardStreamCallback {
            host: host.v3.v1,
            sender,
        });
        let state = Box::into_raw(state).cast::<c_void>();
        let status = match HostString::from_json(&host.v3.v1, &request) {
            Some(request) => unsafe {
                (host.async_llm_next_forward_stream_v2)(
                    self.inner.continuation.next,
                    request.as_ptr(),
                    self.inner.output,
                    forward_stream_callback,
                    state,
                )
            },
            None => {
                unsafe { drop(Box::from_raw(state.cast::<ForwardStreamCallback>())) };
                return Err("failed to serialize streaming pass-through request".into());
            }
        };
        if status != NemoRelayStatus::Ok {
            unsafe { drop(Box::from_raw(state.cast::<ForwardStreamCallback>())) };
            return Err(format!(
                "streaming pass-through continuation failed: {status:?}"
            ));
        }
        receiver.await.unwrap_or_else(|_| {
            Err("streaming pass-through callback closed before settlement".into())
        })
    }
}

// Relay retains the provider stream handle until release and serializes host
// pulls. The Rust wrapper owns the sole plugin reference.
unsafe impl Send for LlmProviderStreamV2 {}

impl Stream for LlmProviderStreamV2 {
    type Item = std::result::Result<Json, LlmContinuationFailureV2>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }
        if self.pending.is_none() {
            let (sender, receiver) = oneshot::channel();
            let state = Box::new(ProviderNextCallback {
                host: self.host.v3.v1,
                sender,
            });
            let state = Box::into_raw(state).cast::<c_void>();
            let status = unsafe {
                (self.host.async_llm_stream_next_v2)(self.raw, provider_next_callback, state)
            };
            if status != NemoRelayStatus::Ok {
                unsafe { drop(Box::from_raw(state.cast::<ProviderNextCallback>())) };
                self.finished = true;
                return Poll::Ready(Some(Err(status_failure(
                    "targeted provider stream poll",
                    status,
                ))));
            }
            self.pending = Some(receiver);
        }

        let receiver = self
            .pending
            .as_mut()
            .expect("provider stream pending receiver was initialized");
        match Pin::new(receiver).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                self.pending = None;
                let item = result.unwrap_or_else(|_| ProviderItem {
                    value: Err(internal_failure(
                        "provider stream callback closed without an event",
                    )),
                    terminal: false,
                });
                self.terminal = item.terminal;
                match item.value {
                    Ok(Some(chunk)) => Poll::Ready(Some(Ok(chunk))),
                    Ok(None) => {
                        self.finished = true;
                        Poll::Ready(None)
                    }
                    Err(error) => {
                        self.finished = true;
                        Poll::Ready(Some(Err(error)))
                    }
                }
            }
        }
    }
}

impl Drop for LlmProviderStreamV2 {
    fn drop(&mut self) {
        if !self.terminal {
            let _ = unsafe { (self.host.async_llm_stream_cancel_v2)(self.raw) };
        }
        unsafe { (self.host.async_llm_stream_release_v2)(self.raw) };
        self.raw = ptr::null();
    }
}

#[derive(Deserialize)]
struct LlmCallbackInvocation {
    name: String,
    request: LlmRequest,
}

struct SafeV2Callback<F> {
    host: NemoRelayNativeHostApiV4,
    callback: Arc<F>,
}

unsafe extern "C" fn drop_safe_v2_callback<F>(user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    let state = unsafe { Box::from_raw(user_data.cast::<SafeV2Callback<F>>()) };
    let host = state.host.v3.v1;
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(state))).is_err() {
        set_last_error(&host, "native API v2 safe callback state drop panicked");
    }
}

impl PluginContext<'_> {
    /// Registers a safe asynchronous native API v2 buffered LLM execution callback.
    ///
    /// The callback receives owned Rust values and a cloneable continuation.
    /// Relay cooperatively polls its future on the host runtime with the
    /// invocation's active scope stack restored for every poll.
    pub fn register_async_llm_execution_v2<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, LlmRequest, LlmContinuationV2) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Json>> + Send + 'static,
    {
        let host = self
            .host_api_v4()
            .copied()
            .ok_or_else(|| "native API v2 requires a complete ABI-v4 host table".to_string())?;
        let name = HostString::try_new(&host.v3.v1, name).map_err(|status| {
            format!("native API v2 buffered LLM registration name failed: {status:?}")
        })?;
        let state = Box::new(SafeV2Callback {
            host,
            callback: Arc::new(callback),
        });
        let user_data = Box::into_raw(state).cast::<c_void>();
        let status = unsafe {
            // Once invoked, the host consumes `user_data` on both success and
            // failure and calls `free_fn` exactly once. Allocate the fallible
            // name first so local ownership is never ambiguous.
            (host.v3.plugin_context_register_async_middleware)(
                self.raw,
                NemoRelayNativeAsyncMiddlewareKind::LlmExecutionIntercept as u32,
                name.as_ptr(),
                priority,
                false,
                safe_buffered_trampoline::<F, Fut>,
                user_data,
                Some(drop_safe_v2_callback::<F>),
            )
        };
        if status == NemoRelayStatus::Ok {
            Ok(())
        } else {
            Err(super::status_error(
                &host.v3.v1,
                status,
                "native API v2 buffered LLM registration",
            ))
        }
    }

    /// Registers a safe asynchronous native API v2 streaming LLM callback.
    ///
    /// The callback may return a Rust stream or request host-owned direct
    /// pass-through. Relay cooperatively polls its future and returned stream
    /// and wakes them when bounded output backpressure clears.
    pub fn register_async_llm_stream_execution_v2<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, LlmRequest, LlmStreamContinuationV2) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<LlmStreamExecutionOutcomeV2>> + Send + 'static,
    {
        let host = self
            .host_api_v4()
            .copied()
            .ok_or_else(|| "native API v2 requires a complete ABI-v4 host table".to_string())?;
        let name = HostString::try_new(&host.v3.v1, name).map_err(|status| {
            format!("native API v2 streaming LLM registration name failed: {status:?}")
        })?;
        let state = Box::new(SafeV2Callback {
            host,
            callback: Arc::new(callback),
        });
        let user_data = Box::into_raw(state).cast::<c_void>();
        let status = unsafe {
            (host.v3.plugin_context_register_async_stream_middleware)(
                self.raw,
                name.as_ptr(),
                priority,
                safe_streaming_trampoline::<F, Fut>,
                user_data,
                Some(drop_safe_v2_callback::<F>),
            )
        };
        if status == NemoRelayStatus::Ok {
            Ok(())
        } else {
            Err(super::status_error(
                &host.v3.v1,
                status,
                "native API v2 streaming LLM registration",
            ))
        }
    }
}

unsafe extern "C" fn safe_buffered_trampoline<F, Fut>(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    next: *const NemoRelayNativeAsyncNext,
    completion: *const NemoRelayNativeAsyncCompletion,
) -> u32
where
    F: Fn(String, LlmRequest, LlmContinuationV2) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Json>> + Send + 'static,
{
    if user_data.is_null() {
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    }
    let state = unsafe { &*user_data.cast::<SafeV2Callback<F>>() };
    if completion.is_null() {
        if !next.is_null() {
            unsafe { (state.host.v3.async_next_release)(next) };
        }
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    }
    let continuation = match unsafe { LlmContinuationV2::from_raw(state.host, next) } {
        Ok(continuation) => continuation,
        Err(status) => {
            reject_completion(
                &state.host,
                completion,
                &format!("invalid native API v2 LLM continuation: {status:?}"),
            );
            return NemoRelayNativeAsyncCallbackState::Complete as u32;
        }
    };
    let invocation: LlmCallbackInvocation = match read_json_value(
        &state.host.v3.v1,
        invocation_json,
        "native API v2 LLM invocation",
    ) {
        Ok(invocation) => invocation,
        Err(status) => {
            reject_completion(
                &state.host,
                completion,
                &format!("invalid native API v2 LLM invocation: {status:?}"),
            );
            drop(continuation);
            return NemoRelayNativeAsyncCallbackState::Complete as u32;
        }
    };
    let host = state.host;
    let completion_handle = CompletionHandle(completion);
    let callback = Arc::clone(&state.callback);
    let callback_continuation = continuation.clone();
    let future = async move {
        let result = std::panic::AssertUnwindSafe(async move {
            callback(invocation.name, invocation.request, callback_continuation).await
        })
        .catch_unwind()
        .await;
        drop(continuation);
        match result {
            Ok(Ok(value)) => resolve_completion(&host, completion_handle.as_ptr(), &value),
            Ok(Err(error)) => reject_completion(&host, completion_handle.as_ptr(), &error),
            Err(_) => reject_completion(
                &host,
                completion_handle.as_ptr(),
                "native API v2 buffered LLM callback panicked",
            ),
        }
    };
    let task = Box::new(HostFutureTask {
        host,
        cancellation: HostTaskCancellation::Completion(completion),
        completion_to_release: Some(completion),
        future: Some(Box::pin(future)),
        waker: None,
    });
    let task = Box::into_raw(task).cast::<c_void>();
    let status = unsafe {
        (host.async_completion_spawn_task_v2)(
            completion,
            poll_host_future_task,
            task,
            Some(drop_host_future_task),
        )
    };
    if status == NemoRelayStatus::Ok {
        NemoRelayNativeAsyncCallbackState::Pending as u32
    } else {
        let mut task = unsafe { Box::from_raw(task.cast::<HostFutureTask>()) };
        task.completion_to_release = None;
        drop(task);
        reject_completion(
            &host,
            completion,
            &format!("native API v2 buffered task spawn failed: {status:?}"),
        );
        NemoRelayNativeAsyncCallbackState::Complete as u32
    }
}

unsafe extern "C" fn safe_streaming_trampoline<F, Fut>(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    next: *const NemoRelayNativeAsyncNext,
    output: *const NemoRelayNativeAsyncStream,
) -> u32
where
    F: Fn(String, LlmRequest, LlmStreamContinuationV2) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<LlmStreamExecutionOutcomeV2>> + Send + 'static,
{
    if user_data.is_null() {
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    }
    let state = unsafe { &*user_data.cast::<SafeV2Callback<F>>() };
    if output.is_null() {
        if !next.is_null() {
            unsafe { (state.host.v3.async_next_release)(next) };
        }
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    }
    let continuation = match unsafe { LlmStreamContinuationV2::from_raw(state.host, next, output) }
    {
        Ok(continuation) => continuation,
        Err(status) => {
            reject_output_once(
                &state.host,
                output,
                &format!("invalid native API v2 stream continuation: {status:?}"),
            );
            unsafe { (state.host.v3.async_stream_release)(output) };
            return NemoRelayNativeAsyncCallbackState::Complete as u32;
        }
    };
    let invocation: LlmCallbackInvocation = match read_json_value(
        &state.host.v3.v1,
        invocation_json,
        "native API v2 streaming LLM invocation",
    ) {
        Ok(invocation) => invocation,
        Err(status) => {
            reject_output_once(
                &state.host,
                output,
                &format!("invalid native API v2 stream invocation: {status:?}"),
            );
            drop(continuation);
            return NemoRelayNativeAsyncCallbackState::Complete as u32;
        }
    };
    let host = state.host;
    let output_handle = OutputHandle(output);
    let callback = Arc::clone(&state.callback);
    let callback_continuation = continuation.clone();
    let future = async move {
        let result = std::panic::AssertUnwindSafe(async move {
            let outcome =
                callback(invocation.name, invocation.request, callback_continuation).await?;
            match outcome {
                LlmStreamExecutionOutcomeV2::Stream(stream) => {
                    pump_output_stream(&continuation, stream).await
                }
                LlmStreamExecutionOutcomeV2::Passthrough(request) => {
                    continuation.forward_passthrough(request).await
                }
            }
        })
        .catch_unwind()
        .await;
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => reject_output(&host, output_handle, &error).await,
            Err(_) => {
                reject_output(
                    &host,
                    output_handle,
                    "native API v2 streaming LLM callback panicked",
                )
                .await;
            }
        }
    };
    let task = Box::new(HostFutureTask {
        host,
        cancellation: HostTaskCancellation::Stream(output),
        completion_to_release: None,
        future: Some(Box::pin(future)),
        waker: None,
    });
    let task = Box::into_raw(task).cast::<c_void>();
    let status = unsafe {
        (host.async_stream_spawn_task_v2)(
            output,
            poll_host_future_task,
            task,
            Some(drop_host_future_task),
        )
    };
    if status == NemoRelayStatus::Ok {
        NemoRelayNativeAsyncCallbackState::Pending as u32
    } else {
        reject_output_once(
            &host,
            output,
            &format!("native API v2 streaming task spawn failed: {status:?}"),
        );
        unsafe { drop(Box::from_raw(task.cast::<HostFutureTask>())) };
        NemoRelayNativeAsyncCallbackState::Complete as u32
    }
}

async fn pump_output_stream(
    continuation: &LlmStreamContinuationV2,
    mut stream: LlmJsonAsyncStreamV2,
) -> Result<()> {
    while let Some(item) = stream.next().await {
        let chunk = item?;
        push_output_json(continuation, &chunk).await?;
    }
    finish_output(continuation)
}

async fn push_output_json(continuation: &LlmStreamContinuationV2, chunk: &Json) -> Result<()> {
    let host = &continuation.inner.continuation.host;
    let chunk = TaskHostString::from_json(&host.v3.v1, chunk)
        .ok_or_else(|| "failed to serialize native API v2 output chunk".to_string())?;
    std::future::poll_fn(move |_context| {
        if unsafe { (host.v3.async_stream_is_cancelled)(continuation.inner.output) } {
            return Poll::Ready(Err("native API v2 output stream was cancelled".into()));
        }
        let status =
            unsafe { (host.v3.async_stream_push_json)(continuation.inner.output, chunk.as_ptr()) };
        match status {
            NemoRelayStatus::Ok => Poll::Ready(Ok(())),
            // For this operation the V3 ABI contract reserves `Internal` for
            // a full bounded queue. Serialization and lifecycle faults use
            // distinct statuses, so retrying cannot mask another ABI error.
            NemoRelayStatus::Internal => Poll::Pending,
            status => Poll::Ready(Err(format!("native API v2 output push failed: {status:?}"))),
        }
    })
    .await
}

fn finish_output(continuation: &LlmStreamContinuationV2) -> Result<()> {
    let host = &continuation.inner.continuation.host;
    if unsafe { (host.v3.async_stream_is_cancelled)(continuation.inner.output) } {
        return Err("native API v2 output stream was cancelled".into());
    }
    let status = unsafe { (host.v3.async_stream_finish)(continuation.inner.output) };
    if status == NemoRelayStatus::Ok {
        Ok(())
    } else {
        Err(format!("native API v2 output finish failed: {status:?}"))
    }
}

async fn reject_output(host: &NemoRelayNativeHostApiV4, output: OutputHandle, error: &str) {
    if unsafe { (host.v3.async_stream_is_cancelled)(output.as_ptr()) } {
        return;
    }
    let error = bounded_error(error);
    let Some(message) = TaskHostString::new(&host.v3.v1, &error) else {
        set_last_error(
            &host.v3.v1,
            "failed to allocate native API v2 stream rejection",
        );
        return;
    };
    std::future::poll_fn(move |_context| {
        if unsafe { (host.v3.async_stream_is_cancelled)(output.as_ptr()) } {
            return Poll::Ready(());
        }
        let status = unsafe { (host.v3.async_stream_reject)(output.as_ptr(), message.as_ptr()) };
        match status {
            NemoRelayStatus::Ok => Poll::Ready(()),
            // `Internal` has the same queue-full-only contract for rejection.
            NemoRelayStatus::Internal => Poll::Pending,
            status => {
                set_last_error(
                    &host.v3.v1,
                    &format!("native API v2 output rejection failed: {status:?}"),
                );
                Poll::Ready(())
            }
        }
    })
    .await
}

fn reject_output_once(
    host: &NemoRelayNativeHostApiV4,
    output: *const NemoRelayNativeAsyncStream,
    error: &str,
) {
    if unsafe { (host.v3.async_stream_is_cancelled)(output) } {
        return;
    }
    let error = bounded_error(error);
    let Some(message) = HostString::new(&host.v3.v1, &error) else {
        set_last_error(
            &host.v3.v1,
            "failed to allocate native API v2 stream rejection",
        );
        return;
    };
    let status = unsafe { (host.v3.async_stream_reject)(output, message.as_ptr()) };
    if status != NemoRelayStatus::Ok {
        set_last_error(
            &host.v3.v1,
            &format!("native API v2 output rejection failed: {status:?}"),
        );
    }
}

fn resolve_completion(
    host: &NemoRelayNativeHostApiV4,
    completion: *const NemoRelayNativeAsyncCompletion,
    value: &Json,
) {
    if unsafe { (host.v3.async_completion_is_cancelled)(completion) } {
        return;
    }
    let Some(value) = HostString::from_json(&host.v3.v1, value) else {
        reject_completion(
            host,
            completion,
            "failed to serialize native API v2 callback result",
        );
        return;
    };
    let status = unsafe { (host.v3.async_completion_resolve_json)(completion, value.as_ptr()) };
    if status != NemoRelayStatus::Ok {
        set_last_error(
            &host.v3.v1,
            &format!("native API v2 callback completion failed: {status:?}"),
        );
    }
}

fn reject_completion(
    host: &NemoRelayNativeHostApiV4,
    completion: *const NemoRelayNativeAsyncCompletion,
    error: &str,
) {
    if unsafe { (host.v3.async_completion_is_cancelled)(completion) } {
        return;
    }
    let error = bounded_error(error);
    let Some(error) = HostString::new(&host.v3.v1, &error) else {
        set_last_error(
            &host.v3.v1,
            "failed to allocate native API v2 callback rejection",
        );
        return;
    };
    let status = unsafe { (host.v3.async_completion_reject)(completion, error.as_ptr()) };
    if status != NemoRelayStatus::Ok {
        set_last_error(
            &host.v3.v1,
            &format!("native API v2 callback rejection failed: {status:?}"),
        );
    }
}

struct TargetedResultCallback {
    host: NemoRelayNativeHostApiV1,
    sender: oneshot::Sender<std::result::Result<Json, LlmContinuationFailureV2>>,
}

unsafe extern "C" fn targeted_result_callback(
    user_data: *mut c_void,
    outcome_json: *const NemoRelayNativeString,
) {
    if user_data.is_null() {
        return;
    }
    let state = unsafe { Box::from_raw(user_data.cast::<TargetedResultCallback>()) };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let outcome: LlmContinuationOutcomeV2 = read_json_value(
            &state.host,
            outcome_json,
            "targeted LLM continuation outcome",
        )
        .map_err(|status| status_failure("targeted LLM continuation outcome", status))?;
        match outcome {
            LlmContinuationOutcomeV2::Success { response } => Ok(response),
            LlmContinuationOutcomeV2::Failure { error } => Err(error),
        }
    }))
    .unwrap_or_else(|_| Err(internal_failure("targeted LLM result callback panicked")));
    let _ = state.sender.send(result);
}

struct PassthroughResultCallback {
    host: NemoRelayNativeHostApiV1,
    sender: oneshot::Sender<Result<Json>>,
}

unsafe extern "C" fn passthrough_result_callback(
    user_data: *mut c_void,
    value_json: *const NemoRelayNativeString,
    error: *const NemoRelayNativeString,
) {
    if user_data.is_null() {
        return;
    }
    let state = unsafe { Box::from_raw(user_data.cast::<PassthroughResultCallback>()) };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !error.is_null() {
            return read_required_host_string(&state.host, error, "LLM pass-through error")
                .map_or_else(
                    |status| Err(format!("invalid LLM pass-through error: {status:?}")),
                    Err,
                );
        }
        read_json_value(&state.host, value_json, "LLM pass-through result")
            .map_err(|status| format!("invalid LLM pass-through result: {status:?}"))
    }))
    .unwrap_or_else(|_| Err("LLM pass-through result callback panicked".into()));
    let _ = state.sender.send(result);
}

struct StreamOpenCallback {
    host: NemoRelayNativeHostApiV4,
    sender: oneshot::Sender<std::result::Result<usize, LlmContinuationFailureV2>>,
}

struct OwnedProviderStream {
    host: NemoRelayNativeHostApiV4,
    raw: *const NemoRelayNativeLlmStreamV2,
    armed: bool,
}

impl OwnedProviderStream {
    fn new(host: NemoRelayNativeHostApiV4, raw: *const NemoRelayNativeLlmStreamV2) -> Self {
        Self {
            host,
            raw,
            armed: !raw.is_null(),
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for OwnedProviderStream {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let _ = unsafe { (self.host.async_llm_stream_cancel_v2)(self.raw) };
        unsafe { (self.host.async_llm_stream_release_v2)(self.raw) };
    }
}

unsafe extern "C" fn stream_open_callback(
    user_data: *mut c_void,
    stream: *const NemoRelayNativeLlmStreamV2,
    error_json: *const NemoRelayNativeString,
) {
    if user_data.is_null() {
        return;
    }
    let state = unsafe { Box::from_raw(user_data.cast::<StreamOpenCallback>()) };
    let mut owned_stream = OwnedProviderStream::new(state.host, stream);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        match (stream.is_null(), error_json.is_null()) {
            (false, true) => Ok(stream as usize),
            (true, false) => read_json_value(
                &state.host.v3.v1,
                error_json,
                "targeted LLM stream setup failure",
            )
            .map_or_else(
                |status| Err(status_failure("targeted LLM stream setup failure", status)),
                Err,
            ),
            _ => Err(internal_failure(
                "targeted LLM stream setup returned an invalid outcome",
            )),
        }
    }))
    .unwrap_or_else(|_| {
        Err(internal_failure(
            "targeted LLM stream setup callback panicked",
        ))
    });
    let transfers_stream = result.is_ok();
    if state.sender.send(result).is_ok() && transfers_stream {
        owned_stream.disarm();
    }
}

struct ProviderNextCallback {
    host: NemoRelayNativeHostApiV1,
    sender: oneshot::Sender<ProviderItem>,
}

unsafe extern "C" fn provider_next_callback(
    user_data: *mut c_void,
    event_json: *const NemoRelayNativeString,
) {
    if user_data.is_null() {
        return;
    }
    let state = unsafe { Box::from_raw(user_data.cast::<ProviderNextCallback>()) };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let event: LlmContinuationStreamEventV2 =
            read_json_value(&state.host, event_json, "targeted provider stream event")
                .map_err(|status| status_failure("targeted provider stream event", status))?;
        Ok::<_, LlmContinuationFailureV2>(match event {
            LlmContinuationStreamEventV2::Chunk { chunk } => ProviderItem {
                value: Ok(Some(chunk)),
                terminal: false,
            },
            LlmContinuationStreamEventV2::Done => ProviderItem {
                value: Ok(None),
                terminal: true,
            },
            LlmContinuationStreamEventV2::Failure { error } => ProviderItem {
                value: Err(error),
                terminal: true,
            },
        })
    }))
    .unwrap_or_else(|_| {
        Err(internal_failure(
            "targeted provider stream callback panicked",
        ))
    })
    .unwrap_or_else(|error| ProviderItem {
        value: Err(error),
        terminal: false,
    });
    let _ = state.sender.send(result);
}

struct ForwardStreamCallback {
    host: NemoRelayNativeHostApiV1,
    sender: oneshot::Sender<Result<()>>,
}

unsafe extern "C" fn forward_stream_callback(
    user_data: *mut c_void,
    error: *const NemoRelayNativeString,
) {
    if user_data.is_null() {
        return;
    }
    let state = unsafe { Box::from_raw(user_data.cast::<ForwardStreamCallback>()) };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if error.is_null() {
            Ok(())
        } else {
            // Relay has already settled the output before this wake-up. Keep
            // the terminal context available for diagnostics, but report
            // completion to the trampoline so it cannot reject a second time.
            let message =
                read_required_host_string(&state.host, error, "streaming pass-through error")
                    .unwrap_or_else(|status| {
                        format!("invalid streaming pass-through error: {status:?}")
                    });
            set_last_error(&state.host, &message);
            Ok(())
        }
    }))
    .unwrap_or_else(|_| Err("streaming pass-through terminal callback panicked".into()));
    let _ = state.sender.send(result);
}

fn status_failure(label: &str, status: NemoRelayStatus) -> LlmContinuationFailureV2 {
    internal_failure(format!("{label} failed at the native boundary: {status:?}"))
}

fn internal_failure(message: impl Into<String>) -> LlmContinuationFailureV2 {
    LlmContinuationFailureV2::NonHttp {
        failure: LlmNonHttpFailureV2 {
            kind: LlmNonHttpFailureKindV2::Internal,
            message: bounded_error(&message.into()),
        },
    }
}

fn bounded_error(message: &str) -> String {
    if message.len() <= MAX_SDK_ERROR_BYTES {
        return message.to_owned();
    }
    let mut boundary = MAX_SDK_ERROR_BYTES;
    while !message.is_char_boundary(boundary) {
        boundary -= 1;
    }
    message[..boundary].to_owned()
}
