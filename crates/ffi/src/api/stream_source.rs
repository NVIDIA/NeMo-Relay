// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owned foreign stream producers and streaming continuation handles.

use super::{block_on, entry_scope_stack, tokio_runtime};
use crate::callable::NemoRelayFreeFn;
use crate::convert::{json_to_c_string, nemo_relay_string_free};
use crate::error::{NemoRelayStatus, clear_last_error, last_error_message, set_last_error};
use libc::{c_char, c_void};
use nemo_relay_runtime::api::runtime::{LlmJsonStream, MiddlewareContinuationContext};
use nemo_relay_types::{
    Json,
    api::runtime::LlmStreamInner,
    error::{FlowError, Result},
};
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
};
use tokio_stream::{Stream, StreamExt};

/// Pull one JSON chunk: 1 writes an owned string, 0 is EOF, -1 is an error.
/// Set the last error before returning -1. Only one read runs at a time.
pub type NemoRelayStreamReadFn = unsafe extern "C" fn(*mut c_void, *mut *mut c_char) -> i32;

/// Nullable C representation of [`NemoRelayStreamReadFn`].
/// `nemo_relay_stream_source_create` requires a non-null read callback.
pub type NemoRelayOptionalStreamReadFn =
    Option<unsafe extern "C" fn(*mut c_void, *mut *mut c_char) -> i32>;
/// Signal cancellation promptly. This can run during a read and must unblock it.
pub type NemoRelayStreamCancelFn = unsafe extern "C" fn(*mut c_void);

/// Nullable version of [`NemoRelayStreamCancelFn`]. Pass null to omit this callback.
pub type NemoRelayOptionalStreamCancelFn = Option<unsafe extern "C" fn(*mut c_void)>;
/// Close producer resources after reads stop. Return zero on success.
/// Set the last error on failure.
pub type NemoRelayStreamCloseFn = unsafe extern "C" fn(*mut c_void) -> NemoRelayStatus;

/// Nullable version of [`NemoRelayStreamCloseFn`]. Pass null to omit this callback.
pub type NemoRelayOptionalStreamCloseFn =
    Option<unsafe extern "C" fn(*mut c_void) -> NemoRelayStatus>;

/// An owned source, transferred to Relay when returned by a streaming callback.
pub struct FfiLlmStreamSource(Arc<SourceHandle>);
struct SourceHandle {
    close_result: tokio::sync::Mutex<Option<std::result::Result<(), String>>>,
    stream: tokio::sync::Mutex<Option<LlmJsonStream>>,
    cancel: tokio::sync::watch::Sender<bool>,
}
impl SourceHandle {
    async fn next(&self) -> Result<Option<Json>> {
        let mut cancel = self.cancel.subscribe();
        if *cancel.borrow() {
            return Ok(None);
        }
        let mut slot = self.stream.lock().await;
        let Some(stream) = slot.as_mut() else {
            return Ok(None);
        };
        tokio::select! {
            biased;
            _ = cancel.changed() => Ok(None),
            item = stream.next() => item.transpose(),
        }
    }
    async fn close(&self) -> Result<()> {
        self.cancel.send_replace(true);
        let mut cached = self.close_result.lock().await;
        if let Some(result) = cached.as_ref() {
            return result.clone().map_err(FlowError::Internal);
        }
        let result = match self.stream.lock().await.take() {
            Some(mut stream) => stream.close().await.map_err(|error| error.to_string()),
            None => Ok(()),
        };
        *cached = Some(result.clone());
        result.map_err(FlowError::Internal)
    }
}
impl Drop for SourceHandle {
    fn drop(&mut self) {
        self.cancel.send_replace(true);
    }
}

pub(crate) fn from_native(stream: LlmJsonStream) -> *mut FfiLlmStreamSource {
    let (cancel, _) = tokio::sync::watch::channel(false);
    Box::into_raw(Box::new(FfiLlmStreamSource(Arc::new(SourceHandle {
        close_result: tokio::sync::Mutex::new(None),
        stream: tokio::sync::Mutex::new(Some(stream)),
        cancel,
    }))))
}

type SourceReadFuture = Pin<Box<dyn Future<Output = Result<Option<Json>>> + Send>>;
struct SourceAdapter {
    source: Arc<SourceHandle>,
    pending: Option<SourceReadFuture>,
}
impl Stream for SourceAdapter {
    type Item = Result<Json>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let pending = this.pending.get_or_insert_with(|| {
            let source = this.source.clone();
            Box::pin(async move { source.next().await })
        });
        match pending.as_mut().poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(item) => {
                this.pending = None;
                Poll::Ready(item.transpose())
            }
        }
    }
}
impl LlmStreamInner for SourceAdapter {
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let this = self.get_mut();
        this.pending = None;
        Box::pin(this.source.close())
    }
}
impl Drop for SourceAdapter {
    fn drop(&mut self) {
        self.source.cancel.send_replace(true);
    }
}
/// Consume a source returned by foreign code, including its ownership.
pub(crate) unsafe fn into_native(source: *mut FfiLlmStreamSource) -> Result<LlmJsonStream> {
    if source.is_null() {
        return Err(FlowError::Internal(
            last_error_message().unwrap_or_else(|| "stream callback returned null".into()),
        ));
    }
    let source = unsafe { Box::from_raw(source) }.0;
    Ok(LlmJsonStream::from_closeable(SourceAdapter {
        source,
        pending: None,
    }))
}

struct ForeignState {
    context: MiddlewareContinuationContext,
    read: NemoRelayStreamReadFn,
    cancel: crate::api::stream_source::NemoRelayOptionalStreamCancelFn,
    close: crate::api::stream_source::NemoRelayOptionalStreamCloseFn,
    data: *mut c_void,
    free: NemoRelayFreeFn,
    cancelled: AtomicBool,
    closed: OnceLock<std::result::Result<(), String>>,
}
// The constructor contract requires callbacks and state to be thread-safe.
unsafe impl Send for ForeignState {}
unsafe impl Sync for ForeignState {}
impl ForeignState {
    fn cancel(&self) {
        if !self.cancelled.swap(true, Ordering::AcqRel)
            && let Some(cancel) = self.cancel
        {
            self.context.run_sync(|| {
                unsafe { cancel(self.data) };
            });
        }
    }
    fn close(&self) -> Result<()> {
        self.cancel();
        self.closed
            .get_or_init(|| {
                self.context.run_sync(|| {
                    clear_last_error();
                    match self.close {
                        Some(close) if unsafe { close(self.data) } != NemoRelayStatus::Ok => {
                            Err(last_error_message()
                                .unwrap_or_else(|| "stream source close failed".into()))
                        }
                        _ => Ok(()),
                    }
                })
            })
            .clone()
            .map_err(FlowError::Internal)
    }
    fn read(&self) -> Result<Option<Json>> {
        clear_last_error();
        let mut output = std::ptr::null_mut();
        let status = unsafe { (self.read)(self.data, &mut output) };
        let result = match status {
            0 => Ok(None),
            1 if !output.is_null() => {
                serde_json::from_str(&unsafe { std::ffi::CStr::from_ptr(output) }.to_string_lossy())
                    .map(Some)
                    .map_err(|e| FlowError::Internal(format!("invalid stream chunk: {e}")))
            }
            _ => Err(FlowError::Internal(
                last_error_message().unwrap_or_else(|| "stream source read failed".into()),
            )),
        };
        unsafe { nemo_relay_string_free(output) };
        result
    }
}
impl Drop for ForeignState {
    fn drop(&mut self) {
        let _ = self.close();
        if let Some(free) = self.free {
            unsafe { free(self.data) };
        }
    }
}
struct ForeignSource {
    state: Arc<ForeignState>,
    pending: Option<tokio::task::JoinHandle<Result<Option<Json>>>>,
    done: bool,
}
impl Stream for ForeignSource {
    type Item = Result<Json>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.done {
            return Poll::Ready(None);
        }
        let pending = this.pending.get_or_insert_with(|| {
            let state = this.state.clone();
            let context = MiddlewareContinuationContext::capture();
            tokio::task::spawn_blocking(move || {
                tokio_runtime().block_on(context.run(async move { state.read() }))
            })
        });
        match Pin::new(pending).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                this.pending = None;
                let item = result.unwrap_or_else(|e| Err(FlowError::Internal(e.to_string())));
                this.done = !matches!(item, Ok(Some(_)));
                Poll::Ready(item.transpose())
            }
        }
    }
}
impl LlmStreamInner for ForeignSource {
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let this = self.get_mut();
        this.state.cancel();
        this.done = true;
        Box::pin(async move {
            if let Some(pending) = this.pending.take() {
                let _ = pending.await;
            }
            let state = this.state.clone();
            tokio::task::spawn_blocking(move || state.close())
                .await
                .map_err(|e| FlowError::Internal(e.to_string()))?
        })
    }
}
impl Drop for ForeignSource {
    fn drop(&mut self) {
        self.state.cancel();
    }
}

/// Construct a foreign stream source. Relay owns the state when the call starts,
/// even if it fails. The free callback runs exactly once after reads and close finish.
/// Reads run on blocking workers. They restore the Relay middleware context
/// so callbacks can emit observability events.
/// Close callbacks restore the source creation scope and active trace context.
/// Callbacks can use supplied capabilities and emit observability events.
/// Other API calls, including managed-stream control and flushes, are invalid.
///
/// # Safety
/// Callbacks and state must stay valid and thread-safe until free. Cancel must
/// promptly unblock reads. Close runs once after reads finish. Output strings
/// transfer ownership and must follow `nemo_relay_string_free`'s allocator contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_stream_source_create(
    read: crate::api::stream_source::NemoRelayOptionalStreamReadFn,
    cancel: crate::api::stream_source::NemoRelayOptionalStreamCancelFn,
    close: crate::api::stream_source::NemoRelayOptionalStreamCloseFn,
    data: *mut c_void,
    free: NemoRelayFreeFn,
) -> *mut FfiLlmStreamSource {
    clear_last_error();
    let Some(read) = read else {
        if let Some(free) = free {
            unsafe { free(data) };
        }
        set_last_error("stream read callback is required");
        return std::ptr::null_mut();
    };
    from_native(LlmJsonStream::from_closeable(ForeignSource {
        state: Arc::new(ForeignState {
            context: MiddlewareContinuationContext::capture(),
            read,
            cancel,
            close,
            data,
            free,
            cancelled: AtomicBool::new(false),
            closed: OnceLock::new(),
        }),
        pending: None,
        done: false,
    }))
}
/// Read an owned source: 1 chunk, 0 EOF, -1 error. Returned strings are owned.
/// # Safety
/// The source and output must be live. Do not free the source during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_stream_source_next(
    source: *const FfiLlmStreamSource,
    output: *mut *mut c_char,
) -> i32 {
    clear_last_error();
    if source.is_null() || output.is_null() {
        set_last_error("null source or output");
        return -1;
    }
    unsafe { *output = std::ptr::null_mut() };
    match block_on(unsafe { &*source }.0.next()) {
        Ok(Some(value)) => {
            unsafe { *output = json_to_c_string(&value) };
            1
        }
        Ok(None) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}
// Preserve middleware context for events emitted during deferred cleanup.
fn close_source_in_background(source: Arc<SourceHandle>) {
    source.cancel.send_replace(true);
    let context = MiddlewareContinuationContext::capture();
    tokio_runtime().spawn(async move {
        context
            .run(async move {
                let _ = source.close().await;
            })
            .await;
    });
}

/// Signal cancellation. This can run during a source read.
/// # Safety
/// Source must be live, or null. Do not free until concurrent calls finish.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_stream_source_cancel(source: *const FfiLlmStreamSource) {
    if let Some(source) = unsafe { source.as_ref() } {
        close_source_in_background(source.0.clone());
    }
}
/// Cancel and await all source cleanup. Does not free the handle.
/// # Safety
/// Source must be a live, non-null owned handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_stream_source_close(
    source: *const FfiLlmStreamSource,
) -> NemoRelayStatus {
    clear_last_error();
    let Some(source) = (unsafe { source.as_ref() }) else {
        set_last_error("null source");
        return NemoRelayStatus::NullPointer;
    };
    match block_on(source.0.close()) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoRelayStatus::Internal
        }
    }
}
/// Release an owned source that has not been transferred to Relay.
/// Close it first to wait for cleanup.
/// # Safety
/// The source must be owned and live, or null. No other call can use it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_stream_source_free(source: *mut FfiLlmStreamSource) {
    if !source.is_null() {
        let source = unsafe { Box::from_raw(source) };
        close_source_in_background(source.0);
    }
}

struct PrefixSource {
    first: Option<Json>,
    tail: Option<LlmJsonStream>,
}
impl Stream for PrefixSource {
    type Item = Result<Json>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if let Some(value) = this.first.take() {
            return Poll::Ready(Some(Ok(value)));
        }
        match this.tail.as_mut() {
            Some(tail) => Pin::new(tail).poll_next(cx),
            None => Poll::Ready(None),
        }
    }
}
impl LlmStreamInner for PrefixSource {
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let this = self.get_mut();
        this.first = None;
        Box::pin(async move {
            if let Some(mut tail) = this.tail.take() {
                tail.close().await?;
            }
            Ok(())
        })
    }
}
pub(crate) fn prefix(first: Json, tail: Option<LlmJsonStream>) -> LlmJsonStream {
    LlmJsonStream::from_closeable(PrefixSource {
        first: Some(first),
        tail,
    })
}

tokio::task_local! {
    // Retained continuations need this data to associate binding-owned sources.
    pub(crate) static STREAM_CALLBACK_STATE: Arc<crate::callable::StreamCallbackState>;
}

/// An owned streaming continuation, retained before its callback returns.
#[derive(Clone)]
pub struct FfiLlmStreamContinuation(
    pub(crate) nemo_relay_runtime::api::runtime::LlmStreamExecutionNextFn,
    Arc<crate::callable::StreamCallbackState>,
);
impl FfiLlmStreamContinuation {
    pub(crate) fn capture(
        next: nemo_relay_runtime::api::runtime::LlmStreamExecutionNextFn,
    ) -> Self {
        use nemo_relay_runtime::api::runtime::scope_stack::{
            active_event_trace_context, active_event_uuid, with_active_event_trace_context,
        };
        let stack = nemo_relay_runtime::api::runtime::scope_stack::snapshot_scope_stack(
            &entry_scope_stack(),
        );
        let event = active_event_uuid();
        let trace = active_event_trace_context();
        let callbacks = STREAM_CALLBACK_STATE
            .try_with(Arc::clone)
            .unwrap_or_default();
        let continuation_callbacks = callbacks.clone();
        Self(
            Arc::new(move |request| {
                let next = next.clone();
                let callbacks = continuation_callbacks.clone();
                let stack = stack.clone();
                let trace = trace.clone();
                Box::pin(async move {
                    let stack = stack?;
                    nemo_relay_runtime::api::runtime::TASK_SCOPE_STACK
                        .scope(
                            stack,
                            STREAM_CALLBACK_STATE.scope(callbacks, async move {
                                match event {
                                    Some(uuid) => {
                                        with_active_event_trace_context(uuid, trace, next(request))
                                            .await
                                    }
                                    None => next(request).await,
                                }
                            }),
                        )
                        .await
                })
            }),
            callbacks,
        )
    }
}
/// Borrow the managed stream's collector/finalizer user data from an interceptor
/// continuation. This lets language bindings associate provider sources with
/// their language owner even when a continuation runs on a different native thread.
/// Returns null for a null continuation or absent callback state.
///
/// # Safety
/// Continuation must be live. The returned data is borrowed and may only be
/// accessed while the stream interceptor callback is active. Do not free it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_stream_continuation_callback_data(
    context: *const FfiLlmStreamContinuation,
) -> *mut c_void {
    unsafe { context.as_ref() }.map_or(std::ptr::null_mut(), |context| context.1.user_data_ptr())
}

/// Retain a streaming next-context before its interceptor callback returns.
/// The result owns the continuation and its original scope and trace context.
/// # Safety
/// Context must be the live next-context from a full-stream interceptor callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_stream_continuation_retain(
    context: *const FfiLlmStreamContinuation,
) -> *mut FfiLlmStreamContinuation {
    clear_last_error();
    match unsafe { context.as_ref() } {
        Some(context) => Box::into_raw(Box::new(context.clone())),
        None => {
            set_last_error("null streaming continuation");
            std::ptr::null_mut()
        }
    }
}
/// Invoke a live borrowed or retained streaming continuation. Returns an owned source.
/// # Safety
/// Continuation must be live and request must be a valid LlmRequest JSON C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_stream_continuation_call(
    context: *const FfiLlmStreamContinuation,
    request: *const c_char,
) -> *mut FfiLlmStreamSource {
    clear_last_error();
    let Some(context) = (unsafe { context.as_ref() }) else {
        set_last_error("null streaming continuation");
        return std::ptr::null_mut();
    };
    let request =
        crate::convert::c_str_to_json(request).and_then(|value| serde_json::from_value(value).ok());
    let Some(request) = request else {
        set_last_error("invalid streaming continuation request");
        return std::ptr::null_mut();
    };
    match block_on((context.0)(request)) {
        Ok(stream) => from_native(stream),
        Err(error) => {
            set_last_error(&error.to_string());
            std::ptr::null_mut()
        }
    }
}
/// Release a retained continuation. Never free a borrowed next-context.
/// # Safety
/// The context must be owned and live, or null. No other call can use it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_stream_continuation_free(
    context: *mut FfiLlmStreamContinuation,
) {
    if !context.is_null() {
        drop(unsafe { Box::from_raw(context) });
    }
}
