// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::type_complexity)]
//! C function pointer typedefs and wrapper functions for FFI callbacks.
//!
//! This module defines the callback signatures used by the C API for tool and
//! LLM guardrails, intercepts, execution functions, and event subscribers. Each
//! `pub type` alias corresponds to a C function pointer that appears in the
//! generated `nemo_relay.h` header.
//!
//! The `wrap_*` functions convert C callbacks (with opaque `user_data` pointers)
//! into Rust closures that the core runtime can invoke. Registry-stored
//! callbacks return `Arc`-backed closures, while one-shot or mutable callback
//! shapes remain boxed. Each wrapper captures the user data and its optional
//! free function in an `Arc<UserData>` so the closure is `Send + Sync` and the
//! free function is called exactly once when all references are dropped.

use std::ffi::{CStr, CString};
use std::future::Future;
use std::pin::Pin;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use libc::c_char;
use nemo_relay::api::runtime::{
    EventSanitizeFn, EventSubscriberFn, LlmCodecIdentity, LlmConditionalFn, LlmExecutionFn,
    LlmExecutionNextFn, LlmJsonStream, LlmRequestInterceptFn, LlmSanitizeRequestContext,
    LlmSanitizeRequestFn, LlmSanitizeResponseContext, LlmSanitizeResponseFn, LlmStreamExecutionFn,
    LlmStreamExecutionNextFn, ToolConditionalFn, ToolExecutionFn, ToolExecutionFrameFn,
    ToolExecutionFrameNextFn, ToolExecutionNextFn, ToolInterceptFn, ToolSanitizeFn,
};
use serde_json::Value as Json;
use tokio_stream::StreamExt;

use nemo_relay::api::event::{Event, EventSanitizeFields};
use nemo_relay::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
use nemo_relay::api::tool::{
    ToolExecutionFrame, ToolExecutionFrameOutcome, ToolExecutionInterceptOutcome,
};
use nemo_relay::codec::request::AnnotatedLlmRequest as AnnotatedLLMRequest;
use nemo_relay::codec::traits::LlmCodec;
use nemo_relay::error::{FlowError, Result};

use crate::convert::{c_str_to_json, json_to_c_string};
use crate::error::{NemoRelayStatus, clear_last_error, last_error_message, set_last_error};
use crate::types::{FfiEvent, FfiLLMRequest, FfiPluginContext};

// ---------------------------------------------------------------------------
// Callback typedefs (mirrored in the C header)
// ---------------------------------------------------------------------------

/// Optional destructor for user data passed to callbacks.
/// Called when the runtime no longer needs the associated callback.
pub type NemoRelayFreeFn = Option<unsafe extern "C" fn(user_data: *mut libc::c_void)>;

/// Indicates whether an async callback settled its completion before returning.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemoRelayAsyncCallbackState {
    /// The callback called a resolve/reject function before returning.
    Complete = 0,
    /// The callback retained the completion and will settle it later.
    Pending = 1,
}

/// One-shot completion passed to asynchronous C callbacks.
pub struct NemoRelayAsyncCompletion {
    sender: std::sync::Mutex<Option<tokio::sync::oneshot::Sender<Result<Json>>>>,
    cancelled: AtomicBool,
}

/// Generic completion-based middleware callback.
///
/// `invocation_json` is borrowed for the duration of the call. The completion
/// has one callback-owned reference. A callback returning `Complete` need not
/// release it; a callback returning `Pending` must eventually settle and call
/// `nemo_relay_async_completion_release`.
pub type NemoRelayAsyncJsonCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    invocation_json: *const c_char,
    completion: *const NemoRelayAsyncCompletion,
) -> NemoRelayAsyncCallbackState;

/// Runtime-owned asynchronous `next` continuation for execution intercepts.
pub struct NemoRelayAsyncNext {
    inner: AsyncNextInner,
    runtime: tokio::runtime::Handle,
}

enum AsyncNextInner {
    Tool(ToolExecutionNextFn),
    Llm(LlmExecutionNextFn),
    LlmStream(LlmStreamExecutionNextFn),
}

const ASYNC_STREAM_MAX_CHUNKS: usize = 4096;
const ASYNC_STREAM_MAX_SERIALIZED_BYTES: usize = 16 * 1024 * 1024;

async fn collect_async_stream_for_completion(mut stream: LlmJsonStream) -> Result<Json> {
    let mut chunks = Vec::new();
    let mut serialized_bytes = 0usize;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if chunks.len() >= ASYNC_STREAM_MAX_CHUNKS {
            return Err(FlowError::Internal(format!(
                "async stream continuation exceeded the {ASYNC_STREAM_MAX_CHUNKS}-chunk completion limit"
            )));
        }
        serialized_bytes = serialized_bytes.saturating_add(
            serde_json::to_vec(&chunk)
                .map_err(|error| {
                    FlowError::Internal(format!("failed to measure async stream chunk: {error}"))
                })?
                .len(),
        );
        if serialized_bytes > ASYNC_STREAM_MAX_SERIALIZED_BYTES {
            return Err(FlowError::Internal(format!(
                "async stream continuation exceeded the {ASYNC_STREAM_MAX_SERIALIZED_BYTES}-byte completion limit"
            )));
        }
        chunks.push(chunk);
    }
    Ok(Json::Array(chunks))
}

/// Completion-based execution-intercept callback.
pub type NemoRelayAsyncInterceptCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    invocation_json: *const c_char,
    next: *const NemoRelayAsyncNext,
    completion: *const NemoRelayAsyncCompletion,
) -> NemoRelayAsyncCallbackState;

/// Result callback used by channel/future-style async `next` wrappers.
///
/// Invoked on a Tokio runtime worker thread, not necessarily the thread that
/// called `nemo_relay_async_next_invoke_callback`; `user_data` must therefore
/// be safe for cross-thread use. `value_json` and `error_message` are borrowed
/// for the duration of the callback only.
pub type NemoRelayAsyncNextResultCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    value_json: *const c_char,
    error_message: *const c_char,
);

struct SendUserData(*mut libc::c_void);

// SAFETY: NemoRelayAsyncNextResultCb requires callers to keep user_data valid
// and safe to access until the asynchronously invoked callback runs.
unsafe impl Send for SendUserData {}

impl SendUserData {
    fn as_ptr(&self) -> *mut libc::c_void {
        self.0
    }
}

struct CompletionWait {
    completion: Arc<NemoRelayAsyncCompletion>,
    receiver: tokio::sync::oneshot::Receiver<Result<Json>>,
}

impl Drop for CompletionWait {
    fn drop(&mut self) {
        self.completion.cancelled.store(true, Ordering::Release);
    }
}

async fn invoke_async_json(
    cb: NemoRelayAsyncJsonCb,
    user_data: Arc<UserData>,
    invocation: Json,
) -> Result<Json> {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NemoRelayAsyncCompletion {
        sender: std::sync::Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
    });
    let callback_ref = Arc::into_raw(completion.clone());
    let invocation = json_to_c_string(&invocation);
    let state = unsafe { cb(user_data.ptr, invocation, callback_ref) };
    unsafe { nemo_relay_string_free_internal(invocation) };
    if state == NemoRelayAsyncCallbackState::Complete {
        unsafe { drop(Arc::from_raw(callback_ref)) };
        if completion
            .sender
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_some()
        {
            return Err(FlowError::Internal(
                "async C callback returned Complete without settling".into(),
            ));
        }
    }
    let mut wait = CompletionWait {
        completion,
        receiver,
    };
    (&mut wait.receiver)
        .await
        .map_err(|_| FlowError::Internal("async C callback dropped without settling".into()))?
}

async fn invoke_async_intercept(
    cb: NemoRelayAsyncInterceptCb,
    user_data: Arc<UserData>,
    invocation: Json,
    next: AsyncNextInner,
) -> Result<Json> {
    let runtime = tokio::runtime::Handle::try_current().map_err(|error| {
        FlowError::Internal(format!(
            "async C intercept requires a Tokio runtime: {error}"
        ))
    })?;
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NemoRelayAsyncCompletion {
        sender: std::sync::Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
    });
    let callback_ref = Arc::into_raw(completion.clone());
    let next = Arc::new(NemoRelayAsyncNext {
        inner: next,
        runtime,
    });
    let next_ref = Arc::into_raw(next);
    let invocation = json_to_c_string(&invocation);
    let state = unsafe { cb(user_data.ptr, invocation, next_ref, callback_ref) };
    unsafe { nemo_relay_string_free_internal(invocation) };
    if state == NemoRelayAsyncCallbackState::Complete {
        unsafe { drop(Arc::from_raw(callback_ref)) };
        unsafe { drop(Arc::from_raw(next_ref)) };
        if completion
            .sender
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_some()
        {
            return Err(FlowError::Internal(
                "async C intercept returned Complete without settling".into(),
            ));
        }
    }
    let mut wait = CompletionWait {
        completion,
        receiver,
    };
    (&mut wait.receiver)
        .await
        .map_err(|_| FlowError::Internal("async C intercept dropped without settling".into()))?
}

/// Release the callback-owned async `next` reference after a pending intercept.
#[allow(clippy::missing_safety_doc)] // The shared C ABI safety contract applies.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_async_next_release(next: *const NemoRelayAsyncNext) {
    if !next.is_null() {
        unsafe { drop(Arc::from_raw(next)) };
    }
}

/// Invoke the next execution layer and settle `completion` with its result.
///
/// A non-`Ok` return means invocation was not scheduled and never settles
/// `completion`; the caller remains responsible for rejecting or releasing it.
#[allow(clippy::missing_safety_doc)] // The shared C ABI safety contract applies.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_async_next_invoke(
    next: *const NemoRelayAsyncNext,
    invocation_json: *const c_char,
    completion: *const NemoRelayAsyncCompletion,
) -> NemoRelayStatus {
    let Some(next) = (unsafe { next.as_ref() }) else {
        return NemoRelayStatus::NullPointer;
    };
    if completion.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let Some(invocation) = c_str_to_json(invocation_json) else {
        return NemoRelayStatus::InvalidJson;
    };
    unsafe { Arc::increment_strong_count(completion) };
    let completion = unsafe { Arc::from_raw(completion) };
    let future: Pin<Box<dyn Future<Output = Result<Json>> + Send>> = match &next.inner {
        AsyncNextInner::Tool(next) => {
            let next = next.clone();
            Box::pin(async move {
                let outcome = next(invocation).await?;
                serde_json::to_value(outcome)
                    .map_err(|error| FlowError::Internal(error.to_string()))
            })
        }
        AsyncNextInner::Llm(next) => {
            let request = match serde_json::from_value(invocation) {
                Ok(request) => request,
                Err(_) => return NemoRelayStatus::InvalidJson,
            };
            let next = next.clone();
            Box::pin(async move { next(request).await })
        }
        AsyncNextInner::LlmStream(next) => {
            let request = match serde_json::from_value(invocation) {
                Ok(request) => request,
                Err(_) => return NemoRelayStatus::InvalidJson,
            };
            let next = next.clone();
            Box::pin(async move { collect_async_stream_for_completion(next(request).await?).await })
        }
    };
    next.runtime.spawn(async move {
        let result = future.await;
        if let Some(sender) = completion
            .sender
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            let _ = sender.send(result);
        }
    });
    NemoRelayStatus::Ok
}

/// Invoke the next execution layer and report its result through a callback.
#[allow(clippy::missing_safety_doc)] // The shared C ABI safety contract applies.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_async_next_invoke_callback(
    next: *const NemoRelayAsyncNext,
    invocation_json: *const c_char,
    callback: NemoRelayAsyncNextResultCb,
    user_data: *mut libc::c_void,
) -> NemoRelayStatus {
    let Some(next) = (unsafe { next.as_ref() }) else {
        return NemoRelayStatus::NullPointer;
    };
    let Some(invocation) = c_str_to_json(invocation_json) else {
        return NemoRelayStatus::InvalidJson;
    };
    let future: Pin<Box<dyn Future<Output = Result<Json>> + Send>> = match &next.inner {
        AsyncNextInner::Tool(next) => {
            let next = next.clone();
            Box::pin(async move {
                serde_json::to_value(next(invocation).await?)
                    .map_err(|error| FlowError::Internal(error.to_string()))
            })
        }
        AsyncNextInner::Llm(next) => {
            let request = match serde_json::from_value(invocation) {
                Ok(request) => request,
                Err(_) => return NemoRelayStatus::InvalidJson,
            };
            let next = next.clone();
            Box::pin(async move { next(request).await })
        }
        AsyncNextInner::LlmStream(next) => {
            let request = match serde_json::from_value(invocation) {
                Ok(request) => request,
                Err(_) => return NemoRelayStatus::InvalidJson,
            };
            let next = next.clone();
            Box::pin(async move { collect_async_stream_for_completion(next(request).await?).await })
        }
    };
    let user_data = SendUserData(user_data);
    next.runtime.spawn(async move {
        match future.await {
            Ok(value) => {
                let value = json_to_c_string(&value);
                unsafe { callback(user_data.as_ptr(), value, ptr::null()) };
                unsafe { nemo_relay_string_free_internal(value) };
            }
            Err(error) => {
                let error = CString::new(error.to_string()).unwrap_or_default();
                unsafe { callback(user_data.as_ptr(), ptr::null(), error.as_ptr()) };
            }
        }
    });
    NemoRelayStatus::Ok
}

/// Resolve an async C callback with owned JSON.
#[allow(clippy::missing_safety_doc)] // The shared C ABI safety contract applies.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_async_completion_resolve_json(
    completion: *const NemoRelayAsyncCompletion,
    value_json: *const c_char,
) -> NemoRelayStatus {
    let Some(completion) = (unsafe { completion.as_ref() }) else {
        return NemoRelayStatus::NullPointer;
    };
    if completion.cancelled.load(Ordering::Acquire) {
        return NemoRelayStatus::InvalidArg;
    }
    let Some(value) = c_str_to_json(value_json) else {
        return NemoRelayStatus::InvalidJson;
    };
    let Some(sender) = completion
        .sender
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .take()
    else {
        return NemoRelayStatus::InvalidArg;
    };
    let _ = sender.send(Ok(value));
    NemoRelayStatus::Ok
}

/// Reject an async C callback with an error message.
#[allow(clippy::missing_safety_doc)] // The shared C ABI safety contract applies.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_async_completion_reject(
    completion: *const NemoRelayAsyncCompletion,
    message: *const c_char,
) -> NemoRelayStatus {
    let Some(completion) = (unsafe { completion.as_ref() }) else {
        return NemoRelayStatus::NullPointer;
    };
    if completion.cancelled.load(Ordering::Acquire) {
        return NemoRelayStatus::InvalidArg;
    }
    let message = if message.is_null() {
        "async C callback rejected".to_string()
    } else {
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    };
    let Some(sender) = completion
        .sender
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .take()
    else {
        return NemoRelayStatus::InvalidArg;
    };
    let _ = sender.send(Err(FlowError::Internal(message)));
    NemoRelayStatus::Ok
}

/// Returns whether an async completion's invocation has been cancelled.
#[allow(clippy::missing_safety_doc)] // The shared C ABI safety contract applies.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_async_completion_is_cancelled(
    completion: *const NemoRelayAsyncCompletion,
) -> bool {
    unsafe { completion.as_ref() }
        .is_none_or(|completion| completion.cancelled.load(Ordering::Acquire))
}

/// Release the callback-owned completion reference after a pending invocation.
#[allow(clippy::missing_safety_doc)] // The shared C ABI safety contract applies.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_async_completion_release(
    completion: *const NemoRelayAsyncCompletion,
) {
    if !completion.is_null() {
        unsafe { drop(Arc::from_raw(completion)) };
    }
}

/// Callback for tool request/response sanitization guardrails and intercepts.
/// Receives tool name and arguments as JSON, returns sanitized arguments as JSON.
/// The returned string must be allocated with `malloc` or equivalent.
pub type NemoRelayToolSanitizeCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    name: *const c_char,
    args_json: *const c_char,
) -> *mut c_char;

/// Callback for tool conditional execution guardrails.
/// Receives tool name and arguments as JSON.
/// Returns NULL to allow execution, or an error message string to reject.
pub type NemoRelayToolConditionalCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    name: *const c_char,
    args_json: *const c_char,
) -> *mut c_char;

/// Callback for tool execution (default callable). Receives arguments as JSON,
/// returns result as JSON. The returned string must be allocated with `malloc`
/// or equivalent.
pub type NemoRelayToolExecCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, args_json: *const c_char) -> *mut c_char;

/// Annotation-aware tool execution callback.
///
/// The returned JSON must serialize a `ToolExecutionFrame`.
pub type NemoRelayToolExecFrameCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, args_json: *const c_char) -> *mut c_char;

/// Runtime-provided "next" callback for tool execution middleware chain.
/// Call this from an intercept to invoke the next layer (or original function).
/// `next_ctx` is an opaque pointer managed by the runtime.
pub type NemoRelayToolExecNextFn =
    unsafe extern "C" fn(args_json: *const c_char, next_ctx: *mut libc::c_void) -> *mut c_char;

/// Runtime-provided annotation-aware continuation.
///
/// The returned JSON serializes a `ToolExecutionFrame`.
pub type NemoRelayToolExecFrameNextFn =
    unsafe extern "C" fn(args_json: *const c_char, next_ctx: *mut libc::c_void) -> *mut c_char;

/// Callback for tool execution intercepts. Receives arguments as JSON plus
/// a `next` callback and its context. Call `next_fn(args, next_ctx)` to invoke
/// the next layer in the middleware chain, or return directly to short-circuit.
/// The `result` field is passed to the remaining middleware and application;
/// `pending_marks` are Relay-owned lifecycle metadata emitted after the
/// tool-end event and are not included in the application-visible result.
/// The returned JSON must contain a `result` field and may contain a
/// `pending_marks` array. The returned string must be allocated with `malloc`
/// or an equivalent allocation compatible with `nemo_relay_string_free`.
/// Ownership transfers to Relay when the callback returns; the callback must
/// not free or reuse the string afterward, and Relay frees it exactly once.
pub type NemoRelayToolExecInterceptCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    args_json: *const c_char,
    next_fn: NemoRelayToolExecNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char;

/// Annotation-aware tool execution intercept callback.
///
/// `next_fn` returns a serialized `ToolExecutionFrame`; this callback must
/// return a serialized `ToolExecutionFrameOutcome`.
pub type NemoRelayToolExecFrameInterceptCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    args_json: *const c_char,
    next_fn: NemoRelayToolExecFrameNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char;

/// Codec identity kind supplied to an LLM sanitizer.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemoRelayLlmSanitizeCodecKind {
    /// No codec was active.
    None = 0,
    /// A Relay built-in codec was active.
    BuiltIn = 1,
    /// A runtime-registered codec was active.
    Runtime = 2,
    /// A codec was active but has no registered identity.
    Opaque = 3,
}

/// Codec identity supplied to an LLM sanitizer. `codec_id` is null for
/// `None` and `Opaque`, and is valid only for the duration of the callback.
#[repr(C)]
pub struct NemoRelayLlmSanitizeRequestContext {
    /// Kind of active codec identity.
    pub codec_kind: NemoRelayLlmSanitizeCodecKind,
    /// Built-in or runtime codec ID, when applicable.
    pub codec_id: *const c_char,
    /// Borrowed request codec capability, or null when no codec is active.
    pub codec: *const crate::types::FfiLlmSanitizeRequestCodec,
}

/// Directional codec context supplied to an LLM response sanitizer.
#[repr(C)]
pub struct NemoRelayLlmSanitizeResponseContext {
    /// Kind of active codec identity.
    pub codec_kind: NemoRelayLlmSanitizeCodecKind,
    /// Built-in or runtime codec ID, when applicable.
    pub codec_id: *const c_char,
    /// Borrowed response codec capability, or null when no codec is active.
    pub codec: *const crate::types::FfiLlmSanitizeResponseCodec,
}

/// LLM request sanitizer. It receives the request first and its codec context
/// second. Return null to omit the observability payload. The request is
/// borrowed, but returning that same pointer is supported as a pass-through.
/// Any other non-null result transfers ownership to Relay.
pub type NemoRelayLlmSanitizeRequestCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
    context: NemoRelayLlmSanitizeRequestContext,
) -> *mut FfiLLMRequest;

/// LLM response sanitizer. It receives response JSON first and its codec
/// context second. Return null to omit the observability payload. The response
/// is borrowed, but returning that same pointer is supported as a pass-through.
/// Any other non-null result transfers ownership to Relay.
pub type NemoRelayLlmSanitizeResponseCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    response_json: *const c_char,
    context: NemoRelayLlmSanitizeResponseContext,
) -> *mut c_char;

/// Callback for LLM conditional execution guardrails.
/// Returns NULL to allow execution, or an error message string to reject.
pub type NemoRelayLlmConditionalCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut c_char;

/// Callback for LLM execution (default callable). Receives a native JSON C string,
/// returns the response as a JSON C string.
pub type NemoRelayLlmExecCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, native_json: *const c_char) -> *mut c_char;

/// Runtime-provided "next" callback for LLM execution middleware chain.
/// Takes a native JSON C string, returns a response JSON C string.
pub type NemoRelayLlmExecNextFn =
    unsafe extern "C" fn(native_json: *const c_char, next_ctx: *mut libc::c_void) -> *mut c_char;

/// Callback for LLM execution intercepts with middleware chain support.
/// Receives native JSON C string plus a `next` callback and its context.
pub type NemoRelayLlmExecInterceptCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    native_json: *const c_char,
    next_fn: NemoRelayLlmExecNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char;

/// Callback for event subscribers. Invoked on each lifecycle event emitted by
/// the runtime. The `FfiEvent` pointer is only valid for the duration of the call.
pub type NemoRelayEventSubscriberCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, event: *const FfiEvent);

/// Callback for mark and scope event sanitizers.
/// The returned JSON string transfers to Relay and is freed exactly once.
pub type NemoRelayEventSanitizeCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    event: *const FfiEvent,
    fields_json: *const c_char,
) -> *mut c_char;

/// Callback for Codec decode: translates an opaque `FfiLLMRequest` into
/// an `AnnotatedLLMRequest` JSON string. Returns a heap-allocated C string
/// on success, or null on error (after setting the last error message).
pub type NemoRelayCodecDecodeCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut c_char;

/// Nullable version of [`NemoRelayCodecDecodeCb`] for use as an optional
/// parameter in FFI execute functions. Pass null to indicate no codec.
pub type NemoRelayCodecDecodeFn = Option<
    unsafe extern "C" fn(
        user_data: *mut libc::c_void,
        request: *const FfiLLMRequest,
    ) -> *mut c_char,
>;

/// Callback for Codec encode: merges structured changes back into opaque
/// request content. Receives the annotated request as a JSON C string and
/// the original `FfiLLMRequest`. Returns a heap-allocated JSON C string
/// representing the new `LlmRequest` content on success, or null on error.
pub type NemoRelayCodecEncodeCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    annotated_json: *const c_char,
    original_request: *const FfiLLMRequest,
) -> *mut c_char;

/// Nullable version of [`NemoRelayCodecEncodeCb`] for use as an optional
/// parameter in FFI execute functions. Pass null to indicate no codec.
pub type NemoRelayCodecEncodeFn = Option<
    unsafe extern "C" fn(
        user_data: *mut libc::c_void,
        annotated_json: *const c_char,
        original_request: *const FfiLLMRequest,
    ) -> *mut c_char,
>;

/// C callback type for LLM request intercepts with unified annotated-aware
/// signature. Receives the intercept name, the opaque `FfiLLMRequest`, and
/// optionally the annotated request as a JSON C string (null if no Codec
/// resolved). Writes one owned canonical outcome JSON string to
/// `out_outcome_json`. Any non-null string written there must be allocated by
/// `nemo_relay_llm_request_intercept_outcome_json_new` or by an allocation
/// compatible with `nemo_relay_string_free`. Ownership transfers to Relay
/// when the callback returns; the callback must not free or reuse the string
/// afterward. Relay frees it exactly once, even when the callback returns an
/// error status. With a Codec, the outcome must preserve request content and
/// return the annotation; only request headers and annotation fields are
/// writable. Returns `NemoRelayStatus`.
pub type NemoRelayLlmRequestInterceptCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    name: *const c_char,
    request: *const FfiLLMRequest,
    annotated_json: *const c_char,
    out_outcome_json: *mut *mut c_char,
) -> NemoRelayStatus;

/// Callback for collecting intercepted stream chunks. Invoked with each chunk
/// (after stream execution intercepts have been applied) as a null-terminated
/// C string. The string is only valid for the duration of the call.
pub type NemoRelayCollectorCb = unsafe extern "C" fn(chunk: *const c_char);

/// Callback for finalizing a collected stream. Invoked once when the stream is
/// exhausted. Must return a JSON C string representing the aggregated response.
/// The returned string must be allocated with `malloc` or equivalent; the
/// runtime will free it.
pub type NemoRelayFinalizerCb = unsafe extern "C" fn() -> *mut c_char;

/// Callback for plugin validation.
/// Receives plugin config JSON and returns a JSON array of diagnostics.
pub type NemoRelayPluginValidateCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    plugin_config_json: *const c_char,
) -> *mut c_char;

/// Callback for plugin registration.
/// Receives plugin config JSON and a plugin context pointer that is
/// only valid for the duration of the call.
pub type NemoRelayPluginRegisterCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    plugin_config_json: *const c_char,
    ctx: *mut FfiPluginContext,
) -> NemoRelayStatus;

// ---------------------------------------------------------------------------
// Shared user_data wrapper (ensures cleanup)
// ---------------------------------------------------------------------------

/// RAII wrapper around a C user-data pointer and its associated free function.
/// Ensures the free function is called exactly once when dropped.
struct UserData {
    ptr: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
}

unsafe impl Send for UserData {}
unsafe impl Sync for UserData {}

impl Drop for UserData {
    fn drop(&mut self) {
        if let Some(free) = self.free_fn {
            unsafe { free(self.ptr) };
        }
    }
}

fn make_user_data(
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> std::sync::Arc<UserData> {
    std::sync::Arc::new(UserData {
        ptr: user_data,
        free_fn,
    })
}

/// Wrap a completion-based C tool sanitizer or request intercept.
pub fn wrap_async_tool_json_fn(
    cb: NemoRelayAsyncJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> ToolSanitizeFn {
    let user_data = make_user_data(user_data, free_fn);
    Arc::new(move |name: String, value: Json| {
        let user_data = user_data.clone();
        Box::pin(invoke_async_json(
            cb,
            user_data,
            serde_json::json!({"name": name, "value": value}),
        ))
    })
}

/// Wrap a completion-based C tool conditional guardrail.
pub fn wrap_async_tool_conditional_fn(
    cb: NemoRelayAsyncJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> ToolConditionalFn {
    let user_data = make_user_data(user_data, free_fn);
    Arc::new(move |name: String, value: Json| {
        let user_data = user_data.clone();
        Box::pin(async move {
            match invoke_async_json(
                cb,
                user_data,
                serde_json::json!({"name": name, "value": value}),
            )
            .await?
            {
                Json::Null => Ok(None),
                Json::String(reason) => Ok(Some(reason)),
                other => Err(FlowError::Internal(format!(
                    "async conditional callback returned {other}; expected string or null"
                ))),
            }
        })
    })
}

/// Wrap a completion-based C event sanitizer.
pub fn wrap_async_event_sanitize_fn(
    cb: NemoRelayAsyncJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> EventSanitizeFn {
    let user_data = make_user_data(user_data, free_fn);
    Arc::new(move |event: Arc<Event>, fields: EventSanitizeFields| {
        let user_data = user_data.clone();
        Box::pin(async move {
            let value = invoke_async_json(
                cb,
                user_data,
                serde_json::json!({"event": event, "fields": fields}),
            )
            .await?;
            serde_json::from_value(value)
                .map_err(|error| FlowError::Internal(format!("invalid event fields: {error}")))
        })
    })
}

/// Wrap a completion-based C LLM conditional guardrail.
pub fn wrap_async_llm_conditional_fn(
    cb: NemoRelayAsyncJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> LlmConditionalFn {
    let user_data = make_user_data(user_data, free_fn);
    Arc::new(move |request: LlmRequest| {
        let user_data = user_data.clone();
        Box::pin(async move {
            match invoke_async_json(cb, user_data, serde_json::json!({"request": request})).await? {
                Json::Null => Ok(None),
                Json::String(reason) => Ok(Some(reason)),
                other => Err(FlowError::Internal(format!(
                    "async conditional callback returned {other}; expected string or null"
                ))),
            }
        })
    })
}

/// Wrap a completion-based C LLM request sanitizer.
pub fn wrap_async_llm_sanitize_request_fn(
    cb: NemoRelayAsyncJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> LlmSanitizeRequestFn {
    let user_data = make_user_data(user_data, free_fn);
    Arc::new(
        move |request: LlmRequest, context: LlmSanitizeRequestContext| {
            let user_data = user_data.clone();
            let codec = ffi_codec_identity_json(context.codec());
            Box::pin(async move {
                let codec = codec?;
                let value = invoke_async_json(
                    cb,
                    user_data,
                    serde_json::json!({"request": request, "context": codec}),
                )
                .await?;
                if value.is_null() {
                    Ok(None)
                } else {
                    serde_json::from_value(value)
                        .map(Some)
                        .map_err(|error| FlowError::Internal(error.to_string()))
                }
            })
        },
    )
}

/// Wrap a completion-based C LLM response sanitizer.
pub fn wrap_async_llm_sanitize_response_fn(
    cb: NemoRelayAsyncJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> LlmSanitizeResponseFn {
    let user_data = make_user_data(user_data, free_fn);
    Arc::new(move |response: Json, context: LlmSanitizeResponseContext| {
        let user_data = user_data.clone();
        let codec = ffi_codec_identity_json(context.codec());
        Box::pin(async move {
            let codec = codec?;
            let value = invoke_async_json(
                cb,
                user_data,
                serde_json::json!({"response": response, "context": codec}),
            )
            .await?;
            Ok((!value.is_null()).then_some(value))
        })
    })
}

/// Wrap a completion-based C LLM request intercept.
pub fn wrap_async_llm_request_intercept_fn(
    cb: NemoRelayAsyncJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> LlmRequestInterceptFn {
    let user_data = make_user_data(user_data, free_fn);
    Arc::new(
        move |name: String, request: LlmRequest, annotated: Option<AnnotatedLLMRequest>| {
            let user_data = user_data.clone();
            Box::pin(async move {
                let value = invoke_async_json(
                    cb,
                    user_data,
                    serde_json::json!({
                        "name": name,
                        "request": request,
                        "annotated": annotated,
                    }),
                )
                .await?;
                serde_json::from_value(value).map_err(|error| {
                    FlowError::Internal(format!("invalid LLM request intercept outcome: {error}"))
                })
            })
        },
    )
}

/// Wrap a completion-based C tool execution intercept.
pub fn wrap_async_tool_execution_intercept_fn(
    cb: NemoRelayAsyncInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> ToolExecutionFn {
    let user_data = make_user_data(user_data, free_fn);
    Arc::new(move |name: &str, args: Json, next: ToolExecutionNextFn| {
        let user_data = user_data.clone();
        let invocation = serde_json::json!({"name": name, "value": args});
        Box::pin(async move {
            let value =
                invoke_async_intercept(cb, user_data, invocation, AsyncNextInner::Tool(next))
                    .await?;
            serde_json::from_value(value)
                .map_err(|error| FlowError::Internal(format!("invalid tool outcome: {error}")))
        })
    })
}

/// Wrap a completion-based C LLM execution intercept.
pub fn wrap_async_llm_execution_intercept_fn(
    cb: NemoRelayAsyncInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> LlmExecutionFn {
    let user_data = make_user_data(user_data, free_fn);
    Arc::new(
        move |name: &str, request: LlmRequest, next: LlmExecutionNextFn| {
            let user_data = user_data.clone();
            let invocation = serde_json::json!({"name": name, "request": request});
            Box::pin(invoke_async_intercept(
                cb,
                user_data,
                invocation,
                AsyncNextInner::Llm(next),
            ))
        },
    )
}

/// Wrap a completion-based C LLM stream execution intercept.
///
/// The completion ABI resolves one JSON value, so a stream intercept must
/// resolve to an array of chunks. Relay replays that array as a stream after
/// completion; incremental chunk delivery is not available through this ABI.
/// Relay rejects more than 4096 chunks or 16 MiB of serialized chunk data.
pub fn wrap_async_llm_stream_execution_intercept_fn(
    cb: NemoRelayAsyncInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> LlmStreamExecutionFn {
    let user_data = make_user_data(user_data, free_fn);
    Arc::new(
        move |name: &str, request: LlmRequest, next: LlmStreamExecutionNextFn| {
            let user_data = user_data.clone();
            let invocation = serde_json::json!({"name": name, "request": request});
            Box::pin(async move {
                let value = invoke_async_intercept(
                    cb,
                    user_data,
                    invocation,
                    AsyncNextInner::LlmStream(next),
                )
                .await?;
                let chunks = value.as_array().cloned().ok_or_else(|| {
                    FlowError::Internal("async stream intercept must resolve to an array".into())
                })?;
                Ok(LlmJsonStream::new(tokio_stream::iter(
                    chunks.into_iter().map(Ok),
                )))
            })
        },
    )
}

// ---------------------------------------------------------------------------
// Wrapper functions: C callback -> core trait objects
// ---------------------------------------------------------------------------

/// Wrap a C tool sanitize callback into a Rust closure for use by the core runtime.
pub fn wrap_tool_sanitize_fn(
    cb: NemoRelayToolSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> ToolSanitizeFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |name: String, args: Json| {
        let ud = ud.clone();
        Box::pin(async move {
            let c_name = CString::new(name).unwrap_or_default();
            let c_args = json_to_c_string(&args);
            let result_ptr = unsafe { cb(ud.ptr, c_name.as_ptr(), c_args) };
            unsafe { nemo_relay_string_free_internal(c_args) };
            let result = ptr_to_json(result_ptr);
            unsafe { nemo_relay_string_free_internal(result_ptr) };
            Ok(result)
        })
    })
}

/// Wrap a C tool conditional callback into a Rust closure for use by the core runtime.
pub fn wrap_tool_conditional_fn(
    cb: NemoRelayToolConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> ToolConditionalFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |name: String, args: Json| {
        let ud = ud.clone();
        Box::pin(async move {
            clear_last_error();
            let c_name = CString::new(name).unwrap_or_default();
            let c_args = json_to_c_string(&args);
            let result_ptr = unsafe { cb(ud.ptr, c_name.as_ptr(), c_args) };
            unsafe { nemo_relay_string_free_internal(c_args) };
            let result = if result_ptr.is_null() {
                match last_error_message() {
                    Some(message) => Err(FlowError::Internal(message)),
                    None => Ok(None),
                }
            } else {
                Ok(ptr_to_opt_string(result_ptr))
            };
            unsafe { nemo_relay_string_free_internal(result_ptr) };
            result
        })
    })
}

/// Wrap a C tool request intercept callback into a Rust closure for use by the core runtime.
pub fn wrap_tool_request_intercept_fn(
    cb: NemoRelayToolSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> ToolInterceptFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |name: String, args: Json| {
        let ud = ud.clone();
        Box::pin(async move {
            clear_last_error();
            let c_name = CString::new(name).unwrap_or_default();
            let c_args = json_to_c_string(&args);
            let result_ptr = unsafe { cb(ud.ptr, c_name.as_ptr(), c_args) };
            unsafe { nemo_relay_string_free_internal(c_args) };
            let result =
                json_result_from_ptr(result_ptr, "tool request intercept callback returned null");
            unsafe { nemo_relay_string_free_internal(result_ptr) };
            result
        })
    })
}

/// Wrap a C tool execution callback into an async Rust closure.
pub fn wrap_tool_exec_fn(
    cb: NemoRelayToolExecCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> Box<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |args: Json| {
        let ud = ud.clone();
        Box::pin(async move {
            let c_args = json_to_c_string(&args);
            let result_ptr = unsafe { cb(ud.ptr, c_args) };
            unsafe { nemo_relay_string_free_internal(c_args) };
            let result = json_result_from_ptr(result_ptr, "tool execution callback failed")?;
            unsafe { nemo_relay_string_free_internal(result_ptr) };
            Ok(result)
        })
    })
}

/// Wrap a C annotation-aware tool execution callback into an async Rust closure.
pub fn wrap_tool_exec_frame_fn(
    cb: NemoRelayToolExecFrameCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> Box<
    dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<ToolExecutionFrame>> + Send>> + Send + Sync,
> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |args: Json| {
        let ud = ud.clone();
        Box::pin(async move {
            let c_args = json_to_c_string(&args);
            let result_ptr = unsafe { cb(ud.ptr, c_args) };
            unsafe { nemo_relay_string_free_internal(c_args) };
            let frame_json =
                json_result_from_ptr(result_ptr, "tool execution frame callback failed")?;
            unsafe { nemo_relay_string_free_internal(result_ptr) };
            serde_json::from_value::<ToolExecutionFrame>(frame_json).map_err(|error| {
                FlowError::Internal(format!("invalid tool execution frame JSON: {error}"))
            })
        })
    })
}

/// Wrap a C tool execution intercept callback into a [`ToolExecutionFn`].
///
/// The wrapper packages the Rust `ToolExecutionNextFn` into a C-callable
/// `(next_fn, next_ctx)` pair and passes both to the C intercept callback. The
/// callback must return a serialized [`ToolExecutionInterceptOutcome`].
pub fn wrap_tool_exec_intercept_fn(
    cb: NemoRelayToolExecInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> ToolExecutionFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |_name: &str, args: Json, next: ToolExecutionNextFn| {
        let ud = ud.clone();
        Box::pin(async move {
            // Package the Rust next fn into an FFI-safe pair
            let next_box = Box::new(next);
            let next_ctx = Box::into_raw(next_box) as *mut libc::c_void;

            /// C trampoline that calls the boxed Rust next fn
            unsafe extern "C" fn tool_next_trampoline(
                args_json: *const c_char,
                next_ctx: *mut libc::c_void,
            ) -> *mut c_char {
                let next_arc = unsafe { &*(next_ctx as *const ToolExecutionNextFn) };
                let next = next_arc.clone();
                let args = if args_json.is_null() {
                    Json::Null
                } else {
                    let s = unsafe { CStr::from_ptr(args_json) }.to_string_lossy();
                    serde_json::from_str(&s).unwrap_or(Json::Null)
                };
                // Use block_in_place to allow nested block_on within the
                // multi-threaded tokio runtime (the outer block_on in
                // nemo_relay_tool_call_execute already occupies this worker).
                let handle = tokio::runtime::Handle::current();
                let result = tokio::task::block_in_place(|| handle.block_on(next(args)));
                match result {
                    Ok(json) => json_to_c_string(&json),
                    Err(e) => {
                        set_last_error(&e.to_string());
                        std::ptr::null_mut()
                    }
                }
            }

            let c_args = json_to_c_string(&args);
            let result_ptr = unsafe { cb(ud.ptr, c_args, tool_next_trampoline, next_ctx) };
            unsafe { drop(Box::from_raw(next_ctx as *mut ToolExecutionNextFn)) };
            unsafe { nemo_relay_string_free_internal(c_args) };
            let outcome_json =
                json_result_from_ptr(result_ptr, "tool execution intercept callback failed")?;
            unsafe { nemo_relay_string_free_internal(result_ptr) };
            serde_json::from_value::<ToolExecutionInterceptOutcome>(outcome_json).map_err(|error| {
                FlowError::Internal(format!(
                    "invalid tool execution intercept outcome JSON: {error}"
                ))
            })
        })
    })
}

/// Wrap a C annotation-aware tool execution intercept into a [`ToolExecutionFrameFn`].
pub fn wrap_tool_exec_frame_intercept_fn(
    cb: NemoRelayToolExecFrameInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> ToolExecutionFrameFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(
        move |_name: &str, args: Json, next: ToolExecutionFrameNextFn| {
            let ud = ud.clone();
            Box::pin(async move {
                let next_box = Box::new(next);
                let next_ctx = Box::into_raw(next_box) as *mut libc::c_void;

                unsafe extern "C" fn tool_frame_next_trampoline(
                    args_json: *const c_char,
                    next_ctx: *mut libc::c_void,
                ) -> *mut c_char {
                    let next_arc = unsafe { &*(next_ctx as *const ToolExecutionFrameNextFn) };
                    let next = next_arc.clone();
                    let args = if args_json.is_null() {
                        Json::Null
                    } else {
                        let value = unsafe { CStr::from_ptr(args_json) }.to_string_lossy();
                        serde_json::from_str(&value).unwrap_or(Json::Null)
                    };
                    let handle = tokio::runtime::Handle::current();
                    let result = tokio::task::block_in_place(|| handle.block_on(next(args)));
                    match result {
                        Ok(frame) => match serde_json::to_value(frame) {
                            Ok(value) => json_to_c_string(&value),
                            Err(error) => {
                                set_last_error(&error.to_string());
                                std::ptr::null_mut()
                            }
                        },
                        Err(error) => {
                            set_last_error(&error.to_string());
                            std::ptr::null_mut()
                        }
                    }
                }

                let c_args = json_to_c_string(&args);
                let result_ptr =
                    unsafe { cb(ud.ptr, c_args, tool_frame_next_trampoline, next_ctx) };
                unsafe { drop(Box::from_raw(next_ctx as *mut ToolExecutionFrameNextFn)) };
                unsafe { nemo_relay_string_free_internal(c_args) };
                let outcome_json = json_result_from_ptr(
                    result_ptr,
                    "tool execution frame intercept callback failed",
                )?;
                unsafe { nemo_relay_string_free_internal(result_ptr) };
                serde_json::from_value::<ToolExecutionFrameOutcome>(outcome_json).map_err(|error| {
                    FlowError::Internal(format!(
                        "invalid tool execution frame outcome JSON: {error}"
                    ))
                })
            })
        },
    )
}

/// Wrap a C LLM execution intercept callback into an `Arc<dyn Fn(LlmRequest, LlmExecutionNextFn) -> ...>`.
pub fn wrap_llm_exec_intercept_fn(
    cb: NemoRelayLlmExecInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(
        move |_name: &str, request: LlmRequest, next: LlmExecutionNextFn| {
            let ud = ud.clone();
            Box::pin(async move {
                let next_box = Box::new(next);
                let next_ctx = Box::into_raw(next_box) as *mut libc::c_void;

                /// C trampoline that calls the boxed Rust next fn.
                /// Takes a JSON string representing an LlmRequest, deserializes it,
                /// and calls the Rust LlmExecutionNextFn.
                unsafe extern "C" fn llm_next_trampoline(
                    native_json: *const c_char,
                    next_ctx: *mut libc::c_void,
                ) -> *mut c_char {
                    let next_arc = unsafe { &*(next_ctx as *const LlmExecutionNextFn) };
                    let next = next_arc.clone();
                    let request = if native_json.is_null() {
                        LlmRequest {
                            headers: serde_json::Map::new(),
                            content: Json::Null,
                        }
                    } else {
                        let s = unsafe { CStr::from_ptr(native_json) }.to_string_lossy();
                        serde_json::from_str::<LlmRequest>(&s).unwrap_or(LlmRequest {
                            headers: serde_json::Map::new(),
                            content: Json::Null,
                        })
                    };
                    let handle = tokio::runtime::Handle::current();
                    let result = tokio::task::block_in_place(|| handle.block_on(next(request)));
                    match result {
                        Ok(json) => json_to_c_string(&json),
                        Err(e) => {
                            set_last_error(&e.to_string());
                            std::ptr::null_mut()
                        }
                    }
                }

                let request_json = serde_json::to_value(&request).unwrap_or(Json::Null);
                let c_request = json_to_c_string(&request_json);
                let result_ptr = unsafe { cb(ud.ptr, c_request, llm_next_trampoline, next_ctx) };
                unsafe { drop(Box::from_raw(next_ctx as *mut LlmExecutionNextFn)) };
                unsafe { nemo_relay_string_free_internal(c_request) };
                let result =
                    json_result_from_ptr(result_ptr, "LLM execution intercept callback failed")?;
                unsafe { nemo_relay_string_free_internal(result_ptr) };
                Ok(result)
            })
        },
    )
}

/// Wrap a C LLM stream execution intercept callback.
/// Since the C callback returns a single string (not a real stream), this wraps
/// it as a single-item stream, same as `wrap_llm_stream_exec_fn`.
pub fn wrap_llm_stream_exec_intercept_fn(
    cb: NemoRelayLlmExecInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmStreamExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<LlmJsonStream>> + Send>>
        + Send
        + Sync,
> {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(
        move |_name: &str, request: LlmRequest, next: LlmStreamExecutionNextFn| {
            let ud = ud.clone();
            Box::pin(async move {
                let next_box = Box::new(next);
                let next_ctx = Box::into_raw(next_box) as *mut libc::c_void;

                unsafe extern "C" fn llm_stream_next_trampoline(
                    native_json: *const c_char,
                    next_ctx: *mut libc::c_void,
                ) -> *mut c_char {
                    let next_arc = unsafe { &*(next_ctx as *const LlmStreamExecutionNextFn) };
                    let next = next_arc.clone();
                    let request = if native_json.is_null() {
                        LlmRequest {
                            headers: serde_json::Map::new(),
                            content: Json::Null,
                        }
                    } else {
                        let s = unsafe { CStr::from_ptr(native_json) }.to_string_lossy();
                        serde_json::from_str::<LlmRequest>(&s).unwrap_or(LlmRequest {
                            headers: serde_json::Map::new(),
                            content: Json::Null,
                        })
                    };
                    let handle = tokio::runtime::Handle::current();
                    let result = tokio::task::block_in_place(|| {
                        handle.block_on(async move {
                            let mut stream = next(request).await?;
                            match stream.next().await {
                                Some(item) => item,
                                None => Ok(Json::Null),
                            }
                        })
                    });
                    match result {
                        Ok(json) => json_to_c_string(&json),
                        Err(e) => {
                            set_last_error(&e.to_string());
                            std::ptr::null_mut()
                        }
                    }
                }

                let request_json = serde_json::to_value(&request).unwrap_or(Json::Null);
                let c_request = json_to_c_string(&request_json);
                let result_ptr =
                    unsafe { cb(ud.ptr, c_request, llm_stream_next_trampoline, next_ctx) };
                unsafe { drop(Box::from_raw(next_ctx as *mut LlmStreamExecutionNextFn)) };
                unsafe { nemo_relay_string_free_internal(c_request) };
                let result = json_result_from_ptr(
                    result_ptr,
                    "LLM stream execution intercept callback failed",
                )?;
                unsafe { nemo_relay_string_free_internal(result_ptr) };
                let stream = tokio_stream::once(Ok(result));
                Ok(LlmJsonStream::new(stream))
            })
        },
    )
}

/// Wrap a C LLM request intercept callback (annotated-aware) into a Rust
/// `LlmRequestInterceptFn` closure. The callback receives the intercept name,
/// the opaque `FfiLLMRequest`, and the annotated JSON (or null). It writes one
/// owned canonical outcome JSON string.
pub fn wrap_llm_request_intercept_fn(
    cb: NemoRelayLlmRequestInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> LlmRequestInterceptFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(
        move |name: String, request: LlmRequest, annotated: Option<AnnotatedLLMRequest>| {
            let ud = ud.clone();
            Box::pin(async move {
                clear_last_error();
                let c_name = CString::new(name).unwrap_or_default();
                let ffi_req = Box::into_raw(Box::new(FfiLLMRequest(request)));

                // Serialize annotated to JSON C string if present, else null
                let c_annotated = match &annotated {
                    Some(a) => {
                        let s = serde_json::to_string(a).unwrap_or_else(|_| "null".to_string());
                        CString::new(s).unwrap_or_default()
                    }
                    None => CString::default(),
                };
                let annotated_ptr = if annotated.is_some() {
                    c_annotated.as_ptr()
                } else {
                    std::ptr::null()
                };

                let mut out_outcome: *mut c_char = std::ptr::null_mut();

                let status = unsafe {
                    cb(
                        ud.ptr,
                        c_name.as_ptr(),
                        ffi_req,
                        annotated_ptr,
                        &mut out_outcome,
                    )
                };

                // Free the input request
                unsafe { drop(Box::from_raw(ffi_req)) };

                if status != NemoRelayStatus::Ok {
                    unsafe { nemo_relay_string_free_internal(out_outcome) };
                    let message = last_error_message()
                        .unwrap_or_else(|| "request intercept callback failed".to_string());
                    return Err(FlowError::Internal(message));
                }

                if out_outcome.is_null() {
                    return Err(FlowError::Internal(
                        "request intercept returned null out_outcome_json".to_string(),
                    ));
                }
                let outcome = unsafe { CStr::from_ptr(out_outcome) }
                    .to_str()
                    .map_err(|error| FlowError::Internal(format!("invalid outcome UTF-8: {error}")))
                    .and_then(|json| {
                        serde_json::from_str::<LlmRequestInterceptOutcome>(json).map_err(|error| {
                            FlowError::Internal(format!(
                                "invalid LLM request intercept outcome JSON: {error}"
                            ))
                        })
                    });
                unsafe { nemo_relay_string_free_internal(out_outcome) };
                outcome
            })
        },
    )
}

/// Wrap a C LLM request sanitizer into a Rust closure.
pub fn wrap_llm_sanitize_request_fn(
    cb: NemoRelayLlmSanitizeRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> LlmSanitizeRequestFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(
        move |request: LlmRequest, context: LlmSanitizeRequestContext| {
            let ud = ud.clone();
            Box::pin(async move {
                clear_last_error();
                let (codec_kind, codec_id) = match ffi_codec_identity(context.codec()) {
                    Ok(identity) => identity,
                    Err(error) => {
                        set_last_error(&error.to_string());
                        return Err(error);
                    }
                };
                let codec = context
                    .resolve_codec()
                    .map(crate::types::FfiLlmSanitizeRequestCodec);
                let ffi_context = NemoRelayLlmSanitizeRequestContext {
                    codec_kind,
                    codec_id: codec_id
                        .as_ref()
                        .map_or(std::ptr::null(), |name| name.as_ptr()),
                    codec: codec.as_ref().map_or(std::ptr::null(), std::ptr::from_ref),
                };
                let ffi_req = Box::into_raw(Box::new(FfiLLMRequest(request)));
                let result_ptr = unsafe { cb(ud.ptr, ffi_req, ffi_context) };
                if result_ptr.is_null() {
                    unsafe { drop(Box::from_raw(ffi_req)) };
                    return Ok(None);
                }
                if result_ptr == ffi_req {
                    return Ok(Some(unsafe { Box::from_raw(ffi_req) }.0));
                }
                unsafe { drop(Box::from_raw(ffi_req)) };
                Ok(Some(unsafe { Box::from_raw(result_ptr) }.0))
            })
        },
    )
}

/// Wrap a C LLM response sanitizer into a Rust closure.
pub fn wrap_llm_sanitize_response_fn(
    cb: NemoRelayLlmSanitizeResponseCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> LlmSanitizeResponseFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |response: Json, context: LlmSanitizeResponseContext| {
        let ud = ud.clone();
        Box::pin(async move {
            clear_last_error();
            let (codec_kind, codec_id) = match ffi_codec_identity(context.codec()) {
                Ok(identity) => identity,
                Err(error) => {
                    set_last_error(&error.to_string());
                    return Err(error);
                }
            };
            let codec = context
                .resolve_codec()
                .map(crate::types::FfiLlmSanitizeResponseCodec);
            let ffi_context = NemoRelayLlmSanitizeResponseContext {
                codec_kind,
                codec_id: codec_id
                    .as_ref()
                    .map_or(std::ptr::null(), |name| name.as_ptr()),
                codec: codec.as_ref().map_or(std::ptr::null(), std::ptr::from_ref),
            };
            let response_json = json_to_c_string(&response);
            let result_ptr = unsafe { cb(ud.ptr, response_json, ffi_context) };
            if result_ptr.is_null() {
                unsafe { nemo_relay_string_free_internal(response_json) };
                return Ok(None);
            }
            let result = c_str_to_json(result_ptr);
            unsafe {
                nemo_relay_string_free_internal(response_json);
                if result_ptr != response_json {
                    nemo_relay_string_free_internal(result_ptr);
                }
            }
            Ok(result)
        })
    })
}

fn ffi_codec_identity(
    identity: &LlmCodecIdentity,
) -> Result<(NemoRelayLlmSanitizeCodecKind, Option<CString>)> {
    Ok(match identity {
        LlmCodecIdentity::None => (NemoRelayLlmSanitizeCodecKind::None, None),
        LlmCodecIdentity::BuiltIn(codec) => (
            NemoRelayLlmSanitizeCodecKind::BuiltIn,
            Some(CString::new(codec.id()).expect("built-in codec IDs never contain NUL")),
        ),
        LlmCodecIdentity::Runtime(id) => (
            NemoRelayLlmSanitizeCodecKind::Runtime,
            Some(CString::new(id.as_str()).map_err(|_| {
                FlowError::InvalidArgument("runtime codec ID contains an embedded NUL".to_string())
            })?),
        ),
        LlmCodecIdentity::Opaque => (NemoRelayLlmSanitizeCodecKind::Opaque, None),
    })
}

fn ffi_codec_identity_json(identity: &LlmCodecIdentity) -> Result<Json> {
    let (kind, id) = ffi_codec_identity(identity)?;
    let id = id
        .as_ref()
        .map(|id| {
            id.to_str()
                .map(str::to_owned)
                .map_err(|error| FlowError::Internal(error.to_string()))
        })
        .transpose()?;
    Ok(serde_json::json!({
        "codec_kind": kind as u32,
        "codec_id": id,
    }))
}

/// Wrap a C LLM conditional callback into a Rust closure.
pub fn wrap_llm_conditional_fn(
    cb: NemoRelayLlmConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> LlmConditionalFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |request: LlmRequest| {
        let ud = ud.clone();
        Box::pin(async move {
            clear_last_error();
            let ffi_req = FfiLLMRequest(request);
            let result_ptr = unsafe { cb(ud.ptr, &ffi_req) };
            let result = if result_ptr.is_null() {
                match last_error_message() {
                    Some(message) => Err(FlowError::Internal(message)),
                    None => Ok(None),
                }
            } else {
                Ok(ptr_to_opt_string(result_ptr))
            };
            unsafe { nemo_relay_string_free_internal(result_ptr) };
            result
        })
    })
}

/// Wrap a C LLM execution callback into an async Rust closure.
/// The C callback receives an `LlmRequest` serialized as a JSON string.
pub fn wrap_llm_exec_fn(
    cb: NemoRelayLlmExecCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> Box<dyn Fn(LlmRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |request: LlmRequest| {
        let ud = ud.clone();
        Box::pin(async move {
            let request_json = serde_json::to_value(&request).unwrap_or(Json::Null);
            let c_request = json_to_c_string(&request_json);
            let result_ptr = unsafe { cb(ud.ptr, c_request) };
            unsafe { nemo_relay_string_free_internal(c_request) };
            let result = json_result_from_ptr(result_ptr, "LLM execution callback failed")?;
            unsafe { nemo_relay_string_free_internal(result_ptr) };
            Ok(result)
        })
    })
}

/// Wrap a C LLM execution callback into an async Rust closure that returns a stream.
/// The C callback returns the full response as a single JSON string, which is emitted
/// as a single-item stream of Json values.
pub fn wrap_llm_stream_exec_fn(
    cb: NemoRelayLlmExecCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> Box<
    dyn Fn(LlmRequest) -> Pin<Box<dyn Future<Output = Result<LlmJsonStream>> + Send>> + Send + Sync,
> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |request: LlmRequest| {
        let ud = ud.clone();
        Box::pin(async move {
            let request_json = serde_json::to_value(&request).unwrap_or(Json::Null);
            let c_request = json_to_c_string(&request_json);
            let result_ptr = unsafe { cb(ud.ptr, c_request) };
            unsafe { nemo_relay_string_free_internal(c_request) };
            let result = json_result_from_ptr(result_ptr, "LLM stream execution callback failed")?;
            unsafe { nemo_relay_string_free_internal(result_ptr) };
            // The C callback returns the full response as a single JSON value for stream
            // We emit it as a single-item stream
            let stream = tokio_stream::once(Ok(result));
            Ok(LlmJsonStream::new(stream))
        })
    })
}

/// Wrap a C collector callback into a `Box<dyn FnMut(Json) -> Result<()> + Send>`
/// for use by the core runtime. Each intercepted chunk Json is serialized to a
/// JSON string and passed to the callback.
///
/// Because the C collector callback signature returns `void`, the wrapper
/// always returns `Ok(())`. C callers that need to signal errors from the
/// collector should use a side-channel (e.g., setting a flag) and check it
/// after the stream is consumed.
///
/// # Safety
/// The caller must ensure `cb` remains valid for the lifetime of the returned
/// closure. The C callback is invoked synchronously from the stream-consumption
/// task.
pub fn wrap_collector_fn(cb: NemoRelayCollectorCb) -> Box<dyn FnMut(Json) -> Result<()> + Send> {
    // NemoRelayCollectorCb is a plain `extern "C" fn` pointer (no user_data),
    // which is Copy + Send, so it can be moved into the closure directly.
    Box::new(move |chunk: Json| {
        let c_chunk = json_to_c_string(&chunk);
        unsafe { cb(c_chunk) };
        unsafe { nemo_relay_string_free_internal(c_chunk) };
        Ok(())
    })
}

/// Wrap a C finalizer callback into a `Box<dyn FnOnce() -> Json + Send>` for
/// use by the core runtime. The callback is invoked exactly once when the
/// stream is exhausted. The returned C string is parsed as JSON and then freed.
///
/// # Safety
/// The caller must ensure `cb` remains valid until the returned closure is
/// invoked. The C callback must return a valid, heap-allocated JSON C string
/// (or null, in which case `Json::Null` is returned).
pub fn wrap_finalizer_fn(cb: NemoRelayFinalizerCb) -> Box<dyn FnOnce() -> Json + Send> {
    Box::new(move || {
        let result_ptr = unsafe { cb() };
        let result = ptr_to_json(result_ptr);
        unsafe { nemo_relay_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C event subscriber callback into a Rust closure.
pub fn wrap_event_subscriber(
    cb: NemoRelayEventSubscriberCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> EventSubscriberFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |event: &Event| {
        let ffi_event = FfiEvent(event.clone());
        unsafe { cb(ud.ptr, &ffi_event) };
    })
}

/// Wrap a C event sanitizer callback into a Rust closure.
pub fn wrap_event_sanitize_fn(
    cb: NemoRelayEventSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> EventSanitizeFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |event: Arc<Event>, fields: EventSanitizeFields| {
        let ud = ud.clone();
        Box::pin(async move {
            let ffi_event = FfiEvent((*event).clone());
            let fields_json =
                json_to_c_string(&serde_json::to_value(&fields).unwrap_or(Json::Null));
            let result_ptr = unsafe { cb(ud.ptr, &ffi_event, fields_json) };
            unsafe { nemo_relay_string_free_internal(fields_json) };
            let result = serde_json::from_value(ptr_to_json(result_ptr)).unwrap_or_default();
            unsafe { nemo_relay_string_free_internal(result_ptr) };
            Ok(result)
        })
    })
}

// ---------------------------------------------------------------------------
// Codec wrapper: C callbacks -> Arc<dyn LlmCodec>
// ---------------------------------------------------------------------------

/// FFI-backed Codec that delegates `decode`/`encode` to C callback pointers.
struct FfiCodec {
    decode_cb: NemoRelayCodecDecodeCb,
    encode_cb: NemoRelayCodecEncodeCb,
    user_data: Arc<UserData>,
}

unsafe impl Send for FfiCodec {}
unsafe impl Sync for FfiCodec {}

impl LlmCodec for FfiCodec {
    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLLMRequest> {
        clear_last_error();
        let ffi_req = Box::into_raw(Box::new(FfiLLMRequest(request.clone())));
        let result_ptr = unsafe { (self.decode_cb)(self.user_data.ptr, ffi_req) };
        // Free the input request
        unsafe { drop(Box::from_raw(ffi_req)) };
        if result_ptr.is_null() {
            let message = last_error_message()
                .unwrap_or_else(|| "codec decode callback returned null".to_string());
            return Err(FlowError::Internal(message));
        }
        let result_str = unsafe { CStr::from_ptr(result_ptr) }.to_string_lossy();
        let annotated: AnnotatedLLMRequest = serde_json::from_str(&result_str).map_err(|e| {
            unsafe { nemo_relay_string_free_internal(result_ptr) };
            FlowError::Internal(format!("codec decode: invalid JSON: {e}"))
        })?;
        unsafe { nemo_relay_string_free_internal(result_ptr) };
        Ok(annotated)
    }

    fn encode(&self, annotated: &AnnotatedLLMRequest, original: &LlmRequest) -> Result<LlmRequest> {
        clear_last_error();
        let annotated_str = serde_json::to_string(annotated)
            .map_err(|e| FlowError::Internal(format!("codec encode: serialize failed: {e}")))?;
        let c_annotated = CString::new(annotated_str)
            .map_err(|e| FlowError::Internal(format!("codec encode: CString failed: {e}")))?;
        let ffi_req = Box::into_raw(Box::new(FfiLLMRequest(original.clone())));
        let result_ptr =
            unsafe { (self.encode_cb)(self.user_data.ptr, c_annotated.as_ptr(), ffi_req) };
        // Free the input request
        unsafe { drop(Box::from_raw(ffi_req)) };
        if result_ptr.is_null() {
            let message = last_error_message()
                .unwrap_or_else(|| "codec encode callback returned null".to_string());
            return Err(FlowError::Internal(message));
        }
        let result_str = unsafe { CStr::from_ptr(result_ptr) }.to_string_lossy();
        let content: serde_json::Value = serde_json::from_str(&result_str).map_err(|e| {
            unsafe { nemo_relay_string_free_internal(result_ptr) };
            FlowError::Internal(format!("codec encode: invalid result JSON: {e}"))
        })?;
        unsafe { nemo_relay_string_free_internal(result_ptr) };
        Ok(LlmRequest {
            headers: original.headers.clone(),
            content,
        })
    }
}

/// Wrap a pair of C codec callbacks into an `Arc<dyn LlmCodec>`.
pub fn wrap_codec_fn(
    decode_cb: NemoRelayCodecDecodeCb,
    encode_cb: NemoRelayCodecEncodeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> Arc<dyn LlmCodec> {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(FfiCodec {
        decode_cb,
        encode_cb,
        user_data: ud,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ptr_to_json(ptr: *mut c_char) -> Json {
    if ptr.is_null() {
        return Json::Null;
    }
    let s = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
    serde_json::from_str(&s).unwrap_or(Json::Null)
}

fn json_result_from_ptr(ptr: *mut c_char, fallback: &str) -> Result<Json> {
    if ptr.is_null() {
        let message = last_error_message().unwrap_or_else(|| fallback.to_string());
        return Err(FlowError::Internal(message));
    }
    Ok(ptr_to_json(ptr))
}

fn ptr_to_opt_string(ptr: *mut c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// Internal helper to free C strings we allocated.
unsafe fn nemo_relay_string_free_internal(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}

#[cfg(test)]
#[path = "../tests/support/mod.rs"]
mod test_support;

#[cfg(test)]
#[path = "../tests/unit/callable_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/unit/callable_private_tests.rs"]
mod private_tests;
