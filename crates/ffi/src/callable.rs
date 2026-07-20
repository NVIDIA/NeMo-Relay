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
use std::sync::Arc;

use libc::c_char;
use nemo_relay::api::runtime::{
    EventSanitizeFn, EventSubscriberFn, LlmConditionalFn, LlmExecutionNextFn, LlmJsonStream,
    LlmRequestInterceptFn, LlmSanitizeRequestFn, LlmSanitizeResponseFn, LlmStreamExecutionNextFn,
    ToolConditionalFn, ToolExecutionFn, ToolExecutionNextFn, ToolInterceptFn, ToolSanitizeFn,
};
use serde_json::Value as Json;
use tokio_stream::StreamExt;

use nemo_relay::api::event::{Event, EventSanitizeFields};
use nemo_relay::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
use nemo_relay::api::tool::ToolExecutionInterceptOutcome;
use nemo_relay::codec::request::AnnotatedLlmRequest as AnnotatedLLMRequest;
use nemo_relay::codec::traits::LlmCodec;
use nemo_relay::error::{FlowError, Result};

use crate::convert::json_to_c_string;
use crate::error::{NemoRelayStatus, clear_last_error, last_error_message, set_last_error};
use crate::types::{FfiEvent, FfiLLMRequest, FfiPluginContext};

// ---------------------------------------------------------------------------
// Callback typedefs (mirrored in the C header)
// ---------------------------------------------------------------------------

/// Optional destructor for user data passed to callbacks.
/// Called when the runtime no longer needs the associated callback.
pub type NemoRelayFreeFn = Option<unsafe extern "C" fn(user_data: *mut libc::c_void)>;

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

/// Runtime-provided "next" callback for tool execution middleware chain.
/// Call this from an intercept to invoke the next layer (or original function).
/// `next_ctx` is an opaque pointer managed by the runtime.
pub type NemoRelayToolExecNextFn =
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

/// Generic JSON-to-JSON callback, used for LLM response sanitization and intercepts.
/// The returned string must be allocated with `malloc` or equivalent.
pub type NemoRelayJsonCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, json: *const c_char) -> *mut c_char;

/// Callback for LLM request sanitization. Receives an `FfiLLMRequest` and returns
/// a new (possibly modified) `FfiLLMRequest`. Return null to use defaults.
pub type NemoRelayLlmRequestCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut FfiLLMRequest;

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
    Arc::new(move |name: &str, args: Json| {
        let c_name = CString::new(name).unwrap_or_default();
        let c_args = json_to_c_string(&args);
        let result_ptr = unsafe { cb(ud.ptr, c_name.as_ptr(), c_args) };
        unsafe { nemo_relay_string_free_internal(c_args) };
        let result = ptr_to_json(result_ptr);
        unsafe { nemo_relay_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C tool conditional callback into a Rust closure for use by the core runtime.
pub fn wrap_tool_conditional_fn(
    cb: NemoRelayToolConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> ToolConditionalFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |name: &str, args: &Json| {
        clear_last_error();
        let c_name = CString::new(name).unwrap_or_default();
        let c_args = json_to_c_string(args);
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
}

/// Wrap a C tool request intercept callback into a Rust closure for use by the core runtime.
pub fn wrap_tool_request_intercept_fn(
    cb: NemoRelayToolSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> ToolInterceptFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |name: &str, args: Json| {
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

/// Wrap a generic C JSON callback into a Rust closure.
pub fn wrap_json_fn(
    cb: NemoRelayJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> Box<dyn Fn(Json) -> Json + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |value: Json| {
        let c_json = json_to_c_string(&value);
        let result_ptr = unsafe { cb(ud.ptr, c_json) };
        unsafe { nemo_relay_string_free_internal(c_json) };
        let result = ptr_to_json(result_ptr);
        unsafe { nemo_relay_string_free_internal(result_ptr) };
        result
    })
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
        move |name: &str, request: LlmRequest, annotated: Option<AnnotatedLLMRequest>| {
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
        },
    )
}

/// Wrap a C JSON callback into a `Fn(Json) -> Json` closure for LLM response
/// sanitization. The callback receives the response as a JSON string and
/// returns the (possibly modified) JSON string.
pub fn wrap_llm_response_fn(
    cb: NemoRelayJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> LlmSanitizeResponseFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |response: Json| {
        let c_json = json_to_c_string(&response);
        let result_ptr = unsafe { cb(ud.ptr, c_json) };
        unsafe { nemo_relay_string_free_internal(c_json) };
        let result_json = ptr_to_json(result_ptr);
        unsafe { nemo_relay_string_free_internal(result_ptr) };
        result_json
    })
}

/// Wrap a C LLM request sanitize callback into a Rust closure.
pub fn wrap_llm_sanitize_request_fn(
    cb: NemoRelayLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> LlmSanitizeRequestFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |request: LlmRequest| {
        let ffi_req = Box::into_raw(Box::new(FfiLLMRequest(request)));
        let result_ptr = unsafe { cb(ud.ptr, ffi_req) };
        // Free the input request
        unsafe { drop(Box::from_raw(ffi_req)) };
        if result_ptr.is_null() {
            // If callback returns null, return a default
            LlmRequest {
                headers: serde_json::Map::new(),
                content: Json::Null,
            }
        } else {
            let result = unsafe { Box::from_raw(result_ptr) };
            result.0
        }
    })
}

/// Wrap a C LLM conditional callback into a Rust closure.
pub fn wrap_llm_conditional_fn(
    cb: NemoRelayLlmConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> LlmConditionalFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |request: &LlmRequest| {
        clear_last_error();
        let ffi_req = FfiLLMRequest(request.clone());
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
    Arc::new(move |event: &Event, fields: EventSanitizeFields| {
        let ffi_event = FfiEvent(event.clone());
        let fields_json = json_to_c_string(&serde_json::to_value(&fields).unwrap_or(Json::Null));
        let result_ptr = unsafe { cb(ud.ptr, &ffi_event, fields_json) };
        unsafe { nemo_relay_string_free_internal(fields_json) };
        let result = serde_json::from_value(ptr_to_json(result_ptr)).unwrap_or_default();
        unsafe { nemo_relay_string_free_internal(result_ptr) };
        result
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
#[path = "../tests/unit/callable_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/unit/callable_private_tests.rs"]
mod private_tests;
