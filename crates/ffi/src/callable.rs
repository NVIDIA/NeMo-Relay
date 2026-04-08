// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::type_complexity)]
//! C function pointer typedefs and wrapper functions for FFI callbacks.
//!
//! This module defines the callback signatures used by the C API for tool and
//! LLM guardrails, intercepts, execution functions, and event subscribers. Each
//! `pub type` alias corresponds to a C function pointer that appears in the
//! generated `nat_nexus.h` header.
//!
//! The `wrap_*` functions convert C callbacks (with opaque `user_data` pointers)
//! into Rust closures (`Box<dyn Fn(...)>`) that the core runtime can invoke.
//! Each wrapper captures the user data and its optional free function in an
//! `Arc<UserData>` so the closure is `Send + Sync` and the free function is
//! called exactly once when all references are dropped.

use std::ffi::{CStr, CString};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use libc::c_char;
use serde_json::Value as Json;
use tokio_stream::{Stream, StreamExt};

use nvidia_nat_nexus_core::codec::{AnnotatedLLMRequest, LlmCodec};
use nvidia_nat_nexus_core::types::LLMRequest;
use nvidia_nat_nexus_core::{
    LlmConditionalFn, LlmExecutionNextFn, LlmRequestInterceptFn, LlmStreamExecutionNextFn,
    NexusError, Result, ToolConditionalFn, ToolExecutionNextFn, ToolInterceptFn,
};

use crate::convert::json_to_c_string;
use crate::error::{clear_last_error, last_error_message, set_last_error, NatNexusStatus};
use crate::types::{FfiEvent, FfiLLMRequest};

// ---------------------------------------------------------------------------
// Callback typedefs (mirrored in the C header)
// ---------------------------------------------------------------------------

/// Optional destructor for user data passed to callbacks.
/// Called when the runtime no longer needs the associated callback.
pub type NatNexusFreeFn = Option<unsafe extern "C" fn(user_data: *mut libc::c_void)>;

/// Callback for tool request/response sanitization guardrails and intercepts.
/// Receives tool name and arguments as JSON, returns sanitized arguments as JSON.
/// The returned string must be allocated with `malloc` or equivalent.
pub type NatNexusToolSanitizeCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    name: *const c_char,
    args_json: *const c_char,
) -> *mut c_char;

/// Callback for tool conditional execution guardrails.
/// Receives tool name and arguments as JSON.
/// Returns NULL to allow execution, or an error message string to reject.
pub type NatNexusToolConditionalCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    name: *const c_char,
    args_json: *const c_char,
) -> *mut c_char;

/// Callback for tool execution (default callable). Receives arguments as JSON,
/// returns result as JSON. The returned string must be allocated with `malloc`
/// or equivalent.
pub type NatNexusToolExecCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, args_json: *const c_char) -> *mut c_char;

/// Runtime-provided "next" callback for tool execution middleware chain.
/// Call this from an intercept to invoke the next layer (or original function).
/// `next_ctx` is an opaque pointer managed by the runtime.
pub type NatNexusToolExecNextFn =
    unsafe extern "C" fn(args_json: *const c_char, next_ctx: *mut libc::c_void) -> *mut c_char;

/// Callback for tool execution intercepts. Receives arguments as JSON plus
/// a `next` callback and its context. Call `next_fn(args, next_ctx)` to invoke
/// the next layer in the middleware chain, or return directly to short-circuit.
pub type NatNexusToolExecInterceptCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    args_json: *const c_char,
    next_fn: NatNexusToolExecNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char;

/// Generic JSON-to-JSON callback, used for LLM response sanitization and intercepts.
/// The returned string must be allocated with `malloc` or equivalent.
pub type NatNexusJsonCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, json: *const c_char) -> *mut c_char;

/// Callback for LLM request sanitization. Receives an `FfiLLMRequest` and returns
/// a new (possibly modified) `FfiLLMRequest`. Return null to use defaults.
pub type NatNexusLlmRequestCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut FfiLLMRequest;

/// Callback for LLM conditional execution guardrails.
/// Returns NULL to allow execution, or an error message string to reject.
pub type NatNexusLlmConditionalCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut c_char;

/// Callback for LLM execution (default callable). Receives a native JSON C string,
/// returns the response as a JSON C string.
pub type NatNexusLlmExecCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, native_json: *const c_char) -> *mut c_char;

/// Runtime-provided "next" callback for LLM execution middleware chain.
/// Takes a native JSON C string, returns a response JSON C string.
pub type NatNexusLlmExecNextFn =
    unsafe extern "C" fn(native_json: *const c_char, next_ctx: *mut libc::c_void) -> *mut c_char;

/// Callback for LLM execution intercepts with middleware chain support.
/// Receives native JSON C string plus a `next` callback and its context.
pub type NatNexusLlmExecInterceptCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    native_json: *const c_char,
    next_fn: NatNexusLlmExecNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char;

/// Callback for event subscribers. Invoked on each lifecycle event emitted by
/// the runtime. The `FfiEvent` pointer is only valid for the duration of the call.
pub type NatNexusEventSubscriberCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, event: *const FfiEvent);

/// Callback for Codec decode: translates an opaque `FfiLLMRequest` into
/// an `AnnotatedLLMRequest` JSON string. Returns a heap-allocated C string
/// on success, or null on error (after setting the last error message).
pub type NatNexusCodecDecodeCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut c_char;

/// Nullable version of [`NatNexusCodecDecodeCb`] for use as an optional
/// parameter in FFI execute functions. Pass null to indicate no codec.
pub type NatNexusCodecDecodeFn = Option<
    unsafe extern "C" fn(
        user_data: *mut libc::c_void,
        request: *const FfiLLMRequest,
    ) -> *mut c_char,
>;

/// Callback for Codec encode: merges structured changes back into opaque
/// request content. Receives the annotated request as a JSON C string and
/// the original `FfiLLMRequest`. Returns a heap-allocated JSON C string
/// representing the new `LLMRequest` content on success, or null on error.
pub type NatNexusCodecEncodeCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    annotated_json: *const c_char,
    original_request: *const FfiLLMRequest,
) -> *mut c_char;

/// Nullable version of [`NatNexusCodecEncodeCb`] for use as an optional
/// parameter in FFI execute functions. Pass null to indicate no codec.
pub type NatNexusCodecEncodeFn = Option<
    unsafe extern "C" fn(
        user_data: *mut libc::c_void,
        annotated_json: *const c_char,
        original_request: *const FfiLLMRequest,
    ) -> *mut c_char,
>;

/// C callback type for LLM request intercepts with unified annotated-aware
/// signature. Receives the intercept name, the opaque `FfiLLMRequest`, and
/// optionally the annotated request as a JSON C string (null if no Codec
/// resolved). Writes transformed outputs to `out_request` and
/// `out_annotated_json`. Returns `NatNexusStatus`.
pub type NatNexusLlmRequestInterceptCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    name: *const c_char,
    request: *const FfiLLMRequest,
    annotated_json: *const c_char,
    out_request: *mut *mut FfiLLMRequest,
    out_annotated_json: *mut *mut c_char,
) -> NatNexusStatus;

/// Callback for collecting intercepted stream chunks. Invoked with each chunk
/// (after stream execution intercepts have been applied) as a null-terminated
/// C string. The string is only valid for the duration of the call.
pub type NatNexusCollectorCb = unsafe extern "C" fn(chunk: *const c_char);

/// Callback for finalizing a collected stream. Invoked once when the stream is
/// exhausted. Must return a JSON C string representing the aggregated response.
/// The returned string must be allocated with `malloc` or equivalent; the
/// runtime will free it.
pub type NatNexusFinalizerCb = unsafe extern "C" fn() -> *mut c_char;

// ---------------------------------------------------------------------------
// Shared user_data wrapper (ensures cleanup)
// ---------------------------------------------------------------------------

/// RAII wrapper around a C user-data pointer and its associated free function.
/// Ensures the free function is called exactly once when dropped.
struct UserData {
    ptr: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
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
    free_fn: NatNexusFreeFn,
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
    cb: NatNexusToolSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> Box<dyn Fn(&str, Json) -> Json + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |name: &str, args: Json| {
        let c_name = CString::new(name).unwrap_or_default();
        let c_args = json_to_c_string(&args);
        let result_ptr = unsafe { cb(ud.ptr, c_name.as_ptr(), c_args) };
        unsafe { nat_nexus_string_free_internal(c_args) };
        let result = ptr_to_json(result_ptr);
        unsafe { nat_nexus_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C tool conditional callback into a Rust closure for use by the core runtime.
pub fn wrap_tool_conditional_fn(
    cb: NatNexusToolConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> ToolConditionalFn {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |name: &str, args: &Json| {
        clear_last_error();
        let c_name = CString::new(name).unwrap_or_default();
        let c_args = json_to_c_string(args);
        let result_ptr = unsafe { cb(ud.ptr, c_name.as_ptr(), c_args) };
        unsafe { nat_nexus_string_free_internal(c_args) };
        let result = if result_ptr.is_null() {
            match last_error_message() {
                Some(message) => Err(NexusError::Internal(message)),
                None => Ok(None),
            }
        } else {
            Ok(ptr_to_opt_string(result_ptr))
        };
        unsafe { nat_nexus_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C tool request intercept callback into a Rust closure for use by the core runtime.
pub fn wrap_tool_request_intercept_fn(
    cb: NatNexusToolSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> ToolInterceptFn {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |name: &str, args: Json| {
        clear_last_error();
        let c_name = CString::new(name).unwrap_or_default();
        let c_args = json_to_c_string(&args);
        let result_ptr = unsafe { cb(ud.ptr, c_name.as_ptr(), c_args) };
        unsafe { nat_nexus_string_free_internal(c_args) };
        let result =
            json_result_from_ptr(result_ptr, "tool request intercept callback returned null");
        unsafe { nat_nexus_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C tool execution callback into an async Rust closure.
pub fn wrap_tool_exec_fn(
    cb: NatNexusToolExecCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> Box<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |args: Json| {
        let ud = ud.clone();
        Box::pin(async move {
            let c_args = json_to_c_string(&args);
            let result_ptr = unsafe { cb(ud.ptr, c_args) };
            unsafe { nat_nexus_string_free_internal(c_args) };
            let result = json_result_from_ptr(result_ptr, "tool execution callback failed")?;
            unsafe { nat_nexus_string_free_internal(result_ptr) };
            Ok(result)
        })
    })
}

/// Wrap a C tool execution intercept callback into an `Arc<dyn Fn(Json, ToolExecutionNextFn) -> ...>`.
///
/// The wrapper packages the Rust `ToolExecutionNextFn` into a C-callable
/// `(next_fn, next_ctx)` pair and passes both to the C intercept callback.
pub fn wrap_tool_exec_intercept_fn(
    cb: NatNexusToolExecInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> Arc<
    dyn Fn(&str, Json, ToolExecutionNextFn) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
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
                // nat_nexus_tool_call_execute already occupies this worker).
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
            unsafe { nat_nexus_string_free_internal(c_args) };
            let result =
                json_result_from_ptr(result_ptr, "tool execution intercept callback failed")?;
            unsafe { nat_nexus_string_free_internal(result_ptr) };
            Ok(result)
        })
    })
}

/// Wrap a C LLM execution intercept callback into an `Arc<dyn Fn(LLMRequest, LlmExecutionNextFn) -> ...>`.
pub fn wrap_llm_exec_intercept_fn(
    cb: NatNexusLlmExecInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> Arc<
    dyn Fn(
            &str,
            LLMRequest,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(
        move |_name: &str, request: LLMRequest, next: LlmExecutionNextFn| {
            let ud = ud.clone();
            Box::pin(async move {
                let next_box = Box::new(next);
                let next_ctx = Box::into_raw(next_box) as *mut libc::c_void;

                /// C trampoline that calls the boxed Rust next fn.
                /// Takes a JSON string representing an LLMRequest, deserializes it,
                /// and calls the Rust LlmExecutionNextFn.
                unsafe extern "C" fn llm_next_trampoline(
                    native_json: *const c_char,
                    next_ctx: *mut libc::c_void,
                ) -> *mut c_char {
                    let next_arc = unsafe { &*(next_ctx as *const LlmExecutionNextFn) };
                    let next = next_arc.clone();
                    let request = if native_json.is_null() {
                        LLMRequest {
                            headers: serde_json::Map::new(),
                            content: Json::Null,
                        }
                    } else {
                        let s = unsafe { CStr::from_ptr(native_json) }.to_string_lossy();
                        serde_json::from_str::<LLMRequest>(&s).unwrap_or(LLMRequest {
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
                unsafe { nat_nexus_string_free_internal(c_request) };
                let result =
                    json_result_from_ptr(result_ptr, "LLM execution intercept callback failed")?;
                unsafe { nat_nexus_string_free_internal(result_ptr) };
                Ok(result)
            })
        },
    )
}

/// Wrap a C LLM stream execution intercept callback.
/// Since the C callback returns a single string (not a real stream), this wraps
/// it as a single-item stream, same as `wrap_llm_stream_exec_fn`.
pub fn wrap_llm_stream_exec_intercept_fn(
    cb: NatNexusLlmExecInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> Arc<
    dyn Fn(
            &str,
            LLMRequest,
            LlmStreamExecutionNextFn,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<Pin<Box<dyn Stream<Item = Result<Json>> + Send>>>>
                    + Send,
            >,
        > + Send
        + Sync,
> {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(
        move |_name: &str, request: LLMRequest, next: LlmStreamExecutionNextFn| {
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
                        LLMRequest {
                            headers: serde_json::Map::new(),
                            content: Json::Null,
                        }
                    } else {
                        let s = unsafe { CStr::from_ptr(native_json) }.to_string_lossy();
                        serde_json::from_str::<LLMRequest>(&s).unwrap_or(LLMRequest {
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
                unsafe { nat_nexus_string_free_internal(c_request) };
                let result = json_result_from_ptr(
                    result_ptr,
                    "LLM stream execution intercept callback failed",
                )?;
                unsafe { nat_nexus_string_free_internal(result_ptr) };
                let stream = tokio_stream::once(Ok(result));
                Ok(Box::pin(stream) as Pin<Box<dyn Stream<Item = Result<Json>> + Send>>)
            })
        },
    )
}

/// Wrap a generic C JSON callback into a Rust closure.
pub fn wrap_json_fn(
    cb: NatNexusJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> Box<dyn Fn(Json) -> Json + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |value: Json| {
        let c_json = json_to_c_string(&value);
        let result_ptr = unsafe { cb(ud.ptr, c_json) };
        unsafe { nat_nexus_string_free_internal(c_json) };
        let result = ptr_to_json(result_ptr);
        unsafe { nat_nexus_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C LLM request intercept callback (annotated-aware) into a Rust
/// `LlmRequestInterceptFn` closure. The callback receives the intercept name,
/// the opaque `FfiLLMRequest`, and the annotated JSON (or null). It writes
/// the transformed request and annotated JSON to output pointers.
pub fn wrap_llm_request_intercept_fn(
    cb: NatNexusLlmRequestInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> LlmRequestInterceptFn {
    let ud = make_user_data(user_data, free_fn);
    Box::new(
        move |name: &str, request: LLMRequest, annotated: Option<AnnotatedLLMRequest>| {
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

            // Initialize output pointers
            let mut out_request: *mut FfiLLMRequest = std::ptr::null_mut();
            let mut out_annotated: *mut c_char = std::ptr::null_mut();

            let status = unsafe {
                cb(
                    ud.ptr,
                    c_name.as_ptr(),
                    ffi_req,
                    annotated_ptr,
                    &mut out_request,
                    &mut out_annotated,
                )
            };

            // Free the input request
            unsafe { drop(Box::from_raw(ffi_req)) };

            if status != NatNexusStatus::Ok {
                let message = last_error_message()
                    .unwrap_or_else(|| "request intercept callback failed".to_string());
                return Err(NexusError::Internal(message));
            }

            // Read output request
            let new_request = if out_request.is_null() {
                return Err(NexusError::Internal(
                    "request intercept returned null out_request".to_string(),
                ));
            } else {
                let boxed = unsafe { Box::from_raw(out_request) };
                boxed.0
            };

            // Read output annotated
            let new_annotated = if out_annotated.is_null() {
                None
            } else {
                let s = unsafe { CStr::from_ptr(out_annotated) }.to_string_lossy();
                let parsed: Option<AnnotatedLLMRequest> = serde_json::from_str(&s).ok();
                unsafe { nat_nexus_string_free_internal(out_annotated) };
                parsed
            };

            Ok((new_request, new_annotated))
        },
    )
}

/// Wrap a C JSON callback into a `Fn(Json) -> Json` closure for LLM response
/// sanitization. The callback receives the response as a JSON string and
/// returns the (possibly modified) JSON string.
pub fn wrap_llm_response_fn(
    cb: NatNexusJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> Box<dyn Fn(Json) -> Json + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |response: Json| {
        let c_json = json_to_c_string(&response);
        let result_ptr = unsafe { cb(ud.ptr, c_json) };
        unsafe { nat_nexus_string_free_internal(c_json) };
        let result_json = ptr_to_json(result_ptr);
        unsafe { nat_nexus_string_free_internal(result_ptr) };
        result_json
    })
}

/// Wrap a C LLM request sanitize callback into a Rust closure.
pub fn wrap_llm_sanitize_request_fn(
    cb: NatNexusLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> Box<dyn Fn(LLMRequest) -> LLMRequest + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |request: LLMRequest| {
        let ffi_req = Box::into_raw(Box::new(FfiLLMRequest(request)));
        let result_ptr = unsafe { cb(ud.ptr, ffi_req) };
        // Free the input request
        unsafe { drop(Box::from_raw(ffi_req)) };
        if result_ptr.is_null() {
            // If callback returns null, return a default
            LLMRequest {
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
    cb: NatNexusLlmConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> LlmConditionalFn {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |request: &LLMRequest| {
        clear_last_error();
        let ffi_req = FfiLLMRequest(request.clone());
        let result_ptr = unsafe { cb(ud.ptr, &ffi_req) };
        let result = if result_ptr.is_null() {
            match last_error_message() {
                Some(message) => Err(NexusError::Internal(message)),
                None => Ok(None),
            }
        } else {
            Ok(ptr_to_opt_string(result_ptr))
        };
        unsafe { nat_nexus_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C LLM execution callback into an async Rust closure.
/// The C callback receives an `LLMRequest` serialized as a JSON string.
pub fn wrap_llm_exec_fn(
    cb: NatNexusLlmExecCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> Box<dyn Fn(LLMRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |request: LLMRequest| {
        let ud = ud.clone();
        Box::pin(async move {
            let request_json = serde_json::to_value(&request).unwrap_or(Json::Null);
            let c_request = json_to_c_string(&request_json);
            let result_ptr = unsafe { cb(ud.ptr, c_request) };
            unsafe { nat_nexus_string_free_internal(c_request) };
            let result = json_result_from_ptr(result_ptr, "LLM execution callback failed")?;
            unsafe { nat_nexus_string_free_internal(result_ptr) };
            Ok(result)
        })
    })
}

/// Wrap a C LLM execution callback into an async Rust closure that returns a stream.
/// The C callback returns the full response as a single JSON string, which is emitted
/// as a single-item stream of Json values.
pub fn wrap_llm_stream_exec_fn(
    cb: NatNexusLlmExecCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> Box<
    dyn Fn(
            LLMRequest,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<Pin<Box<dyn Stream<Item = Result<Json>> + Send>>>>
                    + Send,
            >,
        > + Send
        + Sync,
> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |request: LLMRequest| {
        let ud = ud.clone();
        Box::pin(async move {
            let request_json = serde_json::to_value(&request).unwrap_or(Json::Null);
            let c_request = json_to_c_string(&request_json);
            let result_ptr = unsafe { cb(ud.ptr, c_request) };
            unsafe { nat_nexus_string_free_internal(c_request) };
            let result = json_result_from_ptr(result_ptr, "LLM stream execution callback failed")?;
            unsafe { nat_nexus_string_free_internal(result_ptr) };
            // The C callback returns the full response as a single JSON value for stream
            // We emit it as a single-item stream
            let stream = tokio_stream::once(Ok(result));
            Ok(Box::pin(stream) as Pin<Box<dyn Stream<Item = Result<Json>> + Send>>)
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
pub fn wrap_collector_fn(cb: NatNexusCollectorCb) -> Box<dyn FnMut(Json) -> Result<()> + Send> {
    // NatNexusCollectorCb is a plain `extern "C" fn` pointer (no user_data),
    // which is Copy + Send, so it can be moved into the closure directly.
    Box::new(move |chunk: Json| {
        let c_chunk = json_to_c_string(&chunk);
        unsafe { cb(c_chunk) };
        unsafe { nat_nexus_string_free_internal(c_chunk) };
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
pub fn wrap_finalizer_fn(cb: NatNexusFinalizerCb) -> Box<dyn FnOnce() -> Json + Send> {
    Box::new(move || {
        let result_ptr = unsafe { cb() };
        let result = ptr_to_json(result_ptr);
        unsafe { nat_nexus_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C event subscriber callback into a Rust closure.
pub fn wrap_event_subscriber(
    cb: NatNexusEventSubscriberCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> nvidia_nat_nexus_core::EventSubscriberFn {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |event: &nvidia_nat_nexus_core::Event| {
        let ffi_event = FfiEvent(event.clone());
        unsafe { cb(ud.ptr, &ffi_event) };
    })
}

// ---------------------------------------------------------------------------
// Codec wrapper: C callbacks -> Arc<dyn LlmCodec>
// ---------------------------------------------------------------------------

/// FFI-backed Codec that delegates `decode`/`encode` to C callback pointers.
struct FfiCodec {
    decode_cb: NatNexusCodecDecodeCb,
    encode_cb: NatNexusCodecEncodeCb,
    user_data: Arc<UserData>,
}

unsafe impl Send for FfiCodec {}
unsafe impl Sync for FfiCodec {}

impl LlmCodec for FfiCodec {
    fn decode(&self, request: &LLMRequest) -> nvidia_nat_nexus_core::Result<AnnotatedLLMRequest> {
        clear_last_error();
        let ffi_req = Box::into_raw(Box::new(FfiLLMRequest(request.clone())));
        let result_ptr = unsafe { (self.decode_cb)(self.user_data.ptr, ffi_req) };
        // Free the input request
        unsafe { drop(Box::from_raw(ffi_req)) };
        if result_ptr.is_null() {
            let message = last_error_message()
                .unwrap_or_else(|| "codec decode callback returned null".to_string());
            return Err(NexusError::Internal(message));
        }
        let result_str = unsafe { CStr::from_ptr(result_ptr) }.to_string_lossy();
        let annotated: AnnotatedLLMRequest = serde_json::from_str(&result_str).map_err(|e| {
            unsafe { nat_nexus_string_free_internal(result_ptr) };
            NexusError::Internal(format!("codec decode: invalid JSON: {e}"))
        })?;
        unsafe { nat_nexus_string_free_internal(result_ptr) };
        Ok(annotated)
    }

    fn encode(
        &self,
        annotated: &AnnotatedLLMRequest,
        original: &LLMRequest,
    ) -> nvidia_nat_nexus_core::Result<LLMRequest> {
        clear_last_error();
        let annotated_str = serde_json::to_string(annotated)
            .map_err(|e| NexusError::Internal(format!("codec encode: serialize failed: {e}")))?;
        let c_annotated = CString::new(annotated_str)
            .map_err(|e| NexusError::Internal(format!("codec encode: CString failed: {e}")))?;
        let ffi_req = Box::into_raw(Box::new(FfiLLMRequest(original.clone())));
        let result_ptr =
            unsafe { (self.encode_cb)(self.user_data.ptr, c_annotated.as_ptr(), ffi_req) };
        // Free the input request
        unsafe { drop(Box::from_raw(ffi_req)) };
        if result_ptr.is_null() {
            let message = last_error_message()
                .unwrap_or_else(|| "codec encode callback returned null".to_string());
            return Err(NexusError::Internal(message));
        }
        let result_str = unsafe { CStr::from_ptr(result_ptr) }.to_string_lossy();
        let content: serde_json::Value = serde_json::from_str(&result_str).map_err(|e| {
            unsafe { nat_nexus_string_free_internal(result_ptr) };
            NexusError::Internal(format!("codec encode: invalid result JSON: {e}"))
        })?;
        unsafe { nat_nexus_string_free_internal(result_ptr) };
        Ok(LLMRequest {
            headers: original.headers.clone(),
            content,
        })
    }
}

/// Wrap a pair of C codec callbacks into an `Arc<dyn LlmCodec>`.
pub fn wrap_codec_fn(
    decode_cb: NatNexusCodecDecodeCb,
    encode_cb: NatNexusCodecEncodeCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
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
        return Err(NexusError::Internal(message));
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
unsafe fn nat_nexus_string_free_internal(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use nvidia_nat_nexus_core::types::{Event, LLMAttributes, LLMHandle};
    use serde_json::json;
    use tokio_stream::StreamExt;
    use uuid::Uuid;

    extern "C" fn free_arc_counter(user_data: *mut libc::c_void) {
        let counter = unsafe { Box::from_raw(user_data as *mut Arc<AtomicUsize>) };
        counter.fetch_add(1, Ordering::SeqCst);
    }

    fn user_data_counter() -> (*mut libc::c_void, Arc<AtomicUsize>) {
        let counter = Arc::new(AtomicUsize::new(0));
        let ptr = Box::into_raw(Box::new(counter.clone())) as *mut libc::c_void;
        (ptr, counter)
    }

    unsafe extern "C" fn tool_sanitize_cb(
        user_data: *mut libc::c_void,
        name: *const c_char,
        args_json: *const c_char,
    ) -> *mut c_char {
        let counter = unsafe { &*(user_data as *const Arc<AtomicUsize>) };
        counter.fetch_add(1, Ordering::SeqCst);
        let mut args: Json = serde_json::from_str(
            unsafe { CStr::from_ptr(args_json) }
                .to_str()
                .unwrap_or("null"),
        )
        .unwrap();
        args["name"] = json!(unsafe { CStr::from_ptr(name) }.to_str().unwrap_or_default());
        CString::new(args.to_string()).unwrap().into_raw()
    }

    unsafe extern "C" fn tool_conditional_cb(
        _user_data: *mut libc::c_void,
        _name: *const c_char,
        args_json: *const c_char,
    ) -> *mut c_char {
        let args: Json = serde_json::from_str(
            unsafe { CStr::from_ptr(args_json) }
                .to_str()
                .unwrap_or("null"),
        )
        .unwrap();
        if args["block"] == json!(true) {
            CString::new("blocked").unwrap().into_raw()
        } else {
            std::ptr::null_mut()
        }
    }

    unsafe extern "C" fn tool_exec_cb(
        _user_data: *mut libc::c_void,
        args_json: *const c_char,
    ) -> *mut c_char {
        let mut args: Json = serde_json::from_str(
            unsafe { CStr::from_ptr(args_json) }
                .to_str()
                .unwrap_or("null"),
        )
        .unwrap();
        args["executed"] = json!(true);
        CString::new(args.to_string()).unwrap().into_raw()
    }

    unsafe extern "C" fn tool_exec_error_cb(
        _user_data: *mut libc::c_void,
        _args_json: *const c_char,
    ) -> *mut c_char {
        set_last_error("tool callback failed");
        std::ptr::null_mut()
    }

    unsafe extern "C" fn tool_exec_intercept_cb(
        _user_data: *mut libc::c_void,
        args_json: *const c_char,
        next_fn: NatNexusToolExecNextFn,
        next_ctx: *mut libc::c_void,
    ) -> *mut c_char {
        let result_ptr = unsafe { next_fn(args_json, next_ctx) };
        if result_ptr.is_null() {
            return std::ptr::null_mut();
        }
        let mut result: Json =
            serde_json::from_str(unsafe { CStr::from_ptr(result_ptr) }.to_str().unwrap()).unwrap();
        unsafe { nat_nexus_string_free_internal(result_ptr) };
        result["intercepted"] = json!(true);
        CString::new(result.to_string()).unwrap().into_raw()
    }

    /// Intercept-specific callback with the unified annotated-aware signature
    /// for callable.rs unit tests.
    unsafe extern "C" fn llm_request_intercept_cb(
        _user_data: *mut libc::c_void,
        _name: *const c_char,
        request: *const FfiLLMRequest,
        annotated_json: *const c_char,
        out_request: *mut *mut FfiLLMRequest,
        out_annotated_json: *mut *mut c_char,
    ) -> NatNexusStatus {
        let mut req = unsafe { (&*request).0.clone() };
        req.content["intercepted"] = json!(true);
        unsafe { *out_request = Box::into_raw(Box::new(FfiLLMRequest(req))) };
        if annotated_json.is_null() {
            unsafe { *out_annotated_json = std::ptr::null_mut() };
        } else {
            let s = unsafe { CStr::from_ptr(annotated_json) }
                .to_string_lossy()
                .into_owned();
            unsafe { *out_annotated_json = CString::new(s).unwrap().into_raw() };
        }
        NatNexusStatus::Ok
    }

    unsafe extern "C" fn llm_request_null_cb(
        _user_data: *mut libc::c_void,
        _request: *const FfiLLMRequest,
    ) -> *mut FfiLLMRequest {
        std::ptr::null_mut()
    }

    unsafe extern "C" fn llm_conditional_cb(
        _user_data: *mut libc::c_void,
        request: *const FfiLLMRequest,
    ) -> *mut c_char {
        if unsafe { (&*request).0.content.get("block").cloned() } == Some(json!(true)) {
            CString::new("blocked llm").unwrap().into_raw()
        } else {
            std::ptr::null_mut()
        }
    }

    unsafe extern "C" fn json_cb(
        _user_data: *mut libc::c_void,
        json: *const c_char,
    ) -> *mut c_char {
        let mut value: Json =
            serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
        value["wrapped"] = json!(true);
        CString::new(value.to_string()).unwrap().into_raw()
    }

    unsafe extern "C" fn llm_exec_cb(
        _user_data: *mut libc::c_void,
        native_json: *const c_char,
    ) -> *mut c_char {
        let request: Json =
            serde_json::from_str(unsafe { CStr::from_ptr(native_json) }.to_str().unwrap()).unwrap();
        let response = json!({
            "model": request["content"]["model"].clone(),
            "ok": true,
        });
        CString::new(response.to_string()).unwrap().into_raw()
    }

    unsafe extern "C" fn llm_exec_error_cb(
        _user_data: *mut libc::c_void,
        _native_json: *const c_char,
    ) -> *mut c_char {
        set_last_error("llm callback failed");
        std::ptr::null_mut()
    }

    unsafe extern "C" fn llm_exec_intercept_cb(
        _user_data: *mut libc::c_void,
        native_json: *const c_char,
        next_fn: NatNexusLlmExecNextFn,
        next_ctx: *mut libc::c_void,
    ) -> *mut c_char {
        let result_ptr = unsafe { next_fn(native_json, next_ctx) };
        if result_ptr.is_null() {
            return std::ptr::null_mut();
        }
        let mut value: Json =
            serde_json::from_str(unsafe { CStr::from_ptr(result_ptr) }.to_str().unwrap()).unwrap();
        unsafe { nat_nexus_string_free_internal(result_ptr) };
        value["intercepted"] = json!(true);
        CString::new(value.to_string()).unwrap().into_raw()
    }

    unsafe extern "C" fn llm_exec_short_circuit_cb(
        _user_data: *mut libc::c_void,
        native_json: *const c_char,
        _next_fn: NatNexusLlmExecNextFn,
        _next_ctx: *mut libc::c_void,
    ) -> *mut c_char {
        let request: Json =
            serde_json::from_str(unsafe { CStr::from_ptr(native_json) }.to_str().unwrap()).unwrap();
        let response = json!({
            "model": request["content"]["model"].clone(),
            "intercepted": true,
        });
        CString::new(response.to_string()).unwrap().into_raw()
    }

    static COLLECTED_COUNT: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn collector_cb(_chunk: *const c_char) {
        COLLECTED_COUNT.fetch_add(1, Ordering::SeqCst);
    }

    unsafe extern "C" fn finalizer_cb() -> *mut c_char {
        CString::new(r#"{"done":true}"#).unwrap().into_raw()
    }

    unsafe extern "C" fn subscriber_cb(user_data: *mut libc::c_void, event: *const FfiEvent) {
        let counter = unsafe { &*(user_data as *const Arc<AtomicUsize>) };
        if unsafe { (&*event).0.name() } == "ffi-event" {
            counter.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn make_request() -> LLMRequest {
        LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"model": "test-model"}),
        }
    }

    #[test]
    fn test_wrap_tool_request_and_conditional_callbacks() {
        let (user_data, called) = user_data_counter();
        let wrapped = wrap_tool_sanitize_fn(tool_sanitize_cb, user_data, Some(free_arc_counter));
        let result = wrapped("tool-name", json!({"value": 1}));
        assert_eq!(result["value"], json!(1));
        assert_eq!(result["name"], json!("tool-name"));
        assert_eq!(called.load(Ordering::SeqCst), 1);
        drop(wrapped);
        assert_eq!(called.load(Ordering::SeqCst), 2);

        let wrapped_conditional =
            wrap_tool_conditional_fn(tool_conditional_cb, std::ptr::null_mut(), None);
        assert_eq!(
            wrapped_conditional("tool", &json!({"block": true})).unwrap(),
            Some("blocked".into())
        );
        assert_eq!(
            wrapped_conditional("tool", &json!({"block": false})).unwrap(),
            None
        );
    }

    #[test]
    fn test_wrap_tool_exec_and_intercept_callbacks() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        let exec = wrap_tool_exec_fn(tool_exec_cb, std::ptr::null_mut(), None);
        let result = runtime.block_on(exec(json!({"value": 2}))).unwrap();
        assert_eq!(result["executed"], json!(true));

        let exec_err = wrap_tool_exec_fn(tool_exec_error_cb, std::ptr::null_mut(), None);
        let err = runtime.block_on(exec_err(json!({}))).unwrap_err();
        assert!(err.to_string().contains("tool callback failed"));

        let intercept =
            wrap_tool_exec_intercept_fn(tool_exec_intercept_cb, std::ptr::null_mut(), None);
        let next: ToolExecutionNextFn =
            Arc::new(|args| Box::pin(async move { Ok(json!({"from_next": args})) }));
        let intercepted = runtime
            .block_on(intercept("tool", json!({"v": 1}), next))
            .unwrap();
        assert_eq!(intercepted["intercepted"], json!(true));
        assert_eq!(intercepted["from_next"]["v"], json!(1));

        let failing_intercept =
            wrap_tool_exec_intercept_fn(tool_exec_intercept_cb, std::ptr::null_mut(), None);
        let failing_next: ToolExecutionNextFn =
            Arc::new(|_| Box::pin(async { Err(NexusError::Internal("next failed".into())) }));
        let err = runtime
            .block_on(failing_intercept("tool", json!({"v": 2}), failing_next))
            .unwrap_err();
        assert!(err.to_string().contains("next failed"));
    }

    #[test]
    fn test_wrap_llm_request_response_and_conditional_callbacks() {
        let request_intercept =
            wrap_llm_request_intercept_fn(llm_request_intercept_cb, std::ptr::null_mut(), None);
        let (intercepted, _annotated) = request_intercept("llm", make_request(), None).unwrap();
        assert_eq!(intercepted.content["intercepted"], json!(true));

        let sanitize_request =
            wrap_llm_sanitize_request_fn(llm_request_null_cb, std::ptr::null_mut(), None);
        let sanitized = sanitize_request(make_request());
        assert_eq!(sanitized.headers.len(), 0);
        assert_eq!(sanitized.content, Json::Null);

        let conditional = wrap_llm_conditional_fn(llm_conditional_cb, std::ptr::null_mut(), None);
        assert_eq!(
            conditional(&LLMRequest {
                headers: serde_json::Map::new(),
                content: json!({"block": true}),
            })
            .unwrap(),
            Some("blocked llm".into())
        );
        assert_eq!(conditional(&make_request()).unwrap(), None);

        let wrapped_json = wrap_json_fn(json_cb, std::ptr::null_mut(), None);
        assert_eq!(wrapped_json(json!({"value": 1}))["wrapped"], json!(true));

        let wrapped_response = wrap_llm_response_fn(json_cb, std::ptr::null_mut(), None);
        assert_eq!(
            wrapped_response(json!({"value": 2}))["wrapped"],
            json!(true)
        );
    }

    #[test]
    fn test_wrap_llm_exec_stream_and_event_callbacks() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        let exec = wrap_llm_exec_fn(llm_exec_cb, std::ptr::null_mut(), None);
        let result = runtime.block_on(exec(make_request())).unwrap();
        assert_eq!(result["ok"], json!(true));
        assert_eq!(result["model"], json!("test-model"));

        let exec_err = wrap_llm_exec_fn(llm_exec_error_cb, std::ptr::null_mut(), None);
        let err = runtime.block_on(exec_err(make_request())).unwrap_err();
        assert!(err.to_string().contains("llm callback failed"));

        let intercept =
            wrap_llm_exec_intercept_fn(llm_exec_intercept_cb, std::ptr::null_mut(), None);
        let next: LlmExecutionNextFn = Arc::new(|request| {
            Box::pin(async move { Ok(json!({"model": request.content["model"]})) })
        });
        let intercepted = runtime
            .block_on(intercept("llm", make_request(), next))
            .unwrap();
        assert_eq!(intercepted["intercepted"], json!(true));

        let stream_exec = wrap_llm_stream_exec_fn(llm_exec_cb, std::ptr::null_mut(), None);
        let mut stream = runtime.block_on(stream_exec(make_request())).unwrap();
        let first = runtime.block_on(async { stream.next().await.unwrap().unwrap() });
        assert_eq!(first["ok"], json!(true));

        let stream_intercept = wrap_llm_stream_exec_intercept_fn(
            llm_exec_short_circuit_cb,
            std::ptr::null_mut(),
            None,
        );
        let next_stream: LlmStreamExecutionNextFn = Arc::new(|_request| {
            Box::pin(async {
                Ok(
                    Box::pin(tokio_stream::iter(vec![Ok(json!({"ignored": true}))]))
                        as Pin<Box<dyn Stream<Item = Result<Json>> + Send>>,
                )
            })
        });
        let mut intercepted_stream = runtime
            .block_on(stream_intercept("llm", make_request(), next_stream))
            .unwrap();
        let first = runtime.block_on(async { intercepted_stream.next().await.unwrap().unwrap() });
        assert_eq!(first["intercepted"], json!(true));

        let stream_intercept_with_next =
            wrap_llm_stream_exec_intercept_fn(llm_exec_intercept_cb, std::ptr::null_mut(), None);
        let next_stream: LlmStreamExecutionNextFn = Arc::new(|request| {
            Box::pin(async move {
                Ok(Box::pin(tokio_stream::iter(vec![Ok(json!({
                    "model": request.content["model"].clone()
                }))]))
                    as Pin<Box<dyn Stream<Item = Result<Json>> + Send>>)
            })
        });
        let mut intercepted_stream = runtime
            .block_on(stream_intercept_with_next(
                "llm",
                make_request(),
                next_stream,
            ))
            .unwrap();
        let first = runtime.block_on(async { intercepted_stream.next().await.unwrap().unwrap() });
        assert_eq!(first["intercepted"], json!(true));
        assert_eq!(first["model"], json!("test-model"));

        COLLECTED_COUNT.store(0, Ordering::SeqCst);
        let mut collector = wrap_collector_fn(collector_cb);
        collector(json!({"chunk": 1})).unwrap();
        assert_eq!(COLLECTED_COUNT.load(Ordering::SeqCst), 1);

        let finalizer = wrap_finalizer_fn(finalizer_cb);
        assert_eq!(finalizer(), json!({"done": true}));

        let (user_data, seen) = user_data_counter();
        let subscriber = wrap_event_subscriber(subscriber_cb, user_data, Some(free_arc_counter));
        let event = Event::llm_start(
            None,
            Uuid::new_v4(),
            "ffi-event",
            None,
            None,
            LLMAttributes::empty(),
            None,
            Some("test-model".into()),
        );
        subscriber(&event);
        assert_eq!(seen.load(Ordering::SeqCst), 1);
        drop(subscriber);
        assert_eq!(seen.load(Ordering::SeqCst), 2);

        let handle = LLMHandle::new("llm".into(), LLMAttributes::STATELESS, None, None, None);
        assert_eq!(handle.name, "llm");
    }
}
