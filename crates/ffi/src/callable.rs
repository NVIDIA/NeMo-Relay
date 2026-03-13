// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::type_complexity)]
//! C function pointer typedefs and wrapper functions for FFI callbacks.
//!
//! This module defines the callback signatures used by the C API for tool and
//! LLM guardrails, intercepts, execution functions, and event subscribers. Each
//! `pub type` alias corresponds to a C function pointer that appears in the
//! generated `nvmagic.h` header.
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
use tokio_stream::Stream;

use nvmagic_core::types::{
    IdentityConverter, LLMConverter, LLMRequest, LLMResponse, ToRequestFn, ToResponseFn,
};
use nvmagic_core::{LlmExecutionNextFn, LlmStreamExecutionNextFn, Result, ToolExecutionNextFn};

use crate::convert::json_to_c_string;
use crate::types::{FfiEvent, FfiLLMRequest};

// ---------------------------------------------------------------------------
// Callback typedefs (mirrored in the C header)
// ---------------------------------------------------------------------------

/// Optional destructor for user data passed to callbacks.
/// Called when the runtime no longer needs the associated callback.
pub type NvMagicFreeFn = Option<unsafe extern "C" fn(user_data: *mut libc::c_void)>;

/// Callback for tool request/response sanitization guardrails and intercepts.
/// Receives tool name and arguments as JSON, returns sanitized arguments as JSON.
/// The returned string must be allocated with `malloc` or equivalent.
pub type NvMagicToolSanitizeCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    name: *const c_char,
    args_json: *const c_char,
) -> *mut c_char;

/// Callback for tool conditional execution guardrails.
/// Receives tool name and arguments as JSON.
/// Returns NULL to allow execution, or an error message string to reject.
pub type NvMagicToolConditionalCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    name: *const c_char,
    args_json: *const c_char,
) -> *mut c_char;

/// Callback for tool execution intercept conditions.
/// Receives tool name and arguments as JSON.
/// Returns `true` if this intercept should handle the execution.
pub type NvMagicToolExecConditionalCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    name: *const c_char,
    args_json: *const c_char,
) -> bool;

/// Callback for tool execution (default callable). Receives arguments as JSON,
/// returns result as JSON. The returned string must be allocated with `malloc`
/// or equivalent.
pub type NvMagicToolExecCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, args_json: *const c_char) -> *mut c_char;

/// Runtime-provided "next" callback for tool execution middleware chain.
/// Call this from an intercept to invoke the next layer (or original function).
/// `next_ctx` is an opaque pointer managed by the runtime.
pub type NvMagicToolExecNextFn =
    unsafe extern "C" fn(args_json: *const c_char, next_ctx: *mut libc::c_void) -> *mut c_char;

/// Callback for tool execution intercepts. Receives arguments as JSON plus
/// a `next` callback and its context. Call `next_fn(args, next_ctx)` to invoke
/// the next layer in the middleware chain, or return directly to short-circuit.
pub type NvMagicToolExecInterceptCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    args_json: *const c_char,
    next_fn: NvMagicToolExecNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char;

/// Generic JSON-to-JSON callback, used for LLM response sanitization and intercepts.
/// The returned string must be allocated with `malloc` or equivalent.
pub type NvMagicJsonCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, json: *const c_char) -> *mut c_char;

/// Callback for LLM request sanitization. Receives an `FfiLLMRequest` and returns
/// a new (possibly modified) `FfiLLMRequest`. Return null to use defaults.
pub type NvMagicLlmRequestCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut FfiLLMRequest;

/// Callback for LLM conditional execution guardrails.
/// Returns NULL to allow execution, or an error message string to reject.
pub type NvMagicLlmConditionalCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut c_char;

/// Callback for LLM execution intercept conditions.
/// Receives native JSON string. Returns `true` if this intercept should handle the execution.
pub type NvMagicLlmExecConditionalCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, native_json: *const c_char) -> bool;

/// Callback for LLM execution (default callable). Receives a native JSON C string,
/// returns the response as a JSON C string.
pub type NvMagicLlmExecCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, native_json: *const c_char) -> *mut c_char;

/// Runtime-provided "next" callback for LLM execution middleware chain.
/// Takes a native JSON C string, returns a response JSON C string.
pub type NvMagicLlmExecNextFn =
    unsafe extern "C" fn(native_json: *const c_char, next_ctx: *mut libc::c_void) -> *mut c_char;

/// Callback for LLM execution intercepts with middleware chain support.
/// Receives native JSON C string plus a `next` callback and its context.
pub type NvMagicLlmExecInterceptCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    native_json: *const c_char,
    next_fn: NvMagicLlmExecNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char;

/// Callback for event subscribers. Invoked on each lifecycle event emitted by
/// the runtime. The `FfiEvent` pointer is only valid for the duration of the call.
pub type NvMagicEventSubscriberCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, event: *const FfiEvent);

/// Callback for collecting intercepted stream chunks. Invoked with each chunk
/// (after stream response intercepts have been applied) as a null-terminated
/// C string. The string is only valid for the duration of the call.
pub type NvMagicCollectorCb = unsafe extern "C" fn(chunk: *const c_char);

/// Callback for finalizing a collected stream. Invoked once when the stream is
/// exhausted. Must return a JSON C string representing the aggregated response.
/// The returned string must be allocated with `malloc` or equivalent; the
/// runtime will free it.
pub type NvMagicFinalizerCb = unsafe extern "C" fn() -> *mut c_char;

// ---------------------------------------------------------------------------
// Shared user_data wrapper (ensures cleanup)
// ---------------------------------------------------------------------------

/// RAII wrapper around a C user-data pointer and its associated free function.
/// Ensures the free function is called exactly once when dropped.
struct UserData {
    ptr: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
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
    free_fn: NvMagicFreeFn,
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
    cb: NvMagicToolSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> Box<dyn Fn(&str, Json) -> Json + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |name: &str, args: Json| {
        let c_name = CString::new(name).unwrap_or_default();
        let c_args = json_to_c_string(&args);
        let result_ptr = unsafe { cb(ud.ptr, c_name.as_ptr(), c_args) };
        unsafe { nvmagic_string_free_internal(c_args) };
        let result = ptr_to_json(result_ptr);
        unsafe { nvmagic_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C tool conditional callback into a Rust closure for use by the core runtime.
pub fn wrap_tool_conditional_fn(
    cb: NvMagicToolConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> Box<dyn Fn(&str, &Json) -> Option<String> + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |name: &str, args: &Json| {
        let c_name = CString::new(name).unwrap_or_default();
        let c_args = json_to_c_string(args);
        let result_ptr = unsafe { cb(ud.ptr, c_name.as_ptr(), c_args) };
        unsafe { nvmagic_string_free_internal(c_args) };
        let result = ptr_to_opt_string(result_ptr);
        unsafe { nvmagic_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C tool execution conditional callback into a Rust closure.
pub fn wrap_tool_exec_conditional_fn(
    cb: NvMagicToolExecConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> Box<dyn Fn(&str, &Json) -> bool + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |name: &str, args: &Json| {
        let c_name = CString::new(name).unwrap_or_default();
        let c_args = json_to_c_string(args);
        let result = unsafe { cb(ud.ptr, c_name.as_ptr(), c_args) };
        unsafe { nvmagic_string_free_internal(c_args) };
        result
    })
}

/// Wrap a C tool execution callback into an async Rust closure.
pub fn wrap_tool_exec_fn(
    cb: NvMagicToolExecCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> Box<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |args: Json| {
        let ud = ud.clone();
        Box::pin(async move {
            let c_args = json_to_c_string(&args);
            let result_ptr = unsafe { cb(ud.ptr, c_args) };
            unsafe { nvmagic_string_free_internal(c_args) };
            let result = ptr_to_json(result_ptr);
            unsafe { nvmagic_string_free_internal(result_ptr) };
            Ok(result)
        })
    })
}

/// Wrap a C tool execution intercept callback into an `Arc<dyn Fn(Json, ToolExecutionNextFn) -> ...>`.
///
/// The wrapper packages the Rust `ToolExecutionNextFn` into a C-callable
/// `(next_fn, next_ctx)` pair and passes both to the C intercept callback.
pub fn wrap_tool_exec_intercept_fn(
    cb: NvMagicToolExecInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> Arc<
    dyn Fn(Json, ToolExecutionNextFn) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |args: Json, next: ToolExecutionNextFn| {
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
                let next = unsafe { Box::from_raw(next_ctx as *mut ToolExecutionNextFn) };
                let args = if args_json.is_null() {
                    Json::Null
                } else {
                    let s = unsafe { CStr::from_ptr(args_json) }.to_string_lossy();
                    serde_json::from_str(&s).unwrap_or(Json::Null)
                };
                // Block on the async next fn (we're already in a tokio context)
                let handle = tokio::runtime::Handle::current();
                let result = handle.block_on(next(args));
                match result {
                    Ok(json) => json_to_c_string(&json),
                    Err(_) => std::ptr::null_mut(),
                }
            }

            let c_args = json_to_c_string(&args);
            let result_ptr = unsafe { cb(ud.ptr, c_args, tool_next_trampoline, next_ctx) };
            unsafe { nvmagic_string_free_internal(c_args) };
            let result = ptr_to_json(result_ptr);
            unsafe { nvmagic_string_free_internal(result_ptr) };
            Ok(result)
        })
    })
}

/// Wrap a C LLM execution intercept callback into an `Arc<dyn Fn(Json, LlmExecutionNextFn) -> ...>`.
pub fn wrap_llm_exec_intercept_fn(
    cb: NvMagicLlmExecInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> Arc<
    dyn Fn(Json, LlmExecutionNextFn) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let ud = make_user_data(user_data, free_fn);
    Arc::new(move |native: Json, next: LlmExecutionNextFn| {
        let ud = ud.clone();
        Box::pin(async move {
            let next_box = Box::new(next);
            let next_ctx = Box::into_raw(next_box) as *mut libc::c_void;

            /// C trampoline that calls the boxed Rust next fn.
            /// Takes a native JSON string, deserializes to Json, calls the Rust LlmExecutionNextFn.
            unsafe extern "C" fn llm_next_trampoline(
                native_json: *const c_char,
                next_ctx: *mut libc::c_void,
            ) -> *mut c_char {
                let next = unsafe { Box::from_raw(next_ctx as *mut LlmExecutionNextFn) };
                let native = if native_json.is_null() {
                    Json::Null
                } else {
                    let s = unsafe { CStr::from_ptr(native_json) }.to_string_lossy();
                    serde_json::from_str(&s).unwrap_or(Json::Null)
                };
                let handle = tokio::runtime::Handle::current();
                let result = handle.block_on(next(native));
                match result {
                    Ok(json) => json_to_c_string(&json),
                    Err(_) => std::ptr::null_mut(),
                }
            }

            let c_native = json_to_c_string(&native);
            let result_ptr = unsafe { cb(ud.ptr, c_native, llm_next_trampoline, next_ctx) };
            unsafe { nvmagic_string_free_internal(c_native) };
            let result = ptr_to_json(result_ptr);
            unsafe { nvmagic_string_free_internal(result_ptr) };
            Ok(result)
        })
    })
}

/// Wrap a C LLM stream execution intercept callback.
/// Since the C callback returns a single string (not a real stream), this wraps
/// it as a single-item stream, same as `wrap_llm_stream_exec_fn`.
pub fn wrap_llm_stream_exec_intercept_fn(
    cb: NvMagicLlmExecInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> Arc<
    dyn Fn(
            Json,
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
    Arc::new(move |native: Json, _next: LlmStreamExecutionNextFn| {
        let ud = ud.clone();
        Box::pin(async move {
            // For stream intercepts from C, we ignore next and just call the C callback
            // with a no-op next (the C API doesn't support chaining streams easily)

            unsafe extern "C" fn noop_llm_next(
                _native_json: *const c_char,
                _next_ctx: *mut libc::c_void,
            ) -> *mut c_char {
                std::ptr::null_mut()
            }

            let c_native = json_to_c_string(&native);
            let result_ptr = unsafe { cb(ud.ptr, c_native, noop_llm_next, std::ptr::null_mut()) };
            unsafe { nvmagic_string_free_internal(c_native) };
            let result = ptr_to_json(result_ptr);
            unsafe { nvmagic_string_free_internal(result_ptr) };
            let stream = tokio_stream::once(Ok(result));
            Ok(Box::pin(stream) as Pin<Box<dyn Stream<Item = Result<Json>> + Send>>)
        })
    })
}

/// Wrap a generic C JSON callback into a Rust closure.
pub fn wrap_json_fn(
    cb: NvMagicJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> Box<dyn Fn(Json) -> Json + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |value: Json| {
        let c_json = json_to_c_string(&value);
        let result_ptr = unsafe { cb(ud.ptr, c_json) };
        unsafe { nvmagic_string_free_internal(c_json) };
        let result = ptr_to_json(result_ptr);
        unsafe { nvmagic_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C JSON callback into a `Fn(LLMResponse) -> LLMResponse` closure.
/// The LLMResponse is serialized to a JSON string for the C callback, and the
/// returned JSON string is deserialized back to LLMResponse.
pub fn wrap_llm_response_fn(
    cb: NvMagicJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> Box<dyn Fn(LLMResponse) -> LLMResponse + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |response: LLMResponse| {
        let json_value = serde_json::to_value(&response).unwrap_or(Json::Null);
        let c_json = json_to_c_string(&json_value);
        let result_ptr = unsafe { cb(ud.ptr, c_json) };
        unsafe { nvmagic_string_free_internal(c_json) };
        let result_json = ptr_to_json(result_ptr);
        unsafe { nvmagic_string_free_internal(result_ptr) };
        // Try to deserialize as LLMResponse, fall back to wrapping in data field
        serde_json::from_value::<LLMResponse>(result_json.clone())
            .unwrap_or(LLMResponse { data: result_json })
    })
}

/// Wrap a C LLM request sanitize callback into a Rust closure.
pub fn wrap_llm_sanitize_request_fn(
    cb: NvMagicLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
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
    cb: NvMagicLlmConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> Box<dyn Fn(&LLMRequest) -> Option<String> + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |request: &LLMRequest| {
        let ffi_req = FfiLLMRequest(request.clone());
        let result_ptr = unsafe { cb(ud.ptr, &ffi_req) };
        let result = ptr_to_opt_string(result_ptr);
        unsafe { nvmagic_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C LLM execution conditional callback into a Rust closure.
/// The C callback receives the native Json serialized as a JSON string.
pub fn wrap_llm_exec_conditional_fn(
    cb: NvMagicLlmExecConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> Box<dyn Fn(&Json) -> bool + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |native: &Json| {
        let c_native = json_to_c_string(native);
        let result = unsafe { cb(ud.ptr, c_native) };
        unsafe { nvmagic_string_free_internal(c_native) };
        result
    })
}

/// Wrap a C LLM execution callback into an async Rust closure.
/// The C callback receives native Json serialized as a JSON string.
pub fn wrap_llm_exec_fn(
    cb: NvMagicLlmExecCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> Box<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |native: Json| {
        let ud = ud.clone();
        Box::pin(async move {
            let c_native = json_to_c_string(&native);
            let result_ptr = unsafe { cb(ud.ptr, c_native) };
            unsafe { nvmagic_string_free_internal(c_native) };
            let result = ptr_to_json(result_ptr);
            unsafe { nvmagic_string_free_internal(result_ptr) };
            Ok(result)
        })
    })
}

/// Wrap a C LLM execution callback into an async Rust closure that returns a stream.
/// The C callback returns the full response as a single JSON string, which is emitted
/// as a single-item stream of Json values.
pub fn wrap_llm_stream_exec_fn(
    cb: NvMagicLlmExecCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> Box<
    dyn Fn(
            Json,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<Pin<Box<dyn Stream<Item = Result<Json>> + Send>>>>
                    + Send,
            >,
        > + Send
        + Sync,
> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |native: Json| {
        let ud = ud.clone();
        Box::pin(async move {
            let c_native = json_to_c_string(&native);
            let result_ptr = unsafe { cb(ud.ptr, c_native) };
            unsafe { nvmagic_string_free_internal(c_native) };
            let result = ptr_to_json(result_ptr);
            unsafe { nvmagic_string_free_internal(result_ptr) };
            // The C callback returns the full response as a single JSON value for stream
            // We emit it as a single-item stream
            let stream = tokio_stream::once(Ok(result));
            Ok(Box::pin(stream) as Pin<Box<dyn Stream<Item = Result<Json>> + Send>>)
        })
    })
}

/// Wrap a C collector callback into a `Box<dyn FnMut(Json) + Send>` for use
/// by the core runtime. Each intercepted chunk Json is serialized to a JSON
/// string and passed to the callback.
///
/// # Safety
/// The caller must ensure `cb` remains valid for the lifetime of the returned
/// closure. The C callback is invoked synchronously from the stream-consumption
/// task.
pub fn wrap_collector_fn(cb: NvMagicCollectorCb) -> Box<dyn FnMut(Json) + Send> {
    // NvMagicCollectorCb is a plain `extern "C" fn` pointer (no user_data),
    // which is Copy + Send, so it can be moved into the closure directly.
    Box::new(move |chunk: Json| {
        let c_chunk = json_to_c_string(&chunk);
        unsafe { cb(c_chunk) };
        unsafe { nvmagic_string_free_internal(c_chunk) };
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
pub fn wrap_finalizer_fn(cb: NvMagicFinalizerCb) -> Box<dyn FnOnce() -> Json + Send> {
    Box::new(move || {
        let result_ptr = unsafe { cb() };
        let result = ptr_to_json(result_ptr);
        unsafe { nvmagic_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C event subscriber callback into a Rust closure.
pub fn wrap_event_subscriber(
    cb: NvMagicEventSubscriberCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> Box<dyn Fn(&nvmagic_core::Event) + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |event: &nvmagic_core::Event| {
        let ffi_event = FfiEvent(event.clone());
        unsafe { cb(ud.ptr, &ffi_event) };
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
unsafe fn nvmagic_string_free_internal(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}

// ---------------------------------------------------------------------------
// Per-call LLM converter wrappers
// ---------------------------------------------------------------------------

/// Wrap an FFI callback triple into a [`ToRequestFn`].
///
/// The C callback receives the native JSON payload as a C string and must return
/// a heap-allocated JSON C string representing a serialized [`LLMRequest`].
/// If the callback returns null or an unparseable result, the identity converter
/// is used as a fallback.
pub(crate) fn wrap_to_request_fn(
    cb: NvMagicJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> ToRequestFn {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |native: &Json| {
        let c_json = json_to_c_string(native);
        let result_ptr = unsafe { cb(ud.ptr, c_json) };
        unsafe { nvmagic_string_free_internal(c_json) };
        if result_ptr.is_null() {
            return IdentityConverter.to_request(native);
        }
        let result_json = ptr_to_json(result_ptr);
        unsafe { nvmagic_string_free_internal(result_ptr) };
        serde_json::from_value::<LLMRequest>(result_json)
            .unwrap_or_else(|_| IdentityConverter.to_request(native))
    })
}

/// Wrap an FFI callback triple into a [`ToResponseFn`].
///
/// The C callback receives the native JSON payload as a C string and must return
/// a heap-allocated JSON C string representing a serialized [`LLMResponse`].
/// If the callback returns null or an unparseable result, the identity converter
/// is used as a fallback.
pub(crate) fn wrap_to_response_fn(
    cb: NvMagicJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> ToResponseFn {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |native: &Json| {
        let c_json = json_to_c_string(native);
        let result_ptr = unsafe { cb(ud.ptr, c_json) };
        unsafe { nvmagic_string_free_internal(c_json) };
        if result_ptr.is_null() {
            return IdentityConverter.to_response(native);
        }
        let result_json = ptr_to_json(result_ptr);
        unsafe { nvmagic_string_free_internal(result_ptr) };
        serde_json::from_value::<LLMResponse>(result_json)
            .unwrap_or_else(|_| IdentityConverter.to_response(native))
    })
}
