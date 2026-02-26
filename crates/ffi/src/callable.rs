#![allow(clippy::type_complexity)]
//! C function pointer typedefs and wrapper functions for FFI callbacks.
//!
//! This module defines the callback signatures used by the C API for tool and
//! LLM guardrails, intercepts, execution functions, and event subscribers. Each
//! `pub type` alias corresponds to a C function pointer that appears in the
//! generated `nvagentrt.h` header.
//!
//! The `wrap_*` functions convert C callbacks (with opaque `user_data` pointers)
//! into Rust closures (`Box<dyn Fn(...)>`) that the core runtime can invoke.
//! Each wrapper captures the user data and its optional free function in an
//! `Arc<UserData>` so the closure is `Send + Sync` and the free function is
//! called exactly once when all references are dropped.

use std::ffi::{CStr, CString};
use std::future::Future;
use std::pin::Pin;

use libc::c_char;
use serde_json::Value as Json;
use tokio_stream::Stream;

use nvagentrt_core::types::{LLMRequest, SseEvent};
use nvagentrt_core::Result;

use crate::convert::json_to_c_string;
use crate::types::{FfiEvent, FfiLLMRequest};

// ---------------------------------------------------------------------------
// Callback typedefs (mirrored in the C header)
// ---------------------------------------------------------------------------

/// Optional destructor for user data passed to callbacks.
/// Called when the runtime no longer needs the associated callback.
pub type NvAgentRtFreeFn = Option<unsafe extern "C" fn(user_data: *mut libc::c_void)>;

/// Callback for tool request/response sanitization guardrails and intercepts.
/// Receives tool name and arguments as JSON, returns sanitized arguments as JSON.
/// The returned string must be allocated with `malloc` or equivalent.
pub type NvAgentRtToolSanitizeCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    name: *const c_char,
    args_json: *const c_char,
) -> *mut c_char;

/// Callback for tool conditional execution guardrails.
/// Receives tool name and arguments as JSON.
/// Returns NULL to allow execution, or an error message string to reject.
pub type NvAgentRtToolConditionalCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    name: *const c_char,
    args_json: *const c_char,
) -> *mut c_char;

/// Callback for tool execution intercept conditions.
/// Receives tool name and arguments as JSON.
/// Returns `true` if this intercept should handle the execution.
pub type NvAgentRtToolExecConditionalCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    name: *const c_char,
    args_json: *const c_char,
) -> bool;

/// Callback for tool execution. Receives arguments as JSON, returns result as JSON.
/// The returned string must be allocated with `malloc` or equivalent.
pub type NvAgentRtToolExecCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, args_json: *const c_char) -> *mut c_char;

/// Generic JSON-to-JSON callback, used for LLM response sanitization and intercepts.
/// The returned string must be allocated with `malloc` or equivalent.
pub type NvAgentRtJsonCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, json: *const c_char) -> *mut c_char;

/// Callback for LLM request sanitization. Receives an `FfiLLMRequest` and returns
/// a new (possibly modified) `FfiLLMRequest`. Return null to use defaults.
pub type NvAgentRtLlmRequestCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut FfiLLMRequest;

/// Callback for LLM conditional execution guardrails.
/// Returns NULL to allow execution, or an error message string to reject.
pub type NvAgentRtLlmConditionalCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut c_char;

/// Callback for LLM execution intercept conditions.
/// Returns `true` if this intercept should handle the execution.
pub type NvAgentRtLlmExecConditionalCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, request: *const FfiLLMRequest) -> bool;

/// Callback for LLM execution. Receives an `FfiLLMRequest`, returns the response
/// as a JSON C string. The returned string must be allocated with `malloc` or equivalent.
pub type NvAgentRtLlmExecCb = unsafe extern "C" fn(
    user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut c_char;

/// Callback for SSE stream response intercepts. Receives the SSE event serialized
/// as a JSON C string and returns a (possibly modified) JSON C string.
pub type NvAgentRtSseInterceptCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, sse_json: *const c_char) -> *mut c_char;

/// Callback for event subscribers. Invoked on each lifecycle event emitted by
/// the runtime. The `FfiEvent` pointer is only valid for the duration of the call.
pub type NvAgentRtEventSubscriberCb =
    unsafe extern "C" fn(user_data: *mut libc::c_void, event: *const FfiEvent);

// ---------------------------------------------------------------------------
// Shared user_data wrapper (ensures cleanup)
// ---------------------------------------------------------------------------

/// RAII wrapper around a C user-data pointer and its associated free function.
/// Ensures the free function is called exactly once when dropped.
struct UserData {
    ptr: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
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
    free_fn: NvAgentRtFreeFn,
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
    cb: NvAgentRtToolSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> Box<dyn Fn(&str, Json) -> Json + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |name: &str, args: Json| {
        let c_name = CString::new(name).unwrap_or_default();
        let c_args = json_to_c_string(&args);
        let result_ptr = unsafe { cb(ud.ptr, c_name.as_ptr(), c_args) };
        unsafe { nv_agentrt_string_free_internal(c_args) };
        let result = ptr_to_json(result_ptr);
        unsafe { nv_agentrt_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C tool conditional callback into a Rust closure for use by the core runtime.
pub fn wrap_tool_conditional_fn(
    cb: NvAgentRtToolConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> Box<dyn Fn(&str, &Json) -> Option<String> + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |name: &str, args: &Json| {
        let c_name = CString::new(name).unwrap_or_default();
        let c_args = json_to_c_string(args);
        let result_ptr = unsafe { cb(ud.ptr, c_name.as_ptr(), c_args) };
        unsafe { nv_agentrt_string_free_internal(c_args) };
        let result = ptr_to_opt_string(result_ptr);
        unsafe { nv_agentrt_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C tool execution conditional callback into a Rust closure.
pub fn wrap_tool_exec_conditional_fn(
    cb: NvAgentRtToolExecConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> Box<dyn Fn(&str, &Json) -> bool + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |name: &str, args: &Json| {
        let c_name = CString::new(name).unwrap_or_default();
        let c_args = json_to_c_string(args);
        let result = unsafe { cb(ud.ptr, c_name.as_ptr(), c_args) };
        unsafe { nv_agentrt_string_free_internal(c_args) };
        result
    })
}

/// Wrap a C tool execution callback into an async Rust closure.
pub fn wrap_tool_exec_fn(
    cb: NvAgentRtToolExecCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> Box<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |args: Json| {
        let ud = ud.clone();
        Box::pin(async move {
            let c_args = json_to_c_string(&args);
            let result_ptr = unsafe { cb(ud.ptr, c_args) };
            unsafe { nv_agentrt_string_free_internal(c_args) };
            let result = ptr_to_json(result_ptr);
            unsafe { nv_agentrt_string_free_internal(result_ptr) };
            Ok(result)
        })
    })
}

/// Wrap a generic C JSON callback into a Rust closure.
pub fn wrap_json_fn(
    cb: NvAgentRtJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> Box<dyn Fn(Json) -> Json + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |value: Json| {
        let c_json = json_to_c_string(&value);
        let result_ptr = unsafe { cb(ud.ptr, c_json) };
        unsafe { nv_agentrt_string_free_internal(c_json) };
        let result = ptr_to_json(result_ptr);
        unsafe { nv_agentrt_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C LLM request sanitize callback into a Rust closure.
pub fn wrap_llm_sanitize_request_fn(
    cb: NvAgentRtLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
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
                method: String::new(),
                url: String::new(),
                headers: serde_json::Map::new(),
                body: Json::Null,
            }
        } else {
            let result = unsafe { Box::from_raw(result_ptr) };
            result.0
        }
    })
}

/// Wrap a C LLM conditional callback into a Rust closure.
pub fn wrap_llm_conditional_fn(
    cb: NvAgentRtLlmConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> Box<dyn Fn(&LLMRequest) -> Option<String> + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |request: &LLMRequest| {
        let ffi_req = FfiLLMRequest(request.clone());
        let result_ptr = unsafe { cb(ud.ptr, &ffi_req) };
        let result = ptr_to_opt_string(result_ptr);
        unsafe { nv_agentrt_string_free_internal(result_ptr) };
        result
    })
}

/// Wrap a C LLM execution conditional callback into a Rust closure.
pub fn wrap_llm_exec_conditional_fn(
    cb: NvAgentRtLlmExecConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> Box<dyn Fn(&LLMRequest) -> bool + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |request: &LLMRequest| {
        let ffi_req = FfiLLMRequest(request.clone());
        unsafe { cb(ud.ptr, &ffi_req) }
    })
}

/// Wrap a C LLM execution callback into an async Rust closure.
pub fn wrap_llm_exec_fn(
    cb: NvAgentRtLlmExecCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> Box<dyn Fn(LLMRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |request: LLMRequest| {
        let ud = ud.clone();
        Box::pin(async move {
            let ffi_req = FfiLLMRequest(request);
            let result_ptr = unsafe { cb(ud.ptr, &ffi_req) };
            let result = ptr_to_json(result_ptr);
            unsafe { nv_agentrt_string_free_internal(result_ptr) };
            Ok(result)
        })
    })
}

/// Wrap a C LLM execution callback into an async Rust closure that returns a stream.
/// The C callback returns the full response as a single string, which is emitted
/// as a single-item stream.
pub fn wrap_llm_stream_exec_fn(
    cb: NvAgentRtLlmExecCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> Box<
    dyn Fn(
            LLMRequest,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>>>
                    + Send,
            >,
        > + Send
        + Sync,
> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |request: LLMRequest| {
        let ud = ud.clone();
        Box::pin(async move {
            let ffi_req = FfiLLMRequest(request);
            let result_ptr = unsafe { cb(ud.ptr, &ffi_req) };
            let raw = ptr_to_string(result_ptr).unwrap_or_default();
            unsafe { nv_agentrt_string_free_internal(result_ptr) };
            // The C callback returns the full response as a single string for stream
            // We emit it as a single-item stream
            let stream = tokio_stream::once(Ok(raw));
            Ok(Box::pin(stream) as Pin<Box<dyn Stream<Item = Result<String>> + Send>>)
        })
    })
}

/// Wrap a C SSE intercept callback into a Rust closure. The SSE event is
/// serialized to JSON before being passed to the C callback.
pub fn wrap_sse_intercept_fn(
    cb: NvAgentRtSseInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> Box<dyn Fn(SseEvent) -> SseEvent + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |event: SseEvent| {
        let sse_json = serde_json::to_string(&event).unwrap_or_default();
        let c_json = CString::new(sse_json).unwrap_or_default();
        let result_ptr = unsafe { cb(ud.ptr, c_json.as_ptr()) };
        if result_ptr.is_null() {
            return event;
        }
        let result_str = ptr_to_string(result_ptr).unwrap_or_default();
        unsafe { nv_agentrt_string_free_internal(result_ptr) };
        serde_json::from_str(&result_str).unwrap_or(event)
    })
}

/// Wrap a C event subscriber callback into a Rust closure.
pub fn wrap_event_subscriber(
    cb: NvAgentRtEventSubscriberCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> Box<dyn Fn(&nvagentrt_core::Event) + Send + Sync> {
    let ud = make_user_data(user_data, free_fn);
    Box::new(move |event: &nvagentrt_core::Event| {
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

fn ptr_to_string(ptr: *mut c_char) -> Option<String> {
    ptr_to_opt_string(ptr)
}

/// Internal helper to free C strings we allocated.
unsafe fn nv_agentrt_string_free_internal(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}
