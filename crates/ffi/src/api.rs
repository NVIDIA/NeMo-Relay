// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Top-level FFI API functions exported as `extern "C"`.
//!
//! Each function clears the thread-local error before executing and returns an
//! [`NvMagicStatus`]. On failure, call [`nvmagic_last_error`] to retrieve
//! the error message.

use std::sync::OnceLock;

use libc::c_char;
use nvmagic_core as core;
use nvmagic_core::types as core_types;
use tokio::runtime::Runtime;
use tokio_stream::StreamExt;

use crate::callable::*;
use crate::convert::*;
use crate::error::*;
use crate::types::*;

// ---------------------------------------------------------------------------
// Tokio runtime singleton (for blocking on async functions)
// ---------------------------------------------------------------------------

fn tokio_runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime")
    })
}

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

/// Retrieve the current scope handle from the thread-local scope stack.
///
/// # Parameters
/// - `out`: On success, receives a heap-allocated `FfiScopeHandle` that must be
///   freed with `nvmagic_scope_handle_free`.
///
/// # Safety
/// `out` must be a valid, non-null pointer.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_get_handle(out: *mut *mut FfiScopeHandle) -> NvMagicStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NvMagicStatus::NullPointer;
    }
    match core::nvmagic_get_handle() {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiScopeHandle(h))) };
            NvMagicStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Push a new scope onto the scope stack.
///
/// # Parameters
/// - `name`: Null-terminated scope name.
/// - `scope_type`: The type of scope to create.
/// - `parent`: Optional parent scope handle, or null for auto-parenting.
/// - `attributes`: Bitfield of scope attributes.
/// - `out`: On success, receives a heap-allocated `FfiScopeHandle`.
///
/// # Safety
/// `name` must be a valid C string. `out` must be non-null. `parent` may be null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_push_scope(
    name: *const c_char,
    scope_type: NvMagicScopeType,
    parent: *const FfiScopeHandle,
    attributes: u32,
    out: *mut *mut FfiScopeHandle,
) -> NvMagicStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NvMagicStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = core_types::ScopeAttributes::from_bits_truncate(attributes);

    match core::nvmagic_push_scope(&name, scope_type.into(), parent_ref, attrs) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiScopeHandle(h))) };
            NvMagicStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Pop a scope from the scope stack by its handle.
///
/// # Parameters
/// - `handle`: The scope handle to pop.
///
/// # Safety
/// `handle` must be a valid, non-null `FfiScopeHandle` pointer.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_pop_scope(handle: *const FfiScopeHandle) -> NvMagicStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NvMagicStatus::NullPointer;
    }
    match core::nvmagic_pop_scope(&unsafe { &*handle }.0.uuid) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Emit a named lifecycle event.
///
/// # Parameters
/// - `name`: Null-terminated event name.
/// - `parent`: Optional parent scope handle, or null.
/// - `data_json`: Optional JSON data payload, or null.
/// - `metadata_json`: Optional JSON metadata payload, or null.
///
/// # Safety
/// `name` must be a valid C string. Other pointer args may be null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_event(
    name: *const c_char,
    parent: *const FfiScopeHandle,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvMagicStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvMagicStatus::InvalidJson,
    };

    match core::nvmagic_event(&name, parent_ref, data, metadata) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Tool lifecycle
// ---------------------------------------------------------------------------

/// Begin a tool call, running pre-call guardrails and intercepts.
///
/// # Parameters
/// - `name`: Null-terminated tool name.
/// - `args_json`: Tool arguments as a JSON C string.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of tool attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `tool_call_id`: Optional external correlation ID for the tool call, or null.
/// - `out`: On success, receives a heap-allocated `FfiToolHandle`.
///
/// # Safety
/// `name` and `args_json` must be valid C strings. `out` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_tool_call(
    name: *const c_char,
    args_json: *const c_char,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    tool_call_id: *const c_char,
    out: *mut *mut FfiToolHandle,
) -> NvMagicStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NvMagicStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NvMagicStatus::InvalidJson,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = core_types::ToolAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvMagicStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvMagicStatus::InvalidJson,
    };
    let tool_call_id_opt = if tool_call_id.is_null() {
        None
    } else {
        match c_str_to_string(tool_call_id) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };

    match core::nvmagic_tool_call(
        &name,
        args,
        parent_ref,
        attrs,
        data,
        metadata,
        tool_call_id_opt,
    ) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiToolHandle(h))) };
            NvMagicStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// End a tool call, running post-call guardrails and intercepts.
///
/// # Parameters
/// - `handle`: The tool handle from `nvmagic_tool_call`.
/// - `result_json`: Tool result as a JSON C string.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
///
/// # Safety
/// `handle` and `result_json` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_tool_call_end(
    handle: *const FfiToolHandle,
    result_json: *const c_char,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NvMagicStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NvMagicStatus::NullPointer;
    }
    let result = match c_str_to_json(result_json) {
        Some(r) => r,
        None => return NvMagicStatus::InvalidJson,
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvMagicStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvMagicStatus::InvalidJson,
    };

    match core::nvmagic_tool_call_end(&unsafe { &*handle }.0, result, data, metadata) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Execute a tool call end-to-end: run conditional-execution guardrails (on raw
/// args), then request intercepts, sanitize-request guardrails, execution
/// intercepts, the callback, response intercepts, and sanitize-response
/// guardrails. On rejection, only a standalone Mark event is emitted (no
/// Start/End pair) and `GuardrailRejected` is returned. Blocks the calling
/// thread until completion.
///
/// # Parameters
/// - `name`: Null-terminated tool name.
/// - `args_json`: Tool arguments as a JSON C string.
/// - `func`: C callback that performs the actual tool execution.
/// - `func_user_data`: Opaque pointer passed to `func`.
/// - `func_free`: Optional destructor for `func_user_data`.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of tool attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `out`: On success, receives the result as a JSON C string. Caller must free
///   with `nvmagic_string_free`.
///
/// # Safety
/// `name`, `args_json`, and `out` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_tool_call_execute(
    name: *const c_char,
    args_json: *const c_char,
    func: NvMagicToolExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NvMagicFreeFn,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    out: *mut *mut c_char,
) -> NvMagicStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NvMagicStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NvMagicStatus::InvalidJson,
    };
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = core_types::ToolAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvMagicStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvMagicStatus::InvalidJson,
    };

    let exec_fn = wrap_tool_exec_fn(func, func_user_data, func_free);
    let default_fn: nvmagic_core::ToolExecutionNextFn = Box::new(move |args| exec_fn(args));

    let result = tokio_runtime().block_on(async {
        core::nvmagic_tool_call_execute(
            &name,
            args,
            default_fn,
            parent_handle,
            attrs,
            data,
            metadata,
        )
        .await
    });

    match result {
        Ok(json) => {
            unsafe { *out = json_to_c_string(&json) };
            NvMagicStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// LLM lifecycle
// ---------------------------------------------------------------------------

/// Begin an LLM call, running pre-call guardrails and intercepts.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `native_json`: The native request payload as a JSON C string.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `model_name`: Optional LLM model identifier, or null.
/// - `to_request_cb`: Optional callback to convert native JSON to `LLMRequest`, or null.
/// - `to_request_ud`: Opaque pointer passed to `to_request_cb`.
/// - `to_request_free`: Optional destructor for `to_request_ud`.
/// - `out`: On success, receives a heap-allocated `FfiLLMHandle`.
///
/// # Safety
/// `name`, `native_json`, and `out` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_llm_call(
    name: *const c_char,
    native_json: *const c_char,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    to_request_cb: Option<NvMagicJsonCb>,
    to_request_ud: *mut libc::c_void,
    to_request_free: NvMagicFreeFn,
    out: *mut *mut FfiLLMHandle,
) -> NvMagicStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null pointer argument");
        return NvMagicStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let native = match c_str_to_json(native_json) {
        Some(n) => n,
        None => return NvMagicStatus::InvalidJson,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvMagicStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvMagicStatus::InvalidJson,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };
    let to_request = to_request_cb.map(|cb| wrap_to_request_fn(cb, to_request_ud, to_request_free));

    match core::nvmagic_llm_call(
        &name,
        &native,
        parent_ref,
        attrs,
        data,
        metadata,
        model_name_opt,
        to_request.as_ref(),
    ) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiLLMHandle(h))) };
            NvMagicStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// End an LLM call, running post-call guardrails and intercepts.
///
/// # Parameters
/// - `handle`: The LLM handle from `nvmagic_llm_call`.
/// - `response_json`: LLM response as a JSON C string.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `to_response_cb`: Optional callback to convert native JSON to `LLMResponse`, or null.
/// - `to_response_ud`: Opaque pointer passed to `to_response_cb`.
/// - `to_response_free`: Optional destructor for `to_response_ud`.
///
/// # Safety
/// `handle` and `response_json` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_llm_call_end(
    handle: *const FfiLLMHandle,
    response_json: *const c_char,
    data_json: *const c_char,
    metadata_json: *const c_char,
    to_response_cb: Option<NvMagicJsonCb>,
    to_response_ud: *mut libc::c_void,
    to_response_free: NvMagicFreeFn,
) -> NvMagicStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NvMagicStatus::NullPointer;
    }
    let response = match c_str_to_json(response_json) {
        Some(r) => r,
        None => return NvMagicStatus::InvalidJson,
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvMagicStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvMagicStatus::InvalidJson,
    };
    let to_response =
        to_response_cb.map(|cb| wrap_to_response_fn(cb, to_response_ud, to_response_free));

    match core::nvmagic_llm_call_end(
        &unsafe { &*handle }.0,
        response,
        data,
        metadata,
        to_response.as_ref(),
    ) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Execute an LLM call end-to-end: run conditional-execution guardrails (on raw
/// request), then request intercepts, sanitize-request guardrails, execution
/// intercepts, the callback, response intercepts, and sanitize-response
/// guardrails. On rejection, only a standalone Mark event is emitted (no
/// Start/End pair) and `GuardrailRejected` is returned. Blocks the calling
/// thread until completion.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `native_json`: The native request payload as a JSON C string.
/// - `func`: C callback that performs the actual LLM call.
/// - `func_user_data`: Opaque pointer passed to `func`.
/// - `func_free`: Optional destructor for `func_user_data`.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `model_name`: Optional LLM model identifier, or null.
/// - `to_request_cb`: Optional callback to convert native JSON to `LLMRequest`, or null.
/// - `to_request_ud`: Opaque pointer passed to `to_request_cb`.
/// - `to_request_free`: Optional destructor for `to_request_ud`.
/// - `to_response_cb`: Optional callback to convert native JSON to `LLMResponse`, or null.
/// - `to_response_ud`: Opaque pointer passed to `to_response_cb`.
/// - `to_response_free`: Optional destructor for `to_response_ud`.
/// - `out`: On success, receives the response as a JSON C string. Caller must
///   free with `nvmagic_string_free`.
///
/// # Safety
/// `name`, `native_json`, and `out` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_llm_call_execute(
    name: *const c_char,
    native_json: *const c_char,
    func: NvMagicLlmExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NvMagicFreeFn,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    to_request_cb: Option<NvMagicJsonCb>,
    to_request_ud: *mut libc::c_void,
    to_request_free: NvMagicFreeFn,
    to_response_cb: Option<NvMagicJsonCb>,
    to_response_ud: *mut libc::c_void,
    to_response_free: NvMagicFreeFn,
    out: *mut *mut c_char,
) -> NvMagicStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null pointer argument");
        return NvMagicStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let native = match c_str_to_json(native_json) {
        Some(n) => n,
        None => return NvMagicStatus::InvalidJson,
    };
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvMagicStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvMagicStatus::InvalidJson,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };
    let to_request = to_request_cb.map(|cb| wrap_to_request_fn(cb, to_request_ud, to_request_free));
    let to_response =
        to_response_cb.map(|cb| wrap_to_response_fn(cb, to_response_ud, to_response_free));

    let exec_fn = wrap_llm_exec_fn(func, func_user_data, func_free);
    let default_fn: nvmagic_core::LlmExecutionNextFn = Box::new(move |native| exec_fn(native));

    let result = tokio_runtime().block_on(async {
        core::nvmagic_llm_call_execute(
            &name,
            native,
            default_fn,
            parent_handle,
            attrs,
            data,
            metadata,
            model_name_opt,
            to_request,
            to_response,
        )
        .await
    });

    match result {
        Ok(json) => {
            unsafe { *out = json_to_c_string(&json) };
            NvMagicStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

/// Opaque stream handle for consuming LLM streaming responses chunk by chunk.
/// Use `nvmagic_stream_next` to poll and `nvmagic_stream_free` to release.
pub struct FfiStream {
    receiver:
        tokio::sync::Mutex<tokio::sync::mpsc::Receiver<nvmagic_core::Result<serde_json::Value>>>,
}

/// Execute a streaming LLM call end-to-end. Conditional-execution guardrails
/// run first on the raw request. Returns a stream handle that can be polled
/// with `nvmagic_stream_next`. Blocks until the stream is set up.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `native_json`: The native request payload as a JSON C string.
/// - `func`: C callback that performs the actual LLM call.
/// - `func_user_data`: Opaque pointer passed to `func`.
/// - `func_free`: Optional destructor for `func_user_data`.
/// - `collector`: Callback invoked with each intercepted chunk as a JSON string.
///   May be null, in which case chunks are not collected.
/// - `finalizer`: Callback invoked once when the stream is exhausted to produce
///   the aggregated response as a JSON C string. May be null, in which case the
///   finalizer returns `Json::Null`.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `model_name`: Optional LLM model identifier, or null.
/// - `to_request_cb`: Optional callback to convert native JSON to `LLMRequest`, or null.
/// - `to_request_ud`: Opaque pointer passed to `to_request_cb`.
/// - `to_request_free`: Optional destructor for `to_request_ud`.
/// - `to_response_cb`: Optional callback to convert native JSON to `LLMResponse`, or null.
/// - `to_response_ud`: Opaque pointer passed to `to_response_cb`.
/// - `to_response_free`: Optional destructor for `to_response_ud`.
/// - `out`: On success, receives a heap-allocated `FfiStream`.
///
/// # Safety
/// `name`, `native_json`, and `out` must be valid, non-null pointers. `collector`
/// and `finalizer` may be null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_llm_stream_call_execute(
    name: *const c_char,
    native_json: *const c_char,
    func: NvMagicLlmExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NvMagicFreeFn,
    collector: Option<NvMagicCollectorCb>,
    finalizer: Option<NvMagicFinalizerCb>,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    to_request_cb: Option<NvMagicJsonCb>,
    to_request_ud: *mut libc::c_void,
    to_request_free: NvMagicFreeFn,
    to_response_cb: Option<NvMagicJsonCb>,
    to_response_ud: *mut libc::c_void,
    to_response_free: NvMagicFreeFn,
    out: *mut *mut FfiStream,
) -> NvMagicStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null pointer argument");
        return NvMagicStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let native = match c_str_to_json(native_json) {
        Some(n) => n,
        None => return NvMagicStatus::InvalidJson,
    };
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvMagicStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvMagicStatus::InvalidJson,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };
    let to_request = to_request_cb.map(|cb| wrap_to_request_fn(cb, to_request_ud, to_request_free));
    let to_response =
        to_response_cb.map(|cb| wrap_to_response_fn(cb, to_response_ud, to_response_free));

    let exec_fn = wrap_llm_stream_exec_fn(func, func_user_data, func_free);
    let default_fn: nvmagic_core::LlmStreamExecutionNextFn =
        Box::new(move |native| exec_fn(native));

    let wrapped_collector: Box<dyn FnMut(serde_json::Value) + Send> = match collector {
        Some(cb) => wrap_collector_fn(cb),
        None => Box::new(|_: serde_json::Value| {}),
    };

    let wrapped_finalizer: Box<dyn FnOnce() -> serde_json::Value + Send> = match finalizer {
        Some(cb) => wrap_finalizer_fn(cb),
        None => Box::new(|| serde_json::Value::Null),
    };

    let result = tokio_runtime().block_on(async {
        core::nvmagic_llm_stream_call_execute(
            &name,
            native,
            default_fn,
            wrapped_collector,
            wrapped_finalizer,
            parent_handle,
            attrs,
            data,
            metadata,
            model_name_opt,
            to_request,
            to_response,
        )
        .await
    });

    match result {
        Ok(rust_stream) => {
            let (tx, rx) = tokio::sync::mpsc::channel(32);
            tokio_runtime().spawn(async move {
                let mut stream = rust_stream;
                while let Some(item) = stream.next().await {
                    if tx.send(item).await.is_err() {
                        break;
                    }
                }
            });
            let ffi_stream = Box::new(FfiStream {
                receiver: tokio::sync::Mutex::new(rx),
            });
            unsafe { *out = Box::into_raw(ffi_stream) };
            NvMagicStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Poll the next chunk from a streaming LLM response. Blocks until a chunk is
/// available.
///
/// # Returns
/// - `1`: A chunk was written to `*out_chunk`. Caller must free with
///   `nvmagic_string_free`.
/// - `0`: The stream is complete (no more chunks).
/// - `-1`: An error occurred. Call `nvmagic_last_error` for details.
///
/// # Safety
/// `stream` and `out_chunk` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_stream_next(
    stream: *mut FfiStream,
    out_chunk: *mut *mut c_char,
) -> i32 {
    if stream.is_null() || out_chunk.is_null() {
        return -1;
    }
    let stream = unsafe { &*stream };
    let result = tokio_runtime().block_on(async {
        let mut guard = stream.receiver.lock().await;
        guard.recv().await
    });
    match result {
        None => 0, // stream done
        Some(Ok(chunk)) => {
            unsafe { *out_chunk = json_to_c_string(&chunk) };
            1
        }
        Some(Err(e)) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Free a stream handle and release its resources.
///
/// # Safety
/// `stream` must be a valid `FfiStream` pointer returned by
/// `nvmagic_llm_stream_call_execute`, or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_stream_free(stream: *mut FfiStream) {
    if !stream.is_null() {
        drop(unsafe { Box::from_raw(stream) });
    }
}

// ---------------------------------------------------------------------------
// Tool guardrail registrations
// ---------------------------------------------------------------------------

macro_rules! ffi_guardrail_tool_api {
    ($(#[$reg_doc:meta])* $register_name:ident,
     $(#[$dereg_doc:meta])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:ident) => {
        $(#[$reg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $register_name(
            name: *const c_char,
            priority: i32,
            cb: NvMagicToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NvMagicFreeFn,
        ) -> NvMagicStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            let wrapped = $wrapper(cb, user_data, free_fn);
            match $core_register(&name, priority, wrapped) {
                Ok(()) => NvMagicStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $deregister_name(
            name: *const c_char,
        ) -> NvMagicStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            match $core_deregister(&name) {
                Ok(_) => NvMagicStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }
    };
}

ffi_guardrail_tool_api!(
    /// Register a tool request sanitization guardrail. The callback can inspect
    /// and modify tool arguments before the tool executes.
    ///
    /// # Parameters
    /// - `name`: Unique guardrail name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `cb`: Sanitize callback that receives tool name and args JSON, returns sanitized args JSON.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `name` must be a valid C string. `cb` must be a valid function pointer.
    nvmagic_register_tool_sanitize_request_guardrail,
    /// Deregister a tool request sanitization guardrail by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nvmagic_deregister_tool_sanitize_request_guardrail,
    core::nvmagic_register_tool_sanitize_request_guardrail,
    core::nvmagic_deregister_tool_sanitize_request_guardrail,
    wrap_tool_sanitize_fn
);

ffi_guardrail_tool_api!(
    /// Register a tool response sanitization guardrail. The callback can inspect
    /// and modify tool results after the tool executes.
    ///
    /// # Parameters
    /// - `name`: Unique guardrail name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `cb`: Sanitize callback that receives tool name and result JSON, returns sanitized result JSON.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `name` must be a valid C string. `cb` must be a valid function pointer.
    nvmagic_register_tool_sanitize_response_guardrail,
    /// Deregister a tool response sanitization guardrail by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nvmagic_deregister_tool_sanitize_response_guardrail,
    core::nvmagic_register_tool_sanitize_response_guardrail,
    core::nvmagic_deregister_tool_sanitize_response_guardrail,
    wrap_tool_sanitize_fn
);

/// Register a tool conditional execution guardrail. The callback decides whether
/// a tool call should proceed. Returns an error message to reject, or null to allow.
///
/// # Parameters
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: Conditional callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_register_tool_conditional_execution_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NvMagicToolConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_conditional_fn(cb, user_data, free_fn);
    match core::nvmagic_register_tool_conditional_execution_guardrail(&name, priority, wrapped) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a tool conditional execution guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_deregister_tool_conditional_execution_guardrail(
    name: *const c_char,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nvmagic_deregister_tool_conditional_execution_guardrail(&name) {
        Ok(_) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Tool intercept registrations
// ---------------------------------------------------------------------------

macro_rules! ffi_intercept_tool_api {
    ($(#[$reg_doc:meta])* $register_name:ident,
     $(#[$dereg_doc:meta])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:ident) => {
        $(#[$reg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $register_name(
            name: *const c_char,
            priority: i32,
            break_chain: bool,
            cb: NvMagicToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NvMagicFreeFn,
        ) -> NvMagicStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            let wrapped = $wrapper(cb, user_data, free_fn);
            match $core_register(&name, priority, break_chain, wrapped) {
                Ok(()) => NvMagicStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $deregister_name(
            name: *const c_char,
        ) -> NvMagicStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            match $core_deregister(&name) {
                Ok(_) => NvMagicStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }
    };
}

ffi_intercept_tool_api!(
    /// Register a tool request intercept. The callback can transform tool
    /// arguments before execution. Runs after request guardrails in the
    /// middleware pipeline.
    ///
    /// # Parameters
    /// - `name`: Unique intercept name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `break_chain`: If true, stop processing further intercepts after this one.
    /// - `cb`: Transform callback that receives tool name and args JSON, returns modified args JSON.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `name` must be a valid C string. `cb` must be a valid function pointer.
    nvmagic_register_tool_request_intercept,
    /// Deregister a tool request intercept by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nvmagic_deregister_tool_request_intercept,
    core::nvmagic_register_tool_request_intercept,
    core::nvmagic_deregister_tool_request_intercept,
    wrap_tool_sanitize_fn
);

ffi_intercept_tool_api!(
    /// Register a tool response intercept. The callback can transform tool
    /// results after execution. Runs before response guardrails in the
    /// middleware pipeline.
    ///
    /// # Parameters
    /// - `name`: Unique intercept name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `break_chain`: If true, stop processing further intercepts after this one.
    /// - `cb`: Transform callback that receives tool name and result JSON, returns modified result JSON.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `name` must be a valid C string. `cb` must be a valid function pointer.
    nvmagic_register_tool_response_intercept,
    /// Deregister a tool response intercept by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nvmagic_deregister_tool_response_intercept,
    core::nvmagic_register_tool_response_intercept,
    core::nvmagic_deregister_tool_response_intercept,
    wrap_tool_sanitize_fn
);

/// Register a tool execution intercept following the middleware chain pattern.
/// When the condition callback returns true, the execution callback is included
/// in the chain. The callback receives `(args, next_fn, next_ctx)` — call
/// `next_fn(args, next_ctx)` to invoke the next intercept or the original
/// tool function, or skip calling it to short-circuit.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `cond_cb`: Condition callback that decides if this intercept applies.
/// - `cond_user_data`: Opaque pointer for the condition callback.
/// - `cond_free`: Optional destructor for `cond_user_data`.
/// - `exec_cb`: Middleware callback receiving args and a next function.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `name` must be a valid C string. Callback pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_register_tool_execution_intercept(
    name: *const c_char,
    priority: i32,
    cond_cb: NvMagicToolExecConditionalCb,
    cond_user_data: *mut libc::c_void,
    cond_free: NvMagicFreeFn,
    exec_cb: NvMagicToolExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NvMagicFreeFn,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let cond = wrap_tool_exec_conditional_fn(cond_cb, cond_user_data, cond_free);
    let exec = wrap_tool_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core::nvmagic_register_tool_execution_intercept(&name, priority, cond, exec) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a tool execution intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_deregister_tool_execution_intercept(
    name: *const c_char,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nvmagic_deregister_tool_execution_intercept(&name) {
        Ok(_) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// LLM guardrail registrations
// ---------------------------------------------------------------------------

/// Register an LLM request sanitization guardrail. The callback can modify or
/// replace the LLM request before it is sent.
///
/// # Parameters
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: Request sanitize callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_register_llm_sanitize_request_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NvMagicLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_sanitize_request_fn(cb, user_data, free_fn);
    match core::nvmagic_register_llm_sanitize_request_guardrail(&name, priority, wrapped) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM request sanitization guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_deregister_llm_sanitize_request_guardrail(
    name: *const c_char,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nvmagic_deregister_llm_sanitize_request_guardrail(&name) {
        Ok(_) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM response sanitization guardrail. The callback can inspect
/// and modify the LLM response after it is received.
///
/// # Parameters
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: JSON-to-JSON callback that receives LLMResponse JSON and returns sanitized JSON.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_register_llm_sanitize_response_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NvMagicJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_response_fn(cb, user_data, free_fn);
    match core::nvmagic_register_llm_sanitize_response_guardrail(&name, priority, wrapped) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM response sanitization guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_deregister_llm_sanitize_response_guardrail(
    name: *const c_char,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nvmagic_deregister_llm_sanitize_response_guardrail(&name) {
        Ok(_) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM conditional execution guardrail. The callback decides
/// whether an LLM call should proceed.
///
/// # Parameters
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: Conditional callback. Returns null to allow, or error message to reject.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_register_llm_conditional_execution_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NvMagicLlmConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_conditional_fn(cb, user_data, free_fn);
    match core::nvmagic_register_llm_conditional_execution_guardrail(&name, priority, wrapped) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM conditional execution guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_deregister_llm_conditional_execution_guardrail(
    name: *const c_char,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nvmagic_deregister_llm_conditional_execution_guardrail(&name) {
        Ok(_) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// LLM intercept registrations
// ---------------------------------------------------------------------------

/// Register an LLM request intercept. The callback can transform the native
/// request JSON before it reaches the LLM provider.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `break_chain`: If true, stop processing further intercepts after this one.
/// - `cb`: JSON transform callback (receives/returns native JSON C string).
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_register_llm_request_intercept(
    name: *const c_char,
    priority: i32,
    break_chain: bool,
    cb: NvMagicJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_json_fn(cb, user_data, free_fn);
    match core::nvmagic_register_llm_request_intercept(&name, priority, break_chain, wrapped) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM request intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_deregister_llm_request_intercept(
    name: *const c_char,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nvmagic_deregister_llm_request_intercept(&name) {
        Ok(_) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM response intercept. The callback can transform the
/// LLM response after it is received from the LLM provider.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `break_chain`: If true, stop processing further intercepts after this one.
/// - `cb`: JSON transform callback (receives/returns LLMResponse as JSON C string).
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_register_llm_response_intercept(
    name: *const c_char,
    priority: i32,
    break_chain: bool,
    cb: NvMagicJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_response_fn(cb, user_data, free_fn);
    match core::nvmagic_register_llm_response_intercept(&name, priority, break_chain, wrapped) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM response intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_deregister_llm_response_intercept(
    name: *const c_char,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nvmagic_deregister_llm_response_intercept(&name) {
        Ok(_) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM streaming response intercept. The callback transforms
/// individual chunk strings as they arrive during a streaming LLM call.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `break_chain`: If true, stop processing further intercepts after this one.
/// - `cb`: Chunk string transform callback (receives/returns C string).
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_register_llm_stream_response_intercept(
    name: *const c_char,
    priority: i32,
    break_chain: bool,
    cb: NvMagicSseInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_string_intercept_fn(cb, user_data, free_fn);
    match core::nvmagic_register_llm_stream_response_intercept(
        &name,
        priority,
        break_chain,
        wrapped,
    ) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM streaming response intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_deregister_llm_stream_response_intercept(
    name: *const c_char,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nvmagic_deregister_llm_stream_response_intercept(&name) {
        Ok(_) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM execution intercept following the middleware chain pattern.
/// When the condition callback returns true, the execution callback is included
/// in the chain. The callback receives `(request, next_fn, next_ctx)` — call
/// `next_fn(request, next_ctx)` to invoke the next intercept or the original
/// LLM call, or skip calling it to short-circuit.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `cond_cb`: Condition callback.
/// - `cond_user_data`: Opaque pointer for the condition callback.
/// - `cond_free`: Optional destructor for `cond_user_data`.
/// - `exec_cb`: Middleware callback receiving request and a next function.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `name` must be a valid C string. Callback pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_register_llm_execution_intercept(
    name: *const c_char,
    priority: i32,
    cond_cb: NvMagicLlmExecConditionalCb,
    cond_user_data: *mut libc::c_void,
    cond_free: NvMagicFreeFn,
    exec_cb: NvMagicLlmExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NvMagicFreeFn,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let cond = wrap_llm_exec_conditional_fn(cond_cb, cond_user_data, cond_free);
    let exec = wrap_llm_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core::nvmagic_register_llm_execution_intercept(&name, priority, cond, exec) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM execution intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_deregister_llm_execution_intercept(
    name: *const c_char,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nvmagic_deregister_llm_execution_intercept(&name) {
        Ok(_) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM streaming execution intercept following the middleware chain
/// pattern. When the condition callback returns true, the execution callback is
/// included in the chain. The callback receives `(request, next_fn, next_ctx)` —
/// call `next_fn(request, next_ctx)` to invoke the next intercept or the original
/// streaming LLM call, or skip calling it to short-circuit.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `cond_cb`: Condition callback.
/// - `cond_user_data`: Opaque pointer for the condition callback.
/// - `cond_free`: Optional destructor for `cond_user_data`.
/// - `exec_cb`: Middleware callback receiving request and a next function.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `name` must be a valid C string. Callback pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_register_llm_stream_execution_intercept(
    name: *const c_char,
    priority: i32,
    cond_cb: NvMagicLlmExecConditionalCb,
    cond_user_data: *mut libc::c_void,
    cond_free: NvMagicFreeFn,
    exec_cb: NvMagicLlmExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NvMagicFreeFn,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let cond = wrap_llm_exec_conditional_fn(cond_cb, cond_user_data, cond_free);
    let exec = wrap_llm_stream_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core::nvmagic_register_llm_stream_execution_intercept(&name, priority, cond, exec) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM streaming execution intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_deregister_llm_stream_execution_intercept(
    name: *const c_char,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nvmagic_deregister_llm_stream_execution_intercept(&name) {
        Ok(_) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Subscriber registrations
// ---------------------------------------------------------------------------

/// Register an event subscriber. The callback is invoked for every lifecycle
/// event emitted by the runtime.
///
/// # Parameters
/// - `name`: Unique subscriber name.
/// - `cb`: Event callback. The `FfiEvent` is valid only during the call.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_register_subscriber(
    name: *const c_char,
    cb: NvMagicEventSubscriberCb,
    user_data: *mut libc::c_void,
    free_fn: NvMagicFreeFn,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_event_subscriber(cb, user_data, free_fn);
    match core::nvmagic_register_subscriber(&name, wrapped) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an event subscriber by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_deregister_subscriber(name: *const c_char) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nvmagic_deregister_subscriber(&name) {
        Ok(_) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Scope stack isolation
// ---------------------------------------------------------------------------

/// Create a new isolated scope stack with its own root scope.
///
/// Each scope stack is independent: scopes pushed on one do not appear on another.
/// Use `nvmagic_scope_stack_set_thread` to bind a stack to the current thread
/// before making other NVMagic API calls.
///
/// # Parameters
/// - `out`: On success, receives a heap-allocated `FfiScopeStack` that must be
///   freed with `nvmagic_scope_stack_free`.
///
/// # Safety
/// `out` must be a valid, non-null pointer.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_scope_stack_create(out: *mut *mut FfiScopeStack) -> NvMagicStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NvMagicStatus::NullPointer;
    }
    let handle = core::create_scope_stack();
    unsafe { *out = Box::into_raw(Box::new(FfiScopeStack(handle))) };
    NvMagicStatus::Ok
}

/// Bind an isolated scope stack to the current OS thread.
///
/// After this call, all NVMagic scope operations on the current thread
/// (e.g. `nvmagic_push_scope`, `nvmagic_get_handle`) will use the
/// given scope stack. This is typically used from Go goroutines that have
/// called `runtime.LockOSThread()`.
///
/// The `FfiScopeStack` is **not** consumed — the caller retains ownership
/// and must still free it when done.
///
/// # Safety
/// `stack` must be a valid, non-null `FfiScopeStack` pointer.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_scope_stack_set_thread(
    stack: *const FfiScopeStack,
) -> NvMagicStatus {
    clear_last_error();
    if stack.is_null() {
        set_last_error("stack pointer is null");
        return NvMagicStatus::NullPointer;
    }
    let handle = unsafe { &*stack }.0.clone();
    core::set_thread_scope_stack(handle);
    NvMagicStatus::Ok
}

// ---------------------------------------------------------------------------
// ATIF exporter
// ---------------------------------------------------------------------------

/// Creates a new ATIF exporter.
///
/// # Parameters
/// - `session_id`: Session identifier string (required, non-null).
/// - `agent_name`: Agent name string (required, non-null).
/// - `agent_version`: Agent version string (required, non-null).
/// - `model_name`: Default model name (nullable).
/// - `out`: On success, receives a heap-allocated `FfiAtifExporter`.
///
/// # Safety
/// All non-null string pointers must be valid C strings. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_atif_exporter_create(
    session_id: *const c_char,
    agent_name: *const c_char,
    agent_version: *const c_char,
    model_name: *const c_char,
    out: *mut *mut FfiAtifExporter,
) -> NvMagicStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NvMagicStatus::NullPointer;
    }
    let session_id = match c_str_to_string(session_id) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let agent_name = match c_str_to_string(agent_name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let agent_version = match c_str_to_string(agent_version) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };

    let agent_info = nvmagic_core::atif::AtifAgentInfo {
        name: agent_name,
        version: agent_version,
        model_name: model_name_opt,
        tool_definitions: None,
        extra: None,
    };

    let exporter = nvmagic_core::atif::AtifExporter::new(session_id, agent_info);
    unsafe { *out = Box::into_raw(Box::new(FfiAtifExporter(exporter))) };
    NvMagicStatus::Ok
}

/// Registers the exporter as an event subscriber.
///
/// # Parameters
/// - `exporter`: The exporter handle.
/// - `name`: Subscriber name (required, non-null).
///
/// # Safety
/// `exporter` and `name` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_atif_exporter_register(
    exporter: *const FfiAtifExporter,
    name: *const c_char,
) -> NvMagicStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NvMagicStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let subscriber = unsafe { &*exporter }.0.subscriber();
    match core::nvmagic_register_subscriber(&name, subscriber) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregisters the exporter subscriber.
///
/// # Parameters
/// - `name`: Subscriber name (required, non-null).
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_atif_exporter_deregister(name: *const c_char) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nvmagic_deregister_subscriber(&name) {
        Ok(_) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Exports collected events as an ATIF trajectory JSON string.
///
/// # Parameters
/// - `exporter`: The exporter handle.
/// - `root_uuid`: Optional root UUID filter (nullable C string).
/// - `out`: On success, receives a JSON string (caller must free with
///   `nvmagic_string_free`).
///
/// # Safety
/// `exporter` and `out` must be valid, non-null pointers. `root_uuid` may be
/// null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_atif_exporter_export(
    exporter: *const FfiAtifExporter,
    root_uuid: *const c_char,
    out: *mut *mut c_char,
) -> NvMagicStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NvMagicStatus::NullPointer;
    }
    if out.is_null() {
        set_last_error("out pointer is null");
        return NvMagicStatus::NullPointer;
    }
    let root_uuid_opt = if root_uuid.is_null() {
        None
    } else {
        let uuid_str = match c_str_to_string(root_uuid) {
            Ok(s) => s,
            Err(status) => return status,
        };
        match uuid::Uuid::parse_str(&uuid_str) {
            Ok(u) => Some(u),
            Err(e) => {
                set_last_error(&format!("invalid UUID: {e}"));
                return NvMagicStatus::Internal;
            }
        }
    };

    let trajectory = unsafe { &*exporter }.0.export(root_uuid_opt);
    match serde_json::to_string(&trajectory) {
        Ok(json_str) => {
            unsafe { *out = str_to_c_string(&json_str) };
            NvMagicStatus::Ok
        }
        Err(e) => {
            set_last_error(&format!("failed to serialize trajectory: {e}"));
            NvMagicStatus::Internal
        }
    }
}

/// Clears all collected events from the exporter.
///
/// # Parameters
/// - `exporter`: The exporter handle.
///
/// # Safety
/// `exporter` must be a valid, non-null `FfiAtifExporter` pointer.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_atif_exporter_clear(
    exporter: *const FfiAtifExporter,
) -> NvMagicStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NvMagicStatus::NullPointer;
    }
    unsafe { &*exporter }.0.clear();
    NvMagicStatus::Ok
}

// ---------------------------------------------------------------------------
// Standalone middleware chains
// ---------------------------------------------------------------------------

/// Run the registered tool request intercept chain on the given arguments.
///
/// # Parameters
/// - `name`: Tool name (null-terminated C string).
/// - `args_json`: Tool arguments as a JSON C string.
/// - `out`: On success, receives the transformed JSON string (caller must free
///   with `nvmagic_string_free`).
///
/// # Safety
/// All pointers must be valid. `out` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_tool_request_intercepts(
    name: *const c_char,
    args_json: *const c_char,
    out: *mut *mut c_char,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NvMagicStatus::InvalidJson,
    };
    match core::nvmagic_tool_request_intercepts(&name, args) {
        Ok(result) => {
            unsafe { *out = json_to_c_string(&result) };
            NvMagicStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Run the registered tool conditional execution guardrail chain.
///
/// Returns `NvMagicStatus::Ok` if all guardrails pass, or
/// `NvMagicStatus::GuardrailRejected` if blocked.
///
/// # Parameters
/// - `name`: Tool name (null-terminated C string).
/// - `args_json`: Tool arguments as a JSON C string.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_tool_conditional_execution(
    name: *const c_char,
    args_json: *const c_char,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NvMagicStatus::InvalidJson,
    };
    match core::nvmagic_tool_conditional_execution(&name, &args) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Run the registered tool response intercept chain on the given result.
///
/// # Parameters
/// - `name`: Tool name (null-terminated C string).
/// - `result_json`: Tool result as a JSON C string.
/// - `out`: On success, receives the transformed JSON string (caller must free
///   with `nvmagic_string_free`).
///
/// # Safety
/// All pointers must be valid. `out` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_tool_response_intercepts(
    name: *const c_char,
    result_json: *const c_char,
    out: *mut *mut c_char,
) -> NvMagicStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let result = match c_str_to_json(result_json) {
        Some(r) => r,
        None => return NvMagicStatus::InvalidJson,
    };
    match core::nvmagic_tool_response_intercepts(&name, result) {
        Ok(transformed) => {
            unsafe { *out = json_to_c_string(&transformed) };
            NvMagicStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Run the registered LLM request intercept chain on the given native request.
///
/// # Parameters
/// - `native_json`: Native LLM request as a JSON C string.
/// - `out`: On success, receives the transformed JSON string (caller must free
///   with `nvmagic_string_free`).
///
/// # Safety
/// All pointers must be valid. `out` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_llm_request_intercepts(
    native_json: *const c_char,
    out: *mut *mut c_char,
) -> NvMagicStatus {
    clear_last_error();
    let native = match c_str_to_json(native_json) {
        Some(j) => j,
        None => return NvMagicStatus::InvalidJson,
    };
    match core::nvmagic_llm_request_intercepts(native) {
        Ok(transformed) => {
            unsafe { *out = json_to_c_string(&transformed) };
            NvMagicStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Run the registered LLM conditional execution guardrail chain.
///
/// Returns `NvMagicStatus::Ok` if all guardrails pass, or
/// `NvMagicStatus::GuardrailRejected` if blocked.
///
/// # Parameters
/// - `native_json`: Native LLM request as a JSON C string.
/// - `to_request_cb`: Optional callback to convert native JSON to `LLMRequest`, or null.
/// - `to_request_ud`: Opaque pointer passed to `to_request_cb`.
/// - `to_request_free`: Optional destructor for `to_request_ud`.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_llm_conditional_execution(
    native_json: *const c_char,
    to_request_cb: Option<NvMagicJsonCb>,
    to_request_ud: *mut libc::c_void,
    to_request_free: NvMagicFreeFn,
) -> NvMagicStatus {
    clear_last_error();
    let native = match c_str_to_json(native_json) {
        Some(j) => j,
        None => return NvMagicStatus::InvalidJson,
    };
    let to_request = to_request_cb.map(|cb| wrap_to_request_fn(cb, to_request_ud, to_request_free));
    match core::nvmagic_llm_conditional_execution(&native, to_request.as_ref()) {
        Ok(()) => NvMagicStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Run the registered LLM response intercept chain on the given response.
///
/// # Parameters
/// - `response_json`: LLM response as a JSON C string (serialized LLMResponse).
/// - `out`: On success, receives the transformed JSON string (caller must free
///   with `nvmagic_string_free`).
///
/// # Safety
/// All pointers must be valid. `out` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_llm_response_intercepts(
    response_json: *const c_char,
    out: *mut *mut c_char,
) -> NvMagicStatus {
    clear_last_error();
    let response_str = match c_str_to_string(response_json) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let response: core_types::LLMResponse = match serde_json::from_str(&response_str) {
        Ok(r) => r,
        Err(_) => return NvMagicStatus::InvalidJson,
    };
    match core::nvmagic_llm_response_intercepts(response) {
        Ok(transformed) => match serde_json::to_string(&transformed) {
            Ok(json_str) => {
                unsafe { *out = str_to_c_string(&json_str) };
                NvMagicStatus::Ok
            }
            Err(e) => {
                set_last_error(&format!("failed to serialize LLMResponse: {e}"));
                NvMagicStatus::Internal
            }
        },
        Err(e) => status_from_error(&e),
    }
}
