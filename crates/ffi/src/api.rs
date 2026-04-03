// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Top-level FFI API functions exported as `extern "C"`.
//!
//! Each function clears the thread-local error before executing and returns an
//! [`NatNexusStatus`]. On failure, call [`nat_nexus_last_error`] to retrieve
//! the error message.

use std::sync::{Arc, OnceLock};

use libc::c_char;
use nvidia_nat_nexus_core as core;
use nvidia_nat_nexus_core::types as core_types;
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
///   freed with `nat_nexus_scope_handle_free`.
///
/// # Safety
/// `out` must be a valid, non-null pointer.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_get_handle(out: *mut *mut FfiScopeHandle) -> NatNexusStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NatNexusStatus::NullPointer;
    }
    match core::nat_nexus_get_handle() {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiScopeHandle(h))) };
            NatNexusStatus::Ok
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
/// - `data_json`: Optional null-terminated JSON string for scope data, or null.
/// - `metadata_json`: Optional null-terminated JSON string for scope metadata, or null.
/// - `out`: On success, receives a heap-allocated `FfiScopeHandle`.
///
/// # Safety
/// `name` must be a valid C string. `out` must be non-null. `parent`,
/// `data_json`, and `metadata_json` may be null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_push_scope(
    name: *const c_char,
    scope_type: NatNexusScopeType,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    out: *mut *mut FfiScopeHandle,
) -> NatNexusStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NatNexusStatus::NullPointer;
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
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NatNexusStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NatNexusStatus::InvalidJson,
    };

    match core::nat_nexus_push_scope(&name, scope_type.into(), parent_ref, attrs, data, metadata) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiScopeHandle(h))) };
            NatNexusStatus::Ok
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
pub unsafe extern "C" fn nat_nexus_pop_scope(handle: *const FfiScopeHandle) -> NatNexusStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NatNexusStatus::NullPointer;
    }
    match core::nat_nexus_pop_scope(&unsafe { &*handle }.0.uuid) {
        Ok(()) => NatNexusStatus::Ok,
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
pub unsafe extern "C" fn nat_nexus_event(
    name: *const c_char,
    parent: *const FfiScopeHandle,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NatNexusStatus {
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
        None => return NatNexusStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NatNexusStatus::InvalidJson,
    };

    match core::nat_nexus_event(&name, parent_ref, data, metadata) {
        Ok(()) => NatNexusStatus::Ok,
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
pub unsafe extern "C" fn nat_nexus_tool_call(
    name: *const c_char,
    args_json: *const c_char,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    tool_call_id: *const c_char,
    out: *mut *mut FfiToolHandle,
) -> NatNexusStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NatNexusStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NatNexusStatus::InvalidJson,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = core_types::ToolAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NatNexusStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NatNexusStatus::InvalidJson,
    };
    let tool_call_id_opt = if tool_call_id.is_null() {
        None
    } else {
        match c_str_to_string(tool_call_id) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };

    match core::nat_nexus_tool_call(
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
            NatNexusStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// End a tool call, running post-call guardrails and intercepts.
///
/// # Parameters
/// - `handle`: The tool handle from `nat_nexus_tool_call`.
/// - `result_json`: Tool result as a JSON C string.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
///
/// # Safety
/// `handle` and `result_json` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_tool_call_end(
    handle: *const FfiToolHandle,
    result_json: *const c_char,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NatNexusStatus::NullPointer;
    }
    let result = match c_str_to_json(result_json) {
        Some(r) => r,
        None => return NatNexusStatus::InvalidJson,
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NatNexusStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NatNexusStatus::InvalidJson,
    };

    match core::nat_nexus_tool_call_end(&unsafe { &*handle }.0, result, data, metadata) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Execute a tool call end-to-end: run conditional-execution guardrails (on raw
/// args), then request intercepts, sanitize-request guardrails, execution
/// intercepts, the callback, and sanitize-response
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
///   with `nat_nexus_string_free`.
///
/// # Safety
/// `name`, `args_json`, and `out` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_tool_call_execute(
    name: *const c_char,
    args_json: *const c_char,
    func: NatNexusToolExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NatNexusFreeFn,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    out: *mut *mut c_char,
) -> NatNexusStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NatNexusStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NatNexusStatus::InvalidJson,
    };
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = core_types::ToolAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NatNexusStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NatNexusStatus::InvalidJson,
    };

    let exec_fn = wrap_tool_exec_fn(func, func_user_data, func_free);
    let default_fn: nvidia_nat_nexus_core::ToolExecutionNextFn =
        Arc::new(move |args| exec_fn(args));

    let scope_stack = core::current_scope_stack();
    let result = tokio_runtime().block_on(nvidia_nat_nexus_core::TASK_SCOPE_STACK.scope(
        scope_stack,
        async {
            core::nat_nexus_tool_call_execute(
                &name,
                args,
                default_fn,
                parent_handle,
                attrs,
                data,
                metadata,
            )
            .await
        },
    ));

    match result {
        Ok(json) => {
            unsafe { *out = json_to_c_string(&json) };
            NatNexusStatus::Ok
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
/// - `native_json`: The request payload as a JSON C string representing an
///   `LLMRequest` (`{"headers": {...}, "content": {...}}`).
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `model_name`: Optional LLM model identifier, or null.
/// - `out`: On success, receives a heap-allocated `FfiLLMHandle`.
///
/// # Safety
/// `name`, `native_json`, and `out` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_llm_call(
    name: *const c_char,
    native_json: *const c_char,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    out: *mut *mut FfiLLMHandle,
) -> NatNexusStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null pointer argument");
        return NatNexusStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let native = match c_str_to_json(native_json) {
        Some(n) => n,
        None => return NatNexusStatus::InvalidJson,
    };
    let request: core_types::LLMRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LLMRequest");
            return NatNexusStatus::InvalidJson;
        }
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NatNexusStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NatNexusStatus::InvalidJson,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };

    match core::nat_nexus_llm_call(
        &name,
        &request,
        parent_ref,
        attrs,
        data,
        metadata,
        model_name_opt,
    ) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiLLMHandle(h))) };
            NatNexusStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// End an LLM call, running post-call guardrails and intercepts.
///
/// # Parameters
/// - `handle`: The LLM handle from `nat_nexus_llm_call`.
/// - `response_json`: LLM response as a JSON C string.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
///
/// # Safety
/// `handle` and `response_json` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_llm_call_end(
    handle: *const FfiLLMHandle,
    response_json: *const c_char,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NatNexusStatus::NullPointer;
    }
    let response = match c_str_to_json(response_json) {
        Some(r) => r,
        None => return NatNexusStatus::InvalidJson,
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NatNexusStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NatNexusStatus::InvalidJson,
    };

    match core::nat_nexus_llm_call_end(&unsafe { &*handle }.0, response, data, metadata) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Execute an LLM call end-to-end: run conditional-execution guardrails (on raw
/// request), then request intercepts, sanitize-request guardrails, execution
/// intercepts, the callback, and sanitize-response
/// guardrails. On rejection, only a standalone Mark event is emitted (no
/// Start/End pair) and `GuardrailRejected` is returned. Blocks the calling
/// thread until completion.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `native_json`: The request payload as a JSON C string representing an
///   `LLMRequest` (`{"headers": {...}, "content": {...}}`).
/// - `func`: C callback that performs the actual LLM call.
/// - `func_user_data`: Opaque pointer passed to `func`.
/// - `func_free`: Optional destructor for `func_user_data`.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `model_name`: Optional LLM model identifier, or null.
/// - `out`: On success, receives the response as a JSON C string. Caller must
///   free with `nat_nexus_string_free`.
///
/// # Safety
/// `name`, `native_json`, and `out` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_llm_call_execute(
    name: *const c_char,
    native_json: *const c_char,
    func: NatNexusLlmExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NatNexusFreeFn,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    out: *mut *mut c_char,
) -> NatNexusStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null pointer argument");
        return NatNexusStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let native = match c_str_to_json(native_json) {
        Some(n) => n,
        None => return NatNexusStatus::InvalidJson,
    };
    let request: core_types::LLMRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LLMRequest");
            return NatNexusStatus::InvalidJson;
        }
    };
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NatNexusStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NatNexusStatus::InvalidJson,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };

    let exec_fn = wrap_llm_exec_fn(func, func_user_data, func_free);
    let default_fn: nvidia_nat_nexus_core::LlmExecutionNextFn =
        Arc::new(move |request| exec_fn(request));

    let scope_stack = core::current_scope_stack();
    let result = tokio_runtime().block_on(nvidia_nat_nexus_core::TASK_SCOPE_STACK.scope(
        scope_stack,
        async {
            core::nat_nexus_llm_call_execute(
                &name,
                request,
                default_fn,
                parent_handle,
                attrs,
                data,
                metadata,
                model_name_opt,
            )
            .await
        },
    ));

    match result {
        Ok(json) => {
            unsafe { *out = json_to_c_string(&json) };
            NatNexusStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

/// Opaque stream handle for consuming LLM streaming responses chunk by chunk.
/// Use `nat_nexus_stream_next` to poll and `nat_nexus_stream_free` to release.
pub struct FfiStream {
    receiver: tokio::sync::Mutex<
        tokio::sync::mpsc::Receiver<nvidia_nat_nexus_core::Result<serde_json::Value>>,
    >,
}

/// Execute a streaming LLM call end-to-end. Conditional-execution guardrails
/// run first on the raw request. Returns a stream handle that can be polled
/// with `nat_nexus_stream_next`. Blocks until the stream is set up.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `native_json`: The request payload as a JSON C string representing an
///   `LLMRequest` (`{"headers": {...}, "content": {...}}`).
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
/// - `out`: On success, receives a heap-allocated `FfiStream`.
///
/// # Safety
/// `name`, `native_json`, and `out` must be valid, non-null pointers. `collector`
/// and `finalizer` may be null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_llm_stream_call_execute(
    name: *const c_char,
    native_json: *const c_char,
    func: NatNexusLlmExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NatNexusFreeFn,
    collector: Option<NatNexusCollectorCb>,
    finalizer: Option<NatNexusFinalizerCb>,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    out: *mut *mut FfiStream,
) -> NatNexusStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null pointer argument");
        return NatNexusStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let native = match c_str_to_json(native_json) {
        Some(n) => n,
        None => return NatNexusStatus::InvalidJson,
    };
    let request: core_types::LLMRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LLMRequest");
            return NatNexusStatus::InvalidJson;
        }
    };
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NatNexusStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NatNexusStatus::InvalidJson,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };

    let exec_fn = wrap_llm_stream_exec_fn(func, func_user_data, func_free);
    let default_fn: nvidia_nat_nexus_core::LlmStreamExecutionNextFn =
        Arc::new(move |request| exec_fn(request));

    let wrapped_collector: Box<
        dyn FnMut(serde_json::Value) -> nvidia_nat_nexus_core::Result<()> + Send,
    > = match collector {
        Some(cb) => wrap_collector_fn(cb),
        None => Box::new(|_: serde_json::Value| Ok(())),
    };

    let wrapped_finalizer: Box<dyn FnOnce() -> serde_json::Value + Send> = match finalizer {
        Some(cb) => wrap_finalizer_fn(cb),
        None => Box::new(|| serde_json::Value::Null),
    };

    let scope_stack = core::current_scope_stack();
    let result = tokio_runtime().block_on(nvidia_nat_nexus_core::TASK_SCOPE_STACK.scope(
        scope_stack,
        async {
            core::nat_nexus_llm_stream_call_execute(
                &name,
                request,
                default_fn,
                wrapped_collector,
                wrapped_finalizer,
                parent_handle,
                attrs,
                data,
                metadata,
                model_name_opt,
            )
            .await
        },
    ));

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
            NatNexusStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Poll the next chunk from a streaming LLM response. Blocks until a chunk is
/// available.
///
/// # Returns
/// - `1`: A chunk was written to `*out_chunk`. Caller must free with
///   `nat_nexus_string_free`.
/// - `0`: The stream is complete (no more chunks).
/// - `-1`: An error occurred. Call `nat_nexus_last_error` for details.
///
/// # Safety
/// `stream` and `out_chunk` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_stream_next(
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
/// `nat_nexus_llm_stream_call_execute`, or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_stream_free(stream: *mut FfiStream) {
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
            cb: NatNexusToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NatNexusFreeFn,
        ) -> NatNexusStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            let wrapped = $wrapper(cb, user_data, free_fn);
            match $core_register(&name, priority, wrapped) {
                Ok(()) => NatNexusStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $deregister_name(
            name: *const c_char,
        ) -> NatNexusStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            match $core_deregister(&name) {
                Ok(_) => NatNexusStatus::Ok,
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
    nat_nexus_register_tool_sanitize_request_guardrail,
    /// Deregister a tool request sanitization guardrail by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nat_nexus_deregister_tool_sanitize_request_guardrail,
    core::nat_nexus_register_tool_sanitize_request_guardrail,
    core::nat_nexus_deregister_tool_sanitize_request_guardrail,
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
    nat_nexus_register_tool_sanitize_response_guardrail,
    /// Deregister a tool response sanitization guardrail by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nat_nexus_deregister_tool_sanitize_response_guardrail,
    core::nat_nexus_register_tool_sanitize_response_guardrail,
    core::nat_nexus_deregister_tool_sanitize_response_guardrail,
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
pub unsafe extern "C" fn nat_nexus_register_tool_conditional_execution_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NatNexusToolConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_conditional_fn(cb, user_data, free_fn);
    match core::nat_nexus_register_tool_conditional_execution_guardrail(&name, priority, wrapped) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a tool conditional execution guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_deregister_tool_conditional_execution_guardrail(
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_deregister_tool_conditional_execution_guardrail(&name) {
        Ok(_) => NatNexusStatus::Ok,
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
            cb: NatNexusToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NatNexusFreeFn,
        ) -> NatNexusStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            let wrapped = $wrapper(cb, user_data, free_fn);
            match $core_register(&name, priority, break_chain, wrapped) {
                Ok(()) => NatNexusStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $deregister_name(
            name: *const c_char,
        ) -> NatNexusStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            match $core_deregister(&name) {
                Ok(_) => NatNexusStatus::Ok,
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
    nat_nexus_register_tool_request_intercept,
    /// Deregister a tool request intercept by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nat_nexus_deregister_tool_request_intercept,
    core::nat_nexus_register_tool_request_intercept,
    core::nat_nexus_deregister_tool_request_intercept,
    wrap_tool_sanitize_fn
);

/// Register a tool execution intercept following the middleware chain pattern.
/// The callback receives `(args, next_fn, next_ctx)` — call
/// `next_fn(args, next_ctx)` to invoke the next intercept or the original
/// tool function, or skip calling it to short-circuit.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `exec_cb`: Middleware callback receiving args and a next function.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `name` must be a valid C string. Callback pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_register_tool_execution_intercept(
    name: *const c_char,
    priority: i32,
    exec_cb: NatNexusToolExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let exec = wrap_tool_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core::nat_nexus_register_tool_execution_intercept(&name, priority, exec) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a tool execution intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_deregister_tool_execution_intercept(
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_deregister_tool_execution_intercept(&name) {
        Ok(_) => NatNexusStatus::Ok,
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
pub unsafe extern "C" fn nat_nexus_register_llm_sanitize_request_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NatNexusLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_sanitize_request_fn(cb, user_data, free_fn);
    match core::nat_nexus_register_llm_sanitize_request_guardrail(&name, priority, wrapped) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM request sanitization guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_deregister_llm_sanitize_request_guardrail(
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_deregister_llm_sanitize_request_guardrail(&name) {
        Ok(_) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM response sanitization guardrail. The callback can inspect
/// and modify the LLM response after it is received.
///
/// # Parameters
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: JSON-to-JSON callback that receives the response JSON and returns sanitized JSON.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_register_llm_sanitize_response_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NatNexusJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_response_fn(cb, user_data, free_fn);
    match core::nat_nexus_register_llm_sanitize_response_guardrail(&name, priority, wrapped) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM response sanitization guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_deregister_llm_sanitize_response_guardrail(
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_deregister_llm_sanitize_response_guardrail(&name) {
        Ok(_) => NatNexusStatus::Ok,
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
pub unsafe extern "C" fn nat_nexus_register_llm_conditional_execution_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NatNexusLlmConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_conditional_fn(cb, user_data, free_fn);
    match core::nat_nexus_register_llm_conditional_execution_guardrail(&name, priority, wrapped) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM conditional execution guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_deregister_llm_conditional_execution_guardrail(
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_deregister_llm_conditional_execution_guardrail(&name) {
        Ok(_) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// LLM intercept registrations
// ---------------------------------------------------------------------------

/// Register an LLM request intercept. The callback can transform the
/// `LLMRequest` before it reaches the LLM provider.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `break_chain`: If true, stop processing further intercepts after this one.
/// - `cb`: LLM request transform callback (receives/returns `FfiLLMRequest`).
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_register_llm_request_intercept(
    name: *const c_char,
    priority: i32,
    break_chain: bool,
    cb: NatNexusLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_request_intercept_fn(cb, user_data, free_fn);
    match core::nat_nexus_register_llm_request_intercept(&name, priority, break_chain, wrapped) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM request intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_deregister_llm_request_intercept(
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_deregister_llm_request_intercept(&name) {
        Ok(_) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM execution intercept following the middleware chain pattern.
/// The callback receives `(request, next_fn, next_ctx)` — call
/// `next_fn(request, next_ctx)` to invoke the next intercept or the original
/// LLM call, or skip calling it to short-circuit.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `exec_cb`: Middleware callback receiving request and a next function.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `name` must be a valid C string. Callback pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_register_llm_execution_intercept(
    name: *const c_char,
    priority: i32,
    exec_cb: NatNexusLlmExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let exec = wrap_llm_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core::nat_nexus_register_llm_execution_intercept(&name, priority, exec) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM execution intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_deregister_llm_execution_intercept(
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_deregister_llm_execution_intercept(&name) {
        Ok(_) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM streaming execution intercept following the middleware chain
/// pattern. The callback receives `(request, next_fn, next_ctx)` — call
/// `next_fn(request, next_ctx)` to invoke the next intercept or the original
/// streaming LLM call, or skip calling it to short-circuit.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `exec_cb`: Middleware callback receiving request and a next function.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `name` must be a valid C string. Callback pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_register_llm_stream_execution_intercept(
    name: *const c_char,
    priority: i32,
    exec_cb: NatNexusLlmExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let exec = wrap_llm_stream_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core::nat_nexus_register_llm_stream_execution_intercept(&name, priority, exec) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM streaming execution intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_deregister_llm_stream_execution_intercept(
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_deregister_llm_stream_execution_intercept(&name) {
        Ok(_) => NatNexusStatus::Ok,
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
pub unsafe extern "C" fn nat_nexus_register_subscriber(
    name: *const c_char,
    cb: NatNexusEventSubscriberCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_event_subscriber(cb, user_data, free_fn);
    match core::nat_nexus_register_subscriber(&name, wrapped) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an event subscriber by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_deregister_subscriber(name: *const c_char) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_deregister_subscriber(&name) {
        Ok(_) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Scope stack isolation
// ---------------------------------------------------------------------------

/// Create a new isolated scope stack with its own root scope.
///
/// Each scope stack is independent: scopes pushed on one do not appear on another.
/// Use `nat_nexus_scope_stack_set_thread` to bind a stack to the current thread
/// before making other Nexus API calls.
///
/// # Parameters
/// - `out`: On success, receives a heap-allocated `FfiScopeStack` that must be
///   freed with `nat_nexus_scope_stack_free`.
///
/// # Safety
/// `out` must be a valid, non-null pointer.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_stack_create(
    out: *mut *mut FfiScopeStack,
) -> NatNexusStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NatNexusStatus::NullPointer;
    }
    let handle = core::create_scope_stack();
    unsafe { *out = Box::into_raw(Box::new(FfiScopeStack(handle))) };
    NatNexusStatus::Ok
}

/// Bind an isolated scope stack to the current OS thread.
///
/// After this call, all Nexus scope operations on the current thread
/// (e.g. `nat_nexus_push_scope`, `nat_nexus_get_handle`) will use the
/// given scope stack. This is typically used from Go goroutines that have
/// called `runtime.LockOSThread()`.
///
/// The `FfiScopeStack` is **not** consumed — the caller retains ownership
/// and must still free it when done.
///
/// # Safety
/// `stack` must be a valid, non-null `FfiScopeStack` pointer.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_stack_set_thread(
    stack: *const FfiScopeStack,
) -> NatNexusStatus {
    clear_last_error();
    if stack.is_null() {
        set_last_error("stack pointer is null");
        return NatNexusStatus::NullPointer;
    }
    let handle = unsafe { &*stack }.0.clone();
    core::set_thread_scope_stack(handle);
    NatNexusStatus::Ok
}

/// Returns whether the current execution context has an explicitly-initialized
/// scope stack.
///
/// Returns `true` if `nat_nexus_scope_stack_set_thread` has been called on the
/// current OS thread (or the caller is inside a tokio task-local scope).
/// Returns `false` when only the auto-created default is present.
#[no_mangle]
pub extern "C" fn nat_nexus_scope_stack_active() -> bool {
    core::scope_stack_active()
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
pub unsafe extern "C" fn nat_nexus_atif_exporter_create(
    session_id: *const c_char,
    agent_name: *const c_char,
    agent_version: *const c_char,
    model_name: *const c_char,
    out: *mut *mut FfiAtifExporter,
) -> NatNexusStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NatNexusStatus::NullPointer;
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

    let agent_info = nvidia_nat_nexus_core::atif::AtifAgentInfo {
        name: agent_name,
        version: agent_version,
        model_name: model_name_opt,
        tool_definitions: None,
        extra: None,
    };

    let exporter = nvidia_nat_nexus_core::atif::AtifExporter::new(session_id, agent_info);
    unsafe { *out = Box::into_raw(Box::new(FfiAtifExporter(exporter))) };
    NatNexusStatus::Ok
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
pub unsafe extern "C" fn nat_nexus_atif_exporter_register(
    exporter: *const FfiAtifExporter,
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NatNexusStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let subscriber = unsafe { &*exporter }.0.subscriber();
    match core::nat_nexus_register_subscriber(&name, subscriber) {
        Ok(()) => NatNexusStatus::Ok,
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
pub unsafe extern "C" fn nat_nexus_atif_exporter_deregister(name: *const c_char) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_deregister_subscriber(&name) {
        Ok(_) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Exports collected events as an ATIF trajectory JSON string.
///
/// # Parameters
/// - `exporter`: The exporter handle.
/// - `root_uuid`: Optional root UUID filter (nullable C string).
/// - `out`: On success, receives a JSON string (caller must free with
///   `nat_nexus_string_free`).
///
/// # Safety
/// `exporter` and `out` must be valid, non-null pointers. `root_uuid` may be
/// null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_atif_exporter_export(
    exporter: *const FfiAtifExporter,
    root_uuid: *const c_char,
    out: *mut *mut c_char,
) -> NatNexusStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NatNexusStatus::NullPointer;
    }
    if out.is_null() {
        set_last_error("out pointer is null");
        return NatNexusStatus::NullPointer;
    }
    let root_uuid_opt = if root_uuid.is_null() {
        None
    } else {
        match parse_scope_uuid(root_uuid) {
            Ok(u) => Some(u),
            Err(status) => return status,
        }
    };

    let trajectory = unsafe { &*exporter }.0.export(root_uuid_opt);
    match serde_json::to_string(&trajectory) {
        Ok(json_str) => {
            unsafe { *out = str_to_c_string(&json_str) };
            NatNexusStatus::Ok
        }
        Err(e) => {
            set_last_error(&format!("failed to serialize trajectory: {e}"));
            NatNexusStatus::Internal
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
pub unsafe extern "C" fn nat_nexus_atif_exporter_clear(
    exporter: *const FfiAtifExporter,
) -> NatNexusStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NatNexusStatus::NullPointer;
    }
    unsafe { &*exporter }.0.clear();
    NatNexusStatus::Ok
}

// ---------------------------------------------------------------------------
// Scope-local tool guardrail registrations
// ---------------------------------------------------------------------------

/// Helper to parse a scope UUID from a C string.
fn parse_scope_uuid(scope_uuid: *const c_char) -> Result<uuid::Uuid, NatNexusStatus> {
    let uuid_str = c_str_to_string(scope_uuid)?;
    uuid::Uuid::parse_str(&uuid_str).map_err(|e| {
        set_last_error(&format!("invalid scope UUID: {e}"));
        NatNexusStatus::InvalidArg
    })
}

macro_rules! ffi_scope_guardrail_tool_api {
    ($(#[$reg_doc:meta])* $register_name:ident,
     $(#[$dereg_doc:meta])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:ident) => {
        $(#[$reg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $register_name(
            scope_uuid: *const c_char,
            name: *const c_char,
            priority: i32,
            cb: NatNexusToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NatNexusFreeFn,
        ) -> NatNexusStatus {
            clear_last_error();
            let uuid = match parse_scope_uuid(scope_uuid) {
                Ok(u) => u,
                Err(status) => return status,
            };
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            let wrapped = $wrapper(cb, user_data, free_fn);
            match $core_register(&uuid, &name, priority, wrapped) {
                Ok(()) => NatNexusStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $deregister_name(
            scope_uuid: *const c_char,
            name: *const c_char,
        ) -> NatNexusStatus {
            clear_last_error();
            let uuid = match parse_scope_uuid(scope_uuid) {
                Ok(u) => u,
                Err(status) => return status,
            };
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            match $core_deregister(&uuid, &name) {
                Ok(_) => NatNexusStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }
    };
}

ffi_scope_guardrail_tool_api!(
    /// Register a scope-local tool request sanitization guardrail.
    ///
    /// # Parameters
    /// - `scope_uuid`: UUID of the target scope (null-terminated C string).
    /// - `name`: Unique guardrail name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `cb`: Sanitize callback.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
    nat_nexus_scope_register_tool_sanitize_request_guardrail,
    /// Deregister a scope-local tool request sanitization guardrail by name.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings.
    nat_nexus_scope_deregister_tool_sanitize_request_guardrail,
    core::nat_nexus_scope_register_tool_sanitize_request_guardrail,
    core::nat_nexus_scope_deregister_tool_sanitize_request_guardrail,
    wrap_tool_sanitize_fn
);

ffi_scope_guardrail_tool_api!(
    /// Register a scope-local tool response sanitization guardrail.
    ///
    /// # Parameters
    /// - `scope_uuid`: UUID of the target scope (null-terminated C string).
    /// - `name`: Unique guardrail name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `cb`: Sanitize callback.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
    nat_nexus_scope_register_tool_sanitize_response_guardrail,
    /// Deregister a scope-local tool response sanitization guardrail by name.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings.
    nat_nexus_scope_deregister_tool_sanitize_response_guardrail,
    core::nat_nexus_scope_register_tool_sanitize_response_guardrail,
    core::nat_nexus_scope_deregister_tool_sanitize_response_guardrail,
    wrap_tool_sanitize_fn
);

/// Register a scope-local tool conditional execution guardrail.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: Conditional callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_register_tool_conditional_execution_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NatNexusToolConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_conditional_fn(cb, user_data, free_fn);
    match core::nat_nexus_scope_register_tool_conditional_execution_guardrail(
        &uuid, &name, priority, wrapped,
    ) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local tool conditional execution guardrail by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_deregister_tool_conditional_execution_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_scope_deregister_tool_conditional_execution_guardrail(&uuid, &name) {
        Ok(_) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Scope-local tool intercept registrations
// ---------------------------------------------------------------------------

macro_rules! ffi_scope_intercept_tool_api {
    ($(#[$reg_doc:meta])* $register_name:ident,
     $(#[$dereg_doc:meta])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:ident) => {
        $(#[$reg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $register_name(
            scope_uuid: *const c_char,
            name: *const c_char,
            priority: i32,
            break_chain: bool,
            cb: NatNexusToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NatNexusFreeFn,
        ) -> NatNexusStatus {
            clear_last_error();
            let uuid = match parse_scope_uuid(scope_uuid) {
                Ok(u) => u,
                Err(status) => return status,
            };
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            let wrapped = $wrapper(cb, user_data, free_fn);
            match $core_register(&uuid, &name, priority, break_chain, wrapped) {
                Ok(()) => NatNexusStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $deregister_name(
            scope_uuid: *const c_char,
            name: *const c_char,
        ) -> NatNexusStatus {
            clear_last_error();
            let uuid = match parse_scope_uuid(scope_uuid) {
                Ok(u) => u,
                Err(status) => return status,
            };
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            match $core_deregister(&uuid, &name) {
                Ok(_) => NatNexusStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }
    };
}

ffi_scope_intercept_tool_api!(
    /// Register a scope-local tool request intercept.
    ///
    /// # Parameters
    /// - `scope_uuid`: UUID of the target scope (null-terminated C string).
    /// - `name`: Unique intercept name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `break_chain`: If true, stop processing further intercepts after this one.
    /// - `cb`: Transform callback.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
    nat_nexus_scope_register_tool_request_intercept,
    /// Deregister a scope-local tool request intercept by name.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings.
    nat_nexus_scope_deregister_tool_request_intercept,
    core::nat_nexus_scope_register_tool_request_intercept,
    core::nat_nexus_scope_deregister_tool_request_intercept,
    wrap_tool_sanitize_fn
);

/// Register a scope-local tool execution intercept following the middleware
/// chain pattern.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `exec_cb`: Middleware callback receiving args and a next function.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. Callback pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_register_tool_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    exec_cb: NatNexusToolExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let exec = wrap_tool_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core::nat_nexus_scope_register_tool_execution_intercept(&uuid, &name, priority, exec) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local tool execution intercept by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_deregister_tool_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_scope_deregister_tool_execution_intercept(&uuid, &name) {
        Ok(_) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Scope-local LLM guardrail registrations
// ---------------------------------------------------------------------------

/// Register a scope-local LLM request sanitization guardrail.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: Request sanitize callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_register_llm_sanitize_request_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NatNexusLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_sanitize_request_fn(cb, user_data, free_fn);
    match core::nat_nexus_scope_register_llm_sanitize_request_guardrail(
        &uuid, &name, priority, wrapped,
    ) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM request sanitization guardrail by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_deregister_llm_sanitize_request_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_scope_deregister_llm_sanitize_request_guardrail(&uuid, &name) {
        Ok(_) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register a scope-local LLM response sanitization guardrail.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: JSON-to-JSON callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_register_llm_sanitize_response_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NatNexusJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_response_fn(cb, user_data, free_fn);
    match core::nat_nexus_scope_register_llm_sanitize_response_guardrail(
        &uuid, &name, priority, wrapped,
    ) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM response sanitization guardrail by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_deregister_llm_sanitize_response_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_scope_deregister_llm_sanitize_response_guardrail(&uuid, &name) {
        Ok(_) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register a scope-local LLM conditional execution guardrail.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: Conditional callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_register_llm_conditional_execution_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NatNexusLlmConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_conditional_fn(cb, user_data, free_fn);
    match core::nat_nexus_scope_register_llm_conditional_execution_guardrail(
        &uuid, &name, priority, wrapped,
    ) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM conditional execution guardrail by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_deregister_llm_conditional_execution_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_scope_deregister_llm_conditional_execution_guardrail(&uuid, &name) {
        Ok(_) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Scope-local LLM intercept registrations
// ---------------------------------------------------------------------------

/// Register a scope-local LLM request intercept.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `break_chain`: If true, stop processing further intercepts after this one.
/// - `cb`: LLM request transform callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_register_llm_request_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    break_chain: bool,
    cb: NatNexusLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_request_intercept_fn(cb, user_data, free_fn);
    match core::nat_nexus_scope_register_llm_request_intercept(
        &uuid,
        &name,
        priority,
        break_chain,
        wrapped,
    ) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM request intercept by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_deregister_llm_request_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_scope_deregister_llm_request_intercept(&uuid, &name) {
        Ok(_) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register a scope-local LLM execution intercept following the middleware
/// chain pattern.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `exec_cb`: Middleware callback receiving request and a next function.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. Callback pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_register_llm_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    exec_cb: NatNexusLlmExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let exec = wrap_llm_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core::nat_nexus_scope_register_llm_execution_intercept(&uuid, &name, priority, exec) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM execution intercept by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_deregister_llm_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_scope_deregister_llm_execution_intercept(&uuid, &name) {
        Ok(_) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register a scope-local LLM streaming execution intercept following the
/// middleware chain pattern.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `exec_cb`: Middleware callback receiving request and a next function.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. Callback pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_register_llm_stream_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    exec_cb: NatNexusLlmExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let exec = wrap_llm_stream_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core::nat_nexus_scope_register_llm_stream_execution_intercept(
        &uuid, &name, priority, exec,
    ) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM streaming execution intercept by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_deregister_llm_stream_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_scope_deregister_llm_stream_execution_intercept(&uuid, &name) {
        Ok(_) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Scope-local subscriber registrations
// ---------------------------------------------------------------------------

/// Register a scope-local event subscriber.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique subscriber name.
/// - `cb`: Event callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_register_subscriber(
    scope_uuid: *const c_char,
    name: *const c_char,
    cb: NatNexusEventSubscriberCb,
    user_data: *mut libc::c_void,
    free_fn: NatNexusFreeFn,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_event_subscriber(cb, user_data, free_fn);
    match core::nat_nexus_scope_register_subscriber(&uuid, &name, wrapped) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local event subscriber by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_deregister_subscriber(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nat_nexus_scope_deregister_subscriber(&uuid, &name) {
        Ok(_) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
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
///   with `nat_nexus_string_free`).
///
/// # Safety
/// All pointers must be valid. `out` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_tool_request_intercepts(
    name: *const c_char,
    args_json: *const c_char,
    out: *mut *mut c_char,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NatNexusStatus::InvalidJson,
    };
    match core::nat_nexus_tool_request_intercepts(&name, args) {
        Ok(result) => {
            unsafe { *out = json_to_c_string(&result) };
            NatNexusStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Run the registered tool conditional execution guardrail chain.
///
/// Returns `NatNexusStatus::Ok` if all guardrails pass, or
/// `NatNexusStatus::GuardrailRejected` if blocked.
///
/// # Parameters
/// - `name`: Tool name (null-terminated C string).
/// - `args_json`: Tool arguments as a JSON C string.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_tool_conditional_execution(
    name: *const c_char,
    args_json: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NatNexusStatus::InvalidJson,
    };
    match core::nat_nexus_tool_conditional_execution(&name, &args) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Run the registered LLM request intercept chain on the given request.
///
/// # Parameters
/// - `native_json`: The request payload as a JSON C string representing an
///   `LLMRequest` (`{"headers": {...}, "content": {...}}`).
/// - `out`: On success, receives the transformed JSON string (caller must free
///   with `nat_nexus_string_free`). The output is a serialized `LLMRequest`.
///
/// # Safety
/// All pointers must be valid. `out` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_llm_request_intercepts(
    name: *const c_char,
    native_json: *const c_char,
    out: *mut *mut c_char,
) -> NatNexusStatus {
    clear_last_error();
    let name_str = if name.is_null() {
        ""
    } else {
        unsafe { std::ffi::CStr::from_ptr(name) }
            .to_str()
            .unwrap_or_default()
    };
    let native = match c_str_to_json(native_json) {
        Some(j) => j,
        None => return NatNexusStatus::InvalidJson,
    };
    let request: core_types::LLMRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LLMRequest");
            return NatNexusStatus::InvalidJson;
        }
    };
    match core::nat_nexus_llm_request_intercepts(name_str, request) {
        Ok(transformed) => {
            let result_json = serde_json::to_value(&transformed).unwrap_or(serde_json::Value::Null);
            unsafe { *out = json_to_c_string(&result_json) };
            NatNexusStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Run the registered LLM conditional execution guardrail chain.
///
/// Returns `NatNexusStatus::Ok` if all guardrails pass, or
/// `NatNexusStatus::GuardrailRejected` if blocked.
///
/// # Parameters
/// - `native_json`: The request payload as a JSON C string representing an
///   `LLMRequest` (`{"headers": {...}, "content": {...}}`).
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_llm_conditional_execution(
    native_json: *const c_char,
) -> NatNexusStatus {
    clear_last_error();
    let native = match c_str_to_json(native_json) {
        Some(j) => j,
        None => return NatNexusStatus::InvalidJson,
    };
    let request: core_types::LLMRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LLMRequest");
            return NatNexusStatus::InvalidJson;
        }
    };
    match core::nat_nexus_llm_conditional_execution(&request) {
        Ok(()) => NatNexusStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::ptr;
    use std::sync::{Mutex, OnceLock};

    use serde_json::{json, Value as Json};
    use uuid::Uuid;

    use crate::convert::nat_nexus_string_free;
    use crate::error::{nat_nexus_last_error, NatNexusStatus};
    use crate::types::{
        nat_nexus_atif_exporter_free, nat_nexus_event_data, nat_nexus_event_input,
        nat_nexus_event_metadata, nat_nexus_event_model_name, nat_nexus_event_name,
        nat_nexus_event_output, nat_nexus_event_parent_uuid, nat_nexus_event_root_uuid,
        nat_nexus_event_scope_type, nat_nexus_event_timestamp, nat_nexus_event_tool_call_id,
        nat_nexus_event_type, nat_nexus_event_uuid, nat_nexus_llm_handle_attributes,
        nat_nexus_llm_handle_free, nat_nexus_llm_handle_name, nat_nexus_llm_handle_parent_uuid,
        nat_nexus_llm_handle_uuid, nat_nexus_llm_request_content, nat_nexus_llm_request_free,
        nat_nexus_llm_request_headers, nat_nexus_llm_request_new,
        nat_nexus_scope_handle_attributes, nat_nexus_scope_handle_data,
        nat_nexus_scope_handle_free, nat_nexus_scope_handle_metadata, nat_nexus_scope_handle_name,
        nat_nexus_scope_handle_parent_uuid, nat_nexus_scope_handle_scope_type,
        nat_nexus_scope_handle_uuid, nat_nexus_scope_stack_free, nat_nexus_tool_handle_attributes,
        nat_nexus_tool_handle_free, nat_nexus_tool_handle_name, nat_nexus_tool_handle_parent_uuid,
        nat_nexus_tool_handle_uuid, FfiAtifExporter, FfiEvent, FfiLLMHandle, FfiLLMRequest,
        FfiScopeStack, FfiToolHandle, NatNexusEventType,
    };

    static TEST_MUTEX: Mutex<()> = Mutex::new(());
    static EVENT_LOG: OnceLock<Mutex<Vec<Json>>> = OnceLock::new();
    static COLLECTED_CHUNKS: OnceLock<Mutex<Vec<Json>>> = OnceLock::new();
    static FINALIZER_CALLS: OnceLock<Mutex<usize>> = OnceLock::new();

    fn event_log() -> &'static Mutex<Vec<Json>> {
        EVENT_LOG.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn collected_chunks() -> &'static Mutex<Vec<Json>> {
        COLLECTED_CHUNKS.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn finalizer_calls() -> &'static Mutex<usize> {
        FINALIZER_CALLS.get_or_init(|| Mutex::new(0))
    }

    fn unique_name(prefix: &str) -> String {
        format!("{prefix}_{}", Uuid::new_v4().simple())
    }

    fn lock_unpoisoned<T>(mutex: &'static Mutex<T>) -> std::sync::MutexGuard<'static, T> {
        mutex.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn cstring(s: &str) -> CString {
        CString::new(s).unwrap()
    }

    unsafe fn take_string(ptr: *mut c_char) -> Option<String> {
        if ptr.is_null() {
            return None;
        }
        let s = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();
        unsafe { nat_nexus_string_free(ptr) };
        Some(s)
    }

    unsafe fn read_last_error() -> Option<String> {
        let ptr = nat_nexus_last_error();
        if ptr.is_null() {
            None
        } else {
            Some(
                unsafe { CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    }

    unsafe fn returned_json(ptr: *mut c_char) -> Json {
        serde_json::from_str(&unsafe { take_string(ptr) }.unwrap()).unwrap()
    }

    unsafe fn fresh_scope_stack() -> *mut FfiScopeStack {
        let mut stack = ptr::null_mut();
        assert_eq!(
            unsafe { nat_nexus_scope_stack_create(&mut stack) },
            NatNexusStatus::Ok
        );
        assert!(!stack.is_null());
        assert_eq!(
            unsafe { nat_nexus_scope_stack_set_thread(stack) },
            NatNexusStatus::Ok
        );
        stack
    }

    fn reset_globals() {
        lock_unpoisoned(event_log()).clear();
        lock_unpoisoned(collected_chunks()).clear();
        *lock_unpoisoned(finalizer_calls()) = 0;
    }

    unsafe extern "C" fn subscriber_cb(_user_data: *mut libc::c_void, event: *const FfiEvent) {
        let payload = json!({
            "uuid": unsafe { take_string(nat_nexus_event_uuid(event)) }.unwrap_or_default(),
            "name": unsafe { take_string(nat_nexus_event_name(event)) }.unwrap_or_default(),
            "type": unsafe { nat_nexus_event_type(event) as i32 },
            "data": unsafe { take_string(nat_nexus_event_data(event)) }
                .map(|s| serde_json::from_str::<Json>(&s).unwrap()),
            "metadata": unsafe { take_string(nat_nexus_event_metadata(event)) }
                .map(|s| serde_json::from_str::<Json>(&s).unwrap()),
            "timestamp": unsafe { take_string(nat_nexus_event_timestamp(event)) }.unwrap_or_default(),
            "input": unsafe { take_string(nat_nexus_event_input(event)) }
                .map(|s| serde_json::from_str::<Json>(&s).unwrap()),
            "output": unsafe { take_string(nat_nexus_event_output(event)) }
                .map(|s| serde_json::from_str::<Json>(&s).unwrap()),
            "model_name": unsafe { take_string(nat_nexus_event_model_name(event)) },
            "tool_call_id": unsafe { take_string(nat_nexus_event_tool_call_id(event)) },
            "root_uuid": unsafe { take_string(nat_nexus_event_root_uuid(event)) },
            "parent_uuid": unsafe { take_string(nat_nexus_event_parent_uuid(event)) },
            "scope_type": unsafe { take_string(nat_nexus_event_scope_type(event)) },
        });
        lock_unpoisoned(event_log()).push(payload);
    }

    unsafe extern "C" fn tool_request_cb(
        _user_data: *mut libc::c_void,
        _name: *const c_char,
        args_json: *const c_char,
    ) -> *mut c_char {
        let mut args: Json = serde_json::from_str(
            unsafe { CStr::from_ptr(args_json) }
                .to_str()
                .unwrap_or("null"),
        )
        .unwrap();
        args["intercepted"] = json!(true);
        CString::new(args.to_string()).unwrap().into_raw()
    }

    unsafe extern "C" fn tool_allow_cb(
        _user_data: *mut libc::c_void,
        _name: *const c_char,
        _args_json: *const c_char,
    ) -> *mut c_char {
        ptr::null_mut()
    }

    unsafe extern "C" fn tool_reject_cb(
        _user_data: *mut libc::c_void,
        _name: *const c_char,
        _args_json: *const c_char,
    ) -> *mut c_char {
        CString::new("blocked tool").unwrap().into_raw()
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

    unsafe extern "C" fn tool_exec_intercept_cb(
        _user_data: *mut libc::c_void,
        args_json: *const c_char,
        next_fn: crate::callable::NatNexusToolExecNextFn,
        next_ctx: *mut libc::c_void,
    ) -> *mut c_char {
        next_fn(args_json, next_ctx)
    }

    unsafe extern "C" fn llm_request_cb(
        _user_data: *mut libc::c_void,
        request: *const FfiLLMRequest,
    ) -> *mut FfiLLMRequest {
        let headers = unsafe { take_string(nat_nexus_llm_request_headers(request)) }
            .unwrap_or_else(|| "{}".to_string());
        let content = unsafe { take_string(nat_nexus_llm_request_content(request)) }
            .unwrap_or_else(|| "null".to_string());
        let mut content_json: Json = serde_json::from_str(&content).unwrap();
        content_json["intercepted"] = json!(true);
        let headers_c = CString::new(headers).unwrap();
        let content_c = CString::new(content_json.to_string()).unwrap();
        unsafe { nat_nexus_llm_request_new(headers_c.as_ptr(), content_c.as_ptr()) }
    }

    unsafe extern "C" fn llm_allow_cb(
        _user_data: *mut libc::c_void,
        _request: *const FfiLLMRequest,
    ) -> *mut c_char {
        ptr::null_mut()
    }

    unsafe extern "C" fn llm_reject_cb(
        _user_data: *mut libc::c_void,
        _request: *const FfiLLMRequest,
    ) -> *mut c_char {
        CString::new("blocked llm").unwrap().into_raw()
    }

    unsafe extern "C" fn llm_response_cb(
        _user_data: *mut libc::c_void,
        response_json: *const c_char,
    ) -> *mut c_char {
        let mut response: Json = serde_json::from_str(
            unsafe { CStr::from_ptr(response_json) }
                .to_str()
                .unwrap_or("null"),
        )
        .unwrap();
        response["sanitized"] = json!(true);
        CString::new(response.to_string()).unwrap().into_raw()
    }

    unsafe extern "C" fn llm_exec_cb(
        _user_data: *mut libc::c_void,
        native_json: *const c_char,
    ) -> *mut c_char {
        let request: Json = serde_json::from_str(
            unsafe { CStr::from_ptr(native_json) }
                .to_str()
                .unwrap_or("null"),
        )
        .unwrap();
        let model = request
            .get("content")
            .and_then(|v| v.get("model"))
            .cloned()
            .unwrap_or(Json::Null);
        let response = json!({
            "content": "hello from ffi",
            "role": "assistant",
            "tool_calls": [],
            "model_seen": model,
        });
        CString::new(response.to_string()).unwrap().into_raw()
    }

    unsafe extern "C" fn llm_exec_intercept_cb(
        _user_data: *mut libc::c_void,
        native_json: *const c_char,
        next_fn: crate::callable::NatNexusLlmExecNextFn,
        next_ctx: *mut libc::c_void,
    ) -> *mut c_char {
        next_fn(native_json, next_ctx)
    }

    unsafe extern "C" fn collector_cb(chunk: *const c_char) {
        let chunk: Json =
            serde_json::from_str(unsafe { CStr::from_ptr(chunk) }.to_str().unwrap_or("null"))
                .unwrap();
        lock_unpoisoned(collected_chunks()).push(chunk);
    }

    unsafe extern "C" fn finalizer_cb() -> *mut c_char {
        *lock_unpoisoned(finalizer_calls()) += 1;
        CString::new(json!({"finalized": true}).to_string())
            .unwrap()
            .into_raw()
    }

    #[test]
    fn test_ffi_error_paths_and_scope_stack() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        reset_globals();

        unsafe {
            assert_eq!(
                nat_nexus_get_handle(ptr::null_mut()),
                NatNexusStatus::NullPointer
            );
            assert!(read_last_error().unwrap().contains("out pointer is null"));

            let name = cstring("ffi_invalid_scope");
            let invalid_json = cstring("{");
            let mut handle = ptr::null_mut();
            assert_eq!(
                nat_nexus_push_scope(
                    name.as_ptr(),
                    NatNexusScopeType::Agent,
                    ptr::null(),
                    0,
                    invalid_json.as_ptr(),
                    ptr::null(),
                    &mut handle,
                ),
                NatNexusStatus::InvalidJson
            );

            let stack = fresh_scope_stack();
            assert!(nat_nexus_scope_stack_active());

            let mut root = ptr::null_mut();
            assert_eq!(nat_nexus_get_handle(&mut root), NatNexusStatus::Ok);
            let root_uuid = take_string(nat_nexus_scope_handle_uuid(root)).unwrap();
            assert!(!root_uuid.is_empty());
            assert_eq!(
                nat_nexus_scope_handle_scope_type(root) as i32,
                NatNexusScopeType::Agent as i32
            );
            assert_eq!(nat_nexus_scope_handle_attributes(root), 0);
            nat_nexus_scope_handle_free(root);

            let scope_name = cstring("ffi_scope");
            let scope_data = cstring(r#"{"scope":true}"#);
            let scope_metadata = cstring(r#"{"meta":"ok"}"#);
            let mut scope = ptr::null_mut();
            assert_eq!(
                nat_nexus_push_scope(
                    scope_name.as_ptr(),
                    NatNexusScopeType::Function,
                    ptr::null(),
                    1,
                    scope_data.as_ptr(),
                    scope_metadata.as_ptr(),
                    &mut scope,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                take_string(nat_nexus_scope_handle_name(scope)).unwrap(),
                "ffi_scope"
            );
            assert_eq!(
                nat_nexus_scope_handle_scope_type(scope) as i32,
                NatNexusScopeType::Function as i32
            );
            assert_eq!(nat_nexus_scope_handle_attributes(scope), 1);
            assert!(take_string(nat_nexus_scope_handle_parent_uuid(scope)).is_some());
            assert_eq!(
                serde_json::from_str::<Json>(
                    &take_string(nat_nexus_scope_handle_data(scope)).unwrap()
                )
                .unwrap(),
                json!({"scope": true})
            );
            assert_eq!(
                serde_json::from_str::<Json>(
                    &take_string(nat_nexus_scope_handle_metadata(scope)).unwrap()
                )
                .unwrap(),
                json!({"meta": "ok"})
            );
            assert_eq!(nat_nexus_pop_scope(scope), NatNexusStatus::Ok);
            nat_nexus_scope_handle_free(scope);

            nat_nexus_scope_stack_free(stack);
        }
    }

    #[test]
    fn test_ffi_tool_lifecycle_execute_and_helpers() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        reset_globals();

        unsafe {
            let stack = fresh_scope_stack();
            let subscriber_name = unique_name("ffi_subscriber");
            let subscriber_name_c = cstring(&subscriber_name);
            assert_eq!(
                nat_nexus_register_subscriber(
                    subscriber_name_c.as_ptr(),
                    subscriber_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );

            let intercept_name = unique_name("ffi_tool_intercept");
            let intercept_name_c = cstring(&intercept_name);
            assert_eq!(
                nat_nexus_register_tool_request_intercept(
                    intercept_name_c.as_ptr(),
                    1,
                    false,
                    tool_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );

            let conditional_name = unique_name("ffi_tool_conditional");
            let conditional_name_c = cstring(&conditional_name);
            assert_eq!(
                nat_nexus_register_tool_conditional_execution_guardrail(
                    conditional_name_c.as_ptr(),
                    1,
                    tool_allow_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );

            let tool_name = cstring("ffi_tool");
            let args = cstring(r#"{"value": 1}"#);
            let mut intercepted_out = ptr::null_mut();
            assert_eq!(
                nat_nexus_tool_request_intercepts(
                    tool_name.as_ptr(),
                    args.as_ptr(),
                    &mut intercepted_out
                ),
                NatNexusStatus::Ok
            );
            let intercepted_json = returned_json(intercepted_out);
            assert_eq!(intercepted_json["intercepted"], json!(true));

            assert_eq!(
                nat_nexus_tool_conditional_execution(tool_name.as_ptr(), args.as_ptr()),
                NatNexusStatus::Ok
            );

            let tool_call_id = cstring("call_ffi_123");
            let metadata = cstring(r#"{"source":"ffi-test"}"#);
            let mut handle: *mut FfiToolHandle = ptr::null_mut();
            assert_eq!(
                nat_nexus_tool_call(
                    tool_name.as_ptr(),
                    args.as_ptr(),
                    ptr::null(),
                    1,
                    ptr::null(),
                    metadata.as_ptr(),
                    tool_call_id.as_ptr(),
                    &mut handle,
                ),
                NatNexusStatus::Ok
            );
            assert!(take_string(nat_nexus_tool_handle_uuid(handle)).is_some());
            assert_eq!(
                take_string(nat_nexus_tool_handle_name(handle)).unwrap(),
                "ffi_tool"
            );
            assert_eq!(nat_nexus_tool_handle_attributes(handle), 1);
            assert!(take_string(nat_nexus_tool_handle_parent_uuid(handle)).is_some());

            let result = cstring(r#"{"ok": true}"#);
            assert_eq!(
                nat_nexus_tool_call_end(handle, result.as_ptr(), ptr::null(), ptr::null()),
                NatNexusStatus::Ok
            );
            nat_nexus_tool_handle_free(handle);

            let mut execute_out = ptr::null_mut();
            assert_eq!(
                nat_nexus_tool_call_execute(
                    tool_name.as_ptr(),
                    args.as_ptr(),
                    tool_exec_cb,
                    ptr::null_mut(),
                    None,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    &mut execute_out,
                ),
                NatNexusStatus::Ok
            );
            let executed_json = returned_json(execute_out);
            assert_eq!(executed_json["intercepted"], json!(true));
            assert_eq!(executed_json["executed"], json!(true));

            let events = lock_unpoisoned(event_log()).clone();
            assert!(events.iter().any(|event| event["name"] == "ffi_tool"));
            assert!(events
                .iter()
                .any(|event| event["tool_call_id"] == "call_ffi_123"));
            assert!(events.iter().any(|event| event["root_uuid"].is_string()));
            assert!(events
                .iter()
                .any(|event| event["timestamp"].as_str().is_some_and(|s| !s.is_empty())));

            let mark_name = cstring("ffi_mark");
            let mark_data = cstring(r#"{"mark":true}"#);
            let mark_metadata = cstring(r#"{"origin":"ffi"}"#);
            assert_eq!(
                nat_nexus_event(
                    mark_name.as_ptr(),
                    ptr::null(),
                    mark_data.as_ptr(),
                    mark_metadata.as_ptr(),
                ),
                NatNexusStatus::Ok
            );
            let events = lock_unpoisoned(event_log()).clone();
            assert!(events.iter().any(|event| {
                event["name"] == "ffi_mark"
                    && event["type"] == json!(NatNexusEventType::Mark as i32)
                    && event["data"] == json!({"mark": true})
                    && event["metadata"] == json!({"origin": "ffi"})
            }));

            assert_eq!(
                nat_nexus_deregister_tool_request_intercept(intercept_name_c.as_ptr()),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_deregister_tool_conditional_execution_guardrail(
                    conditional_name_c.as_ptr()
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_deregister_subscriber(subscriber_name_c.as_ptr()),
                NatNexusStatus::Ok
            );
            nat_nexus_scope_stack_free(stack);
        }
    }

    #[test]
    fn test_ffi_additional_null_and_invalid_json_paths() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        reset_globals();

        unsafe {
            let stack = fresh_scope_stack();
            let name = cstring("ffi_edge_paths");
            let args = cstring(r#"{"value": 1}"#);
            let invalid_json = cstring("{");
            let invalid_request_shape = cstring(r#"{"headers":[],"content":"bad"}"#);
            let request = cstring(r#"{"headers":{},"content":{"model":"ffi-model"}}"#);
            let mut handle: *mut FfiToolHandle = ptr::null_mut();
            let mut llm_handle: *mut FfiLLMHandle = ptr::null_mut();
            let mut out_json: *mut c_char = ptr::null_mut();
            let mut stream: *mut FfiStream = ptr::null_mut();

            assert_eq!(
                nat_nexus_tool_call(
                    name.as_ptr(),
                    args.as_ptr(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                ),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_tool_call(
                    name.as_ptr(),
                    invalid_json.as_ptr(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    &mut handle,
                ),
                NatNexusStatus::InvalidJson
            );
            assert_eq!(
                nat_nexus_tool_call(
                    name.as_ptr(),
                    args.as_ptr(),
                    ptr::null(),
                    0,
                    invalid_json.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    &mut handle,
                ),
                NatNexusStatus::InvalidJson
            );

            assert_eq!(
                nat_nexus_tool_call(
                    name.as_ptr(),
                    args.as_ptr(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    &mut handle,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_tool_call_end(ptr::null(), args.as_ptr(), ptr::null(), ptr::null()),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_tool_call_end(handle, invalid_json.as_ptr(), ptr::null(), ptr::null()),
                NatNexusStatus::InvalidJson
            );
            assert_eq!(
                nat_nexus_tool_call_end(handle, args.as_ptr(), invalid_json.as_ptr(), ptr::null(),),
                NatNexusStatus::InvalidJson
            );
            assert_eq!(
                nat_nexus_tool_call_end(handle, args.as_ptr(), ptr::null(), ptr::null()),
                NatNexusStatus::Ok
            );
            nat_nexus_tool_handle_free(handle);

            assert_eq!(
                nat_nexus_tool_call_execute(
                    name.as_ptr(),
                    args.as_ptr(),
                    tool_exec_cb,
                    ptr::null_mut(),
                    None,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                ),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_tool_call_execute(
                    name.as_ptr(),
                    invalid_json.as_ptr(),
                    tool_exec_cb,
                    ptr::null_mut(),
                    None,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    &mut out_json,
                ),
                NatNexusStatus::InvalidJson
            );

            assert_eq!(
                nat_nexus_llm_call(
                    name.as_ptr(),
                    request.as_ptr(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                ),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_llm_call(
                    name.as_ptr(),
                    invalid_json.as_ptr(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    &mut llm_handle,
                ),
                NatNexusStatus::InvalidJson
            );
            assert_eq!(
                nat_nexus_llm_call(
                    name.as_ptr(),
                    invalid_request_shape.as_ptr(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    &mut llm_handle,
                ),
                NatNexusStatus::InvalidJson
            );
            assert!(read_last_error()
                .unwrap_or_default()
                .contains("failed to parse native_json as LLMRequest"));

            assert_eq!(
                nat_nexus_llm_call(
                    name.as_ptr(),
                    request.as_ptr(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    &mut llm_handle,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_llm_call_end(ptr::null(), args.as_ptr(), ptr::null(), ptr::null()),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_llm_call_end(llm_handle, invalid_json.as_ptr(), ptr::null(), ptr::null(),),
                NatNexusStatus::InvalidJson
            );
            assert_eq!(
                nat_nexus_llm_call_end(llm_handle, args.as_ptr(), ptr::null(), ptr::null()),
                NatNexusStatus::Ok
            );
            nat_nexus_llm_handle_free(llm_handle);

            assert_eq!(
                nat_nexus_llm_call_execute(
                    name.as_ptr(),
                    request.as_ptr(),
                    llm_exec_cb,
                    ptr::null_mut(),
                    None,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                ),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_llm_call_execute(
                    name.as_ptr(),
                    invalid_request_shape.as_ptr(),
                    llm_exec_cb,
                    ptr::null_mut(),
                    None,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    &mut out_json,
                ),
                NatNexusStatus::InvalidJson
            );

            assert_eq!(
                nat_nexus_llm_stream_call_execute(
                    name.as_ptr(),
                    request.as_ptr(),
                    llm_exec_cb,
                    ptr::null_mut(),
                    None,
                    None,
                    None,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                ),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_llm_stream_call_execute(
                    name.as_ptr(),
                    invalid_request_shape.as_ptr(),
                    llm_exec_cb,
                    ptr::null_mut(),
                    None,
                    None,
                    None,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    &mut stream,
                ),
                NatNexusStatus::InvalidJson
            );

            nat_nexus_scope_stack_free(stack);
        }
    }

    #[test]
    fn test_ffi_registration_and_exporter_error_paths() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        reset_globals();

        unsafe {
            assert_eq!(
                nat_nexus_scope_stack_create(ptr::null_mut()),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_scope_stack_set_thread(ptr::null()),
                NatNexusStatus::NullPointer
            );

            let stack = fresh_scope_stack();
            let scope_name = cstring("ffi_scope_local");
            let mut scope = ptr::null_mut();
            assert_eq!(
                nat_nexus_push_scope(
                    scope_name.as_ptr(),
                    NatNexusScopeType::Function,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    &mut scope,
                ),
                NatNexusStatus::Ok
            );
            let scope_uuid = cstring(&take_string(nat_nexus_scope_handle_uuid(scope)).unwrap());
            let invalid_uuid = cstring("not-a-uuid");

            let global_tool_san_req = cstring(&unique_name("ffi_tool_san_req"));
            assert_eq!(
                nat_nexus_register_tool_sanitize_request_guardrail(
                    global_tool_san_req.as_ptr(),
                    1,
                    tool_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_register_tool_sanitize_request_guardrail(
                    global_tool_san_req.as_ptr(),
                    1,
                    tool_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::AlreadyExists
            );
            assert_eq!(
                nat_nexus_deregister_tool_sanitize_request_guardrail(global_tool_san_req.as_ptr()),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_deregister_tool_sanitize_request_guardrail(global_tool_san_req.as_ptr()),
                NatNexusStatus::Ok
            );

            let global_tool_san_resp = cstring(&unique_name("ffi_tool_san_resp"));
            assert_eq!(
                nat_nexus_register_tool_sanitize_response_guardrail(
                    global_tool_san_resp.as_ptr(),
                    1,
                    tool_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_deregister_tool_sanitize_response_guardrail(
                    global_tool_san_resp.as_ptr()
                ),
                NatNexusStatus::Ok
            );

            let global_tool_exec = cstring(&unique_name("ffi_tool_exec"));
            assert_eq!(
                nat_nexus_register_tool_execution_intercept(
                    global_tool_exec.as_ptr(),
                    1,
                    tool_exec_intercept_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_deregister_tool_execution_intercept(global_tool_exec.as_ptr()),
                NatNexusStatus::Ok
            );

            let global_llm_san_req = cstring(&unique_name("ffi_llm_san_req"));
            assert_eq!(
                nat_nexus_register_llm_sanitize_request_guardrail(
                    global_llm_san_req.as_ptr(),
                    1,
                    llm_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_deregister_llm_sanitize_request_guardrail(global_llm_san_req.as_ptr()),
                NatNexusStatus::Ok
            );

            let global_llm_exec = cstring(&unique_name("ffi_llm_exec"));
            assert_eq!(
                nat_nexus_register_llm_execution_intercept(
                    global_llm_exec.as_ptr(),
                    1,
                    llm_exec_intercept_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_deregister_llm_execution_intercept(global_llm_exec.as_ptr()),
                NatNexusStatus::Ok
            );

            let global_llm_stream_exec = cstring(&unique_name("ffi_llm_stream_exec"));
            assert_eq!(
                nat_nexus_register_llm_stream_execution_intercept(
                    global_llm_stream_exec.as_ptr(),
                    1,
                    llm_exec_intercept_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_deregister_llm_stream_execution_intercept(
                    global_llm_stream_exec.as_ptr()
                ),
                NatNexusStatus::Ok
            );

            let scope_tool_san_req = cstring(&unique_name("scope_tool_san_req"));
            assert_eq!(
                nat_nexus_scope_register_tool_sanitize_request_guardrail(
                    invalid_uuid.as_ptr(),
                    scope_tool_san_req.as_ptr(),
                    1,
                    tool_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::InvalidArg
            );
            assert_eq!(
                nat_nexus_scope_register_tool_sanitize_request_guardrail(
                    scope_uuid.as_ptr(),
                    scope_tool_san_req.as_ptr(),
                    1,
                    tool_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_scope_deregister_tool_sanitize_request_guardrail(
                    scope_uuid.as_ptr(),
                    scope_tool_san_req.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let scope_tool_san_resp = cstring(&unique_name("scope_tool_san_resp"));
            assert_eq!(
                nat_nexus_scope_register_tool_sanitize_response_guardrail(
                    scope_uuid.as_ptr(),
                    scope_tool_san_resp.as_ptr(),
                    1,
                    tool_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_scope_deregister_tool_sanitize_response_guardrail(
                    scope_uuid.as_ptr(),
                    scope_tool_san_resp.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let scope_tool_cond = cstring(&unique_name("scope_tool_cond"));
            assert_eq!(
                nat_nexus_scope_register_tool_conditional_execution_guardrail(
                    scope_uuid.as_ptr(),
                    scope_tool_cond.as_ptr(),
                    1,
                    tool_allow_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_scope_deregister_tool_conditional_execution_guardrail(
                    scope_uuid.as_ptr(),
                    scope_tool_cond.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let scope_tool_req = cstring(&unique_name("scope_tool_req"));
            assert_eq!(
                nat_nexus_scope_register_tool_request_intercept(
                    scope_uuid.as_ptr(),
                    scope_tool_req.as_ptr(),
                    1,
                    false,
                    tool_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_scope_deregister_tool_request_intercept(
                    scope_uuid.as_ptr(),
                    scope_tool_req.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let scope_tool_exec = cstring(&unique_name("scope_tool_exec"));
            assert_eq!(
                nat_nexus_scope_register_tool_execution_intercept(
                    scope_uuid.as_ptr(),
                    scope_tool_exec.as_ptr(),
                    1,
                    tool_exec_intercept_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_scope_deregister_tool_execution_intercept(
                    scope_uuid.as_ptr(),
                    scope_tool_exec.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let scope_llm_san_req = cstring(&unique_name("scope_llm_san_req"));
            assert_eq!(
                nat_nexus_scope_register_llm_sanitize_request_guardrail(
                    scope_uuid.as_ptr(),
                    scope_llm_san_req.as_ptr(),
                    1,
                    llm_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_scope_deregister_llm_sanitize_request_guardrail(
                    scope_uuid.as_ptr(),
                    scope_llm_san_req.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let scope_llm_san_resp = cstring(&unique_name("scope_llm_san_resp"));
            assert_eq!(
                nat_nexus_scope_register_llm_sanitize_response_guardrail(
                    scope_uuid.as_ptr(),
                    scope_llm_san_resp.as_ptr(),
                    1,
                    llm_response_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_scope_deregister_llm_sanitize_response_guardrail(
                    scope_uuid.as_ptr(),
                    scope_llm_san_resp.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let scope_llm_cond = cstring(&unique_name("scope_llm_cond"));
            assert_eq!(
                nat_nexus_scope_register_llm_conditional_execution_guardrail(
                    scope_uuid.as_ptr(),
                    scope_llm_cond.as_ptr(),
                    1,
                    llm_allow_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_scope_deregister_llm_conditional_execution_guardrail(
                    scope_uuid.as_ptr(),
                    scope_llm_cond.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let scope_llm_req = cstring(&unique_name("scope_llm_req"));
            assert_eq!(
                nat_nexus_scope_register_llm_request_intercept(
                    scope_uuid.as_ptr(),
                    scope_llm_req.as_ptr(),
                    1,
                    false,
                    llm_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_scope_deregister_llm_request_intercept(
                    scope_uuid.as_ptr(),
                    scope_llm_req.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let scope_llm_exec = cstring(&unique_name("scope_llm_exec"));
            assert_eq!(
                nat_nexus_scope_register_llm_execution_intercept(
                    scope_uuid.as_ptr(),
                    scope_llm_exec.as_ptr(),
                    1,
                    llm_exec_intercept_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_scope_deregister_llm_execution_intercept(
                    scope_uuid.as_ptr(),
                    scope_llm_exec.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let scope_llm_stream_exec = cstring(&unique_name("scope_llm_stream_exec"));
            assert_eq!(
                nat_nexus_scope_register_llm_stream_execution_intercept(
                    scope_uuid.as_ptr(),
                    scope_llm_stream_exec.as_ptr(),
                    1,
                    llm_exec_intercept_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_scope_deregister_llm_stream_execution_intercept(
                    scope_uuid.as_ptr(),
                    scope_llm_stream_exec.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let scope_subscriber = cstring(&unique_name("scope_subscriber"));
            assert_eq!(
                nat_nexus_scope_register_subscriber(
                    scope_uuid.as_ptr(),
                    scope_subscriber.as_ptr(),
                    subscriber_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_scope_deregister_subscriber(
                    scope_uuid.as_ptr(),
                    scope_subscriber.as_ptr(),
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_scope_deregister_subscriber(
                    scope_uuid.as_ptr(),
                    scope_subscriber.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let mut exporter: *mut FfiAtifExporter = ptr::null_mut();
            let session = cstring("ffi-session");
            let agent = cstring("ffi-agent");
            let version = cstring("1.0.0");
            assert_eq!(
                nat_nexus_atif_exporter_create(
                    session.as_ptr(),
                    agent.as_ptr(),
                    version.as_ptr(),
                    ptr::null(),
                    &mut exporter,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_atif_exporter_create(
                    session.as_ptr(),
                    agent.as_ptr(),
                    version.as_ptr(),
                    ptr::null(),
                    ptr::null_mut(),
                ),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_atif_exporter_register(ptr::null(), scope_subscriber.as_ptr()),
                NatNexusStatus::NullPointer
            );
            let mut null_export = ptr::null_mut();
            assert_eq!(
                nat_nexus_atif_exporter_export(ptr::null(), ptr::null(), &mut null_export),
                NatNexusStatus::NullPointer
            );
            let exporter_name = cstring(&unique_name("ffi_exporter_sub"));
            assert_eq!(
                nat_nexus_atif_exporter_register(exporter, exporter_name.as_ptr()),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_atif_exporter_register(exporter, exporter_name.as_ptr()),
                NatNexusStatus::AlreadyExists
            );
            let mut exported = ptr::null_mut();
            assert_eq!(
                nat_nexus_atif_exporter_export(exporter, invalid_uuid.as_ptr(), &mut exported),
                NatNexusStatus::InvalidArg
            );
            assert_eq!(
                nat_nexus_atif_exporter_export(exporter, ptr::null(), ptr::null_mut()),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_atif_exporter_clear(ptr::null()),
                NatNexusStatus::NullPointer
            );
            let missing_exporter = cstring("missing_exporter");
            assert_eq!(
                nat_nexus_atif_exporter_deregister(missing_exporter.as_ptr()),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_atif_exporter_deregister(exporter_name.as_ptr()),
                NatNexusStatus::Ok
            );
            nat_nexus_atif_exporter_free(exporter);

            let mut chunk = ptr::null_mut();
            assert_eq!(nat_nexus_stream_next(ptr::null_mut(), &mut chunk), -1);
            assert_eq!(nat_nexus_stream_next(ptr::null_mut(), ptr::null_mut()), -1);

            assert_eq!(nat_nexus_pop_scope(scope), NatNexusStatus::Ok);
            nat_nexus_scope_handle_free(scope);
            nat_nexus_scope_stack_free(stack);
        }
    }

    #[test]
    fn test_ffi_helper_rejection_and_null_name_paths() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        reset_globals();

        unsafe {
            let stack = fresh_scope_stack();
            let args = cstring(r#"{"value": 7}"#);
            let request =
                cstring(r#"{"headers":{},"content":{"model":"ffi-model","messages":[]}}"#);
            let invalid_json = cstring("{");
            let tool_name = cstring("tool");
            let llm_name = cstring("llm");
            let mut null_llm_out = ptr::null_mut();

            assert_eq!(
                nat_nexus_tool_request_intercepts(ptr::null(), args.as_ptr(), ptr::null_mut()),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_tool_request_intercepts(
                    tool_name.as_ptr(),
                    invalid_json.as_ptr(),
                    ptr::null_mut()
                ),
                NatNexusStatus::InvalidJson
            );
            assert_eq!(
                nat_nexus_tool_conditional_execution(ptr::null(), args.as_ptr()),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_tool_conditional_execution(tool_name.as_ptr(), invalid_json.as_ptr()),
                NatNexusStatus::InvalidJson
            );

            let tool_guard = cstring(&unique_name("ffi_tool_reject"));
            assert_eq!(
                nat_nexus_register_tool_conditional_execution_guardrail(
                    tool_guard.as_ptr(),
                    1,
                    tool_reject_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_tool_conditional_execution(tool_name.as_ptr(), args.as_ptr()),
                NatNexusStatus::GuardrailRejected
            );
            assert_eq!(
                nat_nexus_deregister_tool_conditional_execution_guardrail(tool_guard.as_ptr()),
                NatNexusStatus::Ok
            );

            let mut llm_out = ptr::null_mut();
            assert_eq!(
                nat_nexus_llm_request_intercepts(ptr::null(), request.as_ptr(), &mut llm_out),
                NatNexusStatus::Ok
            );
            let llm_json = returned_json(llm_out);
            assert_eq!(llm_json["content"]["model"], json!("ffi-model"));

            assert_eq!(
                nat_nexus_llm_request_intercepts(
                    llm_name.as_ptr(),
                    invalid_json.as_ptr(),
                    &mut null_llm_out
                ),
                NatNexusStatus::InvalidJson
            );
            assert_eq!(
                nat_nexus_llm_conditional_execution(invalid_json.as_ptr()),
                NatNexusStatus::InvalidJson
            );

            let llm_guard = cstring(&unique_name("ffi_llm_reject"));
            assert_eq!(
                nat_nexus_register_llm_conditional_execution_guardrail(
                    llm_guard.as_ptr(),
                    1,
                    llm_reject_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_llm_conditional_execution(request.as_ptr()),
                NatNexusStatus::GuardrailRejected
            );
            assert_eq!(
                nat_nexus_deregister_llm_conditional_execution_guardrail(llm_guard.as_ptr()),
                NatNexusStatus::Ok
            );

            nat_nexus_scope_stack_free(stack);
        }
    }

    #[test]
    fn test_ffi_registration_name_and_uuid_error_sweep() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        reset_globals();

        macro_rules! assert_invalid_arg {
            ($expr:expr) => {
                assert_eq!($expr, NatNexusStatus::InvalidArg);
            };
        }
        macro_rules! assert_null_pointer {
            ($expr:expr) => {
                assert_eq!($expr, NatNexusStatus::NullPointer);
            };
        }

        unsafe {
            let stack = fresh_scope_stack();
            let scope_name = cstring("ffi_error_sweep_scope");
            let mut scope = ptr::null_mut();
            assert_eq!(
                nat_nexus_push_scope(
                    scope_name.as_ptr(),
                    NatNexusScopeType::Function,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    &mut scope,
                ),
                NatNexusStatus::Ok
            );

            let valid_scope_uuid =
                cstring(&take_string(nat_nexus_scope_handle_uuid(scope)).unwrap());
            let invalid_scope_uuid = cstring("not-a-uuid");

            assert_null_pointer!(nat_nexus_register_tool_sanitize_request_guardrail(
                ptr::null(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_deregister_tool_sanitize_request_guardrail(
                ptr::null()
            ));
            assert_null_pointer!(nat_nexus_register_tool_sanitize_response_guardrail(
                ptr::null(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_deregister_tool_sanitize_response_guardrail(
                ptr::null()
            ));
            assert_null_pointer!(nat_nexus_register_tool_conditional_execution_guardrail(
                ptr::null(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_deregister_tool_conditional_execution_guardrail(
                ptr::null()
            ));
            assert_null_pointer!(nat_nexus_register_tool_request_intercept(
                ptr::null(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_deregister_tool_request_intercept(ptr::null()));
            assert_null_pointer!(nat_nexus_register_tool_execution_intercept(
                ptr::null(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_deregister_tool_execution_intercept(ptr::null()));
            assert_null_pointer!(nat_nexus_register_llm_sanitize_request_guardrail(
                ptr::null(),
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_deregister_llm_sanitize_request_guardrail(
                ptr::null()
            ));
            assert_null_pointer!(nat_nexus_register_llm_sanitize_response_guardrail(
                ptr::null(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_deregister_llm_sanitize_response_guardrail(
                ptr::null()
            ));
            assert_null_pointer!(nat_nexus_register_llm_conditional_execution_guardrail(
                ptr::null(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_deregister_llm_conditional_execution_guardrail(
                ptr::null()
            ));
            assert_null_pointer!(nat_nexus_register_llm_request_intercept(
                ptr::null(),
                1,
                false,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_deregister_llm_request_intercept(ptr::null()));
            assert_null_pointer!(nat_nexus_register_llm_execution_intercept(
                ptr::null(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_deregister_llm_execution_intercept(ptr::null()));
            assert_null_pointer!(nat_nexus_register_llm_stream_execution_intercept(
                ptr::null(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_deregister_llm_stream_execution_intercept(
                ptr::null()
            ));
            assert_null_pointer!(nat_nexus_register_subscriber(
                ptr::null(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_deregister_subscriber(ptr::null()));

            assert_invalid_arg!(nat_nexus_scope_register_tool_sanitize_request_guardrail(
                invalid_scope_uuid.as_ptr(),
                ptr::null(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ));
            assert_invalid_arg!(nat_nexus_scope_deregister_tool_sanitize_request_guardrail(
                invalid_scope_uuid.as_ptr(),
                ptr::null(),
            ));
            assert_null_pointer!(nat_nexus_scope_register_tool_sanitize_response_guardrail(
                valid_scope_uuid.as_ptr(),
                ptr::null(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_scope_deregister_tool_sanitize_response_guardrail(
                valid_scope_uuid.as_ptr(),
                ptr::null(),
            ));
            assert_invalid_arg!(
                nat_nexus_scope_register_tool_conditional_execution_guardrail(
                    invalid_scope_uuid.as_ptr(),
                    ptr::null(),
                    1,
                    tool_allow_cb,
                    ptr::null_mut(),
                    None,
                )
            );
            assert_invalid_arg!(
                nat_nexus_scope_deregister_tool_conditional_execution_guardrail(
                    invalid_scope_uuid.as_ptr(),
                    ptr::null(),
                )
            );
            assert_null_pointer!(nat_nexus_scope_register_tool_request_intercept(
                valid_scope_uuid.as_ptr(),
                ptr::null(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_scope_deregister_tool_request_intercept(
                valid_scope_uuid.as_ptr(),
                ptr::null(),
            ));
            assert_invalid_arg!(nat_nexus_scope_register_tool_execution_intercept(
                invalid_scope_uuid.as_ptr(),
                ptr::null(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ));
            assert_invalid_arg!(nat_nexus_scope_deregister_tool_execution_intercept(
                invalid_scope_uuid.as_ptr(),
                ptr::null(),
            ));
            assert_null_pointer!(nat_nexus_scope_register_llm_sanitize_request_guardrail(
                valid_scope_uuid.as_ptr(),
                ptr::null(),
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_scope_deregister_llm_sanitize_request_guardrail(
                valid_scope_uuid.as_ptr(),
                ptr::null(),
            ));
            assert_invalid_arg!(nat_nexus_scope_register_llm_sanitize_response_guardrail(
                invalid_scope_uuid.as_ptr(),
                ptr::null(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ));
            assert_invalid_arg!(nat_nexus_scope_deregister_llm_sanitize_response_guardrail(
                invalid_scope_uuid.as_ptr(),
                ptr::null(),
            ));
            assert_null_pointer!(
                nat_nexus_scope_register_llm_conditional_execution_guardrail(
                    valid_scope_uuid.as_ptr(),
                    ptr::null(),
                    1,
                    llm_allow_cb,
                    ptr::null_mut(),
                    None,
                )
            );
            assert_null_pointer!(
                nat_nexus_scope_deregister_llm_conditional_execution_guardrail(
                    valid_scope_uuid.as_ptr(),
                    ptr::null(),
                )
            );
            assert_invalid_arg!(nat_nexus_scope_register_llm_request_intercept(
                invalid_scope_uuid.as_ptr(),
                ptr::null(),
                1,
                false,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ));
            assert_invalid_arg!(nat_nexus_scope_deregister_llm_request_intercept(
                invalid_scope_uuid.as_ptr(),
                ptr::null(),
            ));
            assert_null_pointer!(nat_nexus_scope_register_llm_execution_intercept(
                valid_scope_uuid.as_ptr(),
                ptr::null(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_scope_deregister_llm_execution_intercept(
                valid_scope_uuid.as_ptr(),
                ptr::null(),
            ));
            assert_invalid_arg!(nat_nexus_scope_register_llm_stream_execution_intercept(
                invalid_scope_uuid.as_ptr(),
                ptr::null(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ));
            assert_invalid_arg!(nat_nexus_scope_deregister_llm_stream_execution_intercept(
                invalid_scope_uuid.as_ptr(),
                ptr::null(),
            ));
            assert_null_pointer!(nat_nexus_scope_register_subscriber(
                valid_scope_uuid.as_ptr(),
                ptr::null(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ));
            assert_null_pointer!(nat_nexus_scope_deregister_subscriber(
                valid_scope_uuid.as_ptr(),
                ptr::null(),
            ));

            assert_eq!(nat_nexus_pop_scope(scope), NatNexusStatus::Ok);
            nat_nexus_scope_handle_free(scope);
            nat_nexus_scope_stack_free(stack);
        }
    }

    #[test]
    fn test_ffi_duplicate_registration_sweep_and_helper_callbacks() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        reset_globals();

        macro_rules! assert_already_exists {
            ($expr:expr) => {
                assert_eq!($expr, NatNexusStatus::AlreadyExists);
            };
        }

        unsafe extern "C" fn tool_next_passthrough(
            _args_json: *const c_char,
            _next_ctx: *mut libc::c_void,
        ) -> *mut c_char {
            CString::new(r#"{"next":true}"#).unwrap().into_raw()
        }

        unsafe extern "C" fn llm_next_passthrough(
            _native_json: *const c_char,
            _next_ctx: *mut libc::c_void,
        ) -> *mut c_char {
            CString::new(r#"{"role":"assistant","content":"next","tool_calls":[]}"#)
                .unwrap()
                .into_raw()
        }

        unsafe {
            clear_last_error();
            assert!(read_last_error().is_none());

            let stack = fresh_scope_stack();
            let scope_name = cstring("ffi_duplicate_scope");
            let mut scope = ptr::null_mut();
            assert_eq!(
                nat_nexus_push_scope(
                    scope_name.as_ptr(),
                    NatNexusScopeType::Function,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    &mut scope,
                ),
                NatNexusStatus::Ok
            );
            let scope_uuid = cstring(&take_string(nat_nexus_scope_handle_uuid(scope)).unwrap());

            let tool_cond = cstring(&unique_name("dup_tool_cond"));
            assert_eq!(
                nat_nexus_register_tool_conditional_execution_guardrail(
                    tool_cond.as_ptr(),
                    1,
                    tool_allow_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_already_exists!(nat_nexus_register_tool_conditional_execution_guardrail(
                tool_cond.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ));
            assert_eq!(
                nat_nexus_deregister_tool_conditional_execution_guardrail(tool_cond.as_ptr()),
                NatNexusStatus::Ok
            );

            let tool_req = cstring(&unique_name("dup_tool_req"));
            assert_eq!(
                nat_nexus_register_tool_request_intercept(
                    tool_req.as_ptr(),
                    1,
                    false,
                    tool_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_already_exists!(nat_nexus_register_tool_request_intercept(
                tool_req.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ));
            assert_eq!(
                nat_nexus_deregister_tool_request_intercept(tool_req.as_ptr()),
                NatNexusStatus::Ok
            );

            let llm_san_resp = cstring(&unique_name("dup_llm_san_resp"));
            assert_eq!(
                nat_nexus_register_llm_sanitize_response_guardrail(
                    llm_san_resp.as_ptr(),
                    1,
                    llm_response_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_already_exists!(nat_nexus_register_llm_sanitize_response_guardrail(
                llm_san_resp.as_ptr(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ));
            assert_eq!(
                nat_nexus_deregister_llm_sanitize_response_guardrail(llm_san_resp.as_ptr()),
                NatNexusStatus::Ok
            );

            let llm_cond = cstring(&unique_name("dup_llm_cond"));
            assert_eq!(
                nat_nexus_register_llm_conditional_execution_guardrail(
                    llm_cond.as_ptr(),
                    1,
                    llm_allow_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_already_exists!(nat_nexus_register_llm_conditional_execution_guardrail(
                llm_cond.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ));
            assert_eq!(
                nat_nexus_deregister_llm_conditional_execution_guardrail(llm_cond.as_ptr()),
                NatNexusStatus::Ok
            );

            let llm_req = cstring(&unique_name("dup_llm_req"));
            assert_eq!(
                nat_nexus_register_llm_request_intercept(
                    llm_req.as_ptr(),
                    1,
                    false,
                    llm_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_already_exists!(nat_nexus_register_llm_request_intercept(
                llm_req.as_ptr(),
                1,
                false,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ));
            assert_eq!(
                nat_nexus_deregister_llm_request_intercept(llm_req.as_ptr()),
                NatNexusStatus::Ok
            );

            let subscriber = cstring(&unique_name("dup_subscriber"));
            assert_eq!(
                nat_nexus_register_subscriber(
                    subscriber.as_ptr(),
                    subscriber_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_already_exists!(nat_nexus_register_subscriber(
                subscriber.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ));
            assert_eq!(
                nat_nexus_deregister_subscriber(subscriber.as_ptr()),
                NatNexusStatus::Ok
            );

            let scope_tool_cond = cstring(&unique_name("dup_scope_tool_cond"));
            assert_eq!(
                nat_nexus_scope_register_tool_conditional_execution_guardrail(
                    scope_uuid.as_ptr(),
                    scope_tool_cond.as_ptr(),
                    1,
                    tool_allow_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_already_exists!(
                nat_nexus_scope_register_tool_conditional_execution_guardrail(
                    scope_uuid.as_ptr(),
                    scope_tool_cond.as_ptr(),
                    1,
                    tool_allow_cb,
                    ptr::null_mut(),
                    None,
                )
            );
            assert_eq!(
                nat_nexus_scope_deregister_tool_conditional_execution_guardrail(
                    scope_uuid.as_ptr(),
                    scope_tool_cond.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let scope_tool_req = cstring(&unique_name("dup_scope_tool_req"));
            assert_eq!(
                nat_nexus_scope_register_tool_request_intercept(
                    scope_uuid.as_ptr(),
                    scope_tool_req.as_ptr(),
                    1,
                    false,
                    tool_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_already_exists!(nat_nexus_scope_register_tool_request_intercept(
                scope_uuid.as_ptr(),
                scope_tool_req.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ));
            assert_eq!(
                nat_nexus_scope_deregister_tool_request_intercept(
                    scope_uuid.as_ptr(),
                    scope_tool_req.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let scope_llm_cond = cstring(&unique_name("dup_scope_llm_cond"));
            assert_eq!(
                nat_nexus_scope_register_llm_conditional_execution_guardrail(
                    scope_uuid.as_ptr(),
                    scope_llm_cond.as_ptr(),
                    1,
                    llm_allow_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_already_exists!(
                nat_nexus_scope_register_llm_conditional_execution_guardrail(
                    scope_uuid.as_ptr(),
                    scope_llm_cond.as_ptr(),
                    1,
                    llm_allow_cb,
                    ptr::null_mut(),
                    None,
                )
            );
            assert_eq!(
                nat_nexus_scope_deregister_llm_conditional_execution_guardrail(
                    scope_uuid.as_ptr(),
                    scope_llm_cond.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let scope_llm_req = cstring(&unique_name("dup_scope_llm_req"));
            assert_eq!(
                nat_nexus_scope_register_llm_request_intercept(
                    scope_uuid.as_ptr(),
                    scope_llm_req.as_ptr(),
                    1,
                    false,
                    llm_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_already_exists!(nat_nexus_scope_register_llm_request_intercept(
                scope_uuid.as_ptr(),
                scope_llm_req.as_ptr(),
                1,
                false,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ));
            assert_eq!(
                nat_nexus_scope_deregister_llm_request_intercept(
                    scope_uuid.as_ptr(),
                    scope_llm_req.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let scope_subscriber = cstring(&unique_name("dup_scope_subscriber"));
            assert_eq!(
                nat_nexus_scope_register_subscriber(
                    scope_uuid.as_ptr(),
                    scope_subscriber.as_ptr(),
                    subscriber_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );
            assert_already_exists!(nat_nexus_scope_register_subscriber(
                scope_uuid.as_ptr(),
                scope_subscriber.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ));
            assert_eq!(
                nat_nexus_scope_deregister_subscriber(
                    scope_uuid.as_ptr(),
                    scope_subscriber.as_ptr(),
                ),
                NatNexusStatus::Ok
            );

            let session = cstring("dup-session");
            let agent = cstring("dup-agent");
            let version = cstring("1.0.0");
            let mut exporter = ptr::null_mut();
            assert_eq!(
                nat_nexus_atif_exporter_create(
                    ptr::null(),
                    agent.as_ptr(),
                    version.as_ptr(),
                    ptr::null(),
                    &mut exporter,
                ),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_atif_exporter_create(
                    session.as_ptr(),
                    ptr::null(),
                    version.as_ptr(),
                    ptr::null(),
                    &mut exporter,
                ),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_atif_exporter_create(
                    session.as_ptr(),
                    agent.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    &mut exporter,
                ),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_atif_exporter_create(
                    session.as_ptr(),
                    agent.as_ptr(),
                    version.as_ptr(),
                    ptr::null(),
                    &mut exporter,
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_atif_exporter_register(exporter, ptr::null()),
                NatNexusStatus::NullPointer
            );
            let exporter_name = cstring(&unique_name("dup_exporter_subscriber"));
            assert_eq!(
                nat_nexus_atif_exporter_register(exporter, exporter_name.as_ptr()),
                NatNexusStatus::Ok
            );
            assert_already_exists!(nat_nexus_atif_exporter_register(
                exporter,
                exporter_name.as_ptr(),
            ));
            assert_eq!(
                nat_nexus_atif_exporter_deregister(ptr::null()),
                NatNexusStatus::NullPointer
            );
            assert_eq!(
                nat_nexus_atif_exporter_deregister(exporter_name.as_ptr()),
                NatNexusStatus::Ok
            );
            nat_nexus_atif_exporter_free(exporter);

            let args = cstring(r#"{"value":1}"#);
            let tool_intercept_json = take_string(tool_exec_intercept_cb(
                ptr::null_mut(),
                args.as_ptr(),
                tool_next_passthrough,
                ptr::null_mut(),
            ))
            .unwrap();
            assert_eq!(
                serde_json::from_str::<Json>(&tool_intercept_json).unwrap(),
                json!({"next": true})
            );

            let request =
                cstring(r#"{"headers":{},"content":{"model":"ffi-model","messages":[]}}"#);
            let llm_intercept_json = take_string(llm_exec_intercept_cb(
                ptr::null_mut(),
                request.as_ptr(),
                llm_next_passthrough,
                ptr::null_mut(),
            ))
            .unwrap();
            assert_eq!(
                serde_json::from_str::<Json>(&llm_intercept_json).unwrap(),
                json!({"role":"assistant","content":"next","tool_calls":[]})
            );

            assert_eq!(nat_nexus_pop_scope(scope), NatNexusStatus::Ok);
            nat_nexus_scope_handle_free(scope);
            nat_nexus_scope_stack_free(stack);
        }
    }

    #[test]
    fn test_ffi_llm_execute_stream_and_atif_exporter() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        reset_globals();

        unsafe {
            let stack = fresh_scope_stack();

            let subscriber_name = unique_name("ffi_llm_subscriber");
            let subscriber_name_c = cstring(&subscriber_name);
            assert_eq!(
                nat_nexus_register_subscriber(
                    subscriber_name_c.as_ptr(),
                    subscriber_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );

            let mut root = ptr::null_mut();
            assert_eq!(nat_nexus_get_handle(&mut root), NatNexusStatus::Ok);
            let root_uuid = take_string(nat_nexus_scope_handle_uuid(root)).unwrap();
            nat_nexus_scope_handle_free(root);

            let intercept_name = unique_name("ffi_llm_intercept");
            let intercept_name_c = cstring(&intercept_name);
            assert_eq!(
                nat_nexus_register_llm_request_intercept(
                    intercept_name_c.as_ptr(),
                    1,
                    false,
                    llm_request_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );

            let conditional_name = unique_name("ffi_llm_conditional");
            let conditional_name_c = cstring(&conditional_name);
            assert_eq!(
                nat_nexus_register_llm_conditional_execution_guardrail(
                    conditional_name_c.as_ptr(),
                    1,
                    llm_allow_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );

            let sanitize_name = unique_name("ffi_llm_sanitize");
            let sanitize_name_c = cstring(&sanitize_name);
            assert_eq!(
                nat_nexus_register_llm_sanitize_response_guardrail(
                    sanitize_name_c.as_ptr(),
                    1,
                    llm_response_cb,
                    ptr::null_mut(),
                    None,
                ),
                NatNexusStatus::Ok
            );

            let mut exporter: *mut FfiAtifExporter = ptr::null_mut();
            let session = cstring("ffi-session");
            let agent = cstring("ffi-agent");
            let version = cstring("1.0.0");
            let model_name = cstring("ffi-model");
            assert_eq!(
                nat_nexus_atif_exporter_create(
                    session.as_ptr(),
                    agent.as_ptr(),
                    version.as_ptr(),
                    model_name.as_ptr(),
                    &mut exporter,
                ),
                NatNexusStatus::Ok
            );

            let exporter_sub = unique_name("ffi_exporter");
            let exporter_sub_c = cstring(&exporter_sub);
            assert_eq!(
                nat_nexus_atif_exporter_register(exporter, exporter_sub_c.as_ptr()),
                NatNexusStatus::Ok
            );

            let llm_name = cstring("ffi_llm");
            let request = cstring(
                r#"{"headers":{},"content":{"messages":[{"role":"user","content":"hi"}],"model":"ffi-model"}}"#,
            );
            let headers = cstring(r#"{"Authorization":"Bearer token"}"#);
            let content = cstring(r#"{"messages":[],"model":"ffi-model"}"#);
            let llm_request = nat_nexus_llm_request_new(headers.as_ptr(), content.as_ptr());
            assert!(!llm_request.is_null());
            assert_eq!(
                serde_json::from_str::<Json>(
                    &take_string(nat_nexus_llm_request_headers(llm_request)).unwrap()
                )
                .unwrap(),
                json!({"Authorization": "Bearer token"})
            );
            assert_eq!(
                serde_json::from_str::<Json>(
                    &take_string(nat_nexus_llm_request_content(llm_request)).unwrap()
                )
                .unwrap(),
                json!({"messages": [], "model": "ffi-model"})
            );
            nat_nexus_llm_request_free(llm_request);

            let mut helper_out = ptr::null_mut();
            assert_eq!(
                nat_nexus_llm_request_intercepts(
                    llm_name.as_ptr(),
                    request.as_ptr(),
                    &mut helper_out
                ),
                NatNexusStatus::Ok
            );
            let helper_json = returned_json(helper_out);
            assert_eq!(helper_json["content"]["intercepted"], json!(true));

            assert_eq!(
                nat_nexus_llm_conditional_execution(request.as_ptr()),
                NatNexusStatus::Ok
            );

            let mut handle: *mut FfiLLMHandle = ptr::null_mut();
            assert_eq!(
                nat_nexus_llm_call(
                    llm_name.as_ptr(),
                    request.as_ptr(),
                    ptr::null(),
                    2,
                    ptr::null(),
                    ptr::null(),
                    model_name.as_ptr(),
                    &mut handle,
                ),
                NatNexusStatus::Ok
            );
            assert!(take_string(nat_nexus_llm_handle_uuid(handle)).is_some());
            assert_eq!(
                take_string(nat_nexus_llm_handle_name(handle)).unwrap(),
                "ffi_llm"
            );
            assert_eq!(nat_nexus_llm_handle_attributes(handle), 2);
            assert!(take_string(nat_nexus_llm_handle_parent_uuid(handle)).is_some());

            let response =
                cstring(r#"{"content":"manual end","role":"assistant","tool_calls":[]}"#);
            assert_eq!(
                nat_nexus_llm_call_end(handle, response.as_ptr(), ptr::null(), ptr::null()),
                NatNexusStatus::Ok
            );
            nat_nexus_llm_handle_free(handle);

            let mut execute_out = ptr::null_mut();
            assert_eq!(
                nat_nexus_llm_call_execute(
                    llm_name.as_ptr(),
                    request.as_ptr(),
                    llm_exec_cb,
                    ptr::null_mut(),
                    None,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    model_name.as_ptr(),
                    &mut execute_out,
                ),
                NatNexusStatus::Ok
            );
            let execute_json = returned_json(execute_out);
            assert_eq!(execute_json["content"], json!("hello from ffi"));
            assert_eq!(execute_json["model_seen"], json!("ffi-model"));
            let events = lock_unpoisoned(event_log()).clone();
            assert!(events
                .iter()
                .any(|event| event["output"]["sanitized"] == json!(true)));
            assert!(events
                .iter()
                .any(|event| event["model_name"] == "ffi-model"));

            let mut stream = ptr::null_mut();
            assert_eq!(
                nat_nexus_llm_stream_call_execute(
                    llm_name.as_ptr(),
                    request.as_ptr(),
                    llm_exec_cb,
                    ptr::null_mut(),
                    None,
                    Some(collector_cb),
                    Some(finalizer_cb),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    model_name.as_ptr(),
                    &mut stream,
                ),
                NatNexusStatus::Ok
            );
            let mut chunk = ptr::null_mut();
            assert_eq!(nat_nexus_stream_next(stream, &mut chunk), 1);
            let chunk_json = returned_json(chunk);
            assert_eq!(chunk_json["content"], json!("hello from ffi"));
            assert_eq!(nat_nexus_stream_next(stream, &mut chunk), 0);
            nat_nexus_stream_free(stream);

            assert_eq!(lock_unpoisoned(collected_chunks()).len(), 1);
            assert_eq!(*lock_unpoisoned(finalizer_calls()), 1);

            let mut exported = ptr::null_mut();
            let root_uuid_c = cstring(&root_uuid);
            assert_eq!(
                nat_nexus_atif_exporter_export(exporter, root_uuid_c.as_ptr(), &mut exported),
                NatNexusStatus::Ok
            );
            let trajectory = returned_json(exported);
            assert_eq!(trajectory["schema_version"], json!("ATIF-v1.6"));
            assert!(trajectory["steps"].as_array().unwrap().len() >= 4);

            assert_eq!(nat_nexus_atif_exporter_clear(exporter), NatNexusStatus::Ok);
            let mut cleared = ptr::null_mut();
            assert_eq!(
                nat_nexus_atif_exporter_export(exporter, root_uuid_c.as_ptr(), &mut cleared),
                NatNexusStatus::Ok
            );
            let cleared_json = returned_json(cleared);
            assert_eq!(cleared_json["steps"].as_array().unwrap().len(), 0);

            assert_eq!(
                nat_nexus_atif_exporter_deregister(exporter_sub_c.as_ptr()),
                NatNexusStatus::Ok
            );
            nat_nexus_atif_exporter_free(exporter);
            assert_eq!(
                nat_nexus_deregister_subscriber(subscriber_name_c.as_ptr()),
                NatNexusStatus::Ok
            );

            assert_eq!(
                nat_nexus_deregister_llm_request_intercept(intercept_name_c.as_ptr()),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_deregister_llm_conditional_execution_guardrail(
                    conditional_name_c.as_ptr()
                ),
                NatNexusStatus::Ok
            );
            assert_eq!(
                nat_nexus_deregister_llm_sanitize_response_guardrail(sanitize_name_c.as_ptr()),
                NatNexusStatus::Ok
            );
            nat_nexus_scope_stack_free(stack);
        }
    }
}
