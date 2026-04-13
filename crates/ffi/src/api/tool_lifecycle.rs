// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_tool_call(
    name: *const c_char,
    args_json: *const c_char,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    tool_call_id: *const c_char,
    out: *mut *mut FfiToolHandle,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NemoFlowStatus::InvalidJson,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = ToolAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };
    let tool_call_id_opt = if tool_call_id.is_null() {
        None
    } else {
        match c_str_to_string(tool_call_id) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };

    match core_tool_api::tool_call(
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
            NemoFlowStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// End a tool call, running post-call guardrails and intercepts.
///
/// # Parameters
/// - `handle`: The tool handle from `nemo_flow_tool_call`.
/// - `result_json`: Tool result as a JSON C string.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
///
/// # Safety
/// `handle` and `result_json` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_tool_call_end(
    handle: *const FfiToolHandle,
    result_json: *const c_char,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NemoFlowStatus::NullPointer;
    }
    let result = match c_str_to_json(result_json) {
        Some(r) => r,
        None => return NemoFlowStatus::InvalidJson,
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };

    match core_tool_api::tool_call_end(&unsafe { &*handle }.0, result, data, metadata) {
        Ok(()) => NemoFlowStatus::Ok,
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
///   with `nemo_flow_string_free`.
///
/// # Safety
/// `name`, `args_json`, and `out` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_tool_call_execute(
    name: *const c_char,
    args_json: *const c_char,
    func: NemoFlowToolExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NemoFlowFreeFn,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    out: *mut *mut c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NemoFlowStatus::InvalidJson,
    };
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = ToolAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };

    let exec_fn = wrap_tool_exec_fn(func, func_user_data, func_free);
    let default_fn: ToolExecutionNextFn = Arc::new(move |args| exec_fn(args));

    let scope_stack = current_scope_stack();
    let result = tokio_runtime().block_on(TASK_SCOPE_STACK.scope(scope_stack, async {
        core_tool_api::tool_call_execute(
            &name,
            args,
            default_fn,
            parent_handle,
            attrs,
            data,
            metadata,
        )
        .await
    }));

    match result {
        Ok(json) => {
            unsafe { *out = json_to_c_string(&json) };
            NemoFlowStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}
