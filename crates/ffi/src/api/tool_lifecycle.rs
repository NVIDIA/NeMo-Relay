// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    Arc, FfiScopeHandle, FfiToolHandle, NemoRelayFreeFn, NemoRelayStatus, NemoRelayToolExecCb,
    TASK_SCOPE_STACK, ToolAttributes, ToolExecutionNextFn, c_char, c_str_to_json,
    c_str_to_opt_json, c_str_to_string, clear_last_error, core_tool_api, current_scope_stack,
    json_to_c_string, set_last_error, status_from_error, tokio_runtime,
    unix_micros_to_opt_timestamp, wrap_tool_exec_fn,
};

// ---------------------------------------------------------------------------
// Tool lifecycle
// ---------------------------------------------------------------------------

/// Begin a manual tool call lifecycle span.
///
/// This emits a tool Start event after applying sanitize-request guardrails to
/// the observability payload. Request and execution intercepts only run through
/// `nemo_relay_tool_call_execute`.
///
/// # Parameters
/// - `name`: Null-terminated tool name.
/// - `args_json`: Tool arguments as a null-terminated JSON C string. These
///   arguments become the start-event data after sanitize-request guardrails.
/// - `parent`: Optional parent scope handle, or null to use the current top of
///   stack.
/// - `attributes`: Bitfield of tool attributes.
/// - `data_json`: Optional null-terminated JSON string stored on the tool
///   handle, or null.
/// - `metadata_json`: Optional null-terminated JSON metadata string recorded
///   on the start event, or null.
/// - `tool_call_id`: Optional null-terminated external correlation ID recorded
///   in the tool event category profile, or null.
/// - `timestamp_unix_micros`: Optional Unix microseconds timestamp for the
///   handle start time and start event, or null to use the current UTC time.
/// - `out`: On success, receives a heap-allocated `FfiToolHandle` that must be
///   freed with `nemo_relay_tool_handle_free`.
///
/// # Errors
/// Returns `InvalidJson` for invalid JSON inputs and `InvalidArg` when
/// `timestamp_unix_micros` is outside the supported timestamp range.
///
/// # Safety
/// `name` and `args_json` must be valid C strings. `out` must be non-null.
/// Optional pointer arguments may be null; when non-null, they must be valid
/// for reads for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_tool_call(
    name: *const c_char,
    args_json: *const c_char,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    tool_call_id: *const c_char,
    timestamp_unix_micros: *const i64,
    out: *mut *mut FfiToolHandle,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NemoRelayStatus::InvalidJson,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = ToolAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoRelayStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoRelayStatus::InvalidJson,
    };
    let tool_call_id_opt = if tool_call_id.is_null() {
        None
    } else {
        match c_str_to_string(tool_call_id) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };
    let timestamp = match unix_micros_to_opt_timestamp(timestamp_unix_micros) {
        Some(v) => v,
        None => return NemoRelayStatus::InvalidArg,
    };

    match core_tool_api::tool_call(
        core_tool_api::ToolCallParams::builder()
            .name(name.as_str())
            .args(args)
            .parent_opt(parent_ref)
            .attributes(attrs)
            .data_opt(data)
            .metadata_opt(metadata)
            .tool_call_id_opt(tool_call_id_opt)
            .timestamp_opt(timestamp)
            .build(),
    ) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiToolHandle(h))) };
            NemoRelayStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// End a manual tool call lifecycle span.
///
/// This emits a tool End event after applying sanitize-response guardrails to
/// the observability payload. Response intercepts only run through
/// `nemo_relay_tool_call_execute`.
///
/// # Parameters
/// - `handle`: The tool handle from `nemo_relay_tool_call`.
/// - `result_json`: Tool result as a null-terminated JSON C string. This
///   result becomes the end-event data after sanitize-response guardrails unless
///   it sanitizes to JSON null.
/// - `data_json`: Optional null-terminated JSON data used when the sanitized
///   result is JSON null, or null.
/// - `metadata_json`: Optional null-terminated JSON metadata recorded on the
///   end event, or null.
/// - `timestamp_unix_micros`: Optional Unix microseconds timestamp for the end
///   event, or null to use the runtime default end timestamp.
///
/// # Errors
/// Returns `InvalidJson` for invalid JSON inputs and `InvalidArg` when
/// `timestamp_unix_micros` is outside the supported timestamp range.
///
/// # Safety
/// `handle` and `result_json` must be valid, non-null pointers. Optional
/// pointer arguments may be null; when non-null, they must be valid for reads
/// for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_tool_call_end(
    handle: *const FfiToolHandle,
    result_json: *const c_char,
    data_json: *const c_char,
    metadata_json: *const c_char,
    timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NemoRelayStatus::NullPointer;
    }
    let result = match c_str_to_json(result_json) {
        Some(r) => r,
        None => return NemoRelayStatus::InvalidJson,
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoRelayStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoRelayStatus::InvalidJson,
    };
    let timestamp = match unix_micros_to_opt_timestamp(timestamp_unix_micros) {
        Some(v) => v,
        None => return NemoRelayStatus::InvalidArg,
    };

    match core_tool_api::tool_call_end(
        core_tool_api::ToolCallEndParams::builder()
            .handle(&unsafe { &*handle }.0)
            .result(result)
            .data_opt(data)
            .metadata_opt(metadata)
            .timestamp_opt(timestamp)
            .build(),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
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
///   with `nemo_relay_string_free`.
///
/// # Safety
/// `name`, `args_json`, and `out` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_tool_call_execute(
    name: *const c_char,
    args_json: *const c_char,
    func: NemoRelayToolExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NemoRelayFreeFn,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    out: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NemoRelayStatus::InvalidJson,
    };
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = ToolAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoRelayStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoRelayStatus::InvalidJson,
    };

    let exec_fn = wrap_tool_exec_fn(func, func_user_data, func_free);
    let default_fn: ToolExecutionNextFn = Arc::new(move |args| exec_fn(args));

    let scope_stack = current_scope_stack();
    let result = tokio_runtime().block_on(TASK_SCOPE_STACK.scope(scope_stack, async {
        core_tool_api::tool_call_execute(
            core_tool_api::ToolCallExecuteParams::builder()
                .name(name)
                .args(args)
                .func(default_fn)
                .parent_opt(parent_handle)
                .attributes(attrs)
                .data_opt(data)
                .metadata_opt(metadata)
                .build(),
        )
        .await
    }));

    match result {
        Ok(json) => {
            unsafe { *out = json_to_c_string(&json) };
            NemoRelayStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}
