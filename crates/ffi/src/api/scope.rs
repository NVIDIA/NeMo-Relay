// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    FfiScopeHandle, NemoRelayScopeType, NemoRelayStatus, ScopeAttributes, c_char,
    c_str_to_opt_json, c_str_to_string, clear_last_error, core_scope_api, set_last_error,
    status_from_error, unix_micros_to_opt_timestamp,
};

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

/// Retrieve the current scope handle from the thread-local scope stack.
///
/// # Parameters
/// - `out`: On success, receives a heap-allocated `FfiScopeHandle` that must be
///   freed with `nemo_relay_scope_handle_free`.
///
/// # Safety
/// `out` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_get_handle(out: *mut *mut FfiScopeHandle) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    match core_scope_api::get_handle() {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiScopeHandle(h))) };
            NemoRelayStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Push a new scope onto the scope stack.
///
/// This creates a scope handle, emits a scope Start event, and makes the new
/// scope the current top of the active stack.
///
/// # Parameters
/// - `name`: Null-terminated scope name.
/// - `scope_type`: The type of scope to create.
/// - `parent`: Optional parent scope handle, or null to use the current top of
///   stack.
/// - `attributes`: Bitfield of scope attributes.
/// - `data_json`: Optional null-terminated JSON string stored on the scope
///   handle, or null.
/// - `metadata_json`: Optional null-terminated JSON metadata string recorded
///   on the start event, or null.
/// - `input_json`: Optional null-terminated JSON string exported as the
///   semantic scope input on the start event, or null.
/// - `timestamp_unix_micros`: Optional Unix microseconds timestamp for the
///   handle start time and start event, or null to use the current UTC time.
/// - `out`: On success, receives a heap-allocated `FfiScopeHandle` that must
///   be freed with `nemo_relay_scope_handle_free`.
///
/// # Errors
/// Returns `InvalidJson` for invalid JSON inputs and `InvalidArg` when
/// `timestamp_unix_micros` is outside the supported timestamp range.
///
/// # Safety
/// `name` must be a valid C string. `out` must be non-null. `parent`,
/// `data_json`, `metadata_json`, `input_json`, and `timestamp_unix_micros` may
/// be null; when non-null, optional pointers must be valid for reads for the
/// duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_push_scope(
    name: *const c_char,
    scope_type: NemoRelayScopeType,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    input_json: *const c_char,
    timestamp_unix_micros: *const i64,
    out: *mut *mut FfiScopeHandle,
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
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = ScopeAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoRelayStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoRelayStatus::InvalidJson,
    };
    let input = match c_str_to_opt_json(input_json) {
        Some(v) => v,
        None => return NemoRelayStatus::InvalidJson,
    };
    let timestamp = match unix_micros_to_opt_timestamp(timestamp_unix_micros) {
        Some(v) => v,
        None => return NemoRelayStatus::InvalidArg,
    };

    match core_scope_api::push_scope(
        core_scope_api::PushScopeParams::builder()
            .name(name.as_str())
            .scope_type(scope_type.into())
            .parent_opt(parent_ref)
            .attributes(attrs)
            .data_opt(data)
            .metadata_opt(metadata)
            .input_opt(input)
            .timestamp_opt(timestamp)
            .build(),
    ) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiScopeHandle(h))) };
            NemoRelayStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Pop a scope from the scope stack by its handle.
///
/// This emits a scope End event and removes scope-local registrations owned by
/// the popped scope.
///
/// # Parameters
/// - `handle`: The current top-of-stack scope handle to pop.
/// - `output_json`: Optional null-terminated JSON string exported as semantic
///   scope output on the end event, or null.
/// - `timestamp_unix_micros`: Optional Unix microseconds timestamp for the end
///   event, or null to use the runtime default end timestamp.
///
/// # Errors
/// Returns `InvalidJson` for invalid output JSON, `InvalidArg` when
/// `timestamp_unix_micros` is outside the supported timestamp range, or an
/// error status when `handle` is not the current top scope.
///
/// # Safety
/// `handle` must be a valid, non-null `FfiScopeHandle` pointer. `output_json` and
/// `timestamp_unix_micros` may be null; when non-null, optional pointers must
/// be valid for reads for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_pop_scope(
    handle: *const FfiScopeHandle,
    output_json: *const c_char,
    timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
    unsafe { pop_scope_impl(handle, output_json, std::ptr::null(), timestamp_unix_micros) }
}

/// Pop a scope from the scope stack by its handle with end metadata.
///
/// This emits a scope End event and removes scope-local registrations owned by
/// the popped scope. Incoming metadata is merged over metadata stored on the
/// scope handle.
///
/// # Parameters
/// - `handle`: The current top-of-stack scope handle to pop.
/// - `output_json`: Optional null-terminated JSON string exported as semantic
///   scope output on the end event, or null.
/// - `metadata_json`: Optional null-terminated JSON metadata string recorded
///   on the end event, or null.
/// - `timestamp_unix_micros`: Optional Unix microseconds timestamp for the end
///   event, or null to use the runtime default end timestamp.
///
/// # Errors
/// Returns `InvalidJson` for invalid output or metadata JSON, `InvalidArg` when
/// `timestamp_unix_micros` is outside the supported timestamp range, or an
/// error status when `handle` is not the current top scope.
///
/// # Safety
/// `handle` must be a valid, non-null `FfiScopeHandle` pointer. Optional
/// pointer arguments may be null; when non-null, they must be valid for reads
/// for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_pop_scope_with_metadata(
    handle: *const FfiScopeHandle,
    output_json: *const c_char,
    metadata_json: *const c_char,
    timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
    unsafe { pop_scope_impl(handle, output_json, metadata_json, timestamp_unix_micros) }
}

unsafe fn pop_scope_impl(
    handle: *const FfiScopeHandle,
    output_json: *const c_char,
    metadata_json: *const c_char,
    timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NemoRelayStatus::NullPointer;
    }
    let output = match c_str_to_opt_json(output_json) {
        Some(v) => v,
        None => return NemoRelayStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(v) => v,
        None => return NemoRelayStatus::InvalidJson,
    };
    let timestamp = match unix_micros_to_opt_timestamp(timestamp_unix_micros) {
        Some(v) => v,
        None => return NemoRelayStatus::InvalidArg,
    };
    match core_scope_api::pop_scope(
        core_scope_api::PopScopeParams::builder()
            .handle_uuid(&unsafe { &*handle }.0.uuid)
            .output_opt(output)
            .metadata_opt(metadata)
            .timestamp_opt(timestamp)
            .build(),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Emit a named lifecycle event.
///
/// This creates a point-in-time Mark event without pushing or popping a scope.
///
/// # Parameters
/// - `name`: Null-terminated event name.
/// - `parent`: Optional parent scope handle, or null to use the current top of
///   stack.
/// - `data_json`: Optional null-terminated JSON data payload recorded on the
///   mark event, or null.
/// - `metadata_json`: Optional null-terminated JSON metadata payload recorded
///   on the mark event, or null.
/// - `timestamp_unix_micros`: Optional Unix microseconds timestamp for the
///   mark event, or null to use the current UTC time.
///
/// # Errors
/// Returns `InvalidJson` for invalid JSON inputs and `InvalidArg` when
/// `timestamp_unix_micros` is outside the supported timestamp range.
///
/// # Safety
/// `name` must be a valid C string. Other pointer args may be null; when
/// non-null, optional pointers must be valid for reads for the duration of the
/// call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event(
    name: *const c_char,
    parent: *const FfiScopeHandle,
    data_json: *const c_char,
    metadata_json: *const c_char,
    timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
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

    match core_scope_api::event(
        core_scope_api::EmitMarkEventParams::builder()
            .name(&name)
            .parent_opt(parent_ref)
            .data_opt(data)
            .metadata_opt(metadata)
            .timestamp_opt(timestamp)
            .build(),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}
