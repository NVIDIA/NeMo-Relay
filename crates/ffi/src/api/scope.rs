// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

/// Retrieve the current scope handle from the thread-local scope stack.
///
/// # Parameters
/// - `out`: On success, receives a heap-allocated `FfiScopeHandle` that must be
///   freed with `nemo_flow_scope_handle_free`.
///
/// # Safety
/// `out` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_get_handle(out: *mut *mut FfiScopeHandle) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    match core_scope_api::get_handle() {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiScopeHandle(h))) };
            NemoFlowStatus::Ok
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_push_scope(
    name: *const c_char,
    scope_type: NemoFlowScopeType,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    out: *mut *mut FfiScopeHandle,
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
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = ScopeAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };

    match core_scope_api::push_scope(&name, scope_type.into(), parent_ref, attrs, data, metadata) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiScopeHandle(h))) };
            NemoFlowStatus::Ok
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_pop_scope(handle: *const FfiScopeHandle) -> NemoFlowStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NemoFlowStatus::NullPointer;
    }
    match core_scope_api::pop_scope(&unsafe { &*handle }.0.uuid) {
        Ok(()) => NemoFlowStatus::Ok,
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_event(
    name: *const c_char,
    parent: *const FfiScopeHandle,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NemoFlowStatus {
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
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };

    match core_scope_api::event(&name, parent_ref, data, metadata) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}
