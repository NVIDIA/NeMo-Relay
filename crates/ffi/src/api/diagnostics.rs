// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental read-only diagnostics ABI. Successful strings are caller-owned.
use crate::error::{NemoRelayStatus, clear_last_error, set_last_error};
use nemo_relay::diagnostics::embedded::{self, EmbeddedDiagnosticError, EmbeddedDiagnosticsReader};
use std::ffi::{CString, c_char};

/// Opaque inspection authority. Caller must not copy, free, or forge the allocation.
pub struct FfiDiagnosticsReader(EmbeddedDiagnosticsReader);

/// Return the core's maximum encoded embedded diagnostics query size.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_relay_diagnostics_query_max_bytes() -> u32 {
    embedded::MAX_BINDING_QUERY_BYTES as u32
}

fn failure(error: EmbeddedDiagnosticError) -> NemoRelayStatus {
    set_last_error(error.reason());
    match error.reason() {
        "not_initialized" | "disabled" | "queries_disabled" => NemoRelayStatus::NotFound,
        "invalid_query" | "invalid_cursor" | "invalid_limits" => NemoRelayStatus::InvalidArg,
        _ => NemoRelayStatus::Internal,
    }
}

/// Obtain an embedded reader without starting or enabling diagnostics.
/// Errors set a stable machine-readable reason in nemo_relay_last_error.
/// # Safety
/// `out` must be writable pointer storage. Success transfers one reader allocation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_diagnostics_reader(
    out: *mut *mut FfiDiagnosticsReader,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null_pointer");
        return NemoRelayStatus::NullPointer;
    }
    unsafe {
        *out = std::ptr::null_mut();
    }
    match embedded::reader() {
        Ok(reader) => {
            unsafe {
                *out = Box::into_raw(Box::new(FfiDiagnosticsReader(reader)));
            }
            NemoRelayStatus::Ok
        }
        Err(error) => failure(error),
    }
}

/// Free a reader once and clear its slot. Null slots are harmless.
/// # Safety
/// `reader` must be null or writable storage containing null or one live reader
/// returned by nemo_relay_diagnostics_reader. No concurrent read may use it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_diagnostics_reader_free(
    reader: *mut *mut FfiDiagnosticsReader,
) {
    if !reader.is_null() {
        let value = unsafe { *reader };
        unsafe {
            *reader = std::ptr::null_mut();
        }
        if !value.is_null() {
            drop(unsafe { Box::from_raw(value) });
        }
    }
}

/// Capture a canonical page. Release successful output with nemo_relay_string_free.
/// The optional query is bounded by the core's encoded query byte limit and
/// uses canonical query fields.
/// Failure always leaves `*out` null. A reader/filter does not restrict authority.
/// # Safety
/// Reader must be live; out must be writable and non-aliasing. Query must be null
/// or readable through its NUL terminator (up to 4097 bytes). Freed/forged pointers
/// cannot be detected. Native library must remain loaded while handles are live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_diagnostics_snapshot_json(
    reader: *const FfiDiagnosticsReader,
    query: *const c_char,
    out: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null_pointer");
        return NemoRelayStatus::NullPointer;
    }
    unsafe {
        *out = std::ptr::null_mut();
    }
    if reader.is_null() {
        set_last_error("null_pointer");
        return NemoRelayStatus::NullPointer;
    }
    let query = if query.is_null() {
        "{}"
    } else {
        let mut length = 0;
        while length <= embedded::MAX_BINDING_QUERY_BYTES && unsafe { *query.add(length) } != 0 {
            length += 1;
        }
        if length > embedded::MAX_BINDING_QUERY_BYTES {
            set_last_error("invalid_query");
            return NemoRelayStatus::InvalidArg;
        }
        match std::str::from_utf8(unsafe { std::slice::from_raw_parts(query.cast::<u8>(), length) })
        {
            Ok(value) => value,
            Err(_) => {
                set_last_error("invalid_query");
                return NemoRelayStatus::InvalidUtf8;
            }
        }
    };
    match unsafe { &*reader }.0.snapshot_json(query) {
        Ok(page) => {
            // Hold copy admission through C allocation, release core lease before it.
            let value = CString::new(page.as_str()).expect("JSON contains no raw NUL");
            unsafe {
                *out = value.into_raw();
            }
            NemoRelayStatus::Ok
        }
        Err(error) => failure(error),
    }
}

/// Inspect core and adapter capacities, occupancy, and logging coverage as JSON.
/// # Safety
/// Reader must be live; out must be writable and non-aliasing. Free successful
/// output using nemo_relay_string_free. Failure always leaves `*out` null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_diagnostics_info_json(
    reader: *const FfiDiagnosticsReader,
    out: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null_pointer");
        return NemoRelayStatus::NullPointer;
    }
    unsafe {
        *out = std::ptr::null_mut();
    }
    if reader.is_null() {
        set_last_error("null_pointer");
        return NemoRelayStatus::NullPointer;
    }
    match unsafe { &*reader }.0.info() {
        Ok(info) => {
            let value = serde_json::to_string(&info).expect("diagnostic metadata serializes");
            unsafe {
                *out = CString::new(value).expect("JSON has no raw NUL").into_raw();
            }
            NemoRelayStatus::Ok
        }
        Err(error) => failure(error),
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/diagnostics_tests.rs"]
mod tests;
