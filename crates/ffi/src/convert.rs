// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::not_unsafe_ptr_arg_deref)]
//! Conversion utilities between C strings/JSON and Rust types.
//!
//! All functions in this module handle null pointers gracefully and use
//! thread-local error storage for failure reporting.

use std::ffi::{CStr, CString};
use std::ptr;

use chrono::{DateTime, Utc};
use libc::c_char;
use serde_json::Value as Json;

#[cfg(test)]
use crate::error;
use crate::error::{NemoRelayStatus, set_last_error};

/// Parse a null-terminated C string as JSON. Returns `None` on error and sets last_error.
pub fn c_str_to_json(ptr: *const c_char) -> Option<Json> {
    if ptr.is_null() {
        return Some(Json::Null);
    }
    let s = match unsafe { CStr::from_ptr(ptr) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8: {e}"));
            return None;
        }
    };
    match serde_json::from_str(s) {
        Ok(v) => Some(v),
        Err(e) => {
            set_last_error(&format!("invalid JSON: {e}"));
            None
        }
    }
}

/// Parse a null-terminated C string to an optional JSON value.
/// Returns `Ok(None)` for null pointers, `Ok(Some(v))` for valid JSON.
pub fn c_str_to_opt_json(ptr: *const c_char) -> Option<Option<Json>> {
    if ptr.is_null() {
        return Some(None);
    }
    c_str_to_json(ptr).map(Some)
}

/// Parse an optional Unix timestamp in microseconds since epoch to UTC.
pub fn unix_micros_to_opt_timestamp(ptr: *const i64) -> Option<Option<DateTime<Utc>>> {
    if ptr.is_null() {
        return Some(None);
    }
    DateTime::<Utc>::from_timestamp_micros(unsafe { ptr::read(ptr) })
        .map(Some)
        .or_else(|| {
            set_last_error("invalid timestamp: unix microseconds are outside supported range");
            None
        })
}

/// Convert a JSON value to a library-owned C string.
/// The caller must free with `nemo_relay_string_free`.
pub fn json_to_c_string(value: &Json) -> *mut c_char {
    match serde_json::to_string(value) {
        Ok(s) => CString::new(s).unwrap_or_default().into_raw(),
        Err(_) => CString::new("null").unwrap().into_raw(),
    }
}

/// Convert a Rust &str to a library-owned C string.
pub fn str_to_c_string(s: &str) -> *mut c_char {
    CString::new(s).unwrap_or_default().into_raw()
}

/// Parse a C string to a Rust String. Returns Err status on failure.
pub fn c_str_to_string(ptr: *const c_char) -> Result<String, NemoRelayStatus> {
    if ptr.is_null() {
        set_last_error("null string pointer");
        return Err(NemoRelayStatus::NullPointer);
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map(|s| s.to_string())
        .map_err(|e| {
            set_last_error(&format!("invalid UTF-8: {e}"));
            NemoRelayStatus::InvalidUtf8
        })
}

/// Free a C string previously returned by any `nemo_relay_*` accessor function.
/// Passing null is a safe no-op.
///
/// # Safety
/// `ptr` must be a pointer returned by this library, or null. Double-free is
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}

#[cfg(test)]
#[path = "../tests/coverage/convert_tests.rs"]
mod tests;
