#![allow(clippy::not_unsafe_ptr_arg_deref)]
//! Conversion utilities between C strings/JSON and Rust types.
//!
//! All functions in this module handle null pointers gracefully and use
//! thread-local error storage for failure reporting.

use std::ffi::{CStr, CString};

use libc::c_char;
use serde_json::Value as Json;

use crate::error::{set_last_error, NvAgentRtStatus};

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

/// Convert a JSON value to a library-owned C string.
/// The caller must free with `nv_agentrt_string_free`.
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
pub fn c_str_to_string(ptr: *const c_char) -> Result<String, NvAgentRtStatus> {
    if ptr.is_null() {
        set_last_error("null string pointer");
        return Err(NvAgentRtStatus::NullPointer);
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map(|s| s.to_string())
        .map_err(|e| {
            set_last_error(&format!("invalid UTF-8: {e}"));
            NvAgentRtStatus::InvalidUtf8
        })
}

/// Free a C string previously returned by any `nv_agentrt_*` accessor function.
/// Passing null is a safe no-op.
///
/// # Safety
/// `ptr` must be a pointer returned by this library, or null. Double-free is
/// undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}
