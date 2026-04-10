// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error handling for the FFI layer.
//!
//! This module defines the [`NemoFlowStatus`] enum returned by every exported
//! FFI function, along with thread-local storage for human-readable error
//! messages. After any non-`Ok` return, the caller should invoke
//! [`nemo_flow_last_error`] on the same thread to obtain a diagnostic string.
//! The error message remains valid until the next FFI call on that thread clears
//! it via [`clear_last_error`].

use std::cell::RefCell;
use std::ffi::CStr;
use std::ffi::CString;

use libc::c_char;

use nemo_flow::FlowError;
use nemo_flow::PluginError;

/// Status codes returned by all FFI functions.
///
/// Every `extern "C"` function in this library returns an `NemoFlowStatus`.
/// On non-`Ok` returns, call [`nemo_flow_last_error`] on the same thread to
/// retrieve a human-readable error message.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemoFlowStatus {
    /// Operation completed successfully.
    Ok = 0,
    /// A resource with the given name already exists.
    AlreadyExists = 1,
    /// The requested resource was not found.
    NotFound = 2,
    /// The scope stack is empty (no active scope).
    ScopeStackEmpty = 3,
    /// A guardrail rejected the operation.
    GuardrailRejected = 4,
    /// An internal runtime error occurred.
    Internal = 5,
    /// A required pointer argument was null.
    NullPointer = 6,
    /// A JSON string argument could not be parsed.
    InvalidJson = 7,
    /// A C string argument contained invalid UTF-8.
    InvalidUtf8 = 8,
    /// A function argument had an invalid value (e.g. malformed UUID).
    InvalidArg = 9,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Store an error message in thread-local storage for later retrieval.
pub fn set_last_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(msg).ok();
    });
}

/// Clear the thread-local last-error message.
pub fn clear_last_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Retrieve the last error message set on this thread, if any.
pub fn last_error_message() -> Option<String> {
    LAST_ERROR.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|s| s.to_string_lossy().into_owned())
    })
}

/// Retrieve the last error message set on this thread, or null if no error
/// has occurred since the last [`clear_last_error`] call.
///
/// The returned pointer borrows from thread-local storage and is valid only
/// until the next FFI call on the same thread. Do **not** free the returned
/// pointer.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_flow_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null())
    })
}

/// Set the thread-local last-error message from foreign code.
///
/// Intended for callback trampolines that need to propagate an error through
/// the existing FFI last-error channel.
///
/// # Safety
/// `msg` must be either null or a valid, null-terminated C string for the
/// duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_set_last_error_message(msg: *const c_char) {
    if msg.is_null() {
        set_last_error("unknown callback error");
        return;
    }
    match unsafe { CStr::from_ptr(msg) }.to_str() {
        Ok(s) => set_last_error(s),
        Err(_) => set_last_error("callback error was not valid UTF-8"),
    }
}

impl From<&FlowError> for NemoFlowStatus {
    fn from(e: &FlowError) -> Self {
        match e {
            FlowError::AlreadyExists(_) => NemoFlowStatus::AlreadyExists,
            FlowError::NotFound(_) => NemoFlowStatus::NotFound,
            FlowError::InvalidArgument(_) => NemoFlowStatus::InvalidArg,
            FlowError::ScopeStackEmpty => NemoFlowStatus::ScopeStackEmpty,
            FlowError::GuardrailRejected(_) => NemoFlowStatus::GuardrailRejected,
            FlowError::Internal(_) => NemoFlowStatus::Internal,
        }
    }
}

/// Convert an `FlowError` to an `NemoFlowStatus`, storing the error message
/// in thread-local storage.
pub fn status_from_error(e: &FlowError) -> NemoFlowStatus {
    set_last_error(&e.to_string());
    NemoFlowStatus::from(e)
}

/// Convert a `PluginError` to an `NemoFlowStatus`, storing the error message
/// in thread-local storage.
pub fn status_from_plugin_error(e: &PluginError) -> NemoFlowStatus {
    set_last_error(&e.to_string());
    match e {
        PluginError::NotFound(_) => NemoFlowStatus::NotFound,
        PluginError::InvalidConfig(_) | PluginError::Serialization(_) => NemoFlowStatus::InvalidArg,
        PluginError::Internal(_) | PluginError::RegistrationFailed(_) => NemoFlowStatus::Internal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{CStr, CString};

    #[test]
    fn test_last_error_round_trip_and_clear() {
        clear_last_error();
        assert_eq!(last_error_message(), None);
        assert!(nemo_flow_last_error().is_null());

        set_last_error("ffi failure");
        assert_eq!(last_error_message(), Some("ffi failure".into()));

        let raw = nemo_flow_last_error();
        assert_eq!(
            unsafe { CStr::from_ptr(raw) }.to_str().unwrap(),
            "ffi failure"
        );

        clear_last_error();
        assert_eq!(last_error_message(), None);
        assert!(nemo_flow_last_error().is_null());
    }

    #[test]
    fn test_set_last_error_message_handles_null_and_invalid_utf8() {
        unsafe { nemo_flow_set_last_error_message(std::ptr::null()) };
        assert_eq!(last_error_message(), Some("unknown callback error".into()));

        let invalid_utf8 = [0xffu8, 0];
        unsafe {
            nemo_flow_set_last_error_message(invalid_utf8.as_ptr() as *const c_char);
        }
        assert_eq!(
            last_error_message(),
            Some("callback error was not valid UTF-8".into())
        );

        let valid = CString::new("callback failed").unwrap();
        unsafe { nemo_flow_set_last_error_message(valid.as_ptr()) };
        assert_eq!(last_error_message(), Some("callback failed".into()));
    }

    #[test]
    fn test_status_from_error_maps_variants_and_sets_message() {
        let cases = [
            (
                FlowError::AlreadyExists("dup".into()),
                NemoFlowStatus::AlreadyExists,
            ),
            (
                FlowError::NotFound("missing".into()),
                NemoFlowStatus::NotFound,
            ),
            (
                FlowError::InvalidArgument("bad arg".into()),
                NemoFlowStatus::InvalidArg,
            ),
            (
                FlowError::GuardrailRejected("blocked".into()),
                NemoFlowStatus::GuardrailRejected,
            ),
            (FlowError::Internal("boom".into()), NemoFlowStatus::Internal),
            (FlowError::ScopeStackEmpty, NemoFlowStatus::ScopeStackEmpty),
        ];

        for (error, expected_status) in cases {
            clear_last_error();
            let status = status_from_error(&error);
            assert_eq!(status, expected_status);
            assert_eq!(NemoFlowStatus::from(&error), expected_status);
            assert!(last_error_message().unwrap().contains(&error.to_string()));
        }
    }
}
