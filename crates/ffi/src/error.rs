// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error handling for the FFI layer.
//!
//! This module defines the [`NemoRelayStatus`] enum returned by every exported
//! FFI function, along with thread-local storage for human-readable error
//! messages. After any non-`Ok` return, the caller should invoke
//! [`nemo_relay_last_error`] on the same thread to obtain a diagnostic string.
//! The error message remains valid until the next FFI call on that thread clears
//! it via [`clear_last_error`].

use std::cell::RefCell;
use std::ffi::CStr;
use std::ffi::CString;

use libc::c_char;

use nemo_relay_plugin_host::PluginError;
use nemo_relay_types::error::FlowError;

/// Status codes returned by all FFI functions.
///
/// Every `extern "C"` function in this library returns an `NemoRelayStatus`.
/// On non-`Ok` returns, call [`nemo_relay_last_error`] on the same thread to
/// retrieve a human-readable error message.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemoRelayStatus {
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
    /// A managed execution was cancelled.
    Cancelled = 10,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
    static LAST_ERROR_CAUSE_IDS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

/// Store an error message in thread-local storage for later retrieval.
pub fn set_last_error(msg: &str) {
    LAST_ERROR_CAUSE_IDS.with_borrow_mut(Vec::clear);
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(msg).ok();
    });
}

/// Clear the thread-local last-error message.
pub fn clear_last_error() {
    LAST_ERROR_CAUSE_IDS.with_borrow_mut(Vec::clear);
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
pub extern "C" fn nemo_relay_last_error() -> *const c_char {
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
pub unsafe extern "C" fn nemo_relay_set_last_error_message(msg: *const c_char) {
    if msg.is_null() {
        set_last_error("unknown callback error");
        return;
    }
    match unsafe { CStr::from_ptr(msg) }.to_str() {
        Ok(s) => set_last_error(s),
        Err(_) => set_last_error("callback error was not valid UTF-8"),
    }
}

impl From<&FlowError> for NemoRelayStatus {
    fn from(e: &FlowError) -> Self {
        match e {
            FlowError::AlreadyExists(_) => NemoRelayStatus::AlreadyExists,
            FlowError::NotFound(_) => NemoRelayStatus::NotFound,
            FlowError::InvalidArgument(_) => NemoRelayStatus::InvalidArg,
            FlowError::ScopeStackEmpty => NemoRelayStatus::ScopeStackEmpty,
            FlowError::GuardrailRejected(_) => NemoRelayStatus::GuardrailRejected,
            FlowError::Upstream(_)
            | FlowError::Internal(_)
            | FlowError::CallbackException { .. } => NemoRelayStatus::Internal,
        }
    }
}

/// Read the opaque callback failure IDs associated with the last error.
/// Writes the number of IDs to `out_count`. An empty list returns null.
/// Read on the same thread as `nemo_relay_last_error`. The returned array is
/// borrowed until the error message or cause IDs change. Do not free it.
///
/// # Safety
/// `out_count` must point to writable storage for one `uintptr_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_last_error_cause_ids(out_count: *mut usize) -> *const u64 {
    if out_count.is_null() {
        return std::ptr::null();
    }
    LAST_ERROR_CAUSE_IDS.with_borrow(|ids| {
        unsafe {
            *out_count = ids.len();
        }
        if ids.is_empty() {
            std::ptr::null()
        } else {
            ids.as_ptr()
        }
    })
}

/// Copy opaque callback failure IDs into the current last error.
/// Call after `nemo_relay_set_last_error_message` on the same thread.
/// Relay carries every selected ID without changing the error text.
/// A count of zero clears the IDs. The binding owns the data behind each ID.
///
/// # Safety
/// `ids` must point to `count` readable `uint64_t` values for this call.
/// It may be null when `count` is zero. IDs must be nonzero.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_set_last_error_cause_ids(ids: *const u64, count: usize) {
    let ids = if count == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(ids, count) }.to_vec()
    };
    LAST_ERROR_CAUSE_IDS.replace(ids);
}

#[derive(Debug)]
struct CallbackCauseIds(Vec<u64>);
impl std::fmt::Display for CallbackCauseIds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("foreign callback failure")
    }
}
impl std::error::Error for CallbackCauseIds {}

pub(crate) fn last_error_as_flow_error(fallback: &str) -> FlowError {
    let message = last_error_message().unwrap_or_else(|| fallback.to_owned());
    let ids = LAST_ERROR_CAUSE_IDS.with_borrow(Clone::clone);
    if ids.is_empty() {
        return FlowError::Internal(message);
    }
    FlowError::CallbackException {
        message,
        exception_type: "ForeignCallbackError".into(),
        source: Some(std::sync::Arc::new(CallbackCauseIds(ids))),
    }
}

/// Convert an `FlowError` to an `NemoRelayStatus`, storing the error message
/// in thread-local storage.
pub fn status_from_error(e: &FlowError) -> NemoRelayStatus {
    set_last_error(&e.to_string());
    LAST_ERROR_CAUSE_IDS.replace(callback_cause_ids(e));
    NemoRelayStatus::from(e)
}

pub(crate) fn callback_cause_ids(error: &FlowError) -> Vec<u64> {
    if let FlowError::CallbackException {
        source: Some(source),
        ..
    } = error
        && let Some(cause) = source.downcast_ref::<CallbackCauseIds>()
    {
        return cause.0.clone();
    }
    Vec::new()
}

/// Convert a `PluginError` to an `NemoRelayStatus`, storing the error message
/// in thread-local storage.
pub fn status_from_plugin_error(e: &PluginError) -> NemoRelayStatus {
    set_last_error(&e.to_string());
    match e {
        PluginError::NotFound(_) => NemoRelayStatus::NotFound,
        PluginError::Conflict(_) => NemoRelayStatus::AlreadyExists,
        PluginError::InvalidConfig(_) | PluginError::Serialization(_) => {
            NemoRelayStatus::InvalidArg
        }
        PluginError::Internal(_) | PluginError::RegistrationFailed(_) => NemoRelayStatus::Internal,
    }
}

#[cfg(test)]
#[path = "../tests/coverage/error_tests.rs"]
mod tests;

#[cfg(test)]
mod callback_cause_tests {
    use super::*;

    fn cause_ids() -> Vec<u64> {
        let mut count = 0;
        let ids = unsafe { nemo_relay_last_error_cause_ids(&mut count) };
        if count == 0 {
            assert!(ids.is_null());
            return Vec::new();
        }
        unsafe { std::slice::from_raw_parts(ids, count) }.to_vec()
    }

    #[test]
    fn selected_callback_cause_survives_thread_hops_and_clears() {
        set_last_error("same message");
        unsafe {
            nemo_relay_set_last_error_cause_ids([1].as_ptr(), 1);
        }
        let first = last_error_as_flow_error("fallback");
        set_last_error("same message");
        unsafe {
            nemo_relay_set_last_error_cause_ids([2, 3].as_ptr(), 2);
        }
        let selected = last_error_as_flow_error("fallback");
        std::thread::spawn(move || {
            assert!(cause_ids().is_empty());
            assert_eq!(status_from_error(&first), NemoRelayStatus::Internal);
            assert_eq!(cause_ids(), [1]);
            assert_eq!(status_from_error(&selected), NemoRelayStatus::Internal);
            assert_eq!(cause_ids(), [2, 3]);
            assert_eq!(
                last_error_message().as_deref(),
                Some("internal error: same message")
            );
            set_last_error("another error");
            assert!(cause_ids().is_empty());
            unsafe {
                nemo_relay_set_last_error_cause_ids([4].as_ptr(), 1);
            }
            clear_last_error();
            assert!(cause_ids().is_empty());
        })
        .join()
        .unwrap();
        clear_last_error();
    }
}
