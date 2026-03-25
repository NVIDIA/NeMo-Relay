// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error handling for the FFI layer.
//!
//! This module defines the [`NatNexusStatus`] enum returned by every exported
//! FFI function, along with thread-local storage for human-readable error
//! messages. After any non-`Ok` return, the caller should invoke
//! [`nat_nexus_last_error`] on the same thread to obtain a diagnostic string.
//! The error message remains valid until the next FFI call on that thread clears
//! it via [`clear_last_error`].

use std::cell::RefCell;
use std::ffi::CString;

use libc::c_char;

use nvidia_nat_nexus_core::NexusError;

/// Status codes returned by all FFI functions.
///
/// Every `extern "C"` function in this library returns an `NatNexusStatus`.
/// On non-`Ok` returns, call [`nat_nexus_last_error`] on the same thread to
/// retrieve a human-readable error message.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NatNexusStatus {
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

/// Retrieve the last error message set on this thread, or null if no error
/// has occurred since the last [`clear_last_error`] call.
///
/// The returned pointer borrows from thread-local storage and is valid only
/// until the next FFI call on the same thread. Do **not** free the returned
/// pointer.
#[no_mangle]
pub extern "C" fn nat_nexus_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null())
    })
}

impl From<&NexusError> for NatNexusStatus {
    fn from(e: &NexusError) -> Self {
        match e {
            NexusError::AlreadyExists(_) => NatNexusStatus::AlreadyExists,
            NexusError::NotFound(_) => NatNexusStatus::NotFound,
            NexusError::ScopeStackEmpty => NatNexusStatus::ScopeStackEmpty,
            NexusError::GuardrailRejected(_) => NatNexusStatus::GuardrailRejected,
            NexusError::Internal(_) => NatNexusStatus::Internal,
        }
    }
}

/// Convert an `NexusError` to an `NatNexusStatus`, storing the error message
/// in thread-local storage.
pub fn status_from_error(e: &NexusError) -> NatNexusStatus {
    set_last_error(&e.to_string());
    NatNexusStatus::from(e)
}
