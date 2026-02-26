//! Error handling for the FFI layer.
//!
//! This module defines the [`NvAgentRtStatus`] enum returned by every exported
//! FFI function, along with thread-local storage for human-readable error
//! messages. After any non-`Ok` return, the caller should invoke
//! [`nv_agentrt_last_error`] on the same thread to obtain a diagnostic string.
//! The error message remains valid until the next FFI call on that thread clears
//! it via [`clear_last_error`].

use std::cell::RefCell;
use std::ffi::CString;

use libc::c_char;

use nvagentrt_core::AgentRtError;

/// Status codes returned by all FFI functions.
///
/// Every `extern "C"` function in this library returns an `NvAgentRtStatus`.
/// On non-`Ok` returns, call [`nv_agentrt_last_error`] on the same thread to
/// retrieve a human-readable error message.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvAgentRtStatus {
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
pub extern "C" fn nv_agentrt_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null())
    })
}

impl From<&AgentRtError> for NvAgentRtStatus {
    fn from(e: &AgentRtError) -> Self {
        match e {
            AgentRtError::AlreadyExists(_) => NvAgentRtStatus::AlreadyExists,
            AgentRtError::NotFound(_) => NvAgentRtStatus::NotFound,
            AgentRtError::ScopeStackEmpty => NvAgentRtStatus::ScopeStackEmpty,
            AgentRtError::GuardrailRejected(_) => NvAgentRtStatus::GuardrailRejected,
            AgentRtError::Internal(_) => NvAgentRtStatus::Internal,
        }
    }
}

/// Convert an `AgentRtError` to an `NvAgentRtStatus`, storing the error message
/// in thread-local storage.
pub fn status_from_error(e: &AgentRtError) -> NvAgentRtStatus {
    set_last_error(&e.to_string());
    NvAgentRtStatus::from(e)
}
