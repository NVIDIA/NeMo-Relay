// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for the Nexus runtime.
//!
//! All fallible operations in the runtime return [`Result<T>`], which uses
//! [`NexusError`] as the error type. Errors are categorized by cause
//! (duplicate registration, missing entity, guardrail rejection, etc.).

use thiserror::Error;

/// The error type for all Nexus runtime operations.
///
/// Each variant represents a distinct failure mode that callers can match on
/// to determine the appropriate recovery strategy.
#[derive(Debug, Error)]
pub enum NexusError {
    /// A resource with the given name is already registered.
    ///
    /// Returned when attempting to register a guardrail, intercept, or subscriber
    /// with a name that is already in use. Deregister the existing entry first,
    /// or choose a different name.
    #[error("already exists: {0}")]
    AlreadyExists(String),

    /// The requested resource was not found.
    ///
    /// Returned when attempting to remove a scope handle by UUID that does not
    /// exist in the scope stack, or when looking up a non-existent entity.
    #[error("not found: {0}")]
    NotFound(String),

    /// The scope stack is empty.
    ///
    /// This should not occur under normal operation because the root scope is
    /// always present and cannot be removed.
    #[error("scope stack empty")]
    ScopeStackEmpty,

    /// A conditional execution guardrail rejected the operation.
    ///
    /// The contained string is the rejection reason provided by the guardrail.
    /// This is returned during `tool_call_execute` or `llm_call_execute` when
    /// a conditional guardrail returns `Some(reason)`.
    #[error("guardrail rejected: {0}")]
    GuardrailRejected(String),

    /// An internal runtime error (e.g., lock poisoning).
    #[error("internal error: {0}")]
    Internal(String),
}

/// A specialized [`Result`](std::result::Result) type for Nexus operations.
pub type Result<T> = std::result::Result<T, NexusError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_already_exists_display() {
        let e = NexusError::AlreadyExists("foo".into());
        assert_eq!(format!("{e}"), "already exists: foo");
    }

    #[test]
    fn test_not_found_display() {
        let e = NexusError::NotFound("bar".into());
        assert_eq!(format!("{e}"), "not found: bar");
    }

    #[test]
    fn test_scope_stack_empty_display() {
        let e = NexusError::ScopeStackEmpty;
        assert_eq!(format!("{e}"), "scope stack empty");
    }

    #[test]
    fn test_guardrail_rejected_display() {
        let e = NexusError::GuardrailRejected("blocked".into());
        assert_eq!(format!("{e}"), "guardrail rejected: blocked");
    }

    #[test]
    fn test_internal_display() {
        let e = NexusError::Internal("oops".into());
        assert_eq!(format!("{e}"), "internal error: oops");
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(NexusError::Internal("test".into()));
        assert!(e.to_string().contains("internal error"));
    }

    #[test]
    fn test_error_debug() {
        let e = NexusError::AlreadyExists("x".into());
        let debug = format!("{e:?}");
        assert!(debug.contains("AlreadyExists"));
    }
}
