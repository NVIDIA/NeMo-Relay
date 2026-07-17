// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for the NeMo Relay runtime.
//!
//! All fallible operations in the runtime return [`Result<T>`], which uses
//! [`FlowError`] as the error type. Errors are categorized by cause
//! (duplicate registration, missing entity, guardrail rejection, etc.).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Stable classification for an upstream provider failure captured by retry-aware dispatch.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UpstreamFailureClass {
    /// Provider connection could not be established or was interrupted.
    Connection,
    /// Provider request timed out.
    Timeout,
    /// Retryable HTTP status without a more specific provider classification.
    RetryableStatus,
    /// Provider rejected the request because its context window was exceeded.
    ContextWindow,
    /// Requested provider model is temporarily unavailable.
    ModelUnavailable,
    /// Provider authentication or authorization failed.
    Authentication,
    /// Provider rejected an invalid request.
    InvalidRequest,
    /// Other non-retryable provider failure.
    Other,
}

/// Structured provider failure surfaced only for explicitly retry-aware dispatches.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UpstreamFailure {
    /// HTTP status when a provider response was received.
    pub status: Option<u16>,
    /// Bounded response body or transport error message.
    pub body: String,
    /// Safe response headers captured from the provider.
    pub headers: BTreeMap<String, String>,
    /// Retry classification.
    pub class: UpstreamFailureClass,
}

impl UpstreamFailure {
    /// Whether Switchyard may be consulted for another bounded provider attempt.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self.class,
            UpstreamFailureClass::Connection
                | UpstreamFailureClass::Timeout
                | UpstreamFailureClass::RetryableStatus
                | UpstreamFailureClass::ContextWindow
                | UpstreamFailureClass::ModelUnavailable
        )
    }
}

impl std::fmt::Display for UpstreamFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.status {
            Some(status) => write!(
                formatter,
                "upstream provider returned HTTP {status} ({:?}): {}",
                self.class, self.body
            ),
            None => write!(
                formatter,
                "upstream provider transport failure ({:?}): {}",
                self.class, self.body
            ),
        }
    }
}

/// The error type for all NeMo Relay runtime operations.
///
/// Each variant represents a distinct failure mode that callers can match on
/// to determine the appropriate recovery strategy.
#[derive(Clone, Debug, Error)]
pub enum FlowError {
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

    /// A function argument was invalid for the requested operation.
    ///
    /// Returned when a provided value is well-formed but violates an API
    /// precondition, such as attempting to pop a scope that is not currently
    /// at the top of the stack.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

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

    /// Structured upstream provider failure from retry-aware gateway dispatch.
    #[error("{0}")]
    Upstream(UpstreamFailure),

    /// An internal runtime error (e.g., lock poisoning).
    #[error("internal error: {0}")]
    Internal(String),
}

/// A specialized [`Result`](std::result::Result) type for NeMo Relay operations.
pub type Result<T> = std::result::Result<T, FlowError>;

#[cfg(test)]
#[path = "../tests/coverage/error_tests.rs"]
mod tests;
