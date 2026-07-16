// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for error in the NeMo Relay core crate.

use super::*;

#[test]
fn test_already_exists_display() {
    let e = FlowError::AlreadyExists("foo".into());
    assert_eq!(format!("{e}"), "already exists: foo");
}

#[test]
fn test_not_found_display() {
    let e = FlowError::NotFound("bar".into());
    assert_eq!(format!("{e}"), "not found: bar");
}

#[test]
fn test_scope_stack_empty_display() {
    let e = FlowError::ScopeStackEmpty;
    assert_eq!(format!("{e}"), "scope stack empty");
}

#[test]
fn test_invalid_argument_display() {
    let e = FlowError::InvalidArgument("bad scope".into());
    assert_eq!(format!("{e}"), "invalid argument: bad scope");
}

#[test]
fn test_guardrail_rejected_display() {
    let e = FlowError::GuardrailRejected("blocked".into());
    assert_eq!(format!("{e}"), "guardrail rejected: blocked");
}

#[test]
fn test_internal_display() {
    let e = FlowError::Internal("oops".into());
    assert_eq!(format!("{e}"), "internal error: oops");
}

#[test]
fn test_error_is_std_error() {
    let e: Box<dyn std::error::Error> = Box::new(FlowError::Internal("test".into()));
    assert!(e.to_string().contains("internal error"));
}

#[test]
fn test_error_debug() {
    let e = FlowError::AlreadyExists("x".into());
    let debug = format!("{e:?}");
    assert!(debug.contains("AlreadyExists"));
}

#[test]
fn upstream_failures_classify_retryability_and_render_status() {
    use std::collections::BTreeMap;

    let retryable = UpstreamFailure {
        status: Some(503),
        body: "temporarily unavailable".into(),
        headers: BTreeMap::new(),
        class: UpstreamFailureClass::RetryableStatus,
    };
    assert!(retryable.is_retryable());
    assert!(retryable.to_string().contains("HTTP 503"));

    let rejected = UpstreamFailure {
        status: None,
        body: "invalid API key".into(),
        headers: BTreeMap::new(),
        class: UpstreamFailureClass::Authentication,
    };
    assert!(!rejected.is_retryable());
    assert!(rejected.to_string().contains("transport failure"));
}
