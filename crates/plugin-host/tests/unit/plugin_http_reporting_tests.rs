// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn plugin_reporting_context_survives_continuations_without_duplication() {
    let error = from_message("HTTP 403: denied by provider".into());
    let error = reported_by(error, "nvidia.switchyard", "native");
    let wire = to_message(&error);
    let error = reported_by(from_message(wire), "nvidia.switchyard", "native");
    let FlowError::Upstream(failure) = error else {
        panic!("expected upstream failure");
    };
    assert_eq!(failure.status, Some(403));
    assert_eq!(
        failure.body,
        "Reported by plugin 'nvidia.switchyard': denied by provider"
    );
    assert!(!failure.is_retryable());
    assert!(failure.headers.is_empty());
}

#[test]
fn nested_plugin_reporting_preserves_the_downstream_context() {
    let error = reported_by(
        from_message("HTTP 404: missing model".into()),
        "downstream",
        "worker",
    );
    let error = reported_by(from_message(to_message(&error)), "router", "native");
    let FlowError::Upstream(failure) = error else {
        panic!("expected upstream failure");
    };
    assert_eq!(failure.status, Some(404));
    assert_eq!(
        failure.body,
        "Reported by plugin 'router': Reported by plugin 'downstream': missing model"
    );
    assert!(
        matches!(reported_by(FlowError::Internal("local error".into()), "router", "native"), FlowError::Internal(message) if message == "local error")
    );
}
