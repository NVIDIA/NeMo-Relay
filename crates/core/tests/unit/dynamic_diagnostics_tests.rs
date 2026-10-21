// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::diagnostics::{DiagnosticQuery, DiagnosticsSettings, SourceIdentity};

#[test]
fn native_control_statuses_are_not_diagnostic_failures() {
    for status in [
        NemoRelayStatus::Ok,
        NemoRelayStatus::StreamEnd,
        NemoRelayStatus::Backpressured,
    ] {
        assert_eq!(native_failure(status), None);
    }
    assert_eq!(
        native_failure(NemoRelayStatus::InvalidJson),
        Some(FailureClass::InvalidInput)
    );
    assert_eq!(
        native_failure(NemoRelayStatus::Internal),
        Some(FailureClass::Internal)
    );
}

#[test]
fn plugin_lifetime_does_not_own_context_or_project_restricted_detail() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let context = Diagnostics::from_settings(
        SourceIdentity {
            instance_id: Uuid::now_v7(),
            process_id: None,
            process_role: "test".into(),
            relay_version: "test".into(),
        },
        DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        },
    )
    .unwrap();
    let projected = Arc::new(AtomicUsize::new(0));
    let calls = projected.clone();
    let _projection = context
        .subscribe_projection(move |_| {
            calls.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    let observer = PluginDiagnostics::new(
        &context,
        "example",
        PluginExecution::NativeAbi,
        Uuid::now_v7(),
    );
    observer.native_status(
        Operation::NativeCallback,
        NemoRelayStatus::Backpressured,
        PluginPhase::Runtime,
    );
    observer.native_status(
        Operation::NativeCallback,
        NemoRelayStatus::InvalidArg,
        PluginPhase::Runtime,
    );
    observer.success(SuccessCode::PluginNativeLoaded, PluginPhase::Activation);
    drop(observer);
    assert_eq!(projected.load(Ordering::Relaxed), 0);
    let view = context
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(view.lifetime.totals.failures_total.value.get(), 1);
    assert_eq!(view.lifetime.totals.successes_total.value.get(), 1);
    assert!(
        context
            .register_component(ComponentRegistration::new("other", &[]))
            .is_ok()
    );
}

#[cfg(feature = "worker-grpc")]
#[test]
fn worker_transport_classifier_uses_structured_status() {
    for (status, class) in [
        (
            tonic::Status::unavailable("arbitrary"),
            FailureClass::Unavailable,
        ),
        (tonic::Status::internal("not found"), FailureClass::Internal),
        (
            tonic::Status::unknown("permission denied"),
            FailureClass::Unknown,
        ),
    ] {
        let value = WorkerStatusClassifier(Operation::WorkerTransport).classify(&status);
        assert_eq!(value.failure_class(), Some(class.as_str()));
    }
}
