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

#[test]
fn native_statuses_classify_failure_semantics_without_error_text() {
    for (status, expected) in [
        (NemoRelayStatus::AlreadyExists, FailureClass::Conflict),
        (NemoRelayStatus::NotFound, FailureClass::NotFound),
        (NemoRelayStatus::ScopeStackEmpty, FailureClass::InvalidState),
        (
            NemoRelayStatus::GuardrailRejected,
            FailureClass::PolicyRejected,
        ),
        (NemoRelayStatus::NullPointer, FailureClass::InvalidInput),
        (NemoRelayStatus::InvalidUtf8, FailureClass::InvalidInput),
        (NemoRelayStatus::InvalidArg, FailureClass::InvalidInput),
    ] {
        assert_eq!(native_failure(status), Some(expected));
    }
}

#[cfg(feature = "worker-grpc")]
#[test]
fn worker_status_classes_and_caller_cancellation_are_distinct() {
    use tonic::Code;
    for (code, expected) in [
        (Code::InvalidArgument, FailureClass::InvalidInput),
        (Code::OutOfRange, FailureClass::InvalidInput),
        (Code::DeadlineExceeded, FailureClass::Timeout),
        (Code::Unauthenticated, FailureClass::Authentication),
        (Code::PermissionDenied, FailureClass::PermissionDenied),
        (Code::ResourceExhausted, FailureClass::ResourceExhausted),
        (Code::Unimplemented, FailureClass::Unsupported),
        (Code::NotFound, FailureClass::NotFound),
        (Code::AlreadyExists, FailureClass::Conflict),
        (Code::DataLoss, FailureClass::Internal),
        (Code::FailedPrecondition, FailureClass::InvalidState),
        (Code::Cancelled, FailureClass::Unknown),
    ] {
        let classified = WorkerStatusClassifier(Operation::WorkerTransport)
            .classify(&tonic::Status::new(code, "misleading input text"));
        assert_eq!(classified.failure_class(), Some(expected.as_str()));
    }
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
    let observer = PluginDiagnostics::new(
        &context,
        "worker",
        PluginExecution::WorkerProtocol,
        Uuid::now_v7(),
    );
    observer.rpc_failure(
        Operation::WorkerTransport,
        &tonic::Status::cancelled("caller stopped"),
        PluginPhase::Runtime,
    );
    assert_eq!(
        context
            .snapshot(&DiagnosticQuery::default())
            .unwrap()
            .to_snapshot()
            .unwrap()
            .lifetime
            .totals
            .failures_total
            .value
            .get(),
        0
    );
    observer.rpc_failure(
        Operation::WorkerCancellation,
        &tonic::Status::cancelled("cancellation delivery failed"),
        PluginPhase::Runtime,
    );
    let view = context
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(view.lifetime.totals.failures_total.value.get(), 1);
    assert_eq!(view.events[0].code(), "plugin.worker.cancellation_failed");
}

#[test]
fn trust_classification_does_not_guess_io_or_authentication_from_messages() {
    use super::super::DynamicPluginTrustFailure as E;
    let path = std::path::PathBuf::from("restricted-artifact");
    for (error, expected) in [
        (E::MissingArtifact, FailureClass::InvalidInput),
        (E::MissingIntegrityDigest, FailureClass::InvalidInput),
        (E::MissingSignature, FailureClass::InvalidInput),
        (E::MissingTrustedKeys, FailureClass::InvalidInput),
        (
            E::ArtifactRead {
                path: path.clone(),
                error: "signature invalid".into(),
            },
            FailureClass::Unknown,
        ),
        (
            E::SignatureRead {
                path: path.clone(),
                error: "permission denied".into(),
            },
            FailureClass::Unknown,
        ),
        (
            E::IntegrityMismatch {
                path: path.clone(),
                expected: "expected".into(),
                actual: "observed".into(),
            },
            FailureClass::Authentication,
        ),
        (
            E::SignatureVerification {
                path,
                parse_errors: vec!["missing file".into()],
            },
            FailureClass::Authentication,
        ),
    ] {
        let value = TrustFailureClassifier.classify(&error);
        assert_eq!(value.code(), "plugin.artifact_verification_failed");
        assert_eq!(value.failure_class(), Some(expected.as_str()));
    }
}
