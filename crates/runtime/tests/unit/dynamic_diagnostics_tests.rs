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

fn plugin_diagnostic_context(settings: DiagnosticsSettings) -> Diagnostics {
    Diagnostics::from_settings(
        SourceIdentity {
            instance_id: Uuid::now_v7(),
            process_id: None,
            process_role: "test".into(),
            relay_version: "test".into(),
        },
        settings,
    )
    .unwrap()
}

#[test]
fn plugin_observers_remain_inert_when_collection_or_registration_is_unavailable() {
    for mode in ["disabled", "components", "counters"] {
        let mut settings = DiagnosticsSettings {
            enabled: mode != "disabled",
            ..Default::default()
        };
        match mode {
            "components" => settings.limits.max_components = 1,
            "counters" => settings.limits.max_counters = 1,
            _ => {}
        }
        let context = plugin_diagnostic_context(settings);
        // Keep the occupied registration alive: it cannot be reclaimed for a plugin.
        let _unrelated = (mode != "disabled").then(|| {
            context
                .register_component(ComponentRegistration::new(
                    "application.untouched",
                    &[BuiltinDiagnosticCode::Success(
                        SuccessCode::LoggingInitialized,
                    )],
                ))
                .unwrap()
        });
        let before = context.memory_usage();
        for (execution, operation, success) in [
            (
                PluginExecution::NativeAbi,
                Operation::NativeCallback,
                SuccessCode::PluginNativeLoaded,
            ),
            (
                PluginExecution::WorkerProtocol,
                Operation::WorkerCallback,
                SuccessCode::PluginWorkerStarted,
            ),
        ] {
            let observer = PluginDiagnostics::new(&context, "example", execution, Uuid::now_v7());
            observer.failure(operation, FailureClass::Unknown, PluginPhase::Runtime);
            observer.success(success, PluginPhase::Activation);
        }
        let after = context.memory_usage();
        assert_eq!(after.reserved_bytes, before.reserved_bytes, "{mode}");
        assert_eq!(after.components, before.components, "{mode}");
        assert_eq!(after.counters, before.counters, "{mode}");
        if mode == "disabled" {
            assert!(matches!(
                context.snapshot(&DiagnosticQuery::default()),
                Err(crate::diagnostics::DiagnosticError::Disabled)
            ));
            assert_eq!(after.reserved_bytes, 0);
            continue;
        }
        assert_unavailable_plugin_observer_snapshot(&context, mode);
    }
}

fn assert_unavailable_plugin_observer_snapshot(context: &Diagnostics, mode: &str) {
    let view = context
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(view.components.len(), 1);
    assert_eq!(view.components[0].identity.key, "application.untouched");
    assert_eq!(view.lifetime.totals.failures_total.value.get(), 0);
    assert_eq!(view.lifetime.totals.successes_total.value.get(), 0);
    assert!(view.events.is_empty());
    assert_eq!(view.coverage.loss.components_rejected.value.get(), 2);
    assert_eq!(
        view.coverage.loss.keys_rejected.value.get(),
        if mode == "counters" { 2 } else { 0 }
    );
    assert_eq!(view.coverage.loss.observations_rejected.value.get(), 0);
}

#[test]
fn replacement_plugin_observers_preserve_separate_instance_attribution() {
    for (execution, operation, success) in [
        (
            PluginExecution::NativeAbi,
            Operation::NativeCallback,
            SuccessCode::PluginNativeLoaded,
        ),
        (
            PluginExecution::WorkerProtocol,
            Operation::WorkerCallback,
            SuccessCode::PluginWorkerStarted,
        ),
    ] {
        let context = plugin_diagnostic_context(DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        });
        let old_instance = Uuid::now_v7();
        let new_instance = Uuid::now_v7();
        let old = PluginDiagnostics::new(&context, "same.plugin", execution, old_instance);
        let in_flight = old.clone();
        old.failure(operation, FailureClass::Unknown, PluginPhase::Runtime);
        let replacement = PluginDiagnostics::new(&context, "same.plugin", execution, new_instance);
        replacement.success(success, PluginPhase::Activation);
        // Work already in flight retains the old owner's identity after replacement.
        in_flight.failure(operation, FailureClass::Unknown, PluginPhase::Runtime);
        replacement.failure(operation, FailureClass::Unknown, PluginPhase::Runtime);
        drop(old);
        drop(in_flight);
        replacement.failure(operation, FailureClass::Unknown, PluginPhase::Runtime);
        let view = context
            .snapshot(&DiagnosticQuery::default())
            .unwrap()
            .to_snapshot()
            .unwrap();
        assert_eq!(view.components.len(), 2);
        let component_for = |instance| {
            view.components
                .iter()
                .find(|row| row.since_refresh.ownership.plugin_instance_id == Some(instance))
                .unwrap()
        };
        let old_component = component_for(old_instance);
        let new_component = component_for(new_instance);
        assert_eq!(old_component.identity.key, new_component.identity.key);
        assert_ne!(old_component.identity, new_component.identity);
        assert_eq!(old_component.lifetime.totals.failures_total.value.get(), 2);
        assert_eq!(old_component.lifetime.totals.successes_total.value.get(), 0);
        assert_eq!(new_component.lifetime.totals.failures_total.value.get(), 2);
        assert_eq!(new_component.lifetime.totals.successes_total.value.get(), 1);
        assert_eq!(view.lifetime.totals.failures_total.value.get(), 4);
        assert_eq!(view.lifetime.totals.successes_total.value.get(), 1);
        assert_eq!(view.events.len(), 5);
        assert_replacement_plugin_events(
            &view,
            old_component,
            new_component,
            (old_instance, new_instance),
            execution,
            success,
        );
    }
}

fn assert_replacement_plugin_events(
    view: &crate::diagnostics::DiagnosticSnapshot,
    old_component: &crate::diagnostics::ComponentSnapshot,
    new_component: &crate::diagnostics::ComponentSnapshot,
    instances: (Uuid, Uuid),
    execution: PluginExecution,
    success: SuccessCode,
) {
    let (old_instance, new_instance) = instances;
    for event in &view.events {
        let expected_owner = if event.component == old_component.identity {
            old_instance
        } else {
            assert_eq!(event.component, new_component.identity);
            new_instance
        };
        assert_eq!(
            event.origin.ownership.plugin_instance_id,
            Some(expected_owner)
        );
        assert_eq!(
            event.context.execution.as_ref().unwrap().as_str(),
            execution.as_str()
        );
        assert_eq!(
            event.context.phase.as_ref().unwrap().as_str(),
            if event.code() == success.canonical_name() {
                "activation"
            } else {
                "runtime"
            }
        );
    }
}
