// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Producer-owned classification and state-only operation observations.

use crate::diagnostics::{
    BuiltinDiagnosticCode, Classification, ComponentRecorder, ComponentRegistration,
    DiagnosticClassifier, Diagnostics, FailureClass, FailureCode, Observation, PluginExecution,
    PluginFailureCode, PluginObservationContext, PluginOperationFailureCode as Operation,
    PluginPhase, SuccessCode,
};
use nemo_relay_plugin::NemoRelayStatus;
use std::num::NonZeroU64;
use uuid::Uuid;

#[derive(Clone)]
pub(super) struct PluginDiagnostics {
    recorder: Option<ComponentRecorder>,
    execution: PluginExecution,
}
impl Default for PluginDiagnostics {
    fn default() -> Self {
        Self {
            recorder: None,
            execution: PluginExecution::NativeAbi,
        }
    }
}
impl PluginDiagnostics {
    pub(super) fn new(
        context: &Diagnostics,
        plugin: &str,
        execution: PluginExecution,
        instance: Uuid,
    ) -> Self {
        if context.reader().availability() != crate::diagnostics::DiagnosticAvailability::Available
        {
            return Self {
                recorder: None,
                execution,
            };
        }
        let codes = BuiltinDiagnosticCode::all()
            .iter()
            .copied()
            .filter(|code| match code {
                BuiltinDiagnosticCode::Failure(FailureCode::Plugin(
                    PluginFailureCode::Operation(operation),
                )) => match execution {
                    PluginExecution::NativeAbi => matches!(
                        operation,
                        Operation::ArtifactVerification
                            | Operation::Policy
                            | Operation::SchemaValidation
                            | Operation::NativeLoad
                            | Operation::NativeValidation
                            | Operation::NativeRegistration
                            | Operation::NativeCallback
                    ),
                    PluginExecution::WorkerProtocol => !matches!(
                        operation,
                        Operation::NativeLoad
                            | Operation::NativeValidation
                            | Operation::NativeRegistration
                            | Operation::NativeCallback
                    ),
                    _ => false,
                },
                BuiltinDiagnosticCode::Success(code) => match execution {
                    PluginExecution::NativeAbi => matches!(
                        code,
                        SuccessCode::PluginNativeLoaded
                            | SuccessCode::PluginNativeValidated
                            | SuccessCode::PluginNativeRegistered
                    ),
                    PluginExecution::WorkerProtocol => matches!(
                        code,
                        SuccessCode::PluginWorkerStarted
                            | SuccessCode::PluginWorkerHandshakeCompleted
                            | SuccessCode::PluginWorkerValidated
                            | SuccessCode::PluginWorkerRegistered
                            | SuccessCode::PluginWorkerProcessCleaned
                    ),
                    _ => false,
                },
                _ => false,
            })
            .collect::<Vec<_>>();
        let key = format!("plugin.{}.{}", execution.as_str(), plugin);
        let mut end = key.len().min(context.limits().max_identity_bytes);
        while !key.is_char_boundary(end) {
            end -= 1;
        }
        let mut registration = ComponentRegistration::new(&key[..end], &codes);
        registration.ownership.plugin_instance_id = Some(instance);
        Self {
            recorder: context.register_component(registration).ok(),
            execution,
        }
    }
    pub(super) fn failure(&self, operation: Operation, class: FailureClass, phase: PluginPhase) {
        self.classified_failure(classification(operation, class), phase);
    }
    pub(super) fn classified_failure(&self, classification: Classification, phase: PluginPhase) {
        if let Some(recorder) = &self.recorder {
            let _ = recorder.record_without_projection_lazy(|| {
                Observation::new(
                    classification,
                    "A dynamic plugin operation failed",
                    NonZeroU64::MIN,
                )
                .with_plugin_context(PluginObservationContext {
                    execution: Some(self.execution.into()),
                    phase: Some(phase.into()),
                })
            });
        }
    }
    pub(super) fn success(&self, code: SuccessCode, phase: PluginPhase) {
        if let Some(recorder) = &self.recorder {
            let _ = recorder.record_without_projection_lazy(|| {
                Observation::new(
                    Classification::success(code),
                    "A dynamic plugin operation succeeded",
                    NonZeroU64::MIN,
                )
                .with_plugin_context(PluginObservationContext {
                    execution: Some(self.execution.into()),
                    phase: Some(phase.into()),
                })
            });
        }
    }
    pub(super) fn native_status(
        &self,
        operation: Operation,
        status: NemoRelayStatus,
        phase: PluginPhase,
    ) {
        if let Some(class) = native_failure(status) {
            self.failure(operation, class, phase);
        }
    }
    #[cfg(feature = "worker-grpc")]
    pub(super) fn rpc_failure(
        &self,
        operation: Operation,
        status: &tonic::Status,
        phase: PluginPhase,
    ) {
        if status.code() == tonic::Code::Cancelled && operation == Operation::WorkerTransport {
            return; // Caller-owned cancellation is not a transport failure.
        }
        let classification = WorkerStatusClassifier(operation).classify(status);
        self.classified_failure(classification, phase);
    }
}

fn classification(operation: Operation, class: FailureClass) -> Classification {
    Classification::failure(
        FailureCode::Plugin(PluginFailureCode::Operation(operation)),
        class,
    )
    .expect("supported plugin operation class")
}

/// Pure native status interpretation. Completion and backpressure are not failures.
pub(super) fn native_failure(status: NemoRelayStatus) -> Option<FailureClass> {
    Some(match status {
        NemoRelayStatus::Ok | NemoRelayStatus::StreamEnd | NemoRelayStatus::Backpressured => {
            return None;
        }
        NemoRelayStatus::AlreadyExists => FailureClass::Conflict,
        NemoRelayStatus::NotFound => FailureClass::NotFound,
        NemoRelayStatus::ScopeStackEmpty => FailureClass::InvalidState,
        NemoRelayStatus::GuardrailRejected => FailureClass::PolicyRejected,
        NemoRelayStatus::Internal => FailureClass::Internal,
        NemoRelayStatus::NullPointer
        | NemoRelayStatus::InvalidJson
        | NemoRelayStatus::InvalidUtf8
        | NemoRelayStatus::InvalidArg => FailureClass::InvalidInput,
    })
}

pub(super) struct TrustFailureClassifier;
impl DiagnosticClassifier<super::DynamicPluginTrustFailure> for TrustFailureClassifier {
    fn classify(&self, error: &super::DynamicPluginTrustFailure) -> Classification {
        use super::DynamicPluginTrustFailure as E;
        classification(
            Operation::ArtifactVerification,
            match error {
                E::MissingArtifact
                | E::MissingIntegrityDigest
                | E::MissingSignature
                | E::MissingTrustedKeys => FailureClass::InvalidInput,
                // Safe-file admission also rejects links, kinds, and excessive size.
                E::ArtifactRead { .. } => FailureClass::Unknown,
                // SignatureRead combines reading and decoding; do not infer from text.
                E::SignatureRead { .. } => FailureClass::Unknown,
                E::IntegrityMismatch { .. } | E::SignatureVerification { .. } => {
                    FailureClass::Authentication
                }
            },
        )
    }
}
pub(super) struct PolicyFailureClassifier;
impl DiagnosticClassifier<super::DynamicPluginHostPolicyFailure> for PolicyFailureClassifier {
    fn classify(&self, _: &super::DynamicPluginHostPolicyFailure) -> Classification {
        classification(Operation::Policy, FailureClass::PolicyRejected)
    }
}
#[cfg(feature = "worker-grpc")]
struct WorkerStatusClassifier(Operation);
#[cfg(feature = "worker-grpc")]
impl DiagnosticClassifier<tonic::Status> for WorkerStatusClassifier {
    fn classify(&self, error: &tonic::Status) -> Classification {
        use tonic::Code;
        classification(
            self.0,
            match error.code() {
                Code::InvalidArgument | Code::OutOfRange => FailureClass::InvalidInput,
                Code::DeadlineExceeded => FailureClass::Timeout,
                Code::Unavailable => FailureClass::Unavailable,
                Code::Unauthenticated => FailureClass::Authentication,
                Code::PermissionDenied => FailureClass::PermissionDenied,
                Code::ResourceExhausted => FailureClass::ResourceExhausted,
                Code::Unimplemented => FailureClass::Unsupported,
                Code::NotFound => FailureClass::NotFound,
                Code::AlreadyExists => FailureClass::Conflict,
                Code::Internal | Code::DataLoss => FailureClass::Internal,
                Code::FailedPrecondition => FailureClass::InvalidState,
                _ => FailureClass::Unknown,
            },
        )
    }
}

pub(super) fn execution(kind: super::DynamicPluginKind) -> PluginExecution {
    match kind {
        super::DynamicPluginKind::RustDynamic => PluginExecution::NativeAbi,
        super::DynamicPluginKind::Worker => PluginExecution::WorkerProtocol,
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/dynamic_diagnostics_tests.rs"]
mod tests;
