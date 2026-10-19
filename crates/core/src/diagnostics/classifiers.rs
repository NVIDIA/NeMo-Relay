// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure classifiers implemented where Relay still owns structured source errors.
//!
//! These adapters never record, format source errors, infer causes from messages,
//! alter retry/cancellation behavior, or assign application health. Producers call
//! them inside lazy recording factories so disabled diagnostics does no extra work.

use nemo_relay_plugin::NemoRelayStatus;
use nemo_relay_types::diagnostics::{
    BindingFailureCode, Classification, CommonPluginFailureCode, DiagnosticClassifier,
    FailureClass, FailureCode, NativePluginFailureCode, NoticeCode, PluginFailureCode, PluginPhase,
    RuntimeFailureCode, UpstreamFailureCode,
};

use crate::error::{FlowError, UpstreamFailure, UpstreamFailureClass};
use crate::plugin::{
    DynamicPluginFailurePhase, DynamicPluginHostPolicyFailure, DynamicPluginTrustFailure,
    PluginError,
};

fn failure(code: FailureCode, class: FailureClass) -> Classification {
    // Every pair below is an explicit entry in the shared catalogue. This is a
    // programmer invariant, not validation of untrusted producer input.
    Classification::failure(code, class).expect("classifier uses a supported catalogue pair")
}

fn common_plugin(code: CommonPluginFailureCode, class: FailureClass) -> Classification {
    failure(FailureCode::Plugin(PluginFailureCode::Common(code)), class)
}

/// Fallback mapping of core runtime errors while preserving their execution meaning.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlowErrorClassifier;

impl DiagnosticClassifier<FlowError> for FlowErrorClassifier {
    fn classify(&self, error: &FlowError) -> Classification {
        match error {
            FlowError::AlreadyExists(_) => failure(
                FailureCode::Runtime(RuntimeFailureCode::AlreadyExists),
                FailureClass::Conflict,
            ),
            FlowError::NotFound(_) => failure(
                FailureCode::Runtime(RuntimeFailureCode::NotFound),
                FailureClass::NotFound,
            ),
            FlowError::InvalidArgument(_) => failure(
                FailureCode::Runtime(RuntimeFailureCode::InvalidArgument),
                FailureClass::InvalidInput,
            ),
            FlowError::ScopeStackEmpty => failure(
                FailureCode::Runtime(RuntimeFailureCode::ScopeStackEmpty),
                FailureClass::InvalidState,
            ),
            FlowError::GuardrailRejected(_) => {
                Classification::notice(NoticeCode::GuardrailRejected)
            }
            FlowError::Upstream(error) => UpstreamFailureClassifier.classify(error),
            FlowError::Internal(_) => failure(
                FailureCode::Runtime(RuntimeFailureCode::Internal),
                FailureClass::Internal,
            ),
            FlowError::CallbackException { .. } => failure(
                FailureCode::Binding(BindingFailureCode::CallbackException),
                FailureClass::ExecutionFailed,
            ),
        }
    }
}

/// Structured upstream mapping; HTTP bodies and retry eligibility are not causes.
#[derive(Debug, Clone, Copy, Default)]
pub struct UpstreamFailureClassifier;

impl DiagnosticClassifier<UpstreamFailure> for UpstreamFailureClassifier {
    fn classify(&self, error: &UpstreamFailure) -> Classification {
        self.classify(&error.class)
    }
}

impl DiagnosticClassifier<UpstreamFailureClass> for UpstreamFailureClassifier {
    fn classify(&self, class: &UpstreamFailureClass) -> Classification {
        let (code, class) = match class {
            UpstreamFailureClass::Connection => (
                UpstreamFailureCode::ConnectionFailed,
                FailureClass::Unavailable,
            ),
            UpstreamFailureClass::Timeout => (UpstreamFailureCode::TimedOut, FailureClass::Timeout),
            UpstreamFailureClass::RetryableStatus => {
                (UpstreamFailureCode::StatusRejected, FailureClass::Unknown)
            }
            UpstreamFailureClass::ContextWindow => (
                UpstreamFailureCode::ContextLimit,
                FailureClass::ResourceExhausted,
            ),
            UpstreamFailureClass::ModelUnavailable => (
                UpstreamFailureCode::ModelUnavailable,
                FailureClass::Unavailable,
            ),
            UpstreamFailureClass::Authentication => (
                UpstreamFailureCode::AuthenticationFailed,
                FailureClass::Authentication,
            ),
            UpstreamFailureClass::InvalidRequest => (
                UpstreamFailureCode::RequestInvalid,
                FailureClass::InvalidInput,
            ),
            UpstreamFailureClass::Other => (UpstreamFailureCode::Failed, FailureClass::Unknown),
        };
        failure(FailureCode::Upstream(code), class)
    }
}

/// Plugin-operation mapping before errors are flattened into reports or SDK strings.
#[derive(Debug, Clone, Copy, Default)]
pub struct PluginErrorClassifier;

impl DiagnosticClassifier<PluginError> for PluginErrorClassifier {
    fn classify(&self, error: &PluginError) -> Classification {
        let (code, class) = match error {
            PluginError::InvalidConfig(_) => (
                CommonPluginFailureCode::ConfigurationInvalid,
                FailureClass::InvalidInput,
            ),
            PluginError::Conflict(_) => (
                CommonPluginFailureCode::StateConflict,
                FailureClass::Conflict,
            ),
            PluginError::NotFound(_) => (
                CommonPluginFailureCode::ResourceNotFound,
                FailureClass::NotFound,
            ),
            PluginError::Serialization(_) => (
                CommonPluginFailureCode::SerializationFailed,
                FailureClass::Unknown,
            ),
            PluginError::Internal(_) => (CommonPluginFailureCode::Internal, FailureClass::Internal),
            PluginError::RegistrationFailed(_) => (
                CommonPluginFailureCode::RegistrationFailed,
                FailureClass::Unknown,
            ),
        };
        common_plugin(code, class)
    }
}

/// Trust-check mapping based on the original typed failure, never attestation text.
#[derive(Debug, Clone, Copy, Default)]
pub struct DynamicPluginTrustFailureClassifier;

impl DiagnosticClassifier<DynamicPluginTrustFailure> for DynamicPluginTrustFailureClassifier {
    fn classify(&self, error: &DynamicPluginTrustFailure) -> Classification {
        let (code, class) = match error {
            DynamicPluginTrustFailure::MissingArtifact
            | DynamicPluginTrustFailure::MissingIntegrityDigest
            | DynamicPluginTrustFailure::IntegrityMismatch { .. } => (
                CommonPluginFailureCode::IntegrityCheckFailed,
                FailureClass::InvalidInput,
            ),
            DynamicPluginTrustFailure::ArtifactRead { .. } => (
                CommonPluginFailureCode::IntegrityCheckFailed,
                FailureClass::Io,
            ),
            DynamicPluginTrustFailure::MissingSignature
            | DynamicPluginTrustFailure::MissingTrustedKeys => (
                CommonPluginFailureCode::AttestationFailed,
                FailureClass::InvalidInput,
            ),
            DynamicPluginTrustFailure::SignatureRead { .. }
            | DynamicPluginTrustFailure::SignatureVerification { .. } => (
                CommonPluginFailureCode::AttestationFailed,
                FailureClass::Unknown,
            ),
        };
        common_plugin(code, class)
    }
}

/// Existing dynamic-plugin policy refusal; classification never applies the policy.
#[derive(Debug, Clone, Copy, Default)]
pub struct DynamicPluginHostPolicyFailureClassifier;

impl DiagnosticClassifier<DynamicPluginHostPolicyFailure>
    for DynamicPluginHostPolicyFailureClassifier
{
    fn classify(&self, error: &DynamicPluginHostPolicyFailure) -> Classification {
        match error {
            DynamicPluginHostPolicyFailure::Blocked => common_plugin(
                CommonPluginFailureCode::PolicyRejected,
                FailureClass::PolicyRejected,
            ),
        }
    }
}

/// Native ABI status mapping, preserving normal stream and backpressure notices.
///
/// `Ok` returns no observation: ABI helper success is not counted implicitly.
/// Callers at an explicitly instrumented operation may record its own success.
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeStatusClassifier;

impl NativeStatusClassifier {
    /// Interpret an already validated native status without altering its meaning.
    #[must_use]
    pub fn classify(&self, status: &NemoRelayStatus) -> Option<Classification> {
        let class = match status {
            NemoRelayStatus::Ok => return None,
            NemoRelayStatus::GuardrailRejected => {
                return Some(Classification::notice(NoticeCode::GuardrailRejected));
            }
            NemoRelayStatus::StreamEnd => {
                return Some(Classification::notice(NoticeCode::NativeStreamEnded));
            }
            NemoRelayStatus::Backpressured => {
                return Some(Classification::notice(
                    NoticeCode::NativeStreamBackpressured,
                ));
            }
            NemoRelayStatus::AlreadyExists => FailureClass::Conflict,
            NemoRelayStatus::NotFound => FailureClass::NotFound,
            NemoRelayStatus::ScopeStackEmpty => FailureClass::InvalidState,
            NemoRelayStatus::Internal => FailureClass::Internal,
            NemoRelayStatus::NullPointer
            | NemoRelayStatus::InvalidJson
            | NemoRelayStatus::InvalidUtf8
            | NemoRelayStatus::InvalidArg => FailureClass::InvalidInput,
        };
        Some(failure(
            FailureCode::Plugin(PluginFailureCode::Native(
                NativePluginFailureCode::AbiCallFailed,
            )),
            class,
        ))
    }

    /// Validate an integer status before constructing an enum or classifying it.
    ///
    /// Unknown raw values are errors at the ABI boundary, not an invalid Rust enum.
    pub fn classify_raw(&self, raw: i32) -> Result<Option<Classification>, UnknownNativeStatus> {
        let status = match raw {
            0 => NemoRelayStatus::Ok,
            1 => NemoRelayStatus::AlreadyExists,
            2 => NemoRelayStatus::NotFound,
            3 => NemoRelayStatus::ScopeStackEmpty,
            4 => NemoRelayStatus::GuardrailRejected,
            5 => NemoRelayStatus::Internal,
            6 => NemoRelayStatus::NullPointer,
            7 => NemoRelayStatus::InvalidJson,
            8 => NemoRelayStatus::InvalidUtf8,
            9 => NemoRelayStatus::InvalidArg,
            10 => NemoRelayStatus::StreamEnd,
            11 => NemoRelayStatus::Backpressured,
            _ => return Err(UnknownNativeStatus(raw)),
        };
        Ok(self.classify(&status))
    }
}

/// An integer outside the supported native ABI execution-status catalogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unknown native ABI status {0}")]
pub struct UnknownNativeStatus(pub i32);

/// Convert an existing dynamic-plugin phase into portable observation context.
#[must_use]
pub fn plugin_phase(phase: DynamicPluginFailurePhase) -> PluginPhase {
    match phase {
        DynamicPluginFailurePhase::Validation => PluginPhase::Validation,
        DynamicPluginFailurePhase::Activation => PluginPhase::Activation,
        DynamicPluginFailurePhase::Runtime => PluginPhase::Runtime,
        DynamicPluginFailurePhase::Policy => PluginPhase::Policy,
    }
}
