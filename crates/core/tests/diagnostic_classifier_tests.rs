// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Producer classifiers preserve structured evidence without changing execution.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use nemo_relay::diagnostics::{
    Classification, DiagnosticClassifier, DynamicPluginHostPolicyFailureClassifier,
    DynamicPluginTrustFailureClassifier, FailureClass, FlowErrorClassifier, NativeStatusClassifier,
    PluginErrorClassifier, PluginPhase, UpstreamFailureClassifier, plugin_phase,
};
use nemo_relay::error::{FlowError, UpstreamFailure, UpstreamFailureClass};
use nemo_relay::plugin::{
    DynamicPluginFailurePhase, DynamicPluginHostPolicyFailure, DynamicPluginTrustFailure,
    PluginError,
};
use nemo_relay_plugin::NemoRelayStatus;

fn assert_classification(actual: Classification, code: &str, class: Option<FailureClass>) {
    assert_eq!(actual.code().canonical_name(), code);
    assert_eq!(actual.failure_class(), class);
    assert!(actual.code().allows_class(actual.failure_class()));
}

#[test]
fn runtime_error_variants_use_supported_causes_and_guardrail_rejection_stays_a_notice() {
    let cases = [
        (
            FlowError::AlreadyExists("timeout".into()),
            "runtime.already_exists",
            Some(FailureClass::Conflict),
        ),
        (
            FlowError::NotFound("permission denied".into()),
            "runtime.not_found",
            Some(FailureClass::NotFound),
        ),
        (
            FlowError::InvalidArgument("internal failure".into()),
            "runtime.invalid_argument",
            Some(FailureClass::InvalidInput),
        ),
        (
            FlowError::ScopeStackEmpty,
            "runtime.scope_stack_empty",
            Some(FailureClass::InvalidState),
        ),
        (
            FlowError::GuardrailRejected("the configured decision was reject".into()),
            "runtime.guardrail_rejected",
            None,
        ),
        (
            FlowError::Internal("timeout: retry this".into()),
            "runtime.internal",
            Some(FailureClass::Internal),
        ),
    ];
    for (error, code, class) in cases {
        assert_classification(FlowErrorClassifier.classify(&error), code, class);
    }
}

#[test]
fn binding_callback_classification_does_not_format_or_invoke_the_source_exception() {
    #[derive(Debug)]
    struct NoFormatting;
    impl fmt::Display for NoFormatting {
        fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
            panic!("classification must not format the binding source");
        }
    }
    impl std::error::Error for NoFormatting {}
    let error = FlowError::CallbackException {
        message: "callback failed".into(),
        exception_type: "TimeoutError".into(),
        source: Some(Arc::new(NoFormatting)),
    };
    // The exception class name alone is not a supported typed timeout contract.
    assert_classification(
        FlowErrorClassifier.classify(&error),
        "binding.callback_exception",
        Some(FailureClass::ExecutionFailed),
    );
}

#[test]
fn upstream_classifiers_use_structured_causes_without_reinterpreting_retry_policy() {
    let cases = [
        (
            UpstreamFailureClass::Connection,
            "upstream.connection_failed",
            FailureClass::Unavailable,
        ),
        (
            UpstreamFailureClass::Timeout,
            "upstream.timed_out",
            FailureClass::Timeout,
        ),
        (
            UpstreamFailureClass::RetryableStatus,
            "upstream.status_rejected",
            FailureClass::Unknown,
        ),
        (
            UpstreamFailureClass::ContextWindow,
            "upstream.context_limit",
            FailureClass::ResourceExhausted,
        ),
        (
            UpstreamFailureClass::ModelUnavailable,
            "upstream.model_unavailable",
            FailureClass::Unavailable,
        ),
        (
            UpstreamFailureClass::Authentication,
            "upstream.authentication_failed",
            FailureClass::Authentication,
        ),
        (
            UpstreamFailureClass::InvalidRequest,
            "upstream.request_invalid",
            FailureClass::InvalidInput,
        ),
        (
            UpstreamFailureClass::Other,
            "upstream.failed",
            FailureClass::Unknown,
        ),
    ];
    for (source_class, code, class) in cases {
        let error = UpstreamFailure {
            status: Some(401),
            body: "timeout; permission denied; invalid request".into(),
            headers: BTreeMap::new(),
            class: source_class,
        };
        let retryable_before = error.is_retryable();
        let direct = UpstreamFailureClassifier.classify(&error);
        assert_classification(direct, code, Some(class));
        assert_eq!(UpstreamFailureClassifier.classify(&source_class), direct);
        assert_eq!(
            FlowErrorClassifier.classify(&FlowError::Upstream(error.clone())),
            direct
        );
        assert_eq!(error.is_retryable(), retryable_before);
        assert_eq!(error.status, Some(401));
        assert_eq!(error.body, "timeout; permission denied; invalid request");
    }
    let retryable = UpstreamFailure {
        status: Some(503),
        body: String::new(),
        headers: BTreeMap::new(),
        class: UpstreamFailureClass::RetryableStatus,
    };
    assert!(retryable.is_retryable());
    assert_eq!(
        UpstreamFailureClassifier
            .classify(&retryable)
            .failure_class(),
        Some(FailureClass::Unknown)
    );
}

#[test]
fn plugin_error_classifiers_do_not_infer_a_cause_from_flattened_strings() {
    let json_error = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
    let cases = [
        (
            PluginError::InvalidConfig("timeout".into()),
            "plugin.configuration_invalid",
            FailureClass::InvalidInput,
        ),
        (
            PluginError::Conflict("not found".into()),
            "plugin.state_conflict",
            FailureClass::Conflict,
        ),
        (
            PluginError::NotFound("access denied".into()),
            "plugin.resource_not_found",
            FailureClass::NotFound,
        ),
        (
            PluginError::Serialization(json_error),
            "plugin.serialization_failed",
            FailureClass::Unknown,
        ),
        (
            PluginError::Internal("permission denied".into()),
            "plugin.internal",
            FailureClass::Internal,
        ),
        (
            PluginError::RegistrationFailed("duplicate conflict: invalid config".into()),
            "plugin.registration_failed",
            FailureClass::Unknown,
        ),
    ];
    for (error, code, class) in cases {
        assert_classification(PluginErrorClassifier.classify(&error), code, Some(class));
    }
}

#[test]
fn trust_classifiers_keep_typed_artifact_and_signature_evidence_conservative() {
    let path = PathBuf::from("plugin-artifact");
    let cases = [
        (
            DynamicPluginTrustFailure::MissingArtifact,
            "plugin.integrity_check_failed",
            FailureClass::InvalidInput,
        ),
        (
            DynamicPluginTrustFailure::MissingIntegrityDigest,
            "plugin.integrity_check_failed",
            FailureClass::InvalidInput,
        ),
        (
            DynamicPluginTrustFailure::ArtifactRead {
                path: path.clone(),
                error: "permission denied: timeout".into(),
            },
            "plugin.integrity_check_failed",
            FailureClass::Io,
        ),
        (
            DynamicPluginTrustFailure::IntegrityMismatch {
                path: path.clone(),
                expected: "a".into(),
                actual: "b".into(),
            },
            "plugin.integrity_check_failed",
            FailureClass::InvalidInput,
        ),
        (
            DynamicPluginTrustFailure::MissingSignature,
            "plugin.attestation_failed",
            FailureClass::InvalidInput,
        ),
        (
            DynamicPluginTrustFailure::MissingTrustedKeys,
            "plugin.attestation_failed",
            FailureClass::InvalidInput,
        ),
        (
            DynamicPluginTrustFailure::SignatureRead {
                path: path.clone(),
                error: "permission denied".into(),
            },
            "plugin.attestation_failed",
            FailureClass::Unknown,
        ),
        (
            DynamicPluginTrustFailure::SignatureVerification {
                path,
                parse_errors: vec!["unsupported key".into()],
            },
            "plugin.attestation_failed",
            FailureClass::Unknown,
        ),
    ];
    for (error, code, class) in cases {
        assert_classification(
            DynamicPluginTrustFailureClassifier.classify(&error),
            code,
            Some(class),
        );
    }
    assert_classification(
        DynamicPluginHostPolicyFailureClassifier.classify(&DynamicPluginHostPolicyFailure::Blocked),
        "plugin.policy_rejected",
        Some(FailureClass::PolicyRejected),
    );
}

#[test]
fn native_status_classification_preserves_completion_and_backpressure_contracts() {
    assert!(
        NativeStatusClassifier
            .classify(&NemoRelayStatus::Ok)
            .is_none()
    );
    assert!(NativeStatusClassifier.classify_raw(0).unwrap().is_none());
    for (raw, code) in [
        (4, "runtime.guardrail_rejected"),
        (10, "plugin.native.stream_ended"),
        (11, "plugin.native.stream_backpressured"),
    ] {
        assert_classification(
            NativeStatusClassifier.classify_raw(raw).unwrap().unwrap(),
            code,
            None,
        );
    }
    for (raw, class) in [
        (1, FailureClass::Conflict),
        (2, FailureClass::NotFound),
        (3, FailureClass::InvalidState),
        (5, FailureClass::Internal),
        (6, FailureClass::InvalidInput),
        (7, FailureClass::InvalidInput),
        (8, FailureClass::InvalidInput),
        (9, FailureClass::InvalidInput),
    ] {
        assert_classification(
            NativeStatusClassifier.classify_raw(raw).unwrap().unwrap(),
            "plugin.native.abi_call_failed",
            Some(class),
        );
    }
    for raw in [-1, 12, i32::MIN, i32::MAX] {
        assert_eq!(NativeStatusClassifier.classify_raw(raw).unwrap_err().0, raw);
    }
}

#[test]
fn plugin_phase_conversion_retains_context_without_adding_a_code_dimension() {
    for (source, target) in [
        (
            DynamicPluginFailurePhase::Validation,
            PluginPhase::Validation,
        ),
        (
            DynamicPluginFailurePhase::Activation,
            PluginPhase::Activation,
        ),
        (DynamicPluginFailurePhase::Runtime, PluginPhase::Runtime),
        (DynamicPluginFailurePhase::Policy, PluginPhase::Policy),
    ] {
        assert_eq!(plugin_phase(source), target);
    }
}
