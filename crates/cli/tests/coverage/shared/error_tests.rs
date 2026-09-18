// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use nemo_relay::error::{FlowError, UpstreamFailure, UpstreamFailureClass};

use super::*;

#[test]
fn log_kinds_cover_every_operational_error_class() {
    let upstream = reqwest::Client::new()
        .get("not a valid URL")
        .build()
        .expect_err("invalid URL should produce a reqwest error");
    let http = http::Request::builder()
        .uri("not a valid URI")
        .body(())
        .expect_err("invalid URI should produce an HTTP error");
    let errors = [
        (
            CliError::GuardrailRejected("blocked".into()),
            "guardrail_rejected",
        ),
        (
            CliError::InvalidPayload("invalid".into()),
            "invalid_payload",
        ),
        (
            CliError::PayloadTooLarge("large".into()),
            "payload_too_large",
        ),
        (
            CliError::Unauthorized("missing token".into()),
            "unauthorized",
        ),
        (CliError::Upstream(upstream), "upstream"),
        (
            CliError::ProviderFailure(UpstreamFailure {
                status: Some(503),
                body: "unavailable".into(),
                headers: BTreeMap::new(),
                class: UpstreamFailureClass::ModelUnavailable,
            }),
            "provider_failure",
        ),
        (CliError::Http(http), "http"),
        (CliError::Io(std::io::Error::other("io")), "io"),
        (CliError::Install("install".into()), "install"),
        (CliError::Config("config".into()), "configuration"),
        (CliError::Launch("launch".into()), "launch"),
        (
            CliError::Launch("launch".into())
                .with_mcp_failure_reason(McpFailureReason::GatewayRecoveryFailed),
            "launch",
        ),
        (
            CliError::HookDelivery {
                source: Box::new(CliError::Install("hook transport".into())),
            },
            "install",
        ),
        (
            CliError::PluginLifecycle {
                command: "add",
                target: None,
                kind: PluginLifecycleFailureKind::Failed,
                code: None,
                message: "plugin".into(),
            },
            "plugin_lifecycle",
        ),
        (
            CliError::Flow(FlowError::Internal("runtime".into())),
            "runtime",
        ),
    ];

    for (error, expected) in errors {
        assert_eq!(error.log_kind(), expected);
    }
}

#[test]
fn mcp_failure_reasons_are_static_and_preserve_source_errors() {
    let source = CliError::Launch("secret local detail".into());
    let error = source.with_mcp_failure_reason(McpFailureReason::GatewayRecoveryFailed);

    assert_eq!(
        error.mcp_failure_reason(),
        Some(McpFailureReason::GatewayRecoveryFailed)
    );
    assert_eq!(error.log_kind(), "launch");
    assert_eq!(error.to_string(), "launcher error: secret local detail");

    for reason in [
        McpFailureReason::GenerationCaptureFailed,
        McpFailureReason::GatewayConfigurationFailed,
        McpFailureReason::HeartbeatConfigurationFailed,
        McpFailureReason::GatewayAcquisitionFailed,
        McpFailureReason::StdinReaderStartFailed,
        McpFailureReason::GatewayHeartbeatTaskFailed,
        McpFailureReason::GenerationLifecycleInvalid,
        McpFailureReason::GenerationLifecycleInvalidDuringRecovery,
        McpFailureReason::GatewayLifecycleVerificationTaskFailed,
        McpFailureReason::GatewayRecoveryTaskFailed,
        McpFailureReason::GatewayRecoveryFailed,
        McpFailureReason::GatewayLeaseClosedDuringRecovery,
        McpFailureReason::GatewayRecoveredThenUnhealthy,
        McpFailureReason::GatewayRecoveredThenReplaced,
        McpFailureReason::GatewayMonitorTaskFailed,
        McpFailureReason::TransparentGatewayInitialVerificationFailed,
        McpFailureReason::TransparentGatewayHeartbeatVerificationFailed,
        McpFailureReason::TransparentGatewayUnavailable,
        McpFailureReason::TransparentGatewayReplaced,
        McpFailureReason::StdinFrameReadFailed,
        McpFailureReason::McpResponseSerializationFailed,
        McpFailureReason::StdoutWriteFailed,
        McpFailureReason::StdoutFlushFailed,
        McpFailureReason::UnknownMcpFailure,
    ] {
        assert!(
            reason
                .as_str()
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'_'),
            "{} must be snake case",
            reason.as_str()
        );
    }
}
