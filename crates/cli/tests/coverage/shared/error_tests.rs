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
            CliError::HookHttpFailure {
                status: StatusCode::FORBIDDEN,
                message: "denied".into(),
            },
            "hook_http",
        ),
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

#[test]
fn http_error_statuses_survive_runtime_and_delivery_wrappers() {
    for status in 400..=599 {
        for variant in 0..4 {
            let failure = UpstreamFailure {
                status: Some(status),
                body: "backend error".into(),
                headers: BTreeMap::new(),
                class: UpstreamFailureClass::Other,
            };
            let error = match variant {
                0 => CliError::ProviderFailure(failure),
                1 => CliError::Flow(FlowError::Upstream(failure)),
                2 => CliError::HookDelivery {
                    source: Box::new(CliError::Flow(FlowError::Upstream(failure))),
                },
                _ => CliError::HookDelivery {
                    source: Box::new(CliError::McpFailure {
                        reason: McpFailureReason::UnknownMcpFailure,
                        source: Box::new(CliError::ProviderFailure(failure)),
                    }),
                },
            };
            assert_eq!(error.into_response().status().as_u16(), status);
        }
    }
    for status in [None, Some(0), Some(1000)] {
        let error = CliError::ProviderFailure(UpstreamFailure {
            status,
            body: "invalid status".into(),
            headers: BTreeMap::new(),
            class: UpstreamFailureClass::Other,
        });
        assert_eq!(error.into_response().status(), StatusCode::BAD_GATEWAY);
    }
}

#[tokio::test]
async fn upstream_transport_errors_return_bad_gateway_without_hook_payload_prefix() {
    use crate::daemon::common::transport::TransportError;
    use http_body_util::BodyExt;

    let error = CliError::UpstreamTransport(TransportError::InsecureDestination);
    assert_eq!(error.log_kind(), "upstream");
    let response = error.into_response();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        body["error"]["message"],
        "upstream forwarding error: plain HTTP upstreams are allowed only for loopback hosts; use https"
    );
}
