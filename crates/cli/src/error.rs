// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use nemo_relay::error::{FlowError, UpstreamFailure};
use serde::Serialize;
use serde_json::{Map, Value, json};
use strum::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub(crate) enum PluginLifecycleFailureKind {
    Failed,
    NotFound,
    Refused,
}

pub(crate) type PluginLifecycleErrorContext<'a> = (
    &'static str,
    Option<&'a str>,
    PluginLifecycleFailureKind,
    Option<&'static str>,
    &'a str,
);

/// An allowlisted operational reason for an MCP command failure.
///
/// Values are intentionally static: source errors can contain local paths, endpoints, or
/// configuration values and must remain on stderr rather than becoming operational log fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub(crate) enum McpFailureReason {
    GenerationCaptureFailed,
    GatewayConfigurationFailed,
    HeartbeatConfigurationFailed,
    GatewayAcquisitionFailed,
    StdinReaderStartFailed,
    GatewayHeartbeatTaskFailed,
    GenerationLifecycleInvalid,
    GenerationLifecycleInvalidDuringRecovery,
    GatewayLifecycleVerificationTaskFailed,
    GatewayRecoveryTaskFailed,
    GatewayRecoveryFailed,
    GatewayLeaseClosedDuringRecovery,
    GatewayRecoveredThenUnhealthy,
    GatewayRecoveredThenReplaced,
    GatewayMonitorTaskFailed,
    TransparentGatewayInitialVerificationFailed,
    TransparentGatewayHeartbeatVerificationFailed,
    TransparentGatewayUnavailable,
    TransparentGatewayReplaced,
    StdinFrameReadFailed,
    McpResponseSerializationFailed,
    StdoutWriteFailed,
    StdoutFlushFailed,
    UnknownMcpFailure,
}

impl McpFailureReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::GenerationCaptureFailed => "generation_capture_failed",
            Self::GatewayConfigurationFailed => "gateway_configuration_failed",
            Self::HeartbeatConfigurationFailed => "heartbeat_configuration_failed",
            Self::GatewayAcquisitionFailed => "gateway_acquisition_failed",
            Self::StdinReaderStartFailed => "stdin_reader_start_failed",
            Self::GatewayHeartbeatTaskFailed => "gateway_heartbeat_task_failed",
            Self::GenerationLifecycleInvalid => "generation_lifecycle_invalid",
            Self::GenerationLifecycleInvalidDuringRecovery => {
                "generation_lifecycle_invalid_during_recovery"
            }
            Self::GatewayLifecycleVerificationTaskFailed => {
                "gateway_lifecycle_verification_task_failed"
            }
            Self::GatewayRecoveryTaskFailed => "gateway_recovery_task_failed",
            Self::GatewayRecoveryFailed => "gateway_recovery_failed",
            Self::GatewayLeaseClosedDuringRecovery => "gateway_lease_closed_during_recovery",
            Self::GatewayRecoveredThenUnhealthy => "gateway_recovered_then_unhealthy",
            Self::GatewayRecoveredThenReplaced => "gateway_recovered_then_replaced",
            Self::GatewayMonitorTaskFailed => "gateway_monitor_task_failed",
            Self::TransparentGatewayInitialVerificationFailed => {
                "transparent_gateway_initial_verification_failed"
            }
            Self::TransparentGatewayHeartbeatVerificationFailed => {
                "transparent_gateway_heartbeat_verification_failed"
            }
            Self::TransparentGatewayUnavailable => "transparent_gateway_unavailable",
            Self::TransparentGatewayReplaced => "transparent_gateway_replaced",
            Self::StdinFrameReadFailed => "stdin_frame_read_failed",
            Self::McpResponseSerializationFailed => "mcp_response_serialization_failed",
            Self::StdoutWriteFailed => "stdout_write_failed",
            Self::StdoutFlushFailed => "stdout_flush_failed",
            Self::UnknownMcpFailure => "unknown_mcp_failure",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum CliError {
    #[error("guardrail rejected: {0}")]
    GuardrailRejected(String),
    #[error("invalid hook payload: {0}")]
    InvalidPayload(String),
    #[error("payload too large: {0}")]
    PayloadTooLarge(String),
    #[error("unauthorized gateway client: {0}")]
    Unauthorized(String),
    #[error("gateway upstream error: {0}")]
    Upstream(#[from] reqwest::Error),
    #[error("{0}")]
    ProviderFailure(UpstreamFailure),
    #[error("http error: {0}")]
    Http(#[from] http::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("installer error: {0}")]
    Install(String),
    #[error("configuration error: {0}")]
    Config(String),
    #[error("launcher error: {0}")]
    Launch(String),
    #[error("{source}")]
    McpFailure {
        reason: McpFailureReason,
        #[source]
        source: Box<CliError>,
    },
    #[error("nemo-relay hook forward failed: {source}")]
    HookDelivery {
        #[source]
        source: Box<CliError>,
    },
    #[error("{message}")]
    PluginLifecycle {
        command: &'static str,
        target: Option<String>,
        kind: PluginLifecycleFailureKind,
        code: Option<&'static str>,
        message: String,
    },
    #[error("NeMo Relay runtime error: {0}")]
    Flow(#[from] nemo_relay::error::FlowError),
}

impl CliError {
    pub(crate) fn with_mcp_failure_reason(self, reason: McpFailureReason) -> Self {
        match self {
            Self::McpFailure { .. } => self,
            source => Self::McpFailure {
                reason,
                source: Box::new(source),
            },
        }
    }

    pub(crate) fn mcp_failure_reason(&self) -> Option<McpFailureReason> {
        match self {
            Self::McpFailure { reason, .. } => Some(*reason),
            _ => None,
        }
    }

    pub(crate) fn log_kind(&self) -> &'static str {
        match self {
            Self::GuardrailRejected(_) => "guardrail_rejected",
            Self::InvalidPayload(_) => "invalid_payload",
            Self::PayloadTooLarge(_) => "payload_too_large",
            Self::Unauthorized(_) => "unauthorized",
            Self::Upstream(_) => "upstream",
            Self::ProviderFailure(_) => "provider_failure",
            Self::Http(_) => "http",
            Self::Io(_) => "io",
            Self::Install(_) => "install",
            Self::Config(_) => "configuration",
            Self::Launch(_) => "launch",
            Self::McpFailure { source, .. } => source.log_kind(),
            Self::HookDelivery { source } => source.log_kind(),
            Self::PluginLifecycle { .. } => "plugin_lifecycle",
            Self::Flow(FlowError::GuardrailRejected(_)) => "guardrail_rejected",
            Self::Flow(_) => "runtime",
        }
    }

    pub(crate) fn guardrail_rejection_reason(&self) -> Option<&str> {
        match self {
            Self::GuardrailRejected(reason) => Some(reason),
            Self::Flow(FlowError::GuardrailRejected(reason)) => Some(reason),
            Self::McpFailure { source, .. } => source.guardrail_rejection_reason(),
            Self::HookDelivery { source } => source.guardrail_rejection_reason(),
            _ => None,
        }
    }

    pub(crate) fn as_plugin_lifecycle_error_context(
        &self,
    ) -> Option<PluginLifecycleErrorContext<'_>> {
        match self {
            Self::PluginLifecycle {
                command,
                target,
                kind,
                code,
                message,
            } => Some((command, target.as_deref(), *kind, *code, message.as_str())),
            Self::McpFailure { source, .. } => source.as_plugin_lifecycle_error_context(),
            _ => None,
        }
    }
}

impl IntoResponse for CliError {
    // Maps gateway errors into a compact JSON HTTP response. Bad hook payloads are client errors,
    // network-level upstream failures are bad gateway responses, provider failures mirror the
    // upstream status when available, and local install/config/runtime faults remain internal
    // errors so callers do not mistake them for agent policy decisions.
    fn into_response(self) -> Response {
        let message = self.to_string();
        let guardrail_reason = self.guardrail_rejection_reason().map(ToOwned::to_owned);
        let status = if guardrail_reason.is_some() {
            StatusCode::FORBIDDEN
        } else {
            response_status(&self)
        };
        let error_type = if guardrail_reason.is_some() {
            "nemo_relay_guardrail_rejected"
        } else {
            "nemo_relay_gateway_error"
        };
        let mut error = Map::from_iter([
            ("message".to_string(), json!(message)),
            ("type".to_string(), json!(error_type)),
        ]);
        if let Some(reason) = guardrail_reason {
            error.insert("reason".to_string(), json!(reason));
        }
        let body = Json(json!({
            "error": Value::Object(error)
        }));
        (status, body).into_response()
    }
}

fn response_status(error: &CliError) -> StatusCode {
    match error {
        CliError::PayloadTooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
        CliError::Unauthorized(_) => StatusCode::UNAUTHORIZED,
        CliError::InvalidPayload(_) => StatusCode::BAD_REQUEST,
        CliError::Upstream(_) => StatusCode::BAD_GATEWAY,
        CliError::ProviderFailure(failure) => failure
            .status
            .and_then(|status| StatusCode::from_u16(status).ok())
            .unwrap_or(StatusCode::BAD_GATEWAY),
        CliError::McpFailure { source, .. } => response_status(source),
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

#[cfg(test)]
#[path = "../tests/coverage/shared/error_tests.rs"]
mod tests;
