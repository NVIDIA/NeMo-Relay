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
    #[error("openinference error: {0}")]
    OpenInference(#[from] nemo_relay::observability::openinference::OpenInferenceError),
}

impl CliError {
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
            Self::HookDelivery { source } => source.log_kind(),
            Self::PluginLifecycle { .. } => "plugin_lifecycle",
            Self::Flow(FlowError::GuardrailRejected(_)) => "guardrail_rejected",
            Self::Flow(_) => "runtime",
            Self::OpenInference(_) => "openinference",
        }
    }

    pub(crate) fn guardrail_rejection_reason(&self) -> Option<&str> {
        match self {
            Self::GuardrailRejected(reason) => Some(reason),
            Self::Flow(FlowError::GuardrailRejected(reason)) => Some(reason),
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
        let status = match (guardrail_reason.is_some(), &self) {
            (true, _) => StatusCode::FORBIDDEN,
            (false, Self::PayloadTooLarge(_)) => StatusCode::PAYLOAD_TOO_LARGE,
            (false, Self::Unauthorized(_)) => StatusCode::UNAUTHORIZED,
            (false, Self::InvalidPayload(_)) => StatusCode::BAD_REQUEST,
            (false, Self::Upstream(_)) => StatusCode::BAD_GATEWAY,
            (false, Self::ProviderFailure(failure)) => failure
                .status
                .and_then(|status| StatusCode::from_u16(status).ok())
                .unwrap_or(StatusCode::BAD_GATEWAY),
            (
                false,
                Self::Http(_)
                | Self::Io(_)
                | Self::Install(_)
                | Self::Config(_)
                | Self::Launch(_)
                | Self::Flow(_)
                | Self::OpenInference(_),
            ) => StatusCode::INTERNAL_SERVER_ERROR,
            (false, _) => StatusCode::INTERNAL_SERVER_ERROR,
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

#[cfg(test)]
#[path = "../tests/coverage/shared/error_tests.rs"]
mod tests;
