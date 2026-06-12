// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use nemo_relay::error::FlowError;
use serde_json::{Map, Value, json};

#[derive(Debug, thiserror::Error)]
pub(crate) enum CliError {
    #[error("guardrail rejected: {0}")]
    GuardrailRejected(String),
    #[error("invalid hook payload: {0}")]
    InvalidPayload(String),
    #[error("payload too large: {0}")]
    PayloadTooLarge(String),
    #[error("gateway upstream error: {0}")]
    Upstream(#[from] reqwest::Error),
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
    #[error("NeMo Relay runtime error: {0}")]
    Flow(#[from] nemo_relay::error::FlowError),
    #[error("openinference error: {0}")]
    OpenInference(#[from] nemo_relay::observability::openinference::OpenInferenceError),
}

impl CliError {
    pub(crate) fn guardrail_rejection_reason(&self) -> Option<&str> {
        match self {
            Self::GuardrailRejected(reason) => Some(reason),
            Self::Flow(FlowError::GuardrailRejected(reason)) => Some(reason),
            _ => None,
        }
    }
}

impl IntoResponse for CliError {
    // Maps gateway errors into a compact JSON HTTP response. Bad hook payloads are client errors,
    // upstream gateway failures are bad gateway responses, and local install/config/runtime faults
    // remain internal errors so callers do not mistake them for agent policy decisions.
    fn into_response(self) -> Response {
        let message = self.to_string();
        let guardrail_reason = self.guardrail_rejection_reason().map(ToOwned::to_owned);
        let status = match (guardrail_reason.is_some(), self) {
            (true, _) => StatusCode::FORBIDDEN,
            (false, Self::PayloadTooLarge(_)) => StatusCode::PAYLOAD_TOO_LARGE,
            (false, Self::InvalidPayload(_)) => StatusCode::BAD_REQUEST,
            (false, Self::Upstream(_)) => StatusCode::BAD_GATEWAY,
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
