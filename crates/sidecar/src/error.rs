// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub(crate) enum SidecarError {
    #[error("invalid hook payload: {0}")]
    InvalidPayload(String),
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
    #[error("NeMo Flow runtime error: {0}")]
    Flow(#[from] nemo_flow::error::FlowError),
    #[error("openinference error: {0}")]
    OpenInference(#[from] nemo_flow::observability::openinference::OpenInferenceError),
}

impl IntoResponse for SidecarError {
    // Maps sidecar errors into a compact JSON HTTP response. Bad hook payloads are client errors,
    // upstream gateway failures are bad gateway responses, and local install/config/runtime faults
    // remain internal errors so callers do not mistake them for agent policy decisions.
    fn into_response(self) -> Response {
        let status = match self {
            Self::InvalidPayload(_) => StatusCode::BAD_REQUEST,
            Self::Upstream(_) => StatusCode::BAD_GATEWAY,
            Self::Http(_)
            | Self::Io(_)
            | Self::Install(_)
            | Self::Config(_)
            | Self::Launch(_)
            | Self::Flow(_)
            | Self::OpenInference(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = Json(json!({
            "error": {
                "message": self.to_string(),
                "type": "nemo_flow_sidecar_error"
            }
        }));
        (status, body).into_response()
    }
}
