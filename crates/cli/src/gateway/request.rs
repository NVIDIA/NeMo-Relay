// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gateway request validation, buffering, and normalized LLM start construction.

use std::error::Error;

use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, Method, Request};
use http_body_util::LengthLimitError;
use nemo_relay::api::llm::LlmRequest;
use serde_json::{Value, json};

use crate::configuration::BOOTSTRAP_CLIENT_TOKEN_HEADER;
use crate::error::CliError;
use crate::sessions::LlmGatewayStart;

use super::response::observable_headers;
use super::routes::{
    ProviderRoute, gateway_identifier, gateway_session_id, gateway_subagent_id,
    gateway_upstream_url_override,
};

pub(super) struct PreparedGatewayRequest {
    pub(super) method: Method,
    pub(super) headers: HeaderMap,
    pub(super) path: String,
    pub(super) provider: ProviderRoute,
    pub(super) upstream_url: String,
    pub(super) body_bytes: Bytes,
    pub(super) request_json: Value,
    pub(super) streaming: bool,
    pub(super) authorization: crate::provider_auth::ProviderRequestAuthorization,
}

pub(super) async fn prepare_gateway_request(
    config: &crate::configuration::GatewayConfig,
    request: Request<Body>,
    mut authorization: crate::provider_auth::ProviderRequestAuthorization,
) -> Result<PreparedGatewayRequest, CliError> {
    let (mut parts, body) = request.into_parts();
    parts.headers.remove(BOOTSTRAP_CLIENT_TOKEN_HEADER);
    let provider = ProviderRoute::from_path(parts.uri.path()).ok_or_else(|| {
        CliError::InvalidPayload(format!("unsupported gateway path {}", parts.uri.path()))
    })?;
    let body_bytes = axum::body::to_bytes(body, config.max_passthrough_body_bytes)
        .await
        .map_err(passthrough_body_error)?;
    let request_json = serde_json::from_slice::<Value>(&body_bytes).unwrap_or(Value::Null);
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|path| path.as_str())
        .unwrap_or(parts.uri.path());
    let upstream_url = gateway_upstream_url_override(
        provider,
        &parts.headers,
        path_and_query,
        authorization.allow_environment_provider_auth,
        config,
    )
    .unwrap_or_else(|| provider.upstream_url(config, path_and_query));
    parts.headers = super::routes::strip_replaceable_agent_auth_headers(
        &parts.headers,
        provider,
        authorization.allow_environment_provider_auth,
        provider.configured_auth_header(config),
    );
    authorization.source_credential = authorization
        .source_credential
        .after_source_normalization(&parts.headers);
    let streaming = request_json
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok(PreparedGatewayRequest {
        method: parts.method,
        headers: parts.headers,
        path: parts.uri.path().to_string(),
        provider,
        upstream_url,
        body_bytes,
        request_json,
        streaming,
        authorization,
    })
}

fn passthrough_body_error(error: axum::Error) -> CliError {
    if error.source().is_some_and(|source| {
        source.is::<LengthLimitError>()
            || source
                .source()
                .is_some_and(|source| source.is::<LengthLimitError>())
    }) {
        CliError::PayloadTooLarge(error.to_string())
    } else {
        CliError::InvalidPayload(error.to_string())
    }
}

pub(super) fn build_llm_gateway_start(request: &PreparedGatewayRequest) -> LlmGatewayStart {
    LlmGatewayStart {
        session_id: gateway_session_id(&request.headers, &request.request_json, request.provider),
        provider: request.provider.name().to_string(),
        model_name: request
            .request_json
            .get("model")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        subagent_id: gateway_subagent_id(&request.headers, &request.request_json, request.provider),
        conversation_id: gateway_identifier(
            &request.headers,
            &request.request_json,
            "x-nemo-relay-conversation-id",
            &[
                &["conversation_id"],
                &["conversationId"],
                &["conversation", "id"],
            ],
        ),
        generation_id: gateway_identifier(
            &request.headers,
            &request.request_json,
            "x-nemo-relay-generation-id",
            &[&["generation_id"], &["generationId"], &["generation", "id"]],
        ),
        request_id: gateway_identifier(
            &request.headers,
            &request.request_json,
            "x-nemo-relay-request-id",
            &[
                &["request_id"],
                &["requestId"],
                &["request", "id"],
                &["metadata", "request_id"],
            ],
        )
        .or_else(|| crate::configuration::header_string(&request.headers, "x-request-id")),
        request: LlmRequest {
            headers: observable_headers(&request.headers),
            content: request.request_json.clone(),
        },
        streaming: request.streaming,
        metadata: json!({ "gateway_path": request.path }),
    }
}
