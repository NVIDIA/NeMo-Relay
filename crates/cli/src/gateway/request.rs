// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gateway request validation, buffering, and normalized LLM start construction.

use std::borrow::Cow;
use std::error::Error;
use std::io::Read;

use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, Method, Request, header};
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
    let request_json = request_body_for_observability(
        &body_bytes,
        &parts.headers,
        config.max_passthrough_body_bytes,
    )
    .and_then(|body| serde_json::from_slice::<Value>(&body).ok())
    .unwrap_or(Value::Null);
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

// Decodes the transport body only for Relay's managed request representation. The original bytes
// and Content-Encoding header remain on PreparedGatewayRequest so unsupported or malformed
// encodings still pass through unchanged. When the managed pipeline reserializes decoded JSON,
// effective_dispatch_request removes Content-Encoding from the identity-encoded upstream body.
fn request_body_for_observability<'a>(
    body: &'a [u8],
    headers: &HeaderMap,
    max_decoded_bytes: usize,
) -> Option<Cow<'a, [u8]>> {
    let mut encodings = Vec::new();
    for value in headers.get_all(header::CONTENT_ENCODING) {
        for encoding in value.to_str().ok()?.split(',') {
            let encoding = encoding.trim();
            if encoding.is_empty() {
                return None;
            }
            encodings.push(encoding.to_ascii_lowercase());
        }
    }
    if encodings.is_empty() {
        return Some(Cow::Borrowed(body));
    }

    let mut decoded = Cow::Borrowed(body);
    for encoding in encodings.iter().rev() {
        match encoding.as_str() {
            "identity" => {}
            "zstd" => decoded = Cow::Owned(decode_zstd(&decoded, max_decoded_bytes)?),
            _ => return None,
        }
    }
    (decoded.len() <= max_decoded_bytes).then_some(decoded)
}

fn decode_zstd(body: &[u8], max_decoded_bytes: usize) -> Option<Vec<u8>> {
    let mut decoder = zstd::stream::read::Decoder::new(body).ok()?;
    decoder
        .window_log_max(zstd_window_log_max(max_decoded_bytes))
        .ok()?;
    let limit = u64::try_from(max_decoded_bytes)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let mut decoded = Vec::new();
    decoder.take(limit).read_to_end(&mut decoded).ok()?;
    (decoded.len() <= max_decoded_bytes).then_some(decoded)
}

pub(super) fn zstd_window_log_max(max_decoded_bytes: usize) -> u32 {
    const ZSTD_WINDOW_LOG_MIN: u32 = 10;
    const ZSTD_WINDOW_LOG_MAX: u32 = if usize::BITS == 32 { 30 } else { 31 };

    let required_log = usize::BITS - max_decoded_bytes.saturating_sub(1).leading_zeros();
    required_log.clamp(ZSTD_WINDOW_LOG_MIN, ZSTD_WINDOW_LOG_MAX)
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
