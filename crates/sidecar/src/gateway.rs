// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, Method, Request, Response, StatusCode};
use futures_util::StreamExt;
use nemo_flow::api::llm::LlmRequest;
use serde_json::{Map, Value, json};

use crate::config::header_string;
use crate::error::SidecarError;
use crate::server::AppState;
use crate::session::LlmGatewayStart;

pub(crate) async fn passthrough(
    State(state): State<AppState>,
    request: Request<Body>,
) -> Result<Response<Body>, SidecarError> {
    let (parts, body) = request.into_parts();
    let provider = ProviderRoute::from_path(parts.uri.path()).ok_or_else(|| {
        SidecarError::InvalidPayload(format!("unsupported gateway path {}", parts.uri.path()))
    })?;
    let body_bytes = axum::body::to_bytes(body, usize::MAX)
        .await
        .map_err(|error| SidecarError::InvalidPayload(error.to_string()))?;
    let request_json = serde_json::from_slice::<Value>(&body_bytes).unwrap_or(Value::Null);
    let upstream_url = provider.upstream_url(
        &state.config,
        parts
            .uri
            .path_and_query()
            .map(|p| p.as_str())
            .unwrap_or(parts.uri.path()),
    );
    let streaming = request_json
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let session_id = gateway_session_id(&parts.headers);
    let llm_request = LlmRequest {
        headers: observable_headers(&parts.headers),
        content: request_json.clone(),
    };
    let active = state
        .sessions
        .start_llm(
            &parts.headers,
            LlmGatewayStart {
                session_id,
                provider: provider.name().to_string(),
                model_name: request_json
                    .get("model")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                request: llm_request,
                streaming,
                metadata: json!({ "gateway_path": parts.uri.path() }),
            },
        )
        .await?;

    let mut upstream = state
        .http
        .request(parts.method.clone(), upstream_url)
        .body(body_bytes.clone());
    for (name, value) in &parts.headers {
        if should_forward_request_header(name) {
            upstream = upstream.header(name, value);
        }
    }
    let upstream_response = match upstream.send().await {
        Ok(response) => response,
        Err(error) => {
            state
                .sessions
                .end_llm(
                    active,
                    json!({ "error": error.to_string() }),
                    json!({ "gateway_error": true, "stage": "send" }),
                )
                .await?;
            return Err(SidecarError::Upstream(error));
        }
    };
    let status = upstream_response.status();
    let headers = response_headers(upstream_response.headers());
    let content_type = upstream_response
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let is_stream = streaming || content_type.contains("text/event-stream");

    if is_stream {
        let sessions = state.sessions.clone();
        let stream = upstream_response.bytes_stream();
        let body = Body::from_stream(async_stream::stream! {
            let mut stream = stream;
            let mut active = Some(active);
            let mut collected = Vec::new();
            let mut truncated = false;
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        if collected.len() + bytes.len() <= 1_048_576 {
                            collected.extend_from_slice(&bytes);
                        } else {
                            truncated = true;
                        }
                        yield Ok::<Bytes, reqwest::Error>(bytes);
                    }
                    Err(error) => {
                        if let Some(active) = active.take() {
                            let _ = sessions
                                .end_llm(
                                    active,
                                    json!({ "error": error.to_string() }),
                                    json!({ "http_status": status.as_u16(), "streaming": true, "gateway_error": true, "stage": "stream" }),
                                )
                                .await;
                        }
                        yield Err(error);
                        return;
                    }
                }
            }
            let response = stream_response_json(&collected, truncated);
            if let Some(active) = active.take() {
                let _ = sessions
                    .end_llm(
                        active,
                        response,
                        json!({ "http_status": status.as_u16(), "streaming": true, "stream_truncated": truncated }),
                    )
                    .await;
            }
        });
        return build_response(status, headers, body);
    }

    let bytes = match upstream_response.bytes().await {
        Ok(bytes) => bytes,
        Err(error) => {
            state
                .sessions
                .end_llm(
                    active,
                    json!({ "error": error.to_string() }),
                    json!({ "http_status": status.as_u16(), "streaming": false, "gateway_error": true, "stage": "body" }),
                )
                .await?;
            return Err(SidecarError::Upstream(error));
        }
    };
    let response_json = serde_json::from_slice::<Value>(&bytes)
        .unwrap_or_else(|_| json!({ "body_bytes": bytes.len() }));
    state
        .sessions
        .end_llm(
            active,
            response_json,
            json!({ "http_status": status.as_u16(), "streaming": false }),
        )
        .await?;
    build_response(status, headers, Body::from(bytes))
}

pub(crate) async fn models(
    State(state): State<AppState>,
    request: Request<Body>,
) -> Result<Response<Body>, SidecarError> {
    let (parts, _body) = request.into_parts();
    if parts.method != Method::GET {
        return build_response(
            StatusCode::METHOD_NOT_ALLOWED,
            HeaderMap::new(),
            Body::empty(),
        );
    }
    let provider = ProviderRoute::OpenAiModels;
    let upstream_url = provider.upstream_url(
        &state.config,
        parts
            .uri
            .path_and_query()
            .map(|p| p.as_str())
            .unwrap_or(parts.uri.path()),
    );
    let mut upstream = state.http.get(upstream_url);
    for (name, value) in &parts.headers {
        if should_forward_request_header(name) {
            upstream = upstream.header(name, value);
        }
    }
    let upstream_response = upstream.send().await?;
    let status = upstream_response.status();
    let headers = response_headers(upstream_response.headers());
    let bytes = upstream_response.bytes().await?;
    build_response(status, headers, Body::from(bytes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderRoute {
    OpenAiResponses,
    OpenAiChatCompletions,
    OpenAiModels,
    AnthropicMessages,
    AnthropicCountTokens,
}

impl ProviderRoute {
    fn from_path(path: &str) -> Option<Self> {
        match path {
            "/v1/responses" => Some(Self::OpenAiResponses),
            "/v1/chat/completions" => Some(Self::OpenAiChatCompletions),
            "/v1/models" => Some(Self::OpenAiModels),
            "/v1/messages" => Some(Self::AnthropicMessages),
            "/v1/messages/count_tokens" => Some(Self::AnthropicCountTokens),
            _ => None,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::OpenAiResponses => "openai.responses",
            Self::OpenAiChatCompletions => "openai.chat_completions",
            Self::OpenAiModels => "openai.models",
            Self::AnthropicMessages => "anthropic.messages",
            Self::AnthropicCountTokens => "anthropic.count_tokens",
        }
    }

    fn upstream_url(self, config: &crate::config::SidecarConfig, path_and_query: &str) -> String {
        let base = match self {
            Self::OpenAiResponses | Self::OpenAiChatCompletions | Self::OpenAiModels => {
                config.openai_base_url.trim_end_matches('/')
            }
            Self::AnthropicMessages | Self::AnthropicCountTokens => {
                config.anthropic_base_url.trim_end_matches('/')
            }
        };
        format!("{base}{path_and_query}")
    }
}

fn gateway_session_id(headers: &HeaderMap) -> Option<String> {
    header_string(headers, "x-nemo-flow-session-id")
        .or_else(|| header_string(headers, "x-claude-code-session-id"))
}

fn observable_headers(headers: &HeaderMap) -> Map<String, Value> {
    let mut output = Map::new();
    for (name, value) in headers {
        if should_record_header(name)
            && let Ok(value) = value.to_str()
        {
            output.insert(name.as_str().to_string(), json!(value));
        }
    }
    output
}

fn response_headers(headers: &HeaderMap) -> HeaderMap {
    let mut output = HeaderMap::new();
    for (name, value) in headers {
        if !is_hop_by_hop(name) {
            output.insert(name.clone(), value.clone());
        }
    }
    output
}

fn build_response(
    status: StatusCode,
    headers: HeaderMap,
    body: Body,
) -> Result<Response<Body>, SidecarError> {
    let mut builder = Response::builder().status(status);
    for (name, value) in headers {
        if let Some(name) = name {
            builder = builder.header(name, value);
        }
    }
    Ok(builder.body(body)?)
}

fn should_forward_request_header(name: &HeaderName) -> bool {
    !is_hop_by_hop(name) && name != http::header::HOST && name != http::header::CONTENT_LENGTH
}

fn should_record_header(name: &HeaderName) -> bool {
    should_forward_request_header(name)
        && name != http::header::AUTHORIZATION
        && name.as_str() != "x-api-key"
        && name.as_str() != "anthropic-api-key"
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

fn stream_response_json(collected: &[u8], truncated: bool) -> Value {
    if truncated {
        return json!({
            "stream_preview": String::from_utf8_lossy(collected),
            "stream_truncated": true
        });
    }
    json!({ "stream": String::from_utf8_lossy(collected) })
}

#[cfg(test)]
#[path = "../tests/coverage/gateway_tests.rs"]
mod tests;
