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
use crate::session::{ActiveLlm, LlmGatewayStart, SessionManager};

const MAX_BODY_BYTES: usize = 100 * 1024 * 1024;

/// Proxies supported LLM API requests while recording a NeMo Flow LLM call around the upstream work.
///
/// The gateway reads the full request body once so it can both forward exact bytes and derive
/// observable metadata. Upstream send/body failures close the active LLM with gateway-error
/// metadata before surfacing an HTTP error. Streaming responses are forwarded chunk-by-chunk while
/// collecting at most 1 MiB for the end event, so client-visible streaming is not delayed by
/// observability capture.
pub(crate) async fn passthrough(
    State(state): State<AppState>,
    request: Request<Body>,
) -> Result<Response<Body>, SidecarError> {
    let prepared = prepare_gateway_request(&state.config, request).await?;
    let active = start_gateway_llm(&state.sessions, &prepared).await?;
    let upstream_response = send_upstream_or_end(&state, &prepared, active.clone()).await?;
    let status = upstream_response.status();
    let headers = response_headers(upstream_response.headers());
    if is_stream_response(prepared.streaming, upstream_response.headers()) {
        return streaming_gateway_response(
            state.sessions,
            active,
            status,
            headers,
            upstream_response,
        );
    }
    buffered_gateway_response(state.sessions, active, status, headers, upstream_response).await
}

struct PreparedGatewayRequest {
    method: Method,
    headers: HeaderMap,
    path: String,
    provider: ProviderRoute,
    upstream_url: String,
    body_bytes: Bytes,
    request_json: Value,
    streaming: bool,
}

// Validates the gateway route, buffers the request body exactly once, and derives the metadata used
// for both upstream forwarding and NeMo Flow LLM start events. Provider JSON parse failures are not
// request failures because the gateway still forwards raw bytes unchanged.
async fn prepare_gateway_request(
    config: &crate::config::SidecarConfig,
    request: Request<Body>,
) -> Result<PreparedGatewayRequest, SidecarError> {
    let (parts, body) = request.into_parts();
    let provider = ProviderRoute::from_path(parts.uri.path()).ok_or_else(|| {
        SidecarError::InvalidPayload(format!("unsupported gateway path {}", parts.uri.path()))
    })?;
    let body_bytes = axum::body::to_bytes(body, MAX_BODY_BYTES)
        .await
        .map_err(|error| SidecarError::InvalidPayload(error.to_string()))?;
    let request_json = serde_json::from_slice::<Value>(&body_bytes).unwrap_or(Value::Null);
    let upstream_url = provider.upstream_url(
        config,
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
    Ok(PreparedGatewayRequest {
        method: parts.method,
        headers: parts.headers,
        path: parts.uri.path().to_string(),
        provider,
        upstream_url,
        body_bytes,
        request_json,
        streaming,
    })
}

// Starts the NeMo Flow LLM lifecycle for a prepared gateway request. Session and subagent
// correlation identifiers are read from headers first and then from provider body fields.
async fn start_gateway_llm(
    sessions: &SessionManager,
    request: &PreparedGatewayRequest,
) -> Result<ActiveLlm, SidecarError> {
    let llm_request = LlmRequest {
        headers: observable_headers(&request.headers),
        content: request.request_json.clone(),
    };
    sessions
        .start_llm(
            &request.headers,
            LlmGatewayStart {
                session_id: gateway_session_id(&request.headers),
                provider: request.provider.name().to_string(),
                model_name: request
                    .request_json
                    .get("model")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                subagent_id: gateway_subagent_id(&request.headers),
                conversation_id: gateway_identifier(
                    &request.headers,
                    &request.request_json,
                    "x-nemo-flow-conversation-id",
                    &[
                        &["conversation_id"],
                        &["conversationId"],
                        &["conversation", "id"],
                    ],
                ),
                generation_id: gateway_identifier(
                    &request.headers,
                    &request.request_json,
                    "x-nemo-flow-generation-id",
                    &[&["generation_id"], &["generationId"], &["generation", "id"]],
                ),
                request_id: gateway_identifier(
                    &request.headers,
                    &request.request_json,
                    "x-nemo-flow-request-id",
                    &[
                        &["request_id"],
                        &["requestId"],
                        &["request", "id"],
                        &["metadata", "request_id"],
                    ],
                )
                .or_else(|| header_string(&request.headers, "x-request-id")),
                request: llm_request,
                streaming: request.streaming,
                metadata: json!({ "gateway_path": request.path }),
            },
        )
        .await
}

// Builds and sends the upstream request, copying only safe request headers. Send failures close the
// active LLM immediately because no response path will later own that lifecycle.
async fn send_upstream_or_end(
    state: &AppState,
    request: &PreparedGatewayRequest,
    active: ActiveLlm,
) -> Result<reqwest::Response, SidecarError> {
    let mut upstream = state
        .http
        .request(request.method.clone(), request.upstream_url.clone())
        .body(request.body_bytes.clone());
    for (name, value) in &request.headers {
        if should_forward_request_header(name) {
            upstream = upstream.header(name, value);
        }
    }
    match upstream.send().await {
        Ok(response) => Ok(response),
        Err(error) => {
            state
                .sessions
                .end_llm(
                    active,
                    json!({ "error": error.to_string() }),
                    json!({ "gateway_error": true, "stage": "send" }),
                )
                .await?;
            Err(SidecarError::Upstream(error))
        }
    }
}

// Determines whether the response should be proxied as a stream. The explicit request `stream`
// flag wins, but upstream SSE content type is also respected for providers that infer streaming.
fn is_stream_response(request_streaming: bool, headers: &HeaderMap) -> bool {
    let content_type = headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    request_streaming || content_type.contains("text/event-stream")
}

// Builds a streaming response body that forwards chunks as they arrive while retaining a bounded
// preview for the LLM end event. Stream errors end the LLM with gateway-error metadata before the
// client sees the propagated stream error.
fn streaming_gateway_response(
    sessions: SessionManager,
    active: ActiveLlm,
    status: StatusCode,
    headers: HeaderMap,
    upstream_response: reqwest::Response,
) -> Result<Response<Body>, SidecarError> {
    let stream = upstream_response.bytes_stream();
    let body = Body::from_stream(async_stream::stream! {
        let mut stream = stream;
        let mut llm = StreamingLlmGuard::new(sessions, active, status);
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
                    llm.end_error("stream", error.to_string()).await;
                    yield Err(error);
                    return;
                }
            }
        }
        let response = stream_response_json(&collected, truncated);
        llm.end_success(response, truncated).await;
    });
    build_response(status, headers, body)
}

// Buffers a non-streaming upstream response, records its JSON body or byte count, and then returns
// the original bytes to the client. Body read errors close the LLM before surfacing upstream error.
async fn buffered_gateway_response(
    sessions: SessionManager,
    active: ActiveLlm,
    status: StatusCode,
    headers: HeaderMap,
    upstream_response: reqwest::Response,
) -> Result<Response<Body>, SidecarError> {
    let bytes = match upstream_response.bytes().await {
        Ok(bytes) => bytes,
        Err(error) => {
            sessions
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
    sessions
        .end_llm(
            active,
            response_json,
            json!({ "http_status": status.as_u16(), "streaming": false }),
        )
        .await?;
    build_response(status, headers, Body::from(bytes))
}

struct StreamingLlmGuard {
    sessions: SessionManager,
    active: Option<ActiveLlm>,
    status: StatusCode,
}

impl StreamingLlmGuard {
    // Creates a guard that owns the active LLM until a stream reaches exactly one terminal path.
    // The option prevents double-ending when success, stream error, or drop cleanup races with
    // normal control flow.
    fn new(sessions: SessionManager, active: ActiveLlm, status: StatusCode) -> Self {
        Self {
            sessions,
            active: Some(active),
            status,
        }
    }

    // Ends a completed streaming LLM with the collected stream preview and truncation marker.
    // Errors from the observability layer are swallowed because the response body has already been
    // delivered to the client and the sidecar must not retroactively fail the stream.
    async fn end_success(&mut self, response: Value, truncated: bool) {
        if let Some(active) = self.active.take() {
            let _ = self
                .sessions
                .end_llm(
                    active,
                    response,
                    json!({ "http_status": self.status.as_u16(), "streaming": true, "stream_truncated": truncated }),
                )
                .await;
        }
    }

    // Ends a streaming LLM after an upstream stream error. The stage is preserved in metadata so
    // observers can distinguish mid-body failures from client drops or initial send failures.
    async fn end_error(&mut self, stage: &'static str, error: String) {
        if let Some(active) = self.active.take() {
            let _ = self
                .sessions
                .end_llm(
                    active,
                    json!({ "error": error }),
                    json!({ "http_status": self.status.as_u16(), "streaming": true, "gateway_error": true, "stage": stage }),
                )
                .await;
        }
    }
}

impl Drop for StreamingLlmGuard {
    // Best-effort cleanup for streams abandoned before success or error handling runs. Drop cannot
    // block, so it spawns onto the current Tokio runtime when one is available and otherwise leaves
    // cleanup to process shutdown.
    fn drop(&mut self) {
        let Some(active) = self.active.take() else {
            return;
        };
        let sessions = self.sessions.clone();
        let status = self.status;
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _ = sessions
                    .end_llm(
                        active,
                        json!({ "error": "stream body dropped before completion" }),
                        json!({ "http_status": status.as_u16(), "streaming": true, "gateway_error": true, "stage": "client_drop" }),
                    )
                    .await;
            });
        }
    }
}

/// Proxies OpenAI model-list requests without creating LLM runtime events.
///
/// The route is registered as GET-only but still verifies the method so direct tests or future
/// router changes return a 405 instead of forwarding a nonsensical request upstream.
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
    // Maps public sidecar paths to known upstream provider routes. Unsupported paths return `None`
    // so the caller can fail as a bad hook/gateway payload instead of constructing arbitrary URLs.
    fn from_path(path: &str) -> Option<Self> {
        match path {
            "/responses" => Some(Self::OpenAiResponses),
            "/v1/responses" => Some(Self::OpenAiResponses),
            "/chat/completions" => Some(Self::OpenAiChatCompletions),
            "/v1/chat/completions" => Some(Self::OpenAiChatCompletions),
            "/models" => Some(Self::OpenAiModels),
            "/v1/models" => Some(Self::OpenAiModels),
            "/v1/messages" => Some(Self::AnthropicMessages),
            "/v1/messages/count_tokens" => Some(Self::AnthropicCountTokens),
            _ => None,
        }
    }

    // Returns the provider route name recorded in LLM event metadata. These names split OpenAI API
    // variants because their request/response schemas differ even when they share a base URL.
    const fn name(self) -> &'static str {
        match self {
            Self::OpenAiResponses => "openai.responses",
            Self::OpenAiChatCompletions => "openai.chat_completions",
            Self::OpenAiModels => "openai.models",
            Self::AnthropicMessages => "anthropic.messages",
            Self::AnthropicCountTokens => "anthropic.count_tokens",
        }
    }

    // Builds the upstream URL by combining the configured provider base with the original path and
    // query string. Trailing slashes are stripped from the base to avoid double-slash variants in
    // configured enterprise or local proxy endpoints.
    fn upstream_url(self, config: &crate::config::SidecarConfig, path_and_query: &str) -> String {
        let base = match self {
            Self::OpenAiResponses | Self::OpenAiChatCompletions | Self::OpenAiModels => {
                config.openai_base_url.trim_end_matches('/')
            }
            Self::AnthropicMessages | Self::AnthropicCountTokens => {
                config.anthropic_base_url.trim_end_matches('/')
            }
        };
        let path_and_query = match self {
            Self::OpenAiResponses | Self::OpenAiChatCompletions | Self::OpenAiModels
                if !path_and_query.starts_with("/v1/") =>
            {
                format!("/v1{path_and_query}")
            }
            _ => path_and_query.to_string(),
        };
        format!("{base}{path_and_query}")
    }
}

// Reads the gateway session id from explicit sidecar headers first, with Claude's session header
// accepted for compatibility with Claude Code environments that already propagate it.
fn gateway_session_id(headers: &HeaderMap) -> Option<String> {
    header_string(headers, "x-nemo-flow-session-id")
        .or_else(|| header_string(headers, "x-claude-code-session-id"))
}

fn gateway_subagent_id(headers: &HeaderMap) -> Option<String> {
    header_string(headers, "x-nemo-flow-subagent-id")
}

// Resolves a correlation identifier from a dedicated header before trying known JSON body paths.
// Header precedence lets callers disambiguate requests even when provider payloads contain stale
// or differently scoped identifiers.
fn gateway_identifier(
    headers: &HeaderMap,
    body: &Value,
    header_name: &'static str,
    body_paths: &[&[&str]],
) -> Option<String> {
    header_string(headers, header_name).or_else(|| {
        body_paths
            .iter()
            .find_map(|path| string_at(body, path))
            .filter(|value| !value.is_empty())
    })
}

// Reads nested JSON as a string, accepting scalar numeric and boolean forms for provider metadata
// fields that are not consistently serialized as strings. Arrays and objects are ignored.
fn string_at(payload: &Value, path: &[&str]) -> Option<String> {
    let mut current = payload;
    for key in path {
        current = current.get(*key)?;
    }
    match current {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

// Copies only non-sensitive, forwardable request headers into LLM request metadata. This preserves
// correlation headers while excluding credentials and hop-by-hop transport details.
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

// Copies upstream response headers except hop-by-hop transport headers that Axum/hyper must manage
// for the downstream connection. Multiple values are appended to preserve provider behavior.
fn response_headers(headers: &HeaderMap) -> HeaderMap {
    let mut output = HeaderMap::new();
    for (name, value) in headers {
        if !is_hop_by_hop(name) {
            output.append(name.clone(), value.clone());
        }
    }
    output
}

// Reconstructs an Axum response from upstream status, filtered headers, and the selected body. All
// builder errors are converted into sidecar HTTP errors rather than panics.
fn build_response(
    status: StatusCode,
    headers: HeaderMap,
    body: Body,
) -> Result<Response<Body>, SidecarError> {
    let mut builder = Response::builder().status(status);
    for (name, value) in &headers {
        builder = builder.header(name, value);
    }
    Ok(builder.body(body)?)
}

// Allows provider request headers through unless they are transport-owned or must be recalculated
// for the forwarded body. Host and content length are intentionally excluded because reqwest sets
// them for the upstream connection.
fn should_forward_request_header(name: &HeaderName) -> bool {
    !is_hop_by_hop(name)
        && name != http::header::HOST
        && name != http::header::CONTENT_LENGTH
        // Strip Accept-Encoding so upstreams return identity-encoded bodies; otherwise the
        // observability capture (`output.value` on LLM spans, ATIF trajectory bodies) records
        // gzip/br/zstd bytes that downstream consumers can't read. Bandwidth cost is paid only
        // on the sidecar-upstream hop. The client never asked for the encoding it would have
        // received from upstream, so its decoders never trigger.
        && name != http::header::ACCEPT_ENCODING
}

// Allows headers into observability metadata only after removing credentials and provider API keys.
// The forwarding filter runs first so hop-by-hop transport headers are also excluded from recorded
// LLM request attributes.
fn should_record_header(name: &HeaderName) -> bool {
    should_forward_request_header(name)
        && name != http::header::AUTHORIZATION
        && name.as_str() != "x-api-key"
        && name.as_str() != "anthropic-api-key"
}

// Identifies headers that describe a single transport hop and therefore must not be proxied across
// the client-sidecar-upstream boundary.
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

// Builds the streaming end-event payload from the collected prefix. Truncated streams are marked
// explicitly so downstream analysis does not mistake the preview for a complete provider response.
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
