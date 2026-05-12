// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use async_stream::stream;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, Method, Request, Response, StatusCode};
use futures_util::StreamExt;
use nemo_flow::api::llm::{
    LlmCallExecuteParams, LlmRequest, LlmStreamCallExecuteParams, llm_call_execute,
    llm_stream_call_execute,
};
use nemo_flow::api::runtime::{
    LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn, TASK_SCOPE_STACK,
};
use nemo_flow::codec::anthropic::{AnthropicMessagesCodec, AnthropicMessagesStreamingCodec};
use nemo_flow::codec::openai_chat::{OpenAIChatCodec, OpenAIChatStreamingCodec};
use nemo_flow::codec::openai_responses::{OpenAIResponsesCodec, OpenAIResponsesStreamingCodec};
use nemo_flow::codec::streaming::StreamingCodec;
use nemo_flow::codec::traits::LlmResponseCodec;
use nemo_flow::error::FlowError;
use serde_json::{Map, Value, json};

use crate::config::header_string;
use crate::error::CliError;
use crate::server::AppState;
use crate::session::{GatewayCallPrep, LlmGatewayStart};

const MAX_BODY_BYTES: usize = 100 * 1024 * 1024;

/// Proxies supported LLM API requests through NeMo Flow's managed execution pipeline.
///
/// The gateway buffers the inbound body once, opens a managed LLM call against the resolved
/// session, and lets the runtime own the start/end events. Provider routes that have a built-in
/// codec round-trip the response through the codec so observability records the same annotated
/// response shape as direct in-process calls; routes without a codec still emit raw JSON to the
/// runtime so the LLM scope is preserved.
///
/// Streaming responses are decoded into per-event JSON values, fed through the runtime collector,
/// and re-encoded as SSE frames for the client. This Option B approach (re-encode) keeps the
/// runtime in the streaming hot path so chunk-level observability matches non-streaming output;
/// the trade-off is one extra JSON parse + serialize per chunk versus the alternative byte-tee
/// design that splits a raw byte stream between client and runtime.
pub(crate) async fn passthrough(
    State(state): State<AppState>,
    request: Request<Body>,
) -> Result<Response<Body>, CliError> {
    let prepared = prepare_gateway_request(&state.config, request).await?;
    let prep = state
        .sessions
        .prepare_gateway_call(&prepared.headers, build_llm_gateway_start(&prepared))
        .await?;
    run_managed_gateway(state, prepared, prep).await
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
    config: &crate::config::GatewayConfig,
    request: Request<Body>,
) -> Result<PreparedGatewayRequest, CliError> {
    let (parts, body) = request.into_parts();
    let provider = ProviderRoute::from_path(parts.uri.path()).ok_or_else(|| {
        CliError::InvalidPayload(format!("unsupported gateway path {}", parts.uri.path()))
    })?;
    let body_bytes = axum::body::to_bytes(body, MAX_BODY_BYTES)
        .await
        .map_err(|error| CliError::InvalidPayload(error.to_string()))?;
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

// Builds the [`LlmGatewayStart`] payload from a prepared request. Identifier resolution is shared
// across streaming and non-streaming paths so correlation behavior is consistent for every route.
fn build_llm_gateway_start(request: &PreparedGatewayRequest) -> LlmGatewayStart {
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
        request: LlmRequest {
            headers: observable_headers(&request.headers),
            content: request.request_json.clone(),
        },
        streaming: request.streaming,
        metadata: json!({ "gateway_path": request.path }),
    }
}

// Captures upstream HTTP status and response headers from inside the managed `func`. The runtime's
// LLM execution callback returns only a Json (or Json stream), so the outer gateway needs a side
// channel to recover the bytes the client expects.
type UpstreamResponseInfo = Arc<Mutex<Option<(StatusCode, HeaderMap)>>>;

// Captures the original `reqwest::Error` from an upstream send failure so the gateway can return
// a 502 Bad Gateway on connection-level failures. The runtime collapses every callback failure to
// `FlowError::Internal`, which would otherwise map to a generic 400.
type UpstreamErrorSlot = Arc<Mutex<Option<reqwest::Error>>>;

// Runs the managed pipeline for a prepared gateway request. Streaming and non-streaming branches
// share the same prep + codec dispatch but diverge in how the runtime drives the upstream call.
async fn run_managed_gateway(
    state: AppState,
    prepared: PreparedGatewayRequest,
    prep: GatewayCallPrep,
) -> Result<Response<Body>, CliError> {
    let codecs = codecs_for_route(prepared.provider);
    if prepared.streaming {
        run_managed_streaming(state, prepared, prep, codecs).await
    } else {
        run_managed_buffered(state, prepared, prep, codecs).await
    }
}

// Codecs registered for each managed provider route. Routes that emit LLM events but lack a typed
// codec (count_tokens) return `None` so the runtime still wraps the call but skips annotation.
struct RouteCodecs {
    streaming: Option<Box<dyn StreamingCodec>>,
    response: Option<Arc<dyn LlmResponseCodec>>,
}

fn codecs_for_route(route: ProviderRoute) -> RouteCodecs {
    match route {
        ProviderRoute::AnthropicMessages => RouteCodecs {
            streaming: Some(Box::new(AnthropicMessagesStreamingCodec::new())),
            response: Some(Arc::new(AnthropicMessagesCodec) as Arc<dyn LlmResponseCodec>),
        },
        ProviderRoute::OpenAiResponses => RouteCodecs {
            streaming: Some(Box::new(OpenAIResponsesStreamingCodec::new())),
            response: Some(Arc::new(OpenAIResponsesCodec) as Arc<dyn LlmResponseCodec>),
        },
        ProviderRoute::OpenAiChatCompletions => RouteCodecs {
            streaming: Some(Box::new(OpenAIChatStreamingCodec::new())),
            response: Some(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>),
        },
        ProviderRoute::AnthropicCountTokens | ProviderRoute::OpenAiModels => RouteCodecs {
            streaming: None,
            response: None,
        },
    }
}

// Runs a non-streaming gateway request through `llm_call_execute`. The runtime handles start/end
// events and codec annotation; the gateway only sends the upstream request, parses bytes, and
// forwards the captured status/headers back to the client.
async fn run_managed_buffered(
    state: AppState,
    prepared: PreparedGatewayRequest,
    prep: GatewayCallPrep,
    codecs: RouteCodecs,
) -> Result<Response<Body>, CliError> {
    let upstream_info: UpstreamResponseInfo = Arc::new(Mutex::new(None));
    let upstream_error: UpstreamErrorSlot = Arc::new(Mutex::new(None));
    let response_bytes: Arc<Mutex<Option<Bytes>>> = Arc::new(Mutex::new(None));
    let func = build_buffered_func(
        state.clone(),
        &prepared,
        upstream_info.clone(),
        upstream_error.clone(),
        response_bytes.clone(),
    );
    let GatewayCallPrep {
        scope_stack,
        session_id,
        provider_name,
        request,
        parent,
        attributes,
        metadata,
        model_name,
        owner_subagent_id,
    } = prep;
    let provider_for_event = provider_name.clone();
    let params = LlmCallExecuteParams::builder()
        .name(provider_for_event)
        .request(request)
        .func(func)
        .parent_opt(parent)
        .attributes(attributes)
        .metadata(metadata)
        .model_name_opt(model_name)
        .response_codec_opt(codecs.response)
        .build();
    let result = TASK_SCOPE_STACK
        .scope(scope_stack, async move { llm_call_execute(params).await })
        .await;
    match result {
        Ok(response_json) => {
            state
                .sessions
                .record_gateway_response_hints(&session_id, owner_subagent_id, response_json)
                .await;
            let (status, headers) = upstream_info
                .lock()
                .expect("upstream info lock poisoned")
                .take()
                .unwrap_or((StatusCode::OK, HeaderMap::new()));
            let bytes = response_bytes
                .lock()
                .expect("response bytes lock poisoned")
                .take()
                .unwrap_or_default();
            build_response(status, headers, Body::from(bytes))
        }
        Err(error) => Err(translate_runtime_error(error, &upstream_error)),
    }
}

// Builds the managed-execution callback for a non-streaming route. The closure forwards the
// buffered request bytes upstream, captures the status and headers into `upstream_info` so the
// outer code can rebuild the client response, and returns the upstream JSON payload to the runtime.
fn build_buffered_func(
    state: AppState,
    prepared: &PreparedGatewayRequest,
    upstream_info: UpstreamResponseInfo,
    upstream_error: UpstreamErrorSlot,
    response_bytes: Arc<Mutex<Option<Bytes>>>,
) -> LlmExecutionNextFn {
    let http = state.http.clone();
    let method = prepared.method.clone();
    let url = prepared.upstream_url.clone();
    let body_bytes = prepared.body_bytes.clone();
    let headers = prepared.headers.clone();
    let route = prepared.provider;
    Arc::new(move |_request| {
        let http = http.clone();
        let method = method.clone();
        let url = url.clone();
        let body_bytes = body_bytes.clone();
        let headers = headers.clone();
        let upstream_info = upstream_info.clone();
        let upstream_error = upstream_error.clone();
        let response_bytes = response_bytes.clone();
        Box::pin(async move {
            let response =
                match forward_upstream_request(&http, &method, &url, &body_bytes, &headers, route)
                    .await
                {
                    Ok(response) => response,
                    Err(error) => {
                        let message = error.to_string();
                        *upstream_error.lock().expect("upstream error lock poisoned") = Some(error);
                        return Err(FlowError::Internal(message));
                    }
                };
            let status = response.status();
            let response_headers = response_headers(response.headers());
            let bytes = match response.bytes().await {
                Ok(bytes) => bytes,
                Err(error) => {
                    let message = error.to_string();
                    *upstream_error.lock().expect("upstream error lock poisoned") = Some(error);
                    return Err(FlowError::Internal(message));
                }
            };
            let json = serde_json::from_slice::<Value>(&bytes)
                .unwrap_or_else(|_| json!({ "body_bytes": bytes.len() }));
            *upstream_info.lock().expect("upstream info lock poisoned") =
                Some((status, response_headers));
            *response_bytes.lock().expect("response bytes lock poisoned") = Some(bytes);
            Ok(json)
        })
    })
}

// Runs a streaming gateway request through `llm_stream_call_execute`. The runtime wraps the
// upstream byte stream as `LlmJsonStream`; the gateway then re-encodes the parsed events back into
// SSE frames for the client (Option B trade-off: simpler chunk-level observability, one extra
// JSON parse/serialize per chunk).
async fn run_managed_streaming(
    state: AppState,
    prepared: PreparedGatewayRequest,
    prep: GatewayCallPrep,
    codecs: RouteCodecs,
) -> Result<Response<Body>, CliError> {
    let upstream_info: UpstreamResponseInfo = Arc::new(Mutex::new(None));
    let upstream_error: UpstreamErrorSlot = Arc::new(Mutex::new(None));
    let func = build_streaming_func(
        state.clone(),
        &prepared,
        upstream_info.clone(),
        upstream_error.clone(),
    );
    let provider_route = prepared.provider;

    // Streaming routes that lack a codec fall back to byte passthrough. The runtime requires a
    // collector and finalizer for managed streaming, so without a codec we cannot use the managed
    // pipeline. This keeps non-LLM streaming paths working while typed codecs remain optional.
    let Some(streaming_codec) = codecs.streaming else {
        return passthrough_streaming(state, prepared).await;
    };
    let collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();

    let GatewayCallPrep {
        scope_stack,
        session_id,
        provider_name,
        request,
        parent,
        attributes,
        metadata,
        model_name,
        owner_subagent_id,
    } = prep;
    let params = LlmStreamCallExecuteParams::builder()
        .name(provider_name)
        .request(request)
        .func(func)
        .collector(collector)
        .finalizer(finalizer)
        .parent_opt(parent)
        .attributes(attributes)
        .metadata(metadata)
        .model_name_opt(model_name)
        .response_codec_opt(codecs.response)
        .build();
    let json_stream_result = TASK_SCOPE_STACK
        .scope(
            scope_stack,
            async move { llm_stream_call_execute(params).await },
        )
        .await;
    let json_stream =
        json_stream_result.map_err(|error| translate_runtime_error(error, &upstream_error))?;
    let (status, headers) = upstream_info
        .lock()
        .expect("upstream info lock poisoned")
        .take()
        .unwrap_or((StatusCode::OK, HeaderMap::new()));
    let body = client_sse_body(json_stream, provider_route);

    // Tool hint extraction from streamed responses is intentionally deferred: the runtime
    // synthesizes the aggregate response inside `LlmStreamWrapper::emit_end_event` and does not
    // surface it back to the caller. Non-streamed responses continue to feed
    // `record_gateway_response_hints` from `run_managed_buffered`. Wiring streamed hints back would
    // require either a runtime hook or a finalizer-tap channel; neither is in scope for the
    // initial managed-execution refactor.
    let _ = (session_id, owner_subagent_id);

    build_response(status, headers, body)
}

// Builds the streaming managed-execution callback. The runtime drives the returned future, which
// fires the upstream request, captures the status + headers into `upstream_info`, and yields a
// stream of parsed SSE event JSON values for the runtime collector.
fn build_streaming_func(
    state: AppState,
    prepared: &PreparedGatewayRequest,
    upstream_info: UpstreamResponseInfo,
    upstream_error: UpstreamErrorSlot,
) -> LlmStreamExecutionNextFn {
    let http = state.http.clone();
    let method = prepared.method.clone();
    let url = prepared.upstream_url.clone();
    let body_bytes = prepared.body_bytes.clone();
    let headers = prepared.headers.clone();
    let route = prepared.provider;
    Arc::new(move |_request| {
        let http = http.clone();
        let method = method.clone();
        let url = url.clone();
        let body_bytes = body_bytes.clone();
        let headers = headers.clone();
        let upstream_info = upstream_info.clone();
        let upstream_error = upstream_error.clone();
        Box::pin(async move {
            let response =
                match forward_upstream_request(&http, &method, &url, &body_bytes, &headers, route)
                    .await
                {
                    Ok(response) => response,
                    Err(error) => {
                        let message = error.to_string();
                        *upstream_error.lock().expect("upstream error lock poisoned") = Some(error);
                        return Err(FlowError::Internal(message));
                    }
                };
            let status = response.status();
            let response_headers = response_headers(response.headers());
            *upstream_info.lock().expect("upstream info lock poisoned") =
                Some((status, response_headers));
            let json_stream = sse_json_stream(response);
            Ok(json_stream)
        })
    })
}

// Decodes an upstream SSE byte stream into a stream of parsed `data:` JSON payloads. Frames with no
// `data:` line (heartbeats), comments, and the `data: [DONE]` sentinel are filtered out by the
// shared `SseEventDecoder`. Trailing partial frames are surfaced to the runtime so the collector
// observes whatever the upstream sent before disconnect.
fn sse_json_stream(response: reqwest::Response) -> LlmJsonStream {
    use nemo_flow::codec::streaming::SseEventDecoder;
    let mut decoder = SseEventDecoder::new();
    let mut bytes = response.bytes_stream();
    let stream = stream! {
        while let Some(chunk) = bytes.next().await {
            match chunk {
                Ok(buffer) => {
                    match decoder.push_bytes(&buffer) {
                        Ok(events) => {
                            for event in events {
                                yield Ok(event.data);
                            }
                        }
                        Err(error) => {
                            yield Err(error);
                            return;
                        }
                    }
                }
                Err(error) => {
                    yield Err(FlowError::Internal(error.to_string()));
                    return;
                }
            }
        }
        match decoder.finish() {
            Ok(Some(event)) => yield Ok(event.data),
            Ok(None) => {}
            Err(error) => yield Err(error),
        }
    };
    Box::pin(stream)
}

// Re-encodes a runtime JSON stream as `text/event-stream` frames for the downstream client. Event
// names are reconstructed from the JSON `type` field where providers populate it (Anthropic
// Messages, OpenAI Responses); OpenAI Chat omits the `event:` line and appends the original
// `data: [DONE]` terminator after the runtime stream completes.
fn client_sse_body(json_stream: LlmJsonStream, route: ProviderRoute) -> Body {
    let mut json_stream = json_stream;
    let stream = stream! {
        while let Some(item) = json_stream.next().await {
            match item {
                Ok(event_json) => {
                    let frame = encode_sse_frame(&event_json, route);
                    yield Ok::<Bytes, CliError>(Bytes::from(frame));
                }
                Err(error) => {
                    yield Err(CliError::InvalidPayload(error.to_string()));
                    return;
                }
            }
        }
        if matches!(route, ProviderRoute::OpenAiChatCompletions) {
            yield Ok::<Bytes, CliError>(Bytes::from_static(b"data: [DONE]\n\n"));
        }
    };
    Body::from_stream(stream)
}

// Formats one SSE frame from a parsed event payload. Anthropic and OpenAI Responses events carry
// the event name in the `type` field, so it is mirrored back onto the `event:` line; OpenAI Chat
// chunks have no event name and emit only `data:`.
fn encode_sse_frame(event_json: &Value, route: ProviderRoute) -> String {
    let serialized = serde_json::to_string(event_json).unwrap_or_else(|_| "null".to_string());
    let event_name = match route {
        ProviderRoute::AnthropicMessages | ProviderRoute::OpenAiResponses => event_json
            .get("type")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        _ => None,
    };
    match event_name {
        Some(name) => format!("event: {name}\ndata: {serialized}\n\n"),
        None => format!("data: {serialized}\n\n"),
    }
}

// Forwards the buffered request to the upstream provider with only the safe request headers. This
// is shared by the buffered and streaming managed funcs so header filtering stays consistent. When
// the inbound request carries no auth (e.g., codex with `requires_openai_auth=false` per NMF-86)
// the gateway injects the provider's API key from environment so the upstream sees authenticated
// traffic without forcing the agent to manage credentials.
async fn forward_upstream_request(
    http: &reqwest::Client,
    method: &Method,
    url: &str,
    body_bytes: &Bytes,
    headers: &HeaderMap,
    route: ProviderRoute,
) -> Result<reqwest::Response, reqwest::Error> {
    // Only strip the inbound JWT when we actually have a replacement key to inject. Without one
    // the upstream just receives no auth and 401s, which is no better than letting it reject the
    // JWT itself — and stripping silently can break setups that point the gateway at an upstream
    // that happens to accept the ChatGPT-Plus token.
    // Whitespace-only keys are effectively missing: stripping the inbound JWT and injecting an
    // empty/whitespace bearer just trades one 401 for another while losing observability.
    let has_openai_env = std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .is_some();
    let sanitized = strip_chatgpt_oauth_for_openai_route(headers, route, has_openai_env);
    let mut upstream = http.request(method.clone(), url).body(body_bytes.clone());
    for (name, value) in &sanitized {
        if should_forward_request_header(name) {
            upstream = upstream.header(name, value);
        }
    }
    upstream = inject_provider_auth(upstream, route, &sanitized);
    upstream.send().await
}

// Removes JWT-shaped bearer tokens from inbound `Authorization` on OpenAI routes when we have
// a replacement `OPENAI_API_KEY` to inject. The detector triggers strictly on the `Bearer eyJ`
// prefix (base64-encoded JSON header), which is what Codex 0.130 sends from `~/.codex/auth.json`
// — that JWT is a consumer ChatGPT-Plus token rejected by `api.openai.com` / LiteLLM-fronted
// endpoints (NVIDIA's `inference-api.nvidia.com`) with 401. After stripping, `inject_provider_auth`
// substitutes the env-provided key and the upstream sees valid auth. ChatGPT OAuth flows in
// general may use opaque tokens too, but those don't match the prefix and are forwarded as-is.
// Real `sk-...` API keys are likewise unaffected. Tracks NMF-86.
fn strip_chatgpt_oauth_for_openai_route(
    headers: &HeaderMap,
    route: ProviderRoute,
    has_replacement_key: bool,
) -> HeaderMap {
    if !matches!(
        route,
        ProviderRoute::OpenAiResponses
            | ProviderRoute::OpenAiChatCompletions
            | ProviderRoute::OpenAiModels
    ) || !has_replacement_key
    {
        return headers.clone();
    }
    let mut out = headers.clone();
    let looks_like_jwt = out
        .get(http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.starts_with("Bearer eyJ"))
        .unwrap_or(false);
    if looks_like_jwt {
        out.remove(http::header::AUTHORIZATION);
    }
    out
}

// If the inbound request has no provider auth header (Authorization / x-api-key / api-key), read
// the provider's standard API key env var and attach it to the outbound request. Tracks NMF-86:
// codex 0.130 prefers `~/.codex/auth.json` ChatGPT-Plus OAuth over `OPENAI_API_KEY` and that JWT is
// rejected by `api.openai.com`, so codex now runs with `requires_openai_auth=false` and the
// gateway owns credentials. If neither inbound auth nor the env var is present, the request is
// forwarded as-is and the upstream returns a real 401 (caller can detect and surface).
fn inject_provider_auth(
    builder: reqwest::RequestBuilder,
    route: ProviderRoute,
    inbound: &HeaderMap,
) -> reqwest::RequestBuilder {
    inject_provider_auth_with_env(builder, route, inbound, |key| std::env::var(key).ok())
}

// Pure variant exposed for tests. The env lookup is injected so cases can be exercised without
// mutating process env state (which races with parallel test execution).
fn inject_provider_auth_with_env<F>(
    builder: reqwest::RequestBuilder,
    route: ProviderRoute,
    inbound: &HeaderMap,
    env_lookup: F,
) -> reqwest::RequestBuilder
where
    F: Fn(&str) -> Option<String>,
{
    let already_authed = inbound.contains_key(http::header::AUTHORIZATION)
        || inbound.contains_key("x-api-key")
        || inbound.contains_key("api-key")
        || inbound.contains_key("anthropic-api-key");
    if already_authed {
        return builder;
    }
    let (env_var, header_name) = match route {
        ProviderRoute::OpenAiResponses
        | ProviderRoute::OpenAiChatCompletions
        | ProviderRoute::OpenAiModels => ("OPENAI_API_KEY", http::header::AUTHORIZATION.as_str()),
        ProviderRoute::AnthropicMessages | ProviderRoute::AnthropicCountTokens => {
            ("ANTHROPIC_API_KEY", "x-api-key")
        }
    };
    let Some(value) = env_lookup(env_var) else {
        return builder;
    };
    // Trim before testing emptiness — a value of "   " is no more useful than "" and sending
    // `Bearer ` with leading whitespace can confuse upstream auth parsers further down.
    let value = value.trim().to_string();
    if value.is_empty() {
        return builder;
    }
    let header_value = match route {
        ProviderRoute::OpenAiResponses
        | ProviderRoute::OpenAiChatCompletions
        | ProviderRoute::OpenAiModels => format!("Bearer {value}"),
        ProviderRoute::AnthropicMessages | ProviderRoute::AnthropicCountTokens => value,
    };
    builder.header(header_name, header_value)
}

// Plain byte passthrough used for streaming routes that lack a typed codec. The managed pipeline
// requires a collector + finalizer, so without a codec we keep the simpler proxy behavior and skip
// the LLM lifecycle event for that single request.
async fn passthrough_streaming(
    state: AppState,
    prepared: PreparedGatewayRequest,
) -> Result<Response<Body>, CliError> {
    let response = forward_upstream_request(
        &state.http,
        &prepared.method,
        &prepared.upstream_url,
        &prepared.body_bytes,
        &prepared.headers,
        prepared.provider,
    )
    .await?;
    let status = response.status();
    let headers = response_headers(response.headers());
    let mut bytes = response.bytes_stream();
    let body = Body::from_stream(stream! {
        while let Some(chunk) = bytes.next().await {
            yield chunk;
        }
    });
    build_response(status, headers, body)
}

// Translates a runtime [`FlowError`] from managed execution into a gateway HTTP error. When the
// failure originated from upstream send/body work, the captured `reqwest::Error` is preferred so
// the response status reflects 502 Bad Gateway rather than the generic 400 from a guardrail or
// internal gateway error.
fn translate_runtime_error(error: FlowError, upstream_error: &UpstreamErrorSlot) -> CliError {
    if let Some(upstream) = upstream_error
        .lock()
        .expect("upstream error lock poisoned")
        .take()
    {
        return CliError::Upstream(upstream);
    }
    match error {
        FlowError::GuardrailRejected(reason) => CliError::InvalidPayload(reason),
        other => CliError::InvalidPayload(other.to_string()),
    }
}

/// Proxies OpenAI model-list requests without creating LLM runtime events.
///
/// The route is registered as GET-only but still verifies the method so direct tests or future
/// router changes return a 405 instead of forwarding a nonsensical request upstream.
pub(crate) async fn models(
    State(state): State<AppState>,
    request: Request<Body>,
) -> Result<Response<Body>, CliError> {
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
    // Whitespace-only keys are effectively missing: stripping the inbound JWT and injecting an
    // empty/whitespace bearer just trades one 401 for another while losing observability.
    let has_openai_env = std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .is_some();
    let sanitized = strip_chatgpt_oauth_for_openai_route(&parts.headers, provider, has_openai_env);
    let mut upstream = state.http.get(upstream_url);
    for (name, value) in &sanitized {
        if should_forward_request_header(name) {
            upstream = upstream.header(name, value);
        }
    }
    upstream = inject_provider_auth(upstream, provider, &sanitized);
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
    // Maps public gateway paths to known upstream provider routes. Unsupported paths return `None`
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
    fn upstream_url(self, config: &crate::config::GatewayConfig, path_and_query: &str) -> String {
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

// Reads the gateway session id from explicit gateway headers first, with Claude's session header
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
// Content-Length is also dropped because the gateway re-encodes streaming responses and the
// upstream-reported length will not match the bytes the client sees.
fn response_headers(headers: &HeaderMap) -> HeaderMap {
    let mut output = HeaderMap::new();
    for (name, value) in headers {
        if !is_hop_by_hop(name) && name != http::header::CONTENT_LENGTH {
            output.append(name.clone(), value.clone());
        }
    }
    output
}

// Reconstructs an Axum response from upstream status, filtered headers, and the selected body. All
// builder errors are converted into gateway HTTP errors rather than panics.
fn build_response(
    status: StatusCode,
    headers: HeaderMap,
    body: Body,
) -> Result<Response<Body>, CliError> {
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
        // on the gateway-upstream hop. The client never asked for the encoding it would have
        // received from upstream, so its decoders never trigger.
        && name != http::header::ACCEPT_ENCODING
}

// Allows headers into observability metadata only after removing credentials and provider API keys.
// The forwarding filter runs first so hop-by-hop transport headers are also excluded from recorded
// LLM request attributes. The credential blocklist covers the four canonical cases we see in
// practice: `Authorization` (most providers), `Cookie` (session credentials), `x-api-key` (OpenAI
// SDK and similar), `anthropic-api-key` (Anthropic), and the generic `api-key` alias used by some
// providers/proxies (e.g., Azure OpenAI). `HeaderName::as_str()` already returns the canonical
// lowercase form so string comparisons are case-insensitive by construction.
fn should_record_header(name: &HeaderName) -> bool {
    should_forward_request_header(name)
        && name != http::header::AUTHORIZATION
        && name != http::header::COOKIE
        && name.as_str() != "x-api-key"
        && name.as_str() != "api-key"
        && name.as_str() != "anthropic-api-key"
}

// Identifies headers that describe a single transport hop and therefore must not be proxied across
// the client-gateway-upstream boundary.
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

#[cfg(test)]
#[path = "../tests/coverage/gateway_tests.rs"]
mod tests;
