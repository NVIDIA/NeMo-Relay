// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::sync::{Arc, Mutex};

use async_stream::stream;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode};
use futures_util::StreamExt;
use http_body_util::LengthLimitError;
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmRequest, LlmStreamCallExecuteParams, llm_call_execute,
    llm_stream_call_execute,
};
use nemo_relay::api::runtime::{
    LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn, TASK_SCOPE_STACK,
};
use nemo_relay::codec::resolve::{
    ProviderSurface, response_codec as build_response_codec,
    streaming_codec as build_streaming_codec,
};
use nemo_relay::codec::streaming::StreamingCodec;
use nemo_relay::codec::traits::LlmResponseCodec;
use nemo_relay::error::FlowError;
use serde_json::{Map, Value, json};

use crate::alignment::{self, GatewayRouteKind};
use crate::config::header_string;
use crate::error::CliError;
use crate::server::AppState;
use crate::session::{GatewayCallPrep, LlmGatewayStart, SessionManager};

/// Proxies supported LLM API requests through NeMo Relay's managed execution pipeline.
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
    state.touch();
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
// for both upstream forwarding and NeMo Relay LLM start events. Provider JSON parse failures are not
// request failures because the gateway still forwards raw bytes unchanged.
async fn prepare_gateway_request(
    config: &crate::config::GatewayConfig,
    request: Request<Body>,
) -> Result<PreparedGatewayRequest, CliError> {
    let (parts, body) = request.into_parts();
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
        .map(|p| p.as_str())
        .unwrap_or(parts.uri.path());
    let upstream_url = gateway_upstream_url_override(provider, &parts.headers, path_and_query)
        .unwrap_or_else(|| provider.upstream_url(config, path_and_query));
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

// Builds the [`LlmGatewayStart`] payload from a prepared request. Identifier resolution is shared
// across streaming and non-streaming paths so correlation behavior is consistent for every route.
// Provider-specific fallbacks are resolved here, before request execution leaves the gateway path,
// because the later runtime-managed LLM call only sees this normalized start payload.
fn build_llm_gateway_start(request: &PreparedGatewayRequest) -> LlmGatewayStart {
    LlmGatewayStart {
        // Explicit NeMo Relay headers still win, but alignment can recover agent-native session
        // signals when available. Applies to Claude Code's session header and Codex's Responses
        // prompt-cache thread id today.
        session_id: gateway_session_id(&request.headers, &request.request_json, request.provider),
        provider: request.provider.name().to_string(),
        model_name: request
            .request_json
            .get("model")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        // Subagent ownership is intentionally header-only at the gateway layer. Body fields can be
        // provider payload content rather than scope identity, so the session layer handles other
        // ownership hints.
        subagent_id: gateway_subagent_id(&request.headers),
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
        // Preserve a transport request id as a weak fallback for debugging even when the provider
        // body does not expose an LLM request id.
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
    if prep.bypass_managed_pipeline {
        let session_id = prep.session_id.clone();
        let prune_empty_session = prep.prune_empty_session_on_finish;
        let model = prep.model_name.as_deref().unwrap_or("<unknown>");
        eprintln!(
            "nemo-relay CLI gateway: bypassing managed LLM observability for Claude Code startup probe session={session_id} provider={} model={model}",
            prep.provider_name
        );
        state
            .sessions
            .finish_gateway_call(&session_id, prune_empty_session)
            .await;
        return run_unmanaged_gateway(state, prepared).await;
    }
    let codecs = codecs_for_route(prepared.provider);
    if prepared.streaming {
        run_managed_streaming(state, prepared, prep, codecs).await
    } else {
        run_managed_buffered(state, prepared, prep, codecs).await
    }
}

async fn run_unmanaged_gateway(
    state: AppState,
    prepared: PreparedGatewayRequest,
) -> Result<Response<Body>, CliError> {
    if prepared.streaming {
        return passthrough_streaming(state, prepared).await;
    }
    let response = forward_upstream_request(
        &state.http,
        &prepared.method,
        &prepared.upstream_url,
        &prepared.body_bytes,
        &prepared.headers,
        None,
        prepared.provider,
    )
    .await?;
    let status = response.status();
    let headers = response_headers(response.headers());
    let bytes = response.bytes().await?;
    build_response(status, headers, Body::from(bytes))
}

// Codecs registered for each managed provider route. Routes that emit LLM events but lack a typed
// codec (count_tokens) return `None` so the runtime still wraps the call but skips annotation.
struct RouteCodecs {
    streaming: Option<Box<dyn StreamingCodec>>,
    response: Option<Arc<dyn LlmResponseCodec>>,
}

fn codecs_for_route(route: ProviderRoute) -> RouteCodecs {
    // Routes with a typed provider surface get the built-in codecs from the core
    // factory; count_tokens/models emit LLM events but have no codec.
    match route.provider_surface() {
        Some(surface) => RouteCodecs {
            streaming: Some(build_streaming_codec(surface)),
            response: Some(build_response_codec(surface)),
        },
        None => RouteCodecs {
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
        bypass_managed_pipeline: _,
        prune_empty_session_on_finish: _,
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
            state.sessions.finish_gateway_call(&session_id, false).await;
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
        Err(error) => {
            state.sessions.finish_gateway_call(&session_id, false).await;
            Err(translate_runtime_error(error, &upstream_error))
        }
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
    Arc::new(move |request| {
        let http = http.clone();
        let method = method.clone();
        let url = url.clone();
        let body_bytes = body_bytes.clone();
        let headers = headers.clone();
        let upstream_info = upstream_info.clone();
        let upstream_error = upstream_error.clone();
        let response_bytes = response_bytes.clone();
        Box::pin(async move {
            let response = match forward_upstream_request(
                &http,
                &method,
                &url,
                &body_bytes,
                &headers,
                Some(&request),
                route,
            )
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
        state
            .sessions
            .finish_gateway_call(&prep.session_id, false)
            .await;
        return passthrough_streaming(state, prepared).await;
    };
    let collector = streaming_codec.collector();
    let final_response = Arc::new(Mutex::new(None));
    let final_response_for_finalizer = final_response.clone();
    let original_finalizer = streaming_codec.finalizer();
    let finalizer = Box::new(move || {
        let response = original_finalizer();
        *final_response_for_finalizer
            .lock()
            .expect("stream final response lock poisoned") = Some(response.clone());
        response
    });

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
        bypass_managed_pipeline: _,
        prune_empty_session_on_finish: _,
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
    let json_stream = match json_stream_result {
        Ok(json_stream) => json_stream,
        Err(error) => {
            state.sessions.finish_gateway_call(&session_id, false).await;
            return Err(translate_runtime_error(error, &upstream_error));
        }
    };
    let (status, headers) = upstream_info
        .lock()
        .expect("upstream info lock poisoned")
        .take()
        .unwrap_or((StatusCode::OK, HeaderMap::new()));
    let body = client_sse_body(
        json_stream,
        provider_route,
        state.sessions.clone(),
        session_id.clone(),
        owner_subagent_id,
        final_response,
    );

    // Streamed responses are finalized inside the runtime stream wrapper. The small finalizer tap
    // above copies only the aggregate JSON payload so the session can update turn output and tool
    // hints after the downstream client consumes the stream, without buffering SSE bytes here.
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
    Arc::new(move |request| {
        let http = http.clone();
        let method = method.clone();
        let url = url.clone();
        let body_bytes = body_bytes.clone();
        let headers = headers.clone();
        let upstream_info = upstream_info.clone();
        let upstream_error = upstream_error.clone();
        Box::pin(async move {
            let response = match forward_upstream_request(
                &http,
                &method,
                &url,
                &body_bytes,
                &headers,
                Some(&request),
                route,
            )
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
    use nemo_relay::codec::streaming::SseEventDecoder;
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
fn client_sse_body(
    json_stream: LlmJsonStream,
    route: ProviderRoute,
    sessions: SessionManager,
    session_id: String,
    owner_subagent_id: Option<String>,
    final_response: Arc<Mutex<Option<Value>>>,
) -> Body {
    let mut json_stream = json_stream;
    let mut guard = GatewayCallGuard::new(sessions, session_id, owner_subagent_id, final_response);
    let stream = stream! {
        while let Some(item) = json_stream.next().await {
            match item {
                Ok(event_json) => {
                    let frame = encode_sse_frame(&event_json, route);
                    yield Ok::<Bytes, CliError>(Bytes::from(frame));
                }
                Err(error) => {
                    guard.finish().await;
                    yield Err(CliError::InvalidPayload(error.to_string()));
                    return;
                }
            }
        }
        guard.finish().await;
        if matches!(route, ProviderRoute::OpenAiChatCompletions) {
            yield Ok::<Bytes, CliError>(Bytes::from_static(b"data: [DONE]\n\n"));
        }
    };
    Body::from_stream(stream)
}

// Keeps the session idle detector honest for streaming responses. Normal completion calls
// `finish`, while early client disconnects drop the body stream and use the drop path to release
// the in-flight gateway call asynchronously.
struct GatewayCallGuard {
    sessions: Option<SessionManager>,
    session_id: String,
    owner_subagent_id: Option<String>,
    final_response: Arc<Mutex<Option<Value>>>,
}

impl GatewayCallGuard {
    fn new(
        sessions: SessionManager,
        session_id: String,
        owner_subagent_id: Option<String>,
        final_response: Arc<Mutex<Option<Value>>>,
    ) -> Self {
        Self {
            sessions: Some(sessions),
            session_id,
            owner_subagent_id,
            final_response,
        }
    }

    async fn finish(&mut self) {
        if let Some(sessions) = self.sessions.take() {
            let response = self
                .final_response
                .lock()
                .expect("stream final response lock poisoned")
                .take();
            if let Some(response) = response {
                sessions
                    .record_gateway_response_hints(
                        &self.session_id,
                        self.owner_subagent_id.clone(),
                        response,
                    )
                    .await;
            }
            sessions.finish_gateway_call(&self.session_id, false).await;
        }
    }
}

impl Drop for GatewayCallGuard {
    fn drop(&mut self) {
        let Some(sessions) = self.sessions.take() else {
            return;
        };
        let session_id = self.session_id.clone();
        let owner_subagent_id = self.owner_subagent_id.clone();
        let response = self
            .final_response
            .lock()
            .expect("stream final response lock poisoned")
            .take();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Some(response) = response {
                    sessions
                        .record_gateway_response_hints(&session_id, owner_subagent_id, response)
                        .await;
                }
                sessions.finish_gateway_call(&session_id, false).await;
            });
        }
    }
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
// is shared by the buffered and streaming managed funcs so header filtering stays consistent.
// Agent-native credential quirks are normalized by alignment before provider auth injection runs.
async fn forward_upstream_request(
    http: &reqwest::Client,
    method: &Method,
    url: &str,
    body_bytes: &Bytes,
    headers: &HeaderMap,
    effective_request: Option<&LlmRequest>,
    route: ProviderRoute,
) -> Result<reqwest::Response, reqwest::Error> {
    let (body_bytes, headers) = effective_upstream_request(body_bytes, headers, effective_request);
    let sanitized = strip_replaceable_agent_auth_headers(&headers, route);
    let mut upstream = http.request(method.clone(), url).body(body_bytes.clone());
    for (name, value) in &sanitized {
        if should_forward_request_header(name) {
            upstream = upstream.header(name, value);
        }
    }
    upstream = inject_provider_auth(upstream, route, &sanitized);
    upstream.send().await
}

fn effective_upstream_request(
    body_bytes: &Bytes,
    headers: &HeaderMap,
    effective_request: Option<&LlmRequest>,
) -> (Bytes, HeaderMap) {
    let Some(request) = effective_request else {
        return (body_bytes.clone(), headers.clone());
    };

    let body_bytes = if request.content.is_null() {
        body_bytes.clone()
    } else {
        match serde_json::to_vec(&request.content) {
            Ok(serialized) => Bytes::from(serialized),
            Err(error) => {
                eprintln!(
                    "nemo-relay CLI gateway: failed to serialize rewritten LLM request body; forwarding original request: {error}"
                );
                return (body_bytes.clone(), headers.clone());
            }
        }
    };
    let mut headers = headers.clone();
    for (name, value) in &request.headers {
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        let Some(value) = json_header_value(value) else {
            continue;
        };
        headers.insert(name, value);
    }
    (body_bytes, headers)
}

fn json_header_value(value: &Value) -> Option<HeaderValue> {
    let rendered = match value {
        Value::String(value) => value.clone(),
        value => serde_json::to_string(value).ok()?,
    };
    HeaderValue::from_str(&rendered).ok()
}

// If the inbound request has no provider auth header (Authorization / x-api-key / api-key), read
// the provider's standard API key env var and attach it to the outbound request. Alignment may
// have already normalized agent-native auth material; this function remains provider-generic and
// only handles standard upstream auth injection.
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
        None,
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
        FlowError::GuardrailRejected(reason) => CliError::GuardrailRejected(reason),
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
    state.touch();
    let (parts, _body) = request.into_parts();
    if parts.method != Method::GET {
        return build_response(
            StatusCode::METHOD_NOT_ALLOWED,
            HeaderMap::new(),
            Body::empty(),
        );
    }
    let provider = ProviderRoute::OpenAiModels;
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or(parts.uri.path());
    let upstream_url = gateway_upstream_url_override(provider, &parts.headers, path_and_query)
        .unwrap_or_else(|| provider.upstream_url(&state.config, path_and_query));
    let sanitized = strip_replaceable_agent_auth_headers(&parts.headers, provider);
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

    // Maps a route to its built-in codec surface, or `None` for routes that emit LLM events but
    // have no typed request/response codec (count_tokens, models).
    const fn provider_surface(self) -> Option<ProviderSurface> {
        match self {
            Self::OpenAiResponses => Some(ProviderSurface::OpenAIResponses),
            Self::OpenAiChatCompletions => Some(ProviderSurface::OpenAIChat),
            Self::AnthropicMessages => Some(ProviderSurface::AnthropicMessages),
            Self::AnthropicCountTokens | Self::OpenAiModels => None,
        }
    }

    // Returns the provider route name recorded on managed LLM events. These names split OpenAI API
    // variants because their request/response schemas differ even when they share a base URL, and
    // they double as codec hints for ambiguous provider request shapes.
    const fn name(self) -> &'static str {
        self.alignment_route().name()
    }

    // Builds the upstream URL by combining the configured provider base with the original path and
    // query string. Trailing slashes are stripped from the base to avoid double-slash variants in
    // configured enterprise or local proxy endpoints.
    fn upstream_url(self, config: &crate::config::GatewayConfig, path_and_query: &str) -> String {
        let base = match self {
            Self::OpenAiResponses | Self::OpenAiChatCompletions | Self::OpenAiModels => {
                config.openai_base_url.as_str()
            }
            Self::AnthropicMessages | Self::AnthropicCountTokens => {
                config.anthropic_base_url.as_str()
            }
        };
        self.upstream_url_with_base(base, path_and_query)
    }

    // Like `upstream_url` but with an explicit base URL. This keeps OpenAI `/v1` normalization in
    // one place for configured public, enterprise, or local proxy bases.
    fn upstream_url_with_base(self, base: &str, path_and_query: &str) -> String {
        let base = base.trim_end_matches('/');
        let path_and_query = match self {
            Self::OpenAiResponses | Self::OpenAiChatCompletions | Self::OpenAiModels => {
                normalize_openai_path_for_base(base, path_and_query)
            }
            _ => path_and_query.to_string(),
        };
        format!("{base}{path_and_query}")
    }

    // Narrows gateway routing to the smaller taxonomy used by trace alignment. Keeping this
    // conversion here prevents provider-specific alignment code from depending on gateway URL
    // routing internals.
    const fn alignment_route(self) -> GatewayRouteKind {
        match self {
            Self::OpenAiResponses => GatewayRouteKind::OpenAiResponses,
            Self::OpenAiChatCompletions => GatewayRouteKind::OpenAiChatCompletions,
            Self::OpenAiModels => GatewayRouteKind::OpenAiModels,
            Self::AnthropicMessages => GatewayRouteKind::AnthropicMessages,
            Self::AnthropicCountTokens => GatewayRouteKind::AnthropicCountTokens,
        }
    }
}

fn normalize_openai_path_for_base(base: &str, path_and_query: &str) -> String {
    match (base.ends_with("/v1"), path_and_query.starts_with("/v1/")) {
        (true, true) => path_and_query
            .strip_prefix("/v1")
            .expect("path was checked to start with /v1")
            .to_string(),
        (false, false) => format!("/v1{path_and_query}"),
        _ => path_and_query.to_string(),
    }
}

// Gives alignment adapters a chance to choose an agent-native upstream before default provider
// routing runs. Today this supports Codex ChatGPT auth; future harness fallbacks should stay in
// alignment rather than adding provider-shaped checks here.
fn gateway_upstream_url_override(
    route: ProviderRoute,
    headers: &HeaderMap,
    path_and_query: &str,
) -> Option<String> {
    gateway_upstream_url_override_with_openai_key_state(
        route,
        headers,
        path_and_query,
        env_var_is_nonempty("OPENAI_API_KEY"),
    )
}

fn gateway_upstream_url_override_with_openai_key_state(
    route: ProviderRoute,
    headers: &HeaderMap,
    path_and_query: &str,
    has_openai_replacement_key: bool,
) -> Option<String> {
    alignment::gateway_upstream_url_override(
        headers,
        route.alignment_route(),
        path_and_query,
        has_openai_replacement_key,
    )
}

// Lets alignment adapters strip agent-native credentials only when the gateway can replace them
// with standard provider API keys. Whitespace-only env vars are treated as missing because
// forwarding an empty bearer value only replaces one authentication failure with another.
fn strip_replaceable_agent_auth_headers(headers: &HeaderMap, route: ProviderRoute) -> HeaderMap {
    strip_replaceable_agent_auth_headers_with_openai_key_state(
        headers,
        route,
        env_var_is_nonempty("OPENAI_API_KEY"),
    )
}

fn strip_replaceable_agent_auth_headers_with_openai_key_state(
    headers: &HeaderMap,
    route: ProviderRoute,
    has_openai_replacement_key: bool,
) -> HeaderMap {
    alignment::gateway_forward_headers(headers, route.alignment_route(), has_openai_replacement_key)
}

fn env_var_is_nonempty(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .is_some()
}

// Delegates provider-specific session fallbacks to `alignment` so request construction stays
// generic and each coding-agent quirk has one documented adapter.
fn gateway_session_id(headers: &HeaderMap, body: &Value, route: ProviderRoute) -> Option<String> {
    alignment::gateway_session_id(headers, body, route.alignment_route())
}

fn gateway_subagent_id(headers: &HeaderMap) -> Option<String> {
    alignment::gateway_subagent_id(headers)
}

// Keeps the gateway-facing helper local for tests while the generic extraction pattern lives in
// `alignment`.
fn gateway_identifier(
    headers: &HeaderMap,
    body: &Value,
    header_name: &'static str,
    body_paths: &[&[&str]],
) -> Option<String> {
    alignment::gateway_identifier(headers, body, header_name, body_paths)
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
