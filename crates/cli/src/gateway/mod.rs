// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod client;
mod request;
mod response;
mod routes;
pub(crate) mod tls;

use request::*;
use response::*;
use routes::*;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_stream::stream;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{
    HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode, header,
};
use futures_util::StreamExt;
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
use nemo_relay::error::{FlowError, UpstreamFailure, UpstreamFailureClass};
use serde_json::{Value, json};

use crate::agents::shared::alignment::{self, GatewayRouteKind};
use crate::configuration::BOOTSTRAP_CLIENT_TOKEN_HEADER;
use crate::error::CliError;
use crate::server::AppState;
use crate::sessions::{GatewayCallPrep, GatewaySessionFinish, SessionManager};

#[cfg(test)]
#[path = "../../tests/coverage/shared/gateway_tests.rs"]
mod tests;

const INTERNAL_DISPATCH_URL_HEADER: &str = "x-nemo-relay-internal-dispatch-url";
const INTERNAL_DISPATCH_ROUTE_HEADER: &str = "x-nemo-relay-internal-dispatch-route";
const INTERNAL_RETRY_AWARE_HEADER: &str = "x-nemo-relay-internal-retry-aware";
const MAX_UPSTREAM_ERROR_BODY_BYTES: usize = 64 * 1024;

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
    mut request: Request<Body>,
) -> Result<Response<Body>, CliError> {
    state.touch();
    let authorization = state.authorize_provider_request(request.headers_mut())?;
    let prepared = prepare_gateway_request(&state.config, request, authorization).await?;
    let prep = state
        .sessions
        .prepare_gateway_call(&prepared.headers, build_llm_gateway_start(&prepared))
        .await?;
    run_managed_gateway(state, prepared, prep).await
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
        let session_finish = prep.session_finish;
        let model = prep.model_name.as_deref().unwrap_or("<unknown>");
        log::warn!(
            target: "nemo_relay.gateway",
            event = "observability_bypassed",
            session_id = session_id.as_str(),
            provider = prep.provider_name.as_str(),
            model = model,
            reason = "startup_probe";
            "Managed observability was bypassed for a startup probe"
        );
        state
            .sessions
            .finish_gateway_call(&session_id, session_finish)
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
        ProviderForwarding::new(prepared.provider, prepared.authorization, &state.config),
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
        session_finish,
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
            state
                .sessions
                .finish_gateway_call(&session_id, session_finish)
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
        Err(error) => {
            state
                .sessions
                .finish_gateway_call(&session_id, session_finish)
                .await;
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
    let forwarding =
        ProviderForwarding::new(prepared.provider, prepared.authorization, &state.config);
    Arc::new(move |request| {
        let http = http.clone();
        let forwarding = forwarding.clone();
        let method = method.clone();
        let url = url.clone();
        let body_bytes = body_bytes.clone();
        let headers = headers.clone();
        let upstream_info = upstream_info.clone();
        let upstream_error = upstream_error.clone();
        let response_bytes = response_bytes.clone();
        Box::pin(async move {
            let retry_aware = retry_aware_dispatch(&request);
            let response = match forward_upstream_request(
                &http,
                &method,
                &url,
                &body_bytes,
                &headers,
                Some(&request),
                forwarding,
            )
            .await
            {
                Ok(response) => response,
                Err(error) => {
                    let message = error.to_string();
                    if retry_aware {
                        return Err(FlowError::Upstream(transport_failure(&error)));
                    }
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
                    if retry_aware {
                        return Err(FlowError::Upstream(transport_failure(&error)));
                    }
                    *upstream_error.lock().expect("upstream error lock poisoned") = Some(error);
                    return Err(FlowError::Internal(message));
                }
            };
            if retry_aware && !status.is_success() {
                return Err(FlowError::Upstream(http_failure(
                    status,
                    &response_headers,
                    &bytes,
                )));
            }
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
        let session_finish = prep.session_finish;
        state
            .sessions
            .finish_gateway_call(&prep.session_id, session_finish)
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
        session_finish,
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
            state
                .sessions
                .finish_gateway_call(&session_id, session_finish)
                .await;
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
        session_finish,
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
    let forwarding =
        ProviderForwarding::new(prepared.provider, prepared.authorization, &state.config);
    Arc::new(move |request| {
        let http = http.clone();
        let forwarding = forwarding.clone();
        let method = method.clone();
        let url = url.clone();
        let body_bytes = body_bytes.clone();
        let headers = headers.clone();
        let upstream_info = upstream_info.clone();
        let upstream_error = upstream_error.clone();
        Box::pin(async move {
            let retry_aware = retry_aware_dispatch(&request);
            let response = match forward_upstream_request(
                &http,
                &method,
                &url,
                &body_bytes,
                &headers,
                Some(&request),
                forwarding,
            )
            .await
            {
                Ok(response) => response,
                Err(error) => {
                    let message = error.to_string();
                    if retry_aware {
                        return Err(FlowError::Upstream(transport_failure(&error)));
                    }
                    *upstream_error.lock().expect("upstream error lock poisoned") = Some(error);
                    return Err(FlowError::Internal(message));
                }
            };
            let status = response.status();
            let response_headers = response_headers(response.headers());
            if retry_aware && !status.is_success() {
                let bytes = response
                    .bytes()
                    .await
                    .map_err(|error| FlowError::Upstream(transport_failure(&error)))?;
                return Err(FlowError::Upstream(http_failure(
                    status,
                    &response_headers,
                    &bytes,
                )));
            }
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
    session_finish: GatewaySessionFinish,
) -> Body {
    let mut json_stream = json_stream;
    let mut guard = GatewayCallGuard::new(
        sessions,
        session_id,
        owner_subagent_id,
        final_response,
        session_finish,
    );
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
    session_finish: GatewaySessionFinish,
}

impl GatewayCallGuard {
    fn new(
        sessions: SessionManager,
        session_id: String,
        owner_subagent_id: Option<String>,
        final_response: Arc<Mutex<Option<Value>>>,
        session_finish: GatewaySessionFinish,
    ) -> Self {
        Self {
            sessions: Some(sessions),
            session_id,
            owner_subagent_id,
            final_response,
            session_finish,
        }
    }

    async fn finish(&mut self) {
        if let Some(sessions) = self.sessions.take() {
            let response = self
                .final_response
                .lock()
                .expect("stream final response lock poisoned")
                .take();
            complete_gateway_call(
                sessions,
                self.session_id.clone(),
                self.owner_subagent_id.clone(),
                response,
                self.session_finish,
            )
            .await;
        }
    }
}

async fn complete_gateway_call(
    sessions: SessionManager,
    session_id: String,
    owner_subagent_id: Option<String>,
    response: Option<Value>,
    session_finish: GatewaySessionFinish,
) {
    if let Some(response) = response {
        sessions
            .record_gateway_response_hints(&session_id, owner_subagent_id, response)
            .await;
    }
    sessions
        .finish_gateway_call(&session_id, session_finish)
        .await;
}

impl Drop for GatewayCallGuard {
    fn drop(&mut self) {
        let Some(sessions) = self.sessions.take() else {
            return;
        };
        let session_id = self.session_id.clone();
        let owner_subagent_id = self.owner_subagent_id.clone();
        let session_finish = self.session_finish;
        let response = self
            .final_response
            .lock()
            .expect("stream final response lock poisoned")
            .take();
        let cleanup = complete_gateway_call(
            sessions,
            session_id,
            owner_subagent_id,
            response,
            session_finish,
        );
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(cleanup);
        } else {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("gateway cleanup runtime should build")
                .block_on(cleanup);
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
// Source authentication is normalized at ingress, before interceptors can select a target.
async fn forward_upstream_request(
    http: &reqwest::Client,
    method: &Method,
    url: &str,
    body_bytes: &Bytes,
    headers: &HeaderMap,
    effective_request: Option<&LlmRequest>,
    forwarding: ProviderForwarding,
) -> Result<reqwest::Response, reqwest::Error> {
    debug_assert_eq!(
        forwarding
            .authorization
            .source_credential
            .provider_credential_present(),
        crate::provider_auth::has_provider_credential(headers)
    );
    let effective = effective_dispatch_request(
        body_bytes,
        headers,
        effective_request,
        url,
        forwarding.source_route,
    );
    let configured_auth_header = forwarding.configured_auth_header(effective.target_route);
    let mut upstream = http
        .request(method.clone(), &effective.url)
        .body(effective.body_bytes.clone());
    for (name, value) in &effective.headers {
        if should_forward_request_header(name, &effective.headers) {
            upstream = upstream.header(name, value);
        }
    }
    upstream = inject_provider_auth(
        upstream,
        effective.target_route,
        &effective.headers,
        matches!(
            effective.credential_policy,
            TargetCredentialPolicy::SourceOrEnvironment
        ) && forwarding.authorization.allow_environment_provider_auth,
        configured_auth_header,
    );
    upstream.send().await
}

#[derive(Clone)]
struct EffectiveUpstreamRequest {
    body_bytes: Bytes,
    headers: HeaderMap,
    url: String,
    target_route: ProviderRoute,
    credential_policy: TargetCredentialPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TargetCredentialPolicy {
    SourceOrEnvironment,
    ExplicitTarget,
}

#[cfg(test)]
fn effective_upstream_request(
    body_bytes: &Bytes,
    headers: &HeaderMap,
    effective_request: Option<&LlmRequest>,
) -> (Bytes, HeaderMap) {
    let effective = effective_dispatch_request(
        body_bytes,
        headers,
        effective_request,
        "",
        ProviderRoute::OpenAiChatCompletions,
    );
    (effective.body_bytes, effective.headers)
}

fn effective_dispatch_request(
    body_bytes: &Bytes,
    headers: &HeaderMap,
    effective_request: Option<&LlmRequest>,
    url: &str,
    route: ProviderRoute,
) -> EffectiveUpstreamRequest {
    let mut headers = headers.clone();
    strip_internal_dispatch_headers(&mut headers);
    let Some(request) = effective_request else {
        return EffectiveUpstreamRequest {
            body_bytes: body_bytes.clone(),
            headers,
            url: url.to_string(),
            target_route: route,
            credential_policy: TargetCredentialPolicy::SourceOrEnvironment,
        };
    };

    let mut body_reencoded = false;
    let body_bytes = if request.content.is_null() {
        body_bytes.clone()
    } else {
        match serde_json::to_vec(&request.content) {
            Ok(serialized) => {
                body_reencoded = true;
                Bytes::from(serialized)
            }
            Err(_) => {
                log::warn!(
                    target: "nemo_relay.gateway",
                    event = "request_rewrite_fallback",
                    reason = "serialization_failed";
                    "The original upstream request was retained after rewrite serialization failed"
                );
                return EffectiveUpstreamRequest {
                    body_bytes: body_bytes.clone(),
                    headers,
                    url: url.to_string(),
                    target_route: route,
                    credential_policy: TargetCredentialPolicy::SourceOrEnvironment,
                };
            }
        }
    };
    let mut override_url = None;
    let mut override_route = None;
    let mut dispatch_route_header_seen = false;
    for (name, value) in &request.headers {
        if name.eq_ignore_ascii_case(INTERNAL_DISPATCH_URL_HEADER) {
            override_url = json_header_string(value);
            continue;
        }
        if name.eq_ignore_ascii_case(INTERNAL_DISPATCH_ROUTE_HEADER) {
            dispatch_route_header_seen = true;
            override_route = json_header_string(value)
                .and_then(|value| ProviderRoute::from_dispatch_override(&value));
            continue;
        }
    }
    let credential_policy = if override_url.is_some() || dispatch_route_header_seen {
        crate::provider_auth::remove_provider_credentials(&mut headers);
        TargetCredentialPolicy::ExplicitTarget
    } else {
        TargetCredentialPolicy::SourceOrEnvironment
    };
    // Observable source headers exclude credentials. Applying the rewritten map only after source
    // credentials are removed lets an explicit target binding add its own authorization without
    // inheriting credentials intended for the original provider.
    for (name, value) in &request.headers {
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        if is_internal_dispatch_header(&name) {
            continue;
        }
        let Some(value) = json_header_value(value) else {
            continue;
        };
        headers.insert(name, value);
    }
    if body_reencoded {
        headers.remove(header::CONTENT_ENCODING);
    }
    EffectiveUpstreamRequest {
        body_bytes,
        headers,
        url: if dispatch_route_header_seen && override_route.is_none() {
            url.to_string()
        } else {
            override_url.unwrap_or_else(|| url.to_string())
        },
        target_route: override_route.unwrap_or(route),
        credential_policy,
    }
}

fn json_header_string(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn strip_internal_dispatch_headers(headers: &mut HeaderMap) {
    headers.remove(INTERNAL_DISPATCH_URL_HEADER);
    headers.remove(INTERNAL_DISPATCH_ROUTE_HEADER);
    headers.remove(INTERNAL_RETRY_AWARE_HEADER);
}

fn is_internal_dispatch_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        INTERNAL_DISPATCH_URL_HEADER | INTERNAL_DISPATCH_ROUTE_HEADER | INTERNAL_RETRY_AWARE_HEADER
    )
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
    allow_environment_provider_auth: bool,
    configured_auth_header: Option<&str>,
) -> reqwest::RequestBuilder {
    inject_provider_auth_with_env(
        builder,
        route,
        inbound,
        allow_environment_provider_auth,
        configured_auth_header,
        |key| std::env::var(key).ok(),
    )
}

// Pure variant exposed for tests. The env lookup is injected so cases can be exercised without
// mutating process env state (which races with parallel test execution).
fn inject_provider_auth_with_env<F>(
    builder: reqwest::RequestBuilder,
    route: ProviderRoute,
    inbound: &HeaderMap,
    allow_environment_provider_auth: bool,
    configured_auth_header: Option<&str>,
    env_lookup: F,
) -> reqwest::RequestBuilder
where
    F: Fn(&str) -> Option<String>,
{
    if !allow_environment_provider_auth {
        return builder;
    }
    let already_authed = inbound.contains_key(http::header::AUTHORIZATION)
        || inbound.contains_key("x-api-key")
        || inbound.contains_key("api-key")
        || inbound.contains_key("anthropic-api-key");
    if already_authed {
        return builder;
    }
    if let Some(value) = configured_auth_header {
        return builder.header(http::header::AUTHORIZATION, value);
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
        ProviderForwarding::new(prepared.provider, prepared.authorization, &state.config),
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
        FlowError::Upstream(failure) => CliError::ProviderFailure(failure),
        other => CliError::InvalidPayload(other.to_string()),
    }
}

fn retry_aware_dispatch(request: &LlmRequest) -> bool {
    request
        .headers
        .get(INTERNAL_RETRY_AWARE_HEADER)
        .and_then(Value::as_str)
        .is_some_and(|value| value.eq_ignore_ascii_case("true"))
}

fn transport_failure(error: &reqwest::Error) -> UpstreamFailure {
    UpstreamFailure {
        status: error.status().map(|status| status.as_u16()),
        body: bounded_error_body(error.to_string().as_bytes()),
        headers: BTreeMap::new(),
        class: if error.is_timeout() {
            UpstreamFailureClass::Timeout
        } else {
            UpstreamFailureClass::Connection
        },
    }
}

fn http_failure(status: StatusCode, headers: &HeaderMap, body: &[u8]) -> UpstreamFailure {
    let body = bounded_error_body(body);
    let normalized = body.to_ascii_lowercase();
    let class = if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
        UpstreamFailureClass::Authentication
    } else if normalized.contains("context_length_exceeded")
        || normalized.contains("context window")
        || normalized.contains("too many tokens")
    {
        UpstreamFailureClass::ContextWindow
    } else if normalized.contains("model_not_found")
        || normalized.contains("model unavailable")
        || normalized.contains("model_overloaded")
    {
        UpstreamFailureClass::ModelUnavailable
    } else if matches!(status.as_u16(), 408 | 429 | 500 | 502 | 503 | 504) {
        UpstreamFailureClass::RetryableStatus
    } else if status.is_client_error() {
        UpstreamFailureClass::InvalidRequest
    } else {
        UpstreamFailureClass::Other
    };
    UpstreamFailure {
        status: Some(status.as_u16()),
        body,
        headers: failure_headers(headers),
        class,
    }
}

// Captures only safe provider metadata for retry and fallback diagnostics. This is intentionally
// separate from `response_headers`, which preserves response headers for ordinary downstream
// forwarding and therefore must not apply this failure-specific credential filter.
fn failure_headers(headers: &HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter(|(name, _)| {
            !is_hop_by_hop(name)
                && *name != http::header::CONTENT_LENGTH
                && !is_sensitive_response_header(name)
        })
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_string(), value.to_string()))
        })
        .collect()
}

fn is_sensitive_response_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "set-cookie"
            | "www-authenticate"
            | "authorization"
            | "cookie"
            | "x-api-key"
            | "api-key"
            | "anthropic-api-key"
    )
}

fn bounded_error_body(body: &[u8]) -> String {
    String::from_utf8_lossy(&body[..body.len().min(MAX_UPSTREAM_ERROR_BODY_BYTES)]).into_owned()
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
    let (mut parts, _body) = request.into_parts();
    if parts.method != Method::GET {
        return build_response(
            StatusCode::METHOD_NOT_ALLOWED,
            HeaderMap::new(),
            Body::empty(),
        );
    }
    let provider = ProviderRoute::OpenAiModels;
    let configured_auth_header = provider.configured_auth_header(&state.config);
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or(parts.uri.path());
    let authorization = state.authorize_provider_request(&mut parts.headers)?;
    let allow_environment_provider_auth = authorization.allow_environment_provider_auth;
    parts.headers.remove(BOOTSTRAP_CLIENT_TOKEN_HEADER);
    let upstream_url = gateway_upstream_url_override(
        provider,
        &parts.headers,
        path_and_query,
        allow_environment_provider_auth,
        &state.config,
    )
    .unwrap_or_else(|| provider.upstream_url(&state.config, path_and_query));
    let sanitized = strip_replaceable_agent_auth_headers(
        &parts.headers,
        provider,
        allow_environment_provider_auth,
        configured_auth_header,
    );
    let mut upstream = state.http.get(upstream_url);
    for (name, value) in &sanitized {
        if should_forward_request_header(name, &sanitized) {
            upstream = upstream.header(name, value);
        }
    }
    upstream = inject_provider_auth(
        upstream,
        provider,
        &sanitized,
        allow_environment_provider_auth,
        configured_auth_header,
    );
    let upstream_response = upstream.send().await?;
    let status = upstream_response.status();
    let headers = response_headers(upstream_response.headers());
    let bytes = upstream_response.bytes().await?;
    build_response(status, headers, Body::from(bytes))
}
