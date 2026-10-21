// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod client;
mod request;
mod response;
mod routes;
pub(crate) mod tls;
pub(crate) mod websocket;

use request::*;
pub(crate) use response::strip_redirect_location;
use response::*;
use routes::*;

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use async_stream::stream;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{
    HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode, header,
};
use futures_util::StreamExt;
use nemo_relay::api::llm::gateway::{ResponseCompletionFn, execute as execute_gateway_stream};
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmRequest, LlmStreamCallExecuteParams, llm_call_execute,
};
use nemo_relay::api::runtime::{
    LlmCodecIdentity, LlmExecutionNextFn, LlmFinalizerFn, LlmJsonStream, LlmStreamExecutionNextFn,
    TASK_SCOPE_STACK,
};
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::resolve::{
    ProviderSurface, request_codec as build_request_codec, response_codec as build_response_codec,
    streaming_codec as build_streaming_codec,
};
use nemo_relay::codec::streaming::StreamingCodec;
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay::error::{FlowError, UpstreamFailure, UpstreamFailureClass};
use serde_json::Value;
use tokio::sync::oneshot;

use crate::agents::shared::alignment::{self, GatewayRouteKind};
use crate::configuration::BOOTSTRAP_CLIENT_TOKEN_HEADER;
use crate::error::CliError;
use crate::operational::{
    self, OperationalContext, UPSTREAM_RESPONSE_THRESHOLD_MILLIS,
    UPSTREAM_STREAM_STALL_THRESHOLD_MILLIS,
};
use crate::server::AppState;
use crate::sessions::{GatewayCallPrep, GatewaySessionFinish, SessionManager};

/// Builds the same correlation input used by the foreground gateway without selecting its
/// response-reencoding delivery path. Daemon workers use this before handing the provider body to
/// the raw frame transport.
pub(crate) fn daemon_gateway_start(
    headers: &HeaderMap,
    path: &str,
    request_json: Value,
    streaming: bool,
) -> Option<crate::sessions::LlmGatewayStart> {
    let provider = ProviderRoute::from_request(path, headers)?;
    Some(request::build_llm_gateway_start_from_parts(
        headers,
        path,
        provider,
        request_json,
        streaming,
    ))
}

/// Resolves the provider destination for an authenticated daemon delivery.
///
/// Codex ChatGPT OAuth retains the personal gateway's alignment behavior on every OpenAI route.
/// Other paths honor Pi's validated named-upstream header; a bearer token alone never selects a
/// different authority.
pub(crate) fn daemon_provider_upstream_url(
    headers: &HeaderMap,
    path_and_query: &str,
    config: &crate::configuration::GatewayConfig,
) -> Result<Option<String>, CliError> {
    daemon_provider_upstream_url_with_access(headers, path_and_query, config, true, true)
}

/// Resolves a daemon provider destination for a request with explicit access rights.
///
/// `invocation_authenticated` controls whether the client may name its own upstream.
/// `allow_provider_auth` controls whether daemon-held provider credentials may replace the
/// client's agent-native credential when selecting the upstream.
pub(crate) fn daemon_provider_upstream_url_with_access(
    headers: &HeaderMap,
    path_and_query: &str,
    config: &crate::configuration::GatewayConfig,
    invocation_authenticated: bool,
    allow_provider_auth: bool,
) -> Result<Option<String>, CliError> {
    let path = path_and_query
        .split_once('?')
        .map_or(path_and_query, |(path, _)| path);
    let Some(provider) = ProviderRoute::from_request(path, headers) else {
        return Ok(None);
    };
    match client_named_upstream_url(provider, headers, path_and_query, invocation_authenticated) {
        crate::agents::pi::alignment::NamedUpstream::Named(destination) => {
            if !crate::provider_auth::has_provider_credential(headers) {
                return Err(CliError::InvalidPayload(
                    "a client-named upstream requires a caller-supplied provider credential".into(),
                ));
            }
            Ok(Some(destination))
        }
        crate::agents::pi::alignment::NamedUpstream::Rejected(reason) => {
            Err(CliError::InvalidPayload(reason.to_owned()))
        }
        crate::agents::pi::alignment::NamedUpstream::Absent => Ok(Some(
            gateway_upstream_url_override(
                provider,
                headers,
                path_and_query,
                allow_provider_auth,
                config,
            )
            .unwrap_or_else(|| provider.upstream_url(config, path_and_query)),
        )),
    }
}

/// Whether a daemon delivery goes to an upstream the caller named, and so must be dialed through
/// the named-upstream address policy rather than the shared provider pool.
pub(crate) fn daemon_destination_is_client_named(
    headers: &HeaderMap,
    invocation_authenticated: bool,
) -> bool {
    matches!(
        crate::agents::pi::alignment::client_named_upstream_base(headers, invocation_authenticated),
        crate::agents::pi::alignment::NamedUpstream::Named(_)
    )
}

/// Applies the personal gateway's agent-auth replacement rules to a daemon provider request.
pub(crate) fn daemon_provider_forward_headers(
    headers: &HeaderMap,
    path: &str,
    config: &crate::configuration::GatewayConfig,
) -> Option<HeaderMap> {
    daemon_provider_forward_headers_with_access(
        headers,
        path,
        config,
        daemon_allows_environment_provider_auth(headers),
    )
}

/// Applies agent-auth replacement only when daemon-held provider credentials may be injected, so
/// a request that will not receive a replacement keeps its own credential.
pub(crate) fn daemon_provider_forward_headers_with_access(
    headers: &HeaderMap,
    path: &str,
    config: &crate::configuration::GatewayConfig,
    allow_provider_auth: bool,
) -> Option<HeaderMap> {
    let provider = ProviderRoute::from_request(path, headers)?;
    Some(strip_replaceable_agent_auth_headers(
        headers,
        provider,
        path,
        allow_provider_auth && provider.allows_relay_auth(),
        config,
    ))
}

pub(crate) fn daemon_allows_provider_auth(
    headers: &HeaderMap,
    path: &str,
    _config: &crate::configuration::GatewayConfig,
) -> bool {
    daemon_allows_environment_provider_auth(headers) && provider_route_allows_relay_auth(path)
}

/// Both model APIs share a path. Anthropic protocol headers select Anthropic; token shapes do not.
pub(crate) fn is_anthropic_models_request(path: &str, headers: &HeaderMap) -> bool {
    matches!(path, "/models" | "/v1/models")
        && (headers.contains_key("anthropic-version") || headers.contains_key("anthropic-beta"))
}

/// Model discovery and managed routes may use Relay credentials, including in daemon mode.
pub(crate) fn provider_route_allows_relay_auth(path: &str) -> bool {
    let path = path.split_once('?').map_or(path, |(path, _)| path);
    ProviderRoute::from_path(path).is_some_and(ProviderRoute::allows_relay_auth)
}

pub(crate) fn daemon_allows_environment_provider_auth(headers: &HeaderMap) -> bool {
    !headers.contains_key(crate::agents::pi::alignment::UPSTREAM_BASE_URL_HEADER)
}

/// Returns the exact credential-safe request-header view exposed to daemon-worker middleware.
///
/// The worker uses this same view when applying middleware diffs to the original `HeaderMap`, so
/// headers hidden from middleware remain opaque and unchanged while unmodified multi-value
/// headers retain their original representation.
pub(crate) fn daemon_observable_headers(headers: &HeaderMap) -> serde_json::Map<String, Value> {
    response::observable_headers(headers)
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/gateway_tests.rs"]
mod tests;

const INTERNAL_DISPATCH_URL_HEADER: &str = "x-nemo-relay-internal-dispatch-url";
const INTERNAL_DISPATCH_ROUTE_HEADER: &str = "x-nemo-relay-internal-dispatch-route";
const INTERNAL_DISPATCH_BACKEND_HEADER: &str = "x-nemo-relay-internal-dispatch-backend";
const INTERNAL_RETRY_AWARE_HEADER: &str = "x-nemo-relay-internal-retry-aware";
const MAX_UPSTREAM_ERROR_BODY_BYTES: usize = 64 * 1024;
const CAPTURED_UPSTREAM_FAILURE_PREFIX: &str = "nemo-relay-gateway-upstream-attempt:";

/// Runs an authenticated daemon request through the same execution chain as the gateway.
/// The worker strips hop credentials and untrusted dispatch controls before calling this.
pub(crate) async fn daemon_execute(
    state: AppState,
    request: Request<Body>,
    operational: OperationalContext,
) -> Result<Response<Body>, CliError> {
    let path = request.uri().path().to_owned();
    let authorization = crate::provider_auth::ProviderRequestAuthorization {
        source_credential:
            crate::provider_auth::SourceCredentialDisposition::RelayProxyCredential {
                provider_credential_present: crate::provider_auth::has_provider_credential(
                    request.headers(),
                ),
            },
        allow_environment_provider_auth: daemon_allows_environment_provider_auth(request.headers()),
    };
    let mut prepared =
        prepare_gateway_request(&state.config, request, authorization, &path).await?;
    let start = take_llm_gateway_start(&mut prepared);
    let prep = state
        .sessions
        .prepare_gateway_call(&prepared.headers, start)
        .await?;
    // Balance daemon session accounting if the caller disconnects before response headers.
    // After setup, streaming delivery owns its own guard; unary/error paths finish the call.
    let mut guard = GatewayCallGuard::new(
        state.sessions.clone(),
        prep.session_id.clone(),
        prep.owner_subagent_id.clone(),
        None,
        prep.session_finish,
        operational.clone(),
    );
    let response_timeout = state.config.response_timeout();
    let execution = run_managed_gateway(state, prepared, prep, operational);
    let result = match response_timeout {
        Some(timeout) => match tokio::time::timeout(timeout, execution).await {
            Ok(result) => result,
            Err(_) => return Err(CliError::Launch("provider response-head timeout".into())),
        },
        None => execution.await,
    };
    guard.sessions.take();
    result
}

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
    let operational = OperationalContext::new_gateway(request.headers_mut());
    let request_path = request.uri().path().to_string();
    let (authorization, provider_path) =
        state.authorize_provider_request(request.headers_mut(), &request_path)?;
    let mut prepared = match prepare_gateway_request(
        &state.config,
        request,
        authorization,
        &provider_path,
    )
    .await
    {
        Ok(prepared) => prepared,
        Err(error) => {
            if matches!(error, CliError::PayloadTooLarge(_)) {
                operational::limit_exceeded(
                    &operational,
                    "gateway",
                    "max_passthrough_body_bytes",
                    state.config.max_passthrough_body_bytes,
                );
            }
            return Err(error);
        }
    };
    let start = take_llm_gateway_start(&mut prepared);
    let prep = state
        .sessions
        .prepare_gateway_call(&prepared.headers, start)
        .await?;
    run_managed_gateway(state, prepared, prep, operational).await
}

/// Upgrades an authorized provider WebSocket without entering the managed LLM pipeline.
pub(crate) async fn websocket_passthrough(
    State(state): State<AppState>,
    mut request: Request<Body>,
) -> Result<Response<Body>, CliError> {
    state.touch();
    let path = request.uri().path().to_owned();
    let (authorization, path) = state.authorize_provider_request(request.headers_mut(), &path)?;
    if !websocket::supported_path(&path) {
        return Err(CliError::InvalidPayload(
            "unsupported WebSocket path".into(),
        ));
    }
    let extensions = std::mem::take(request.extensions_mut());
    let prepared = prepare_gateway_request(&state.config, request, authorization, &path).await?;
    let headers = inject_provider_auth(
        state
            .http
            .get(&prepared.upstream_url)
            .headers(prepared.headers.clone()),
        prepared.provider,
        prepared.authorization.allow_environment_provider_auth,
        &state.config,
    )
    .await?
    .headers()
    .clone();
    let mut request = Request::new(Body::empty());
    *request.method_mut() = prepared.method;
    *request.headers_mut() = headers;
    *request.extensions_mut() = extensions;
    websocket::forward_to_provider(
        prepared.client_named_upstream,
        prepared.authorization.allow_environment_provider_auth
            && prepared.provider.has_helper(&state.config),
        request,
        &prepared.upstream_url,
        state.hold_websocket(),
        &state.config,
        std::future::pending(),
    )
    .await
    .map_err(|e| CliError::Launch(e.to_string()))
}

/// Transparently proxies compaction, image generation, and voice call setup without emitting LLM events.
///
/// These APIs do not use the managed text-generation request and response schemas.
pub(crate) async fn unmanaged_passthrough(
    State(state): State<AppState>,
    mut request: Request<Body>,
) -> Result<Response<Body>, CliError> {
    state.touch();
    let operational = OperationalContext::new_gateway(request.headers_mut());
    let request_path = request.uri().path().to_string();
    let (authorization, provider_path) =
        state.authorize_provider_request(request.headers_mut(), &request_path)?;
    let prepared = match prepare_gateway_request(
        &state.config,
        request,
        authorization,
        &provider_path,
    )
    .await
    {
        Ok(prepared) => prepared,
        Err(error) => {
            if matches!(error, CliError::PayloadTooLarge(_)) {
                operational::limit_exceeded(
                    &operational,
                    "gateway",
                    "max_passthrough_body_bytes",
                    state.config.max_passthrough_body_bytes,
                );
            }
            return Err(error);
        }
    };
    run_unmanaged_gateway(state, prepared, Some(operational)).await
}

/// Exact failure material from one ordinary upstream attempt.
///
/// Retry-aware routing attempts use [`FlowError::Upstream`].
/// Ordinary attempts keep their original HTTP bytes and multi-value headers so
/// the selected attempt can be relayed without a shared last-write-wins slot.
enum CapturedUpstreamFailure {
    Response {
        status: StatusCode,
        headers: HeaderMap,
        bytes: Bytes,
    },
    Transport(reqwest::Error),
    Authentication(CliError),
}

#[derive(Default)]
struct CapturedUpstreamFailures {
    entries: Mutex<HashMap<uuid::Uuid, CapturedUpstreamFailure>>,
}

impl CapturedUpstreamFailures {
    fn capture(&self, failure: CapturedUpstreamFailure) -> FlowError {
        let token = uuid::Uuid::now_v7();
        self.entries
            .lock()
            .expect("captured upstream failures lock poisoned")
            .insert(token, failure);
        FlowError::Internal(format!("{CAPTURED_UPSTREAM_FAILURE_PREFIX}{token}"))
    }

    fn take(&self, error: &FlowError) -> Option<CapturedUpstreamFailure> {
        let FlowError::Internal(message) = error else {
            return None;
        };
        let token = message
            .strip_prefix(CAPTURED_UPSTREAM_FAILURE_PREFIX)?
            .parse()
            .ok()?;
        self.entries
            .lock()
            .expect("captured upstream failures lock poisoned")
            .remove(&token)
    }
}

type CapturedUpstreamFailuresRef = Arc<CapturedUpstreamFailures>;

// Runs the managed pipeline for a prepared gateway request. Streaming and non-streaming branches
// share the same prep + codec dispatch but diverge in how the runtime drives the upstream call.
async fn run_managed_gateway(
    state: AppState,
    prepared: PreparedGatewayRequest,
    prep: GatewayCallPrep,
    operational: OperationalContext,
) -> Result<Response<Body>, CliError> {
    let operational = operational.with_session(prep.session_id.clone());
    if prep.bypass_managed_pipeline {
        let session_id = prep.session_id.clone();
        let session_finish = prep.session_finish;
        log::debug!(
            target: "nemo_relay.gateway",
            event = "observability_bypassed",
            operation_id = operational.operation_id(),
            session_id = session_id.as_str(),
            reason = "startup_probe";
            "Managed observability was bypassed for a startup probe"
        );
        state
            .sessions
            .finish_gateway_call(
                &session_id,
                prep.owner_subagent_id.as_deref(),
                session_finish,
            )
            .await;
        return run_unmanaged_gateway(state, prepared, Some(operational)).await;
    }
    let codecs = codecs_for_route(prepared.provider);
    if prepared.streaming {
        run_managed_streaming(state, prepared, prep, codecs, operational).await
    } else {
        run_managed_buffered(state, prepared, prep, codecs, operational).await
    }
}

async fn run_unmanaged_gateway(
    state: AppState,
    mut prepared: PreparedGatewayRequest,
    operational: Option<OperationalContext>,
) -> Result<Response<Body>, CliError> {
    websocket::strip_private_headers(&mut prepared.headers, false);
    if prepared.streaming {
        return passthrough_streaming(state, prepared, operational).await;
    }
    let response = forward_upstream_request(
        upstream_client(&state, prepared.client_named_upstream),
        &state.http_helper,
        UpstreamForwardRequest {
            method: &prepared.method,
            url: &prepared.upstream_url,
            body_bytes: &prepared.body_bytes,
            headers: &prepared.headers,
            effective_request: None,
            source_body: None,
            forwarding: ProviderForwarding::new(
                prepared.provider,
                prepared.authorization,
                &state.config,
            ),
            operational: operational.as_ref(),
            streaming: false,
        },
    )
    .await?;
    let status = response.status();
    let headers = response_headers(response.headers());
    if !status.is_success()
        && let Some(operational) = operational.as_ref()
    {
        operational::upstream_status(operational, status.as_u16());
    }
    let bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(error) => {
            if let Some(operational) = operational.as_ref() {
                operational::upstream_failed(operational, "response_read");
            }
            return Err(CliError::Upstream(error));
        }
    };
    if let Some(operational) = operational.as_ref() {
        operational::upstream_completed(operational, false);
    }
    build_response(status, headers, Body::from(bytes))
}

// Codecs registered for each managed provider route. Routes that emit LLM events but lack a typed
// codec (count_tokens) return `None` so the runtime still wraps the call but skips annotation.
struct RouteCodecs {
    request: Option<Arc<dyn LlmCodec>>,
    streaming: Option<Box<dyn StreamingCodec>>,
    response: Option<Arc<dyn LlmResponseCodec>>,
    source_body: SourceRequestBody,
}

// Retain the consumed source content only while upstream execution can retry. Queued
// event codecs hold weak references so middleware publication cannot prolong its lifetime.
#[derive(Clone, Default)]
struct SourceRequestBody(Arc<OnceLock<Value>>);

impl SourceRequestBody {
    fn matches(&self, content: &Value) -> bool {
        !content.is_null() && self.0.get().is_some_and(|source| source == content)
    }
}

struct GatewayRequestCodec {
    inner: Arc<dyn LlmCodec>,
    source_body: Weak<OnceLock<Value>>,
}

impl LlmCodec for GatewayRequestCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        self.inner.codec_identity()
    }

    fn decode(&self, request: &LlmRequest) -> nemo_relay::error::Result<AnnotatedLlmRequest> {
        self.inner.decode(request)
    }

    fn encode(
        &self,
        annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> nemo_relay::error::Result<LlmRequest> {
        let baseline = self.inner.decode(original)?;
        self.encode_with_baseline(annotated, original, &baseline)
    }

    fn encode_with_baseline(
        &self,
        annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
        baseline: &AnnotatedLlmRequest,
    ) -> nemo_relay::error::Result<LlmRequest> {
        let original_streaming = baseline.stream.unwrap_or(false);
        let edited_streaming = annotated.stream.unwrap_or(false);
        if edited_streaming != original_streaming {
            return Err(FlowError::InvalidArgument(
                "gateway request interceptors cannot change stream mode; preserve the original stream value"
                    .into(),
            ));
        }
        self.inner
            .encode_with_baseline(annotated, original, baseline)
    }

    fn encode_owned_with_baseline(
        &self,
        annotated: &AnnotatedLlmRequest,
        original: LlmRequest,
        baseline: &AnnotatedLlmRequest,
    ) -> nemo_relay::error::Result<LlmRequest> {
        let mut encoded = self.encode_with_baseline(annotated, &original, baseline)?;
        encoded.headers.extend(original.headers);
        if encoded.content == original.content
            && let Some(source_body) = self.source_body.upgrade()
        {
            let _ = source_body.set(original.content);
        }
        Ok(encoded)
    }
}

fn codecs_for_route(route: ProviderRoute) -> RouteCodecs {
    let source_body = SourceRequestBody::default();
    match route.provider_surface() {
        Some(surface) => RouteCodecs {
            request: Some(Arc::new(GatewayRequestCodec {
                inner: build_request_codec(surface),
                source_body: Arc::downgrade(&source_body.0),
            })),
            streaming: Some(build_streaming_codec(surface)),
            response: Some(build_response_codec(surface)),
            source_body,
        },
        None => RouteCodecs {
            request: None,
            streaming: None,
            response: None,
            source_body,
        },
    }
}

// Runs a non-streaming gateway request through `llm_call_execute`. The runtime handles start/end
// events and codec annotation; the gateway only sends the upstream request and parses its JSON.
async fn run_managed_buffered(
    state: AppState,
    prepared: PreparedGatewayRequest,
    prep: GatewayCallPrep,
    codecs: RouteCodecs,
    operational: OperationalContext,
) -> Result<Response<Body>, CliError> {
    let upstream_failures = Arc::new(CapturedUpstreamFailures::default());
    let func = build_buffered_func(
        state.clone(),
        &prepared,
        codecs.source_body,
        upstream_failures.clone(),
        operational.clone(),
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
        .codec_opt(codecs.request)
        .response_codec_opt(codecs.response)
        .build();
    let result = TASK_SCOPE_STACK
        .scope(scope_stack, async move { llm_call_execute(params).await })
        .await;
    match result {
        Ok(response_json) => {
            operational::upstream_completed(&operational, false);
            let mut headers = HeaderMap::new();
            headers.insert(
                http::header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            );
            let response_body = Body::from(response_json.to_string());
            state
                .sessions
                .record_gateway_response_hints(
                    &session_id,
                    owner_subagent_id.clone(),
                    Cow::Owned(response_json),
                )
                .await;
            state
                .sessions
                .finish_gateway_call(&session_id, owner_subagent_id.as_deref(), session_finish)
                .await;
            build_response(StatusCode::OK, headers, response_body)
        }
        Err(error) => {
            state
                .sessions
                .finish_gateway_call(&session_id, owner_subagent_id.as_deref(), session_finish)
                .await;
            if let Some(failure) = upstream_failures.take(&error) {
                return selected_upstream_failure(failure);
            }
            operational::upstream_failed(&operational, "runtime");
            Err(translate_runtime_error(error))
        }
    }
}

// Builds the managed-execution callback for a non-streaming route. Every failure carries either
// retry-routing data or an opaque token for that exact attempt, so concurrent attempts cannot
// overwrite one another's transport metadata.
fn build_buffered_func(
    state: AppState,
    prepared: &PreparedGatewayRequest,
    source_body: SourceRequestBody,
    upstream_failures: CapturedUpstreamFailuresRef,
    operational: OperationalContext,
) -> LlmExecutionNextFn {
    let http = upstream_client(&state, prepared.client_named_upstream).clone();
    let helper_http = state.http_helper.clone();
    let method = prepared.method.clone();
    let url = prepared.upstream_url.clone();
    let body_bytes = prepared.body_bytes.clone();
    let headers = prepared.headers.clone();
    let forwarding =
        ProviderForwarding::new(prepared.provider, prepared.authorization, &state.config);
    Arc::new(move |request| {
        let http = http.clone();
        let helper_http = helper_http.clone();
        let forwarding = forwarding.clone();
        let method = method.clone();
        let url = url.clone();
        let body_bytes = body_bytes.clone();
        let headers = headers.clone();
        let upstream_failures = upstream_failures.clone();
        let operational = operational.clone();
        let source_body = source_body.clone();
        Box::pin(async move {
            let retry_aware = retry_aware_dispatch(&request);
            let response = match forward_upstream_request(
                &http,
                &helper_http,
                UpstreamForwardRequest {
                    method: &method,
                    url: &url,
                    body_bytes: &body_bytes,
                    headers: &headers,
                    effective_request: Some(&request),
                    source_body: Some(&source_body),
                    forwarding,
                    operational: Some(&operational),
                    streaming: false,
                },
            )
            .await
            {
                Ok(response) => response,
                Err(CliError::Upstream(error)) if retry_aware => {
                    return Err(FlowError::Upstream(transport_failure(&error)));
                }
                Err(CliError::Upstream(error)) => {
                    return Err(
                        upstream_failures.capture(CapturedUpstreamFailure::Transport(error))
                    );
                }
                Err(error) => {
                    return Err(
                        upstream_failures.capture(CapturedUpstreamFailure::Authentication(error))
                    );
                }
            };
            let status = response.status();
            if !status.is_success() {
                operational::upstream_status(&operational, status.as_u16());
            }
            let response_headers = response_headers(response.headers());
            let bytes = match response.bytes().await {
                Ok(bytes) => bytes,
                Err(error) if retry_aware => {
                    operational::upstream_failed(&operational, "response_read");
                    return Err(FlowError::Upstream(transport_failure(&error)));
                }
                Err(error) => {
                    operational::upstream_failed(&operational, "response_read");
                    return Err(
                        upstream_failures.capture(CapturedUpstreamFailure::Transport(error))
                    );
                }
            };
            if !status.is_success() {
                if retry_aware {
                    return Err(FlowError::Upstream(http_failure(
                        status,
                        &response_headers,
                        &bytes,
                    )));
                }
                return Err(
                    upstream_failures.capture(CapturedUpstreamFailure::Response {
                        status,
                        headers: response_headers,
                        bytes,
                    }),
                );
            }
            match serde_json::from_slice::<Value>(&bytes) {
                Ok(response) => Ok(response),
                Err(_) if retry_aware => {
                    Err(FlowError::Upstream(invalid_json_failure(&response_headers)))
                }
                Err(_) => Err(
                    upstream_failures.capture(CapturedUpstreamFailure::Response {
                        status,
                        headers: response_headers,
                        bytes,
                    }),
                ),
            }
        })
    })
}

/// Runs a gateway stream through its provider codec and the managed LLM lifecycle.
///
/// The codec checks completion before the runtime finalizes the call. The gateway then
/// encodes each parsed event as an SSE frame for the client.
async fn run_managed_streaming(
    state: AppState,
    prepared: PreparedGatewayRequest,
    prep: GatewayCallPrep,
    codecs: RouteCodecs,
    operational: OperationalContext,
) -> Result<Response<Body>, CliError> {
    let upstream_failures = Arc::new(CapturedUpstreamFailures::default());
    let func = build_streaming_func(
        state.clone(),
        &prepared,
        codecs.source_body,
        upstream_failures.clone(),
        operational.clone(),
    );
    let provider_route = prepared.provider;

    // Streaming routes that lack a codec fall back to byte passthrough. The runtime requires a
    // collector and finalizer for managed streaming, so without a codec we cannot use the managed
    // pipeline. This keeps non-LLM streaming paths working while typed codecs remain optional.
    let Some(streaming_codec) = codecs.streaming else {
        let session_finish = prep.session_finish;
        state
            .sessions
            .finish_gateway_call(
                &prep.session_id,
                prep.owner_subagent_id.as_deref(),
                session_finish,
            )
            .await;
        return passthrough_streaming(state, prepared, Some(operational)).await;
    };
    let streaming_codec: Arc<dyn StreamingCodec> = Arc::from(streaming_codec);
    let codec_for_stream = Arc::clone(&streaming_codec);
    let func: LlmStreamExecutionNextFn = Arc::new(move |request| {
        let future = func(request);
        let codec = Arc::clone(&codec_for_stream);
        Box::pin(async move { Ok(codec.validate_stream(future.await?)) })
    });
    let collector = streaming_codec.collector();
    let borrowed_collector = streaming_codec.borrowed_collector();
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
        bypass_managed_pipeline: _,
        session_finish,
    } = prep;
    let (finalizer, response_completion, response_ready) = gateway_response_completion(
        state.sessions.clone(),
        session_id.clone(),
        owner_subagent_id.clone(),
        finalizer,
    );
    let params = LlmStreamCallExecuteParams::builder()
        .name(provider_name)
        .request(request)
        .func(func)
        .collector(collector)
        .borrowed_collector_opt(borrowed_collector)
        .finalizer(finalizer)
        .parent_opt(parent)
        .attributes(attributes)
        .metadata(metadata)
        .model_name_opt(model_name)
        .codec_opt(codecs.request)
        .response_codec_opt(codecs.response)
        .build();
    let json_stream_result = TASK_SCOPE_STACK
        .scope(scope_stack, async move {
            execute_gateway_stream(params, response_completion).await
        })
        .await;
    let json_stream = match json_stream_result {
        Ok(json_stream) => json_stream,
        Err(error) => {
            state
                .sessions
                .finish_gateway_call(&session_id, owner_subagent_id.as_deref(), session_finish)
                .await;
            if let Some(failure) = upstream_failures.take(&error) {
                return selected_upstream_failure(failure);
            }
            operational::upstream_failed(&operational, "runtime");
            return Err(translate_runtime_error(error));
        }
    };
    let mut headers = HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    let guard = GatewayCallGuard::new(
        state.sessions.clone(),
        session_id.clone(),
        owner_subagent_id,
        Some(response_ready),
        session_finish,
        operational,
    );
    let body = client_sse_body(json_stream, provider_route, guard);

    build_response(StatusCode::OK, headers, body)
}

// Borrow the aggregate before runtime sanitization. Complete session bookkeeping
// inline when uncontended, so client completion does not queue behind unrelated
// response annotation. The observer retains the async fallback for busy sessions.
fn gateway_response_completion(
    sessions: SessionManager,
    session_id: String,
    owner_subagent_id: Option<String>,
    finalizer: LlmFinalizerFn,
) -> (LlmFinalizerFn, ResponseCompletionFn, oneshot::Receiver<()>) {
    let (ready, receiver) = oneshot::channel();
    let ready = Arc::new(Mutex::new(Some(ready)));
    let finalizer_ready = ready.clone();
    let finalizer_sessions = sessions.clone();
    let finalizer_session_id = session_id.clone();
    let finalizer_owner = owner_subagent_id.clone();
    let finalizer: LlmFinalizerFn = Box::new(move || {
        let response = finalizer();
        if finalizer_sessions.try_record_gateway_response_hints(
            &finalizer_session_id,
            finalizer_owner,
            &response,
        ) && let Some(ready) = finalizer_ready
            .lock()
            .expect("stream completion lock poisoned")
            .take()
        {
            let _ = ready.send(());
        }
        response
    });
    let observer: ResponseCompletionFn = Box::new(move |response| {
        let ready = ready
            .lock()
            .expect("stream completion lock poisoned")
            .take();
        Box::pin(async move {
            if let Some(ready) = ready {
                sessions
                    .record_gateway_response_hints(
                        &session_id,
                        owner_subagent_id,
                        Cow::Borrowed(response),
                    )
                    .await;
                let _ = ready.send(());
            }
            Ok(())
        })
    });
    (finalizer, observer, receiver)
}

// Builds the streaming managed-execution callback. Each failure carries either retry-routing data
// or an opaque token for that exact attempt, without publishing a shared last-response slot.
fn build_streaming_func(
    state: AppState,
    prepared: &PreparedGatewayRequest,
    source_body: SourceRequestBody,
    upstream_failures: CapturedUpstreamFailuresRef,
    operational: OperationalContext,
) -> LlmStreamExecutionNextFn {
    let http = upstream_client(&state, prepared.client_named_upstream).clone();
    let helper_http = state.http_helper.clone();
    let method = prepared.method.clone();
    let url = prepared.upstream_url.clone();
    let body_bytes = prepared.body_bytes.clone();
    let headers = prepared.headers.clone();
    let forwarding =
        ProviderForwarding::new(prepared.provider, prepared.authorization, &state.config);
    Arc::new(move |request| {
        let http = http.clone();
        let helper_http = helper_http.clone();
        let forwarding = forwarding.clone();
        let method = method.clone();
        let url = url.clone();
        let body_bytes = body_bytes.clone();
        let headers = headers.clone();
        let upstream_failures = upstream_failures.clone();
        let operational = operational.clone();
        let source_body = source_body.clone();
        Box::pin(async move {
            let retry_aware = retry_aware_dispatch(&request);
            let response = match forward_upstream_request(
                &http,
                &helper_http,
                UpstreamForwardRequest {
                    method: &method,
                    url: &url,
                    body_bytes: &body_bytes,
                    headers: &headers,
                    effective_request: Some(&request),
                    source_body: Some(&source_body),
                    forwarding,
                    operational: Some(&operational),
                    streaming: true,
                },
            )
            .await
            {
                Ok(response) => response,
                Err(CliError::Upstream(error)) if retry_aware => {
                    return Err(FlowError::Upstream(transport_failure(&error)));
                }
                Err(CliError::Upstream(error)) => {
                    return Err(
                        upstream_failures.capture(CapturedUpstreamFailure::Transport(error))
                    );
                }
                Err(error) => {
                    return Err(
                        upstream_failures.capture(CapturedUpstreamFailure::Authentication(error))
                    );
                }
            };
            let status = response.status();
            if !status.is_success() {
                operational::upstream_status(&operational, status.as_u16());
            }
            let response_headers = response_headers(response.headers());
            if !status.is_success() {
                let bytes = match response.bytes().await {
                    Ok(bytes) => bytes,
                    Err(error) if retry_aware => {
                        operational::upstream_failed(&operational, "response_read");
                        return Err(FlowError::Upstream(transport_failure(&error)));
                    }
                    Err(error) => {
                        operational::upstream_failed(&operational, "response_read");
                        return Err(
                            upstream_failures.capture(CapturedUpstreamFailure::Transport(error))
                        );
                    }
                };
                if retry_aware {
                    return Err(FlowError::Upstream(http_failure(
                        status,
                        &response_headers,
                        &bytes,
                    )));
                }
                return Err(
                    upstream_failures.capture(CapturedUpstreamFailure::Response {
                        status,
                        headers: response_headers,
                        bytes,
                    }),
                );
            }
            let json_stream = sse_json_stream(response, operational);
            Ok(json_stream)
        })
    })
}

// Decodes an upstream SSE byte stream into a stream of parsed `data:` JSON payloads. Frames with no
// `data:` line (heartbeats), comments, and the `data: [DONE]` sentinel are filtered out by the
// shared `SseEventDecoder`. Trailing partial frames are surfaced to the runtime so the collector
// observes whatever the upstream sent before disconnect.
fn sse_json_stream(response: reqwest::Response, operational: OperationalContext) -> LlmJsonStream {
    sse_json_stream_with_thresholds(
        response,
        operational,
        std::time::Duration::from_millis(UPSTREAM_RESPONSE_THRESHOLD_MILLIS),
        std::time::Duration::from_millis(UPSTREAM_STREAM_STALL_THRESHOLD_MILLIS),
    )
}

fn sse_json_stream_with_thresholds(
    response: reqwest::Response,
    operational: OperationalContext,
    first_event_threshold: std::time::Duration,
    stall_threshold: std::time::Duration,
) -> LlmJsonStream {
    use nemo_relay::codec::streaming::SseEventDecoder;
    let mut decoder = SseEventDecoder::new();
    let mut bytes = response.bytes_stream();
    let stream = stream! {
        let mut first_event = true;
        let mut first_event_warned = false;
        let mut last_event_at = tokio::time::Instant::now();
        loop {
            let threshold = if first_event {
                first_event_threshold
            } else {
                stall_threshold
            };
            let already_warned = first_event && first_event_warned;
            let mut next = Box::pin(bytes.next());
            let chunk = if already_warned {
                next.await
            } else {
                let mut deadline = last_event_at + threshold;
                loop {
                    let delay = tokio::time::sleep_until(deadline);
                    tokio::pin!(delay);
                    let chunk = tokio::select! {
                        chunk = &mut next => Some(chunk),
                        _ = &mut delay => None,
                    };
                    if let Some(chunk) = chunk {
                        break chunk;
                    }
                    operational::upstream_delayed(
                        &operational,
                        if first_event {
                            "upstream_first_event_delayed"
                        } else {
                            "upstream_stream_stalled"
                        },
                        threshold.as_millis() as u64,
                    );
                    if first_event {
                        first_event_warned = true;
                        break next.await;
                    }
                    last_event_at = tokio::time::Instant::now();
                    deadline = last_event_at + threshold;
                }
            };
            let Some(chunk) = chunk else {
                break;
            };
            match chunk {
                Ok(buffer) => {
                    for result in decoder.push_bytes_results(&buffer) {
                        match result {
                            Ok(event) => {
                                if first_event {
                                    operational::upstream_first_event(&operational);
                                    first_event = false;
                                }
                                last_event_at = tokio::time::Instant::now();
                                yield Ok(event.data);
                            }
                            Err(error) => {
                                operational::upstream_stream_read_failed(&operational);
                                yield Err(error);
                                return;
                            }
                        }
                    }
                }
                Err(error) => {
                    operational::upstream_stream_read_failed(&operational);
                    yield Err(FlowError::Internal(error.to_string()));
                    return;
                }
            }
        }
        match decoder.finish() {
            Ok(Some(event)) => {
                if first_event {
                    operational::upstream_first_event(&operational);
                }
                yield Ok(event.data);
            }
            Ok(None) => {}
            Err(error) => {
                operational::upstream_stream_read_failed(&operational);
                yield Err(error);
                return;
            }
        }
        operational::upstream_completed(&operational, true);
    };
    LlmJsonStream::new(stream)
}

// Re-encodes a runtime JSON stream as `text/event-stream` frames for the downstream client. Event
// names are reconstructed from the JSON `type` field where providers populate it (Anthropic
// Messages, OpenAI Responses); OpenAI Chat omits the `event:` line and appends the original
// `data: [DONE]` terminator after the runtime stream completes.
fn client_sse_body(
    json_stream: LlmJsonStream,
    route: ProviderRoute,
    mut guard: GatewayCallGuard,
) -> Body {
    guard.json_stream = Some(json_stream);
    let stream = stream! {
        while let Some(item) = guard.json_stream.as_mut().expect("gateway stream installed").next().await {
            match item {
                Ok(event_json) => {
                    yield Ok::<Bytes, CliError>(encode_sse_frame(&event_json, route));
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
    response_ready: Option<oneshot::Receiver<()>>,
    json_stream: Option<LlmJsonStream>,
    session_finish: GatewaySessionFinish,
    operational: OperationalContext,
}

impl GatewayCallGuard {
    fn new(
        sessions: SessionManager,
        session_id: String,
        owner_subagent_id: Option<String>,
        response_ready: Option<oneshot::Receiver<()>>,
        session_finish: GatewaySessionFinish,
        operational: OperationalContext,
    ) -> Self {
        Self {
            sessions: Some(sessions),
            session_id,
            owner_subagent_id,
            response_ready,
            json_stream: None,
            session_finish,
            operational,
        }
    }

    async fn finish(&mut self) {
        // Drop first so an interrupted managed stream schedules its observer.
        drop(self.json_stream.take());
        // Keep ownership in the guard across this await so cancellation still
        // runs session cleanup through Drop.
        if let Some(ready) = self.response_ready.as_mut() {
            let _ = ready.await;
        }
        self.response_ready = None;
        if let Some(sessions) = self.sessions.take() {
            complete_gateway_call(
                sessions,
                self.session_id.clone(),
                self.owner_subagent_id.clone(),
                None,
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
    response_ready: Option<oneshot::Receiver<()>>,
    session_finish: GatewaySessionFinish,
) {
    if let Some(ready) = response_ready {
        let _ = ready.await;
    }
    sessions
        .finish_gateway_call(&session_id, owner_subagent_id.as_deref(), session_finish)
        .await;
}

impl Drop for GatewayCallGuard {
    fn drop(&mut self) {
        drop(self.json_stream.take());
        let Some(sessions) = self.sessions.take() else {
            return;
        };
        operational::upstream_cancelled(&self.operational);
        let session_id = self.session_id.clone();
        let owner_subagent_id = self.owner_subagent_id.clone();
        let session_finish = self.session_finish;
        let cleanup = complete_gateway_call(
            sessions,
            session_id,
            owner_subagent_id,
            self.response_ready.take(),
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
fn encode_sse_frame(event_json: &Value, route: ProviderRoute) -> Bytes {
    let event_name = match route {
        ProviderRoute::AnthropicMessages | ProviderRoute::OpenAiResponses => {
            event_json.get("type").and_then(Value::as_str)
        }
        _ => None,
    };
    let mut frame = Vec::with_capacity(64);
    if let Some(name) = event_name {
        frame.extend_from_slice(b"event: ");
        frame.extend_from_slice(name.as_bytes());
        frame.push(b'\n');
    }
    frame.extend_from_slice(b"data: ");
    if serde_json::to_writer(&mut frame, event_json).is_err() {
        frame.extend_from_slice(b"null");
    }
    frame.extend_from_slice(b"\n\n");
    Bytes::from(frame)
}

// Forwards the buffered request to the upstream provider with only the safe request headers. This
// is shared by the buffered and streaming managed funcs so header filtering stays consistent.
// Source authentication is normalized at ingress, before interceptors can select a target.
async fn forward_upstream_request(
    http: &reqwest::Client,
    helper_http: &reqwest::Client,
    request: UpstreamForwardRequest<'_>,
) -> Result<reqwest::Response, CliError> {
    let UpstreamForwardRequest {
        method,
        url,
        body_bytes,
        headers,
        effective_request,
        source_body,
        forwarding,
        operational,
        streaming,
    } = request;
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
        source_body,
    );
    let allow_configured_auth = matches!(
        effective.credential_policy,
        TargetCredentialPolicy::SourceOrEnvironment
    ) && forwarding.authorization.allow_environment_provider_auth;
    let http = if allow_configured_auth && effective.target_route.has_helper(&forwarding.config) {
        helper_http
    } else {
        http
    };
    let mut upstream = http
        .request(method.clone(), &effective.url)
        .body(effective.body_bytes.clone());
    for (name, value) in &effective.headers {
        if should_forward_request_header(name, &effective.headers) {
            upstream = upstream.header(name, value);
        }
    }
    let upstream = inject_provider_auth(
        upstream,
        effective.target_route,
        allow_configured_auth,
        &forwarding.config,
    )
    .await?;
    if let Some(operational) = operational {
        operational::upstream_started(operational, streaming);
        let request = http.execute(upstream);
        tokio::pin!(request);
        let response = tokio::select! {
            response = &mut request => response,
            _ = tokio::time::sleep(std::time::Duration::from_millis(UPSTREAM_RESPONSE_THRESHOLD_MILLIS)) => {
                operational::upstream_delayed(
                    operational,
                    "upstream_headers_delayed",
                    UPSTREAM_RESPONSE_THRESHOLD_MILLIS,
                );
                request.await
            }
        };
        if response.is_ok() {
            operational::upstream_headers_received(operational, streaming);
        } else {
            operational::upstream_failed(operational, "transport");
        }
        response.map_err(CliError::from)
    } else {
        http.execute(upstream).await.map_err(CliError::from)
    }
}

struct UpstreamForwardRequest<'a> {
    method: &'a Method,
    url: &'a str,
    body_bytes: &'a Bytes,
    headers: &'a HeaderMap,
    effective_request: Option<&'a LlmRequest>,
    source_body: Option<&'a SourceRequestBody>,
    forwarding: ProviderForwarding,
    operational: Option<&'a OperationalContext>,
    streaming: bool,
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
        None,
    );
    (effective.body_bytes, effective.headers)
}

fn effective_dispatch_request(
    body_bytes: &Bytes,
    headers: &HeaderMap,
    effective_request: Option<&LlmRequest>,
    url: &str,
    route: ProviderRoute,
    source_body: Option<&SourceRequestBody>,
) -> EffectiveUpstreamRequest {
    let mut headers = headers.clone();
    strip_internal_dispatch_headers(&mut headers);
    let Some(request) = effective_request else {
        return source_request(body_bytes, headers, url, route);
    };
    let body_unchanged = source_body.is_some_and(|source| source.matches(&request.content));
    let Some((body_bytes, body_reencoded)) =
        reencode_request_body(request, body_bytes, body_unchanged)
    else {
        return source_request(body_bytes, headers, url, route);
    };
    let overrides = dispatch_overrides(&request.headers);
    let credential_policy = if overrides.is_explicit_target() {
        crate::provider_auth::remove_provider_credentials(&mut headers);
        TargetCredentialPolicy::ExplicitTarget
    } else {
        TargetCredentialPolicy::SourceOrEnvironment
    };
    // Observable source headers exclude credentials. Applying the rewritten map only after source
    // credentials are removed lets an explicit target binding add its own authorization without
    // inheriting credentials intended for the original provider.
    apply_rewritten_headers(&mut headers, &request.headers, body_unchanged);
    if body_reencoded {
        headers.remove(header::CONTENT_ENCODING);
    }
    EffectiveUpstreamRequest {
        body_bytes,
        headers,
        url: overrides.resolve_url(url),
        target_route: overrides.route.unwrap_or(route),
        credential_policy,
    }
}

fn source_request(
    body_bytes: &Bytes,
    headers: HeaderMap,
    url: &str,
    route: ProviderRoute,
) -> EffectiveUpstreamRequest {
    EffectiveUpstreamRequest {
        body_bytes: body_bytes.clone(),
        headers,
        url: url.to_string(),
        target_route: route,
        credential_policy: TargetCredentialPolicy::SourceOrEnvironment,
    }
}

fn reencode_request_body(
    request: &LlmRequest,
    body_bytes: &Bytes,
    body_unchanged: bool,
) -> Option<(Bytes, bool)> {
    if request.content.is_null() || body_unchanged {
        return Some((body_bytes.clone(), false));
    }
    match serde_json::to_vec(&request.content) {
        Ok(serialized) => Some((Bytes::from(serialized), true)),
        Err(_) => {
            log::warn!(
                target: "nemo_relay.gateway",
                event = "request_rewrite_fallback",
                reason = "serialization_failed";
                "The original upstream request was retained after rewrite serialization failed"
            );
            None
        }
    }
}

struct DispatchOverrides {
    url: Option<String>,
    route: Option<ProviderRoute>,
    route_header_seen: bool,
}

impl DispatchOverrides {
    fn is_explicit_target(&self) -> bool {
        self.url.is_some() || self.route_header_seen
    }

    fn resolve_url(&self, fallback: &str) -> String {
        if self.route_header_seen && self.route.is_none() {
            fallback.to_string()
        } else {
            self.url.clone().unwrap_or_else(|| fallback.to_string())
        }
    }
}

fn dispatch_overrides(headers: &serde_json::Map<String, Value>) -> DispatchOverrides {
    let mut url = None;
    let mut route = None;
    let mut route_header_seen = false;
    for (name, value) in headers {
        if name.eq_ignore_ascii_case(INTERNAL_DISPATCH_URL_HEADER) {
            url = json_header_string(value);
        } else if name.eq_ignore_ascii_case(INTERNAL_DISPATCH_ROUTE_HEADER) {
            route_header_seen = true;
            route = json_header_string(value)
                .and_then(|value| ProviderRoute::from_dispatch_override(&value));
        }
    }
    DispatchOverrides {
        url,
        route,
        route_header_seen,
    }
}

fn apply_rewritten_headers(
    headers: &mut HeaderMap,
    request_headers: &serde_json::Map<String, Value>,
    preserve_content_encoding: bool,
) {
    for (name, value) in request_headers {
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        if is_internal_dispatch_header(&name)
            || (preserve_content_encoding && name == header::CONTENT_ENCODING)
        {
            continue;
        }
        let Some(value) = json_header_value(value) else {
            continue;
        };
        headers.insert(name, value);
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
    headers.remove(INTERNAL_DISPATCH_BACKEND_HEADER);
    headers.remove(INTERNAL_RETRY_AWARE_HEADER);
}

fn is_internal_dispatch_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        INTERNAL_DISPATCH_URL_HEADER
            | INTERNAL_DISPATCH_ROUTE_HEADER
            | INTERNAL_DISPATCH_BACKEND_HEADER
            | INTERNAL_RETRY_AWARE_HEADER
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
async fn inject_provider_auth(
    builder: reqwest::RequestBuilder,
    route: ProviderRoute,
    allow_configured_auth: bool,
    config: &crate::configuration::GatewayConfig,
) -> Result<reqwest::Request, CliError> {
    let mut request = builder.build()?;
    crate::upstream_auth::apply(
        request.headers_mut(),
        route.is_anthropic(),
        allow_configured_auth && route.allows_relay_auth(),
        config,
    )
    .await?;
    Ok(request)
}

// Plain byte passthrough used for streaming routes that lack a typed codec. The managed pipeline
// requires a collector + finalizer, so without a codec we keep the simpler proxy behavior and skip
// the LLM lifecycle event for that single request.
async fn passthrough_streaming(
    state: AppState,
    prepared: PreparedGatewayRequest,
    operational: Option<OperationalContext>,
) -> Result<Response<Body>, CliError> {
    let response = forward_upstream_request(
        upstream_client(&state, prepared.client_named_upstream),
        &state.http_helper,
        UpstreamForwardRequest {
            method: &prepared.method,
            url: &prepared.upstream_url,
            body_bytes: &prepared.body_bytes,
            headers: &prepared.headers,
            effective_request: None,
            source_body: None,
            forwarding: ProviderForwarding::new(
                prepared.provider,
                prepared.authorization,
                &state.config,
            ),
            operational: operational.as_ref(),
            streaming: true,
        },
    )
    .await?;
    let status = response.status();
    let headers = response_headers(response.headers());
    if !status.is_success()
        && let Some(operational) = operational.as_ref()
    {
        operational::upstream_status(operational, status.as_u16());
    }
    let mut bytes = response.bytes_stream();
    let operational_for_stream = operational;
    let body = Body::from_stream(stream! {
        let mut guard = UpstreamStreamGuard::new(operational_for_stream);
        while let Some(chunk) = bytes.next().await {
            if chunk.is_err() {
                if let Some(operational) = guard.context.as_ref() {
                    operational::upstream_stream_read_failed(operational);
                }
                guard.stop();
                yield chunk;
                return;
            }
            yield chunk;
        }
        guard.complete();
    });
    build_response(status, headers, body)
}

/// Logs a downstream stream cancellation when a raw passthrough body is dropped before its
/// upstream bytes are exhausted. Managed streams use [`GatewayCallGuard`] for the same purpose.
struct UpstreamStreamGuard {
    context: Option<OperationalContext>,
    completed: bool,
}

impl UpstreamStreamGuard {
    fn new(context: Option<OperationalContext>) -> Self {
        Self {
            context,
            completed: false,
        }
    }

    fn complete(&mut self) {
        self.completed = true;
        if let Some(context) = self.context.as_ref() {
            operational::upstream_completed(context, true);
        }
    }

    fn stop(&mut self) {
        self.completed = true;
    }
}

impl Drop for UpstreamStreamGuard {
    fn drop(&mut self) {
        if !self.completed
            && let Some(context) = self.context.as_ref()
        {
            operational::upstream_cancelled(context);
        }
    }
}

// Translates a runtime [`FlowError`] from managed execution into a gateway HTTP error. Provider
// callbacks surface transport and HTTP failures as typed upstream failures so no shared
// per-request error side channel is required.
fn translate_runtime_error(error: FlowError) -> CliError {
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

fn selected_upstream_failure(failure: CapturedUpstreamFailure) -> Result<Response<Body>, CliError> {
    match failure {
        CapturedUpstreamFailure::Response {
            status,
            headers,
            bytes,
        } => build_response(status, headers, Body::from(bytes)),
        CapturedUpstreamFailure::Transport(error) => Err(CliError::Upstream(error)),
        CapturedUpstreamFailure::Authentication(error) => Err(error),
    }
}

/// The HTTP client to dispatch with, chosen by where the destination came from.
///
/// Both clients stop at redirects to keep credentials and prompts at the chosen destination.
/// The client for caller-named destinations also checks resolved addresses. The client for
/// configured destinations keeps the operator's network access.
fn upstream_client(state: &AppState, client_named_upstream: bool) -> &reqwest::Client {
    if client_named_upstream {
        &state.http_no_redirect
    } else {
        &state.http
    }
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

fn invalid_json_failure(headers: &HeaderMap) -> UpstreamFailure {
    UpstreamFailure {
        status: None,
        body: "upstream returned a non-JSON success response".to_string(),
        headers: failure_headers(headers),
        class: UpstreamFailureClass::Other,
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

/// Proxies provider model-list requests without creating LLM runtime events.
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
    let request_path = parts.uri.path().to_string();
    let (authorization, provider_path) =
        state.authorize_provider_request(&mut parts.headers, &request_path)?;
    let provider = if is_anthropic_models_request(&provider_path, &parts.headers) {
        ProviderRoute::AnthropicModels
    } else {
        ProviderRoute::OpenAiModels
    };
    let path_and_query = request::provider_path_and_query(&provider_path, parts.uri.query());
    let mut allow_environment_provider_auth =
        authorization.allow_environment_provider_auth && provider.allows_relay_auth();
    parts.headers.remove(BOOTSTRAP_CLIENT_TOKEN_HEADER);
    let mut named_by_client = false;
    let agent_override = gateway_upstream_url_override(
        provider,
        &parts.headers,
        &path_and_query,
        allow_environment_provider_auth,
        &state.config,
    );
    let upstream_url = match agent_override {
        Some(url) => url,
        None => match crate::gateway::routes::client_named_upstream_url(
            provider,
            &parts.headers,
            &path_and_query,
            authorization.source_credential.is_relay_proxy_credential(),
        ) {
            crate::agents::pi::alignment::NamedUpstream::Named(url) => {
                named_by_client = true;
                url
            }
            // Refuse rather than reroute; see `gateway::request::prepare_gateway_request`.
            crate::agents::pi::alignment::NamedUpstream::Rejected(reason) => {
                return Err(CliError::InvalidPayload(reason.to_string()));
            }
            crate::agents::pi::alignment::NamedUpstream::Absent => {
                provider.upstream_url(&state.config, &path_and_query)
            }
        },
    };
    if named_by_client {
        // Same reasoning as the passthrough path: naming a destination is not a way to obtain a
        // credential for it. See `gateway::request::prepare_gateway_request`.
        allow_environment_provider_auth = false;
    }
    let sanitized = strip_replaceable_agent_auth_headers(
        &parts.headers,
        provider,
        &provider_path,
        allow_environment_provider_auth,
        &state.config,
    );
    let http = if allow_environment_provider_auth && provider.has_helper(&state.config) {
        &state.http_helper
    } else {
        upstream_client(&state, named_by_client)
    };
    let mut upstream = http.get(upstream_url);
    for (name, value) in &sanitized {
        if should_forward_request_header(name, &sanitized) {
            upstream = upstream.header(name, value);
        }
    }
    let upstream = inject_provider_auth(
        upstream,
        provider,
        allow_environment_provider_auth,
        &state.config,
    )
    .await?;
    let upstream_response = http.execute(upstream).await?;
    let status = upstream_response.status();
    let headers = response_headers(upstream_response.headers());
    let bytes = upstream_response.bytes().await?;
    build_response(status, headers, Body::from(bytes))
}
