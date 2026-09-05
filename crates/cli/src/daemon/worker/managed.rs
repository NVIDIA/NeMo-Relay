// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Managed worker execution that keeps response delivery on the raw frame path.

use std::collections::BTreeSet;
use std::error::Error as _;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};
use std::time::Duration;

use axum::Json;
use axum::body::{Body, Bytes};
use axum::http::header::{ACCEPT, AUTHORIZATION, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode, Uri};
use axum::response::IntoResponse;
use http_body_util::LengthLimitError;
use hyper::body::{Body as HttpBody, Frame, SizeHint};
use nemo_relay::api::llm::{
    LlmCallEndParams, LlmCallParams, LlmRequest, llm_call, llm_call_end, llm_conditional_execution,
    llm_request_intercepts,
};
use nemo_relay::api::registry::{
    RuntimeRegistrationIdentity, RuntimeRegistrationKind, list_runtime_registrations,
};
use nemo_relay::api::runtime::TASK_SCOPE_STACK;
use nemo_relay::codec::resolve::{ProviderSurface, request_codec, response_codec, streaming_codec};
use nemo_relay::codec::streaming::SseEventDecoder;
use nemo_relay::error::FlowError;
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc};

use super::super::common::control::{
    CLIENT_TOKEN_HEADER, WORKER_ROUTE_FAILURE_HEADER, WORKER_TOKEN_HEADER,
};
use super::super::common::routes::{HookRoute, ProviderRoute};
use super::super::common::transport::{
    BoxError, PooledClient, RelayBody, box_body, prepare_forward_request, prepare_forward_response,
};
use crate::agents::shared::adapters::{claude_code, codex, pi};
use crate::configuration::GatewayConfig;
use crate::error::CliError;
use crate::plugins::lifecycle::ActiveDynamicPluginComponent;
use crate::server::ServerPluginActivation;
use crate::sessions::{GatewayCallPrep, SessionManager};

const RESPONSE_HEAD_TIMEOUT: Duration = Duration::from_secs(60);
const OBSERVATION_QUEUE_FRAMES: usize = 32;
const DEFAULT_OBSERVATION_CAPTURE_BYTES: usize = 4 * 1024 * 1024;
const OBSERVATION_CAPTURE_BYTES_ENV: &str = "NEMO_RELAY_DAEMON_OBSERVATION_CAPTURE_BYTES";
const OBSERVATION_ACTIVE: u8 = 0;
const OBSERVATION_COMPLETE: u8 = 1;
const OBSERVATION_BODY_ERROR: u8 = 2;
const OBSERVATION_CANCELLED: u8 = 3;
const STREAM_MODE_MUTATION_ERROR: &str =
    "daemon worker request middleware cannot change stream mode";
const INTERNAL_HEADER_PREFIX: &str = "x-nemo-relay-";
const INTERNAL_DISPATCH_URL_HEADER: &str = "x-nemo-relay-internal-dispatch-url";
const INTERNAL_DISPATCH_ROUTE_HEADER: &str = "x-nemo-relay-internal-dispatch-route";
const INTERNAL_DISPATCH_BACKEND_HEADER: &str = "x-nemo-relay-internal-dispatch-backend";
const INTERNAL_RETRY_AWARE_HEADER: &str = "x-nemo-relay-internal-retry-aware";

/// Runtime-owned plugin activation, hook sessions, and response observation.
pub(super) struct ManagedRuntime {
    config: GatewayConfig,
    sessions: SessionManager,
    owner: String,
    observation_capture_bytes: usize,
    activation: Mutex<Option<ServerPluginActivation>>,
}

impl ManagedRuntime {
    pub(super) async fn initialize(
        config: GatewayConfig,
        dynamic_plugins: Vec<ActiveDynamicPluginComponent>,
        owner: String,
    ) -> Result<Self, CliError> {
        let observation_capture_bytes = observation_capture_limit_from_environment()?;
        let activation =
            crate::server::initialize_plugin_host(config.plugin_config.clone(), dynamic_plugins)
                .await?;
        if let Err(error) = reject_incompatible_execution_middleware() {
            if let Some(activation) = activation {
                let _ = activation.clear();
            }
            return Err(error);
        }
        let sessions = SessionManager::new(config.clone());
        sessions.start_idle_sweeper();
        Ok(Self {
            config,
            sessions,
            owner,
            observation_capture_bytes,
            activation: Mutex::new(activation),
        })
    }

    /// Rechecks the transport contract before a provider body is polled. Plugin activation is
    /// normally static, but this also fails closed if a component installs middleware later.
    pub(super) fn ensure_streaming_transport_compatible(&self) -> Result<(), CliError> {
        reject_incompatible_execution_middleware()
    }

    pub(super) async fn close(&self) -> Result<(), CliError> {
        let sessions = self.sessions.close_all("daemon_worker_shutdown").await;
        let subscribers = nemo_relay::api::runtime::flush_subscribers().map_err(CliError::from);
        let activation = lock(&self.activation)
            .take()
            .map(ServerPluginActivation::clear);
        sessions?;
        subscribers?;
        activation.transpose()?;
        Ok(())
    }

    pub(super) async fn handle_hook(
        &self,
        route: HookRoute,
        request: Request<Body>,
    ) -> Response<Body> {
        match self.handle_hook_inner(route, request).await {
            Ok(response) => Json(response).into_response(),
            Err(error) => error.into_response(),
        }
    }

    async fn handle_hook_inner(
        &self,
        route: HookRoute,
        request: Request<Body>,
    ) -> Result<Value, CliError> {
        let (mut parts, body) = request.into_parts();
        strip_worker_headers(&mut parts.headers);
        let bytes = axum::body::to_bytes(body, self.config.max_hook_payload_bytes)
            .await
            .map_err(body_read_error)?;
        let payload = serde_json::from_slice::<Value>(&bytes)
            .map_err(|error| CliError::InvalidPayload(error.to_string()))?;
        match route {
            HookRoute::Codex => {
                let outcome = codex::adapt(payload, &parts.headers);
                self.sessions
                    .apply_authenticated_events(&parts.headers, outcome.events, &self.owner)
                    .await?;
                if let Some(permission) = outcome.permission
                    && let Err(error) = self.authorize_permission(permission).await
                {
                    return Ok(json!({
                        "decision": "deny",
                        "reason": permission_denial_reason(error),
                    }));
                }
                Ok(outcome.response)
            }
            HookRoute::Claude => {
                let outcome = claude_code::adapt(payload, &parts.headers);
                self.sessions
                    .apply_authenticated_events(&parts.headers, outcome.events, &self.owner)
                    .await?;
                if let Some(permission) = outcome.permission {
                    let result = self.authorize_permission(permission).await;
                    return Ok(match result {
                        Ok(()) => json!({
                            "continue": true,
                            "hookSpecificOutput": {
                                "hookEventName": "PermissionRequest",
                                "decision": { "behavior": "allow" },
                            },
                        }),
                        Err(error) => json!({
                            "continue": true,
                            "hookSpecificOutput": {
                                "hookEventName": "PermissionRequest",
                                "decision": {
                                    "behavior": "deny",
                                    "message": permission_denial_reason(error),
                                },
                            },
                        }),
                    });
                }
                Ok(outcome.response)
            }
            HookRoute::Pi => {
                let outcome = pi::adapt(payload, &parts.headers);
                // A daemon worker is already isolated to one authenticated machine owner. Keep
                // pi's response-transform behavior while using that isolation as its ownership
                // boundary, just as the personal gateway does for its local extension.
                let effects = self
                    .sessions
                    .apply_events(&parts.headers, outcome.events)
                    .await?;
                Ok(pi::response_with_effects(outcome.response, &effects))
            }
        }
    }

    async fn authorize_permission(
        &self,
        permission: Result<crate::events::ToolEvent, String>,
    ) -> Result<(), CliError> {
        match permission {
            Ok(permission) => {
                self.sessions
                    .authorize_tool_permission(&permission, &self.owner)
                    .await
            }
            Err(reason) => Err(CliError::InvalidPayload(reason)),
        }
    }

    pub(super) async fn proxy_provider(
        &self,
        upstream: PooledClient,
        mut request: Request<Body>,
        route: ProviderRoute,
    ) -> Result<Response<RelayBody>, CliError> {
        let Some(surface) = provider_surface(request.uri().path()) else {
            return dispatch_unmanaged(upstream, request, route, &self.config).await;
        };
        if !request_body_decode_required()? {
            strip_worker_headers(request.headers_mut());
            strip_untrusted_dispatch_headers(request.headers_mut());
            let streaming_hint = request_streaming_hint(request.headers());
            let start = crate::gateway::daemon_gateway_start(
                request.headers(),
                request.uri().path(),
                Value::Null,
                streaming_hint,
            )
            .ok_or_else(|| CliError::InvalidPayload("unsupported provider path".into()))?;
            let prep = self
                .sessions
                .prepare_gateway_call(request.headers(), start)
                .await?;
            return self
                .proxy_unbuffered(upstream, request, route, surface, prep, streaming_hint)
                .await;
        }
        let prepared = PreparedProviderRequest::read(request, &self.config).await?;
        let start = crate::gateway::daemon_gateway_start(
            &prepared.headers,
            &prepared.path,
            prepared.request_json.clone(),
            prepared.streaming,
        )
        .ok_or_else(|| CliError::InvalidPayload("unsupported provider path".into()))?;
        let prep = self
            .sessions
            .prepare_gateway_call(&prepared.headers, start)
            .await?;
        if prep.bypass_managed_pipeline {
            self.sessions
                .finish_gateway_call(&prep.session_id, prep.session_finish)
                .await;
            return dispatch_observed(
                upstream,
                prepared,
                route,
                None,
                &self.config,
                self.observation_capture_bytes,
            )
            .await
            .map(|(response, _)| response);
        }
        self.proxy_managed(upstream, prepared, route, surface, prep)
            .await
    }

    async fn proxy_unbuffered(
        &self,
        upstream: PooledClient,
        request: Request<Body>,
        route: ProviderRoute,
        surface: ProviderSurface,
        prep: GatewayCallPrep,
        streaming_hint: bool,
    ) -> Result<Response<RelayBody>, CliError> {
        let GatewayCallPrep {
            scope_stack,
            session_id,
            provider_name,
            request: request_for_event,
            parent,
            attributes,
            metadata,
            model_name,
            owner_subagent_id,
            bypass_managed_pipeline,
            session_finish,
        } = prep;
        if bypass_managed_pipeline {
            self.sessions
                .finish_gateway_call(&session_id, session_finish)
                .await;
            return dispatch_unmanaged(upstream, request, route, &self.config).await;
        }

        let handle = TASK_SCOPE_STACK
            .scope(scope_stack, async {
                llm_call(
                    LlmCallParams::builder()
                        .name(&provider_name)
                        .request(&request_for_event)
                        .parent_opt(parent.as_ref())
                        .attributes(attributes)
                        .metadata(metadata.clone())
                        .model_name_opt(model_name)
                        .build(),
                )
            })
            .await;
        let handle = match handle {
            Ok(handle) => handle,
            Err(error) => {
                self.sessions
                    .finish_gateway_call(&session_id, session_finish)
                    .await;
                return Err(error.into());
            }
        };
        let response = dispatch_unbuffered_observed(
            upstream,
            request,
            route,
            &self.config,
            self.observation_capture_bytes,
        )
        .await;
        let (response, observation, response_streaming) = match response {
            Ok(result) => result,
            Err(error) => {
                finish_llm_after_dispatch_failure(&handle, metadata, &error);
                self.sessions
                    .finish_gateway_call(&session_id, session_finish)
                    .await;
                return Err(error);
            }
        };
        let sessions = self.sessions.clone();
        tokio::spawn(async move {
            let observed = observation
                .finish(surface, response_streaming || streaming_hint)
                .await;
            let response_value = observed.value.clone().unwrap_or(Value::Null);
            let mut end_metadata = merge_object(metadata, observed.metadata());
            insert_metadata(
                &mut end_metadata,
                "daemon_worker_request_capture",
                json!("head_only"),
            );
            if observed.failure.is_none() {
                insert_metadata(&mut end_metadata, "otel.status_code", json!("OK"));
            } else if let Some(failure) = observed.failure.as_ref() {
                insert_metadata(&mut end_metadata, "otel.status_code", json!("ERROR"));
                insert_metadata(&mut end_metadata, "otel.status_description", json!(failure));
            }
            if let Err(error) = llm_call_end(
                LlmCallEndParams::builder()
                    .handle(&handle)
                    .response(response_value)
                    .metadata(end_metadata)
                    .response_codec_opt(observed.value.as_ref().map(|_| response_codec(surface)))
                    .build(),
            ) {
                log::warn!(
                    target: "nemo_relay.daemon.worker",
                    event = "worker_llm_observation_end_failed",
                    error_kind = "runtime";
                    "Daemon worker failed to close an observed LLM lifecycle: {error}"
                );
            }
            if let Some(value) = observed.value {
                sessions
                    .record_gateway_response_hints(&session_id, owner_subagent_id, value)
                    .await;
            }
            sessions
                .finish_gateway_call(&session_id, session_finish)
                .await;
        });
        Ok(response)
    }

    async fn proxy_managed(
        &self,
        upstream: PooledClient,
        prepared: PreparedProviderRequest,
        route: ProviderRoute,
        surface: ProviderSurface,
        prep: GatewayCallPrep,
    ) -> Result<Response<RelayBody>, CliError> {
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
        let codec = request_codec(surface);
        let middleware = TASK_SCOPE_STACK
            .scope(scope_stack.clone(), async {
                llm_conditional_execution(&request).await?;
                let mut outcome = llm_request_intercepts(&provider_name, request).await?;
                if let Some(annotated) = outcome.annotated_request.as_ref() {
                    outcome.request = codec.encode(annotated, &outcome.request)?;
                }
                Ok::<_, FlowError>(outcome)
            })
            .await;
        let outcome = match middleware {
            Ok(outcome) => outcome,
            Err(error) => {
                self.sessions
                    .finish_gateway_call(&session_id, session_finish)
                    .await;
                return Err(error.into());
            }
        };
        if stream_mode(&outcome.request) != prepared.streaming {
            self.sessions
                .finish_gateway_call(&session_id, session_finish)
                .await;
            return Err(CliError::Flow(FlowError::InvalidArgument(
                STREAM_MODE_MUTATION_ERROR.into(),
            )));
        }
        let annotated_request = outcome.annotated_request.clone().map(Arc::new);
        let request_for_event = outcome.request.clone();
        let handle = TASK_SCOPE_STACK
            .scope(scope_stack, async {
                llm_call(
                    LlmCallParams::builder()
                        .name(&provider_name)
                        .request(&request_for_event)
                        .parent_opt(parent.as_ref())
                        .attributes(attributes)
                        .metadata(metadata.clone())
                        .model_name_opt(model_name)
                        .annotated_request_opt(annotated_request)
                        .build(),
                )
            })
            .await?;
        for contribution in outcome.optimization_contributions {
            let _ = handle.optimization_recorder.record(contribution);
        }
        if !outcome.pending_marks.is_empty() {
            log::warn!(
                target: "nemo_relay.daemon.worker",
                event = "worker_request_marks_unsupported",
                pending_mark_count = outcome.pending_marks.len();
                "Daemon raw-stream observation cannot attach request-interceptor marks to the LLM handle"
            );
        }

        let response = dispatch_observed(
            upstream,
            prepared,
            route,
            Some(&outcome.request),
            &self.config,
            self.observation_capture_bytes,
        )
        .await;
        let (response, observation) = match response {
            Ok(result) => result,
            Err(error) => {
                finish_llm_after_dispatch_failure(&handle, metadata, &error);
                self.sessions
                    .finish_gateway_call(&session_id, session_finish)
                    .await;
                return Err(error);
            }
        };
        let sessions = self.sessions.clone();
        tokio::spawn(async move {
            let observed = observation
                .finish(surface, prepared_streaming(&request_for_event))
                .await;
            let response_value = observed.value.clone().unwrap_or(Value::Null);
            let mut end_metadata = merge_object(metadata, observed.metadata());
            if observed.failure.is_none() {
                insert_metadata(&mut end_metadata, "otel.status_code", json!("OK"));
            } else if let Some(failure) = observed.failure.as_ref() {
                insert_metadata(&mut end_metadata, "otel.status_code", json!("ERROR"));
                insert_metadata(&mut end_metadata, "otel.status_description", json!(failure));
            }
            if let Err(error) = llm_call_end(
                LlmCallEndParams::builder()
                    .handle(&handle)
                    .response(response_value)
                    .metadata(end_metadata)
                    .response_codec_opt(observed.value.as_ref().map(|_| response_codec(surface)))
                    .build(),
            ) {
                log::warn!(
                    target: "nemo_relay.daemon.worker",
                    event = "worker_llm_observation_end_failed",
                    error_kind = "runtime";
                    "Daemon worker failed to close an observed LLM lifecycle: {error}"
                );
            }
            if let Some(value) = observed.value {
                sessions
                    .record_gateway_response_hints(&session_id, owner_subagent_id, value)
                    .await;
            }
            sessions
                .finish_gateway_call(&session_id, session_finish)
                .await;
        });
        Ok(response)
    }
}

pub(super) fn requires_route_pass_through(error: &CliError) -> bool {
    matches!(
        error,
        CliError::Flow(FlowError::InvalidArgument(message))
            if message == STREAM_MODE_MUTATION_ERROR
    )
}

async fn dispatch_unmanaged(
    upstream: PooledClient,
    mut request: Request<Body>,
    route: ProviderRoute,
    config: &GatewayConfig,
) -> Result<Response<RelayBody>, CliError> {
    strip_worker_headers(request.headers_mut());
    strip_untrusted_dispatch_headers(request.headers_mut());
    let path_and_query = request
        .uri()
        .path_and_query()
        .map_or_else(|| request.uri().path().to_owned(), ToString::to_string);
    let destination =
        crate::gateway::daemon_provider_upstream_url(request.headers(), &path_and_query, config)?
            .unwrap_or_else(|| route.upstream_url(config, &path_and_query))
            .parse::<Uri>()
            .map_err(|_| CliError::InvalidPayload("invalid provider destination".into()))?;
    strip_internal_headers(request.headers_mut());
    inject_provider_auth(request.headers_mut(), route, config);
    let strip = [
        HeaderName::from_static(CLIENT_TOKEN_HEADER),
        HeaderName::from_static(WORKER_TOKEN_HEADER),
        HeaderName::from_static(WORKER_ROUTE_FAILURE_HEADER),
    ];
    let request = prepare_forward_request(request, destination, &strip)
        .map_err(|error| CliError::InvalidPayload(error.to_string()))?
        .map(box_body);
    let response = tokio::time::timeout(RESPONSE_HEAD_TIMEOUT, upstream.request(request))
        .await
        .map_err(|_| CliError::Launch("provider response-head timeout".into()))?
        .map_err(|error| CliError::Launch(error.to_string()))?;
    let response = prepare_forward_response(response, &strip)
        .map_err(|error| CliError::Launch(error.to_string()))?;
    let (parts, body) = response.into_parts();
    Ok(Response::from_parts(parts, box_body(body)))
}

async fn dispatch_unbuffered_observed(
    upstream: PooledClient,
    mut request: Request<Body>,
    route: ProviderRoute,
    config: &GatewayConfig,
    capture_limit: usize,
) -> Result<(Response<RelayBody>, ObservationReceiver, bool), CliError> {
    strip_worker_headers(request.headers_mut());
    strip_untrusted_dispatch_headers(request.headers_mut());
    let path_and_query = request
        .uri()
        .path_and_query()
        .map_or_else(|| request.uri().path().to_owned(), ToString::to_string);
    let destination =
        crate::gateway::daemon_provider_upstream_url(request.headers(), &path_and_query, config)?
            .unwrap_or_else(|| route.upstream_url(config, &path_and_query))
            .parse::<Uri>()
            .map_err(|_| CliError::InvalidPayload("invalid provider destination".into()))?;
    if let Some(aligned) = crate::gateway::daemon_provider_forward_headers(
        request.headers(),
        request.uri().path(),
        config,
    ) {
        *request.headers_mut() = aligned;
    }
    strip_internal_headers(request.headers_mut());
    inject_provider_auth(request.headers_mut(), route, config);
    let strip = [
        HeaderName::from_static(CLIENT_TOKEN_HEADER),
        HeaderName::from_static(WORKER_TOKEN_HEADER),
        HeaderName::from_static(WORKER_ROUTE_FAILURE_HEADER),
    ];
    let request = prepare_forward_request(request, destination, &strip)
        .map_err(|error| CliError::InvalidPayload(error.to_string()))?
        .map(box_body);
    let response = tokio::time::timeout(RESPONSE_HEAD_TIMEOUT, upstream.request(request))
        .await
        .map_err(|_| CliError::Launch("provider response-head timeout".into()))?
        .map_err(|error| CliError::Launch(error.to_string()))?;
    let response = prepare_forward_response(response, &strip)
        .map_err(|error| CliError::Launch(error.to_string()))?;
    let status = response.status();
    let streaming = response_streaming(response.headers());
    let (parts, body) = response.into_parts();
    let (body, observation) = observe_body(body, status, capture_limit);
    Ok((Response::from_parts(parts, body), observation, streaming))
}

impl Drop for ManagedRuntime {
    fn drop(&mut self) {
        if let Some(activation) = self
            .activation
            .get_mut()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            let _ = activation.clear();
        }
    }
}

struct PreparedProviderRequest {
    method: Method,
    version: http::Version,
    headers: HeaderMap,
    path: String,
    path_and_query: String,
    body: Bytes,
    request_json: Value,
    streaming: bool,
}

impl PreparedProviderRequest {
    async fn read(request: Request<Body>, config: &GatewayConfig) -> Result<Self, CliError> {
        let (mut parts, body) = request.into_parts();
        strip_worker_headers(&mut parts.headers);
        // Dispatch controls are created only by worker-local middleware. Never allow a value that
        // arrived on the authenticated daemon hop to become an upstream override. Correlation
        // headers remain available to session normalization.
        strip_untrusted_dispatch_headers(&mut parts.headers);
        let bytes = axum::body::to_bytes(body, config.max_passthrough_body_bytes)
            .await
            .map_err(body_read_error)?;
        let request_json = serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null);
        let streaming = request_json
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let path = parts.uri.path().to_owned();
        let path_and_query = parts
            .uri
            .path_and_query()
            .map_or_else(|| path.clone(), ToString::to_string);
        Ok(Self {
            method: parts.method,
            version: parts.version,
            headers: parts.headers,
            path,
            path_and_query,
            body: bytes,
            request_json,
            streaming,
        })
    }
}

async fn dispatch_observed(
    upstream: PooledClient,
    prepared: PreparedProviderRequest,
    route: ProviderRoute,
    effective: Option<&LlmRequest>,
    config: &GatewayConfig,
    capture_limit: usize,
) -> Result<(Response<RelayBody>, ObservationReceiver), CliError> {
    let destination = effective_destination(&prepared, route, effective, config)?;
    let (mut headers, body, explicit_target) = effective_request(&prepared, effective)?;
    if !explicit_target
        && let Some(aligned) =
            crate::gateway::daemon_provider_forward_headers(&headers, &prepared.path, config)
    {
        headers = aligned;
    }
    let mut request = Request::builder()
        .method(prepared.method.clone())
        .version(prepared.version)
        .uri(destination.clone())
        .body(Body::from(body))?;
    *request.headers_mut() = headers;
    if !explicit_target {
        inject_provider_auth(request.headers_mut(), route, config);
    }
    let strip = [
        HeaderName::from_static(CLIENT_TOKEN_HEADER),
        HeaderName::from_static(WORKER_TOKEN_HEADER),
        HeaderName::from_static(WORKER_ROUTE_FAILURE_HEADER),
    ];
    let request = prepare_forward_request(request, destination, &strip)
        .map_err(|error| CliError::InvalidPayload(error.to_string()))?
        .map(box_body);
    let response = tokio::time::timeout(RESPONSE_HEAD_TIMEOUT, upstream.request(request))
        .await
        .map_err(|_| CliError::Launch("provider response-head timeout".into()))?
        .map_err(|error| CliError::Launch(error.to_string()))?;
    let response = prepare_forward_response(response, &strip)
        .map_err(|error| CliError::Launch(error.to_string()))?;
    let status = response.status();
    let (parts, body) = response.into_parts();
    let (body, observation) = observe_body(body, status, capture_limit);
    Ok((Response::from_parts(parts, body), observation))
}

fn effective_destination(
    prepared: &PreparedProviderRequest,
    route: ProviderRoute,
    effective: Option<&LlmRequest>,
    config: &GatewayConfig,
) -> Result<Uri, CliError> {
    let override_url = effective.and_then(|request| {
        request
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(INTERNAL_DISPATCH_URL_HEADER))
            .and_then(|(_, value)| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
    });
    let destination = match override_url {
        Some(destination) => destination.to_owned(),
        None => crate::gateway::daemon_provider_upstream_url(
            &prepared.headers,
            &prepared.path_and_query,
            config,
        )?
        .unwrap_or_else(|| route.upstream_url(config, &prepared.path_and_query)),
    };
    destination
        .parse::<Uri>()
        .map_err(|_| CliError::InvalidPayload("invalid provider destination".into()))
}

fn effective_request(
    prepared: &PreparedProviderRequest,
    effective: Option<&LlmRequest>,
) -> Result<(HeaderMap, Bytes, bool), CliError> {
    let mut headers = prepared.headers.clone();
    strip_internal_headers(&mut headers);
    let Some(effective) = effective else {
        headers.remove(CONTENT_LENGTH);
        return Ok((headers, prepared.body.clone(), false));
    };
    let explicit_target = has_explicit_target(effective);
    if explicit_target {
        crate::provider_auth::remove_provider_credentials(&mut headers);
        headers.remove(http::header::COOKIE);
    }
    let baseline = crate::gateway::daemon_observable_headers(&prepared.headers);
    for name in baseline.keys() {
        if !effective
            .headers
            .keys()
            .any(|effective_name| effective_name.eq_ignore_ascii_case(name))
            && let Ok(name) = HeaderName::from_bytes(name.as_bytes())
        {
            headers.remove(name);
        }
    }
    for (name, value) in &effective.headers {
        if name
            .to_ascii_lowercase()
            .starts_with(INTERNAL_HEADER_PREFIX)
        {
            continue;
        }
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        if baseline.get(name.as_str()) == Some(value) {
            continue;
        }
        let Some(value) = json_header_value(value) else {
            continue;
        };
        headers.insert(name, value);
    }
    strip_internal_headers(&mut headers);
    headers.remove(CONTENT_LENGTH);
    if effective.content == prepared.request_json || effective.content.is_null() {
        return Ok((headers, prepared.body.clone(), explicit_target));
    }
    let body = serde_json::to_vec(&effective.content)
        .map(Bytes::from)
        .map_err(|error| CliError::InvalidPayload(error.to_string()))?;
    headers.remove(CONTENT_ENCODING);
    Ok((headers, body, explicit_target))
}

fn has_explicit_target(request: &LlmRequest) -> bool {
    request.headers.iter().any(|(name, value)| {
        (name.eq_ignore_ascii_case(INTERNAL_DISPATCH_URL_HEADER)
            || name.eq_ignore_ascii_case(INTERNAL_DISPATCH_ROUTE_HEADER))
            && value
                .as_str()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
    })
}

fn strip_worker_headers(headers: &mut HeaderMap) {
    headers.remove(CLIENT_TOKEN_HEADER);
    headers.remove(WORKER_TOKEN_HEADER);
}

fn strip_internal_headers(headers: &mut HeaderMap) {
    let names = headers
        .keys()
        .filter(|name| name.as_str().starts_with(INTERNAL_HEADER_PREFIX))
        .cloned()
        .collect::<Vec<_>>();
    for name in names {
        headers.remove(name);
    }
}

fn strip_untrusted_dispatch_headers(headers: &mut HeaderMap) {
    headers.remove(INTERNAL_DISPATCH_URL_HEADER);
    headers.remove(INTERNAL_DISPATCH_ROUTE_HEADER);
    headers.remove(INTERNAL_DISPATCH_BACKEND_HEADER);
    headers.remove(INTERNAL_RETRY_AWARE_HEADER);
}

fn body_read_error(error: axum::Error) -> CliError {
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

fn inject_provider_auth(headers: &mut HeaderMap, route: ProviderRoute, config: &GatewayConfig) {
    if crate::provider_auth::has_provider_credential(headers) {
        return;
    }
    if let Some(configured) = match route {
        ProviderRoute::OpenAi => config.openai_auth_header.as_deref(),
        ProviderRoute::Anthropic => config.anthropic_auth_header.as_deref(),
    }
    .and_then(|value| HeaderValue::from_str(value).ok())
    {
        headers.insert(AUTHORIZATION, configured);
        return;
    }
    let (name, value) = match route {
        ProviderRoute::OpenAi => {
            let Some(key) = environment_value("OPENAI_API_KEY") else {
                return;
            };
            (AUTHORIZATION, format!("Bearer {key}"))
        }
        ProviderRoute::Anthropic => {
            let Some(key) = environment_value("ANTHROPIC_API_KEY") else {
                return;
            };
            (HeaderName::from_static("x-api-key"), key)
        }
    };
    if let Ok(value) = HeaderValue::from_str(&value) {
        headers.insert(name, value);
    }
}

fn environment_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn observation_capture_limit_from_environment() -> Result<usize, CliError> {
    let Some(raw) = std::env::var_os(OBSERVATION_CAPTURE_BYTES_ENV) else {
        return Ok(DEFAULT_OBSERVATION_CAPTURE_BYTES);
    };
    let raw = raw.to_str().ok_or_else(|| {
        CliError::Config(format!(
            "{OBSERVATION_CAPTURE_BYTES_ENV} must be a positive integer"
        ))
    })?;
    let value = raw.trim().parse::<usize>().map_err(|_| {
        CliError::Config(format!(
            "{OBSERVATION_CAPTURE_BYTES_ENV} must be a positive integer"
        ))
    })?;
    if value == 0 {
        return Err(CliError::Config(format!(
            "{OBSERVATION_CAPTURE_BYTES_ENV} must be a positive integer"
        )));
    }
    Ok(value)
}

fn json_header_value(value: &Value) -> Option<HeaderValue> {
    let value = value
        .as_str()
        .map(ToOwned::to_owned)
        .or_else(|| serde_json::to_string(value).ok())?;
    HeaderValue::from_str(&value).ok()
}

fn provider_surface(path: &str) -> Option<ProviderSurface> {
    match path {
        "/responses" | "/v1/responses" | "/backend-api/codex/responses" => {
            Some(ProviderSurface::OpenAIResponses)
        }
        "/chat/completions" | "/v1/chat/completions" => Some(ProviderSurface::OpenAIChat),
        "/v1/messages" => Some(ProviderSurface::AnthropicMessages),
        _ => None,
    }
}

fn stream_mode(request: &LlmRequest) -> bool {
    request
        .content
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn request_streaming_hint(headers: &HeaderMap) -> bool {
    headers.get_all(ACCEPT).iter().any(|value| {
        value.to_str().ok().is_some_and(|value| {
            value.split(',').any(|media_type| {
                media_type
                    .split(';')
                    .next()
                    .is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"))
            })
        })
    })
}

fn response_streaming(headers: &HeaderMap) -> bool {
    headers.get_all(CONTENT_TYPE).iter().any(|value| {
        value.to_str().ok().is_some_and(|value| {
            value
                .split(';')
                .next()
                .is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"))
        })
    })
}

fn prepared_streaming(request: &LlmRequest) -> bool {
    stream_mode(request)
}

fn request_body_decode_required() -> Result<bool, CliError> {
    let kinds = BTreeSet::from([
        RuntimeRegistrationKind::LlmSanitizeRequestGuardrail,
        RuntimeRegistrationKind::LlmConditionalExecutionGuardrail,
        RuntimeRegistrationKind::LlmRequestIntercept,
    ]);
    let registrations = list_runtime_registrations(Some(&kinds)).map_err(CliError::from)?;
    Ok(registrations.iter().any(registration_reads_request_body))
}

fn registration_reads_request_body(registration: &RuntimeRegistrationIdentity) -> bool {
    matches!(
        registration.kind,
        RuntimeRegistrationKind::LlmSanitizeRequestGuardrail
            | RuntimeRegistrationKind::LlmConditionalExecutionGuardrail
            | RuntimeRegistrationKind::LlmRequestIntercept
    )
}

fn reject_incompatible_execution_middleware() -> Result<(), CliError> {
    // Execution intercepts own the provider callback and may replace, suppress, retry, or mutate
    // its result. The raw worker transport cannot safely invoke that contract while also returning
    // the provider's response head and frames unchanged. Request intercepts and conditional
    // execution guardrails remain supported above the transport boundary.
    let kinds = BTreeSet::from([
        RuntimeRegistrationKind::LlmExecutionIntercept,
        RuntimeRegistrationKind::LlmStreamExecutionIntercept,
    ]);
    let registrations = list_runtime_registrations(Some(&kinds)).map_err(CliError::from)?;
    let incompatible = incompatible_registration_names(&registrations);
    if incompatible.is_empty() {
        return Ok(());
    }
    Err(CliError::Config(format!(
        "daemon worker raw delivery is incompatible with LLM execution middleware: {}",
        incompatible.join(", ")
    )))
}

fn incompatible_registration_names(registrations: &[RuntimeRegistrationIdentity]) -> Vec<String> {
    registrations
        .iter()
        .filter(|registration| {
            matches!(
                registration.kind,
                RuntimeRegistrationKind::LlmExecutionIntercept
                    | RuntimeRegistrationKind::LlmStreamExecutionIntercept
            )
        })
        .map(|registration| registration.effective_name.clone())
        .collect()
}

fn finish_llm_after_dispatch_failure(
    handle: &nemo_relay::api::llm::LlmHandle,
    metadata: Value,
    error: &CliError,
) {
    let mut metadata = metadata;
    insert_metadata(&mut metadata, "otel.status_code", json!("ERROR"));
    insert_metadata(
        &mut metadata,
        "otel.status_description",
        json!(error.to_string()),
    );
    let _ = llm_call_end(
        LlmCallEndParams::builder()
            .handle(handle)
            .response(Value::Null)
            .metadata(metadata)
            .build(),
    );
}

fn permission_denial_reason(error: CliError) -> String {
    error
        .guardrail_rejection_reason()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| error.to_string())
}

fn merge_object(mut base: Value, extra: Value) -> Value {
    if !base.is_object() {
        base = json!({});
    }
    if let (Some(base), Some(extra)) = (base.as_object_mut(), extra.as_object()) {
        base.extend(extra.clone());
    }
    base
}

fn insert_metadata(metadata: &mut Value, name: &str, value: Value) {
    if !metadata.is_object() {
        *metadata = json!({});
    }
    if let Some(metadata) = metadata.as_object_mut() {
        metadata.insert(name.to_owned(), value);
    }
}

struct ObservationSignal {
    terminal: AtomicU8,
    truncated: AtomicBool,
    notify: Notify,
}

impl ObservationSignal {
    fn new() -> Self {
        Self {
            terminal: AtomicU8::new(OBSERVATION_ACTIVE),
            truncated: AtomicBool::new(false),
            notify: Notify::new(),
        }
    }

    fn truncate(&self) {
        self.truncated.store(true, Ordering::Release);
    }

    fn finish(&self, terminal: u8) {
        if self
            .terminal
            .compare_exchange(
                OBSERVATION_ACTIVE,
                terminal,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.notify.notify_waiters();
        }
    }

    async fn wait(&self) -> u8 {
        loop {
            let notified = self.notify.notified();
            let terminal = self.terminal.load(Ordering::Acquire);
            if terminal != OBSERVATION_ACTIVE {
                return terminal;
            }
            notified.await;
        }
    }
}

struct ObservedBody<B> {
    body: B,
    sender: Option<mpsc::Sender<Bytes>>,
    signal: Arc<ObservationSignal>,
    scheduled_bytes: usize,
    capture_limit: usize,
}

impl<B> HttpBody for ObservedBody<B>
where
    B: HttpBody<Data = Bytes> + Unpin,
    B::Error: Into<BoxError>,
{
    type Data = Bytes;
    type Error = BoxError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        match Pin::new(&mut self.body).poll_frame(context) {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref()
                    && let Some(next) = self.scheduled_bytes.checked_add(data.len())
                {
                    if next <= self.capture_limit {
                        let sent = self
                            .sender
                            .as_ref()
                            .is_some_and(|sender| sender.try_send(data.clone()).is_ok());
                        if sent {
                            self.scheduled_bytes = next;
                        } else if self.sender.take().is_some() {
                            self.signal.truncate();
                        }
                    } else if self.sender.take().is_some() {
                        self.signal.truncate();
                    }
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(Some(Err(error))) => {
                self.sender.take();
                self.signal.finish(OBSERVATION_BODY_ERROR);
                Poll::Ready(Some(Err(error.into())))
            }
            Poll::Ready(None) => {
                self.sender.take();
                self.signal.finish(OBSERVATION_COMPLETE);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.body.size_hint()
    }
}

impl<B> Drop for ObservedBody<B> {
    fn drop(&mut self) {
        self.sender.take();
        self.signal.finish(OBSERVATION_CANCELLED);
    }
}

struct ObservationReceiver {
    receiver: mpsc::Receiver<Bytes>,
    signal: Arc<ObservationSignal>,
    status: StatusCode,
}

struct ObservedResponse {
    value: Option<Value>,
    truncated: bool,
    terminal: u8,
    status: StatusCode,
    failure: Option<String>,
}

impl ObservedResponse {
    fn metadata(&self) -> Value {
        json!({
            "daemon_worker_observation": {
                "truncated": self.truncated,
                "http_status": self.status.as_u16(),
                "terminal": match self.terminal {
                    OBSERVATION_COMPLETE => "complete",
                    OBSERVATION_BODY_ERROR => "body_error",
                    OBSERVATION_CANCELLED => "cancelled",
                    _ => "unknown",
                },
            },
        })
    }
}

impl ObservationReceiver {
    async fn finish(mut self, surface: ProviderSurface, streaming: bool) -> ObservedResponse {
        let value = if streaming {
            self.finish_stream(surface).await
        } else {
            self.finish_buffered().await
        };
        let terminal = self.signal.wait().await;
        let truncated = self.signal.truncated.load(Ordering::Acquire);
        let mut failure = match terminal {
            OBSERVATION_COMPLETE => None,
            OBSERVATION_BODY_ERROR => Some("provider response body failed".to_owned()),
            OBSERVATION_CANCELLED => Some("downstream cancelled provider response".to_owned()),
            _ => Some("provider response observation ended unexpectedly".to_owned()),
        };
        if truncated {
            failure = Some("provider response observation was truncated".to_owned());
        } else if !self.status.is_success() {
            failure = Some(format!("provider returned HTTP {}", self.status.as_u16()));
        }
        ObservedResponse {
            value: if truncated { None } else { value },
            truncated,
            terminal,
            status: self.status,
            failure,
        }
    }

    async fn finish_buffered(&mut self) -> Option<Value> {
        let mut bytes = Vec::new();
        while let Some(chunk) = self.receiver.recv().await {
            bytes.extend_from_slice(&chunk);
        }
        if self.signal.truncated.load(Ordering::Acquire) {
            return None;
        }
        serde_json::from_slice(&bytes).ok()
    }

    async fn finish_stream(&mut self, surface: ProviderSurface) -> Option<Value> {
        let mut decoder = SseEventDecoder::new();
        let codec = streaming_codec(surface);
        let mut collector = codec.collector();
        let finalizer = codec.finalizer();
        let mut valid = true;
        while let Some(chunk) = self.receiver.recv().await {
            if !valid || self.signal.truncated.load(Ordering::Acquire) {
                continue;
            }
            for event in decoder.push_bytes_results(&chunk) {
                match event {
                    Ok(event) => {
                        if collector(event.data).is_ok() {
                            continue;
                        }
                        valid = false;
                        self.signal.truncate();
                        break;
                    }
                    Err(_) => {
                        valid = false;
                        self.signal.truncate();
                        break;
                    }
                }
            }
        }
        if valid
            && !self.signal.truncated.load(Ordering::Acquire)
            && let Ok(Some(event)) = decoder.finish()
            && collector(event.data).is_err()
        {
            self.signal.truncate();
            valid = false;
        }
        (valid && !self.signal.truncated.load(Ordering::Acquire)).then(finalizer)
    }
}

fn observe_body<B>(
    body: B,
    status: StatusCode,
    capture_limit: usize,
) -> (RelayBody, ObservationReceiver)
where
    B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
    B::Error: Into<BoxError>,
{
    let signal = Arc::new(ObservationSignal::new());
    let (sender, receiver) = mpsc::channel(OBSERVATION_QUEUE_FRAMES);
    let sender = if body.is_end_stream() {
        signal.finish(OBSERVATION_COMPLETE);
        None
    } else {
        Some(sender)
    };
    let body = box_body(ObservedBody {
        body,
        sender,
        signal: Arc::clone(&signal),
        scheduled_bytes: 0,
        capture_limit,
    });
    (
        body,
        ObservationReceiver {
            receiver,
            signal,
            status,
        },
    )
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/worker_managed_tests.rs"]
mod tests;
