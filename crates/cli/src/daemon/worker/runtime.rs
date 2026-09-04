// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authenticated worker listener and lossless provider data plane.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Request, Response, StatusCode, Uri};
use axum::middleware::{self, Next};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::serve::ListenerExt;
use axum::{Json, Router};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ConnectionBuilder;
use hyper_util::service::TowerToHyperService;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::{Notify, Semaphore};

use super::super::common::control::{
    CLIENT_TOKEN_HEADER, DRAIN_LIFETIME_MS, MAX_CONTROL_BODY_BYTES, RECOVERY_LIFETIME_MS,
    SessionRequest, WORKER_DRAIN_PATH, WORKER_PROBE_PATH, WORKER_ROUTE_FAILURE_HEADER,
    WORKER_TOKEN_HEADER, WorkerDrainRequest, now_unix_ms,
};
use super::super::common::identity::{MachineIdentity, TokenDigest};
use super::super::common::routes::{ProviderRoute, PublicRoute};
use super::super::common::transport::{
    PooledClient, RelayBody, box_body, hold_body, pooled_client, prepare_forward_request,
    prepare_forward_response,
};
use super::control::{self, Registration};
use crate::configuration::GatewayConfig;
use crate::error::CliError;
use crate::plugins::lifecycle::ActiveDynamicPluginComponent;

use super::managed::ManagedRuntime;

const RESPONSE_HEAD_TIMEOUT: Duration = Duration::from_secs(60);
const CONTROL_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(5);
const INITIAL_READY_TIMEOUT: Duration = Duration::from_secs(15);
const RECOVERY_RETRY_INTERVAL: Duration = Duration::from_secs(1);
const MAX_CONCURRENT_TLS_HANDSHAKES: usize = 256;
const TLS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

pub(super) struct RuntimeOptions {
    pub(super) daemon_origin: String,
    pub(super) identity: MachineIdentity,
    pub(super) worker_id: String,
    pub(super) endpoint: String,
    pub(super) worker_tls_root: Option<String>,
    pub(super) tls_config: Option<Arc<rustls::ServerConfig>>,
    pub(super) config: GatewayConfig,
    pub(super) dynamic_plugins: Vec<ActiveDynamicPluginComponent>,
    pub(super) registration: Registration,
}

struct AuthTokens {
    data: TokenDigest,
    pending_data: Option<TokenDigest>,
    readiness_data: Option<TokenDigest>,
    control: TokenDigest,
    last_control_sequence: u64,
    last_control_request_id: String,
}

struct WorkerState {
    worker_id: String,
    config: GatewayConfig,
    upstream: PooledClient,
    managed: Option<Arc<ManagedRuntime>>,
    auth: RwLock<AuthTokens>,
    accepting: AtomicBool,
    draining: AtomicBool,
    exiting: AtomicBool,
    in_flight: AtomicUsize,
    drain_deadline: RwLock<Option<tokio::time::Instant>>,
    lifecycle: Notify,
}

impl WorkerState {
    fn new(
        worker_id: String,
        config: GatewayConfig,
        managed: Option<Arc<ManagedRuntime>>,
        registration: &Registration,
    ) -> Result<Self, CliError> {
        let data_token = registration.data_token_digest();
        Ok(Self {
            worker_id,
            config,
            upstream: pooled_client().map_err(|error| CliError::Launch(error.to_string()))?,
            managed,
            auth: RwLock::new(AuthTokens {
                data: data_token,
                pending_data: None,
                readiness_data: Some(data_token),
                control: registration.session_token_digest(),
                last_control_sequence: 0,
                last_control_request_id: String::new(),
            }),
            accepting: AtomicBool::new(false),
            draining: AtomicBool::new(false),
            exiting: AtomicBool::new(false),
            in_flight: AtomicUsize::new(0),
            drain_deadline: RwLock::new(None),
            lifecycle: Notify::new(),
        })
    }

    fn admit(self: &Arc<Self>) -> Option<InFlight> {
        if !self.accepting.load(Ordering::Acquire) {
            return None;
        }
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        if !self.accepting.load(Ordering::Acquire) {
            self.release_in_flight();
            return None;
        }
        Some(InFlight {
            state: Arc::clone(self),
        })
    }

    fn release_in_flight(&self) {
        let previous = self.in_flight.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "worker in-flight count underflowed");
        if previous == 1 && self.draining.load(Ordering::Acquire) {
            self.lifecycle.notify_waiters();
        }
    }

    fn authenticate_data(&self, headers: &HeaderMap) -> bool {
        let mut values = headers.get_all(WORKER_TOKEN_HEADER).iter();
        let Some(value) = values.next() else {
            return false;
        };
        if values.next().is_some() {
            return false;
        }
        let actual = TokenDigest::from_token(value.as_bytes());
        let auth = read_lock(&self.auth);
        auth.data.matches(&actual)
            || auth
                .pending_data
                .as_ref()
                .is_some_and(|pending| pending.matches(&actual))
    }

    fn activate_pending_readiness(&self, headers: &HeaderMap) {
        let mut values = headers.get_all(WORKER_TOKEN_HEADER).iter();
        let Some(value) = values.next() else {
            return;
        };
        if values.next().is_some() {
            return;
        }
        let actual = TokenDigest::from_token(value.as_bytes());
        let activated = {
            let mut auth = write_lock(&self.auth);
            if auth
                .readiness_data
                .as_ref()
                .is_some_and(|expected| expected.matches(&actual))
            {
                auth.readiness_data = None;
                true
            } else {
                false
            }
        };
        if activated
            && !self.draining.load(Ordering::Acquire)
            && !self.exiting.load(Ordering::Acquire)
        {
            self.accepting.store(true, Ordering::Release);
        }
    }

    fn authenticate_control(&self, request: &SessionRequest<WorkerDrainRequest>) -> bool {
        if request.payload.worker_id != self.worker_id
            || request.session_id != self.worker_id
            || request.request_id.is_empty()
            || request.request_id.len() > 128
            || !request.validate_payload_hash()
        {
            return false;
        }
        let actual = TokenDigest::from_token(request.session_token.expose().as_bytes());
        let mut auth = write_lock(&self.auth);
        if !auth.control.matches(&actual) {
            return false;
        }
        if request.sequence == auth.last_control_sequence
            && request.request_id == auth.last_control_request_id
        {
            return true;
        }
        if request.sequence != auth.last_control_sequence.saturating_add(1) {
            return false;
        }
        auth.last_control_sequence = request.sequence;
        auth.last_control_request_id = request.request_id.clone();
        true
    }

    fn begin_drain(&self, requested_timeout_ms: u64) {
        let timeout = Duration::from_millis(requested_timeout_ms.min(DRAIN_LIFETIME_MS));
        *write_lock(&self.drain_deadline) = Some(tokio::time::Instant::now() + timeout);
        self.draining.store(true, Ordering::Release);
        self.accepting.store(false, Ordering::Release);
        self.lifecycle.notify_waiters();
    }

    fn control_lost(&self) {
        self.accepting.store(false, Ordering::Release);
    }

    fn control_restored(&self, registration: &Registration) {
        {
            let mut auth = write_lock(&self.auth);
            auth.data = registration.data_token_digest();
            auth.pending_data = None;
            auth.readiness_data = None;
            auth.control = registration.session_token_digest();
            auth.last_control_sequence = 0;
            auth.last_control_request_id.clear();
        }
        if !self.draining.load(Ordering::Acquire) && !self.exiting.load(Ordering::Acquire) {
            self.accepting.store(true, Ordering::Release);
        }
    }

    fn stage_recovery_data_token(&self, registration: &Registration) {
        let token = registration.data_token_digest();
        let mut auth = write_lock(&self.auth);
        auth.pending_data = Some(token);
        auth.readiness_data = Some(token);
    }

    fn discard_recovery_data_token(&self) {
        let mut auth = write_lock(&self.auth);
        auth.pending_data = None;
        auth.readiness_data = None;
        self.accepting.store(false, Ordering::Release);
    }

    fn request_exit(&self) {
        self.exiting.store(true, Ordering::Release);
        self.accepting.store(false, Ordering::Release);
        self.lifecycle.notify_waiters();
    }

    async fn wait_until_stopped(&self) {
        loop {
            if self.exiting.load(Ordering::Acquire) {
                return;
            }
            if self.draining.load(Ordering::Acquire) {
                if self.in_flight.load(Ordering::Acquire) == 0 {
                    return;
                }
                let deadline = read_lock(&self.drain_deadline)
                    .as_ref()
                    .copied()
                    .unwrap_or_else(tokio::time::Instant::now);
                if deadline <= tokio::time::Instant::now() {
                    return;
                }
                tokio::select! {
                    _ = self.lifecycle.notified() => {}
                    _ = tokio::time::sleep_until(deadline) => return,
                }
            } else {
                self.lifecycle.notified().await;
            }
        }
    }
}

struct InFlight {
    state: Arc<WorkerState>,
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.state.release_in_flight();
    }
}

pub(super) async fn serve(listener: TcpListener, options: RuntimeOptions) -> Result<(), CliError> {
    let RuntimeOptions {
        daemon_origin,
        identity,
        worker_id,
        endpoint,
        worker_tls_root,
        tls_config,
        config,
        dynamic_plugins,
        mut registration,
    } = options;
    let managed = Arc::new(
        ManagedRuntime::initialize(
            config.clone(),
            dynamic_plugins,
            identity.fingerprint().to_string(),
        )
        .await?,
    );
    let state = Arc::new(WorkerState::new(
        worker_id.clone(),
        config,
        Some(managed),
        &registration,
    )?);
    let app = router(Arc::clone(&state));
    let server = async move {
        match tls_config {
            Some(config) => serve_tls(listener, app, config).await,
            None => axum::serve(
                listener.tap_io(|stream| {
                    let _ = stream.set_nodelay(true);
                }),
                app,
            )
            .await
            .map_err(CliError::Io),
        }
    };
    tokio::pin!(server);
    let readiness = tokio::time::timeout(
        INITIAL_READY_TIMEOUT,
        registration.ready(&daemon_origin, &worker_id),
    );
    let readiness = tokio::select! {
        result = &mut server => {
            return result.and_then(|()| Err(CliError::Launch("worker listener stopped before readiness".into())));
        }
        result = readiness => result,
    };
    match readiness {
        Ok(Ok(())) => state.control_restored(&registration),
        Ok(Err(error)) => return Err(error),
        Err(_) => {
            return Err(CliError::Launch(
                "daemon worker readiness acknowledgement timed out".into(),
            ));
        }
    }
    log::info!(
        target: "nemo_relay.daemon.worker",
        event = "worker_ready",
        worker_id = worker_id.as_str(),
        endpoint = endpoint.as_str();
        "Daemon worker is ready"
    );
    let heartbeat = tokio::spawn(monitor_control(
        Arc::clone(&state),
        daemon_origin,
        identity,
        worker_id.clone(),
        endpoint,
        worker_tls_root,
        registration,
    ));
    let signal_state = Arc::clone(&state);
    let signal = tokio::spawn(async move {
        shutdown_signal().await;
        signal_state.request_exit();
    });
    let result = tokio::select! {
        result = &mut server => result,
        _ = state.wait_until_stopped() => Ok(()),
    };
    state.request_exit();
    heartbeat.abort();
    signal.abort();
    if let Some(managed) = state.managed.as_ref() {
        managed.close().await?;
    }
    result
}

async fn serve_tls(
    listener: TcpListener,
    app: Router,
    config: Arc<rustls::ServerConfig>,
) -> Result<(), CliError> {
    let acceptor = tokio_rustls::TlsAcceptor::from(config);
    let handshake_permits = Arc::new(Semaphore::new(MAX_CONCURRENT_TLS_HANDSHAKES));
    let mut connections = tokio::task::JoinSet::new();
    loop {
        let (stream, _) = listener.accept().await?;
        stream.set_nodelay(true)?;
        let Ok(handshake_permit) = Arc::clone(&handshake_permits).try_acquire_owned() else {
            continue;
        };
        let acceptor = acceptor.clone();
        let service = app.clone();
        connections.spawn(async move {
            let Ok(Ok(stream)) =
                tokio::time::timeout(TLS_HANDSHAKE_TIMEOUT, acceptor.accept(stream)).await
            else {
                return;
            };
            drop(handshake_permit);
            let builder = ConnectionBuilder::new(TokioExecutor::new());
            let _ = builder
                .serve_connection_with_upgrades(
                    TokioIo::new(stream),
                    TowerToHyperService::new(service),
                )
                .await;
        });
        while connections.try_join_next().is_some() {}
    }
}

fn router(state: Arc<WorkerState>) -> Router {
    let control = Router::new()
        .route(WORKER_DRAIN_PATH, post(drain))
        .route(WORKER_PROBE_PATH, get(readiness_probe))
        .layer(DefaultBodyLimit::max(MAX_CONTROL_BODY_BYTES));
    Router::new()
        .merge(control)
        .fallback(proxy)
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            authenticate_daemon_request,
        ))
        .with_state(state)
}

#[cfg(test)]
pub(crate) struct TestWorkerHandle {
    state: Arc<WorkerState>,
}

#[cfg(test)]
impl TestWorkerHandle {
    pub(crate) fn control_lost(&self) {
        self.state.control_lost();
    }

    pub(crate) fn begin_drain(&self, deadline_unix_ms: u64) {
        self.state
            .begin_drain(deadline_unix_ms.saturating_sub(now_unix_ms()));
    }

    pub(crate) fn in_flight(&self) -> usize {
        self.state.in_flight.load(Ordering::Acquire)
    }
}

/// Constructs the real authenticated worker router with an injected process-wide pool. This is a
/// narrow test seam for multi-hop network tests; request admission, authentication, routing, and
/// provider forwarding remain the production handlers above and below it.
#[cfg(test)]
pub(crate) fn test_router(
    config: GatewayConfig,
    upstream: PooledClient,
    data_token: &[u8],
) -> (Router, TestWorkerHandle) {
    let state = Arc::new(WorkerState {
        worker_id: "test-worker".into(),
        config,
        upstream,
        managed: None,
        auth: RwLock::new(AuthTokens {
            data: TokenDigest::from_token(data_token),
            pending_data: None,
            readiness_data: None,
            control: TokenDigest::from_token(b"unused-test-control-token"),
            last_control_sequence: 0,
            last_control_request_id: String::new(),
        }),
        accepting: AtomicBool::new(true),
        draining: AtomicBool::new(false),
        exiting: AtomicBool::new(false),
        in_flight: AtomicUsize::new(0),
        drain_deadline: RwLock::new(None),
        lifecycle: Notify::new(),
    });
    (router(Arc::clone(&state)), TestWorkerHandle { state })
}

async fn readiness_probe(
    State(state): State<Arc<WorkerState>>,
    headers: HeaderMap,
) -> Response<Body> {
    if state.draining.load(Ordering::Acquire) || state.exiting.load(Ordering::Acquire) {
        return message(StatusCode::SERVICE_UNAVAILABLE, "worker is stopping");
    }
    // Only the exact token staged for this registration may open admission. A later health probe
    // using an old, still-authenticated data token must not resurrect a worker after control loss.
    // Opening before the response is returned keeps broker publication from racing local state.
    state.activate_pending_readiness(&headers);
    StatusCode::NO_CONTENT.into_response()
}

async fn authenticate_daemon_request(
    State(state): State<Arc<WorkerState>>,
    request: Request<Body>,
    next: Next,
) -> Response<Body> {
    if !state.authenticate_data(request.headers()) {
        return message(StatusCode::UNAUTHORIZED, "invalid daemon worker credential");
    }
    next.run(request).await
}

async fn drain(
    State(state): State<Arc<WorkerState>>,
    Json(request): Json<SessionRequest<WorkerDrainRequest>>,
) -> Response<Body> {
    if !state.authenticate_control(&request) {
        return message(
            StatusCode::UNAUTHORIZED,
            "invalid daemon control credential",
        );
    }
    state.begin_drain(drain_timeout_ms(&request.payload));
    StatusCode::NO_CONTENT.into_response()
}

fn drain_timeout_ms(request: &WorkerDrainRequest) -> u64 {
    request
        .timeout_ms
        .unwrap_or_else(|| request.deadline_unix_ms.saturating_sub(now_unix_ms()))
        .min(DRAIN_LIFETIME_MS)
}

async fn proxy(State(state): State<Arc<WorkerState>>, request: Request<Body>) -> Response<Body> {
    let Some(route) = PublicRoute::from_path(request.uri().path()) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if matches!(route, PublicRoute::Provider(_))
        && let Some(managed) = state.managed.as_ref()
        && let Err(error) = managed.ensure_streaming_transport_compatible()
    {
        return route_failure_response(error);
    }
    let Some(in_flight) = state.admit() else {
        let mut response = message(
            StatusCode::SERVICE_UNAVAILABLE,
            "worker is not accepting requests",
        );
        if !state.draining.load(Ordering::Acquire) && !state.exiting.load(Ordering::Acquire) {
            response.headers_mut().insert(
                WORKER_ROUTE_FAILURE_HEADER,
                HeaderValue::from_static("pass-through"),
            );
        }
        return response;
    };
    match route {
        PublicRoute::Hook(hook) => {
            if let Some(managed) = state.managed.as_ref() {
                let response = managed.handle_hook(hook, request).await;
                drop(in_flight);
                return response;
            }
            drop(in_flight);
            let mut response = Response::new(Body::from(hook.pass_through_body()));
            response
                .headers_mut()
                .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            response
        }
        PublicRoute::Provider(provider) => {
            if let Some(managed) = state.managed.as_ref() {
                let response = managed
                    .proxy_provider(state.upstream.clone(), request, provider)
                    .await;
                return match response {
                    Ok(response) => {
                        let (parts, body) = response.into_parts();
                        let body: RelayBody = hold_body(body, in_flight);
                        Response::from_parts(parts, Body::new(body))
                    }
                    Err(error) if super::managed::requires_route_pass_through(&error) => {
                        route_failure_response(error)
                    }
                    Err(error) => error.into_response(),
                };
            }
            forward_to_provider(Arc::clone(&state), request, provider, in_flight).await
        }
    }
}

fn route_failure_response(error: CliError) -> Response<Body> {
    let mut response = error.into_response();
    response.headers_mut().insert(
        WORKER_ROUTE_FAILURE_HEADER,
        HeaderValue::from_static("pass-through"),
    );
    response
}

async fn forward_to_provider(
    state: Arc<WorkerState>,
    mut request: Request<Body>,
    route: ProviderRoute,
    in_flight: InFlight,
) -> Response<Body> {
    let path_and_query = request
        .uri()
        .path_and_query()
        .map_or("/", |value| value.as_str());
    let destination = match crate::gateway::daemon_provider_upstream_url(
        request.headers(),
        path_and_query,
        &state.config,
    ) {
        Ok(Some(destination)) => destination,
        Ok(None) => route.upstream_url(&state.config, path_and_query),
        Err(error) => return error.into_response(),
    };
    inject_provider_auth(request.headers_mut(), route, &state.config);
    let destination = match destination.parse::<Uri>() {
        Ok(destination) => destination,
        Err(_) => return message(StatusCode::BAD_GATEWAY, "invalid provider destination"),
    };
    let strip = [
        HeaderName::from_static(CLIENT_TOKEN_HEADER),
        HeaderName::from_static(WORKER_TOKEN_HEADER),
        HeaderName::from_static(WORKER_ROUTE_FAILURE_HEADER),
        HeaderName::from_static(crate::agents::pi::alignment::UPSTREAM_BASE_URL_HEADER),
    ];
    let request = match prepare_forward_request(request, destination, &strip) {
        Ok(request) => request.map(box_body),
        Err(error) => return message(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    let response =
        match tokio::time::timeout(RESPONSE_HEAD_TIMEOUT, state.upstream.request(request)).await {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => return message(StatusCode::BAD_GATEWAY, &error.to_string()),
            Err(_) => {
                return message(
                    StatusCode::GATEWAY_TIMEOUT,
                    "provider response-head timeout",
                );
            }
        };
    let response = match prepare_forward_response(response, &strip) {
        Ok(response) => response,
        Err(error) => return message(StatusCode::BAD_GATEWAY, &error.to_string()),
    };
    let (parts, body) = response.into_parts();
    let body: RelayBody = hold_body(body, in_flight);
    Response::from_parts(parts, Body::new(body))
}

fn inject_provider_auth(headers: &mut HeaderMap, route: ProviderRoute, config: &GatewayConfig) {
    if crate::provider_auth::has_provider_credential(headers) {
        return;
    }
    let configured = match route {
        ProviderRoute::OpenAi => config.openai_auth_header.as_deref(),
        ProviderRoute::Anthropic => config.anthropic_auth_header.as_deref(),
    };
    if let Some(configured) = configured.and_then(header_value) {
        headers.insert(AUTHORIZATION, configured);
        return;
    }
    match route {
        ProviderRoute::OpenAi => {
            let Some(key) = nonempty_environment("OPENAI_API_KEY") else {
                return;
            };
            if let Some(value) = header_value(&format!("Bearer {key}")) {
                headers.insert(AUTHORIZATION, value);
            }
        }
        ProviderRoute::Anthropic => {
            let Some(key) = nonempty_environment("ANTHROPIC_API_KEY") else {
                return;
            };
            if let Some(value) = header_value(&key) {
                headers.insert(HeaderName::from_static("x-api-key"), value);
            }
        }
    }
}

fn nonempty_environment(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn header_value(value: &str) -> Option<HeaderValue> {
    HeaderValue::from_str(value).ok()
}

async fn monitor_control(
    state: Arc<WorkerState>,
    daemon_origin: String,
    identity: MachineIdentity,
    worker_id: String,
    endpoint: String,
    worker_tls_root: Option<String>,
    mut registration: Registration,
) {
    loop {
        tokio::time::sleep(registration.heartbeat_interval()).await;
        if state.draining.load(Ordering::Acquire) || state.exiting.load(Ordering::Acquire) {
            return;
        }
        if heartbeat_attempt(&mut registration, &daemon_origin, &worker_id).await {
            continue;
        }
        state.control_lost();
        log::error!(
            target: "nemo_relay.daemon.worker",
            event = "worker_control_lost",
            worker_id = worker_id.as_str();
            "Worker lost its authenticated daemon control relationship"
        );
        let recovery_deadline =
            tokio::time::Instant::now() + Duration::from_millis(RECOVERY_LIFETIME_MS);
        loop {
            if state.draining.load(Ordering::Acquire) || state.exiting.load(Ordering::Acquire) {
                return;
            }
            if heartbeat_attempt(&mut registration, &daemon_origin, &worker_id).await {
                state.control_restored(&registration);
                log::info!(
                    target: "nemo_relay.daemon.worker",
                    event = "worker_control_restored",
                    worker_id = worker_id.as_str();
                    "Worker restored its daemon control relationship"
                );
                break;
            }
            let recovered = tokio::time::timeout(
                CONTROL_ATTEMPT_TIMEOUT,
                control::recover(
                    &daemon_origin,
                    &identity,
                    &worker_id,
                    &endpoint,
                    worker_tls_root.as_deref(),
                    registration.generation_grant().clone(),
                ),
            )
            .await;
            if let Ok(Ok(mut new_registration)) = recovered {
                state.stage_recovery_data_token(&new_registration);
                let ready = tokio::time::timeout(
                    CONTROL_ATTEMPT_TIMEOUT,
                    new_registration.ready(&daemon_origin, &worker_id),
                )
                .await;
                if matches!(ready, Ok(Ok(()))) {
                    registration = new_registration;
                    state.control_restored(&registration);
                    log::info!(
                        target: "nemo_relay.daemon.worker",
                        event = "worker_reregistered",
                        worker_id = worker_id.as_str();
                        "Worker re-registered with its daemon"
                    );
                    break;
                }
                state.discard_recovery_data_token();
            }
            if tokio::time::Instant::now() >= recovery_deadline {
                log::error!(
                    target: "nemo_relay.daemon.worker",
                    event = "worker_recovery_expired",
                    worker_id = worker_id.as_str();
                    "Worker could not restore daemon control before the recovery deadline"
                );
                state.request_exit();
                return;
            }
            tokio::time::sleep(RECOVERY_RETRY_INTERVAL).await;
        }
    }
}

async fn heartbeat_attempt(
    registration: &mut Registration,
    daemon_origin: &str,
    worker_id: &str,
) -> bool {
    matches!(
        tokio::time::timeout(
            CONTROL_ATTEMPT_TIMEOUT,
            registration.heartbeat(daemon_origin, worker_id),
        )
        .await,
        Ok(Ok(()))
    )
}

fn message(status: StatusCode, text: &str) -> Response<Body> {
    (status, Json(json!({ "error": { "message": text } }))).into_response()
}

fn read_lock<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|error| error.into_inner())
}

fn write_lock<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|error| error.into_inner())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("installing SIGTERM handler should succeed");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    }
    #[cfg(windows)]
    {
        let mut shutdown = tokio::signal::windows::ctrl_shutdown()
            .expect("installing shutdown handler should succeed");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = shutdown.recv() => {}
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/worker_runtime_tests.rs"]
mod tests;
