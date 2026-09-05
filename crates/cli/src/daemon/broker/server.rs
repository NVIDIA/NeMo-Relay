// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public daemon listener, authenticated broker control plane, and streaming data plane.

use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode, Uri};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::serve::ListenerExt;
use axum::{Json, Router};
use base64::Engine;
use bytes::Bytes;
use hyper::body::Body as HttpBody;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ConnectionBuilder;
use hyper_util::service::TowerToHyperService;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::Serialize;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;

use super::lifecycle::{McpSessionId, ResolvedTarget, WorkerRequest, WorkerTarget};
use super::registry::{
    ExpiredActivation, McpRegistration, RecoveryPermit, Registry, RegistryError, ReleaseAction,
    ResolveError, WorkerFailureAction,
};
use crate::configuration::GatewayConfig;
use crate::daemon::ServerOptions;
use crate::daemon::common::address::{daemon_url, validate_bind_ip};
use crate::daemon::common::control::{
    ACTIVATION_LIFETIME_MS, ActivationFailedPayload, CHALLENGE_LIFETIME_MS, CHALLENGE_PATH,
    CLIENT_TOKEN_HEADER, ChallengeRequest, ChallengeResponse, DRAIN_LIFETIME_MS, EmptyPayload,
    MAX_CONTROL_BODY_BYTES, MCP_ACTIVATION_FAILED_PATH, MCP_HEARTBEAT_INTERVAL_MS,
    MCP_HEARTBEAT_PATH, MCP_LEASE_MS, MCP_REGISTER_PATH, MCP_RELEASE_PATH, McpHeartbeatResponse,
    McpRegisterRequest, McpRegisterResponse, RECOVERY_LIFETIME_MS, SessionRequest,
    WORKER_DRAIN_PATH, WORKER_HEARTBEAT_INTERVAL_MS, WORKER_HEARTBEAT_PATH, WORKER_LEASE_MS,
    WORKER_PROBE_PATH, WORKER_READY_PATH, WORKER_RECOVER_PATH, WORKER_REGISTER_PATH,
    WORKER_ROUTE_FAILURE_HEADER, WORKER_TOKEN_HEADER, WorkerDrainRequest, WorkerGenerationGrant,
    WorkerHeartbeatPayload, WorkerNetworkHint, WorkerReadyPayload, WorkerRecoverRequest,
    WorkerRegisterRequest, WorkerRegisterResponse, now_unix_ms, random_secret,
};
use crate::daemon::common::identity::{
    ChallengeId, ChallengeRecord, Fingerprint, MachineIdentity, TokenDigest,
};
use crate::daemon::common::protocol::{
    BrokerDirective, Capabilities, ComponentRole, HandshakeProof, SensitiveString, WorkerLaunch,
};
use crate::daemon::common::routes::{ProviderRoute, PublicRoute};
use crate::daemon::common::state::{
    ActiveWorkerGenerations, ROUTE_TOKEN_ENV, RouteCredential, load_or_create_daemon_identity,
};
use crate::daemon::common::transport::{
    PooledClient, RelayBody, box_body, hold_body, pooled_client, prepare_forward_request,
    prepare_forward_response,
};
use crate::daemon::common::worker_tls::WorkerClientPool;
use crate::error::CliError;

const RESPONSE_HEAD_TIMEOUT: Duration = Duration::from_secs(60);
const MAINTENANCE_INTERVAL: Duration = Duration::from_secs(1);
const MAX_PENDING_CHALLENGES: usize = 512;
const MAX_PENDING_MCP_CHALLENGES: usize = 384;
const MAX_PENDING_WORKER_CHALLENGES: usize = 128;
const MAX_STAGED_WORKER_SESSIONS: usize = 4_096;
const MAX_MCP_CONTROL_SESSIONS: usize = 8_192;
const MAX_ALLOWED_ROUTE_TOKENS: usize = 65_536;
const MAX_CONCURRENT_TLS_HANDSHAKES: usize = 256;
const TLS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

struct PendingChallenge {
    request: ChallengeRequest,
    record: ChallengeRecord,
}

struct Activation {
    fingerprint: Fingerprint,
    secret_digest: TokenDigest,
    deadline_unix_ms: u64,
    consumed: bool,
    bind_ip: Ipv4Addr,
    port: u16,
    advertise_address: Option<String>,
}

struct McpControlSession {
    fingerprint: Fingerprint,
    token_digest: TokenDigest,
    secret: SensitiveString,
    secret_digest: TokenDigest,
    lease_expires_at_unix_ms: u64,
    last_sequence: u64,
    last_request_id: String,
    last_heartbeat: Option<CachedHeartbeat>,
    worker_network: WorkerNetworkHint,
    released: bool,
}

#[derive(Clone)]
struct CachedHeartbeat {
    sequence: u64,
    request_id: String,
    response: McpHeartbeatResponse,
}

struct WorkerControlSession {
    fingerprint: Fingerprint,
    worker_id: String,
    secret: SensitiveString,
    secret_digest: TokenDigest,
    last_sequence: u64,
    last_request_id: String,
    next_daemon_sequence: u64,
    lease_expires_at_unix_ms: u64,
    pending_target: Arc<WorkerTarget>,
    publication: WorkerPublication,
    published: bool,
    generation_grant: WorkerGenerationGrant,
}

#[derive(Clone)]
enum WorkerPublication {
    Activation { activation_id: String },
    Recovery { permit: RecoveryPermit },
}

struct DaemonState {
    registry: Registry,
    identity: MachineIdentity,
    descriptor: crate::daemon::common::protocol::ComponentDescriptor,
    instance_id: String,
    public_origin: String,
    config: GatewayConfig,
    upstream: PooledClient,
    worker_clients: WorkerClientPool,
    allowed_route_tokens: HashSet<TokenDigest>,
    challenges: Mutex<HashMap<ChallengeId, PendingChallenge>>,
    activations: Mutex<HashMap<String, Activation>>,
    mcp_sessions: Mutex<HashMap<String, McpControlSession>>,
    mcp_heartbeat_serialization: Mutex<()>,
    worker_sessions: Mutex<HashMap<String, WorkerControlSession>>,
    pending_directives: Mutex<HashMap<String, BrokerDirective>>,
    active_worker_generations: ActiveWorkerGenerations,
    worker_generation_publication: Mutex<()>,
}

pub(crate) async fn serve(options: ServerOptions) -> Result<(), CliError> {
    validate_bind_ip(options.bind, "daemon")?;
    let bind = SocketAddr::new(IpAddr::V4(options.bind), options.port);
    let listener = TcpListener::bind(bind).await.map_err(|error| {
        CliError::Launch(format!("failed to bind daemon listener {bind}: {error}"))
    })?;
    let local = listener.local_addr()?;
    let public_origin = daemon_origin(&options, local)?;
    let resolved = crate::configuration::resolve_server_config(&options.gateway)?;
    let allowed_route_tokens = load_allowed_route_tokens(options.client_token_file.as_deref())?;
    let state = Arc::new(DaemonState {
        registry: Registry::new(options.pass_through),
        identity: load_or_create_daemon_identity()?,
        descriptor: crate::daemon::common::control::descriptor(ComponentRole::Daemon),
        instance_id: uuid::Uuid::now_v7().to_string(),
        public_origin,
        config: resolved.gateway,
        upstream: pooled_client().map_err(|error| CliError::Launch(error.to_string()))?,
        worker_clients: WorkerClientPool::new()?,
        allowed_route_tokens,
        challenges: Mutex::new(HashMap::new()),
        activations: Mutex::new(HashMap::new()),
        mcp_sessions: Mutex::new(HashMap::new()),
        mcp_heartbeat_serialization: Mutex::new(()),
        worker_sessions: Mutex::new(HashMap::new()),
        pending_directives: Mutex::new(HashMap::new()),
        active_worker_generations: ActiveWorkerGenerations::load()?,
        worker_generation_publication: Mutex::new(()),
    });
    spawn_maintenance(Arc::clone(&state));
    let app = router(Arc::clone(&state));
    let address = local.to_string();
    log::info!(
        target: "nemo_relay.daemon",
        event = "daemon_listening",
        address = address.as_str(),
        public_origin = state.public_origin.as_str(),
        pass_through = options.pass_through;
        "NeMo Relay daemon is listening"
    );
    match (&options.tls_cert, &options.tls_key) {
        (Some(certificate), Some(key)) => {
            let tls = load_tls_config(certificate, key)?;
            serve_tls(listener, app, tls).await
        }
        (None, None) => axum::serve(
            listener.tap_io(|stream| {
                let _ = stream.set_nodelay(true);
            }),
            app,
        )
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(CliError::Io),
        _ => Err(CliError::Config(
            "--tls-cert and --tls-key must be supplied together".into(),
        )),
    }
}

fn router(state: Arc<DaemonState>) -> Router {
    let control = Router::new()
        .route(CHALLENGE_PATH, post(issue_challenge))
        .route(MCP_REGISTER_PATH, post(register_mcp))
        .route(MCP_HEARTBEAT_PATH, post(heartbeat_mcp))
        .route(MCP_RELEASE_PATH, post(release_mcp))
        .route(MCP_ACTIVATION_FAILED_PATH, post(activation_failed))
        .route(WORKER_REGISTER_PATH, post(register_worker))
        .route(WORKER_RECOVER_PATH, post(recover_worker))
        .route(WORKER_READY_PATH, post(ready_worker))
        .route(WORKER_HEARTBEAT_PATH, post(heartbeat_worker))
        .layer(DefaultBodyLimit::max(MAX_CONTROL_BODY_BYTES));
    Router::new()
        .merge(control)
        .fallback(public_proxy)
        .with_state(state)
}

async fn issue_challenge(
    State(state): State<Arc<DaemonState>>,
    Json(request): Json<ChallengeRequest>,
) -> Response<Body> {
    if let Err(error) = request.initiator.validate() {
        return control_error(StatusCode::UNAUTHORIZED, error);
    }
    if !has_required_transport_capabilities(&request.initiator) {
        return control_message(
            StatusCode::UPGRADE_REQUIRED,
            "component lacks required lossless streaming and trailer capabilities",
        );
    }
    if request.initiator.role == ComponentRole::Daemon
        || request.initiator_public_identity.fingerprint() != request.initiator_fingerprint
        || request.initiator_instance_id.is_empty()
        || request.initiator_instance_id.len() > 256
    {
        return control_message(StatusCode::UNAUTHORIZED, "invalid component identity");
    }
    let now = now_unix_ms();
    let record = match ChallengeRecord::generate(now, CHALLENGE_LIFETIME_MS) {
        Ok(record) => record,
        Err(error) => return control_error(StatusCode::INTERNAL_SERVER_ERROR, error),
    };
    let challenge = record.challenge();
    let signed_request = request.clone();
    let initiator_role = request.initiator.role;
    let pending = PendingChallenge { request, record };
    let mut challenges = lock(&state.challenges);
    if !reserve_challenge_slot(&mut challenges, now, initiator_role) {
        return control_message(
            StatusCode::TOO_MANY_REQUESTS,
            "too many pending authentication challenges",
        );
    }
    challenges.insert(challenge.id, pending);
    drop(challenges);
    let mut response = ChallengeResponse {
        daemon: state.descriptor.clone(),
        daemon_instance_id: state.instance_id.clone(),
        daemon_public_identity: state.identity.public_identity(),
        daemon_fingerprint: state.identity.fingerprint(),
        challenge,
        daemon_challenge_proof: state.identity.sign(b"pending-daemon-challenge"),
    };
    let canonical =
        match crate::daemon::common::control::daemon_challenge_bytes(&signed_request, &response) {
            Ok(canonical) => canonical,
            Err(error) => return control_error(StatusCode::INTERNAL_SERVER_ERROR, error),
        };
    response.daemon_challenge_proof = state.identity.sign(&canonical);
    Json(response).into_response()
}

fn reserve_challenge_slot(
    challenges: &mut HashMap<ChallengeId, PendingChallenge>,
    now_unix_ms: u64,
    role: ComponentRole,
) -> bool {
    challenges.retain(|_, pending| pending.record.challenge().expires_at_unix_ms > now_unix_ms);
    let role_limit = match role {
        ComponentRole::Mcp => MAX_PENDING_MCP_CHALLENGES,
        ComponentRole::Worker => MAX_PENDING_WORKER_CHALLENGES,
        ComponentRole::Daemon => return false,
    };
    challenges.len() < MAX_PENDING_CHALLENGES
        && challenges
            .values()
            .filter(|pending| pending.request.initiator.role == role)
            .count()
            < role_limit
}

fn reserve_worker_session_slot(
    sessions: &mut HashMap<String, WorkerControlSession>,
    now_unix_ms: u64,
    worker_id: &str,
    capacity: usize,
) -> bool {
    sessions
        .retain(|_, session| session.published || session.lease_expires_at_unix_ms > now_unix_ms);
    if sessions.contains_key(worker_id) {
        return false;
    }
    sessions
        .values()
        .filter(|session| !session.published)
        .count()
        < capacity
}

async fn register_mcp(
    State(state): State<Arc<DaemonState>>,
    headers: HeaderMap,
    Json(request): Json<McpRegisterRequest>,
) -> Response<Body> {
    let credential = match public_credential(&headers) {
        Ok(credential) => credential,
        Err(response) => return response,
    };
    if !state.allowed_route_tokens.contains(&credential.digest()) {
        return control_message(StatusCode::UNAUTHORIZED, "invalid route credential");
    }
    let transcript = &request.proof.transcript;
    if transcript.initiator.role != ComponentRole::Mcp
        || transcript.route_token_digest != Some(credential.digest())
    {
        return control_message(StatusCode::UNAUTHORIZED, "route credential proof mismatch");
    }
    let daemon_proof = match validate_registration(&state, &request.proof) {
        Ok(proof) => proof,
        Err(response) => return response,
    };
    if request
        .worker_network
        .verify(
            &transcript.daemon_target,
            &transcript.initiator_instance_id,
            &transcript.challenge_id,
            &transcript.initiator_fingerprint,
            &transcript.initiator_public_identity,
        )
        .is_err()
    {
        return control_message(
            StatusCode::UNAUTHORIZED,
            "invalid worker network hint proof",
        );
    }
    let session_id = match McpSessionId::new(transcript.initiator_instance_id.clone()) {
        Ok(session_id) => session_id,
        Err(error) => return control_error(StatusCode::BAD_REQUEST, error),
    };
    let now = now_unix_ms();
    expire_activation_routes(&state, now);
    let launch = match fresh_launch(request.worker_network.hint.clone()) {
        Ok(launch) => launch,
        Err(error) => return control_error(StatusCode::INTERNAL_SERVER_ERROR, error),
    };
    let fresh_session_token = match random_secret(32).and_then(|secret| {
        SensitiveString::new(secret).map_err(|error| CliError::Launch(error.to_string()))
    }) {
        Ok(token) => token,
        Err(error) => return control_error(StatusCode::INTERNAL_SERVER_ERROR, error),
    };
    let lease_expires_at_unix_ms = now.saturating_add(MCP_LEASE_MS);
    let mut sessions = lock(&state.mcp_sessions);
    sessions.retain(|_, session| session.lease_expires_at_unix_ms > now);
    if !sessions.contains_key(session_id.as_str()) && sessions.len() >= MAX_MCP_CONTROL_SESSIONS {
        return control_message(StatusCode::TOO_MANY_REQUESTS, "too many live MCP sessions");
    }
    let (session_token, reuse_session) = match select_mcp_session_token(
        &sessions,
        session_id.as_str(),
        transcript.initiator_fingerprint,
        credential.digest(),
        request.worker_network.hint.clone(),
        now,
        fresh_session_token,
    ) {
        Ok(selection) => selection,
        Err(response) => return response,
    };
    let directive = match state.registry.register_mcp(
        McpRegistration {
            fingerprint: transcript.initiator_fingerprint,
            token_digest: credential.digest(),
            session_id: session_id.clone(),
            lease_expires_at_unix_ms,
        },
        launch,
    ) {
        Ok(directive) => directive,
        Err(error) => return registry_error(error),
    };
    if reuse_session {
        let session = sessions
            .get_mut(session_id.as_str())
            .expect("a selected reusable MCP session must still exist while locked");
        session.lease_expires_at_unix_ms = lease_expires_at_unix_ms;
    } else {
        sessions.insert(
            session_id.as_str().to_owned(),
            McpControlSession {
                fingerprint: transcript.initiator_fingerprint,
                token_digest: credential.digest(),
                secret_digest: TokenDigest::from_token(session_token.expose().as_bytes()),
                secret: session_token.clone(),
                lease_expires_at_unix_ms,
                last_sequence: 0,
                last_request_id: String::new(),
                last_heartbeat: None,
                worker_network: request.worker_network.hint,
                released: false,
            },
        );
    }
    drop(sessions);
    if !reuse_session {
        lock(&state.pending_directives).remove(session_id.as_str());
    }
    remember_activation(&state, transcript.initiator_fingerprint, &directive);
    Json(McpRegisterResponse {
        daemon_proof,
        session_token,
        heartbeat_interval_ms: MCP_HEARTBEAT_INTERVAL_MS,
        directive,
    })
    .into_response()
}

#[allow(clippy::result_large_err)]
fn select_mcp_session_token(
    sessions: &HashMap<String, McpControlSession>,
    session_id: &str,
    fingerprint: Fingerprint,
    token_digest: TokenDigest,
    worker_network: WorkerNetworkHint,
    now_unix_ms: u64,
    fresh: SensitiveString,
) -> Result<(SensitiveString, bool), Response<Body>> {
    let reusable = sessions
        .get(session_id)
        .filter(|session| session.lease_expires_at_unix_ms > now_unix_ms && !session.released);
    if reusable.is_some_and(|session| {
        session.fingerprint != fingerprint
            || !session.token_digest.matches(&token_digest)
            || session.worker_network != worker_network
    }) {
        return Err(control_message(
            StatusCode::UNAUTHORIZED,
            "MCP session ID is already bound to another authenticated route",
        ));
    }
    Ok(reusable.map_or((fresh, false), |session| (session.secret.clone(), true)))
}

async fn heartbeat_mcp(
    State(state): State<Arc<DaemonState>>,
    Json(request): Json<SessionRequest<EmptyPayload>>,
) -> Response<Body> {
    // The critical section contains no I/O. Serializing it closes the small window in which a
    // concurrent lost-response retry could observe the accepted sequence before its response was
    // cached, while leaving the request data plane entirely lock-free.
    let _heartbeat_serialization = lock(&state.mcp_heartbeat_serialization);
    let lease_expires_at_unix_ms = now_unix_ms().saturating_add(MCP_LEASE_MS);
    let authenticated = match authenticate_mcp(&state, &request, Some(lease_expires_at_unix_ms)) {
        Ok(authenticated) => authenticated,
        Err(response) => return response,
    };
    if authenticated.released {
        return control_message(StatusCode::UNAUTHORIZED, "MCP session was already released");
    }
    if authenticated.duplicate {
        return authenticated.cached_heartbeat.map_or_else(
            || {
                control_message(
                    StatusCode::CONFLICT,
                    "duplicate request does not match the cached heartbeat response",
                )
            },
            |response| Json(response).into_response(),
        );
    }
    if let Err(error) = state.registry.renew_mcp(
        authenticated.fingerprint,
        &authenticated.session_id,
        lease_expires_at_unix_ms,
    ) {
        lock(&state.mcp_sessions).remove(authenticated.session_id.as_str());
        lock(&state.pending_directives).remove(authenticated.session_id.as_str());
        return registry_error(error);
    }
    let directive = lock(&state.pending_directives).remove(authenticated.session_id.as_str());
    let response = McpHeartbeatResponse { directive };
    if let Some(session) = lock(&state.mcp_sessions).get_mut(authenticated.session_id.as_str()) {
        session.last_heartbeat = Some(CachedHeartbeat {
            sequence: request.sequence,
            request_id: request.request_id,
            response: response.clone(),
        });
    }
    Json(response).into_response()
}

async fn release_mcp(
    State(state): State<Arc<DaemonState>>,
    Json(request): Json<SessionRequest<EmptyPayload>>,
) -> Response<Body> {
    let authenticated = match authenticate_mcp(&state, &request, None) {
        Ok(authenticated) => authenticated,
        Err(response) => return response,
    };
    if authenticated.released {
        return if authenticated.duplicate {
            StatusCode::NO_CONTENT.into_response()
        } else {
            control_message(StatusCode::UNAUTHORIZED, "MCP session was already released")
        };
    }
    if authenticated.duplicate {
        return StatusCode::NO_CONTENT.into_response();
    }
    if let Some(session) = lock(&state.mcp_sessions).get_mut(authenticated.session_id.as_str()) {
        session.released = true;
    }
    lock(&state.pending_directives).remove(authenticated.session_id.as_str());
    let action = match state.registry.release_mcp(
        authenticated.fingerprint,
        &authenticated.session_id,
        now_unix_ms().saturating_add(DRAIN_LIFETIME_MS),
    ) {
        Ok(action) => action,
        Err(RegistryError::UnknownRoute | RegistryError::UnknownMcpSession) => {
            return StatusCode::NO_CONTENT.into_response();
        }
        Err(error) => return registry_error(error),
    };
    handle_release_action(Arc::clone(&state), authenticated.fingerprint, action);
    StatusCode::NO_CONTENT.into_response()
}

async fn activation_failed(
    State(state): State<Arc<DaemonState>>,
    Json(request): Json<SessionRequest<ActivationFailedPayload>>,
) -> Response<Body> {
    if request.payload.activation_id.len() > 128 || request.payload.reason.len() > 2_048 {
        return control_message(
            StatusCode::BAD_REQUEST,
            "activation failure payload is too large",
        );
    }
    let authenticated = match authenticate_mcp(&state, &request, None) {
        Ok(authenticated) => authenticated,
        Err(response) => return response,
    };
    if authenticated.released {
        return control_message(StatusCode::UNAUTHORIZED, "MCP session was already released");
    }
    if authenticated.duplicate {
        return StatusCode::NO_CONTENT.into_response();
    }
    match state
        .registry
        .mark_activation_failed(authenticated.fingerprint, &request.payload.activation_id)
    {
        Ok(()) => {
            revoke_activation(&state, &request.payload.activation_id);
            let fingerprint = authenticated.fingerprint.to_string();
            log::error!(
                target: "nemo_relay.daemon",
                event = "worker_activation_failed",
                fingerprint = fingerprint.as_str(),
                reason = request.payload.reason.as_str();
                "Worker activation failed; route changed to pass-through"
            );
            StatusCode::NO_CONTENT.into_response()
        }
        Err(error) => registry_error(error),
    }
}

async fn register_worker(
    State(state): State<Arc<DaemonState>>,
    Json(request): Json<WorkerRegisterRequest>,
) -> Response<Body> {
    if request.proof.transcript.initiator.role != ComponentRole::Worker
        || request.proof.transcript.route_token_digest.is_some()
    {
        return control_message(StatusCode::UNAUTHORIZED, "invalid worker proof");
    }
    let daemon_proof = match validate_registration(&state, &request.proof) {
        Ok(proof) => proof,
        Err(response) => return response,
    };
    let now = now_unix_ms();
    let (activation_fingerprint, replay) = {
        let mut activations = lock(&state.activations);
        let Some(activation) = activations.get_mut(&request.activation_id) else {
            return control_message(
                StatusCode::UNAUTHORIZED,
                "unknown or consumed activation grant",
            );
        };
        if activation.deadline_unix_ms <= now
            || activation.fingerprint != request.proof.transcript.initiator_fingerprint
            || !activation.secret_digest.matches(&TokenDigest::from_token(
                request.activation_token.expose().as_bytes(),
            ))
        {
            return control_message(StatusCode::UNAUTHORIZED, "invalid activation grant");
        }
        if !activation_endpoint_matches(&request.endpoint, activation) {
            let fingerprint = activation.fingerprint;
            drop(activations);
            let _ = state
                .registry
                .mark_activation_failed(fingerprint, &request.activation_id);
            revoke_activation(&state, &request.activation_id);
            return control_message(
                StatusCode::BAD_REQUEST,
                "worker endpoint does not match the signed activation policy",
            );
        }
        if activation.consumed {
            (activation.fingerprint, true)
        } else {
            activation.consumed = true;
            (activation.fingerprint, false)
        }
    };
    if replay {
        return replay_worker_registration(
            &state,
            activation_fingerprint,
            &request.worker_id,
            &request.endpoint,
            request.tls_root_certificate.as_deref(),
            Some(&request.activation_id),
            None,
            daemon_proof,
        )
        .unwrap_or_else(|| {
            control_message(
                StatusCode::UNAUTHORIZED,
                "activation grant was consumed by another worker registration",
            )
        });
    }
    let publication = WorkerPublication::Activation {
        activation_id: request.activation_id.clone(),
    };
    let response = stage_worker(
        &state,
        activation_fingerprint,
        request.worker_id,
        request.endpoint,
        request.tls_root_certificate,
        None,
        publication.clone(),
        daemon_proof,
    );
    if !response.status().is_success() {
        fail_worker_publication(&state, activation_fingerprint, &publication);
    }
    response
}

async fn recover_worker(
    State(state): State<Arc<DaemonState>>,
    Json(request): Json<WorkerRecoverRequest>,
) -> Response<Body> {
    if request.proof.transcript.initiator.role != ComponentRole::Worker
        || request.proof.transcript.route_token_digest.is_some()
    {
        return control_message(StatusCode::UNAUTHORIZED, "invalid worker proof");
    }
    let daemon_proof = match validate_registration(&state, &request.proof) {
        Ok(proof) => proof,
        Err(response) => return response,
    };
    let fingerprint = request.proof.transcript.initiator_fingerprint;
    if request
        .generation_grant
        .verify(
            &request.worker_id,
            fingerprint,
            &request.endpoint,
            request.tls_root_certificate.as_deref(),
            &state.identity.public_identity(),
        )
        .is_err()
    {
        return control_message(
            StatusCode::UNAUTHORIZED,
            "invalid worker recovery generation",
        );
    }
    match state
        .active_worker_generations
        .matches(fingerprint, &request.generation_grant.generation_id)
    {
        Ok(true) => {}
        Ok(false) => {
            return control_message(
                StatusCode::UNAUTHORIZED,
                "invalid worker recovery generation",
            );
        }
        Err(error) => return control_error(StatusCode::INTERNAL_SERVER_ERROR, error),
    }
    if let Some(response) = replay_worker_registration(
        &state,
        fingerprint,
        &request.worker_id,
        &request.endpoint,
        request.tls_root_certificate.as_deref(),
        None,
        Some(&request.generation_grant.generation_id),
        daemon_proof.clone(),
    ) {
        return response;
    }
    let permit = match state
        .registry
        .authorize_worker_recovery(fingerprint, &request.worker_id)
    {
        Ok(permit) => permit,
        Err(error) => return registry_error(error),
    };
    let publication = WorkerPublication::Recovery { permit };
    let generation_id = request.generation_grant.generation_id.clone();
    let response = stage_worker(
        &state,
        fingerprint,
        request.worker_id,
        request.endpoint,
        request.tls_root_certificate,
        Some(request.generation_grant),
        publication.clone(),
        daemon_proof,
    );
    if !response.status().is_success()
        && revoke_active_worker_generation(&state, fingerprint, &generation_id)
    {
        fail_worker_publication(&state, fingerprint, &publication);
    }
    response
}

#[allow(clippy::too_many_arguments)]
fn replay_worker_registration(
    state: &DaemonState,
    fingerprint: Fingerprint,
    worker_id: &str,
    endpoint: &str,
    tls_root_certificate: Option<&str>,
    activation_id: Option<&str>,
    generation_id: Option<&str>,
    daemon_proof: HandshakeProof,
) -> Option<Response<Body>> {
    let sessions = lock(&state.worker_sessions);
    let session = sessions.get(worker_id)?;
    let publication_matches = match (&session.publication, activation_id, generation_id) {
        (
            WorkerPublication::Activation {
                activation_id: staged,
            },
            Some(expected),
            None,
        ) => staged == expected,
        (_, None, Some(expected)) => session.generation_grant.generation_id == expected,
        _ => false,
    };
    if session.fingerprint != fingerprint
        || session.pending_target.endpoint() != endpoint
        || !publication_matches
        || session
            .generation_grant
            .verify(
                worker_id,
                fingerprint,
                endpoint,
                tls_root_certificate,
                &state.identity.public_identity(),
            )
            .is_err()
    {
        return None;
    }
    let data_token =
        SensitiveString::new(session.pending_target.session_token().to_owned()).ok()?;
    Some(
        Json(WorkerRegisterResponse {
            daemon_proof,
            session_token: session.secret.clone(),
            data_token,
            heartbeat_interval_ms: WORKER_HEARTBEAT_INTERVAL_MS,
            generation_grant: session.generation_grant.clone(),
        })
        .into_response(),
    )
}

#[allow(clippy::too_many_arguments)]
fn stage_worker(
    state: &Arc<DaemonState>,
    fingerprint: Fingerprint,
    worker_id: String,
    endpoint: String,
    tls_root_certificate: Option<String>,
    generation_grant: Option<WorkerGenerationGrant>,
    publication: WorkerPublication,
    daemon_proof: HandshakeProof,
) -> Response<Body> {
    if worker_id.is_empty()
        || worker_id.len() > 256
        || endpoint.len() > 2_048
        || validate_worker_endpoint(&endpoint, tls_root_certificate.as_deref()).is_err()
    {
        return control_message(StatusCode::BAD_REQUEST, "invalid worker endpoint");
    }
    let generation_grant = match generation_grant {
        Some(grant) => grant,
        None => match WorkerGenerationGrant::issue(
            &worker_id,
            fingerprint,
            &endpoint,
            tls_root_certificate.as_deref(),
            &state.identity,
        ) {
            Ok(grant) => grant,
            Err(error) => return control_error(StatusCode::INTERNAL_SERVER_ERROR, error),
        },
    };
    let control_secret = match random_secret(32).and_then(|secret| {
        SensitiveString::new(secret).map_err(|error| CliError::Launch(error.to_string()))
    }) {
        Ok(secret) => secret,
        Err(error) => return control_error(StatusCode::INTERNAL_SERVER_ERROR, error),
    };
    let data_secret = match random_secret(32).and_then(|secret| {
        SensitiveString::new(secret).map_err(|error| CliError::Launch(error.to_string()))
    }) {
        Ok(secret) => secret,
        Err(error) => return control_error(StatusCode::INTERNAL_SERVER_ERROR, error),
    };
    let mut worker_sessions = lock(&state.worker_sessions);
    if !reserve_worker_session_slot(
        &mut worker_sessions,
        now_unix_ms(),
        &worker_id,
        MAX_STAGED_WORKER_SESSIONS,
    ) {
        return control_message(
            StatusCode::TOO_MANY_REQUESTS,
            "too many staged worker sessions",
        );
    }
    // Select or construct the process-wide pool only after this worker has won admission. Keeping
    // the session lock through the synchronous client construction makes the collision check and
    // insertion atomic, without holding it across connection acquisition or network I/O.
    let worker_client = match state.worker_clients.client(tls_root_certificate.as_deref()) {
        Ok(client) => client,
        Err(error) => return control_error(StatusCode::BAD_REQUEST, error),
    };
    let target = match WorkerTarget::with_shared_client(
        worker_id.clone(),
        endpoint,
        data_secret.clone(),
        worker_client,
    ) {
        Ok(target) => Arc::new(target),
        Err(error) => return control_error(StatusCode::BAD_REQUEST, error),
    };
    worker_sessions.insert(
        worker_id.clone(),
        WorkerControlSession {
            fingerprint,
            worker_id,
            secret_digest: TokenDigest::from_token(control_secret.expose().as_bytes()),
            secret: control_secret.clone(),
            last_sequence: 0,
            last_request_id: String::new(),
            next_daemon_sequence: 0,
            lease_expires_at_unix_ms: now_unix_ms().saturating_add(WORKER_LEASE_MS),
            pending_target: target,
            publication,
            published: false,
            generation_grant: generation_grant.clone(),
        },
    );
    drop(worker_sessions);
    Json(WorkerRegisterResponse {
        daemon_proof,
        session_token: control_secret,
        data_token: data_secret,
        heartbeat_interval_ms: WORKER_HEARTBEAT_INTERVAL_MS,
        generation_grant,
    })
    .into_response()
}

async fn ready_worker(
    State(state): State<Arc<DaemonState>>,
    Json(request): Json<SessionRequest<WorkerReadyPayload>>,
) -> Response<Body> {
    let candidate = {
        let mut sessions = lock(&state.worker_sessions);
        let Some(session) = sessions.get_mut(&request.session_id) else {
            return control_message(StatusCode::UNAUTHORIZED, "unknown worker session");
        };
        if request.payload.worker_id != session.worker_id {
            return control_message(StatusCode::UNAUTHORIZED, "worker identity mismatch");
        }
        if let Err(response) = authenticate_sequence(
            session.secret_digest,
            &mut session.last_sequence,
            &mut session.last_request_id,
            &request,
        ) {
            return response;
        }
        if session.published {
            return StatusCode::NO_CONTENT.into_response();
        }
        (
            session.fingerprint,
            Arc::clone(&session.pending_target),
            session.publication.clone(),
            session.generation_grant.generation_id.clone(),
        )
    };
    let (fingerprint, target, publication, generation_id) = candidate;
    if let Err(error) = probe_worker(&target).await {
        let fail_route = match &publication {
            WorkerPublication::Activation { .. } => true,
            WorkerPublication::Recovery { .. } => {
                revoke_active_worker_generation(&state, fingerprint, &generation_id)
            }
        };
        if fail_route {
            fail_worker_publication(&state, fingerprint, &publication);
        }
        lock(&state.worker_sessions).remove(target.worker_id());
        return control_error(StatusCode::BAD_GATEWAY, error);
    }
    // No network I/O occurs while this lock is held. It serializes the durable generation update
    // with the broker publication so restart recovery cannot race a replacement readiness probe.
    let _generation_publication = lock(&state.worker_generation_publication);
    let previous_generation = match &publication {
        WorkerPublication::Activation { activation_id } => {
            let current = lock(&state.activations)
                .get(activation_id)
                .is_some_and(|activation| {
                    activation.fingerprint == fingerprint && activation.consumed
                });
            if !current {
                lock(&state.worker_sessions).remove(target.worker_id());
                return control_message(
                    StatusCode::UNAUTHORIZED,
                    "worker activation is no longer current",
                );
            }
            match state
                .active_worker_generations
                .publish(fingerprint, &generation_id)
            {
                Ok(previous) => previous,
                Err(error) => {
                    fail_worker_publication(&state, fingerprint, &publication);
                    lock(&state.worker_sessions).remove(target.worker_id());
                    return control_error(StatusCode::INTERNAL_SERVER_ERROR, error);
                }
            }
        }
        WorkerPublication::Recovery { .. } => {
            match state
                .active_worker_generations
                .matches(fingerprint, &generation_id)
            {
                Ok(true) => {}
                Ok(false) => {
                    lock(&state.worker_sessions).remove(target.worker_id());
                    return control_message(
                        StatusCode::UNAUTHORIZED,
                        "worker recovery generation was revoked before readiness",
                    );
                }
                Err(error) => {
                    fail_worker_publication(&state, fingerprint, &publication);
                    lock(&state.worker_sessions).remove(target.worker_id());
                    return control_error(StatusCode::INTERNAL_SERVER_ERROR, error);
                }
            }
            None
        }
    };
    let publication_result = match &publication {
        WorkerPublication::Activation { activation_id } => state
            .registry
            .mark_worker_ready(fingerprint, activation_id, Arc::clone(&target))
            .map(|()| Some(activation_id.clone())),
        WorkerPublication::Recovery { permit } => {
            state
                .registry
                .publish_recovered_worker(fingerprint, permit, Arc::clone(&target))
        }
    };
    let canceled_activation = match publication_result {
        Ok(canceled_activation) => canceled_activation,
        Err(error) => {
            if matches!(&publication, WorkerPublication::Activation { .. })
                && let Err(restore_error) = state.active_worker_generations.restore_if_matches(
                    fingerprint,
                    &generation_id,
                    previous_generation.as_deref(),
                )
            {
                log::error!(
                    target: "nemo_relay.daemon",
                    event = "worker_generation_restore_failed",
                    error_kind = restore_error.log_kind();
                    "Failed to restore durable worker generation after publication race"
                );
            }
            lock(&state.worker_sessions).remove(target.worker_id());
            return registry_error(error);
        }
    };
    if let Some(activation_id) = canceled_activation {
        revoke_activation(&state, &activation_id);
    }
    if let Some(session) = lock(&state.worker_sessions).get_mut(target.worker_id()) {
        session.published = true;
        session.lease_expires_at_unix_ms = now_unix_ms().saturating_add(WORKER_LEASE_MS);
    }
    StatusCode::NO_CONTENT.into_response()
}

fn fail_worker_publication(
    state: &DaemonState,
    fingerprint: Fingerprint,
    publication: &WorkerPublication,
) {
    let canceled_activation = match publication {
        WorkerPublication::Activation { activation_id } => state
            .registry
            .mark_activation_failed(fingerprint, activation_id)
            .ok()
            .map(|()| activation_id.clone()),
        WorkerPublication::Recovery { .. } => state
            .registry
            .mark_route_pass_through(fingerprint)
            .ok()
            .flatten(),
    };
    if let Some(activation_id) = canceled_activation {
        revoke_activation(state, &activation_id);
    }
}

async fn probe_worker(target: &Arc<WorkerTarget>) -> Result<(), CliError> {
    let uri = format!(
        "{}{}",
        target.endpoint().trim_end_matches('/'),
        WORKER_PROBE_PATH
    )
    .parse::<Uri>()
    .map_err(|error| CliError::Launch(format!("invalid worker readiness endpoint: {error}")))?;
    let request = Request::get(uri)
        .header(WORKER_TOKEN_HEADER, target.session_token())
        .body(box_body(http_body_util::Empty::<Bytes>::new()))?;
    let response = tokio::time::timeout(Duration::from_secs(2), target.client().request(request))
        .await
        .map_err(|_| CliError::Launch("worker readiness probe timed out".into()))?
        .map_err(|error| CliError::Launch(format!("worker readiness probe failed: {error}")))?;
    if response.status() != StatusCode::NO_CONTENT {
        return Err(CliError::Launch(format!(
            "worker readiness probe returned HTTP {}",
            response.status()
        )));
    }
    Ok(())
}

async fn heartbeat_worker(
    State(state): State<Arc<DaemonState>>,
    Json(request): Json<SessionRequest<WorkerHeartbeatPayload>>,
) -> Response<Body> {
    let mut sessions = lock(&state.worker_sessions);
    let Some(session) = sessions.get_mut(&request.session_id) else {
        return control_message(StatusCode::UNAUTHORIZED, "unknown worker session");
    };
    if request.payload.worker_id != session.worker_id {
        return control_message(StatusCode::UNAUTHORIZED, "worker identity mismatch");
    }
    match authenticate_sequence(
        session.secret_digest,
        &mut session.last_sequence,
        &mut session.last_request_id,
        &request,
    ) {
        Ok(_) => {
            session.lease_expires_at_unix_ms = now_unix_ms().saturating_add(WORKER_LEASE_MS);
            StatusCode::NO_CONTENT.into_response()
        }
        Err(response) => response,
    }
}

async fn public_proxy(
    State(state): State<Arc<DaemonState>>,
    mut request: Request<Body>,
) -> Response<Body> {
    let Some(route) = PublicRoute::from_path(request.uri().path()) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let credential = match public_credential(request.headers()) {
        Ok(credential) => credential,
        Err(response) => return response,
    };
    if !state.allowed_route_tokens.contains(&credential.digest()) {
        return control_message(StatusCode::UNAUTHORIZED, "invalid route credential");
    }
    strip_public_relay_headers(request.headers_mut(), route);
    if responses_websocket_probe(&request) {
        return StatusCode::UPGRADE_REQUIRED.into_response();
    }
    if !public_method_allowed(request.method(), request.uri().path()) {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }
    let target = if state.registry.is_global_pass_through() {
        ResolvedTarget::PassThrough
    } else {
        match state.registry.resolve_target(&credential.digest()) {
            Ok(target) => target,
            Err(ResolveError::UnknownToken) => {
                return control_message(StatusCode::UNAUTHORIZED, "invalid route credential");
            }
            Err(ResolveError::Unavailable(_)) => return unavailable_response(),
        }
    };
    match (target, route) {
        (ResolvedTarget::PassThrough, PublicRoute::Hook(hook)) => {
            let mut response = Response::new(Body::from(hook.pass_through_body()));
            response
                .headers_mut()
                .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            response
        }
        (ResolvedTarget::PassThrough, PublicRoute::Provider(provider)) => {
            forward_to_provider(&state, request, provider).await
        }
        (ResolvedTarget::Worker(worker), _) => {
            forward_to_worker(Arc::clone(&state), request, worker).await
        }
    }
}

fn responses_websocket_probe(request: &Request<Body>) -> bool {
    request.method() == http::Method::GET
        && matches!(
            request.uri().path(),
            "/responses" | "/v1/responses" | "/backend-api/codex/responses"
        )
        && request
            .headers()
            .get(axum::http::header::UPGRADE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.eq_ignore_ascii_case("websocket"))
}

fn public_method_allowed(method: &Method, path: &str) -> bool {
    if matches!(path, "/models" | "/v1/models") {
        method == Method::GET
    } else {
        method == Method::POST
    }
}

fn strip_public_relay_headers(headers: &mut HeaderMap, route: PublicRoute) {
    let keep_named_upstream = matches!(route, PublicRoute::Provider(_));
    let private_names = headers
        .keys()
        .filter(|name| {
            name.as_str().starts_with("x-nemo-relay-")
                && name.as_str() != CLIENT_TOKEN_HEADER
                && !(keep_named_upstream
                    && name.as_str() == crate::agents::pi::alignment::UPSTREAM_BASE_URL_HEADER)
        })
        .cloned()
        .collect::<Vec<_>>();
    for name in private_names {
        headers.remove(name);
    }
}

async fn forward_to_provider(
    state: &DaemonState,
    mut request: Request<Body>,
    route: ProviderRoute,
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
    if let Some(aligned) = crate::gateway::daemon_provider_forward_headers(
        request.headers(),
        request.uri().path(),
        &state.config,
    ) {
        *request.headers_mut() = aligned;
    }
    request
        .headers_mut()
        .remove(crate::agents::pi::alignment::UPSTREAM_BASE_URL_HEADER);
    inject_provider_auth(request.headers_mut(), route, &state.config);
    forward(&state.upstream, request, &destination, None, None)
        .await
        .response
}

fn inject_provider_auth(headers: &mut HeaderMap, route: ProviderRoute, config: &GatewayConfig) {
    if crate::provider_auth::has_provider_credential(headers) {
        return;
    }
    let configured = match route {
        ProviderRoute::OpenAi => config.openai_auth_header.as_deref(),
        ProviderRoute::Anthropic => config.anthropic_auth_header.as_deref(),
    };
    if let Some(configured) = configured.and_then(|value| HeaderValue::from_str(value).ok()) {
        headers.insert(AUTHORIZATION, configured);
        return;
    }
    let (name, value) = match route {
        ProviderRoute::OpenAi => {
            let Some(key) = nonempty_environment("OPENAI_API_KEY") else {
                return;
            };
            (AUTHORIZATION, format!("Bearer {key}"))
        }
        ProviderRoute::Anthropic => {
            let Some(key) = nonempty_environment("ANTHROPIC_API_KEY") else {
                return;
            };
            (HeaderName::from_static("x-api-key"), key)
        }
    };
    if let Ok(value) = HeaderValue::from_str(&value) {
        headers.insert(name, value);
    }
}

fn nonempty_environment(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

async fn forward_to_worker(
    state: Arc<DaemonState>,
    request: Request<Body>,
    worker: WorkerRequest,
) -> Response<Body> {
    let fingerprint = worker.fingerprint();
    let worker_id = worker.target().worker_id().to_owned();
    let path_and_query = request
        .uri()
        .path_and_query()
        .map_or("/", |value| value.as_str());
    let destination = format!(
        "{}{}",
        worker.target().endpoint().trim_end_matches('/'),
        path_and_query
    );
    let token = worker.session_token().to_owned();
    let client = worker.target().client().clone();
    let mut outcome = forward(
        &client,
        request,
        &destination,
        Some((HeaderName::from_static(WORKER_TOKEN_HEADER), token)),
        Some(worker),
    )
    .await;
    let route_failure = take_worker_route_failure(&mut outcome.response);
    if outcome.communication_failure || route_failure {
        handle_worker_communication_failure(&state, fingerprint, &worker_id);
        return outcome.response;
    }
    let (parts, body) = outcome.response.into_parts();
    let observed = ErrorObservedBody {
        body,
        on_error: Some(move || {
            handle_worker_communication_failure(&state, fingerprint, &worker_id);
        }),
    };
    Response::from_parts(parts, Body::new(observed))
}

fn take_worker_route_failure(response: &mut Response<Body>) -> bool {
    response
        .headers_mut()
        .remove(WORKER_ROUTE_FAILURE_HEADER)
        .is_some()
}

struct ErrorObservedBody<B, F> {
    body: B,
    on_error: Option<F>,
}

impl<B, F> HttpBody for ErrorObservedBody<B, F>
where
    B: HttpBody<Data = Bytes> + Unpin,
    F: FnOnce() + Unpin,
{
    type Data = Bytes;
    type Error = B::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<hyper::body::Frame<Self::Data>, Self::Error>>> {
        let frame = Pin::new(&mut self.body).poll_frame(context);
        if matches!(frame, Poll::Ready(Some(Err(_))))
            && let Some(on_error) = self.on_error.take()
        {
            on_error();
        }
        frame
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> hyper::body::SizeHint {
        self.body.size_hint()
    }
}

struct ForwardOutcome {
    response: Response<Body>,
    communication_failure: bool,
}

impl ForwardOutcome {
    fn response(response: Response<Body>) -> Self {
        Self {
            response,
            communication_failure: false,
        }
    }

    fn communication_failure(response: Response<Body>) -> Self {
        Self {
            response,
            communication_failure: true,
        }
    }
}

async fn forward(
    client: &PooledClient,
    request: Request<Body>,
    destination: &str,
    authentication: Option<(HeaderName, String)>,
    hold: Option<WorkerRequest>,
) -> ForwardOutcome {
    let destination = match destination.parse::<Uri>() {
        Ok(destination) => destination,
        Err(_) => {
            return ForwardOutcome::response(control_message(
                StatusCode::BAD_GATEWAY,
                "invalid upstream destination",
            ));
        }
    };
    let strip = [
        HeaderName::from_static(CLIENT_TOKEN_HEADER),
        HeaderName::from_static(WORKER_TOKEN_HEADER),
    ];
    let mut request = match prepare_forward_request(request, destination, &strip) {
        Ok(request) => request.map(box_body),
        Err(error) => {
            return ForwardOutcome::response(control_error(StatusCode::BAD_REQUEST, error));
        }
    };
    if let Some((name, value)) = authentication {
        let Ok(value) = HeaderValue::from_str(&value) else {
            return ForwardOutcome::response(control_message(
                StatusCode::INTERNAL_SERVER_ERROR,
                "invalid worker credential",
            ));
        };
        request.headers_mut().insert(name, value);
    }
    let response = match tokio::time::timeout(RESPONSE_HEAD_TIMEOUT, client.request(request)).await
    {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            return ForwardOutcome::communication_failure(control_error(
                StatusCode::BAD_GATEWAY,
                error,
            ));
        }
        Err(_) => {
            return ForwardOutcome::communication_failure(control_message(
                StatusCode::GATEWAY_TIMEOUT,
                "response-head timeout",
            ));
        }
    };
    let response = match prepare_forward_response(response, &strip) {
        Ok(response) => response,
        Err(error) => {
            return ForwardOutcome::communication_failure(control_error(
                StatusCode::BAD_GATEWAY,
                error,
            ));
        }
    };
    let (parts, body) = response.into_parts();
    let body: RelayBody = match hold {
        Some(hold) => hold_body(body, hold),
        None => box_body(body),
    };
    ForwardOutcome::response(Response::from_parts(parts, Body::new(body)))
}

fn handle_worker_communication_failure(
    state: &DaemonState,
    fingerprint: Fingerprint,
    worker_id: &str,
) {
    revoke_worker_generation(state, worker_id);
    let Ok(canceled_activation) = state
        .registry
        .mark_worker_communication_failed(fingerprint, worker_id)
    else {
        return;
    };
    if let Some(activation_id) = canceled_activation {
        revoke_activation(state, &activation_id);
    }
    lock(&state.worker_sessions).remove(worker_id);
    let fingerprint = fingerprint.to_string();
    log::error!(
        target: "nemo_relay.daemon",
        event = "worker_communication_failed",
        fingerprint = fingerprint.as_str(),
        worker_id = worker_id;
        "Worker communication failed; route changed to pass-through"
    );
}

struct AuthenticatedMcp {
    fingerprint: Fingerprint,
    session_id: McpSessionId,
    duplicate: bool,
    cached_heartbeat: Option<McpHeartbeatResponse>,
    released: bool,
}

#[allow(clippy::result_large_err)]
fn authenticate_mcp<T: Serialize>(
    state: &DaemonState,
    request: &SessionRequest<T>,
    renew_lease_expires_at_unix_ms: Option<u64>,
) -> Result<AuthenticatedMcp, Response<Body>> {
    let mut sessions = lock(&state.mcp_sessions);
    let Some(session) = sessions.get_mut(&request.session_id) else {
        return Err(control_message(
            StatusCode::UNAUTHORIZED,
            "unknown MCP session",
        ));
    };
    let duplicate = authenticate_sequence(
        session.secret_digest,
        &mut session.last_sequence,
        &mut session.last_request_id,
        request,
    )?;
    if !session.released
        && !duplicate
        && let Some(lease_expires_at_unix_ms) = renew_lease_expires_at_unix_ms
    {
        session.lease_expires_at_unix_ms = lease_expires_at_unix_ms;
    }
    let cached_heartbeat = cached_heartbeat_response(session, request, duplicate);
    let session_id = McpSessionId::new(request.session_id.clone())
        .map_err(|error| control_error(StatusCode::BAD_REQUEST, error))?;
    Ok(AuthenticatedMcp {
        fingerprint: session.fingerprint,
        session_id,
        duplicate,
        cached_heartbeat,
        released: session.released,
    })
}

fn cached_heartbeat_response<T>(
    session: &McpControlSession,
    request: &SessionRequest<T>,
    duplicate: bool,
) -> Option<McpHeartbeatResponse> {
    duplicate
        .then_some(session.last_heartbeat.as_ref())
        .flatten()
        .filter(|cached| {
            cached.sequence == request.sequence && cached.request_id == request.request_id
        })
        .map(|cached| cached.response.clone())
}

#[allow(clippy::result_large_err)]
fn authenticate_sequence<T: Serialize>(
    expected_secret: TokenDigest,
    last_sequence: &mut u64,
    last_request_id: &mut String,
    request: &SessionRequest<T>,
) -> Result<bool, Response<Body>> {
    if !expected_secret.matches(&TokenDigest::from_token(
        request.session_token.expose().as_bytes(),
    )) || !request.validate_payload_hash()
        || request.request_id.is_empty()
        || request.request_id.len() > 128
    {
        return Err(control_message(
            StatusCode::UNAUTHORIZED,
            "invalid control message authentication",
        ));
    }
    if request.sequence == *last_sequence && request.request_id == *last_request_id {
        return Ok(true);
    }
    if request.sequence != last_sequence.saturating_add(1) {
        return Err(control_message(
            StatusCode::CONFLICT,
            "control sequence is stale or out of order",
        ));
    }
    *last_sequence = request.sequence;
    *last_request_id = request.request_id.clone();
    Ok(false)
}

#[allow(clippy::result_large_err)]
fn validate_registration(
    state: &DaemonState,
    proof: &crate::daemon::common::control::RegistrationProof,
) -> Result<HandshakeProof, Response<Body>> {
    let transcript = &proof.transcript;
    if !has_required_transport_capabilities(&transcript.initiator) {
        return Err(control_message(
            StatusCode::UPGRADE_REQUIRED,
            "component lacks required lossless streaming and trailer capabilities",
        ));
    }
    let pending = lock(&state.challenges).remove(&transcript.challenge_id);
    let Some(mut pending) = pending else {
        return Err(control_message(
            StatusCode::UNAUTHORIZED,
            "unknown, expired, or replayed challenge",
        ));
    };
    if let Err(error) = pending
        .record
        .consume(&transcript.challenge_id, now_unix_ms())
    {
        return Err(control_error(StatusCode::UNAUTHORIZED, error));
    }
    let request = pending.request;
    let selected = request
        .initiator
        .protocol
        .negotiate(state.descriptor.protocol)
        .map_err(|error| control_error(StatusCode::UNAUTHORIZED, error))?;
    if transcript.daemon_target != state.public_origin
        || transcript.initiator != request.initiator
        || transcript.responder != state.descriptor
        || transcript.initiator_instance_id != request.initiator_instance_id
        || transcript.responder_instance_id != state.instance_id
        || transcript.selected_protocol != selected
        || transcript.initiator_public_identity != request.initiator_public_identity
        || transcript.responder_public_identity != state.identity.public_identity()
        || transcript.initiator_fingerprint != request.initiator_fingerprint
        || transcript.responder_fingerprint != state.identity.fingerprint()
        || transcript.initiator_nonce != request.initiator_nonce
        || transcript.responder_nonce != pending.record.challenge().nonce
        || proof.initiator_proof.signer != request.initiator.role
    {
        return Err(control_message(
            StatusCode::UNAUTHORIZED,
            "handshake transcript does not match the issued challenge",
        ));
    }
    transcript
        .verify(&proof.initiator_proof)
        .map_err(|error| control_error(StatusCode::UNAUTHORIZED, error))?;
    transcript
        .sign(ComponentRole::Daemon, &state.identity)
        .map_err(|error| control_error(StatusCode::INTERNAL_SERVER_ERROR, error))
}

fn has_required_transport_capabilities(
    descriptor: &crate::daemon::common::protocol::ComponentDescriptor,
) -> bool {
    descriptor
        .capabilities
        .includes(&Capabilities::streaming_transport())
}

fn fresh_launch(worker_network: WorkerNetworkHint) -> Result<WorkerLaunch, CliError> {
    worker_network.validate()?;
    let now = now_unix_ms();
    let loopback = worker_network.is_loopback();
    Ok(WorkerLaunch {
        activation_id: random_secret(16)?,
        activation_token: SensitiveString::new(random_secret(32)?)
            .map_err(|error| CliError::Launch(error.to_string()))?,
        deadline_unix_ms: now.saturating_add(ACTIVATION_LIFETIME_MS),
        bind_ip: if loopback {
            Ipv4Addr::LOCALHOST
        } else {
            Ipv4Addr::UNSPECIFIED
        },
        port: worker_network.port.unwrap_or(0),
        advertise_address: (!loopback).then_some(worker_network.advertised_host),
    })
}

fn remember_activation(state: &DaemonState, fingerprint: Fingerprint, directive: &BrokerDirective) {
    if let BrokerDirective::LaunchWorker {
        activation_id,
        activation_token,
        deadline_unix_ms,
        bind_ip,
        port,
        advertise_address,
        ..
    } = directive
    {
        lock(&state.activations)
            .entry(activation_id.clone())
            .or_insert_with(|| Activation {
                fingerprint,
                secret_digest: TokenDigest::from_token(activation_token.expose().as_bytes()),
                deadline_unix_ms: *deadline_unix_ms,
                consumed: false,
                bind_ip: *bind_ip,
                port: *port,
                advertise_address: advertise_address.clone(),
            });
    }
}

fn revoke_activation(state: &DaemonState, activation_id: &str) {
    lock(&state.activations).remove(activation_id);
    lock(&state.pending_directives).retain(|_, directive| {
        !matches!(
            directive,
            BrokerDirective::LaunchWorker {
                activation_id: pending,
                ..
            } if pending == activation_id
        )
    });
}

fn expire_activation_routes(state: &DaemonState, now_unix_ms: u64) {
    for ExpiredActivation {
        fingerprint,
        activation_id,
    } in state.registry.expire_activations(now_unix_ms)
    {
        revoke_activation(state, &activation_id);
        let fingerprint = fingerprint.to_string();
        log::error!(
            target: "nemo_relay.daemon",
            event = "worker_activation_expired",
            fingerprint = fingerprint.as_str();
            "Worker activation expired; route changed to pass-through"
        );
    }
}

fn handle_release_action(state: Arc<DaemonState>, fingerprint: Fingerprint, action: ReleaseAction) {
    match action {
        ReleaseAction::NoChange => {}
        ReleaseAction::CancelActivation { activation_id } => {
            revoke_activation(&state, &activation_id);
        }
        ReleaseAction::BeginDrain {
            target,
            deadline_unix_ms,
        } => {
            revoke_worker_generation(&state, target.worker_id());
            tokio::spawn(async move {
                request_worker_drain(&state, &target, deadline_unix_ms).await;
                loop {
                    let now = now_unix_ms();
                    if target.in_flight() == 0 || now >= deadline_unix_ms {
                        let _ = state.registry.finish_draining(fingerprint, now);
                        lock(&state.worker_sessions).remove(target.worker_id());
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            });
        }
        ReleaseAction::TransferActivation {
            session_id,
            directive,
        } => {
            lock(&state.pending_directives).insert(session_id.as_str().to_owned(), directive);
        }
        ReleaseAction::NominateMcp { session_id } => {
            nominate_relaunch(&state, fingerprint, session_id);
        }
    }
}

fn revoke_worker_generation(state: &DaemonState, worker_id: &str) {
    if let Some((fingerprint, generation_id)) =
        lock(&state.worker_sessions).get(worker_id).map(|session| {
            (
                session.fingerprint,
                session.generation_grant.generation_id.clone(),
            )
        })
    {
        revoke_active_worker_generation(state, fingerprint, &generation_id);
    }
}

fn revoke_active_worker_generation(
    state: &DaemonState,
    fingerprint: Fingerprint,
    generation_id: &str,
) -> bool {
    let _generation_publication = lock(&state.worker_generation_publication);
    match state
        .active_worker_generations
        .revoke_if_matches(fingerprint, generation_id)
    {
        Ok(revoked) => revoked,
        Err(error) => {
            let fingerprint = fingerprint.to_string();
            log::error!(
                target: "nemo_relay.daemon",
                event = "worker_generation_revocation_failed",
                fingerprint = fingerprint.as_str(),
                error_kind = error.log_kind();
                "Failed to durably revoke worker generation"
            );
            true
        }
    }
}

fn nominate_relaunch(state: &Arc<DaemonState>, fingerprint: Fingerprint, session_id: McpSessionId) {
    let worker_network = lock(&state.mcp_sessions)
        .get(session_id.as_str())
        .filter(|session| !session.released)
        .map(|session| session.worker_network.clone());
    let Some(worker_network) = worker_network else {
        return;
    };
    let Ok(launch) = fresh_launch(worker_network) else {
        return;
    };
    let Ok(directive) = state
        .registry
        .begin_relaunch(fingerprint, &session_id, launch)
    else {
        return;
    };
    remember_activation(state, fingerprint, &directive);
    lock(&state.pending_directives).insert(session_id.as_str().to_owned(), directive);
}

async fn request_worker_drain(
    state: &DaemonState,
    target: &Arc<WorkerTarget>,
    deadline_unix_ms: u64,
) {
    let request = {
        let mut sessions = lock(&state.worker_sessions);
        let Some(session) = sessions.get_mut(target.worker_id()) else {
            return;
        };
        let Some(sequence) = session.next_daemon_sequence.checked_add(1) else {
            return;
        };
        session.next_daemon_sequence = sequence;
        SessionRequest::new(
            target.worker_id().to_owned(),
            session.secret.clone(),
            sequence,
            WorkerDrainRequest {
                worker_id: target.worker_id().to_owned(),
                deadline_unix_ms,
                timeout_ms: Some(
                    deadline_unix_ms
                        .saturating_sub(now_unix_ms())
                        .min(DRAIN_LIFETIME_MS),
                ),
            },
        )
    };
    let Ok(request) = request else {
        return;
    };
    let uri = format!(
        "{}{}",
        target.endpoint().trim_end_matches('/'),
        WORKER_DRAIN_PATH
    );
    let Ok(uri) = uri.parse::<Uri>() else {
        return;
    };
    let payload = match serde_json::to_vec(&request) {
        Ok(payload) => payload,
        Err(_) => return,
    };
    let payload = Bytes::from(payload);
    for _ in 0..2 {
        let request = match Request::post(uri.clone())
            .header(CONTENT_TYPE, "application/json")
            .header(WORKER_TOKEN_HEADER, target.session_token())
            .body(box_body(http_body_util::Full::new(payload.clone())))
        {
            Ok(request) => request,
            Err(_) => return,
        };
        match tokio::time::timeout(Duration::from_secs(2), target.client().request(request)).await {
            Ok(Ok(response)) if response.status() == StatusCode::NO_CONTENT => return,
            _ => {}
        }
    }
}

fn spawn_maintenance(state: Arc<DaemonState>) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(MAINTENANCE_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            let now = now_unix_ms();
            expire_activation_routes(&state, now);
            lock(&state.activations).retain(|_, activation| activation.deadline_unix_ms > now);
            let actions = state
                .registry
                .expire_mcp_leases(now, now.saturating_add(DRAIN_LIFETIME_MS));
            for (fingerprint, action) in actions {
                handle_release_action(Arc::clone(&state), fingerprint, action);
            }
            prune_expired_mcp_control_state(
                &mut lock(&state.mcp_sessions),
                &mut lock(&state.pending_directives),
                now,
            );
            let expired_workers: Vec<_> = {
                let mut sessions = lock(&state.worker_sessions);
                let expired = sessions
                    .iter()
                    .filter(|(_, session)| session.lease_expires_at_unix_ms <= now)
                    .map(|(id, session)| {
                        (
                            id.clone(),
                            session.fingerprint,
                            session.generation_grant.generation_id.clone(),
                        )
                    })
                    .collect::<Vec<_>>();
                for (id, _, _) in &expired {
                    sessions.remove(id);
                }
                expired
            };
            for (worker_id, fingerprint, generation_id) in expired_workers {
                revoke_active_worker_generation(&state, fingerprint, &generation_id);
                if let Ok(WorkerFailureAction::NominateMcp { session_id }) =
                    state.registry.worker_failed(
                        fingerprint,
                        &worker_id,
                        now.saturating_add(RECOVERY_LIFETIME_MS),
                    )
                {
                    nominate_relaunch(&state, fingerprint, session_id);
                }
            }
        }
    });
}

fn prune_expired_mcp_control_state(
    sessions: &mut HashMap<String, McpControlSession>,
    pending_directives: &mut HashMap<String, BrokerDirective>,
    now_unix_ms: u64,
) {
    let expired = sessions
        .iter()
        .filter(|(_, session)| session.lease_expires_at_unix_ms <= now_unix_ms)
        .map(|(session_id, _)| session_id.clone())
        .collect::<Vec<_>>();
    for session_id in expired {
        sessions.remove(&session_id);
        pending_directives.remove(&session_id);
    }
}

#[allow(clippy::result_large_err)]
fn public_credential(headers: &HeaderMap) -> Result<RouteCredential, Response<Body>> {
    let values = headers.get_all(CLIENT_TOKEN_HEADER);
    if values.iter().count() != 1 {
        return Err(control_message(
            StatusCode::UNAUTHORIZED,
            "exactly one route credential is required",
        ));
    }
    let value = values
        .iter()
        .next()
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| control_message(StatusCode::UNAUTHORIZED, "invalid route credential"))?;
    RouteCredential::parse(value.to_owned())
        .map_err(|_| control_message(StatusCode::UNAUTHORIZED, "invalid route credential"))
}

fn load_allowed_route_tokens(path: Option<&Path>) -> Result<HashSet<TokenDigest>, CliError> {
    let mut digests = HashSet::new();
    if let Some(value) = std::env::var_os(ROUTE_TOKEN_ENV) {
        let value = value.into_string().map_err(|_| {
            CliError::Config(format!("{ROUTE_TOKEN_ENV} must contain valid Unicode text"))
        })?;
        let credential = RouteCredential::parse(value)?;
        digests.insert(credential.digest());
    }
    if let Some(path) = path {
        let bytes = crate::filesystem::bounded::read_bounded_regular_file(
            path,
            "daemon client-token allowlist",
        )
        .map_err(CliError::Config)?;
        let text = std::str::from_utf8(&bytes).map_err(|_| {
            CliError::Config(format!(
                "daemon client-token allowlist {} must be UTF-8",
                path.display()
            ))
        })?;
        for (index, line) in text.lines().enumerate() {
            let value = line.trim();
            if value.is_empty() || value.starts_with('#') {
                continue;
            }
            if digests.len() >= MAX_ALLOWED_ROUTE_TOKENS {
                return Err(CliError::Config(format!(
                    "daemon client-token allowlist exceeds {MAX_ALLOWED_ROUTE_TOKENS} entries"
                )));
            }
            let credential = RouteCredential::parse(value.to_owned()).map_err(|_| {
                CliError::Config(format!(
                    "daemon client-token allowlist {} has an invalid token on line {}",
                    path.display(),
                    index + 1
                ))
            })?;
            digests.insert(credential.digest());
        }
    }
    if digests.is_empty() {
        return Err(CliError::Config(format!(
            "daemon requires an administrator-provisioned client token via {ROUTE_TOKEN_ENV} or --client-token-file"
        )));
    }
    Ok(digests)
}

fn validate_worker_endpoint(
    endpoint: &str,
    tls_root_certificate: Option<&str>,
) -> Result<(), CliError> {
    let explicit_port = endpoint
        .parse::<Uri>()
        .ok()
        .and_then(|uri| uri.authority().and_then(http::uri::Authority::port_u16));
    let url = reqwest::Url::parse(endpoint)
        .map_err(|error| CliError::Config(format!("invalid worker endpoint: {error}")))?;
    if !matches!(url.scheme(), "http" | "https")
        || explicit_port.is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
        || url.host_str() == Some("0.0.0.0")
    {
        return Err(CliError::Config("invalid worker endpoint origin".into()));
    }
    let host_is_loopback = url.host_str().is_some_and(|host| {
        host.eq_ignore_ascii_case("localhost")
            || host
                .parse::<IpAddr>()
                .is_ok_and(|address| address.is_loopback())
    });
    match (url.scheme(), host_is_loopback, tls_root_certificate) {
        ("http", true, None) | ("https", _, Some(_)) => Ok(()),
        ("http", false, _) => Err(CliError::Config(
            "non-loopback worker endpoints must use pinned TLS".into(),
        )),
        ("http", true, Some(_)) | ("https", _, None) => Err(CliError::Config(
            "worker endpoint scheme and TLS trust anchor do not match".into(),
        )),
        _ => Err(CliError::Config("invalid worker endpoint origin".into())),
    }
}

fn activation_endpoint_matches(endpoint: &str, activation: &Activation) -> bool {
    let Ok(url) = reqwest::Url::parse(endpoint) else {
        return false;
    };
    let (Some(host), Some(port)) = (url.host_str(), url.port()) else {
        return false;
    };
    let expected_host = activation
        .advertise_address
        .as_deref()
        .unwrap_or("127.0.0.1");
    let expected_scheme = if activation.bind_ip.is_unspecified() {
        "https"
    } else {
        "http"
    };
    url.scheme() == expected_scheme
        && host == expected_host
        && port != 0
        && (activation.port == 0 || activation.port == port)
}

fn daemon_origin(options: &ServerOptions, local: SocketAddr) -> Result<String, CliError> {
    if let Some(advertised) = options.advertise_address.as_deref() {
        let url = daemon_url(advertised)?;
        let host_is_loopback = url.host_str().is_some_and(|host| {
            host.eq_ignore_ascii_case("localhost")
                || host
                    .parse::<IpAddr>()
                    .is_ok_and(|address| address.is_loopback())
        });
        if options.bind == Ipv4Addr::UNSPECIFIED && host_is_loopback {
            return Err(CliError::Config(
                "a daemon bound to 0.0.0.0 must advertise a concrete reachable host".into(),
            ));
        }
        if options.tls_cert.is_some() && url.scheme() != "https" {
            return Err(CliError::Config(
                "the advertised daemon URL must use https when native TLS is configured".into(),
            ));
        }
        return Ok(url.as_str().trim_end_matches('/').to_owned());
    }
    if options.bind == Ipv4Addr::UNSPECIFIED {
        return Err(CliError::Config(
            "--advertise-address is required when the daemon binds to 0.0.0.0".into(),
        ));
    }
    Ok(format!(
        "{}://{local}",
        if options.tls_cert.is_some() {
            "https"
        } else {
            "http"
        }
    ))
}

fn load_tls_config(
    certificate_path: &Path,
    key_path: &Path,
) -> Result<Arc<rustls::ServerConfig>, CliError> {
    let certificate_pem = crate::filesystem::bounded::read_bounded_regular_file(
        certificate_path,
        "daemon TLS certificate",
    )
    .map_err(CliError::Config)?;
    let key_pem =
        crate::filesystem::bounded::read_bounded_regular_file(key_path, "daemon TLS private key")
            .map_err(CliError::Config)?;
    let certificates = decode_pem_blocks(&certificate_pem, "CERTIFICATE")?
        .into_iter()
        .map(CertificateDer::from)
        .collect::<Vec<_>>();
    if certificates.is_empty() {
        return Err(CliError::Config(format!(
            "daemon TLS certificate {} contains no CERTIFICATE blocks",
            certificate_path.display()
        )));
    }
    let mut keys = decode_pem_blocks(&key_pem, "PRIVATE KEY")?;
    if keys.len() != 1 {
        return Err(CliError::Config(format!(
            "daemon TLS key {} must contain exactly one unencrypted PKCS#8 PRIVATE KEY block",
            key_path.display()
        )));
    }
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(keys.remove(0)));
    let _ = rustls::crypto::ring::default_provider().install_default();
    let mut config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certificates, key)
        .map_err(|error| CliError::Config(format!("invalid daemon TLS identity: {error}")))?;
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(Arc::new(config))
}

fn decode_pem_blocks(bytes: &[u8], label: &str) -> Result<Vec<Vec<u8>>, CliError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| CliError::Config(format!("TLS PEM is not UTF-8: {error}")))?;
    let begin = format!("-----BEGIN {label}-----");
    let end = format!("-----END {label}-----");
    let mut blocks = Vec::new();
    let mut remainder = text;
    while let Some((_, after_begin)) = remainder.split_once(&begin) {
        let Some((encoded, after_end)) = after_begin.split_once(&end) else {
            return Err(CliError::Config(format!(
                "TLS PEM has an unterminated {label} block"
            )));
        };
        let compact = encoded
            .chars()
            .filter(|character| !character.is_ascii_whitespace())
            .collect::<String>();
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(compact)
            .map_err(|_| CliError::Config(format!("TLS PEM contains invalid {label} base64")))?;
        if decoded.is_empty() {
            return Err(CliError::Config(format!(
                "TLS PEM contains an empty {label} block"
            )));
        }
        blocks.push(decoded);
        remainder = after_end;
    }
    Ok(blocks)
}

async fn serve_tls(
    listener: TcpListener,
    app: Router,
    config: Arc<rustls::ServerConfig>,
) -> Result<(), CliError> {
    let acceptor = tokio_rustls::TlsAcceptor::from(config);
    let handshake_permits = Arc::new(Semaphore::new(MAX_CONCURRENT_TLS_HANDSHAKES));
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let mut connections = tokio::task::JoinSet::new();
    let mut shutdown = Box::pin(shutdown_signal());
    loop {
        tokio::select! {
            biased;
            _ = &mut shutdown => break,
            accepted = listener.accept() => {
                let (stream, _) = accepted?;
                stream.set_nodelay(true)?;
                let Ok(handshake_permit) = Arc::clone(&handshake_permits).try_acquire_owned() else {
                    continue;
                };
                let acceptor = acceptor.clone();
                let service = app.clone();
                let mut shutdown_rx = shutdown_rx.clone();
                connections.spawn(async move {
                    let Ok(Ok(stream)) = tokio::time::timeout(
                        TLS_HANDSHAKE_TIMEOUT,
                        acceptor.accept(stream),
                    ).await else {
                        return;
                    };
                    drop(handshake_permit);
                    let builder = ConnectionBuilder::new(TokioExecutor::new());
                    let connection = builder.serve_connection_with_upgrades(
                        TokioIo::new(stream),
                        TowerToHyperService::new(service),
                    );
                    tokio::pin!(connection);
                    tokio::select! {
                        _ = &mut connection => {}
                        changed = shutdown_rx.changed() => {
                            if changed.is_ok() {
                                connection.as_mut().graceful_shutdown();
                                let _ = connection.await;
                            }
                        }
                    }
                });
            }
        }
    }
    let _ = shutdown_tx.send(true);
    let drain = async { while connections.join_next().await.is_some() {} };
    let _ = tokio::time::timeout(Duration::from_millis(DRAIN_LIFETIME_MS), drain).await;
    Ok(())
}

fn unavailable_response() -> Response<Body> {
    let mut response = control_message(StatusCode::SERVICE_UNAVAILABLE, "route is not ready");
    response
        .headers_mut()
        .insert(RETRY_AFTER, HeaderValue::from_static("1"));
    response
}

fn registry_error(error: RegistryError) -> Response<Body> {
    let status = match error {
        RegistryError::TokenAlreadyBound | RegistryError::FingerprintTokenMismatch => {
            StatusCode::UNAUTHORIZED
        }
        RegistryError::UnknownRoute | RegistryError::UnknownMcpSession => StatusCode::NOT_FOUND,
        RegistryError::RouteCapacityReached | RegistryError::McpReferenceCapacityReached => {
            StatusCode::TOO_MANY_REQUESTS
        }
        RegistryError::ActivationMismatch
        | RegistryError::WorkerMismatch
        | RegistryError::NoLiveMcpReferences
        | RegistryError::NotLaunchOwner
        | RegistryError::RecoveryNotAuthorized
        | RegistryError::RecoveryGenerationChanged
        | RegistryError::InvalidState { .. }
        | RegistryError::DrainInProgress => StatusCode::CONFLICT,
        #[cfg(test)]
        RegistryError::RecoveryInProgress => StatusCode::CONFLICT,
    };
    control_error(status, error)
}

fn control_error(status: StatusCode, error: impl std::fmt::Display) -> Response<Body> {
    control_message(status, &error.to_string())
}

fn control_message(status: StatusCode, message: &str) -> Response<Body> {
    (status, Json(json!({ "error": { "message": message } }))).into_response()
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
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
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/server_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/daemon_worker_e2e_tests.rs"]
mod daemon_worker_e2e_tests;
