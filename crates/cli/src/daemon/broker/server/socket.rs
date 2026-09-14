// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Persistent control sessions and disconnect-driven grace periods.
use super::*;
use crate::daemon::common::socket::{
    ATTEMPT_TIMEOUT, Command, Event, GRACE, QUEUE_CAPACITY, Request as ControlRequest,
};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::{Notify, mpsc};

#[derive(Default)]
pub(super) struct Hub {
    peers: Mutex<HashMap<String, Peer>>,
    restarting: Mutex<HashMap<Fingerprint, String>>,
    restart_deadline: Option<tokio::time::Instant>,
    pub(super) changed: Notify,
    // Fence callbacks and reconnects for one logical session without blocking other peers.
    gates: Mutex<HashMap<String, std::sync::Weak<tokio::sync::Mutex<()>>>>,
}
struct Peer {
    generation: String,
    sender: Option<mpsc::Sender<Message>>,
    cancel: Arc<Notify>,
    disconnected: Option<tokio::time::Instant>,
    ready: bool,
    acknowledgments: HashMap<String, tokio::time::Instant>,
    last_acknowledged: Option<String>,
    drain: Option<(String, WorkerDrainRequest, tokio::time::Instant)>,
}
fn key(role: ComponentRole, id: &str) -> String {
    format!("{role:?}:{id}")
}
impl Hub {
    async fn transaction(&self, role: ComponentRole, id: &str) -> tokio::sync::OwnedMutexGuard<()> {
        let gate = {
            let mut gates = lock(&self.gates);
            let key = key(role, id);
            match gates.get(&key).and_then(std::sync::Weak::upgrade) {
                Some(gate) => gate,
                None => {
                    gates.retain(|_, gate| gate.strong_count() != 0);
                    let gate = Arc::new(tokio::sync::Mutex::new(()));
                    gates.insert(key, Arc::downgrade(&gate));
                    gate
                }
            }
        };
        gate.lock_owned().await
    }

    pub(super) fn restarting(generations: HashMap<Fingerprint, String>) -> Self {
        Self {
            restarting: Mutex::new(generations),
            restart_deadline: Some(tokio::time::Instant::now() + GRACE),
            ..Self::default()
        }
    }
    pub(super) fn deferred(&self, fingerprint: Fingerprint) -> bool {
        lock(&self.restarting).contains_key(&fingerprint)
    }
    pub(super) fn restored(&self, fingerprint: Fingerprint) {
        lock(&self.restarting).remove(&fingerprint);
    }
    pub(super) fn defer_launch(&self, fingerprint: Fingerprint, launch: &mut WorkerLaunch) {
        if self.deferred(fingerprint) {
            let remaining = self
                .restart_deadline
                .unwrap()
                .saturating_duration_since(tokio::time::Instant::now());
            launch.deadline_unix_ms = now_unix_ms()
                .saturating_add(remaining.as_millis() as u64)
                .saturating_add(ACTIVATION_LIFETIME_MS);
        }
    }
    pub(super) fn directive(
        &self,
        fingerprint: Fingerprint,
        directive: BrokerDirective,
    ) -> BrokerDirective {
        if self.deferred(fingerprint) && matches!(directive, BrokerDirective::LaunchWorker { .. }) {
            BrokerDirective::WaitForWorker {
                retry_after_ms: 100,
            }
        } else {
            directive
        }
    }
    fn emit(peer: &mut Peer, event: Event) {
        let Some(sender) = &peer.sender else { return };
        let request_id = match &event {
            Event::Directive { request_id, .. } | Event::Drain { request_id, .. } => request_id,
            Event::Reply { .. } => return,
        };
        if peer.acknowledgments.len() >= QUEUE_CAPACITY {
            peer.cancel.notify_one();
            return;
        }
        peer.acknowledgments.insert(
            request_id.clone(),
            tokio::time::Instant::now() + ATTEMPT_TIMEOUT,
        );
        let Ok(text) = serde_json::to_string(&event) else {
            peer.cancel.notify_one();
            return;
        };
        if text.len() > MAX_CONTROL_BODY_BYTES
            || sender.try_send(Message::Text(text.into())).is_err()
        {
            peer.cancel.notify_one();
        }
    }
    pub(super) fn cancel_worker(&self, worker_id: &str) {
        if let Some(peer) = lock(&self.peers).get(&key(ComponentRole::Worker, worker_id)) {
            peer.cancel.notify_one();
        }
    }
    pub(super) fn draining(&self, worker_id: &str) -> bool {
        lock(&self.peers)
            .get(&key(ComponentRole::Worker, worker_id))
            .is_some_and(|p| p.drain.is_some())
    }
    pub(super) fn drain(&self, request: WorkerDrainRequest) {
        if let Some(peer) =
            lock(&self.peers).get_mut(&key(ComponentRole::Worker, &request.worker_id))
        {
            let id = uuid::Uuid::now_v7().to_string();
            peer.ready = false;
            peer.drain = Some((
                id.clone(),
                request.clone(),
                tokio::time::Instant::now()
                    + Duration::from_millis(request.timeout_ms.unwrap_or(DRAIN_LIFETIME_MS)),
            ));
            Self::emit(
                peer,
                Event::Drain {
                    request_id: id,
                    request,
                },
            );
        }
        self.changed.notify_waiters();
    }
}

pub(super) async fn mcp(
    State(state): State<Arc<DaemonState>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    local: Option<axum::Extension<LocalAddress>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    upgrade(state, ComponentRole::Mcp, peer, local.map(|v| v.0.0), ws)
}
pub(super) async fn worker(
    State(state): State<Arc<DaemonState>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    local: Option<axum::Extension<LocalAddress>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    upgrade(state, ComponentRole::Worker, peer, local.map(|v| v.0.0), ws)
}
#[derive(Clone, Copy)]
pub(super) struct LocalAddress(pub(super) SocketAddr);
fn upgrade(
    state: Arc<DaemonState>,
    role: ComponentRole,
    peer: SocketAddr,
    local: Option<SocketAddr>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    let local = local.is_some_and(|local| local.ip().is_loopback()) && peer.ip().is_loopback();
    ws.max_message_size(MAX_CONTROL_BODY_BYTES)
        .max_frame_size(MAX_CONTROL_BODY_BYTES)
        .on_upgrade(move |socket| run(state, role, local, socket))
}

pub(super) fn recover_after_restart(state: Arc<DaemonState>) {
    let Some(deadline) = state.sockets.restart_deadline else {
        return;
    };
    tokio::spawn(async move {
        tokio::time::sleep_until(deadline).await;
        let work = state.clone();
        let _ = tokio::task::spawn_blocking(move || {
            // Publication removes restored generations while holding this same durable-state
            // lock. Taking the pending set here prevents expiry from revoking a recovered peer.
            let _publication = lock(&work.worker_generation_publication);
            let generations = std::mem::take(&mut *lock(&work.sockets.restarting));
            for (fingerprint, generation) in generations {
                if let Err(error) = work
                    .active_worker_generations
                    .revoke_if_matches(fingerprint, &generation)
                {
                    log::error!(
                        target: "nemo_relay.daemon",
                        event = "worker_generation_revocation_failed",
                        error_kind = error.log_kind();
                        "Failed to revoke expired worker generation"
                    );
                }
            }
        })
        .await;
        state.sockets.changed.notify_waiters();
    });
}

#[allow(clippy::cognitive_complexity)] // One select loop owns the connection state and cancellation.
async fn run(state: Arc<DaemonState>, role: ComponentRole, local: bool, socket: WebSocket) {
    let (mut writer, mut reader) = socket.split();
    let (send, mut outgoing) = mpsc::channel::<Message>(QUEUE_CAPACITY);
    let cancel = Arc::new(Notify::new());
    let writer_cancel = cancel.clone();
    let writing = tokio::spawn(async move {
        while let Some(message) = outgoing.recv().await {
            if !matches!(
                tokio::time::timeout(ATTEMPT_TIMEOUT, writer.send(message)).await,
                Ok(Ok(()))
            ) {
                break;
            }
        }
        writer_cancel.notify_one();
    });
    let generation = uuid::Uuid::now_v7().to_string();
    let mut id: Option<String> = None;
    let mut challenge: Option<ChallengeId> = None;
    let mut last_directive = None;
    let mut last_reply: Option<(String, String, Event)> = None;
    let auth_deadline = tokio::time::Instant::now() + ATTEMPT_TIMEOUT;
    let mut ping_at = tokio::time::Instant::now() + Duration::from_secs(30);
    let mut pong: Option<(Bytes, tokio::time::Instant)> = None;
    loop {
        let changed = state.sockets.changed.notified();
        tokio::pin!(changed);
        changed.as_mut().enable();
        if let Some(id) = &id {
            let mut peers = lock(&state.sockets.peers);
            let Some(peer) = peers
                .get_mut(&key(role, id))
                .filter(|p| p.generation == generation)
            else {
                break;
            };
            if role == ComponentRole::Mcp {
                let session = lock(&state.mcp_sessions)
                    .get(id)
                    .filter(|s| !s.released)
                    .map(|s| s.fingerprint);
                if let Some(fingerprint) = session {
                    let directive = lock(&state.pending_directives).remove(id).or_else(|| {
                        McpSessionId::new(id.clone())
                            .ok()
                            .and_then(|id| state.registry.current_directive(fingerprint, &id).ok())
                    });
                    if let Some(directive) = directive
                        .map(|d| state.sockets.directive(fingerprint, d))
                        .filter(|d| Some(d) != last_directive.as_ref())
                    {
                        Hub::emit(
                            peer,
                            Event::Directive {
                                request_id: uuid::Uuid::now_v7().to_string(),
                                directive: directive.clone(),
                            },
                        );
                        last_directive = Some(directive);
                    }
                }
            }
        }
        let ack_deadline = id.as_ref().and_then(|id| {
            lock(&state.sockets.peers)
                .get(&key(role, id))
                .and_then(|peer| peer.acknowledgments.values().min().copied())
        });
        let pong_deadline = pong.as_ref().map_or(ping_at, |(_, deadline)| *deadline);
        tokio::select! {
            _ = cancel.notified() => break,
            _ = tokio::time::sleep_until(ack_deadline.unwrap_or(auth_deadline)), if ack_deadline.is_some() => break,
            _ = tokio::time::sleep_until(auth_deadline), if id.is_none() => break,
            _ = &mut changed => continue,
            _ = tokio::time::sleep_until(pong_deadline), if !local && id.is_some() => {
                if pong.is_some() { break; }
                let bytes = Bytes::copy_from_slice(uuid::Uuid::now_v7().as_bytes());
                if send.try_send(Message::Ping(bytes.clone())).is_err() { break; }
                pong = Some((bytes, tokio::time::Instant::now() + Duration::from_secs(10)));
            }
            message = reader.next() => {
                let Some(Ok(message)) = message else { break };
                let text = match message {
                    Message::Text(text) => text,
                    Message::Pong(bytes) => {
                        if pong.as_ref().is_some_and(|(expected, _)| *expected == bytes) {
                            pong = None;
                            ping_at = tokio::time::Instant::now() + Duration::from_secs(30);
                        }
                        continue;
                    }
                    Message::Ping(bytes) => { if send.try_send(Message::Pong(bytes)).is_err() { break; } continue; }
                    _ => break,
                };
                let Ok(request) = serde_json::from_str::<ControlRequest>(&text) else { break };
                if request.request_id.is_empty() || request.request_id.len() > 128 { break; }
                if let Some((id, original, reply)) = &last_reply && *id == request.request_id {
                    if original != text.as_str() { break; }
                    let Ok(text) = serde_json::to_string(reply) else { break };
                    if send.try_send(Message::Text(text.into())).is_err() { break; }
                    continue;
                }
                // ACKs affect only the peer's bounded directive queue. Challenge
                // issuance has no routing effect either, so neither needs to
                // wake every established control connection.
                let local_control_command = matches!(
                    &request.command,
                    Command::Acknowledge { .. } | Command::Challenge(_)
                );
                let readiness = if let Command::Ready(payload) = &request.command {
                    Some(socket_ready(&state, role, id.as_deref(), &generation, payload).await)
                } else {
                    None
                };
                let session_id = match &request.command {
                    Command::RegisterMcp { request, .. } => request.proof.transcript.initiator_instance_id.as_str(),
                    Command::RegisterWorker(request) => request.worker_id.as_str(),
                    Command::RecoverWorker(request) => request.worker_id.as_str(),
                    _ => id.as_deref().unwrap_or(&generation),
                };
                let _transaction = state.sockets.transaction(role, session_id).await;
                if let Some(id) = &id && lock(&state.sockets.peers).get(&key(role, id)).is_none_or(|p| p.generation != generation) { break; }
                let response = match readiness {
                    Some(response) => response,
                    None => dispatch(&state, role, &mut id, &mut challenge, request.command).await,
                };
                let success = response.status().is_success();
                if success && !local_control_command && let Some(id) = &id {
                    let mut peers = lock(&state.sockets.peers);
                    let entry = peers.entry(key(role, id));
                    use std::collections::hash_map::Entry;
                    match entry {
                        Entry::Vacant(entry) => { entry.insert(Peer { generation: generation.clone(), sender: Some(send.clone()), cancel: cancel.clone(), disconnected: None, ready: false, acknowledgments: HashMap::new(), last_acknowledged: None, drain: None }); }
                        Entry::Occupied(mut entry) if entry.get().generation != generation => {
                            let peer = entry.get_mut();
                            peer.cancel.notify_one();
                            peer.generation = generation.clone(); peer.sender = Some(send.clone());
                            peer.cancel = cancel.clone(); peer.disconnected = None; peer.ready = false;
                            peer.acknowledgments.clear(); peer.last_acknowledged = None;
                            if let Some((request_id, mut request, deadline)) = peer.drain.clone() {
                                request.timeout_ms = Some(deadline.saturating_duration_since(tokio::time::Instant::now()).as_millis() as u64);
                                Hub::emit(peer, Event::Drain { request_id, request });
                            }
                        }
                        _ => {}
                    }
                    drop(peers);
                    if role == ComponentRole::Mcp {
                        if let Some(session) = lock(&state.mcp_sessions).get_mut(id) {
                            session.lease_expires_at_unix_ms = u64::MAX;
                            let _ = state.registry.renew_mcp(session.fingerprint, &McpSessionId::new(id.clone()).expect("validated session"), u64::MAX);
                        }
                    } else if let Some(session) = lock(&state.worker_sessions).get_mut(id) { session.lease_expires_at_unix_ms = u64::MAX; }
                }
                let status = response.status().as_u16();
                let Ok(bytes) = axum::body::to_bytes(response.into_body(), MAX_CONTROL_BODY_BYTES).await else { break };
                let payload = if bytes.is_empty() { serde_json::Value::Null } else { serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null) };
                let event = Event::Reply { request_id: request.request_id.clone(), status, payload };
                last_reply = Some((request.request_id, text.to_string(), event.clone()));
                let Ok(text) = serde_json::to_string(&event) else { break };
                if send.try_send(Message::Text(text.into())).is_err() { break; }
                if !local_control_command {
                    state.sockets.changed.notify_waiters();
                }
            }
        }
    }
    writing.abort();
    if let Some(challenge) = challenge {
        lock(&state.challenges).remove(&challenge);
    }
    if let Some(id) = id {
        disconnected(state, role, id, generation).await;
    }
}

// Authenticate and snapshot under the transaction lock, but never hold it over network I/O.
// A replacement connection or drain may run during the probe, so fence publication again.
async fn socket_ready(
    state: &Arc<DaemonState>,
    role: ComponentRole,
    id: Option<&str>,
    generation: &str,
    request: &SessionRequest<WorkerReadyPayload>,
) -> Response<Body> {
    let candidate = {
        let _transaction = state.sockets.transaction(role, &request.session_id).await;
        if role != ComponentRole::Worker || id != Some(request.session_id.as_str()) {
            return control_message(StatusCode::UNAUTHORIZED, "worker connection required");
        }
        if let Err(response) = validate_ready_connection(state, &request.session_id, generation) {
            return response;
        }
        match prepare_ready_worker(state, request) {
            Ok(candidate) => candidate,
            Err(response) => return response,
        }
    };
    let probe = probe_worker(&candidate.target).await;
    let _transaction = state.sockets.transaction(role, &request.session_id).await;
    if let Err(response) = validate_ready_connection(state, &request.session_id, generation) {
        return response;
    }
    let response = finish_ready_worker(state.clone(), candidate, probe).await;
    if response.status().is_success() {
        if let Some(peer) = lock(&state.sockets.peers).get_mut(&key(role, &request.session_id)) {
            peer.ready = true;
        }
        if let Some(session) = lock(&state.worker_sessions).get(&request.session_id) {
            session.pending_target.set_control_available(true);
        }
    }
    response
}

#[allow(clippy::result_large_err)]
fn validate_ready_connection(
    state: &DaemonState,
    id: &str,
    generation: &str,
) -> Result<(), Response<Body>> {
    let peers = lock(&state.sockets.peers);
    let Some(peer) = peers
        .get(&key(ComponentRole::Worker, id))
        .filter(|peer| peer.generation == generation && peer.sender.is_some())
    else {
        return Err(control_message(
            StatusCode::UNAUTHORIZED,
            "worker connection was replaced",
        ));
    };
    if peer.drain.is_some() {
        return Err(control_message(StatusCode::CONFLICT, "worker is draining"));
    }
    Ok(())
}

#[allow(clippy::cognitive_complexity)] // Keep role and authentication guards beside each command.
async fn dispatch(
    state: &Arc<DaemonState>,
    role: ComponentRole,
    id: &mut Option<String>,
    challenge: &mut Option<ChallengeId>,
    command: Command,
) -> Response<Body> {
    match command {
        Command::Challenge(request)
            if id.is_none() && challenge.is_none() && request.initiator.role == role =>
        {
            // A connection gets one challenge. The upgrade endpoint retains peer rate limiting.
            let response = issue_challenge(State(state.clone()), Json(request)).await;
            let status = response.status();
            let Ok(bytes) =
                axum::body::to_bytes(response.into_body(), MAX_CONTROL_BODY_BYTES).await
            else {
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            };
            if status.is_success()
                && let Ok(response) = serde_json::from_slice::<ChallengeResponse>(&bytes)
            {
                *challenge = Some(response.challenge.id);
            }
            Response::builder()
                .status(status)
                .body(Body::from(bytes))
                .expect("valid response")
        }
        Command::RegisterMcp {
            request,
            credential,
        } if role == ComponentRole::Mcp
            && id.is_none()
            && *challenge == Some(request.proof.transcript.challenge_id) =>
        {
            let session_id = request.proof.transcript.initiator_instance_id.clone();
            if expired(state, role, &session_id) {
                return control_message(StatusCode::UNAUTHORIZED, "session recovery grace expired");
            }
            let mut headers = HeaderMap::new();
            let Ok(value) = HeaderValue::from_str(credential.expose()) else {
                return StatusCode::BAD_REQUEST.into_response();
            };
            headers.insert(CLIENT_TOKEN_HEADER, value);
            let response = register_mcp(State(state.clone()), headers, Json(request)).await;
            if response.status().is_success() {
                *id = Some(session_id.clone());
                if let Some(session) = lock(&state.mcp_sessions).get_mut(&session_id) {
                    session.last_sequence = 0;
                    session.last_request_id.clear();
                }
            }
            response
        }
        Command::RegisterWorker(request)
            if role == ComponentRole::Worker
                && id.is_none()
                && *challenge == Some(request.proof.transcript.challenge_id)
                && request.worker_id == request.proof.transcript.initiator_instance_id =>
        {
            let worker_id = request.worker_id.clone();
            if expired(state, role, &worker_id) {
                return control_message(StatusCode::UNAUTHORIZED, "session recovery grace expired");
            }
            let response = register_worker(State(state.clone()), Json(request)).await;
            if response.status().is_success() {
                *id = Some(worker_id.clone());
                reset_worker(state, &worker_id);
            }
            response
        }
        Command::RecoverWorker(request)
            if role == ComponentRole::Worker
                && id.is_none()
                && *challenge == Some(request.proof.transcript.challenge_id)
                && request.worker_id == request.proof.transcript.initiator_instance_id =>
        {
            let worker_id = request.worker_id.clone();
            if expired(state, role, &worker_id) {
                return control_message(StatusCode::UNAUTHORIZED, "session recovery grace expired");
            }
            let response = recover_worker(State(state.clone()), Json(request)).await;
            if response.status().is_success() {
                *id = Some(worker_id.clone());
                reset_worker(state, &worker_id);
            }
            response
        }
        Command::Release(request)
            if role == ComponentRole::Mcp && id.as_ref() == Some(&request.session_id) =>
        {
            release_mcp(State(state.clone()), Json(request)).await
        }
        Command::ActivationFailed(request)
            if role == ComponentRole::Mcp && id.as_ref() == Some(&request.session_id) =>
        {
            activation_failed(State(state.clone()), Json(request)).await
        }
        Command::Acknowledge { request_id } if id.is_some() => {
            let mut peers = lock(&state.sockets.peers);
            let Some(peer) = peers.get_mut(&key(role, id.as_ref().unwrap())) else {
                return StatusCode::UNAUTHORIZED.into_response();
            };
            if peer.last_acknowledged.as_ref() == Some(&request_id) {
                return StatusCode::NO_CONTENT.into_response();
            }
            if peer
                .acknowledgments
                .remove(&request_id)
                .is_some_and(|deadline| tokio::time::Instant::now() <= deadline)
            {
                peer.last_acknowledged = Some(request_id);
                StatusCode::NO_CONTENT.into_response()
            } else {
                control_message(
                    StatusCode::CONFLICT,
                    "unknown or expired command acknowledgement",
                )
            }
        }
        _ => control_message(
            StatusCode::UNAUTHORIZED,
            "command is not authorized on this control connection",
        ),
    }
}
fn expired(state: &DaemonState, role: ComponentRole, id: &str) -> bool {
    lock(&state.sockets.peers)
        .get(&key(role, id))
        .is_some_and(|p| {
            p.disconnected
                .is_some_and(|at| tokio::time::Instant::now() >= at + GRACE)
        })
}
fn reset_worker(state: &DaemonState, id: &str) {
    if let Some(session) = lock(&state.worker_sessions).get_mut(id) {
        session.last_sequence = 0;
        session.last_request_id.clear();
        session.pending_target.set_control_available(false);
        // A previously published worker still needs a fresh HTTP readiness probe.
    }
}
async fn disconnected(
    state: Arc<DaemonState>,
    role: ComponentRole,
    id: String,
    generation: String,
) {
    let _transaction = state.sockets.transaction(role, &id).await;
    let deadline = {
        let mut peers = lock(&state.sockets.peers);
        let Some(peer) = peers
            .get_mut(&key(role, &id))
            .filter(|p| p.generation == generation)
        else {
            return;
        };
        peer.sender = None;
        peer.ready = false;
        let now = tokio::time::Instant::now();
        *peer.disconnected.get_or_insert(now) + GRACE
    };
    if role == ComponentRole::Worker
        && let Some(session) = lock(&state.worker_sessions).get(&id)
    {
        session.pending_target.set_control_available(false);
    }
    state.sockets.changed.notify_waiters();
    drop(_transaction);
    tokio::spawn(async move {
        tokio::time::sleep_until(deadline).await;
        let _transaction = state.sockets.transaction(role, &id).await;
        let current = lock(&state.sockets.peers)
            .get(&key(role, &id))
            .is_some_and(|p| p.generation == generation && p.sender.is_none());
        if !current {
            return;
        }
        if role == ComponentRole::Mcp {
            let session = lock(&state.mcp_sessions).remove(&id);
            if let Some(session) = session
                && let Ok(id) = McpSessionId::new(id.clone())
                && let Ok(action) = state.registry.release_mcp(
                    session.fingerprint,
                    &id,
                    now_unix_ms().saturating_add(DRAIN_LIFETIME_MS),
                )
            {
                handle_release_action(state.clone(), session.fingerprint, action);
            }
            lock(&state.pending_directives).remove(&id);
        } else {
            let session = lock(&state.worker_sessions).remove(&id);
            if let Some(session) = session {
                let work = state.clone();
                let fingerprint = session.fingerprint;
                let generation_id = session.generation_grant.generation_id;
                let _ = tokio::task::spawn_blocking(move || {
                    revoke_active_worker_generation(&work, fingerprint, &generation_id)
                })
                .await;
                if let Ok(WorkerFailureAction::NominateMcp { session_id }) =
                    state.registry.worker_failed(
                        fingerprint,
                        &id,
                        now_unix_ms().saturating_add(RECOVERY_LIFETIME_MS),
                    )
                {
                    nominate_relaunch(&state, fingerprint, session_id);
                }
            }
        }
        lock(&state.sockets.peers).remove(&key(role, &id));
        state.sockets.changed.notify_waiters();
    });
}

#[derive(Clone)]
pub(super) struct SocketInfo {
    peer: SocketAddr,
    local: SocketAddr,
}
impl axum::extract::connect_info::Connected<axum::serve::IncomingStream<'_, TcpListener>>
    for SocketInfo
{
    fn connect_info(stream: axum::serve::IncomingStream<'_, TcpListener>) -> Self {
        let _ = stream.io().set_nodelay(true);
        Self {
            peer: *stream.remote_addr(),
            local: stream
                .io()
                .local_addr()
                .expect("accepted socket has local address"),
        }
    }
}
pub(super) async fn connection_info(mut request: Request<Body>, next: Next) -> Response<Body> {
    if let Some(ConnectInfo(info)) = request
        .extensions()
        .get::<ConnectInfo<SocketInfo>>()
        .cloned()
    {
        request.extensions_mut().insert(ConnectInfo(info.peer));
        request.extensions_mut().insert(LocalAddress(info.local));
    }
    next.run(request).await
}

#[cfg(test)]
#[path = "../../../../tests/coverage/daemon/socket_tests.rs"]
mod tests;

#[cfg(test)]
pub(super) async fn test_expire_worker(state: Arc<DaemonState>, id: String) {
    lock(&state.sockets.peers).insert(
        key(ComponentRole::Worker, &id),
        Peer {
            generation: "fixture".into(),
            sender: None,
            cancel: Arc::new(Notify::new()),
            disconnected: Some(tokio::time::Instant::now() - GRACE),
            ready: false,
            acknowledgments: HashMap::new(),
            last_acknowledged: None,
            drain: None,
        },
    );
    disconnected(state, ComponentRole::Worker, id, "fixture".into()).await;
}
