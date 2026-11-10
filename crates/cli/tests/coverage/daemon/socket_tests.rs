// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::daemon::broker::lifecycle::RouteStateKind;
use crate::daemon::common::control::WorkerNetworkHintProof;
use crate::daemon::common::{client::begin_handshake, socket::Client, state::RouteCredential};

async fn daemon(pass: bool) -> (Arc<DaemonState>, String, tokio::task::JoinHandle<()>) {
    daemon_with_idle_retention(pass, Duration::ZERO, false).await
}

async fn daemon_with_idle_retention(
    pass: bool,
    idle_timeout: Duration,
    advertise_retention: bool,
) -> (Arc<DaemonState>, String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let mut state = super::super::tests::test_daemon_state_at(
        pass,
        "",
        GatewayConfig::default(),
        origin.clone(),
    );
    let configured = Arc::get_mut(&mut state).unwrap();
    configured.registry = Registry::new(pass).with_worker_idle_timeout(idle_timeout);
    configured.descriptor = crate::daemon::common::control::descriptor_with_idle_retention(
        ComponentRole::Daemon,
        advertise_retention,
    );
    let app = super::router(state.clone())
        .layer(axum::middleware::from_fn(connection_info))
        .into_make_service_with_connect_info::<SocketInfo>();
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (state, origin, task)
}

#[tokio::test]
async fn diagnostic_revocation_emits_acknowledged_worker_event() {
    let hub = Hub::default();
    let (sender, mut receiver) = mpsc::channel(1);
    lock(&hub.peers).insert(
        key(ComponentRole::Worker, "worker"),
        Peer {
            generation: "connection".into(),
            sender: Some(sender),
            cancel: Arc::new(ControlCancellation::new()),
            disconnected: None,
            ready: true,
            acknowledgments: HashMap::new(),
            last_acknowledged: None,
            drain: None,
        },
    );

    hub.revoke_diagnostics("worker", true);

    let Some(Message::Text(text)) = receiver.recv().await else {
        panic!("expected diagnostic revocation message");
    };
    let Event::DiagnosticRevoked { request_id } = serde_json::from_str(&text).unwrap() else {
        panic!("expected diagnostic revocation event");
    };
    assert!(!request_id.is_empty());
    assert!(
        lock(&hub.peers)
            .get(&key(ComponentRole::Worker, "worker"))
            .unwrap()
            .acknowledgments
            .contains_key(&request_id)
    );
}

#[tokio::test]
async fn diagnostic_revocation_skips_workers_without_negotiated_support() {
    let hub = Hub::default();
    let (sender, mut receiver) = mpsc::channel(1);
    lock(&hub.peers).insert(
        key(ComponentRole::Worker, "legacy-worker"),
        Peer {
            generation: "connection".into(),
            sender: Some(sender),
            cancel: Arc::new(ControlCancellation::new()),
            disconnected: None,
            ready: true,
            acknowledgments: HashMap::new(),
            last_acknowledged: None,
            drain: None,
        },
    );

    hub.revoke_diagnostics("legacy-worker", false);

    assert!(receiver.try_recv().is_err());
}

async fn mcp(
    client: &Client,
    origin: &str,
    identity: &MachineIdentity,
    id: &str,
) -> McpRegisterResponse {
    let credential =
        RouteCredential::parse(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([42; 32]))
            .unwrap();
    let handshake = begin_handshake(
        client,
        origin,
        ComponentRole::Mcp,
        identity,
        id,
        Some(credential.digest()),
    )
    .await
    .unwrap();
    let worker_network = WorkerNetworkHintProof::sign(
        WorkerNetworkHint::new("127.0.0.1", None).unwrap(),
        origin,
        id,
        &handshake.proof.transcript.challenge_id,
        &identity.fingerprint(),
        identity,
    )
    .unwrap();
    let response: McpRegisterResponse = client
        .request(Command::RegisterMcp {
            request: McpRegisterRequest {
                proof: handshake.proof.clone(),
                worker_network,
            },
            credential: SensitiveString::new(credential.expose()).unwrap(),
        })
        .await
        .unwrap();
    handshake
        .authenticate_daemon(&response.daemon_proof)
        .unwrap();
    response
}
async fn wait_disconnected(state: &DaemonState, role: ComponentRole, id: &str) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if lock(&state.sockets.peers)
                .get(&key(role, id))
                .is_some_and(|p| p.sender.is_none())
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn pending_owner_disconnect_and_reconnect_preserve_session_identity() {
    use crate::daemon::common::control::{
        ActivationCancellation, CancelActivationPayload, WorkerBootstrap,
    };

    for registered_before_disconnect in [true, false] {
        let (state, origin, task) =
            daemon_with_idle_retention(false, Duration::from_secs(30), true).await;
        let identity = MachineIdentity::generate().unwrap().identity;
        let owner = Client::default();
        let registration = mcp(&owner, &origin, &identity, "retained-owner").await;
        let bootstrap = WorkerBootstrap::from_directive(registration.directive).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker = Client::default();
        let handshake = begin_handshake(
            &worker,
            &origin,
            ComponentRole::Worker,
            &identity,
            "retained-worker",
            None,
        )
        .await
        .unwrap();
        let admission = worker.request(Command::RegisterWorker(WorkerRegisterRequest {
            diagnostics: None,
            proof: handshake.proof,
            worker_id: "retained-worker".into(),
            endpoint: format!("http://{}", listener.local_addr().unwrap()),
            activation_id: bootstrap.activation_id.clone(),
            activation_token: bootstrap.activation_token,
            tls_root_certificate: None,
        }));
        tokio::pin!(admission);
        if registered_before_disconnect {
            let _: WorkerRegisterResponse = admission.as_mut().await.unwrap();
        }
        let old_generation = lock(&state.sockets.peers)[&key(ComponentRole::Mcp, "retained-owner")]
            .generation
            .clone();
        drop(owner);
        tokio::time::timeout(Duration::from_secs(2), async {
            while state
                .registry
                .snapshot(identity.fingerprint())
                .unwrap()
                .reference_count
                != 0
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("pending owner disconnect releases its route reference");
        if !registered_before_disconnect {
            let _: WorkerRegisterResponse = admission.as_mut().await.unwrap();
        }
        {
            let sessions = lock(&state.mcp_sessions);
            let session = &sessions["retained-owner"];
            assert!(!session.released);
        }
        assert!(lock(&state.activations).contains_key(&bootstrap.activation_id));
        assert!(lock(&state.worker_sessions).contains_key("retained-worker"));
        let reconnected = Client::default();
        let resumed = mcp(&reconnected, &origin, &identity, "retained-owner").await;
        assert_eq!(resumed.session_token, registration.session_token);
        assert!(matches!(
            resumed.directive,
            BrokerDirective::WaitForWorker { .. }
        ));
        disconnected(
            state.clone(),
            ComponentRole::Mcp,
            "retained-owner".into(),
            old_generation,
            "stale_disconnect",
        )
        .await;
        assert_eq!(
            state
                .registry
                .snapshot(identity.fingerprint())
                .unwrap()
                .reference_count,
            1
        );
        let cancellation: ActivationCancellation = reconnected
            .request(Command::CancelActivation(
                SessionRequest::new(
                    "retained-owner".into(),
                    resumed.session_token,
                    1,
                    CancelActivationPayload {
                        activation_id: bootstrap.activation_id.clone(),
                    },
                )
                .unwrap(),
            ))
            .await
            .unwrap();
        assert_eq!(cancellation, ActivationCancellation::Cancelled);
        assert!(!lock(&state.activations).contains_key(&bootstrap.activation_id));
        assert!(!lock(&state.worker_sessions).contains_key("retained-worker"));
        task.abort();
    }
}

#[tokio::test]
async fn owner_disconnect_without_negotiated_retention_still_cancels_startup() {
    use crate::daemon::common::control::WorkerBootstrap;

    for timeout in [Duration::ZERO, Duration::from_secs(30)] {
        let (state, origin, task) = daemon_with_idle_retention(false, timeout, false).await;
        let identity = MachineIdentity::generate().unwrap().identity;
        let owner = Client::default();
        let registration = mcp(&owner, &origin, &identity, "legacy-owner").await;
        let bootstrap = WorkerBootstrap::from_directive(registration.directive).unwrap();
        assert!(!lock(&state.mcp_sessions)["legacy-owner"].idle_retention);
        drop(owner);
        tokio::time::timeout(Duration::from_secs(2), async {
            while lock(&state.activations).contains_key(&bootstrap.activation_id) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("unsupported owner disconnect cancels startup");
        let snapshot = state.registry.snapshot(identity.fingerprint()).unwrap();
        assert_eq!(snapshot.reference_count, 1);
        assert_eq!(snapshot.state, RouteStateKind::PassThrough);
        task.abort();
    }
}

#[tokio::test]
async fn loopback_idle_session_survives_old_lease_deadlines_and_releases_explicitly() {
    let (state, origin, task) = daemon(true).await;
    let identity = MachineIdentity::generate().unwrap().identity;
    let client = Client::default();
    let registration = mcp(&client, &origin, &identity, "idle").await;
    let Event::Directive {
        request_id,
        directive,
    } = client.next().await.unwrap()
    else {
        panic!("expected directive")
    };
    assert_eq!(directive, BrokerDirective::UsePassThrough);
    client.acknowledge(request_id).await.unwrap();
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(120)).await;
    assert!(
        lock(&state.sockets.peers)[&key(ComponentRole::Mcp, "idle")]
            .sender
            .is_some()
    );
    assert_eq!(
        lock(&state.mcp_sessions)["idle"].lease_expires_at_unix_ms,
        u64::MAX
    );
    tokio::time::resume();
    client
        .request::<()>(Command::Release(
            SessionRequest::new(
                "idle".into(),
                registration.session_token,
                1,
                crate::daemon::common::control::ReleasePayload::default(),
            )
            .unwrap(),
        ))
        .await
        .unwrap();
    assert_eq!(
        state
            .registry
            .snapshot(identity.fingerprint())
            .unwrap()
            .reference_count,
        0
    );
    task.abort();
}
#[tokio::test]
async fn replacement_connection_fences_old_callbacks_and_grace_expires_once() {
    let (state, origin, task) = daemon(true).await;
    let identity = MachineIdentity::generate().unwrap().identity;
    let first = Client::default();
    mcp(&first, &origin, &identity, "reconnect").await;
    let old_generation = lock(&state.sockets.peers)[&key(ComponentRole::Mcp, "reconnect")]
        .generation
        .clone();
    let second = Client::default();
    mcp(&second, &origin, &identity, "reconnect").await;
    disconnected(
        state.clone(),
        ComponentRole::Mcp,
        "reconnect".into(),
        old_generation,
        "test_replacement",
    )
    .await;
    assert!(
        lock(&state.sockets.peers)[&key(ComponentRole::Mcp, "reconnect")]
            .sender
            .is_some()
    );
    assert_eq!(
        state
            .registry
            .snapshot(identity.fingerprint())
            .unwrap()
            .reference_count,
        1
    );
    drop(first);
    drop(second);
    wait_disconnected(&state, ComponentRole::Mcp, "reconnect").await;
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(29)).await;
    assert_eq!(
        state
            .registry
            .snapshot(identity.fingerprint())
            .unwrap()
            .reference_count,
        1
    );
    tokio::time::advance(Duration::from_secs(2)).await;
    tokio::time::resume();
    tokio::time::timeout(Duration::from_secs(2), async {
        while state
            .registry
            .snapshot(identity.fingerprint())
            .unwrap()
            .reference_count
            != 0
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("disconnected session expires");
    task.abort();
}
#[tokio::test]
async fn worker_ready_is_pushed_recovery_reprobes_and_drain_survives_disconnect() {
    let (state, origin, task) = daemon(false).await;
    let identity = MachineIdentity::generate().unwrap().identity;
    let client = Client::default();
    let mcp_registration = mcp(&client, &origin, &identity, "owner").await;
    let crate::daemon::common::control::WorkerBootstrap {
        activation_id,
        activation_token,
        ..
    } = crate::daemon::common::control::WorkerBootstrap::from_directive(mcp_registration.directive)
        .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let worker = Client::default();
    let handshake = begin_handshake(
        &worker,
        &origin,
        ComponentRole::Worker,
        &identity,
        "worker",
        None,
    )
    .await
    .unwrap();
    let diagnostic_offer = Some(crate::daemon::diagnostics::DiagnosticOffer {
        source: uuid::Uuid::now_v7(),
        schema_version: nemo_relay::diagnostics::DIAGNOSTIC_SCHEMA_VERSION,
        enabled: false,
        query_available: false,
        whole_context_read: false,
        max_page_bytes: 0,
    });
    let mut register = WorkerRegisterRequest {
        diagnostics: diagnostic_offer.clone(),
        proof: handshake.proof,
        worker_id: "worker".into(),
        endpoint: endpoint.clone(),
        activation_id: activation_id.clone(),
        activation_token,
        tls_root_certificate: None,
    };
    let registration: WorkerRegisterResponse = worker
        .request(Command::RegisterWorker(register.clone()))
        .await
        .unwrap();
    let data = registration.data_token.clone();
    let probe_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count = probe_count.clone();
    let probe_started = Arc::new(Notify::new());
    let probe_release = Arc::new(Notify::new());
    let started = probe_started.clone();
    let release = probe_release.clone();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().route(
                WORKER_PROBE_PATH,
                axum::routing::get(move |headers: HeaderMap| {
                    let data = data.clone();
                    let count = count.clone();
                    let started = started.clone();
                    let release = release.clone();
                    async move {
                        assert_eq!(headers[WORKER_TOKEN_HEADER], data.expose());
                        if count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
                            started.notify_one();
                            release.notified().await;
                        }
                        StatusCode::NO_CONTENT
                    }
                }),
            ),
        )
        .await
        .unwrap();
    });
    let ready = SessionRequest::new(
        "worker".into(),
        registration.session_token.clone(),
        1,
        WorkerReadyPayload {
            worker_id: "worker".into(),
        },
    )
    .unwrap();
    let first_worker = worker.clone();
    let first_ready = ready.clone();
    let first_probe = tokio::spawn(async move {
        first_worker
            .request::<()>(Command::Ready(first_ready))
            .await
    });
    tokio::time::timeout(Duration::from_secs(2), probe_started.notified())
        .await
        .unwrap();
    assert!(!lock(&state.worker_sessions)["worker"].published);

    // Retry the identical command on a replacement connection while the original probe waits.
    // Only the replacement may publish, even when the stale probe later succeeds.
    let replacement = Client::default();
    register.proof = begin_handshake(
        &replacement,
        &origin,
        ComponentRole::Worker,
        &identity,
        "worker",
        None,
    )
    .await
    .unwrap()
    .proof;
    let replay: WorkerRegisterResponse = replacement
        .request(Command::RegisterWorker(register))
        .await
        .unwrap();
    assert_eq!(
        replay.session_token.expose(),
        registration.session_token.expose()
    );
    replacement
        .request::<()>(Command::Ready(ready.clone()))
        .await
        .unwrap();
    probe_release.notify_one();
    assert!(first_probe.await.unwrap().is_err());
    assert!(lock(&state.worker_sessions)["worker"].published);
    let cancellation: crate::daemon::common::control::ActivationCancellation = client
        .request(Command::CancelActivation(
            SessionRequest::new(
                "owner".into(),
                mcp_registration.session_token.clone(),
                1,
                crate::daemon::common::control::CancelActivationPayload { activation_id },
            )
            .unwrap(),
        ))
        .await
        .unwrap();
    assert_eq!(
        cancellation,
        crate::daemon::common::control::ActivationCancellation::Published
    );
    assert!(
        lock(&state.worker_sessions)["worker"]
            .pending_target
            .control_available()
    );
    let worker = replacement;
    worker.request::<()>(Command::Ready(ready)).await.unwrap();
    loop {
        if matches!(
            client.next().await.unwrap(),
            Event::Directive {
                directive: BrokerDirective::ReuseWorker { .. },
                ..
            }
        ) {
            break;
        }
    }
    let target = lock(&state.worker_sessions)["worker"]
        .pending_target
        .clone();
    drop(worker);
    wait_disconnected(&state, ComponentRole::Worker, "worker").await;
    assert!(!target.control_available());
    assert!(matches!(
        state
            .registry
            .current_directive(identity.fingerprint(), &McpSessionId::new("owner").unwrap())
            .unwrap(),
        BrokerDirective::WaitForWorker { .. }
    ));
    let restored = Client::default();
    let handshake = begin_handshake(
        &restored,
        &origin,
        ComponentRole::Worker,
        &identity,
        "worker",
        None,
    )
    .await
    .unwrap();
    let recovered: WorkerRegisterResponse = restored
        .request(Command::RecoverWorker(WorkerRecoverRequest {
            diagnostics: diagnostic_offer,
            proof: handshake.proof,
            worker_id: "worker".into(),
            endpoint: endpoint.clone(),
            tls_root_certificate: None,
            generation_grant: registration.generation_grant.clone(),
        }))
        .await
        .unwrap();
    assert!(!target.control_available());
    restored
        .request::<()>(Command::Ready(
            SessionRequest::new(
                "worker".into(),
                recovered.session_token,
                1,
                WorkerReadyPayload {
                    worker_id: "worker".into(),
                },
            )
            .unwrap(),
        ))
        .await
        .unwrap();
    assert!(target.control_available());
    assert!(probe_count.load(std::sync::atomic::Ordering::Relaxed) >= 2);
    client
        .request::<()>(Command::Release(
            SessionRequest::new(
                "owner".into(),
                mcp_registration.session_token,
                2,
                crate::daemon::common::control::ReleasePayload::default(),
            )
            .unwrap(),
        ))
        .await
        .unwrap();
    let event = restored.next().await.unwrap();
    let Event::DiagnosticRevoked { request_id } = event else {
        panic!("expected diagnostic revocation before drain, got {event:?}");
    };
    restored.acknowledge(request_id).await.unwrap();
    let event = restored.next().await.unwrap();
    assert!(matches!(event, Event::Drain { .. }));
    if let Event::Drain { request_id, .. } = event {
        restored.acknowledge(request_id).await.unwrap();
    }
    assert!(
        restored
            .request::<()>(Command::Ready(
                SessionRequest::new(
                    "worker".into(),
                    registration.session_token,
                    2,
                    WorkerReadyPayload {
                        worker_id: "worker".into()
                    }
                )
                .unwrap()
            ))
            .await
            .is_err()
    );
    drop(restored);
    wait_disconnected(&state, ComponentRole::Worker, "worker").await;
    let draining = Client::default();
    let handshake = begin_handshake(
        &draining,
        &origin,
        ComponentRole::Worker,
        &identity,
        "worker",
        None,
    )
    .await
    .unwrap();
    let _: WorkerRegisterResponse = draining
        .request(Command::RecoverWorker(WorkerRecoverRequest {
            diagnostics: None,
            proof: handshake.proof,
            worker_id: "worker".into(),
            endpoint,
            tls_root_certificate: None,
            generation_grant: registration.generation_grant,
        }))
        .await
        .unwrap();
    assert!(matches!(
        draining.pending_event().await,
        Some(Event::Drain { .. })
    ));
    assert!(!target.control_available());
    task.abort();
    server.abort();
}
#[tokio::test]
async fn unauthenticated_and_legacy_control_requests_are_rejected() {
    let (_state, origin, task) = daemon(true).await;
    let client = Client::default();
    client.connect(&origin, ComponentRole::Mcp).await.unwrap();
    assert!(matches!(
        client
            .request::<()>(Command::Acknowledge {
                request_id: "unknown".into()
            })
            .await,
        Err(CliError::Unauthorized(_))
    ));
    let response = reqwest::Client::new()
        .post(format!("{origin}/_nemo-relay/control/v1/mcp/heartbeat"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    task.abort();
}

#[tokio::test]
async fn activation_cancellation_before_socket_publication_rejects_admission() {
    use crate::daemon::common::control::{
        ActivationCancellation, CancelActivationPayload, WorkerBootstrap,
    };

    struct AdmissionRelease(Arc<Notify>);
    impl Drop for AdmissionRelease {
        fn drop(&mut self) {
            self.0.notify_one();
        }
    }

    let (state, origin, daemon_task) = daemon(false).await;
    let identity = MachineIdentity::generate().unwrap().identity;
    let owner = Client::default();
    let registration = mcp(&owner, &origin, &identity, "admission-owner").await;
    let bootstrap = WorkerBootstrap::from_directive(registration.directive).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let worker = Client::default();
    let handshake = begin_handshake(
        &worker,
        &origin,
        ComponentRole::Worker,
        &identity,
        "admitting-worker",
        None,
    )
    .await
    .unwrap();
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let _release_on_failure = AdmissionRelease(Arc::clone(&release));
    *lock(&state.sockets.admission_gate) = Some(WorkerAdmissionGate {
        worker_id: "admitting-worker".into(),
        entered: Arc::clone(&entered),
        release: Arc::clone(&release),
    });
    let activation_id = bootstrap.activation_id.clone();
    let pending = tokio::spawn(async move {
        worker
            .request::<WorkerRegisterResponse>(Command::RegisterWorker(WorkerRegisterRequest {
                diagnostics: None,
                proof: handshake.proof,
                worker_id: "admitting-worker".into(),
                endpoint,
                activation_id: bootstrap.activation_id,
                activation_token: bootstrap.activation_token,
                tls_root_certificate: None,
            }))
            .await
    });
    tokio::time::timeout(Duration::from_secs(2), entered.notified())
        .await
        .expect("worker admission must reach the publication gate");
    assert!(lock(&state.worker_sessions).contains_key("admitting-worker"));
    assert!(
        !lock(&state.sockets.peers).contains_key(&key(ComponentRole::Worker, "admitting-worker"))
    );
    let cancellation: ActivationCancellation = owner
        .request(Command::CancelActivation(
            SessionRequest::new(
                "admission-owner".into(),
                registration.session_token,
                1,
                CancelActivationPayload { activation_id },
            )
            .unwrap(),
        ))
        .await
        .unwrap();
    release.notify_one();
    let result = tokio::time::timeout(Duration::from_secs(2), pending)
        .await
        .expect("revoked admission must terminate")
        .unwrap();
    assert_eq!(cancellation, ActivationCancellation::Cancelled);
    let error = result.expect_err("revoked admission must not return a grant");
    // With no admitted peer, the next socket-loop iteration aborts its writer.
    // That can close the connection before its queued rejection is flushed.
    match error {
        CliError::Unauthorized(message) => assert_eq!(
            message,
            "worker admission was revoked before connection publication"
        ),
        CliError::Launch(message) => assert_eq!(message, "control connection lost"),
        unexpected => panic!("unexpected admission failure: {unexpected}"),
    }
    assert!(lock(&state.worker_sessions).is_empty());
    assert!(
        !lock(&state.sockets.peers).contains_key(&key(ComponentRole::Worker, "admitting-worker"))
    );
    assert_eq!(
        state
            .registry
            .snapshot(identity.fingerprint())
            .unwrap()
            .state,
        RouteStateKind::PassThrough
    );
    daemon_task.abort();
}

#[tokio::test(flavor = "current_thread")]
async fn blocked_worker_admission_publication_does_not_stall_health() {
    use crate::daemon::common::control::WorkerBootstrap;

    let (state, origin, daemon_task) = daemon(false).await;
    let identity = MachineIdentity::generate().unwrap().identity;
    let owner = Client::default();
    let registration = mcp(&owner, &origin, &identity, "publication-owner").await;
    let bootstrap = WorkerBootstrap::from_directive(registration.directive).unwrap();
    let worker = Client::default();
    let handshake = begin_handshake(
        &worker,
        &origin,
        ComponentRole::Worker,
        &identity,
        "publication-worker",
        None,
    )
    .await
    .unwrap();
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    *lock(&state.sockets.admission_gate) = Some(WorkerAdmissionGate {
        worker_id: "publication-worker".into(),
        entered: Arc::clone(&entered),
        release: Arc::clone(&release),
    });
    let registration = tokio::spawn(async move {
        worker
            .request::<WorkerRegisterResponse>(Command::RegisterWorker(WorkerRegisterRequest {
                diagnostics: None,
                proof: handshake.proof,
                worker_id: "publication-worker".into(),
                endpoint: "http://127.0.0.1:1".into(),
                activation_id: bootstrap.activation_id,
                activation_token: bootstrap.activation_token,
                tls_root_certificate: None,
            }))
            .await
    });
    tokio::time::timeout(Duration::from_secs(2), entered.notified())
        .await
        .expect("worker admission must reach the publication gate");

    let (locked, acquired) = tokio::sync::oneshot::channel();
    let (unlock, unlocked) = std::sync::mpsc::channel();
    let holder = std::thread::spawn(move || {
        let _publication = lock(&state.worker_generation_publication);
        locked.send(()).unwrap();
        let _ = unlocked.recv_timeout(Duration::from_secs(5));
    });
    acquired.await.unwrap();
    let start = std::time::Instant::now();
    release.notify_one();
    tokio::task::yield_now().await;
    let response = tokio::time::timeout(
        Duration::from_secs(1),
        reqwest::Client::new()
            .get(format!("{origin}/healthz"))
            .send(),
    )
    .await;
    let elapsed = start.elapsed();
    let still_waiting = !registration.is_finished();
    unlock.send(()).ok();
    holder.join().unwrap();
    let result = registration.await.unwrap();
    daemon_task.abort();
    assert!(
        elapsed < Duration::from_secs(2),
        "admission publication parked the runtime for {elapsed:?}"
    );
    assert_eq!(response.unwrap().unwrap().status(), StatusCode::OK);
    assert!(still_waiting, "worker admission must wait for publication");
    result.unwrap();
}

#[tokio::test]
async fn activation_cancellation_fences_an_in_progress_readiness_probe() {
    use crate::daemon::common::control::{
        ActivationCancellation, CancelActivationPayload, WorkerBootstrap,
    };

    let (state, origin, daemon_task) = daemon(false).await;
    let identity = MachineIdentity::generate().unwrap().identity;
    let owner = Client::default();
    let registration = mcp(&owner, &origin, &identity, "cancelling-owner").await;
    let bootstrap = WorkerBootstrap::from_directive(registration.directive).unwrap();
    let other = Client::default();
    let other_registration = mcp(&other, &origin, &identity, "other-mcp").await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let worker = Client::default();
    let handshake = begin_handshake(
        &worker,
        &origin,
        ComponentRole::Worker,
        &identity,
        "cancelled-worker",
        None,
    )
    .await
    .unwrap();
    let worker_registration: WorkerRegisterResponse = worker
        .request(Command::RegisterWorker(WorkerRegisterRequest {
            diagnostics: None,
            proof: handshake.proof,
            worker_id: "cancelled-worker".into(),
            endpoint,
            activation_id: bootstrap.activation_id.clone(),
            activation_token: bootstrap.activation_token,
            tls_root_certificate: None,
        }))
        .await
        .unwrap();
    let probe_started = Arc::new(Notify::new());
    let probe_release = Arc::new(Notify::new());
    let started = probe_started.clone();
    let release = probe_release.clone();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().route(
                WORKER_PROBE_PATH,
                axum::routing::get(move || {
                    let started = started.clone();
                    let release = release.clone();
                    async move {
                        started.notify_one();
                        release.notified().await;
                        StatusCode::NO_CONTENT
                    }
                }),
            ),
        )
        .await
        .unwrap();
    });
    let ready_worker = worker.clone();
    let ready = tokio::spawn(async move {
        ready_worker
            .request::<()>(Command::Ready(
                SessionRequest::new(
                    "cancelled-worker".into(),
                    worker_registration.session_token,
                    1,
                    WorkerReadyPayload {
                        worker_id: "cancelled-worker".into(),
                    },
                )
                .unwrap(),
            ))
            .await
    });
    tokio::time::timeout(Duration::from_secs(2), probe_started.notified())
        .await
        .unwrap();
    let payload = CancelActivationPayload {
        activation_id: bootstrap.activation_id.clone(),
    };
    assert!(
        other
            .request::<ActivationCancellation>(Command::CancelActivation(
                SessionRequest::new(
                    "other-mcp".into(),
                    other_registration.session_token,
                    1,
                    payload.clone(),
                )
                .unwrap()
            ))
            .await
            .is_err()
    );
    assert!(lock(&state.activations).contains_key(&bootstrap.activation_id));
    let cancellation: ActivationCancellation = owner
        .request(Command::CancelActivation(
            SessionRequest::new(
                "cancelling-owner".into(),
                registration.session_token.clone(),
                1,
                payload.clone(),
            )
            .unwrap(),
        ))
        .await
        .unwrap();
    assert_eq!(cancellation, ActivationCancellation::Cancelled);
    assert!(!lock(&state.activations).contains_key(&bootstrap.activation_id));
    assert!(lock(&state.worker_sessions).is_empty());
    probe_release.notify_one();
    assert!(
        tokio::time::timeout(Duration::from_secs(2), ready)
            .await
            .unwrap()
            .unwrap()
            .is_err()
    );
    assert_ne!(
        state
            .registry
            .snapshot(identity.fingerprint())
            .unwrap()
            .state,
        RouteStateKind::Ready
    );
    daemon_task.abort();
    server.abort();
}

#[tokio::test]
async fn slow_consumer_queue_is_bounded_and_cancels_connection() {
    let (sender, _receiver) = mpsc::channel(QUEUE_CAPACITY);
    let cancel = Arc::new(ControlCancellation::new());
    let mut peer = Peer {
        generation: "test".into(),
        sender: Some(sender),
        cancel: cancel.clone(),
        disconnected: None,
        ready: false,
        acknowledgments: HashMap::new(),
        last_acknowledged: None,
        drain: None,
    };
    for _ in 0..=QUEUE_CAPACITY {
        Hub::emit(
            &mut peer,
            Event::Directive {
                request_id: "event".into(),
                directive: BrokerDirective::UsePassThrough,
            },
        );
    }
    let reason = tokio::time::timeout(Duration::from_millis(100), cancel.cancelled())
        .await
        .unwrap();
    assert_eq!(reason, "outbound_queue_full");
}

type RawSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
async fn raw_request(socket: &mut RawSocket, command: Command) -> serde_json::Value {
    use tokio_tungstenite::tungstenite::Message as Wire;
    let request = ControlRequest {
        request_id: uuid::Uuid::now_v7().to_string(),
        command,
    };
    socket
        .send(Wire::Text(serde_json::to_string(&request).unwrap().into()))
        .await
        .unwrap();
    loop {
        let Wire::Text(text) = socket.next().await.unwrap().unwrap() else {
            continue;
        };
        if let Event::Reply {
            request_id,
            status,
            payload,
        } = serde_json::from_str(&text).unwrap()
            && request_id == request.request_id
        {
            assert!((200..300).contains(&status));
            return payload;
        }
    }
}
async fn raw_mcp(origin: &str) -> RawSocket {
    use crate::daemon::common::control::{descriptor, fresh_nonce};
    use crate::daemon::common::protocol::HandshakeTranscript;
    let (mut socket, _) = tokio_tungstenite::connect_async(format!(
        "{}{path}",
        origin.replacen("http", "ws", 1),
        path = crate::daemon::common::socket::MCP_SOCKET_PATH
    ))
    .await
    .unwrap();
    let identity = MachineIdentity::generate().unwrap().identity;
    let credential =
        RouteCredential::parse(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([42; 32]))
            .unwrap();
    let request = ChallengeRequest {
        initiator: descriptor(ComponentRole::Mcp),
        initiator_instance_id: "raw".into(),
        initiator_public_identity: identity.public_identity(),
        initiator_fingerprint: identity.fingerprint(),
        initiator_nonce: fresh_nonce().unwrap(),
    };
    let challenge: ChallengeResponse =
        serde_json::from_value(raw_request(&mut socket, Command::Challenge(request.clone())).await)
            .unwrap();
    let transcript = HandshakeTranscript {
        daemon_target: origin.into(),
        initiator: request.initiator,
        responder: challenge.daemon,
        initiator_instance_id: "raw".into(),
        responder_instance_id: challenge.daemon_instance_id,
        selected_protocol: crate::daemon::common::protocol::PROTOCOL_V1,
        initiator_public_identity: identity.public_identity(),
        responder_public_identity: challenge.daemon_public_identity,
        initiator_fingerprint: identity.fingerprint(),
        responder_fingerprint: challenge.daemon_fingerprint,
        challenge_id: challenge.challenge.id,
        initiator_nonce: request.initiator_nonce,
        responder_nonce: challenge.challenge.nonce,
        route_token_digest: Some(credential.digest()),
    };
    let proof = crate::daemon::common::control::RegistrationProof {
        initiator_proof: transcript.sign(ComponentRole::Mcp, &identity).unwrap(),
        transcript,
    };
    let worker_network = WorkerNetworkHintProof::sign(
        WorkerNetworkHint::new("127.0.0.1", None).unwrap(),
        origin,
        "raw",
        &proof.transcript.challenge_id,
        &identity.fingerprint(),
        &identity,
    )
    .unwrap();
    raw_request(
        &mut socket,
        Command::RegisterMcp {
            request: McpRegisterRequest {
                proof,
                worker_network,
            },
            credential: SensitiveString::new(credential.expose()).unwrap(),
        },
    )
    .await;
    // Drain the initial pushed directive before checking idle traffic.
    let tokio_tungstenite::tungstenite::Message::Text(text) = socket.next().await.unwrap().unwrap()
    else {
        panic!("expected directive")
    };
    let Event::Directive { request_id, .. } = serde_json::from_str(&text).unwrap() else {
        panic!("expected directive")
    };
    raw_request(&mut socket, Command::Acknowledge { request_id }).await;
    socket
}
#[tokio::test]
async fn loopback_socket_sends_no_periodic_frames() {
    let (state, origin, task) = daemon(true).await;
    let mut socket = raw_mcp(&origin).await;
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(120)).await;
    assert!(
        tokio::time::timeout(Duration::from_millis(1), socket.next())
            .await
            .is_err()
    );
    assert!(
        lock(&state.sockets.peers)[&key(ComponentRole::Mcp, "raw")]
            .sender
            .is_some()
    );
    task.abort();
}
#[tokio::test]
async fn remote_keepalive_requires_the_matching_pong() {
    use tokio_tungstenite::tungstenite::Message as Wire;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let state = super::super::tests::test_daemon_state_at(
        true,
        "",
        GatewayConfig::default(),
        origin.clone(),
    );
    let work = state.clone();
    let app = Router::new().route(crate::daemon::common::socket::MCP_SOCKET_PATH, axum::routing::get(move |ws: WebSocketUpgrade| {
        let state = work.clone();
        async move { ws.on_upgrade(move |socket| run(state, ComponentRole::Mcp, false, socket)) }
    }));
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let mut socket = raw_mcp(&origin).await;
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(30)).await;
    tokio::time::resume();
    // Receiving a ping queues Tungstenite's automatic pong, but does not flush it. Drop
    // the read future here and advance time without another socket operation.
    assert!(matches!(
        socket.next().await.unwrap().unwrap(),
        Wire::Ping(_)
    ));
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(11)).await;
    tokio::time::resume();
    wait_disconnected(&state, ComponentRole::Mcp, "raw").await;
    task.abort();
}
#[tokio::test]
async fn malformed_and_oversized_frames_close_the_socket_without_registering() {
    use tokio_tungstenite::tungstenite::Message as Wire;
    let (state, origin, task) = daemon(true).await;
    for text in [
        "not json".to_owned(),
        "x".repeat(MAX_CONTROL_BODY_BYTES + 1),
    ] {
        let (mut socket, _) = tokio_tungstenite::connect_async(format!(
            "{}{path}",
            origin.replacen("http", "ws", 1),
            path = crate::daemon::common::socket::MCP_SOCKET_PATH
        ))
        .await
        .unwrap();
        let _ = socket.send(Wire::Text(text.into())).await;
        let closed = tokio::time::timeout(Duration::from_secs(2), socket.next())
            .await
            .unwrap();
        assert!(!matches!(closed, Some(Ok(Wire::Text(_)))));
    }
    assert!(lock(&state.mcp_sessions).is_empty());
    task.abort();
}

#[tokio::test]
async fn restart_defers_replacement_until_the_generation_recovery_deadline() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let identity = MachineIdentity::generate().unwrap().identity;
    let mut state = super::super::tests::test_daemon_state_at(
        false,
        "",
        GatewayConfig::default(),
        origin.clone(),
    );
    state
        .active_worker_generations
        .publish(identity.fingerprint(), "prior-worker-generation", None)
        .unwrap();
    Arc::get_mut(&mut state).unwrap().sockets = Hub::restarting(HashMap::from([(
        identity.fingerprint(),
        "prior-worker-generation".into(),
    )]));
    recover_after_restart(state.clone());
    let app = super::router(state.clone())
        .layer(axum::middleware::from_fn(connection_info))
        .into_make_service_with_connect_info::<SocketInfo>();
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = Client::default();
    let registration = mcp(&client, &origin, &identity, "restart").await;
    assert!(matches!(
        registration.directive,
        BrokerDirective::WaitForWorker { .. }
    ));
    let Event::Directive { request_id, .. } = client.next().await.unwrap() else {
        panic!("expected directive")
    };
    client.acknowledge(request_id).await.unwrap();
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(29)).await;
    assert!(state.sockets.deferred(identity.fingerprint()));
    tokio::time::advance(Duration::from_secs(2)).await;
    tokio::time::resume();
    let event = tokio::time::timeout(Duration::from_secs(5), client.next())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        event,
        Event::Directive {
            directive: BrokerDirective::LaunchWorker { .. },
            ..
        }
    ));
    assert!(
        !state
            .active_worker_generations
            .matches(identity.fingerprint(), "prior-worker-generation")
            .unwrap()
    );
    task.abort();
}

struct RecoveryFixture {
    state: Arc<DaemonState>,
    origin: String,
    identity: MachineIdentity,
    worker: Client,
    registration: WorkerRegisterResponse,
    target: Arc<WorkerTarget>,
    fail_probe: Arc<std::sync::atomic::AtomicBool>,
    block_probe: Arc<std::sync::atomic::AtomicBool>,
    probe_started: Arc<Notify>,
    probe_release: Arc<Notify>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}
impl Drop for RecoveryFixture {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}
impl RecoveryFixture {
    fn ready(&self, sequence: u64) -> Command {
        Command::Ready(
            SessionRequest::new(
                "worker".into(),
                self.registration.session_token.clone(),
                sequence,
                WorkerReadyPayload {
                    worker_id: "worker".into(),
                },
            )
            .unwrap(),
        )
    }
}
async fn recovering_worker() -> RecoveryFixture {
    let (state, origin, task) = daemon(false).await;
    let identity = MachineIdentity::generate().unwrap().identity;
    let client = Client::default();
    let mcp_registration = mcp(&client, &origin, &identity, "owner").await;
    let crate::daemon::common::control::WorkerBootstrap {
        activation_id,
        activation_token,
        ..
    } = crate::daemon::common::control::WorkerBootstrap::from_directive(mcp_registration.directive)
        .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let worker = Client::default();
    let handshake = begin_handshake(
        &worker,
        &origin,
        ComponentRole::Worker,
        &identity,
        "worker",
        None,
    )
    .await
    .unwrap();
    let registration: WorkerRegisterResponse = worker
        .request(Command::RegisterWorker(WorkerRegisterRequest {
            diagnostics: None,
            proof: handshake.proof,
            worker_id: "worker".into(),
            endpoint: endpoint.clone(),
            activation_id,
            activation_token,
            tls_root_certificate: None,
        }))
        .await
        .unwrap();
    let data = registration.data_token.clone();
    let probe_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count = probe_count.clone();
    let fail_probe = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let block_probe = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let probe_started = Arc::new(Notify::new());
    let probe_release = Arc::new(Notify::new());
    let fail = fail_probe.clone();
    let block = block_probe.clone();
    let started = probe_started.clone();
    let release = probe_release.clone();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().route(
                WORKER_PROBE_PATH,
                axum::routing::get(move |headers: HeaderMap| {
                    let data = data.clone();
                    let count = count.clone();
                    let fail = fail.clone();
                    let block = block.clone();
                    let started = started.clone();
                    let release = release.clone();
                    async move {
                        assert_eq!(headers[WORKER_TOKEN_HEADER], data.expose());
                        count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if block.load(std::sync::atomic::Ordering::Relaxed) {
                            started.notify_one();
                            release.notified().await;
                        }
                        if fail.load(std::sync::atomic::Ordering::Relaxed) {
                            StatusCode::SERVICE_UNAVAILABLE
                        } else {
                            StatusCode::NO_CONTENT
                        }
                    }
                }),
            ),
        )
        .await
        .unwrap();
    });
    let ready = SessionRequest::new(
        "worker".into(),
        registration.session_token.clone(),
        1,
        WorkerReadyPayload {
            worker_id: "worker".into(),
        },
    )
    .unwrap();
    worker
        .request::<()>(Command::Ready(ready.clone()))
        .await
        .unwrap();
    worker.request::<()>(Command::Ready(ready)).await.unwrap();
    acknowledge_until_ready(&client).await;
    let mcp_task = tokio::spawn(acknowledge_directives(client.clone()));
    let target = lock(&state.worker_sessions)["worker"]
        .pending_target
        .clone();
    drop(worker);
    wait_disconnected(&state, ComponentRole::Worker, "worker").await;
    assert!(!target.control_available());
    assert!(matches!(
        state
            .registry
            .current_directive(identity.fingerprint(), &McpSessionId::new("owner").unwrap())
            .unwrap(),
        BrokerDirective::WaitForWorker { .. }
    ));
    let restored = Client::default();
    let handshake = begin_handshake(
        &restored,
        &origin,
        ComponentRole::Worker,
        &identity,
        "worker",
        None,
    )
    .await
    .unwrap();
    let recovered: WorkerRegisterResponse = restored
        .request(Command::RecoverWorker(WorkerRecoverRequest {
            diagnostics: None,
            proof: handshake.proof,
            worker_id: "worker".into(),
            endpoint: endpoint.clone(),
            tls_root_certificate: None,
            generation_grant: registration.generation_grant.clone(),
        }))
        .await
        .unwrap();
    assert!(!target.control_available());
    RecoveryFixture {
        state,
        origin,
        identity,
        worker: restored,
        registration: recovered,
        target,
        fail_probe,
        block_probe,
        probe_started,
        probe_release,
        tasks: vec![task, server, mcp_task],
    }
}

#[tokio::test]
async fn failed_recovery_probe_preserves_session_for_retry_and_expiry() {
    let mut fixture = recovering_worker().await;
    fixture
        .fail_probe
        .store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(
        fixture
            .worker
            .request::<()>(fixture.ready(1))
            .await
            .is_err()
    );
    assert!(lock(&fixture.state.worker_sessions).contains_key("worker"));
    assert!(!fixture.target.control_available());
    fixture
        .fail_probe
        .store(false, std::sync::atomic::Ordering::Relaxed);
    fixture
        .worker
        .request::<()>(fixture.ready(2))
        .await
        .unwrap();
    assert!(fixture.target.control_available());
    // Fail a subsequent probe and disconnect without another recovery attempt.
    fixture
        .fail_probe
        .store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(
        fixture
            .worker
            .request::<()>(fixture.ready(3))
            .await
            .is_err()
    );
    drop(std::mem::take(&mut fixture.worker));
    wait_disconnected(&fixture.state, ComponentRole::Worker, "worker").await;
    let generation = lock(&fixture.state.sockets.peers)[&key(ComponentRole::Worker, "worker")]
        .generation
        .clone();
    lock(&fixture.state.sockets.peers)
        .get_mut(&key(ComponentRole::Worker, "worker"))
        .unwrap()
        .disconnected = Some(tokio::time::Instant::now() - GRACE);
    disconnected(
        fixture.state.clone(),
        ComponentRole::Worker,
        "worker".into(),
        generation,
        "test_disconnect",
    )
    .await;
    tokio::time::timeout(Duration::from_secs(2), async {
        while lock(&fixture.state.worker_sessions).contains_key("worker") {
            tokio::task::yield_now().await;
        }
        while matches!(
            fixture
                .state
                .registry
                .current_directive(
                    fixture.identity.fingerprint(),
                    &McpSessionId::new("owner").unwrap()
                )
                .unwrap(),
            BrokerDirective::WaitForWorker { .. }
        ) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn slow_probe_does_not_block_other_control_sessions_or_stale_publication() {
    let fixture = recovering_worker().await;
    fixture
        .block_probe
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let worker = fixture.worker.clone();
    let ready = fixture.ready(1);
    let probing = tokio::spawn(async move { worker.request::<()>(ready).await });
    fixture.probe_started.notified().await;
    let other = Client::default();
    tokio::time::timeout(
        Duration::from_secs(1),
        mcp(&other, &fixture.origin, &fixture.identity, "other"),
    )
    .await
    .unwrap();
    // Replace the socket while its old probe is still in flight.
    let replacement = Client::default();
    let handshake = begin_handshake(
        &replacement,
        &fixture.origin,
        ComponentRole::Worker,
        &fixture.identity,
        "worker",
        None,
    )
    .await
    .unwrap();
    let _: WorkerRegisterResponse = replacement
        .request(Command::RecoverWorker(WorkerRecoverRequest {
            diagnostics: None,
            proof: handshake.proof,
            worker_id: "worker".into(),
            endpoint: fixture.target.endpoint().into(),
            tls_root_certificate: None,
            generation_grant: fixture.registration.generation_grant.clone(),
        }))
        .await
        .unwrap();
    fixture.probe_release.notify_one();
    assert!(probing.await.unwrap().is_err());
    assert!(!fixture.target.control_available());
}

#[tokio::test]
async fn drain_during_readiness_probe_prevents_publication() {
    let fixture = recovering_worker().await;
    fixture
        .block_probe
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let worker = fixture.worker.clone();
    let ready = fixture.ready(1);
    let probing = tokio::spawn(async move { worker.request::<()>(ready).await });
    fixture.probe_started.notified().await;
    fixture.state.sockets.drain(WorkerDrainRequest {
        worker_id: "worker".into(),
        deadline_unix_ms: now_unix_ms() + DRAIN_LIFETIME_MS,
        timeout_ms: Some(DRAIN_LIFETIME_MS),
    });
    fixture.probe_release.notify_one();
    assert!(probing.await.unwrap().is_err());
    assert!(!fixture.target.control_available());
    assert!(fixture.state.sockets.draining("worker"));
}

async fn acknowledge_until_ready(client: &Client) {
    loop {
        let Event::Directive {
            request_id,
            directive,
        } = client.next().await.unwrap()
        else {
            panic!("directive expected")
        };
        client.acknowledge(request_id).await.unwrap();
        if matches!(directive, BrokerDirective::ReuseWorker { .. }) {
            break;
        }
    }
}

async fn acknowledge_directives(client: Client) {
    while let Ok(Event::Directive { request_id, .. }) = client.next().await {
        if client.acknowledge(request_id).await.is_err() {
            break;
        }
    }
}

#[tokio::test]
async fn blocked_worker_transaction_does_not_block_other_sessions() {
    let fixture = recovering_worker().await;
    let _worker = fixture
        .state
        .sockets
        .transaction(ComponentRole::Worker, "worker")
        .await;
    let other = Client::default();
    tokio::time::timeout(
        Duration::from_secs(1),
        mcp(&other, &fixture.origin, &fixture.identity, "independent"),
    )
    .await
    .unwrap();
    // Another transaction for this same worker must still wait for the current owner.
    assert!(
        tokio::time::timeout(
            Duration::from_millis(20),
            fixture
                .state
                .sockets
                .transaction(ComponentRole::Worker, "worker")
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn both_upgrade_routes_apply_transport_peer_rate_limits() {
    for path in [
        crate::daemon::common::socket::MCP_SOCKET_PATH,
        crate::daemon::common::socket::WORKER_SOCKET_PATH,
    ] {
        let (_state, origin, task) = daemon(true).await;
        let url = format!("{}{path}", origin.replacen("http", "ws", 1));
        for _ in 0..CHALLENGES_PER_PEER_WINDOW {
            let (socket, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
            drop(socket);
        }
        let error = tokio_tungstenite::connect_async(&url).await.unwrap_err();
        let tokio_tungstenite::tungstenite::Error::Http(response) = error else {
            panic!("expected rate-limit response")
        };
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        task.abort();
    }
}

#[tokio::test]
async fn owner_disconnect_revokes_the_grant_and_only_connected_mcp_receives_replacement() {
    let (state, origin, task) = daemon(false).await;
    let identity = MachineIdentity::generate().unwrap().identity;
    let owner = Client::default();
    let initial = mcp(&owner, &origin, &identity, "a-owner").await;
    let old =
        crate::daemon::common::control::WorkerBootstrap::from_directive(initial.directive).unwrap();
    let standby = Client::default();
    mcp(&standby, &origin, &identity, "b-standby").await;
    drop(owner);
    wait_disconnected(&state, ComponentRole::Mcp, "a-owner").await;
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if state
                .registry
                .snapshot(identity.fingerprint())
                .unwrap()
                .launch_owner
                .is_some_and(|owner| owner.as_str() == "b-standby")
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(!lock(&state.activations).contains_key(&old.activation_id));
    assert_eq!(
        state
            .registry
            .snapshot(identity.fingerprint())
            .unwrap()
            .reference_count,
        2
    );
    assert_eq!(
        state
            .registry
            .snapshot(identity.fingerprint())
            .unwrap()
            .launch_owner
            .unwrap()
            .as_str(),
        "b-standby"
    );
    assert!(!lock(&state.pending_directives).contains_key("a-owner"));
    let event = standby.next().await.unwrap();
    if let Event::Directive { request_id, .. } = event {
        standby.acknowledge(request_id).await.unwrap();
    }
    drop(standby);
    wait_disconnected(&state, ComponentRole::Mcp, "b-standby").await;
    assert!(state.registry.activation_candidates(u64::MAX).is_empty());
    task.abort();
}

#[tokio::test(flavor = "current_thread")]
async fn blocked_durable_publication_does_not_stall_control_or_health() {
    use crate::daemon::common::control::{
        ActivationCancellation, CancelActivationPayload, WorkerBootstrap,
    };
    let (state, origin, daemon_task) = daemon(false).await;
    let identity = MachineIdentity::generate().unwrap().identity;
    let owner = Client::default();
    let registration = mcp(&owner, &origin, &identity, "blocked-publication-owner").await;
    let bootstrap = WorkerBootstrap::from_directive(registration.directive).unwrap();
    let held_state = Arc::clone(&state);
    let (locked, acquired) = tokio::sync::oneshot::channel();
    let (release, released) = std::sync::mpsc::channel();
    let holder = std::thread::spawn(move || {
        let _publication = lock(&held_state.worker_generation_publication);
        locked.send(()).unwrap();
        let _ = released.recv_timeout(Duration::from_secs(5));
    });
    acquired.await.unwrap();
    let cancelling_owner = owner.clone();
    let cancellation = tokio::spawn(async move {
        cancelling_owner
            .request::<ActivationCancellation>(Command::CancelActivation(
                SessionRequest::new(
                    "blocked-publication-owner".into(),
                    registration.session_token,
                    1,
                    CancelActivationPayload {
                        activation_id: bootstrap.activation_id,
                    },
                )
                .unwrap(),
            ))
            .await
    });
    spawn_maintenance(Arc::clone(&state));
    let start = std::time::Instant::now();
    tokio::time::sleep(Duration::from_millis(50)).await;
    let response = tokio::time::timeout(
        Duration::from_secs(1),
        reqwest::Client::new()
            .get(format!("{origin}/healthz"))
            .send(),
    )
    .await;
    let elapsed = start.elapsed();
    let still_waiting = !cancellation.is_finished();
    release.send(()).ok();
    holder.join().unwrap();
    assert!(
        elapsed < Duration::from_secs(2),
        "durable lock parked the runtime for {elapsed:?}"
    );
    assert_eq!(response.unwrap().unwrap().status(), StatusCode::OK);
    assert!(
        still_waiting,
        "cancellation must wait for publication to finish"
    );
    assert_eq!(
        cancellation.await.unwrap().unwrap(),
        ActivationCancellation::Cancelled
    );
    daemon_task.abort();
}

#[tokio::test]
async fn worker_setup_failure_releases_staged_sessions_before_disconnect_grace() {
    use crate::daemon::common::control::{WorkerActivationFailureReason, WorkerBootstrap};
    let (state, origin, daemon_task) = daemon(false).await;
    let identity = MachineIdentity::generate().unwrap().identity;
    let owner = Client::default();
    let registration = mcp(&owner, &origin, &identity, "setup-failure-owner").await;
    let bootstrap = WorkerBootstrap::from_directive(registration.directive).unwrap();
    let worker = Client::default();
    let worker_id = "setup-failure-worker";
    let handshake = begin_handshake(
        &worker,
        &origin,
        ComponentRole::Worker,
        &identity,
        worker_id,
        None,
    )
    .await
    .unwrap();
    let staged: WorkerRegisterResponse = worker
        .request(Command::RegisterWorker(WorkerRegisterRequest {
            diagnostics: None,
            proof: handshake.proof,
            worker_id: worker_id.into(),
            endpoint: "http://127.0.0.1:41000".into(),
            activation_id: bootstrap.activation_id.clone(),
            activation_token: bootstrap.activation_token,
            tls_root_certificate: None,
        }))
        .await
        .unwrap();
    let generation = staged.generation_grant.generation_id;
    state
        .active_worker_generations
        .publish(identity.fingerprint(), &generation, None)
        .unwrap();
    // The MCP reports terminal exits from configuration or plugin discovery before readiness.
    drop(worker);
    wait_disconnected(&state, ComponentRole::Worker, worker_id).await;
    owner
        .request::<()>(Command::ActivationFailed(
            SessionRequest::new(
                "setup-failure-owner".into(),
                registration.session_token,
                1,
                ActivationFailedPayload {
                    activation_id: bootstrap.activation_id.clone(),
                    failure_reason: WorkerActivationFailureReason::WorkerExitedBeforeReady,
                },
            )
            .unwrap(),
        ))
        .await
        .unwrap();
    assert!(lock(&state.worker_sessions).is_empty());
    assert!(!lock(&state.activations).contains_key(&bootstrap.activation_id));
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if !state
                .active_worker_generations
                .matches(identity.fingerprint(), &generation)
                .unwrap()
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let snapshot = state.registry.snapshot(identity.fingerprint()).unwrap();
    assert_eq!(snapshot.state, RouteStateKind::PassThrough);
    assert_eq!(snapshot.reference_count, 1);
    daemon_task.abort();
}

struct ReadyWorker {
    state: Arc<DaemonState>,
    identity: MachineIdentity,
    owner: Client,
    worker: Client,
    session_token: SensitiveString,
    origin: String,
    endpoint: String,
    generation_grant: crate::daemon::common::control::WorkerGenerationGrant,
    activation_id: String,
    activation_token: SensitiveString,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

/// A registered worker that passed its readiness probe, with its launching MCP connected.
async fn ready_worker() -> ReadyWorker {
    let (state, origin, task) = daemon(false).await;
    let identity = MachineIdentity::generate().unwrap().identity;
    let owner = Client::default();
    let mcp_registration = mcp(&owner, &origin, &identity, "owner").await;
    let crate::daemon::common::control::WorkerBootstrap {
        activation_id,
        activation_token,
        ..
    } = crate::daemon::common::control::WorkerBootstrap::from_directive(mcp_registration.directive)
        .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let worker = Client::default();
    let handshake = begin_handshake(
        &worker,
        &origin,
        ComponentRole::Worker,
        &identity,
        "worker",
        None,
    )
    .await
    .unwrap();
    let registration: WorkerRegisterResponse = worker
        .request(Command::RegisterWorker(WorkerRegisterRequest {
            diagnostics: None,
            proof: handshake.proof,
            worker_id: "worker".into(),
            endpoint: endpoint.clone(),
            activation_id: activation_id.clone(),
            activation_token: activation_token.clone(),
            tls_root_certificate: None,
        }))
        .await
        .unwrap();
    let data = registration.data_token.clone();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().route(
                WORKER_PROBE_PATH,
                axum::routing::get(move |headers: HeaderMap| {
                    let data = data.clone();
                    async move {
                        assert_eq!(headers[WORKER_TOKEN_HEADER], data.expose());
                        StatusCode::NO_CONTENT
                    }
                }),
            ),
        )
        .await
        .unwrap();
    });
    let ready = SessionRequest::new(
        "worker".into(),
        registration.session_token.clone(),
        1,
        WorkerReadyPayload {
            worker_id: "worker".into(),
        },
    )
    .unwrap();
    worker.request::<()>(Command::Ready(ready)).await.unwrap();
    acknowledge_until_ready(&owner).await;
    let mcp_task = tokio::spawn(acknowledge_directives(owner.clone()));
    ReadyWorker {
        state,
        identity,
        owner,
        worker,
        session_token: registration.session_token,
        origin,
        endpoint,
        generation_grant: registration.generation_grant,
        activation_id,
        activation_token,
        tasks: vec![task, server, mcp_task],
    }
}

fn worker_release(token: SensitiveString) -> Command {
    Command::Release(
        SessionRequest::new(
            "worker".into(),
            token,
            2,
            crate::daemon::common::control::ReleasePayload::default(),
        )
        .unwrap(),
    )
}

/// A worker that announces its exit is removed at once, so the route can launch a replacement
/// instead of waiting out the reconnect grace meant for a live worker.
#[tokio::test]
async fn worker_release_before_exit_skips_the_reconnect_grace() {
    let fixture = ready_worker().await;
    fixture
        .worker
        .request::<()>(worker_release(fixture.session_token.clone()))
        .await
        .unwrap();
    // Released but still tearing down: no more traffic, and no replacement yet.
    let target = lock(&fixture.state.worker_sessions)["worker"]
        .pending_target
        .clone();
    assert!(!target.control_available());
    assert!(lock(&fixture.state.worker_sessions).contains_key("worker"));
    assert!(!matches!(
        owner_directive_now(&fixture.state, &fixture.identity),
        BrokerDirective::LaunchWorker { .. }
    ));
    drop(fixture.worker);
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let removed = !lock(&fixture.state.worker_sessions).contains_key("worker");
            if removed
                && matches!(
                    owner_directive_now(&fixture.state, &fixture.identity),
                    BrokerDirective::LaunchWorker { .. }
                )
            {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("released worker is removed and a replacement is nominated without the grace");
    drop(fixture.owner);
    for task in fixture.tasks {
        task.abort();
    }
}

/// A worker that announced its exit must not be made routable again by a later Ready on the
/// same connection; traffic would reach a worker that is tearing down.
#[tokio::test]
async fn worker_ready_is_rejected_after_an_announced_exit() {
    let fixture = ready_worker().await;
    fixture
        .worker
        .request::<()>(worker_release(fixture.session_token.clone()))
        .await
        .unwrap();
    let ready = fixture
        .worker
        .request::<()>(Command::Ready(
            SessionRequest::new(
                "worker".into(),
                fixture.session_token.clone(),
                3,
                WorkerReadyPayload {
                    worker_id: "worker".into(),
                },
            )
            .unwrap(),
        ))
        .await;
    assert!(
        matches!(&ready, Err(CliError::Launch(message)) if message == "worker announced its exit"),
        "{ready:?}"
    );
    let target = lock(&fixture.state.worker_sessions)["worker"]
        .pending_target
        .clone();
    assert!(!target.control_available());
    assert!(lock(&fixture.state.sockets.exiting_workers).contains("worker"));
    drop(fixture.owner);
    for task in fixture.tasks {
        task.abort();
    }
}

/// A registration replayed after the exit notice is refused for the same reason as recovery.
#[tokio::test]
async fn worker_registration_is_rejected_after_an_announced_exit() {
    let fixture = ready_worker().await;
    fixture
        .worker
        .request::<()>(worker_release(fixture.session_token.clone()))
        .await
        .unwrap();
    let late = Client::default();
    let handshake = begin_handshake(
        &late,
        &fixture.origin,
        ComponentRole::Worker,
        &fixture.identity,
        "worker",
        None,
    )
    .await
    .unwrap();
    let registered = late
        .request::<WorkerRegisterResponse>(Command::RegisterWorker(WorkerRegisterRequest {
            diagnostics: None,
            proof: handshake.proof,
            worker_id: "worker".into(),
            endpoint: fixture.endpoint.clone(),
            activation_id: fixture.activation_id.clone(),
            activation_token: fixture.activation_token.clone(),
            tls_root_certificate: None,
        }))
        .await;
    assert!(
        matches!(&registered, Err(CliError::Launch(message)) if message == "worker announced its exit"),
        "{registered:?}"
    );
    drop(late);
    drop(fixture.owner);
    for task in fixture.tasks {
        task.abort();
    }
}

fn owner_directive_now(state: &DaemonState, identity: &MachineIdentity) -> BrokerDirective {
    state
        .registry
        .current_directive(identity.fingerprint(), &McpSessionId::new("owner").unwrap())
        .unwrap()
}

/// A release must carry the worker's own session token; otherwise the disconnect keeps the
/// reconnect grace a live worker may still use.
#[tokio::test]
async fn worker_release_requires_the_worker_session() {
    let fixture = ready_worker().await;
    let forged = SensitiveString::new("not-the-session-token").unwrap();
    assert!(
        fixture
            .worker
            .request::<()>(worker_release(forged))
            .await
            .is_err()
    );
    let ReadyWorker {
        state,
        identity,
        owner,
        worker,
        tasks,
        ..
    } = fixture;
    drop(worker);
    wait_disconnected(&state, ComponentRole::Worker, "worker").await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(lock(&state.worker_sessions).contains_key("worker"));
    assert!(matches!(
        owner_directive_now(&state, &identity),
        BrokerDirective::WaitForWorker { .. }
    ));
    drop(owner);
    for task in tasks {
        task.abort();
    }
}

/// A late recovery from a worker that announced its exit must not take over its control
/// session, or its disconnect could launch a replacement while teardown is still running.
#[tokio::test]
async fn worker_recovery_is_rejected_after_an_announced_exit() {
    let fixture = ready_worker().await;
    fixture
        .worker
        .request::<()>(worker_release(fixture.session_token.clone()))
        .await
        .unwrap();
    let late = Client::default();
    let handshake = begin_handshake(
        &late,
        &fixture.origin,
        ComponentRole::Worker,
        &fixture.identity,
        "worker",
        None,
    )
    .await
    .unwrap();
    let recovered = late
        .request::<WorkerRegisterResponse>(Command::RecoverWorker(WorkerRecoverRequest {
            diagnostics: None,
            proof: handshake.proof,
            worker_id: "worker".into(),
            endpoint: fixture.endpoint.clone(),
            tls_root_certificate: None,
            generation_grant: fixture.generation_grant.clone(),
        }))
        .await;
    assert!(
        matches!(&recovered, Err(CliError::Launch(message)) if message == "worker announced its exit"),
        "{recovered:?}"
    );
    // The announcing connection still owns the session until it closes.
    assert!(lock(&fixture.state.worker_sessions).contains_key("worker"));
    drop(late);
    drop(fixture.worker);
    tokio::time::timeout(Duration::from_secs(2), async {
        while lock(&fixture.state.worker_sessions).contains_key("worker") {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the announced exit still removes the worker without the grace");
    drop(fixture.owner);
    for task in fixture.tasks {
        task.abort();
    }
}
