// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hook delivery must remain responsive under model response backpressure.

use super::*;

fn tolerant_listener(
    protocol: TestProtocol,
    listener: TcpListener,
    app: Router,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut connections = tokio::task::JoinSet::new();
        loop {
            tokio::select! {
                accepted = listener.accept() => {
                    let (stream, _) = accepted.unwrap();
                    stream.set_nodelay(true).unwrap();
                    let service = TowerToHyperService::new(app.clone());
                    connections.spawn(async move {
                        match protocol {
                            TestProtocol::Http1 => {
                                let _ = http1::Builder::new().serve_connection(TokioIo::new(stream), service).with_upgrades().await;
                            }
                            TestProtocol::Http2 => {
                                let mut builder = http2::Builder::new(TokioExecutor::new());
                                builder.max_concurrent_streams(256);
                                let _ = builder.serve_connection(TokioIo::new(stream), service).await;
                            }
                        }
                    });
                }
                Some(result) = connections.join_next(), if !connections.is_empty() => { result.unwrap(); }
            }
        }
    })
}

async fn tolerant_router(
    protocol: TestProtocol,
    app: Router,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    (address, tolerant_listener(protocol, listener, app))
}

fn unread_worker(counted: Arc<AtomicUsize>) -> Router {
    Router::new()
        .route(
            WORKER_PROBE_PATH,
            axum::routing::get(|| async { StatusCode::NO_CONTENT }),
        )
        .route(
            "/hooks/codex",
            axum::routing::post(|| async { Json(json!({"source": "worker-hook"})) }),
        )
        .route(
            "/v1/responses",
            axum::routing::post(move || {
                let polls = counted.clone();
                async move {
                    Response::new(Body::new(CountedProviderBody {
                        // Reuse one 64 KiB frame; a large logical stream keeps TCP buffers
                        // from absorbing the entire response before the timeout experiment.
                        remaining: 16 * 1024,
                        frame: Bytes::from(vec![b'x'; 64 * 1024]),
                        polls,
                    }))
                }
            }),
        )
}

#[tokio::test]
async fn aep278_http2_unread_responses_do_not_starve_independent_hooks() {
    let polls = Arc::new(AtomicUsize::new(0));
    let worker = unread_worker(polls.clone());
    let (worker_address, worker_task) = tolerant_router(TestProtocol::Http2, worker).await;
    let token = route_token();
    let (app, target, _generation) = daemon_router_with_staged_worker(&token, worker_address);
    let (address, daemon_task) = tolerant_router(TestProtocol::Http1, app).await;
    let client = client_for(TestProtocol::Http1);
    let mut unread = Vec::new();
    let mut blocked_at = None;
    for count in 1..=128 {
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            client.request(provider_request(address, ProviderKind::OpenAi, &token)),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "model response {count} must succeed to create backpressure"
        );
        unread.push(response);
        if ![1, 8, 32, 128].contains(&count) {
            continue;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        let stage = Arc::new(AtomicUsize::new(0));
        let pending = async {
            let response = client
                .request(
                    Request::post(format!("http://{address}/hooks/codex"))
                        .header(CLIENT_TOKEN_HEADER, &token)
                        .body(box_body(Full::new(Bytes::from_static(b"{}"))))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            stage.store(1, Ordering::SeqCst);
            let body = response.into_body().collect().await.unwrap().to_bytes();
            assert_eq!(
                &body[..],
                br#"{"source":"worker-hook"}"#,
                "hook verdict must come from the worker, not daemon pass-through"
            );
            stage.store(2, Ordering::SeqCst);
        };
        tokio::pin!(pending);
        let start = std::time::Instant::now();
        if tokio::time::timeout(Duration::from_secs(2), &mut pending)
            .await
            .is_err()
        {
            let health = tokio::time::timeout(
                Duration::from_secs(2),
                client.request(
                    Request::get(format!("http://{address}/healthz"))
                        .body(box_body(Empty::<Bytes>::new()))
                        .unwrap(),
                ),
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(health.status(), StatusCode::OK);
            let _ = health.into_body().collect().await.unwrap();
            assert!(target.control_available());
            let direct_start = std::time::Instant::now();
            let direct = client_for(TestProtocol::Http2);
            tokio::time::timeout(Duration::from_secs(2), async {
                let response = direct
                    .request(
                        Request::post(format!("http://{worker_address}/hooks/codex"))
                            .header(WORKER_TOKEN_HEADER, WORKER_TOKEN)
                            .body(box_body(Full::new(Bytes::from_static(b"{}"))))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                assert_eq!(
                    &response.into_body().collect().await.unwrap().to_bytes()[..],
                    br#"{"source":"worker-hook"}"#
                );
            })
            .await
            .expect("fresh independent worker connection must remain responsive");
            eprintln!(
                "AEP278 unread_responses blocked_stage={} fresh_worker_connection_hook_ms={:.3}",
                stage.load(Ordering::SeqCst),
                direct_start.elapsed().as_secs_f64() * 1000.0
            );
            unread.clear();
            let resumed = tokio::time::timeout(Duration::from_secs(3), &mut pending)
                .await
                .is_ok();
            eprintln!(
                "AEP278 unread_responses blocked_at={count} health_ok=true control_available=true resumed_after_cancel={resumed} frame_polls={}",
                polls.load(Ordering::SeqCst)
            );
            blocked_at = Some(count);
            break;
        }
        probe_worker(&target)
            .await
            .expect("readiness remains responsive under model backpressure");
        eprintln!(
            "AEP278 unread_responses count={count} hook_ms={:.3} frame_polls={}",
            start.elapsed().as_secs_f64() * 1000.0,
            polls.load(Ordering::SeqCst)
        );
    }
    unread.clear();
    tokio::time::timeout(Duration::from_secs(3), async {
        while target.in_flight() != 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("cancelled model responses release broker ownership");
    worker_task.abort();
    daemon_task.abort();
    assert!(
        blocked_at.is_none(),
        "independent hook starved at {blocked_at:?} unread responses"
    );
}

fn daemon_router_with_staged_worker(
    route_token: &str,
    worker_address: std::net::SocketAddr,
) -> (Router, Arc<WorkerTarget>, tempfile::TempDir) {
    let machine_identity = MachineIdentity::generate()
        .expect("machine identity")
        .identity;
    let fingerprint = machine_identity.fingerprint();
    let credential = RouteCredential::parse(route_token.to_owned()).expect("route credential");
    let registry = Registry::new(false);
    let mcp_session = McpSessionId::new("test-mcp-session").expect("MCP session ID");
    let launch = WorkerLaunch {
        activation_id: "test-activation".into(),
        activation_token: SensitiveString::new("unused-test-activation-token")
            .expect("activation token"),
        deadline_unix_ms: now_unix_ms().saturating_add(ACTIVATION_LIFETIME_MS),
        bind_ip: Ipv4Addr::LOCALHOST,
        port: 0,
        advertise_address: None,
    };
    let directive = registry
        .register_mcp(
            McpRegistration {
                fingerprint,
                token_digest: credential.digest(),
                session_id: mcp_session,
                lease_expires_at_unix_ms: u64::MAX,
            },
            launch,
        )
        .expect("register test MCP");
    assert!(matches!(directive, BrokerDirective::LaunchWorker { .. }));
    let identity = MachineIdentity::generate()
        .expect("daemon identity")
        .identity;
    let generation_state = tempfile::tempdir().expect("generation state directory");
    let active_worker_generations = ActiveWorkerGenerations::load_for_test(
        generation_state
            .path()
            .join("active-worker-generations.json"),
    )
    .expect("active generation state");
    let state = Arc::new(DaemonState {
        collector: None,
        sockets: socket::Hub::default(),
        registry,
        identity,
        descriptor: crate::daemon::common::control::descriptor(ComponentRole::Daemon),
        instance_id: "test-daemon".into(),
        pass_through: false,
        loopback_listener: true,
        public_origin: "http://127.0.0.1:1".into(),
        config: GatewayConfig::default(),
        upstream: pooled_client().expect("daemon pass-through client"),
        worker_clients: WorkerClientPool::new().expect("daemon worker clients"),
        challenges: Mutex::new(HashMap::new()),
        activations: Mutex::new(HashMap::new()),
        mcp_sessions: Mutex::new(HashMap::new()),
        worker_sessions: Mutex::new(HashMap::new()),
        pending_directives: Mutex::new(HashMap::new()),
        active_worker_generations,
        worker_generation_publication: Mutex::new(()),
        diagnostics_http: None,
    });
    let response = stage_worker(
        &state,
        fingerprint,
        "test-worker".into(),
        format!("http://{worker_address}"),
        None,
        None,
        WorkerPublication::Activation {
            activation_id: "test-activation".into(),
        },
        None,
        HandshakeProof {
            signer: ComponentRole::Daemon,
            signature: state.identity.sign(b"test-readiness"),
        },
    );
    assert_eq!(response.status(), StatusCode::OK);
    let target = lock(&state.worker_sessions)["test-worker"]
        .pending_target
        .clone();
    state
        .registry
        .mark_worker_ready(fingerprint, "test-activation", Arc::clone(&target))
        .unwrap();
    (router(state), target, generation_state)
}
