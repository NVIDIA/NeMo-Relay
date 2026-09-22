// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use base64::Engine;
use std::sync::Arc;

use super::*;
use crate::daemon::common::client::{begin_handshake, control_client};
use crate::daemon::common::control::{
    WorkerActivationFailureReason, WorkerNetworkHintProof, WorkerReadyPayload,
    WorkerRegisterResponse,
};
use crate::daemon::common::routes::HookRoute;
use crate::daemon::common::state::ROUTE_TOKEN_ENV;
use crate::daemon::common::worker_tls::pooled_worker_tls_client;
use crate::test_support::{EnvScope, PLUGIN_CONFIG_TEST_LOCK};
use axum::Router;
use axum::extract::State;
use axum::routing::post;
use http_body_util::BodyExt as _;
use tower::ServiceExt as _;

type CapturedProviderRequest = Arc<std::sync::Mutex<Option<(HeaderMap, bytes::Bytes)>>>;

#[tokio::test]
async fn daemon_health_is_a_public_process_probe() {
    let state = test_daemon_state(false, "", GatewayConfig::default());
    let app = router(Arc::clone(&state));

    let health = app
        .clone()
        .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    assert_eq!(health.headers()[CACHE_CONTROL], "no-store");
    let health: serde_json::Value =
        serde_json::from_slice(&health.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(health["status"], "ok");
    assert_eq!(health["service"], "nemo-relay-daemon");
    assert_eq!(health["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(health["instance_id"], "public-proxy-test-daemon");
    assert_eq!(health["deployment_mode"], "managed");

    let pass_through = router(test_daemon_state(true, "", GatewayConfig::default()))
        .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let pass_through: serde_json::Value =
        serde_json::from_slice(&pass_through.into_body().collect().await.unwrap().to_bytes())
            .unwrap();
    assert_eq!(pass_through["deployment_mode"], "pass_through");
}

#[tokio::test]
async fn shared_model_catalogs_disable_cache_reuse_between_credentials() {
    let provider = Router::new().route(
        "/v1/models",
        axum::routing::get(|headers: HeaderMap| async move {
            Response::builder()
                .header(axum::http::header::CACHE_CONTROL, "public, max-age=3600")
                .header(axum::http::header::CACHE_CONTROL, "private")
                .body(Body::from(
                    headers[AUTHORIZATION].to_str().unwrap().to_owned(),
                ))
                .unwrap()
        }),
    );
    let provider_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider_origin = format!("http://{}", provider_listener.local_addr().unwrap());
    let provider_task =
        tokio::spawn(async { axum::serve(provider_listener, provider).await.unwrap() });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let state = test_daemon_state_at(
        true,
        "",
        GatewayConfig {
            openai_base_url: provider_origin,
            ..GatewayConfig::default()
        },
        origin.clone(),
    );
    let daemon_task = tokio::spawn({
        let state = Arc::clone(&state);
        async move { axum::serve(listener, router(state)).await.unwrap() }
    });
    let app = router(Arc::clone(&state));
    for (index, provider_auth) in [(1_u8, "Bearer first-user"), (2, "Bearer second-user")] {
        let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([index; 32]);
        let identity = MachineIdentity::generate().unwrap().identity;
        assert_eq!(
            enroll_test_mcp(&state, &origin, &identity, &token, provider_auth)
                .await
                .status(),
            StatusCode::OK,
        );
        let response = app
            .clone()
            .oneshot(
                Request::get("/models")
                    .header(CLIENT_TOKEN_HEADER, &token)
                    .header(AUTHORIZATION, provider_auth)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[axum::http::header::CACHE_CONTROL],
            "no-store"
        );
        assert_eq!(
            response
                .headers()
                .get_all(axum::http::header::CACHE_CONTROL)
                .iter()
                .count(),
            1
        );
        assert_eq!(
            response.into_body().collect().await.unwrap().to_bytes(),
            provider_auth
        );
    }
    for path in ["/models", "/v1/models"] {
        let response = app
            .clone()
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers()[axum::http::header::CACHE_CONTROL],
            "no-store"
        );
    }
    daemon_task.abort();
    provider_task.abort();
}

#[tokio::test(start_paused = true)]
async fn challenge_admission_limits_transport_peers_before_polling_bodies() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x91; 32]);
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    let peers: ChallengePeers = Arc::new(Mutex::new(HashMap::new()));
    let app = Router::new()
        .route(
            crate::daemon::common::socket::MCP_SOCKET_PATH,
            post(issue_challenge),
        )
        .route_layer(from_fn_with_state(peers, limit_challenges))
        .with_state(state.clone());
    let identity = MachineIdentity::generate().unwrap().identity;
    let mut challenge = ChallengeRequest {
        initiator: crate::daemon::common::control::descriptor(ComponentRole::Mcp),
        initiator_instance_id: "limited-peer".into(),
        initiator_public_identity: identity.public_identity(),
        initiator_fingerprint: identity.fingerprint(),
        initiator_nonce: ChallengeRecord::generate(1, 1).unwrap().challenge().nonce,
    };
    for port in 1..=CHALLENGES_PER_PEER_WINDOW {
        let response = app
            .clone()
            .oneshot(
                Request::post(crate::daemon::common::socket::MCP_SOCKET_PATH)
                    .extension(ConnectInfo(SocketAddr::from(([192, 0, 2, 1], port as u16))))
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(&challenge).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    let response = app
        .clone()
        .oneshot(
            Request::post(crate::daemon::common::socket::MCP_SOCKET_PATH)
                .extension(ConnectInfo(SocketAddr::from(([192, 0, 2, 1], 65535))))
                .header("x-forwarded-for", "192.0.2.99")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from_stream(futures_util::stream::poll_fn(
                    |_| -> std::task::Poll<Option<Result<Bytes, std::io::Error>>> {
                        panic!("rate-limited challenge body polled")
                    },
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()[RETRY_AFTER], "15");
    assert_eq!(
        lock(&state.challenges).len(),
        CHALLENGES_PER_PEER_WINDOW as usize
    );

    // A worker on another peer still has capacity and needs no MCP route-token header.
    challenge.initiator = crate::daemon::common::control::descriptor(ComponentRole::Worker);
    let request = |peer| {
        Request::post(crate::daemon::common::socket::MCP_SOCKET_PATH)
            .extension(ConnectInfo(SocketAddr::from((peer, 1))))
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&challenge).unwrap()))
            .unwrap()
    };
    assert_eq!(
        app.clone()
            .oneshot(request([192, 0, 2, 2]))
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    tokio::time::advance(Duration::from_millis(CHALLENGE_LIFETIME_MS)).await;
    assert_eq!(
        app.oneshot(request([192, 0, 2, 1])).await.unwrap().status(),
        StatusCode::OK
    );
}

#[tokio::test(start_paused = true)]
async fn challenge_peer_tracking_is_bounded_and_expired_entries_are_reclaimed() {
    let now = tokio::time::Instant::now();
    let peers: ChallengePeers = Arc::new(Mutex::new(
        (0..MAX_CHALLENGE_PEERS)
            .map(|index| (IpAddr::V4(Ipv4Addr::from(index as u32)), (now, 1)))
            .collect(),
    ));
    let app = Router::new()
        .route("/", post(|| async { StatusCode::NO_CONTENT }))
        .layer(from_fn_with_state(Arc::clone(&peers), limit_challenges));
    let request = || {
        Request::post("/")
            .extension(ConnectInfo("192.0.2.1:1".parse::<SocketAddr>().unwrap()))
            .body(Body::empty())
            .unwrap()
    };
    assert_eq!(
        app.clone().oneshot(request()).await.unwrap().status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    assert_eq!(lock(&peers).len(), MAX_CHALLENGE_PEERS);
    tokio::time::advance(Duration::from_millis(CHALLENGE_LIFETIME_MS)).await;
    assert_eq!(
        app.oneshot(request()).await.unwrap().status(),
        StatusCode::NO_CONTENT
    );
    assert_eq!(lock(&peers).len(), 1);
}

async fn enroll_test_mcp(
    state: &Arc<DaemonState>,
    origin: &str,
    identity: &MachineIdentity,
    token: &str,
    session: &str,
) -> Response<Body> {
    let credential = RouteCredential::parse(token.to_owned()).unwrap();
    let control = control_client().unwrap();
    let handshake = begin_handshake(
        &control,
        origin,
        ComponentRole::Mcp,
        identity,
        session,
        Some(credential.digest()),
    )
    .await
    .unwrap();
    let worker_network = WorkerNetworkHintProof::sign(
        WorkerNetworkHint::new("127.0.0.1", None).unwrap(),
        &handshake.proof.transcript.daemon_target,
        session,
        &handshake.proof.transcript.challenge_id,
        &identity.fingerprint(),
        identity,
    )
    .unwrap();
    register_mcp(
        State(Arc::clone(state)),
        HeaderMap::from_iter([(
            HeaderName::from_static(CLIENT_TOKEN_HEADER),
            HeaderValue::from_str(token).unwrap(),
        )]),
        Json(McpRegisterRequest {
            proof: handshake.proof,
            worker_network,
        }),
    )
    .await
}

#[tokio::test]
async fn open_enrollment_binds_new_tokens_without_allowlist_and_prevents_route_takeover() {
    for pass_through in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let first = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0xa1; 32]);
        let second = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0xa2; 32]);
        let state = test_daemon_state_at(
            pass_through,
            &first,
            GatewayConfig::default(),
            origin.clone(),
        );
        let server = tokio::spawn({
            let state = Arc::clone(&state);
            async move { axum::serve(listener, router(state)).await.unwrap() }
        });
        let app = router(Arc::clone(&state));
        let identity = MachineIdentity::generate().unwrap().identity;
        let other_identity = MachineIdentity::generate().unwrap().identity;
        for (token, machine, session) in [
            (&first, &identity, "first"),
            (&second, &other_identity, "second"),
        ] {
            // Rejection must happen from the request head, before reading even one body frame.
            let body = Body::from_stream(futures_util::stream::poll_fn(
                |_| -> std::task::Poll<Option<Result<bytes::Bytes, std::io::Error>>> {
                    panic!("unregistered request body polled")
                },
            ));
            let response = app
                .clone()
                .oneshot(
                    Request::post("/hooks/pi")
                        .header(CLIENT_TOKEN_HEADER, token)
                        .body(body)
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            let response = enroll_test_mcp(&state, &origin, machine, token, session).await;
            assert_eq!(response.status(), StatusCode::OK);
            let response: McpRegisterResponse =
                serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                    .unwrap();
            assert_eq!(
                matches!(response.directive, BrokerDirective::UsePassThrough),
                pass_through
            );
            assert_eq!(
                matches!(response.directive, BrokerDirective::LaunchWorker { .. }),
                !pass_through
            );
            let response = app
                .clone()
                .oneshot(
                    Request::post("/hooks/pi")
                        .header(CLIENT_TOKEN_HEADER, token)
                        .body(Body::from("{}"))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                if pass_through {
                    StatusCode::OK
                } else {
                    StatusCode::SERVICE_UNAVAILABLE
                }
            );
        }
        assert_eq!(
            enroll_test_mcp(&state, &origin, &other_identity, &first, "takeover")
                .await
                .status(),
            StatusCode::UNAUTHORIZED,
        );
        let third = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0xa3; 32]);
        assert_eq!(
            enroll_test_mcp(&state, &origin, &identity, &third, "rebind")
                .await
                .status(),
            StatusCode::UNAUTHORIZED,
        );
        state
            .registry
            .release_mcp(
                identity.fingerprint(),
                &McpSessionId::new("first").unwrap(),
                u64::MAX,
            )
            .unwrap();
        let response = app
            .oneshot(
                Request::post("/hooks/pi")
                    .header(CLIENT_TOKEN_HEADER, &first)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        if pass_through {
            assert!(lock(&state.activations).is_empty());
        }
        server.abort();
    }
}

#[test]
fn worker_endpoint_rejects_bind_only_and_non_origin_values() {
    assert!(validate_worker_endpoint("http://127.0.0.1:1234", None).is_ok());
    assert!(validate_worker_endpoint("http://0.0.0.0:1234", None).is_err());
    assert!(validate_worker_endpoint("http://127.0.0.1:1234/path", None).is_err());
    assert!(validate_worker_endpoint("http://127.0.0.1", None).is_err());
    assert!(validate_worker_endpoint("http://192.0.2.2:1234", None).is_err());
    assert!(validate_worker_endpoint("https://192.0.2.2:1234", Some("root")).is_ok());
}

#[test]
fn public_credential_requires_exactly_one_valid_value() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([7_u8; 32]);
    let mut headers = HeaderMap::new();
    assert!(public_credential(&headers).is_err());
    headers.insert(
        CLIENT_TOKEN_HEADER,
        HeaderValue::from_str(&token).expect("header"),
    );
    assert!(public_credential(&headers).is_ok());
    headers.append(
        CLIENT_TOKEN_HEADER,
        HeaderValue::from_str(&token).expect("header"),
    );
    assert!(public_credential(&headers).is_err());

    let mut non_utf8 = HeaderMap::new();
    non_utf8.insert(
        CLIENT_TOKEN_HEADER,
        HeaderValue::from_bytes(&[0xff]).expect("opaque header value"),
    );
    assert!(public_credential(&non_utf8).is_err());
}

#[test]
fn daemon_tls_parsing_rejects_malformed_and_incomplete_pem_material() {
    assert!(decode_pem_blocks(&[0xff], "CERTIFICATE").is_err());
    assert!(decode_pem_blocks(b"-----BEGIN CERTIFICATE-----", "CERTIFICATE").is_err());
    assert!(
        decode_pem_blocks(
            b"-----BEGIN CERTIFICATE-----!-----END CERTIFICATE-----",
            "CERTIFICATE"
        )
        .is_err()
    );
    assert!(
        decode_pem_blocks(
            b"-----BEGIN CERTIFICATE-----\n-----END CERTIFICATE-----",
            "CERTIFICATE"
        )
        .is_err()
    );

    let directory = tempfile::tempdir().unwrap();
    let certificate = directory.path().join("certificate.pem");
    let key = directory.path().join("key.pem");
    std::fs::write(&certificate, b"no certificate blocks").unwrap();
    std::fs::write(&key, b"no key blocks").unwrap();
    assert!(load_tls_config(&certificate, &key).is_err());
    std::fs::write(
        &certificate,
        b"-----BEGIN CERTIFICATE-----AQ==-----END CERTIFICATE-----",
    )
    .unwrap();
    assert!(load_tls_config(&certificate, &key).is_err());
    std::fs::write(
        &key,
        b"-----BEGIN PRIVATE KEY-----AQ==-----END PRIVATE KEY-----\n-----BEGIN PRIVATE KEY-----Ag==-----END PRIVATE KEY-----",
    )
    .unwrap();
    assert!(load_tls_config(&certificate, &key).is_err());
}

#[test]
fn worker_activation_helpers_cover_remote_and_rejected_endpoint_shapes() {
    let loopback = fresh_launch(WorkerNetworkHint::new("localhost", None).unwrap()).unwrap();
    assert_eq!(loopback.bind_ip, Ipv4Addr::LOCALHOST);
    assert_eq!(loopback.port, 0);
    assert!(loopback.advertise_address.is_none());

    let remote = fresh_launch(WorkerNetworkHint::new("worker.example", Some(9443)).unwrap())
        .expect("remote worker launch");
    assert_eq!(remote.bind_ip, Ipv4Addr::UNSPECIFIED);
    assert_eq!(remote.advertise_address.as_deref(), Some("worker.example"));
    let activation = Activation {
        fingerprint: MachineIdentity::generate().unwrap().identity.fingerprint(),
        secret_digest: TokenDigest::from_token(b"secret"),
        deadline_unix_ms: u64::MAX,
        bind_ip: remote.bind_ip,
        port: remote.port,
        advertise_address: remote.advertise_address,
        consumed: false,
    };
    assert!(activation_endpoint_matches(
        "https://worker.example:9443",
        &activation
    ));
    assert!(!activation_endpoint_matches("not a URL", &activation));
    assert!(!activation_endpoint_matches(
        "https://worker.example",
        &activation
    ));
    assert!(!activation_endpoint_matches(
        "http://worker.example:9443",
        &activation
    ));
    assert!(!reserve_challenge_slot(
        &mut HashMap::new(),
        now_unix_ms(),
        ComponentRole::Daemon
    ));
}

#[tokio::test]
async fn forwarding_rejects_invalid_destinations_and_worker_credentials_before_io() {
    let client = pooled_client().unwrap();
    let invalid_destination = forward(
        &client,
        Request::post("/").body(Body::empty()).unwrap(),
        "http://[invalid",
        None,
        None,
    )
    .await;
    assert_eq!(
        invalid_destination.response.status(),
        StatusCode::BAD_GATEWAY
    );
    assert!(matches!(
        invalid_destination.failure,
        Some(ForwardFailure::InvalidDestination)
    ));

    let invalid_credential = forward(
        &client,
        Request::post("/").body(Body::empty()).unwrap(),
        "http://127.0.0.1:1",
        Some((
            HeaderName::from_static(WORKER_TOKEN_HEADER),
            "bad\nvalue".into(),
        )),
        None,
    )
    .await;
    assert_eq!(
        invalid_credential.response.status(),
        StatusCode::INTERNAL_SERVER_ERROR
    );
    assert!(invalid_credential.failure.is_none());
}

#[tokio::test]
async fn worker_response_body_errors_trigger_the_route_callback_once() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let frames = futures_util::stream::iter([Err::<hyper::body::Frame<Bytes>, std::io::Error>(
        std::io::Error::other("body failed"),
    )]);
    let mut body = ErrorObservedBody {
        body: http_body_util::StreamBody::new(frames),
        on_error: Some(move || {
            observed.fetch_add(1, Ordering::SeqCst);
        }),
    };
    assert!(body.frame().await.unwrap().is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(body.frame().await.is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn responses_websocket_probe_is_narrow() {
    let probe = Request::get("/backend-api/codex/responses")
        .header(axum::http::header::UPGRADE, "WebSocket")
        .body(Body::empty())
        .expect("probe");
    assert!(responses_websocket_probe(&probe));

    let ordinary = Request::get("/backend-api/codex/responses")
        .body(Body::empty())
        .expect("ordinary GET");
    assert!(!responses_websocket_probe(&ordinary));
    assert!(!public_method_allowed(
        ordinary.method(),
        ordinary.uri().path()
    ));
    assert!(public_method_allowed(
        &Method::POST,
        "/backend-api/codex/responses"
    ));
    assert!(public_method_allowed(&Method::GET, "/v1/models"));
    assert!(!public_method_allowed(&Method::POST, "/v1/models"));
}

#[test]
fn unavailable_response_includes_retry_after() {
    let response = unavailable_response();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(response.headers().get(RETRY_AFTER).unwrap(), "1");
}

#[tokio::test]
async fn global_pass_through_authenticates_and_forwards_only_provider_headers() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x11_u8; 32]);
    let captured: CapturedProviderRequest = Arc::new(std::sync::Mutex::new(None));
    let provider = Router::new()
        .route(
            "/v1/responses",
            post(
                |State(captured): State<CapturedProviderRequest>,
                 request: Request<Body>| async move {
                    let (parts, body) = request.into_parts();
                    let body = axum::body::to_bytes(body, usize::MAX)
                        .await
                        .expect("provider request body");
                    *captured.lock().expect("capture provider request") =
                        Some((parts.headers, body));
                    Response::builder()
                        .status(StatusCode::CREATED)
                        .header("x-provider-result", "preserved")
                        .body(Body::from("raw provider response"))
                        .expect("provider response")
                },
            ),
        )
        .with_state(Arc::clone(&captured));
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind provider");
    let address = listener.local_addr().expect("provider address");
    let provider_task = tokio::spawn(async move {
        axum::serve(listener, provider)
            .await
            .expect("serve provider");
    });
    let state = test_daemon_state(
        true,
        &token,
        GatewayConfig {
            openai_base_url: format!("http://{address}"),
            openai_auth_header: Some("Bearer configured-provider".into()),
            ..GatewayConfig::default()
        },
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    // The challenge transcript must name the same origin as the serving daemon.
    let mut state = state;
    Arc::get_mut(&mut state).unwrap().public_origin = origin.clone();
    let server = tokio::spawn({
        let state = Arc::clone(&state);
        async move { axum::serve(listener, router(state)).await.unwrap() }
    });
    let identity = MachineIdentity::generate().unwrap().identity;
    assert_eq!(
        enroll_test_mcp(&state, &origin, &identity, &token, "pass-through")
            .await
            .status(),
        StatusCode::OK
    );
    let app = router(state);

    let response = app
        .clone()
        .oneshot(
            Request::post("/v1/responses")
                .header(CLIENT_TOKEN_HEADER, &token)
                .header("x-nemo-relay-session-id", "untrusted-private-header")
                .header("x-provider-feature", "preserved")
                .body(Body::from("provider request bytes"))
                .expect("public request"),
        )
        .await
        .expect("public response");
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(response.headers()["x-provider-result"], "preserved");
    assert_eq!(
        response
            .into_body()
            .collect()
            .await
            .expect("provider response body")
            .to_bytes(),
        "raw provider response"
    );
    let (headers, body) = captured
        .lock()
        .expect("captured provider request")
        .take()
        .expect("provider received request");
    assert_eq!(headers[AUTHORIZATION], "Bearer configured-provider");
    assert_eq!(headers["x-provider-feature"], "preserved");
    assert!(!headers.contains_key(CLIENT_TOKEN_HEADER));
    assert!(!headers.contains_key("x-nemo-relay-session-id"));
    assert_eq!(body, "provider request bytes");

    let hook = app
        .clone()
        .oneshot(
            Request::post("/hooks/claude-code")
                .header(CLIENT_TOKEN_HEADER, &token)
                .body(Body::from("{}"))
                .expect("hook request"),
        )
        .await
        .expect("hook response");
    assert_eq!(hook.status(), StatusCode::OK);
    assert_eq!(
        hook.into_body()
            .collect()
            .await
            .expect("hook body")
            .to_bytes(),
        HookRoute::Claude.pass_through_body()
    );

    for request in [
        Request::get("/v1/responses")
            .header(CLIENT_TOKEN_HEADER, &token)
            .body(Body::empty())
            .expect("disallowed method"),
        Request::post("/not-a-route")
            .header(CLIENT_TOKEN_HEADER, &token)
            .body(Body::empty())
            .expect("unknown route"),
    ] {
        let response = app.clone().oneshot(request).await.expect("response");
        assert!(matches!(
            response.status(),
            StatusCode::METHOD_NOT_ALLOWED | StatusCode::NOT_FOUND
        ));
    }
    provider_task.abort();
    server.abort();
}

#[tokio::test]
async fn unreachable_worker_marks_its_authenticated_route_pass_through() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x22_u8; 32]);
    let credential = RouteCredential::parse(token.clone()).expect("route credential");
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    let fingerprint = MachineIdentity::generate()
        .expect("machine identity")
        .identity
        .fingerprint();
    state
        .registry
        .register_mcp(
            McpRegistration {
                fingerprint,
                token_digest: credential.digest(),
                session_id: McpSessionId::new("worker-route-session").expect("session"),
                lease_expires_at_unix_ms: u64::MAX,
            },
            WorkerLaunch {
                activation_id: "worker-route-activation".into(),
                activation_token: SensitiveString::new("activation-token").expect("token"),
                deadline_unix_ms: u64::MAX,
                bind_ip: Ipv4Addr::LOCALHOST,
                port: 0,
                advertise_address: None,
            },
        )
        .expect("register route");
    state
        .registry
        .mark_worker_ready(
            fingerprint,
            "worker-route-activation",
            Arc::new(
                WorkerTarget::new(
                    "unreachable-worker",
                    "http://127.0.0.1:9",
                    SensitiveString::new("worker-session-token").expect("token"),
                )
                .expect("worker target"),
            ),
        )
        .expect("publish route");

    let response = router(Arc::clone(&state))
        .oneshot(
            Request::post("/v1/responses")
                .header(CLIENT_TOKEN_HEADER, &token)
                .body(Body::from("request bytes"))
                .expect("public request"),
        )
        .await
        .expect("public response");
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(matches!(
        state.registry.resolve_target(&credential.digest()),
        Ok(ResolvedTarget::PassThrough)
    ));
}

#[tokio::test]
async fn authenticated_hook_forwarding_preserves_the_operation_id_for_the_worker() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x67_u8; 32]);
    let credential = RouteCredential::parse(token.clone()).expect("route credential");
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    let fingerprint = MachineIdentity::generate()
        .expect("machine identity")
        .identity
        .fingerprint();
    let activation_id = "operation-id-hook-activation";
    state
        .registry
        .register_mcp(
            McpRegistration {
                fingerprint,
                token_digest: credential.digest(),
                session_id: McpSessionId::new("operation-id-hook-session").expect("session"),
                lease_expires_at_unix_ms: u64::MAX,
            },
            WorkerLaunch {
                activation_id: activation_id.into(),
                activation_token: SensitiveString::new("activation-token").expect("token"),
                deadline_unix_ms: u64::MAX,
                bind_ip: Ipv4Addr::LOCALHOST,
                port: 0,
                advertise_address: None,
            },
        )
        .expect("register route");

    let captured = Arc::new(std::sync::Mutex::new(None));
    let worker = Router::new().fallback({
        let captured = Arc::clone(&captured);
        move |headers: HeaderMap| {
            let captured = Arc::clone(&captured);
            async move {
                *captured.lock().expect("capture worker request") = Some(headers);
                Response::new(Body::from("{}"))
            }
        }
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("worker bind");
    let address = listener.local_addr().expect("worker address");
    let worker_task = tokio::spawn(async move {
        axum::serve(listener, worker).await.expect("worker serve");
    });
    state
        .registry
        .mark_worker_ready(
            fingerprint,
            activation_id,
            Arc::new(
                WorkerTarget::new(
                    "operation-id-worker",
                    format!("http://{address}"),
                    SensitiveString::new("worker-session-token").expect("token"),
                )
                .expect("worker target"),
            ),
        )
        .expect("publish route");

    let operation_id = "018f0f3f-3f7a-7b72-9d0d-e0d8ced91d8b";
    let response = router(Arc::clone(&state))
        .oneshot(
            Request::post("/hooks/codex")
                .header(CLIENT_TOKEN_HEADER, &token)
                .header(crate::operational::OPERATION_ID_HEADER, operation_id)
                .body(Body::from("{}"))
                .expect("hook request"),
        )
        .await
        .expect("hook response");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        captured
            .lock()
            .expect("capture worker request")
            .as_ref()
            .and_then(|headers| headers.get(crate::operational::OPERATION_ID_HEADER))
            .and_then(|value| value.to_str().ok()),
        Some(operation_id)
    );
    worker_task.abort();
}

#[tokio::test]
async fn aborted_public_upload_does_not_demote_a_healthy_worker() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x72_u8; 32]);
    let credential = RouteCredential::parse(token.clone()).expect("route credential");
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    let fingerprint = MachineIdentity::generate()
        .expect("machine identity")
        .identity
        .fingerprint();
    let activation_id = "aborted-upload-activation";
    state
        .registry
        .register_mcp(
            McpRegistration {
                fingerprint,
                token_digest: credential.digest(),
                session_id: McpSessionId::new("aborted-upload-session").expect("session"),
                lease_expires_at_unix_ms: u64::MAX,
            },
            WorkerLaunch {
                activation_id: activation_id.into(),
                activation_token: SensitiveString::new("activation-token").expect("token"),
                deadline_unix_ms: u64::MAX,
                bind_ip: Ipv4Addr::LOCALHOST,
                port: 0,
                advertise_address: None,
            },
        )
        .expect("register route");

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("worker bind");
    let address = listener.local_addr().expect("worker address");
    let worker_task = tokio::spawn(async move {
        let app = Router::new().fallback(|body: Body| async move {
            let _ = body.collect().await;
            StatusCode::NO_CONTENT
        });
        axum::serve(listener, app).await.expect("worker serve");
    });
    state
        .registry
        .mark_worker_ready(
            fingerprint,
            activation_id,
            Arc::new(
                WorkerTarget::new(
                    "healthy-worker",
                    format!("http://{address}"),
                    SensitiveString::new("worker-session-token").expect("token"),
                )
                .expect("worker target"),
            ),
        )
        .expect("publish route");

    let failed_body = futures_util::stream::iter([
        Ok::<_, std::io::Error>(Bytes::from_static(b"partial")),
        Err(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "client upload aborted",
        )),
    ]);
    let response = router(Arc::clone(&state))
        .oneshot(
            Request::post("/v1/responses")
                .header(CLIENT_TOKEN_HEADER, &token)
                .body(Body::from_stream(failed_body))
                .expect("public request"),
        )
        .await
        .expect("public response");
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(matches!(
        state.registry.resolve_target(&credential.digest()),
        Ok(ResolvedTarget::Worker(_))
    ));
    worker_task.abort();
}

#[tokio::test]
async fn public_ingress_rejects_credentials_methods_websockets_and_unready_routes_early() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x23_u8; 32]);
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    let app = router(Arc::clone(&state));

    let unknown = app
        .clone()
        .oneshot(Request::get("/unknown").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(unknown.status(), StatusCode::NOT_FOUND);

    for request in [
        Request::post("/v1/responses").body(Body::empty()).unwrap(),
        Request::post("/v1/responses")
            .header(CLIENT_TOKEN_HEADER, "invalid")
            .body(Body::empty())
            .unwrap(),
    ] {
        assert_eq!(
            app.clone().oneshot(request).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
    }

    let websocket = Request::get("/v1/responses")
        .header(CLIENT_TOKEN_HEADER, &token)
        .header(axum::http::header::UPGRADE, "websocket")
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        app.clone().oneshot(websocket).await.unwrap().status(),
        StatusCode::UPGRADE_REQUIRED
    );

    let fingerprint = MachineIdentity::generate()
        .expect("identity")
        .identity
        .fingerprint();
    let credential = RouteCredential::parse(token.clone()).expect("credential");
    state
        .registry
        .register_mcp(
            McpRegistration {
                fingerprint,
                token_digest: credential.digest(),
                session_id: McpSessionId::new("pending-route").expect("session"),
                lease_expires_at_unix_ms: u64::MAX,
            },
            WorkerLaunch {
                activation_id: "pending-activation".into(),
                activation_token: SensitiveString::new("activation-secret").expect("secret"),
                deadline_unix_ms: u64::MAX,
                bind_ip: Ipv4Addr::LOCALHOST,
                port: 0,
                advertise_address: None,
            },
        )
        .expect("register pending route");
    let unavailable = app
        .oneshot(
            Request::post("/v1/responses")
                .header(CLIENT_TOKEN_HEADER, &token)
                .body(Body::from("must not be forwarded"))
                .expect("request"),
        )
        .await
        .expect("unavailable response");
    assert_eq!(unavailable.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(unavailable.headers()[RETRY_AFTER], "1");
}

#[test]
fn pass_through_auth_injection_supports_environment_and_anthropic_configuration() {
    let _environment = EnvScope::set(&[
        ("OPENAI_API_KEY", Some(std::ffi::OsStr::new(" openai-env "))),
        (
            "ANTHROPIC_API_KEY",
            Some(std::ffi::OsStr::new(" anthropic-env ")),
        ),
    ]);
    let mut openai = HeaderMap::new();
    inject_provider_auth(
        &mut openai,
        ProviderRoute::OpenAi,
        &GatewayConfig::default(),
    );
    assert_eq!(openai[AUTHORIZATION], "Bearer openai-env");

    let mut anthropic = HeaderMap::new();
    inject_provider_auth(
        &mut anthropic,
        ProviderRoute::Anthropic,
        &GatewayConfig::default(),
    );
    assert_eq!(anthropic["x-api-key"], "anthropic-env");

    let mut configured = HeaderMap::new();
    inject_provider_auth(
        &mut configured,
        ProviderRoute::Anthropic,
        &GatewayConfig {
            anthropic_auth_header: Some("configured".into()),
            ..GatewayConfig::default()
        },
    );
    assert_eq!(configured[AUTHORIZATION], "configured");
}

fn test_daemon_state(pass_through: bool, token: &str, config: GatewayConfig) -> Arc<DaemonState> {
    test_daemon_state_at(pass_through, token, config, "http://127.0.0.1:47632".into())
}

pub(super) fn test_daemon_state_at(
    pass_through: bool,
    _token: &str,
    config: GatewayConfig,
    public_origin: String,
) -> Arc<DaemonState> {
    let generation_directory = tempfile::tempdir().expect("generation directory");
    let generation_path = generation_directory
        .keep()
        .join("active-worker-generations.json");
    Arc::new(DaemonState {
        sockets: socket::Hub::default(),
        registry: Registry::new(pass_through),
        descriptor: crate::daemon::common::control::descriptor(ComponentRole::Daemon),
        instance_id: "public-proxy-test-daemon".into(),
        pass_through,
        public_origin,
        config,
        upstream: pooled_client().expect("daemon client"),
        worker_clients: WorkerClientPool::new().expect("worker clients"),
        challenges: Mutex::new(HashMap::new()),
        activations: Mutex::new(HashMap::new()),
        mcp_sessions: Mutex::new(HashMap::new()),
        worker_sessions: Mutex::new(HashMap::new()),
        pending_directives: Mutex::new(HashMap::new()),
        active_worker_generations: ActiveWorkerGenerations::load_for_test(generation_path)
            .expect("generation state"),
        worker_generation_publication: Mutex::new(()),
        identity: MachineIdentity::generate()
            .expect("daemon identity")
            .identity,
    })
}

#[test]
fn registration_requires_the_complete_lossless_transport_capability_set() {
    let complete = crate::daemon::common::control::descriptor(ComponentRole::Mcp);
    assert!(has_required_transport_capabilities(&complete));

    let missing_trailers = crate::daemon::common::protocol::ComponentDescriptor::nemo_relay(
        ComponentRole::Mcp,
        crate::daemon::common::protocol::ProtocolRange::default(),
        Capabilities::new(["http1", "http2", "streaming_body_frames", "sse_passthrough"])
            .expect("capabilities"),
        "future-version",
    );
    assert!(!has_required_transport_capabilities(&missing_trailers));
}

#[test]
fn pending_challenge_storage_is_bounded_and_prunes_expired_entries() {
    let generated = MachineIdentity::generate().expect("identity");
    let identity = generated.identity;
    let descriptor = crate::daemon::common::control::descriptor(ComponentRole::Mcp);
    let mut challenges = HashMap::new();
    for index in 0..MAX_PENDING_CHALLENGES {
        let record = ChallengeRecord::generate(100, 100).expect("challenge");
        let challenge = record.challenge();
        challenges.insert(
            challenge.id,
            PendingChallenge {
                request: ChallengeRequest {
                    initiator: descriptor.clone(),
                    initiator_instance_id: format!("mcp-{index}"),
                    initiator_public_identity: identity.public_identity(),
                    initiator_fingerprint: identity.fingerprint(),
                    initiator_nonce: challenge.nonce,
                },
                record,
            },
        );
    }
    assert_eq!(challenges.len(), MAX_PENDING_CHALLENGES);
    assert!(!reserve_challenge_slot(
        &mut challenges,
        199,
        ComponentRole::Mcp
    ));
    assert!(reserve_challenge_slot(
        &mut challenges,
        200,
        ComponentRole::Mcp
    ));
    assert!(challenges.is_empty());
}

#[test]
fn active_mcp_registration_reuses_its_session_credential() {
    let identity = MachineIdentity::generate().expect("identity").identity;
    let fingerprint = identity.fingerprint();
    let token_digest = TokenDigest::from_token(b"route-token");
    let original = SensitiveString::new("original-session-secret").expect("secret");
    let mut sessions = HashMap::new();
    sessions.insert(
        "mcp-session".to_owned(),
        McpControlSession {
            fingerprint,
            token_digest,
            secret: original.clone(),
            secret_digest: TokenDigest::from_token(original.expose().as_bytes()),
            lease_expires_at_unix_ms: 200,
            last_sequence: 0,
            last_request_id: String::new(),
            worker_network: worker_network(),
            released: false,
        },
    );

    let (selected, reused) = select_mcp_session_token(
        &sessions,
        "mcp-session",
        fingerprint,
        token_digest,
        worker_network(),
        199,
        SensitiveString::new("must-not-rotate").expect("secret"),
    )
    .expect("selection");
    assert!(reused);
    assert_eq!(selected, original);

    let (selected, reused) = select_mcp_session_token(
        &sessions,
        "mcp-session",
        fingerprint,
        token_digest,
        worker_network(),
        200,
        SensitiveString::new("fresh-after-expiry").expect("secret"),
    )
    .expect("expired selection");
    assert!(!reused);
    assert_eq!(selected.expose(), "fresh-after-expiry");
}

#[test]
fn mcp_session_reuse_rejects_identity_token_and_network_rebinding() {
    let identity = MachineIdentity::generate().unwrap().identity;
    let fingerprint = identity.fingerprint();
    let token = TokenDigest::from_token(b"route-token");
    let secret = SensitiveString::new("session-secret").unwrap();
    let sessions = HashMap::from([(
        "mcp".into(),
        McpControlSession {
            fingerprint,
            token_digest: token,
            secret: secret.clone(),
            secret_digest: TokenDigest::from_token(secret.expose().as_bytes()),
            lease_expires_at_unix_ms: 200,
            last_sequence: 0,
            last_request_id: String::new(),
            worker_network: worker_network(),
            released: false,
        },
    )]);
    for (candidate_fingerprint, candidate_token, candidate_network) in [
        (
            MachineIdentity::generate().unwrap().identity.fingerprint(),
            token,
            worker_network(),
        ),
        (
            fingerprint,
            TokenDigest::from_token(b"different"),
            worker_network(),
        ),
        (
            fingerprint,
            token,
            WorkerNetworkHint {
                advertised_host: "worker.example".into(),
                port: Some(443),
            },
        ),
    ] {
        assert!(
            select_mcp_session_token(
                &sessions,
                "mcp",
                candidate_fingerprint,
                candidate_token,
                candidate_network,
                100,
                SensitiveString::new("fresh").unwrap(),
            )
            .is_err()
        );
    }
}

#[test]
fn authenticated_sequence_rejects_tampering_and_out_of_order_messages() {
    let secret = SensitiveString::new("session-secret").unwrap();
    let digest = TokenDigest::from_token(secret.expose().as_bytes());
    let mut sequence = 0;
    let mut request_id = String::new();
    let request =
        SessionRequest::new("session".into(), secret.clone(), 1, EmptyPayload::default()).unwrap();
    assert!(!authenticate_sequence(digest, &mut sequence, &mut request_id, &request).unwrap());
    assert!(authenticate_sequence(digest, &mut sequence, &mut request_id, &request).unwrap());

    let wrong = SessionRequest::new(
        "session".into(),
        SensitiveString::new("wrong").unwrap(),
        2,
        EmptyPayload::default(),
    )
    .unwrap();
    assert!(authenticate_sequence(digest, &mut sequence, &mut request_id, &wrong).is_err());
    let out_of_order =
        SessionRequest::new("session".into(), secret, 3, EmptyPayload::default()).unwrap();
    assert!(authenticate_sequence(digest, &mut sequence, &mut request_id, &out_of_order).is_err());
}

#[test]
fn registry_errors_map_to_stable_control_statuses() {
    for error in [
        RegistryError::UnknownRoute,
        RegistryError::UnknownMcpSession,
        RegistryError::TokenAlreadyBound,
        RegistryError::FingerprintTokenMismatch,
        RegistryError::ActivationMismatch,
        RegistryError::WorkerMismatch,
        RegistryError::RecoveryNotAuthorized,
        RegistryError::RecoveryGenerationChanged,
        RegistryError::RouteCapacityReached,
        RegistryError::McpReferenceCapacityReached,
    ] {
        assert!(registry_error(error).status().is_client_error());
    }
    assert_eq!(
        registry_error(RegistryError::DrainInProgress).status(),
        StatusCode::CONFLICT
    );
}

#[test]
fn expired_mcp_control_sessions_and_pending_directives_are_removed_together() {
    let identity = MachineIdentity::generate().expect("identity").identity;
    let fingerprint = identity.fingerprint();
    let token_digest = TokenDigest::from_token(b"route-token");
    let session = |lease_expires_at_unix_ms| {
        let secret = SensitiveString::new("session-secret").expect("secret");
        McpControlSession {
            fingerprint,
            token_digest,
            secret: secret.clone(),
            secret_digest: TokenDigest::from_token(secret.expose().as_bytes()),
            lease_expires_at_unix_ms,
            last_sequence: 0,
            last_request_id: String::new(),
            worker_network: worker_network(),
            released: false,
        }
    };
    let mut sessions = HashMap::from([
        ("expired".to_owned(), session(100)),
        ("live".to_owned(), session(101)),
    ]);
    let mut pending = HashMap::from([
        (
            "expired".to_owned(),
            BrokerDirective::WaitForWorker {
                retry_after_ms: 100,
            },
        ),
        ("live".to_owned(), BrokerDirective::UsePassThrough),
    ]);

    prune_expired_mcp_control_state(&mut sessions, &mut pending, 100);

    assert!(!sessions.contains_key("expired"));
    assert!(!pending.contains_key("expired"));
    assert!(sessions.contains_key("live"));
    assert!(pending.contains_key("live"));
}

fn worker_network() -> WorkerNetworkHint {
    WorkerNetworkHint {
        advertised_host: Ipv4Addr::LOCALHOST.to_string(),
        port: None,
    }
}

#[test]
fn advertised_https_is_valid_behind_a_reverse_proxy_without_native_tls() {
    let options = crate::daemon::ServerOptions {
        bind: Ipv4Addr::LOCALHOST,
        port: 8080,
        advertise_address: Some("https://relay.example.com:443".into()),
        pass_through: false,
        gateway: crate::server::GatewayOverrides::default(),
        tls_cert: None,
        tls_key: None,
    };
    assert_eq!(
        daemon_origin(&options, "127.0.0.1:8080".parse().unwrap()).expect("proxy origin"),
        "https://relay.example.com"
    );

    let native_http = crate::daemon::ServerOptions {
        tls_cert: Some("cert.pem".into()),
        tls_key: Some("key.pem".into()),
        advertise_address: Some("http://127.0.0.1:8080".into()),
        ..options
    };
    assert!(daemon_origin(&native_http, "127.0.0.1:8080".parse().unwrap()).is_err());
}

#[test]
fn daemon_origin_enforces_bind_and_tls_advertisement_contracts() {
    let local: SocketAddr = "127.0.0.1:47632".parse().unwrap();
    let loopback = ServerOptions {
        bind: Ipv4Addr::LOCALHOST,
        port: 47632,
        advertise_address: None,
        pass_through: false,
        gateway: crate::server::GatewayOverrides::default(),
        tls_cert: None,
        tls_key: None,
    };
    assert_eq!(
        daemon_origin(&loopback, local).unwrap(),
        "http://127.0.0.1:47632"
    );
    assert_eq!(
        daemon_origin(
            &ServerOptions {
                tls_cert: Some("cert".into()),
                ..loopback.clone()
            },
            local
        )
        .unwrap(),
        "https://127.0.0.1:47632"
    );
    for (address, tls, expected) in [
        ("127.0.0.1:80", false, "http://127.0.0.1"),
        ("127.0.0.1:443", true, "https://127.0.0.1"),
    ] {
        let options = ServerOptions {
            tls_cert: tls.then(|| "cert".into()),
            tls_key: tls.then(|| "key".into()),
            ..loopback.clone()
        };
        assert_eq!(
            daemon_origin(&options, address.parse().unwrap()).unwrap(),
            expected
        );
    }
    let unspecified = ServerOptions {
        bind: Ipv4Addr::UNSPECIFIED,
        ..loopback
    };
    assert!(daemon_origin(&unspecified, local).is_err());
    for origin in ["http://relay.example.com", "http://[::1]", "https://[::1]"] {
        assert!(
            daemon_origin(
                &ServerOptions {
                    advertise_address: Some(origin.into()),
                    ..unspecified.clone()
                },
                local
            )
            .is_err(),
            "public bind accepted invalid origin {origin}"
        );
    }
    assert!(
        daemon_origin(
            &ServerOptions {
                advertise_address: Some("https://127.0.0.1:443".into()),
                ..unspecified.clone()
            },
            local
        )
        .is_err()
    );
    assert_eq!(
        daemon_origin(
            &ServerOptions {
                advertise_address: Some("https://relay.example.com:443/".into()),
                ..unspecified
            },
            local
        )
        .unwrap(),
        "https://relay.example.com"
    );
}

#[tokio::test]
async fn daemon_startup_rejects_an_unpaired_tls_identity_after_initializing_state() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let state_directory = tempfile::tempdir().unwrap();
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x46_u8; 32]);
    let _environment = EnvScope::set(&[
        (ROUTE_TOKEN_ENV, Some(std::ffi::OsStr::new(&token))),
        ("XDG_CONFIG_HOME", Some(state_directory.path().as_os_str())),
    ]);
    let error = serve(ServerOptions {
        bind: Ipv4Addr::LOCALHOST,
        port: 0,
        advertise_address: None,
        pass_through: false,
        gateway: crate::server::GatewayOverrides::default(),
        tls_cert: Some(state_directory.path().join("certificate.pem")),
        tls_key: None,
    })
    .await
    .expect_err("unpaired TLS configuration");
    assert!(error.to_string().contains("must be supplied together"));
}

#[tokio::test]
async fn daemon_plain_listener_starts_and_reports_address_conflicts() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let state_directory = tempfile::tempdir().unwrap();
    let _environment = EnvScope::set(&[
        (ROUTE_TOKEN_ENV, None),
        ("XDG_CONFIG_HOME", Some(state_directory.path().as_os_str())),
    ]);
    let reserved = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let running_port = reserved.local_addr().unwrap().port();
    drop(reserved);
    let options = ServerOptions {
        bind: Ipv4Addr::LOCALHOST,
        port: running_port,
        advertise_address: None,
        pass_through: true,
        gateway: crate::server::GatewayOverrides::default(),
        tls_cert: None,
        tls_key: None,
    };
    let running = tokio::spawn(serve(options));
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Ok(response) =
                reqwest::get(format!("http://127.0.0.1:{running_port}/unknown")).await
            {
                assert_eq!(response.status(), StatusCode::NOT_FOUND);
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("daemon starts");
    running.abort();
    let _ = running.await;

    let occupied = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = occupied.local_addr().unwrap().port();
    let error = serve(ServerOptions {
        port,
        ..ServerOptions {
            bind: Ipv4Addr::LOCALHOST,
            port: 0,
            advertise_address: None,
            pass_through: true,
            gateway: crate::server::GatewayOverrides::default(),
            tls_cert: None,
            tls_key: None,
        }
    })
    .await
    .expect_err("occupied daemon port");
    assert!(error.to_string().contains("failed to bind daemon listener"));
}

#[tokio::test]
async fn challenge_endpoint_rejects_incompatible_and_forged_components() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x41_u8; 32]);
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    let identity = MachineIdentity::generate().unwrap().identity;
    let valid = ChallengeRequest {
        initiator: crate::daemon::common::control::descriptor(ComponentRole::Mcp),
        initiator_instance_id: "mcp-instance".into(),
        initiator_public_identity: identity.public_identity(),
        initiator_fingerprint: identity.fingerprint(),
        initiator_nonce: ChallengeRecord::generate(1, 1).unwrap().challenge().nonce,
    };

    let mut incompatible = valid.clone();
    incompatible.initiator = crate::daemon::common::protocol::ComponentDescriptor::nemo_relay(
        ComponentRole::Mcp,
        crate::daemon::common::protocol::ProtocolRange::default(),
        Capabilities::new(["http1"]).unwrap(),
        "test",
    );
    assert_eq!(
        issue_challenge(State(Arc::clone(&state)), Json(incompatible))
            .await
            .status(),
        StatusCode::UPGRADE_REQUIRED
    );

    let mut daemon_role = valid.clone();
    daemon_role.initiator = crate::daemon::common::control::descriptor(ComponentRole::Daemon);
    assert_eq!(
        issue_challenge(State(Arc::clone(&state)), Json(daemon_role))
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let mut forged = valid;
    forged.initiator_fingerprint = MachineIdentity::generate().unwrap().identity.fingerprint();
    assert_eq!(
        issue_challenge(State(state), Json(forged)).await.status(),
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn control_handlers_reject_unknown_sessions_and_oversized_failure_payloads() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x43_u8; 32]);
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    let secret = SensitiveString::new("unknown-session-secret").unwrap();
    let empty =
        SessionRequest::new("unknown".into(), secret.clone(), 1, EmptyPayload::default()).unwrap();
    assert_eq!(
        release_mcp(State(Arc::clone(&state)), Json(empty))
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let ready = SessionRequest::new(
        "unknown".into(),
        secret.clone(),
        1,
        WorkerReadyPayload {
            worker_id: "unknown".into(),
        },
    )
    .unwrap();
    assert_eq!(
        ready_worker(State(Arc::clone(&state)), Json(ready))
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let oversized = SessionRequest::new(
        "unknown".into(),
        secret,
        1,
        ActivationFailedPayload {
            activation_id: "x".repeat(129),
            failure_reason: WorkerActivationFailureReason::WorkerProcessSpawnFailed,
        },
    )
    .unwrap();
    assert_eq!(
        activation_failed(State(state), Json(oversized))
            .await
            .status(),
        StatusCode::BAD_REQUEST
    );
}

#[tokio::test]
async fn worker_control_handlers_bind_payload_identity_before_sequence_authentication() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x44_u8; 32]);
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    lock(&state.worker_sessions).insert("staged".into(), staged_worker_session("staged", u64::MAX));
    let secret = SensitiveString::new("control-secret").unwrap();
    let ready = SessionRequest::new(
        "staged".into(),
        secret.clone(),
        1,
        WorkerReadyPayload {
            worker_id: "different".into(),
        },
    )
    .unwrap();
    assert_eq!(
        ready_worker(State(Arc::clone(&state)), Json(ready))
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn worker_handlers_cover_bad_sequence_and_already_published_readiness() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x49_u8; 32]);
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    let mut session = staged_worker_session("staged", u64::MAX);
    session.published = true;
    lock(&state.worker_sessions).insert("staged".into(), session);

    let wrong = SessionRequest::new(
        "staged".into(),
        SensitiveString::new("wrong-secret").unwrap(),
        1,
        WorkerReadyPayload {
            worker_id: "staged".into(),
        },
    )
    .unwrap();

    assert_eq!(
        ready_worker(State(state.clone()), Json(wrong))
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );
    let ready = SessionRequest::new(
        "staged".into(),
        SensitiveString::new("control-secret").unwrap(),
        1,
        WorkerReadyPayload {
            worker_id: "staged".into(),
        },
    )
    .unwrap();
    assert_eq!(
        ready_worker(State(state.clone()), Json(ready))
            .await
            .status(),
        StatusCode::BAD_GATEWAY
    );
    assert!(lock(&state.worker_sessions).contains_key("staged"));
}

#[tokio::test]
async fn worker_registration_and_recovery_fail_closed_at_each_authority_boundary() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x4a_u8; 32]);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let state = test_daemon_state_at(false, &token, GatewayConfig::default(), origin.clone());
    let server = tokio::spawn({
        let state = Arc::clone(&state);
        async move { axum::serve(listener, router(state)).await.unwrap() }
    });
    let client = control_client().unwrap();
    let worker = MachineIdentity::generate().unwrap().identity;

    let proof = begin_handshake(
        &client,
        &origin,
        ComponentRole::Worker,
        &worker,
        "unknown-activation",
        None,
    )
    .await
    .unwrap()
    .proof;
    let unknown = WorkerRegisterRequest {
        proof,
        worker_id: "unknown-activation".into(),
        endpoint: "http://127.0.0.1:41000".into(),
        activation_id: "missing".into(),
        activation_token: SensitiveString::new("missing-token").unwrap(),
        tls_root_certificate: None,
    };
    assert_eq!(
        register_worker(State(Arc::clone(&state)), Json(unknown))
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let proof = begin_handshake(
        &client,
        &origin,
        ComponentRole::Worker,
        &worker,
        "wrong-secret",
        None,
    )
    .await
    .unwrap()
    .proof;
    lock(&state.activations).insert(
        "wrong-secret".into(),
        Activation {
            fingerprint: worker.fingerprint(),
            secret_digest: TokenDigest::from_token(b"correct-token"),
            deadline_unix_ms: u64::MAX,
            consumed: false,
            bind_ip: Ipv4Addr::LOCALHOST,
            port: 0,
            advertise_address: None,
        },
    );
    let wrong_secret = WorkerRegisterRequest {
        proof,
        worker_id: "wrong-secret".into(),
        endpoint: "http://127.0.0.1:41000".into(),
        activation_id: "wrong-secret".into(),
        activation_token: SensitiveString::new("incorrect-token").unwrap(),
        tls_root_certificate: None,
    };
    assert_eq!(
        register_worker(State(Arc::clone(&state)), Json(wrong_secret))
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let proof = begin_handshake(
        &client,
        &origin,
        ComponentRole::Worker,
        &worker,
        "endpoint-mismatch",
        None,
    )
    .await
    .unwrap()
    .proof;
    lock(&state.activations).insert(
        "endpoint-mismatch".into(),
        Activation {
            fingerprint: worker.fingerprint(),
            secret_digest: TokenDigest::from_token(b"activation-token"),
            deadline_unix_ms: u64::MAX,
            consumed: false,
            bind_ip: Ipv4Addr::LOCALHOST,
            port: 41000,
            advertise_address: None,
        },
    );
    let mismatched = WorkerRegisterRequest {
        proof,
        worker_id: "endpoint-mismatch".into(),
        endpoint: "http://127.0.0.1:41001".into(),
        activation_id: "endpoint-mismatch".into(),
        activation_token: SensitiveString::new("activation-token").unwrap(),
        tls_root_certificate: None,
    };
    assert_eq!(
        register_worker(State(Arc::clone(&state)), Json(mismatched))
            .await
            .status(),
        StatusCode::BAD_REQUEST
    );

    lock(&state.activations).insert(
        "replay-activation".into(),
        Activation {
            fingerprint: worker.fingerprint(),
            secret_digest: TokenDigest::from_token(b"replay-token"),
            deadline_unix_ms: u64::MAX,
            consumed: false,
            bind_ip: Ipv4Addr::LOCALHOST,
            port: 41002,
            advertise_address: None,
        },
    );
    for worker_id in ["replay-worker", "replay-worker"] {
        let proof = begin_handshake(
            &client,
            &origin,
            ComponentRole::Worker,
            &worker,
            worker_id,
            None,
        )
        .await
        .unwrap()
        .proof;
        let replay = WorkerRegisterRequest {
            proof,
            worker_id: worker_id.into(),
            endpoint: "http://127.0.0.1:41002".into(),
            activation_id: "replay-activation".into(),
            activation_token: SensitiveString::new("replay-token").unwrap(),
            tls_root_certificate: None,
        };
        assert!(
            register_worker(State(Arc::clone(&state)), Json(replay))
                .await
                .status()
                .is_success()
        );
    }
    let proof = begin_handshake(
        &client,
        &origin,
        ComponentRole::Worker,
        &worker,
        "activation-thief",
        None,
    )
    .await
    .unwrap()
    .proof;
    let stolen = WorkerRegisterRequest {
        proof,
        worker_id: "activation-thief".into(),
        endpoint: "http://127.0.0.1:41002".into(),
        activation_id: "replay-activation".into(),
        activation_token: SensitiveString::new("replay-token").unwrap(),
        tls_root_certificate: None,
    };
    assert_eq!(
        register_worker(State(Arc::clone(&state)), Json(stolen))
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let proof = begin_handshake(
        &client,
        &origin,
        ComponentRole::Worker,
        &worker,
        "invalid-recovery",
        None,
    )
    .await
    .unwrap()
    .proof;
    let foreign = MachineIdentity::generate().unwrap().identity;
    let invalid_grant = WorkerGenerationGrant::issue(
        "invalid-recovery",
        worker.fingerprint(),
        "http://127.0.0.1:41000",
        None,
        &foreign,
    )
    .unwrap();
    let invalid_recovery = WorkerRecoverRequest {
        proof,
        worker_id: "invalid-recovery".into(),
        endpoint: "http://127.0.0.1:41000".into(),
        tls_root_certificate: None,
        generation_grant: invalid_grant,
    };
    assert_eq!(
        recover_worker(State(Arc::clone(&state)), Json(invalid_recovery))
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let proof = begin_handshake(
        &client,
        &origin,
        ComponentRole::Worker,
        &worker,
        "revoked-recovery",
        None,
    )
    .await
    .unwrap()
    .proof;
    let revoked_grant = WorkerGenerationGrant::issue(
        "revoked-recovery",
        worker.fingerprint(),
        "http://127.0.0.1:41000",
        None,
        &state.identity,
    )
    .unwrap();
    let revoked = WorkerRecoverRequest {
        proof,
        worker_id: "revoked-recovery".into(),
        endpoint: "http://127.0.0.1:41000".into(),
        tls_root_certificate: None,
        generation_grant: revoked_grant,
    };
    assert_eq!(
        recover_worker(State(state), Json(revoked)).await.status(),
        StatusCode::UNAUTHORIZED
    );
    server.abort();
}

#[tokio::test]
async fn mcp_registration_rejects_credential_and_network_proof_mismatches_before_routing() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x4d_u8; 32]);
    let credential = RouteCredential::parse(token.clone()).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let state = test_daemon_state_at(false, &token, GatewayConfig::default(), origin.clone());
    let server = tokio::spawn({
        let state = Arc::clone(&state);
        async move { axum::serve(listener, router(state)).await.unwrap() }
    });
    let client = control_client().unwrap();
    let identity = MachineIdentity::generate().unwrap().identity;
    let handshake = begin_handshake(
        &client,
        &origin,
        ComponentRole::Mcp,
        &identity,
        "mcp-boundary",
        Some(credential.digest()),
    )
    .await
    .unwrap();
    let hint = WorkerNetworkHint::new("127.0.0.1", None).unwrap();
    let hint_proof = WorkerNetworkHintProof::sign(
        hint,
        &handshake.proof.transcript.daemon_target,
        "mcp-boundary",
        &handshake.proof.transcript.challenge_id,
        &identity.fingerprint(),
        &identity,
    )
    .unwrap();
    let request = McpRegisterRequest {
        proof: handshake.proof.clone(),
        worker_network: hint_proof,
    };
    assert_eq!(
        register_mcp(
            State(Arc::clone(&state)),
            HeaderMap::new(),
            Json(request.clone()),
        )
        .await
        .status(),
        StatusCode::UNAUTHORIZED
    );
    let other = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x4e_u8; 32]);
    let headers = HeaderMap::from_iter([(
        HeaderName::from_static(CLIENT_TOKEN_HEADER),
        HeaderValue::from_str(&other).unwrap(),
    )]);
    assert_eq!(
        register_mcp(State(Arc::clone(&state)), headers, Json(request.clone()),)
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let headers = HeaderMap::from_iter([(
        HeaderName::from_static(CLIENT_TOKEN_HEADER),
        HeaderValue::from_str(&token).unwrap(),
    )]);
    let mut mismatched = request.clone();
    mismatched.proof.transcript.route_token_digest = None;
    assert_eq!(
        register_mcp(State(Arc::clone(&state)), headers.clone(), Json(mismatched),)
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let mut bad_network = request;
    bad_network.worker_network.hint.advertised_host = "localhost".into();
    assert_eq!(
        register_mcp(State(Arc::clone(&state)), headers, Json(bad_network))
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );
    server.abort();
}

#[tokio::test]
async fn worker_readiness_probe_failure_removes_the_staged_session() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x4b_u8; 32]);
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    let worker = MachineIdentity::generate().unwrap().identity;
    let fingerprint = worker.fingerprint();
    let daemon_proof = HandshakeProof {
        signer: ComponentRole::Daemon,
        signature: state.identity.sign(b"readiness-failure"),
    };
    let response = stage_worker(
        &state,
        fingerprint,
        "unreachable-worker".into(),
        "http://127.0.0.1:1".into(),
        None,
        None,
        WorkerPublication::Activation {
            activation_id: "unreachable-activation".into(),
        },
        daemon_proof,
    );
    assert_eq!(response.status(), StatusCode::OK);
    let registration: WorkerRegisterResponse =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let ready = SessionRequest::new(
        "unreachable-worker".into(),
        registration.session_token,
        1,
        WorkerReadyPayload {
            worker_id: "unreachable-worker".into(),
        },
    )
    .unwrap();
    assert_eq!(
        ready_worker(State(Arc::clone(&state)), Json(ready))
            .await
            .status(),
        StatusCode::BAD_GATEWAY
    );
    assert!(!lock(&state.worker_sessions).contains_key("unreachable-worker"));
}

#[tokio::test]
async fn native_tls_configuration_serves_a_daemon_request_with_pinned_trust() {
    let rcgen::CertifiedKey { cert, key_pair } =
        rcgen::generate_simple_self_signed(vec!["127.0.0.1".into()]).expect("certificate");
    let directory = tempfile::tempdir().expect("temporary TLS directory");
    let certificate_path = directory.path().join("daemon.crt");
    let key_path = directory.path().join("daemon.pk8");
    std::fs::write(&certificate_path, cert.pem()).expect("write certificate");
    std::fs::write(&key_path, key_pair.serialize_pem()).expect("write key");
    let config = load_tls_config(&certificate_path, &key_path).expect("daemon TLS config");
    assert_eq!(
        config.alpn_protocols,
        [b"h2".to_vec(), b"http/1.1".to_vec()]
    );

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind daemon TLS listener");
    let address = listener.local_addr().expect("daemon TLS address");
    let app = Router::new().route(
        "/probe",
        axum::routing::get(|| async { StatusCode::NO_CONTENT }),
    );
    let server = tokio::spawn(serve_tls(listener, app, config));

    let root = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(cert.der());
    let client = pooled_worker_tls_client(&root).expect("pinned TLS client");
    let request = Request::get(format!("https://127.0.0.1:{}/probe", address.port()))
        .body(box_body(http_body_util::Empty::<Bytes>::new()))
        .expect("probe request");
    let response = client.request(request).await.expect("daemon TLS response");
    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    server.abort();
    assert!(
        server
            .await
            .expect_err("TLS server is stopped")
            .is_cancelled()
    );
}

#[test]
fn public_ingress_keeps_only_authenticated_provider_routing_metadata() {
    let mut headers = HeaderMap::new();
    headers.insert(CLIENT_TOKEN_HEADER, HeaderValue::from_static("route"));
    headers.insert(
        "x-nemo-relay-internal-dispatch-url",
        HeaderValue::from_static("http://attacker.invalid"),
    );
    headers.insert(WORKER_TOKEN_HEADER, HeaderValue::from_static("attacker"));
    headers.insert(
        "x-nemo-relay-bootstrap-proof",
        HeaderValue::from_static("attacker"),
    );
    headers.insert(
        crate::agents::pi::alignment::UPSTREAM_BASE_URL_HEADER,
        HeaderValue::from_static("https://custom.example/v1"),
    );
    headers.insert(
        crate::operational::OPERATION_ID_HEADER,
        HeaderValue::from_static("018f0f3f-3f7a-7b72-9d0d-e0d8ced91d8b"),
    );
    let mut hook_headers = headers.clone();
    strip_public_relay_headers(&mut headers, PublicRoute::Provider(ProviderRoute::OpenAi));
    assert!(headers.contains_key(CLIENT_TOKEN_HEADER));
    assert!(headers.contains_key(crate::agents::pi::alignment::UPSTREAM_BASE_URL_HEADER));
    assert!(!headers.contains_key("x-nemo-relay-internal-dispatch-url"));
    assert!(!headers.contains_key(WORKER_TOKEN_HEADER));
    assert!(!headers.contains_key("x-nemo-relay-bootstrap-proof"));
    assert!(!headers.contains_key(crate::operational::OPERATION_ID_HEADER));

    strip_public_relay_headers(
        &mut hook_headers,
        PublicRoute::Hook(crate::daemon::common::routes::HookRoute::Pi),
    );
    assert!(!hook_headers.contains_key(crate::agents::pi::alignment::UPSTREAM_BASE_URL_HEADER));
    assert!(hook_headers.contains_key(crate::operational::OPERATION_ID_HEADER));

    hook_headers.insert(
        crate::operational::OPERATION_ID_HEADER,
        HeaderValue::from_static("untrusted-native-text"),
    );
    strip_public_relay_headers(
        &mut hook_headers,
        PublicRoute::Hook(crate::daemon::common::routes::HookRoute::Pi),
    );
    assert!(!hook_headers.contains_key(crate::operational::OPERATION_ID_HEADER));
}

#[test]
fn pass_through_provider_auth_preserves_callers_and_fills_missing_configured_auth() {
    let config = GatewayConfig {
        openai_auth_header: Some("Bearer configured".into()),
        ..GatewayConfig::default()
    };
    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer caller"));
    inject_provider_auth(&mut headers, ProviderRoute::OpenAi, &config);
    assert_eq!(
        headers.get(AUTHORIZATION).expect("caller auth"),
        "Bearer caller"
    );

    headers.remove(AUTHORIZATION);
    inject_provider_auth(&mut headers, ProviderRoute::OpenAi, &config);
    assert_eq!(
        headers.get(AUTHORIZATION).expect("configured auth"),
        "Bearer configured"
    );
}

#[test]
fn worker_route_failure_signal_is_consumed_before_the_public_response() {
    let mut response = Response::new(Body::empty());
    response.headers_mut().insert(
        WORKER_ROUTE_FAILURE_HEADER,
        HeaderValue::from_static("pass-through"),
    );

    assert!(take_worker_route_failure(&mut response));
    assert!(!response.headers().contains_key(WORKER_ROUTE_FAILURE_HEADER));
    assert!(!take_worker_route_failure(&mut response));
}

#[test]
fn activation_endpoint_is_bound_to_signed_worker_network_policy() {
    let activation = Activation {
        fingerprint: MachineIdentity::generate().unwrap().identity.fingerprint(),
        secret_digest: TokenDigest::from_token(b"secret"),
        deadline_unix_ms: u64::MAX,
        consumed: false,
        bind_ip: Ipv4Addr::UNSPECIFIED,
        port: 9443,
        advertise_address: Some("worker.example.com".into()),
    };
    assert!(activation_endpoint_matches(
        "https://worker.example.com:9443",
        &activation
    ));
    assert!(!activation_endpoint_matches(
        "http://worker.example.com:9443",
        &activation
    ));
    assert!(!activation_endpoint_matches(
        "https://attacker.example.com:9443",
        &activation
    ));
    let mut activation = activation;
    activation.port = 443;
    assert!(activation_endpoint_matches(
        "https://worker.example.com:443",
        &activation
    ));
    assert!(!activation_endpoint_matches(
        "https://worker.example.com",
        &activation
    ));
    assert!(!activation_endpoint_matches(
        "https://worker.example.com:80",
        &activation
    ));
    activation.bind_ip = Ipv4Addr::LOCALHOST;
    activation.advertise_address = None;
    activation.port = 80;
    assert!(activation_endpoint_matches(
        "http://127.0.0.1:80",
        &activation
    ));
    assert!(!activation_endpoint_matches(
        "http://127.0.0.1",
        &activation
    ));
    assert!(!activation_endpoint_matches(
        "http://127.0.0.1:443",
        &activation
    ));
}

#[test]
fn released_mcp_session_credential_is_never_reused() {
    let identity = MachineIdentity::generate().expect("identity").identity;
    let fingerprint = identity.fingerprint();
    let token_digest = TokenDigest::from_token(b"route-token");
    let secret = SensitiveString::new("released-secret").expect("secret");
    let sessions = HashMap::from([(
        "released".to_owned(),
        McpControlSession {
            fingerprint,
            token_digest,
            secret: secret.clone(),
            secret_digest: TokenDigest::from_token(secret.expose().as_bytes()),
            lease_expires_at_unix_ms: 1_000,
            last_sequence: 1,
            last_request_id: "release-request".into(),
            worker_network: worker_network(),
            released: true,
        },
    )]);
    let fresh = SensitiveString::new("fresh-secret").expect("fresh");
    let (selected, reused) = select_mcp_session_token(
        &sessions,
        "released",
        fingerprint,
        token_digest,
        worker_network(),
        999,
        fresh.clone(),
    )
    .expect("fresh selection");
    assert!(!reused);
    assert_eq!(selected, fresh);
}

#[test]
fn staged_worker_sessions_are_bounded_pruned_and_collision_safe() {
    let mut sessions = HashMap::from([("staged".into(), staged_worker_session("staged", 100))]);
    assert!(!reserve_worker_session_slot(&mut sessions, 99, "other", 1));
    assert!(!reserve_worker_session_slot(&mut sessions, 99, "staged", 2));
    assert!(reserve_worker_session_slot(&mut sessions, 100, "other", 1));
    assert!(sessions.is_empty());
}

#[test]
fn worker_staging_and_replay_are_bound_to_endpoint_and_publication() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x45_u8; 32]);
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    let fingerprint = MachineIdentity::generate().unwrap().identity.fingerprint();
    let proof = HandshakeProof {
        signer: ComponentRole::Daemon,
        signature: state.identity.sign(b"test-daemon-proof"),
    };
    let publication = WorkerPublication::Activation {
        activation_id: "activation".into(),
    };
    assert_eq!(
        stage_worker(
            &state,
            fingerprint,
            String::new(),
            "http://127.0.0.1:41000".into(),
            None,
            None,
            publication.clone(),
            proof.clone(),
        )
        .status(),
        StatusCode::BAD_REQUEST
    );
    assert!(
        stage_worker(
            &state,
            fingerprint,
            "worker".into(),
            "http://127.0.0.1:41000".into(),
            None,
            None,
            publication.clone(),
            proof.clone(),
        )
        .status()
        .is_success()
    );
    assert!(
        replay_worker_registration(
            &state,
            fingerprint,
            "worker",
            "http://127.0.0.1:41000",
            None,
            Some("activation"),
            None,
            proof.clone(),
        )
        .is_some()
    );
    assert!(
        replay_worker_registration(
            &state,
            fingerprint,
            "worker",
            "http://127.0.0.1:41001",
            None,
            Some("activation"),
            None,
            proof.clone(),
        )
        .is_none()
    );
    assert_eq!(
        stage_worker(
            &state,
            fingerprint,
            "worker".into(),
            "http://127.0.0.1:41000".into(),
            None,
            None,
            publication,
            proof,
        )
        .status(),
        StatusCode::TOO_MANY_REQUESTS
    );
}

fn staged_worker_session(worker_id: &str, lease_expires_at_unix_ms: u64) -> WorkerControlSession {
    let worker = MachineIdentity::generate()
        .expect("worker identity")
        .identity;
    let daemon = MachineIdentity::generate()
        .expect("daemon identity")
        .identity;
    let endpoint = "http://127.0.0.1:41000";
    let secret = SensitiveString::new("control-secret").expect("control secret");
    let data = SensitiveString::new("data-secret").expect("data secret");
    WorkerControlSession {
        fingerprint: worker.fingerprint(),
        worker_id: worker_id.into(),
        secret: secret.clone(),
        secret_digest: TokenDigest::from_token(secret.expose().as_bytes()),
        last_sequence: 0,
        last_request_id: String::new(),
        lease_expires_at_unix_ms,
        pending_target: Arc::new(
            WorkerTarget::new(worker_id, endpoint, data).expect("worker target"),
        ),
        publication: WorkerPublication::Activation {
            activation_id: "activation".into(),
        },
        published: false,
        generation_grant: WorkerGenerationGrant::issue(
            worker_id,
            worker.fingerprint(),
            endpoint,
            None,
            &daemon,
        )
        .expect("generation grant"),
    }
}

#[test]
fn worker_generation_snapshot_releases_the_session_lock() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x73_u8; 32]);
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    let session = staged_worker_session("snapshot-worker", u64::MAX);
    let expected = (
        session.fingerprint,
        session.generation_grant.generation_id.clone(),
    );
    lock(&state.worker_sessions).insert("snapshot-worker".into(), session);

    let snapshot = worker_generation(&state, "snapshot-worker").unwrap();
    let mut sessions = state
        .worker_sessions
        .try_lock()
        .expect("snapshot released lock");
    sessions.remove("snapshot-worker");
    assert_eq!(
        snapshot, expected,
        "snapshot owns its generation independently"
    );
}

#[test]
fn communication_failure_invalidates_route_without_waiting_for_durable_revocation() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x74_u8; 32]);
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    let session = staged_worker_session("failed-worker", u64::MAX);
    let fingerprint = session.fingerprint;
    let generation = session.generation_grant.generation_id.clone();
    let digest = RouteCredential::parse(token).unwrap().digest();
    let launch = fresh_launch(WorkerNetworkHint::new("127.0.0.1", None).unwrap()).unwrap();
    let activation_id = launch.activation_id.clone();
    state
        .registry
        .register_mcp(
            McpRegistration {
                fingerprint,
                token_digest: digest,
                session_id: McpSessionId::new("failure-mcp").unwrap(),
                lease_expires_at_unix_ms: u64::MAX,
            },
            launch,
        )
        .unwrap();
    state
        .registry
        .mark_worker_ready(
            fingerprint,
            &activation_id,
            Arc::clone(&session.pending_target),
        )
        .unwrap();
    state
        .active_worker_generations
        .publish(fingerprint, &generation)
        .unwrap();
    lock(&state.worker_sessions).insert("failed-worker".into(), session);

    // A publication transaction may be stalled on disk. Failure handling on a body-polling
    // thread must nevertheless return and invalidate the route before that transaction ends.
    let publication = lock(&state.worker_generation_publication);
    let (sent, received) = std::sync::mpsc::channel();
    let failure_state = Arc::clone(&state);
    let task = runtime.spawn_blocking(move || {
        handle_worker_communication_failure(&failure_state, fingerprint, "failed-worker");
        sent.send(()).unwrap();
    });
    let completed = received.recv_timeout(Duration::from_secs(5));
    let route = state.registry.resolve_target(&digest);
    let session_removed = state
        .worker_sessions
        .try_lock()
        .is_ok_and(|sessions| !sessions.contains_key("failed-worker"));
    // Always release contention before asserting, so a regression cannot strand runtime shutdown.
    drop(publication);
    runtime.block_on(task).unwrap();
    completed.expect("failure handling waited for the publication lock");
    assert!(matches!(route, Ok(ResolvedTarget::PassThrough)));
    assert!(session_removed);
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(5), async {
            while state
                .active_worker_generations
                .matches(fingerprint, &generation)
                .unwrap()
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("generation was not durably revoked");
    });
}

#[tokio::test]
async fn release_actions_revoke_activation_transfer_directives_and_ignore_absent_nominees() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x4d_u8; 32]);
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    let fingerprint = MachineIdentity::generate().unwrap().identity.fingerprint();
    let launch = fresh_launch(WorkerNetworkHint::new("127.0.0.1", None).unwrap()).unwrap();
    let activation_id = launch.activation_id.clone();
    let directive = launch.into_directive();
    remember_activation(&state, fingerprint, &directive);
    lock(&state.pending_directives).insert("launch-owner".into(), directive);

    handle_release_action(
        Arc::clone(&state),
        fingerprint,
        ReleaseAction::CancelActivation {
            activation_id: activation_id.clone(),
        },
    );
    assert!(!lock(&state.activations).contains_key(&activation_id));
    assert!(!lock(&state.pending_directives).contains_key("launch-owner"));

    let session_id = McpSessionId::new("transfer-owner").unwrap();
    handle_release_action(
        Arc::clone(&state),
        fingerprint,
        ReleaseAction::TransferActivation {
            session_id: session_id.clone(),
            directive: BrokerDirective::UsePassThrough,
        },
    );
    assert!(matches!(
        lock(&state.pending_directives).get(session_id.as_str()),
        Some(BrokerDirective::UsePassThrough)
    ));

    handle_release_action(
        Arc::clone(&state),
        fingerprint,
        ReleaseAction::NominateMcp {
            session_id: McpSessionId::new("missing-owner").unwrap(),
        },
    );
    handle_release_action(state, fingerprint, ReleaseAction::NoChange);
}

#[tokio::test]
async fn nominated_live_mcp_receives_a_fresh_relaunch_activation() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x4e_u8; 32]);
    let credential = RouteCredential::parse(token.clone()).unwrap();
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    let fingerprint = MachineIdentity::generate().unwrap().identity.fingerprint();
    let session_id = McpSessionId::new("relaunch-owner").unwrap();
    let activation_id = match state
        .registry
        .register_mcp(
            McpRegistration {
                fingerprint,
                token_digest: credential.digest(),
                session_id: session_id.clone(),
                lease_expires_at_unix_ms: u64::MAX,
            },
            fresh_launch(worker_network()).unwrap(),
        )
        .unwrap()
    {
        BrokerDirective::LaunchWorker { activation_id, .. } => activation_id,
        directive => panic!("unexpected directive: {directive:?}"),
    };
    let target = Arc::new(
        WorkerTarget::new(
            "failed-worker",
            "http://127.0.0.1:41001",
            SensitiveString::new("worker-secret").unwrap(),
        )
        .unwrap(),
    );
    state
        .registry
        .mark_worker_ready(fingerprint, &activation_id, target)
        .unwrap();
    let action = state
        .registry
        .worker_failed(fingerprint, "failed-worker", u64::MAX)
        .unwrap();
    let nominee = match action {
        WorkerFailureAction::NominateMcp { session_id } => session_id,
        WorkerFailureAction::RouteEmpty => panic!("route unexpectedly empty"),
    };
    let secret = SensitiveString::new("mcp-secret").unwrap();
    lock(&state.mcp_sessions).insert(
        nominee.as_str().to_owned(),
        McpControlSession {
            fingerprint,
            token_digest: credential.digest(),
            secret: secret.clone(),
            secret_digest: TokenDigest::from_token(secret.expose().as_bytes()),
            lease_expires_at_unix_ms: u64::MAX,
            last_sequence: 0,
            last_request_id: String::new(),
            worker_network: worker_network(),
            released: false,
        },
    );

    handle_release_action(
        Arc::clone(&state),
        fingerprint,
        ReleaseAction::NominateMcp {
            session_id: nominee.clone(),
        },
    );
    assert!(matches!(
        lock(&state.pending_directives).get(nominee.as_str()),
        Some(BrokerDirective::LaunchWorker { .. })
    ));
    assert_eq!(lock(&state.activations).len(), 1);
}

#[tokio::test]
async fn worker_probe_reports_a_reachable_but_unready_endpoint() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().route(
                WORKER_PROBE_PATH,
                axum::routing::get(|| async { StatusCode::OK }),
            ),
        )
        .await
        .unwrap();
    });
    let target = Arc::new(
        WorkerTarget::new(
            "unready-worker",
            endpoint,
            SensitiveString::new("data-secret").unwrap(),
        )
        .unwrap(),
    );
    let error = probe_worker(&target).await.unwrap_err();
    assert!(error.to_string().contains("returned HTTP 200 OK"));
    server.abort();
}

#[test]
fn publication_failures_revoke_activation_and_fail_recovery_closed() {
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x4f_u8; 32]);
    let credential = RouteCredential::parse(token.clone()).unwrap();
    let state = test_daemon_state(false, &token, GatewayConfig::default());
    let fingerprint = MachineIdentity::generate().unwrap().identity.fingerprint();
    let session_id = McpSessionId::new("publication-owner").unwrap();
    let launch = WorkerLaunch {
        activation_id: "publication-activation".into(),
        activation_token: SensitiveString::new("publication-secret").unwrap(),
        deadline_unix_ms: u64::MAX,
        bind_ip: Ipv4Addr::LOCALHOST,
        port: 0,
        advertise_address: None,
    };
    let directive = state
        .registry
        .register_mcp(
            McpRegistration {
                fingerprint,
                token_digest: credential.digest(),
                session_id,
                lease_expires_at_unix_ms: u64::MAX,
            },
            launch,
        )
        .unwrap();
    remember_activation(&state, fingerprint, &directive);
    fail_worker_publication(
        &state,
        fingerprint,
        &WorkerPublication::Activation {
            activation_id: "publication-activation".into(),
        },
    );
    assert!(lock(&state.activations).is_empty());
    assert!(matches!(
        state.registry.resolve_target(&credential.digest()),
        Ok(ResolvedTarget::PassThrough)
    ));

    let unrelated = MachineIdentity::generate().unwrap().identity.fingerprint();
    fail_worker_publication(
        &state,
        unrelated,
        &WorkerPublication::Recovery {
            permit: RecoveryPermit::ExistingWorker {
                worker_id: "missing-worker".into(),
                recovering: true,
            },
        },
    );
}

#[tokio::test]
async fn readiness_rechecks_activation_and_recovery_authority_after_the_probe() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().route(
                WORKER_PROBE_PATH,
                axum::routing::get(|| async { StatusCode::NO_CONTENT }),
            ),
        )
        .await
        .unwrap();
    });
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x4c_u8; 32]);
    let state = test_daemon_state(false, &token, GatewayConfig::default());

    let mut activation = staged_worker_session("stale-activation", u64::MAX);
    activation.pending_target = Arc::new(
        WorkerTarget::new(
            "stale-activation",
            endpoint.clone(),
            SensitiveString::new("data-secret").unwrap(),
        )
        .unwrap(),
    );
    activation.publication = WorkerPublication::Activation {
        activation_id: "already-revoked".into(),
    };
    lock(&state.worker_sessions).insert("stale-activation".into(), activation);
    let ready = SessionRequest::new(
        "stale-activation".into(),
        SensitiveString::new("control-secret").unwrap(),
        1,
        WorkerReadyPayload {
            worker_id: "stale-activation".into(),
        },
    )
    .unwrap();
    assert_eq!(
        ready_worker(State(Arc::clone(&state)), Json(ready))
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let mut recovery = staged_worker_session("stale-recovery", u64::MAX);
    recovery.pending_target = Arc::new(
        WorkerTarget::new(
            "stale-recovery",
            endpoint,
            SensitiveString::new("data-secret").unwrap(),
        )
        .unwrap(),
    );
    recovery.publication = WorkerPublication::Recovery {
        permit: RecoveryPermit::ExistingWorker {
            worker_id: "stale-recovery".into(),
            recovering: true,
        },
    };
    lock(&state.worker_sessions).insert("stale-recovery".into(), recovery);
    let ready = SessionRequest::new(
        "stale-recovery".into(),
        SensitiveString::new("control-secret").unwrap(),
        1,
        WorkerReadyPayload {
            worker_id: "stale-recovery".into(),
        },
    )
    .unwrap();
    assert_eq!(
        ready_worker(State(Arc::clone(&state)), Json(ready))
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );
    server.abort();
}

fn router(state: Arc<DaemonState>) -> Router {
    super::router(state)
        .layer(axum::Extension(ConnectInfo(
            "127.0.0.1:1".parse::<SocketAddr>().unwrap(),
        )))
        .layer(axum::Extension(socket::LocalAddress(
            "127.0.0.1:2".parse().unwrap(),
        )))
}
