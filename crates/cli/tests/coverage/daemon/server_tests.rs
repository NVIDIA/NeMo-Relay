// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use base64::Engine;

use super::*;
use crate::daemon::common::worker_tls::pooled_worker_tls_client;

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
}

#[test]
fn administrator_token_file_is_hashed_into_the_daemon_allowlist() {
    let directory = tempfile::tempdir().expect("temporary allowlist directory");
    let path = directory.path().join("client-tokens");
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x42_u8; 32]);
    std::fs::write(&path, format!("# managed credentials\n{token}\n")).expect("write allowlist");
    let allowed = load_allowed_route_tokens(Some(&path)).expect("load allowlist");
    assert!(allowed.contains(&TokenDigest::from_token(token.as_bytes())));
    assert!(!allowed.contains(&TokenDigest::from_token(b"not-authorized")));
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
            last_heartbeat: None,
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
fn duplicate_heartbeat_replays_the_exact_cached_directive() {
    let identity = MachineIdentity::generate().expect("identity").identity;
    let secret = SensitiveString::new("session-secret").expect("secret");
    let mut request = SessionRequest::new(
        "mcp-session".to_owned(),
        secret.clone(),
        7,
        EmptyPayload::default(),
    )
    .expect("request");
    request.request_id = "stable-request-id".to_owned();
    let expected = McpHeartbeatResponse {
        directive: Some(BrokerDirective::WaitForWorker {
            retry_after_ms: 321,
        }),
    };
    let session = McpControlSession {
        fingerprint: identity.fingerprint(),
        token_digest: TokenDigest::from_token(b"route-token"),
        secret: secret.clone(),
        secret_digest: TokenDigest::from_token(secret.expose().as_bytes()),
        lease_expires_at_unix_ms: 1_000,
        last_sequence: request.sequence,
        last_request_id: request.request_id.clone(),
        last_heartbeat: Some(CachedHeartbeat {
            sequence: request.sequence,
            request_id: request.request_id.clone(),
            response: expected,
        }),
        worker_network: worker_network(),
        released: false,
    };

    let replayed =
        cached_heartbeat_response(&session, &request, true).expect("duplicate heartbeat response");
    assert!(matches!(
        replayed.directive,
        Some(BrokerDirective::WaitForWorker {
            retry_after_ms: 321
        })
    ));
    assert!(cached_heartbeat_response(&session, &request, false).is_none());
    request.request_id = "different-request-id".to_owned();
    assert!(cached_heartbeat_response(&session, &request, true).is_none());
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
            last_heartbeat: None,
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
        client_token_file: None,
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
    let mut hook_headers = headers.clone();
    strip_public_relay_headers(&mut headers, PublicRoute::Provider(ProviderRoute::OpenAi));
    assert!(headers.contains_key(CLIENT_TOKEN_HEADER));
    assert!(headers.contains_key(crate::agents::pi::alignment::UPSTREAM_BASE_URL_HEADER));
    assert!(!headers.contains_key("x-nemo-relay-internal-dispatch-url"));
    assert!(!headers.contains_key(WORKER_TOKEN_HEADER));
    assert!(!headers.contains_key("x-nemo-relay-bootstrap-proof"));

    strip_public_relay_headers(
        &mut hook_headers,
        PublicRoute::Hook(crate::daemon::common::routes::HookRoute::Pi),
    );
    assert!(!hook_headers.contains_key(crate::agents::pi::alignment::UPSTREAM_BASE_URL_HEADER));
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
            last_heartbeat: None,
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
        next_daemon_sequence: 0,
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
