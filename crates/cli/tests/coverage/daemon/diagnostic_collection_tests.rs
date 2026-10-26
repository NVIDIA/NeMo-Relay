// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use super::*;
use nemo_relay::diagnostics::*;

fn context() -> Diagnostics {
    Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        },
    )
    .unwrap()
}
fn broker() -> (Arc<DaemonState>, String) {
    let mut state =
        super::super::tests::test_daemon_state(false, "route-token", GatewayConfig::default());
    let mutable = Arc::get_mut(&mut state).unwrap();
    let key = crate::configuration::BootstrapChallengeKey::from_bytes(b"operator");
    let token = key.diagnostics_token();
    mutable.diagnostics_http = Some(
        crate::diagnostics_http::DiagnosticsHttp::new(context().reader(), key, vec![]).unwrap(),
    );
    mutable.collector = Some(Arc::new(
        collection::Collector::new(collection::owned_bytes()).unwrap(),
    ));
    (state, token)
}
fn session(context: &Diagnostics, endpoint: &str) -> WorkerControlSession {
    let mut session = super::super::tests::staged_worker_session("worker", u64::MAX);
    let offer = diag::DiagnosticOffer {
        source: context.source().instance_id,
        schema_version: DIAGNOSTIC_SCHEMA_VERSION,
        enabled: true,
        query_available: true,
        whole_context_read: true,
        max_page_bytes: collection::PAGE_BYTES,
    };
    session.diagnostic_grant = diag::DiagnosticReadGrant::issue(
        offer.source,
        &session.generation_grant.generation_id,
        offer.max_page_bytes,
    );
    session.diagnostic_offer = Some(offer);
    session.diagnostic_connected = true;
    session.pending_target = Arc::new(
        WorkerTarget::new(
            "worker",
            endpoint,
            SensitiveString::new("data-token").unwrap(),
        )
        .unwrap(),
    );
    session
}
async fn wrapper(state: &Arc<DaemonState>, token: &str) -> Response<Body> {
    read(
        State(state.clone()),
        AxumPath("worker".into()),
        axum::extract::OriginalUri(
            "/_nemo_relay/v1/workers/worker/diagnostics"
                .parse()
                .unwrap(),
        ),
        HeaderMap::from_iter([(
            HeaderName::from_static(AUTH_HEADER),
            HeaderValue::from_str(token).unwrap(),
        )]),
    )
    .await
}
async fn poll_session(state: &Arc<DaemonState>) {
    let collector = state.collector.as_ref().unwrap().clone();
    let (target, grant, ticket) = {
        let sessions = lock(&state.worker_sessions);
        let session = &sessions["worker"];
        (
            session.pending_target.clone(),
            session.diagnostic_grant.clone().unwrap(),
            lock(&collector.cache)
                .begin(key(session).unwrap(), Instant::now())
                .unwrap(),
        )
    };
    let buffer = collector.ingress.reserve().unwrap();
    poll(
        state.clone(),
        target,
        grant,
        PollGuard {
            collector,
            ticket: Some(ticket),
        },
        buffer,
    )
    .await;
}
#[tokio::test]
async fn daemon_collection_preserves_source_totals_and_reauthorizes_cached_pages() {
    let source = context();
    let owner = diag::DaemonDiagnostics::new(&source, "daemon_worker").unwrap();
    let _projection = source
        .subscribe_projection(|_| panic!("restricted worker detail projected"))
        .unwrap();
    for _ in 0..5 {
        owner.record(
            DaemonFailureCode::Forward,
            FailureClass::Timeout,
            "RESTRICTED",
        );
    }
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let (state, token) = broker();
    let session = session(&source, &endpoint);
    let authority = diag::WorkerReadAuthority::default();
    authority.replace(session.diagnostic_grant.clone());
    let adapter = crate::diagnostics_http::DiagnosticsHttp::with_authority(
        source.reader(),
        crate::diagnostics_http::ReadAuthority::Worker(authority),
        vec![],
    )
    .unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, adapter.router()).await.unwrap();
    });
    lock(&state.worker_sessions).insert("worker".into(), session);
    assert_eq!(
        wrapper(&state, "data-token").await.status(),
        StatusCode::FORBIDDEN
    );
    for _ in 0..2 {
        poll_session(&state).await;
        let response = wrapper(&state, &token).await;
        assert_eq!(response.status(), StatusCode::OK);
        let value: serde_json::Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(value["collection"]["latest_poll"], "succeeded");
        let page: DiagnosticSnapshot = serde_json::from_value(value["snapshot"].clone()).unwrap();
        assert_eq!(page.source, *source.source());
        assert_eq!(
            page.components
                .iter()
                .find(|c| c.identity.key == "daemon.worker.forwarding")
                .unwrap()
                .lifetime
                .totals
                .failures_total
                .value
                .get(),
            5
        );
        tokio::time::sleep(Duration::from_millis(6100)).await;
    }
    let invalid = read(
        State(state.clone()),
        AxumPath("worker".into()),
        axum::extract::OriginalUri(
            "/_nemo_relay/v1/workers/worker/diagnostics?query=private"
                .parse()
                .unwrap(),
        ),
        HeaderMap::from_iter([(
            HeaderName::from_static(AUTH_HEADER),
            HeaderValue::from_str(&token).unwrap(),
        )]),
    )
    .await;
    assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
    let collector = state.collector.as_ref().unwrap();
    let mut held = Vec::new();
    for _ in 0..collection::POLLS {
        held.push(wrapper(&state, &token).await);
    }
    let busy = wrapper(&state, &token).await;
    assert_eq!(busy.status(), StatusCode::TOO_MANY_REQUESTS);
    let body: serde_json::Value =
        serde_json::from_slice(&busy.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["error"]["reason"], "busy");
    drop(held);
    assert!(collector.responses.reserve().is_some());
    lock(&state.worker_sessions)
        .get_mut("worker")
        .unwrap()
        .diagnostic_connected = false;
    assert_eq!(
        wrapper(&state, &token).await.status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    server.abort();
}
#[tokio::test]
async fn daemon_collection_cancels_slow_polls_and_fences_replaced_sessions() {
    let source = context();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let (state, _) = broker();
    lock(&state.worker_sessions).insert("worker".into(), session(&source, &endpoint));
    let entered = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let payload = source
        .snapshot(&Default::default())
        .unwrap()
        .as_bytes()
        .to_vec();
    let app = Router::new().route(
        crate::diagnostics_http::PREFIX,
        get({
            let entered = entered.clone();
            let release = release.clone();
            move || {
                let entered = entered.clone();
                let release = release.clone();
                let payload = payload.clone();
                async move {
                    entered.notify_one();
                    release.notified().await;
                    ([(CONTENT_TYPE, "application/json")], payload)
                }
            }
        }),
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let job = tokio::spawn({
        let state = state.clone();
        async move {
            poll_session(&state).await;
        }
    });
    tokio::time::timeout(Duration::from_secs(5), entered.notified())
        .await
        .unwrap();
    job.abort();
    let _ = job.await;
    // A cancelled job relinquishes both its single-flight ticket and ingress buffer.
    let job = tokio::spawn({
        let state = state.clone();
        async move {
            poll_session(&state).await;
        }
    });
    tokio::time::timeout(Duration::from_secs(5), entered.notified())
        .await
        .unwrap();
    let old_key = {
        let sessions = lock(&state.worker_sessions);
        key(&sessions["worker"]).unwrap()
    };
    {
        let mut sessions = lock(&state.worker_sessions);
        let session = sessions.get_mut("worker").unwrap();
        session.diagnostic_grant = diag::DiagnosticReadGrant::issue(
            source.source().instance_id,
            &session.generation_grant.generation_id,
            collection::PAGE_BYTES,
        );
    }
    release.notify_waiters();
    job.await.unwrap();
    let collector = state.collector.as_ref().unwrap();
    let mut response = collector.responses.reserve().unwrap();
    lock(&collector.cache)
        .write_page(&old_key, Instant::now(), &mut response)
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&response.bytes).unwrap();
    assert!(value["snapshot"].is_null());
    let buffers = (0..collection::POLLS)
        .map(|_| collector.ingress.reserve().unwrap())
        .collect::<Vec<_>>();
    assert!(collector.ingress.reserve().is_none());
    drop(buffers);
    server.abort();
}
#[test]
fn daemon_collection_negotiation_is_optional_and_budget_checked() {
    let (state, _) = broker();
    let source = context();
    let mut session = session(&source, "http://127.0.0.1:1");
    assert!(grant(&state, None, &session.generation_grant.generation_id).is_none());
    let mut offer = session.diagnostic_offer.take().unwrap();
    for flag in [0, 1, 2, 3] {
        let original = offer.clone();
        match flag {
            0 => offer.enabled = false,
            1 => offer.query_available = false,
            2 => offer.whole_context_read = false,
            _ => offer.schema_version += 1,
        }
        assert!(
            grant(
                &state,
                Some(&offer),
                &session.generation_grant.generation_id
            )
            .is_none()
        );
        offer = original;
    }
    offer.max_page_bytes = usize::MAX;
    assert_eq!(
        grant(
            &state,
            Some(&offer),
            &session.generation_grant.generation_id
        )
        .unwrap()
        .max_page_bytes,
        collection::PAGE_BYTES
    );
    assert!(collection::Collector::new(collection::owned_bytes() - 1).is_err());
}

#[tokio::test]
async fn daemon_collection_rejects_oversized_encoded_and_stalled_responses() {
    for scenario in ["oversized", "encoded", "stall", "disabled"] {
        let source = context();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new().route(
            crate::diagnostics_http::PREFIX,
            get(move || async move {
                match scenario {
                    "stall" => {
                        tokio::time::sleep(Duration::from_secs(10)).await;
                        StatusCode::OK.into_response()
                    }
                    "disabled" => error(HttpReason::Disabled),
                    "encoded" => (
                        [
                            (CONTENT_TYPE, "application/json"),
                            (http::header::CONTENT_ENCODING, "gzip"),
                        ],
                        "{}",
                    )
                        .into_response(),
                    _ => (
                        [(CONTENT_TYPE, "application/json")],
                        vec![b' '; collection::PAGE_BYTES + 1],
                    )
                        .into_response(),
                }
            }),
        );
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let (state, token) = broker();
        lock(&state.worker_sessions).insert("worker".into(), session(&source, &endpoint));
        poll_session(&state).await;
        let response = wrapper(&state, &token).await;
        let value: serde_json::Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert!(value["snapshot"].is_null());
        assert_eq!(
            value["collection"]["latest_poll"],
            match scenario {
                "stall" => "timeout",
                "disabled" => "disabled",
                "encoded" => "invalid_page",
                _ => "oversized",
            }
        );
        assert_eq!(
            lock(&state.worker_sessions)["worker"].diagnostic_poll_stopped,
            scenario == "disabled"
        );
        server.abort();
    }
}

#[tokio::test]
async fn daemon_collection_registration_replay_keeps_or_rotates_separate_authority() {
    let (state, _) = broker();
    let source = context();
    let mut session = session(&source, "http://127.0.0.1:1");
    // Match the existing signed generation endpoint in the staging fixture.
    let endpoint = session.generation_grant.endpoint.clone();
    session.pending_target = Arc::new(
        WorkerTarget::new("worker", &endpoint, SensitiveString::new("data").unwrap()).unwrap(),
    );
    // Reissue with the fixture broker's identity, exactly as stage_worker does.
    session.generation_grant = WorkerGenerationGrant::issue(
        "worker",
        session.fingerprint,
        &endpoint,
        None,
        &state.identity,
    )
    .unwrap();
    session.diagnostic_grant = grant(
        &state,
        session.diagnostic_offer.as_ref(),
        &session.generation_grant.generation_id,
    );
    let original = session
        .diagnostic_grant
        .as_ref()
        .unwrap()
        .token(now_unix_ms());
    let authority = diag::WorkerReadAuthority::default();
    authority.replace(session.diagnostic_grant.clone());
    let fingerprint = session.fingerprint;
    let offer = session.diagnostic_offer.clone();
    lock(&state.worker_sessions).insert("worker".into(), session);
    let proof = HandshakeProof {
        signer: ComponentRole::Daemon,
        signature: state.identity.sign(b"test"),
    };
    let replay = || {
        replay_worker_registration(
            &state,
            fingerprint,
            "worker",
            &endpoint,
            None,
            Some("activation"),
            None,
            offer.as_ref(),
            proof.clone(),
        )
        .unwrap()
    };
    let response: WorkerRegisterResponse =
        serde_json::from_slice(&replay().into_body().collect().await.unwrap().to_bytes()).unwrap();
    authority.replace(response.diagnostics);
    assert!(authority.verify(&original));
    lock(&state.worker_sessions)
        .get_mut("worker")
        .unwrap()
        .diagnostic_connected = false;
    let response: WorkerRegisterResponse =
        serde_json::from_slice(&replay().into_body().collect().await.unwrap().to_bytes()).unwrap();
    let new_token = response.diagnostics.as_ref().unwrap().token(now_unix_ms());
    authority.replace(response.diagnostics);
    assert!(!authority.verify(&original));
    assert!(authority.verify(&new_token));
    // Optional negotiation is omitted on the wire for old peers.
    let response = WorkerRegisterResponse {
        diagnostics: None,
        daemon_proof: proof,
        session_token: SensitiveString::new("session").unwrap(),
        data_token: SensitiveString::new("data").unwrap(),
        generation_grant: lock(&state.worker_sessions)["worker"]
            .generation_grant
            .clone(),
    };
    let value = serde_json::to_value(response).unwrap();
    assert!(value.get("diagnostics").is_none());
    assert!(
        serde_json::from_value::<WorkerRegisterResponse>(value)
            .unwrap()
            .diagnostics
            .is_none()
    );
}

#[tokio::test]
async fn daemon_collection_scheduler_bounds_in_flight_and_releases_work_on_shutdown() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let source = context();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let started = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route(
        crate::diagnostics_http::PREFIX,
        get({
            let started = started.clone();
            move || {
                let started = started.clone();
                async move {
                    started.fetch_add(1, Ordering::SeqCst);
                    std::future::pending::<()>().await;
                    StatusCode::OK
                }
            }
        }),
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let (state, _) = broker();
    for n in 0..8 {
        let mut session = session(&source, &endpoint);
        session.worker_id = format!("worker-{n}");
        session.pending_target = Arc::new(
            WorkerTarget::with_shared_client(
                session.worker_id.clone(),
                &endpoint,
                SensitiveString::new("data").unwrap(),
                state.worker_clients.client(None).unwrap(),
            )
            .unwrap(),
        );
        lock(&state.worker_sessions).insert(session.worker_id.clone(), session);
    }
    let task = tokio::spawn(run(Arc::downgrade(&state)));
    let lifetime = Lifetime::new(&state, Some(&task));
    tokio::time::timeout(Duration::from_secs(5), async {
        while started.load(Ordering::SeqCst) < collection::POLLS {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(started.load(Ordering::SeqCst), collection::POLLS);
    assert!(
        state
            .collector
            .as_ref()
            .unwrap()
            .ingress
            .reserve()
            .is_none()
    );
    drop(lifetime);
    let _ = task.await;
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let buffers = (0..collection::POLLS)
                .filter_map(|_| state.collector.as_ref().unwrap().ingress.reserve())
                .collect::<Vec<_>>();
            if buffers.len() == collection::POLLS {
                break;
            }
            drop(buffers);
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(state.collector.as_ref().unwrap().is_closed());
    server.abort();
}

#[tokio::test]
async fn daemon_collection_status_failures_never_publish_a_snapshot() {
    for (status, outcome) in [
        (StatusCode::FORBIDDEN, "unauthorized"),
        (StatusCode::NOT_FOUND, "unsupported"),
        (StatusCode::TOO_MANY_REQUESTS, "busy"),
        (StatusCode::UNPROCESSABLE_ENTITY, "envelope_too_small"),
        (StatusCode::BAD_GATEWAY, "unreachable"),
    ] {
        let source = context();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new().route(
            crate::diagnostics_http::PREFIX,
            get(move || async move { status }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (state, token) = broker();
        lock(&state.worker_sessions).insert("worker".into(), session(&source, &endpoint));
        poll_session(&state).await;
        let response = wrapper(&state, &token).await;
        let value: serde_json::Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(value["collection"]["latest_poll"], outcome, "{status}");
        assert!(value["snapshot"].is_null());
        server.abort();
    }
}

#[tokio::test]
async fn daemon_collection_read_reports_negotiated_unavailability() {
    for (scenario, expected) in [
        ("missing", StatusCode::NOT_FOUND),
        ("unsupported", StatusCode::NOT_IMPLEMENTED),
        ("disabled", StatusCode::SERVICE_UNAVAILABLE),
        ("query_disabled", StatusCode::SERVICE_UNAVAILABLE),
        ("schema", StatusCode::NOT_ACCEPTABLE),
        ("http_disabled", StatusCode::SERVICE_UNAVAILABLE),
        ("collection_disabled", StatusCode::SERVICE_UNAVAILABLE),
        ("closed", StatusCode::SERVICE_UNAVAILABLE),
    ] {
        let (mut state, token) = broker();
        if scenario == "http_disabled" {
            Arc::get_mut(&mut state).unwrap().diagnostics_http = None;
        }
        if scenario == "collection_disabled" {
            Arc::get_mut(&mut state).unwrap().collector = None;
        }
        let source = context();
        let mut session = session(&source, "http://127.0.0.1:1");
        match scenario {
            "unsupported" => session.diagnostic_offer = None,
            "disabled" => session.diagnostic_offer.as_mut().unwrap().enabled = false,
            "query_disabled" => session.diagnostic_offer.as_mut().unwrap().query_available = false,
            "schema" => session.diagnostic_offer.as_mut().unwrap().schema_version += 1,
            "closed" => state.collector.as_ref().unwrap().close(),
            _ => {}
        }
        if scenario != "missing" {
            lock(&state.worker_sessions).insert("worker".into(), session);
        }
        let response = wrapper(&state, &token).await;
        assert_eq!(response.status(), expected, "{scenario}");
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let envelope = DiagnosticHttpError::from_response(expected.as_u16(), &body).unwrap();
        let expected_reason = match scenario {
            "missing" => HttpReason::Missing,
            "unsupported" => HttpReason::Unsupported,
            "disabled" => HttpReason::Disabled,
            "query_disabled" => HttpReason::QueryDisabled,
            "schema" => HttpReason::UnsupportedSchema,
            "http_disabled" => HttpReason::HttpReadDisabled,
            "collection_disabled" => HttpReason::CollectionDisabled,
            "closed" => HttpReason::ShuttingDown,
            _ => unreachable!(),
        };
        assert_eq!(envelope.error.reason, expected_reason, "{scenario}");
    }
}

#[tokio::test]
async fn diagnostic_polling_uses_error_reasons_before_stopping_a_session() {
    // Exercise actual HTTP body consumption and the session's terminal decision.
    for (status, body, outcome, stopped) in [
        (
            503,
            r#"{"error":{"reason":"shutting_down"}}"#,
            "shutting_down",
            false,
        ),
        (503, r#"{"error":{"reason":"disabled"}}"#, "disabled", true),
        (
            503,
            r#"{"error":{"reason":"query_disabled"}}"#,
            "disabled",
            true,
        ),
        (
            503,
            r#"{"error":{"reason":"read_authority_unavailable"}}"#,
            "unreachable",
            false,
        ),
        (
            503,
            r#"{"error":{"reason":"future_reason"}}"#,
            "unreachable",
            false,
        ),
        (503, "", "unreachable", false),
        (503, "malformed", "unreachable", false),
        (
            503,
            r#"{"error":{"reason":"unsupported"}}"#,
            "unreachable",
            false,
        ),
        (
            404,
            r#"{"error":{"reason":"missing"}}"#,
            "unreachable",
            false,
        ),
        (404, "", "unsupported", true),
        (
            501,
            r#"{"error":{"reason":"unsupported"}}"#,
            "unsupported",
            true,
        ),
    ] {
        let source = context();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new().route(
            crate::diagnostics_http::PREFIX,
            get(move || async move { (StatusCode::from_u16(status).unwrap(), body) }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (state, token) = broker();
        lock(&state.worker_sessions).insert("worker".into(), session(&source, &endpoint));
        poll_session(&state).await;
        assert_eq!(
            lock(&state.worker_sessions)["worker"].diagnostic_poll_stopped,
            stopped,
            "{body}"
        );
        let response = wrapper(&state, &token).await;
        let value: serde_json::Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(value["collection"]["latest_poll"], outcome, "{body}");
        assert!(value["snapshot"].is_null());
        server.abort();
    }
}

#[tokio::test]
async fn oversized_diagnostic_errors_are_bounded_and_do_not_stop_polling() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new().route(
        crate::diagnostics_http::PREFIX,
        get(|| async {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "x".repeat(MAX_DIAGNOSTIC_HTTP_ERROR_BYTES + 1),
            )
        }),
    );
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let (state, token) = broker();
    lock(&state.worker_sessions).insert("worker".into(), session(&context(), &endpoint));
    poll_session(&state).await;
    assert!(!lock(&state.worker_sessions)["worker"].diagnostic_poll_stopped);
    let response = wrapper(&state, &token).await;
    let value: serde_json::Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(value["collection"]["latest_poll"], "oversized");
    server.abort();
}

#[tokio::test]
async fn daemon_diagnostic_lifetime_closes_reader_even_without_a_poll_task() {
    let (state, token) = broker();
    let app = state.diagnostics_http.as_ref().unwrap().clone().router();
    drop(Lifetime::new(&state, None));
    assert!(state.collector.as_ref().unwrap().is_closed());
    use tower::ServiceExt as _;
    let response = app
        .oneshot(
            Request::get(crate::diagnostics_http::PREFIX)
                .header(AUTH_HEADER, token)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}
