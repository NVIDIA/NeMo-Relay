// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::daemon::common::protocol::SensitiveString;
use base64::Engine as _;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll};

use http::header::TRAILER;
use http_body_util::BodyExt as _;
use hyper::body::{Frame, Incoming, SizeHint};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use tokio::sync::oneshot;
use tower::ServiceExt as _;

use crate::test_support::EnvScope;

fn state() -> Arc<WorkerState> {
    state_with_config(GatewayConfig::default())
}

fn state_with_config(config: GatewayConfig) -> Arc<WorkerState> {
    Arc::new(WorkerState {
        worker_id: "worker-one".into(),
        config,
        upstream: pooled_client().expect("pooled client"),
        managed: None,
        auth: RwLock::new(AuthTokens {
            data: TokenDigest::from_token(b"data-secret"),
            pending_data: None,
            readiness_data: None,
            control: TokenDigest::from_token(b"control-secret"),
            last_control_sequence: 0,
            last_control_request_id: String::new(),
        }),
        accepting: AtomicBool::new(true),
        draining: AtomicBool::new(false),
        exiting: AtomicBool::new(false),
        in_flight: AtomicUsize::new(0),
        drain_deadline: RwLock::new(None),
        exit_reason: RwLock::new(None),
        lifecycle: Notify::new(),
    })
}

#[tokio::test]
async fn unmanaged_provider_wait_releases_admission_on_completion_cancellation_or_timeout() {
    for (response_timeout_secs, cancel) in [(0, false), (0, true), (90, false)] {
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let provider = axum::Router::new().route(
            "/v1/responses",
            axum::routing::post({
                let started = Arc::clone(&started);
                let release = Arc::clone(&release);
                move || {
                    let started = Arc::clone(&started);
                    let release = Arc::clone(&release);
                    async move {
                        started.notify_one();
                        release.notified().await;
                        "compaction complete"
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let state = state_with_config(GatewayConfig {
            openai_base_url: format!("http://{}", listener.local_addr().unwrap()),
            response_timeout_secs,
            ..GatewayConfig::default()
        });
        let server = tokio::spawn(async move { axum::serve(listener, provider).await.unwrap() });
        let app = router(Arc::clone(&state));
        let task = tokio::spawn(async move {
            app.oneshot(
                Request::post("/v1/responses")
                    .header(WORKER_TOKEN_HEADER, "data-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
        });
        tokio::time::timeout(Duration::from_secs(5), started.notified())
            .await
            .unwrap();
        tokio::time::pause();
        tokio::time::advance(Duration::from_secs(61)).await;
        tokio::task::yield_now().await;
        assert!(!task.is_finished());
        assert_eq!(state.in_flight.load(Ordering::Acquire), 1);
        tokio::time::advance(Duration::from_secs(3600)).await;
        tokio::task::yield_now().await;
        tokio::time::resume();
        if cancel {
            assert!(!task.is_finished());
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
        } else if response_timeout_secs == 0 {
            assert!(!task.is_finished());
            release.notify_one();
            let response = task.await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                response.into_body().collect().await.unwrap().to_bytes(),
                "compaction complete"
            );
        } else {
            let response = task.await.unwrap();
            assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
            drop(response);
        }
        assert_eq!(state.in_flight.load(Ordering::Acquire), 0);
        server.abort();
    }
}

#[test]
fn relative_drain_timeout_does_not_depend_on_the_daemon_wall_clock() {
    let request = WorkerDrainRequest {
        worker_id: "worker-one".into(),
        deadline_unix_ms: 0,
        timeout_ms: Some(321),
    };
    assert_eq!(drain_timeout_ms(&request), 321);
}

#[tokio::test]
async fn authenticated_readiness_probe_opens_admission_before_publication() {
    let state = state();
    state.accepting.store(false, Ordering::Release);
    write_lock(&state.auth).readiness_data = Some(TokenDigest::from_token(b"data-secret"));
    let headers = HeaderMap::from_iter([(
        HeaderName::from_static(WORKER_TOKEN_HEADER),
        HeaderValue::from_static("data-secret"),
    )]);
    let response = readiness_probe(State(Arc::clone(&state)), headers).await;
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert!(state.admit().is_some());
}

#[tokio::test]
async fn worker_stops_immediately_after_exit_or_an_empty_drain() {
    let exited = state();
    exited.request_exit("first");
    exited.request_exit("second");
    assert_eq!(*read_lock(&exited.exit_reason), Some("first"));
    tokio::time::timeout(Duration::from_millis(50), exited.wait_until_stopped())
        .await
        .expect("exit stops the worker");

    let drained = state();
    drained.begin_drain(0);
    tokio::time::timeout(Duration::from_millis(50), drained.wait_until_stopped())
        .await
        .expect("empty drain stops the worker");
}

#[tokio::test]
async fn unmanaged_worker_forwards_hooks_without_contacting_a_provider() {
    let response = router(state())
        .oneshot(
            Request::post("/hooks/pi")
                .header(WORKER_TOKEN_HEADER, "data-secret")
                .body(Body::from("native hook payload"))
                .expect("hook request"),
        )
        .await
        .expect("hook response");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .into_body()
            .collect()
            .await
            .expect("hook body")
            .to_bytes(),
        crate::daemon::common::routes::HookRoute::Pi.pass_through_body()
    );
}

#[test]
fn drain_control_requires_scoped_sequence_hash_and_exact_replay() {
    let state = state();
    let request = SessionRequest::new(
        "worker-one".into(),
        SensitiveString::new("control-secret").expect("secret"),
        1,
        WorkerDrainRequest {
            worker_id: "worker-one".into(),
            deadline_unix_ms: 100,
            timeout_ms: Some(100),
        },
    )
    .expect("drain request");
    assert!(state.authenticate_control(&request));
    assert!(state.authenticate_control(&request));

    let mut replay_mutation = request.clone();
    replay_mutation.request_id = "different-request".into();
    assert!(!state.authenticate_control(&replay_mutation));

    let out_of_order = SessionRequest::new(
        "worker-one".into(),
        SensitiveString::new("control-secret").expect("secret"),
        3,
        WorkerDrainRequest {
            worker_id: "worker-one".into(),
            deadline_unix_ms: 100,
            timeout_ms: Some(100),
        },
    )
    .expect("out of order");
    assert!(!state.authenticate_control(&out_of_order));
}

#[tokio::test]
async fn control_and_readiness_reject_mismatched_credentials_without_admission() {
    let state = state();
    state.accepting.store(false, Ordering::Release);
    write_lock(&state.auth).readiness_data = Some(TokenDigest::from_token(b"data-secret"));
    let mut duplicate = HeaderMap::new();
    duplicate.append(
        HeaderName::from_static(WORKER_TOKEN_HEADER),
        HeaderValue::from_static("data-secret"),
    );
    duplicate.append(
        HeaderName::from_static(WORKER_TOKEN_HEADER),
        HeaderValue::from_static("data-secret"),
    );
    assert_eq!(
        readiness_probe(State(Arc::clone(&state)), duplicate)
            .await
            .status(),
        StatusCode::NO_CONTENT
    );
    assert!(state.admit().is_none());

    let wrong_worker = SessionRequest::new(
        "wrong-worker".into(),
        SensitiveString::new("control-secret").expect("secret"),
        1,
        WorkerDrainRequest {
            worker_id: "wrong-worker".into(),
            deadline_unix_ms: 1,
            timeout_ms: None,
        },
    )
    .expect("control request");
    assert!(!state.authenticate_control(&wrong_worker));
}

struct CausalBody {
    phase: u8,
    release_second: oneshot::Receiver<()>,
    trailers: Option<HeaderMap>,
}

struct PanicBody;

impl hyper::body::Body for PanicBody {
    type Data = bytes::Bytes;
    type Error = Infallible;

    fn poll_frame(
        self: Pin<&mut Self>,
        _context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        panic!("an unauthenticated request body must not be polled")
    }
}

impl hyper::body::Body for CausalBody {
    type Data = bytes::Bytes;
    type Error = Infallible;

    fn poll_frame(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        match this.phase {
            0 => {
                this.phase = 1;
                Poll::Ready(Some(Ok(Frame::data(bytes::Bytes::from_static(
                    b"event: first\r\ndata: A\r\n\r\n",
                )))))
            }
            1 => match Pin::new(&mut this.release_second).poll(context) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(_) => {
                    this.phase = 2;
                    Poll::Ready(Some(Ok(Frame::data(bytes::Bytes::from_static(
                        b": heartbeat\r\ndata: [DONE]\r\n\r\n",
                    )))))
                }
            },
            2 => {
                this.phase = 3;
                Poll::Ready(
                    this.trailers
                        .take()
                        .map(|trailers| Ok(Frame::trailers(trailers))),
                )
            }
            _ => Poll::Ready(None),
        }
    }

    fn is_end_stream(&self) -> bool {
        self.phase >= 3 && self.trailers.is_none()
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}

#[test]
fn data_request_requires_exactly_one_matching_header() {
    let state = state();
    let mut headers = HeaderMap::new();
    assert!(!state.authenticate_data(&headers));
    headers.insert(WORKER_TOKEN_HEADER, HeaderValue::from_static("wrong"));
    assert!(!state.authenticate_data(&headers));
    headers.insert(WORKER_TOKEN_HEADER, HeaderValue::from_static("data-secret"));
    assert!(state.authenticate_data(&headers));
    headers.append(WORKER_TOKEN_HEADER, HeaderValue::from_static("data-secret"));
    assert!(!state.authenticate_data(&headers));
}

#[test]
fn recovery_probe_accepts_staged_token_only_until_commit_or_discard() {
    let state = state();
    let registration = control::test_registration("new-data-secret", "new-control-secret");
    state.stage_recovery_data_token(&registration);

    let mut headers = HeaderMap::new();
    headers.insert(
        WORKER_TOKEN_HEADER,
        HeaderValue::from_static("new-data-secret"),
    );
    assert!(state.authenticate_data(&headers));
    headers.insert(WORKER_TOKEN_HEADER, HeaderValue::from_static("data-secret"));
    assert!(state.authenticate_data(&headers));

    state.discard_recovery_data_token();
    headers.insert(
        WORKER_TOKEN_HEADER,
        HeaderValue::from_static("new-data-secret"),
    );
    assert!(!state.authenticate_data(&headers));

    state.stage_recovery_data_token(&registration);
    state.control_restored(&registration);
    assert!(state.authenticate_data(&headers));
    headers.insert(WORKER_TOKEN_HEADER, HeaderValue::from_static("data-secret"));
    assert!(!state.authenticate_data(&headers));
}

#[tokio::test]
async fn only_the_staged_registration_token_can_reopen_readiness() {
    let state = state();
    state.control_lost();
    let registration = control::test_registration("new-data-secret", "new-control-secret");
    state.stage_recovery_data_token(&registration);

    let old_headers = HeaderMap::from_iter([(
        HeaderName::from_static(WORKER_TOKEN_HEADER),
        HeaderValue::from_static("data-secret"),
    )]);
    assert_eq!(
        readiness_probe(State(Arc::clone(&state)), old_headers)
            .await
            .status(),
        StatusCode::NO_CONTENT
    );
    assert!(state.admit().is_none());

    let staged_headers = HeaderMap::from_iter([(
        HeaderName::from_static(WORKER_TOKEN_HEADER),
        HeaderValue::from_static("new-data-secret"),
    )]);
    assert_eq!(
        readiness_probe(State(Arc::clone(&state)), staged_headers)
            .await
            .status(),
        StatusCode::NO_CONTENT
    );
    assert!(state.admit().is_some());
}

#[tokio::test]
async fn unauthenticated_request_is_rejected_before_its_body_is_polled() {
    let request = Request::post("/v1/responses")
        .body(Body::new(PanicBody))
        .expect("worker request");
    let response = router(state())
        .oneshot(request)
        .await
        .expect("worker response");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn authenticated_readiness_probe_works_before_public_requests_are_admitted() {
    let state = state();
    state.control_lost();
    let probe = Request::get(WORKER_PROBE_PATH)
        .header(WORKER_TOKEN_HEADER, "data-secret")
        .body(Body::empty())
        .expect("probe request");
    assert_eq!(
        router(Arc::clone(&state))
            .oneshot(probe)
            .await
            .expect("probe response")
            .status(),
        StatusCode::NO_CONTENT
    );

    let provider = Request::post("/v1/responses")
        .header(WORKER_TOKEN_HEADER, "data-secret")
        .body(Body::new(PanicBody))
        .expect("provider request");
    let response = router(state)
        .oneshot(provider)
        .await
        .expect("provider response");
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response
            .headers()
            .get(WORKER_ROUTE_FAILURE_HEADER)
            .expect("route failure signal"),
        "pass-through"
    );
}

#[test]
fn route_failure_responses_are_explicitly_signaled_to_the_daemon() {
    let response = route_failure_response(CliError::Config("incompatible middleware".into()));
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        response
            .headers()
            .get(WORKER_ROUTE_FAILURE_HEADER)
            .expect("route failure signal"),
        "pass-through"
    );
}

#[test]
fn control_loss_rejects_new_admissions_without_touching_existing_one() {
    let state = state();
    let accepted = state.admit().expect("request admitted");
    assert_eq!(state.in_flight.load(Ordering::Acquire), 1);
    state.control_lost();
    assert!(state.admit().is_none());
    assert_eq!(state.in_flight.load(Ordering::Acquire), 1);
    drop(accepted);
    assert_eq!(state.in_flight.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn configured_provider_auth_never_replaces_caller_auth() {
    let config = GatewayConfig {
        openai_auth_header: Some("Bearer configured".into()),
        ..GatewayConfig::default()
    };
    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer caller"));
    inject_provider_auth(&mut headers, ProviderRoute::OpenAi, &config)
        .await
        .unwrap();
    assert_eq!(headers.get(AUTHORIZATION).expect("auth"), "Bearer caller");

    headers.remove(AUTHORIZATION);
    inject_provider_auth(&mut headers, ProviderRoute::OpenAi, &config)
        .await
        .unwrap();
    assert_eq!(
        headers.get(AUTHORIZATION).expect("configured auth"),
        "Bearer configured"
    );
}

#[tokio::test]
async fn provider_frames_are_forwarded_causally_with_status_headers_and_trailers() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind provider");
    let provider_address = listener.local_addr().expect("provider address");
    let (release_second, wait_for_release) = oneshot::channel();
    let provider = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept worker");
        let release = Arc::new(Mutex::new(Some(wait_for_release)));
        let service = service_fn(move |_request: Request<Incoming>| {
            let release = release
                .lock()
                .expect("release lock")
                .take()
                .expect("one request");
            async move {
                let mut trailers = HeaderMap::new();
                trailers.append("x-stream-checksum", HeaderValue::from_static("one"));
                trailers.append("x-stream-checksum", HeaderValue::from_static("two"));
                let mut response = Response::new(box_body(CausalBody {
                    phase: 0,
                    release_second: release,
                    trailers: Some(trailers),
                }));
                *response.status_mut() = StatusCode::CREATED;
                response
                    .headers_mut()
                    .append("x-provider", HeaderValue::from_static("first"));
                response
                    .headers_mut()
                    .append("x-provider", HeaderValue::from_static("second"));
                response.headers_mut().insert(
                    WORKER_ROUTE_FAILURE_HEADER,
                    HeaderValue::from_static("provider-spoof"),
                );
                response
                    .headers_mut()
                    .insert(TRAILER, HeaderValue::from_static("x-stream-checksum"));
                Ok::<_, Infallible>(response)
            }
        });
        let mut connection = http1::Builder::new();
        connection.keep_alive(false);
        connection
            .serve_connection(TokioIo::new(stream), service)
            .await
            .expect("serve provider response");
    });

    let config = GatewayConfig {
        openai_base_url: format!("http://{provider_address}/v1"),
        ..GatewayConfig::default()
    };
    let state = state_with_config(config);
    let request = Request::post("/v1/responses")
        .header(WORKER_TOKEN_HEADER, "data-secret")
        .body(Body::empty())
        .expect("worker request");
    let response = router(Arc::clone(&state))
        .oneshot(request)
        .await
        .expect("worker response");
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(response.headers().get_all("x-provider").iter().count(), 2);
    assert!(!response.headers().contains_key(WORKER_ROUTE_FAILURE_HEADER));
    assert_eq!(response.headers()[TRAILER], "x-stream-checksum");
    assert_eq!(state.in_flight.load(Ordering::Acquire), 1);

    let mut body = response.into_body();
    let first = body
        .frame()
        .await
        .expect("first frame")
        .expect("first frame succeeds")
        .into_data()
        .expect("first is data");
    assert_eq!(first, "event: first\r\ndata: A\r\n\r\n");
    let second = body.frame();
    tokio::pin!(second);
    assert!(futures_util::poll!(second.as_mut()).is_pending());
    release_second.send(()).expect("release provider");
    assert_eq!(
        second
            .await
            .expect("second frame")
            .expect("second frame succeeds")
            .into_data()
            .expect("second is data"),
        ": heartbeat\r\ndata: [DONE]\r\n\r\n"
    );
    let trailers = body
        .frame()
        .await
        .expect("trailer frame")
        .expect("trailer succeeds")
        .into_trailers()
        .expect("trailers");
    assert_eq!(
        trailers
            .get_all("x-stream-checksum")
            .iter()
            .map(|value| value.to_str().expect("ASCII trailer"))
            .collect::<Vec<_>>(),
        ["one", "two"]
    );
    assert!(body.frame().await.is_none());
    assert_eq!(state.in_flight.load(Ordering::Acquire), 0);
    provider.await.expect("provider task");
}

#[tokio::test]
async fn draining_waits_for_accepted_work_and_honors_the_deadline() {
    let draining_state = state();
    let accepted = draining_state
        .admit()
        .expect("request admitted before drain");
    draining_state.begin_drain(10_000);

    let waiting = draining_state.wait_until_stopped();
    tokio::pin!(waiting);
    assert!(futures_util::poll!(waiting.as_mut()).is_pending());
    drop(accepted);
    tokio::time::timeout(Duration::from_millis(50), waiting)
        .await
        .expect("releasing the last request completes drain");

    let expired = state();
    let _accepted = expired.admit().expect("request admitted");
    expired.begin_drain(0);
    tokio::time::timeout(Duration::from_millis(50), expired.wait_until_stopped())
        .await
        .expect("expired drain deadline stops waiting");
}

#[tokio::test]
async fn stopping_readiness_unknown_routes_and_control_auth_have_stable_responses() {
    let stopping_state = state();
    stopping_state.begin_drain(0);
    let response = readiness_probe(State(Arc::clone(&stopping_state)), HeaderMap::new()).await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let unknown = router(state())
        .oneshot(
            Request::get("/not-a-public-route")
                .header(WORKER_TOKEN_HEADER, "data-secret")
                .body(Body::empty())
                .expect("unknown request"),
        )
        .await
        .expect("unknown response");
    assert_eq!(unknown.status(), StatusCode::NOT_FOUND);

    let invalid = SessionRequest::new(
        "worker-one".into(),
        SensitiveString::new("wrong-control-secret").expect("secret"),
        1,
        WorkerDrainRequest {
            worker_id: "worker-one".into(),
            deadline_unix_ms: now_unix_ms() + 1_000,
            timeout_ms: None,
        },
    )
    .expect("drain request");
    let response = drain(State(state()), Json(invalid)).await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let valid = SessionRequest::new(
        "worker-one".into(),
        SensitiveString::new("control-secret").expect("secret"),
        1,
        WorkerDrainRequest {
            worker_id: "worker-one".into(),
            deadline_unix_ms: now_unix_ms() + 1_000,
            timeout_ms: None,
        },
    )
    .expect("drain request");
    let state = state();
    let response = drain(State(Arc::clone(&state)), Json(valid)).await;
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert!(state.draining.load(Ordering::Acquire));
}

#[tokio::test]
async fn provider_routing_reports_invalid_destinations_and_connection_failures() {
    let invalid = state_with_config(GatewayConfig {
        openai_base_url: "not a URI".into(),
        ..GatewayConfig::default()
    });
    let response = router(invalid)
        .oneshot(
            Request::post("/v1/responses")
                .header(WORKER_TOKEN_HEADER, "data-secret")
                .body(Body::empty())
                .expect("provider request"),
        )
        .await
        .expect("invalid destination response");
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);

    let unavailable = state_with_config(GatewayConfig {
        openai_base_url: "http://127.0.0.1:9".into(),
        ..GatewayConfig::default()
    });
    let response = router(unavailable)
        .oneshot(
            Request::post("/v1/responses")
                .header(WORKER_TOKEN_HEADER, "data-secret")
                .body(Body::empty())
                .expect("provider request"),
        )
        .await
        .expect("connection failure response");
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn provider_auth_supports_both_configured_and_environment_credentials() {
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
    )
    .await
    .unwrap();
    assert_eq!(openai[AUTHORIZATION], "Bearer openai-env");

    let mut anthropic = HeaderMap::new();
    inject_provider_auth(
        &mut anthropic,
        ProviderRoute::Anthropic,
        &GatewayConfig::default(),
    )
    .await
    .unwrap();
    assert_eq!(anthropic["x-api-key"], "anthropic-env");

    let mut configured = HeaderMap::new();
    inject_provider_auth(
        &mut configured,
        ProviderRoute::Anthropic,
        &GatewayConfig {
            anthropic_auth_header: Some("configured-anthropic".into()),
            ..GatewayConfig::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(configured[AUTHORIZATION], "configured-anthropic");
}

#[test]
fn runtime_state_constructor_stages_registration_credentials_until_readiness() {
    let registration = control::test_registration("fresh-data", "fresh-control");
    let state = Arc::new(
        WorkerState::new(
            "worker-one".into(),
            GatewayConfig::default(),
            None,
            &registration,
        )
        .expect("worker state"),
    );
    assert!(state.admit().is_none());
    let headers = HeaderMap::from_iter([(
        HeaderName::from_static(WORKER_TOKEN_HEADER),
        HeaderValue::from_static("fresh-data"),
    )]);
    assert!(state.authenticate_data(&headers));
    state.activate_pending_readiness(&headers);
    assert!(state.admit().is_some());
}

#[tokio::test]
async fn runtime_serve_reaches_readiness_and_exposes_the_authenticated_listener() {
    let _environment = EnvScope::set(&[("NEMO_RELAY_DAEMON_OBSERVATION_CAPTURE_BYTES", None)]);
    let daemon_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind daemon fixture");
    let daemon_address = daemon_listener.local_addr().expect("daemon address");
    let daemon = tokio::spawn(async move {
        axum::serve(
            daemon_listener,
            Router::new().route(
                crate::daemon::common::socket::WORKER_SOCKET_PATH,
                get(|ws: axum::extract::ws::WebSocketUpgrade| async {
                    ws.on_upgrade(|mut socket| async move {
                        while let Some(Ok(axum::extract::ws::Message::Text(text))) =
                            socket.recv().await
                        {
                            let request: crate::daemon::common::socket::Request =
                                serde_json::from_str(&text).unwrap();
                            let reply = crate::daemon::common::socket::Event::Reply {
                                request_id: request.request_id,
                                status: 204,
                                payload: serde_json::Value::Null,
                            };
                            socket
                                .send(axum::extract::ws::Message::Text(
                                    serde_json::to_string(&reply).unwrap().into(),
                                ))
                                .await
                                .unwrap();
                        }
                    })
                }),
            ),
        )
        .await
        .expect("serve daemon fixture");
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind worker");
    let worker_address = listener.local_addr().expect("worker address");
    let identity = MachineIdentity::generate().expect("identity").identity;
    let registration = control::test_registration("runtime-data", "runtime-control");
    registration
        .test_connect(&format!("http://{daemon_address}"))
        .await;
    let worker = tokio::spawn(serve(
        listener,
        RuntimeOptions {
            daemon_origin: format!("http://{daemon_address}"),
            identity,
            worker_id: "worker-one".into(),
            endpoint: format!("http://{worker_address}"),
            worker_tls_root: None,
            tls_config: None,
            config: GatewayConfig::default(),
            dynamic_plugins: Vec::new(),
            registration,
        },
    ));

    let client = reqwest::Client::new();
    let probe_url = format!("http://{worker_address}{WORKER_PROBE_PATH}");
    let mut response = None;
    for _ in 0..50 {
        if let Ok(candidate) = client
            .get(&probe_url)
            .header(WORKER_TOKEN_HEADER, "runtime-data")
            .send()
            .await
        {
            response = Some(candidate);
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(
        response.expect("worker became reachable").status(),
        StatusCode::NO_CONTENT
    );
    worker.abort();
    daemon.abort();
}

#[tokio::test]
async fn tls_worker_listener_discards_bad_handshakes_and_serves_authenticated_requests() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let rcgen::CertifiedKey { cert, key_pair } =
        rcgen::generate_simple_self_signed(vec!["127.0.0.1".into()]).expect("certificate");
    let key = rustls::pki_types::PrivateKeyDer::Pkcs8(rustls::pki_types::PrivatePkcs8KeyDer::from(
        key_pair.serialize_der(),
    ));
    let mut tls = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert.der().clone()], key)
        .expect("TLS server config");
    tls.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind TLS worker");
    let address = listener.local_addr().expect("worker address");
    let server = tokio::spawn(serve_tls(listener, router(state()), Arc::new(tls)));

    // A non-TLS peer is ignored without terminating the listener.
    let mut plain = tokio::net::TcpStream::connect(address)
        .await
        .expect("plain connection");
    use tokio::io::AsyncWriteExt as _;
    plain.write_all(b"not tls").await.expect("plain write");
    plain.shutdown().await.expect("plain shutdown");

    let root = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(cert.der());
    let client = crate::daemon::common::worker_tls::pooled_worker_tls_client(&root)
        .expect("pinned worker client");
    let request = Request::get(format!(
        "https://127.0.0.1:{}{WORKER_PROBE_PATH}",
        address.port()
    ))
    .header(WORKER_TOKEN_HEADER, "data-secret")
    .body(box_body(http_body_util::Empty::<bytes::Bytes>::new()))
    .expect("TLS probe");
    let response = client.request(request).await.expect("TLS response");
    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    server.abort();
    assert!(server.await.expect_err("server aborted").is_cancelled());
}

#[tokio::test(start_paused = true)]
async fn monitor_control_rejects_work_and_exits_after_recovery_deadline() {
    let state = state();
    state.accepting.store(true, Ordering::Release);
    let identity = MachineIdentity::generate().expect("identity").identity;
    tokio::time::timeout(
        Duration::from_millis(crate::daemon::common::control::RECOVERY_LIFETIME_MS + 1_000),
        monitor_control(
            Arc::clone(&state),
            "http://127.0.0.1:9".into(),
            identity,
            "worker-one".into(),
            "http://127.0.0.1:1".into(),
            None,
            control::test_registration("data", "control"),
        ),
    )
    .await
    .expect("monitor exits after its bounded recovery window");
    assert!(!state.accepting.load(Ordering::Acquire));
    assert!(state.exiting.load(Ordering::Acquire));
}

#[tokio::test]
async fn websocket_proxy_authentication_and_drain_release_worker_admission() {
    use crate::gateway::websocket::tests::{provider, serve};
    use futures_util::StreamExt;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    let (upstream, provider_task, active) = provider().await;
    let state = state_with_config(GatewayConfig {
        openai_base_url: upstream,
        ..GatewayConfig::default()
    });
    let (origin, task) = serve(router(state.clone())).await;
    let url = format!("{}/v1/live?model=voice", origin.replacen("http", "ws", 1));
    let error = tokio_tungstenite::connect_async(&url).await.unwrap_err();
    let tokio_tungstenite::tungstenite::Error::Http(response) = error else {
        panic!("{error}");
    };
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(active.load(Ordering::SeqCst), 0);
    let mut request = url.into_client_request().unwrap();
    request
        .headers_mut()
        .insert(WORKER_TOKEN_HEADER, HeaderValue::from_static("data-secret"));
    request.headers_mut().insert(
        AUTHORIZATION,
        HeaderValue::from_static("Bearer caller-voice-key"),
    );
    let (mut socket, _) = tokio_tungstenite::connect_async(request).await.unwrap();
    socket.next().await.unwrap().unwrap();
    assert_eq!(state.in_flight.load(Ordering::SeqCst), 1);
    state.begin_drain(0);
    tokio::time::timeout(Duration::from_secs(2), async {
        while state.in_flight.load(Ordering::SeqCst) != 0 || active.load(Ordering::SeqCst) != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    task.abort();
    provider_task.abort();
}

#[tokio::test]
async fn websocket_drain_cancels_a_pending_upstream_handshake() {
    use crate::gateway::websocket::tests::serve;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    let started = Arc::new(Notify::new());
    let app = axum::Router::new().fallback({
        let started = Arc::clone(&started);
        move || {
            let started = Arc::clone(&started);
            async move {
                started.notify_one();
                std::future::pending::<StatusCode>().await
            }
        }
    });
    let (upstream, provider_task) = serve(app).await;
    let state = state_with_config(GatewayConfig {
        openai_base_url: upstream,
        ..GatewayConfig::default()
    });
    let (origin, task) = serve(router(Arc::clone(&state))).await;
    let mut request = format!("{}/v1/live", origin.replacen("http", "ws", 1))
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert(WORKER_TOKEN_HEADER, HeaderValue::from_static("data-secret"));
    let connection = tokio::spawn(tokio_tungstenite::connect_async(request));
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .unwrap();
    assert_eq!(state.in_flight.load(Ordering::SeqCst), 1);
    state.begin_drain(0);
    let error = tokio::time::timeout(Duration::from_secs(2), connection)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    let tokio_tungstenite::tungstenite::Error::Http(response) = error else {
        panic!("{error}");
    };
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(state.in_flight.load(Ordering::SeqCst), 0);
    task.abort();
    provider_task.abort();
}

#[test]
fn managed_provider_transport_failures_preserve_operational_reasons() {
    use nemo_relay::error::{UpstreamFailure, UpstreamFailureClass};
    for (class, reason) in [
        (UpstreamFailureClass::Connection, Some("transport_error")),
        (UpstreamFailureClass::Timeout, Some("response_head_timeout")),
        (UpstreamFailureClass::Authentication, None),
    ] {
        let error = CliError::ProviderFailure(UpstreamFailure {
            status: None,
            body: "provider failure".into(),
            headers: Default::default(),
            class,
        });
        assert_eq!(managed_upstream_failure_reason(&error), reason);
    }
    let error = CliError::UpstreamTransport(
        crate::daemon::common::transport::TransportError::InsecureDestination,
    );
    assert_eq!(
        managed_upstream_failure_reason(&error),
        Some("transport_error")
    );
}

#[tokio::test]
async fn unmanaged_routes_forward_only_client_credentials() {
    let _env = crate::test_support::EnvScope::set(&[(
        "OPENAI_API_KEY",
        Some(std::ffi::OsStr::new("relay-environment-secret")),
    )]);
    for configured in [None, Some("Bearer relay-owned-secret")] {
        async fn echo(request: Request<Body>) -> Response<Body> {
            Response::builder()
                .status(StatusCode::CREATED)
                .body(Body::from(
                    request
                        .headers()
                        .get("authorization")
                        .map_or("absent", |value| value.to_str().unwrap())
                        .to_owned(),
                ))
                .unwrap()
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config = GatewayConfig {
            openai_base_url: format!("http://{}/v1", listener.local_addr().unwrap()),
            chatgpt_codex_base_url: Some(format!("http://{}", listener.local_addr().unwrap())),
            openai_auth_header: configured.map(str::to_owned),
            ..GatewayConfig::default()
        };
        let provider = axum::Router::new().fallback(echo);
        let task = tokio::spawn(async move { axum::serve(listener, provider).await.unwrap() });
        let app = router(state_with_config(config));
        for path in [
            "/v1/responses/compact",
            "/v1/images/generations",
            "/v1/realtime/calls",
            "/v1/live/sessions",
            "/v1/images/edits",
            "/v1/alpha/search",
            "/v1/memories/trace_summarize",
        ] {
            for credential in [
                None,
                Some("Bearer caller-key"),
                Some("Bearer eyJhbGciOiJIUzI1NiJ9.deadbeef.signature"),
            ] {
                let mut request = Request::post(path)
                    .header("content-type", "application/json")
                    .header("x-nemo-relay-worker-token", "data-secret");
                if let Some(credential) = credential {
                    request = request.header("authorization", credential);
                }
                let request = request.body(Body::from("{}")).unwrap();
                let response = app.clone().oneshot(request).await.unwrap();
                assert_eq!(response.status(), StatusCode::CREATED, "{path}");
                let bytes = response.into_body().collect().await.unwrap().to_bytes();
                assert_eq!(
                    bytes.as_ref(),
                    credential.unwrap_or("absent").as_bytes(),
                    "{path}"
                );
            }
        }
        task.abort();
    }
}

#[tokio::test]
async fn model_catalog_workers_keep_provider_credentials_on_the_selected_upstream() {
    let _plugins = crate::test_support::PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _env = EnvScope::set(&[
        (
            "OPENAI_API_KEY",
            Some(std::ffi::OsStr::new("environment-openai")),
        ),
        (
            "ANTHROPIC_API_KEY",
            Some(std::ffi::OsStr::new("environment-anthropic")),
        ),
    ]);
    async fn upstream(name: &'static str) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new().fallback(move |request: Request<Body>| async move {
            Json(json!({
                "provider": name,
                "authorization": request.headers().get("authorization").map(|v| v.to_str().unwrap()),
                "key": request.headers().get("x-api-key").map(|v| v.to_str().unwrap()),
                "uri": request.uri().to_string(),
            }))
        });
        (
            base,
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() }),
        )
    }
    let (openai, openai_task) = upstream("openai").await;
    let (anthropic, anthropic_task) = upstream("anthropic").await;
    for configured in [false, true] {
        let config = GatewayConfig {
            openai_base_url: format!("{openai}/v1"),
            anthropic_base_url: anthropic.clone(),
            openai_auth_header: configured.then(|| "Bearer configured-openai".into()),
            anthropic_auth_header: configured.then(|| "Bearer configured-anthropic".into()),
            ..GatewayConfig::default()
        };
        for managed in [false, true] {
            let mut state = state_with_config(config.clone());
            if managed {
                Arc::get_mut(&mut state).unwrap().managed = Some(
                    ManagedRuntime::initialize(
                        config.clone(),
                        Vec::new(),
                        "model-catalog-test".into(),
                    )
                    .await
                    .unwrap()
                    .into(),
                );
            }
            let app = router(state.clone());
            for provider in ["openai", "anthropic"] {
                for path in ["/models", "/v1/models"] {
                    let mut request = Request::get(format!("{path}?limit=1"))
                        .header(WORKER_TOKEN_HEADER, "data-secret");
                    if provider == "anthropic" {
                        request = request.header("anthropic-version", "2023-06-01");
                    }
                    let response = app
                        .clone()
                        .oneshot(request.body(Body::empty()).unwrap())
                        .await
                        .unwrap();
                    assert_eq!(response.status(), StatusCode::OK);
                    let body = response.into_body().collect().await.unwrap().to_bytes();
                    let observed: serde_json::Value = serde_json::from_slice(&body).unwrap();
                    assert_eq!(observed["provider"], provider, "managed={managed}");
                    assert_eq!(observed["uri"], "/v1/models?limit=1");
                    let auth = if configured {
                        Some(format!("Bearer configured-{provider}"))
                    } else {
                        (provider == "openai").then(|| "Bearer environment-openai".into())
                    };
                    assert_eq!(observed["authorization"], json!(auth));
                    assert_eq!(
                        observed["key"],
                        json!(
                            (!configured && provider == "anthropic")
                                .then_some("environment-anthropic")
                        )
                    );
                }
            }
            if let Some(runtime) = &state.managed {
                runtime.close().await.unwrap();
            }
        }
    }
    openai_task.abort();
    anthropic_task.abort();
}
