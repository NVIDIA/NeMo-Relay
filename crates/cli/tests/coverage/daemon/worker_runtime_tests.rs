// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::daemon::common::protocol::SensitiveString;
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
        lifecycle: Notify::new(),
    })
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

#[test]
fn configured_provider_auth_never_replaces_caller_auth() {
    let config = GatewayConfig {
        openai_auth_header: Some("Bearer configured".into()),
        ..GatewayConfig::default()
    };
    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer caller"));
    inject_provider_auth(&mut headers, ProviderRoute::OpenAi, &config);
    assert_eq!(headers.get(AUTHORIZATION).expect("auth"), "Bearer caller");

    headers.remove(AUTHORIZATION);
    inject_provider_auth(&mut headers, ProviderRoute::OpenAi, &config);
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
