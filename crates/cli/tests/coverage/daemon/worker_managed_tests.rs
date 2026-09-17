// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::convert::Infallible;
use std::ffi::OsStr;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use axum::Router;
use axum::extract::State;
use axum::routing::post;
use http_body_util::{BodyExt as _, Full, StreamBody};
use nemo_relay::api::llm::LlmRequestInterceptOutcome;
use nemo_relay::api::registry::{RuntimeRegistrationOwner, RuntimeRegistrationOwnerKind};
use nemo_relay::api::registry::{
    deregister_llm_execution_intercept, deregister_llm_request_intercept,
    register_llm_execution_intercept, register_llm_request_intercept,
};

use crate::daemon::common::transport::pooled_client;
use crate::test_support::{EnvScope, PLUGIN_CONFIG_TEST_LOCK};

type CapturedProviderRequest = Arc<std::sync::Mutex<Option<(HeaderMap, Bytes)>>>;
type ProviderRequests = Arc<std::sync::Mutex<Vec<(HeaderMap, Bytes)>>>;

#[tokio::test]
async fn observation_preserves_delivery_while_capturing_json() {
    let expected = Bytes::from_static(br#"{"ok":true}"#);
    let (body, observation) = observe_body(
        Full::new(expected.clone()),
        StatusCode::OK,
        expected.len(),
        OperationalContext::new(),
    );
    let delivered = body.collect().await.expect("delivered body").to_bytes();
    let observed = observation
        .finish(ProviderSurface::OpenAIResponses, false)
        .await;
    assert_eq!(delivered, expected);
    assert_eq!(observed.value, Some(json!({ "ok": true })));
    assert!(!observed.truncated);
    assert_eq!(observed.terminal, OBSERVATION_COMPLETE);
}

#[tokio::test]
async fn capture_limit_truncates_observation_without_truncating_delivery() {
    let expected = Bytes::from_static(br#"{"too":"large"}"#);
    let (body, observation) = observe_body(
        Full::new(expected.clone()),
        StatusCode::OK,
        3,
        OperationalContext::new(),
    );
    let delivered = body.collect().await.expect("delivered body").to_bytes();
    let observed = observation
        .finish(ProviderSurface::OpenAIResponses, false)
        .await;
    assert_eq!(delivered, expected);
    assert!(observed.value.is_none());
    assert!(observed.truncated);
    assert_eq!(observed.terminal, OBSERVATION_COMPLETE);
}

#[tokio::test]
async fn saturated_observation_queue_never_blocks_or_truncates_delivery() {
    let expected = (0..OBSERVATION_QUEUE_FRAMES + 8)
        .map(|index| Bytes::from(vec![u8::try_from(index).expect("test byte")]))
        .collect::<Vec<_>>();
    let frames = expected
        .clone()
        .into_iter()
        .map(|bytes| Ok::<_, Infallible>(Frame::data(bytes)));
    let (mut body, observation) = observe_body(
        StreamBody::new(futures_util::stream::iter(frames)),
        StatusCode::OK,
        usize::MAX,
        OperationalContext::new(),
    );

    let mut delivered = Vec::new();
    while let Some(frame) = body.frame().await {
        let frame = frame.expect("delivery frame");
        if let Some(data) = frame.data_ref() {
            delivered.extend_from_slice(data);
        }
    }
    let observed = observation
        .finish(ProviderSurface::OpenAIResponses, false)
        .await;

    let expected_delivery = expected
        .iter()
        .flat_map(|bytes| bytes.iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(delivered, expected_delivery);
    assert!(observed.truncated);
    assert_eq!(observed.terminal, OBSERVATION_COMPLETE);
}

#[tokio::test]
async fn dropping_delivery_marks_observation_cancelled_and_terminates_it() {
    let frames = [
        Ok::<_, Infallible>(Frame::data(Bytes::from_static(b"first"))),
        Ok(Frame::data(Bytes::from_static(b"second"))),
    ];
    let (mut body, observation) = observe_body(
        StreamBody::new(futures_util::stream::iter(frames)),
        StatusCode::OK,
        usize::MAX,
        OperationalContext::new(),
    );
    let first = body
        .frame()
        .await
        .expect("first frame")
        .expect("first delivery frame")
        .into_data()
        .expect("first data");
    assert_eq!(first, "first");
    drop(body);

    let observed = tokio::time::timeout(
        Duration::from_secs(1),
        observation.finish(ProviderSurface::OpenAIResponses, false),
    )
    .await
    .expect("observation task must terminate after cancellation");
    assert_eq!(observed.terminal, OBSERVATION_CANCELLED);
    assert!(
        observed
            .failure
            .as_deref()
            .is_some_and(|failure| failure.contains("cancelled"))
    );
}

#[tokio::test(start_paused = true)]
async fn observation_body_deadline_terminates_a_stalled_observer() {
    let started = tokio::time::Instant::now();
    let (_sender, receiver) = tokio::sync::mpsc::channel(1);
    let observation = ObservationReceiver {
        receiver,
        signal: Arc::new(ObservationSignal::new()),
        status: StatusCode::OK,
        operational: OperationalContext::new(),
    };
    let observed = observation
        .finish(ProviderSurface::OpenAIResponses, false)
        .await;
    assert!(observed.truncated);
    assert_eq!(observed.terminal, OBSERVATION_ACTIVE);
    assert_eq!(
        observed.failure.as_deref(),
        Some("provider response observation timed out")
    );
    assert_eq!(started.elapsed(), OBSERVATION_COMPLETION_TIMEOUT);
}

#[tokio::test(start_paused = true)]
async fn successful_stream_observation_can_outlive_the_response_head_deadline() {
    let (sender, receiver) = tokio::sync::mpsc::channel(4);
    let signal = Arc::new(ObservationSignal::new());
    let observation = ObservationReceiver {
        receiver,
        signal: Arc::clone(&signal),
        status: StatusCode::OK,
        operational: OperationalContext::new(),
    };
    let task = tokio::spawn(observation.finish(ProviderSurface::OpenAIChat, true));
    tokio::task::yield_now().await;
    sender.send(Bytes::from_static(b"data: {\"id\":\"long\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"}}]}\n\n")).await.unwrap();
    tokio::time::advance(RESPONSE_HEAD_TIMEOUT + Duration::from_secs(1)).await;
    tokio::task::yield_now().await;
    assert!(
        !task.is_finished(),
        "response-head deadline must not truncate observation"
    );
    sender.send(Bytes::from_static(b"data: {\"id\":\"long\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")).await.unwrap();
    drop(sender);
    signal.finish(OBSERVATION_COMPLETE);
    let observed = task.await.unwrap();
    assert_eq!(observed.terminal, OBSERVATION_COMPLETE);
    assert!(!observed.truncated);
    assert!(observed.failure.is_none());
    assert!(observed.value.is_some());
}

#[test]
fn internal_worker_headers_are_not_forwarded_to_providers() {
    let mut headers = HeaderMap::new();
    headers.insert(
        WORKER_TOKEN_HEADER,
        HeaderValue::from_static("worker-secret"),
    );
    headers.insert(
        CLIENT_TOKEN_HEADER,
        HeaderValue::from_static("client-secret"),
    );
    headers.insert(
        "x-nemo-relay-session-id",
        HeaderValue::from_static("session"),
    );
    headers.insert("x-provider-header", HeaderValue::from_static("kept"));
    strip_worker_headers(&mut headers);
    strip_internal_headers(&mut headers);
    assert!(!headers.contains_key(WORKER_TOKEN_HEADER));
    assert!(!headers.contains_key(CLIENT_TOKEN_HEADER));
    assert!(!headers.contains_key("x-nemo-relay-session-id"));
    assert_eq!(headers["x-provider-header"], "kept");
}

#[tokio::test]
async fn daemon_hop_cannot_supply_worker_local_dispatch_overrides() {
    let request = Request::post("/v1/responses")
        .header(
            INTERNAL_DISPATCH_URL_HEADER,
            "https://attacker.invalid/v1/responses",
        )
        .header(INTERNAL_DISPATCH_ROUTE_HEADER, "anthropic_messages")
        .header(INTERNAL_DISPATCH_BACKEND_HEADER, "attacker")
        .header(INTERNAL_RETRY_AWARE_HEADER, "true")
        .header("x-nemo-relay-session-id", "session-kept-for-correlation")
        .body(Body::from(r#"{"model":"test","stream":true}"#))
        .expect("provider request");
    let prepared = PreparedProviderRequest::read(request, &GatewayConfig::default())
        .await
        .expect("prepared request");

    assert!(!prepared.headers.contains_key(INTERNAL_DISPATCH_URL_HEADER));
    assert!(
        !prepared
            .headers
            .contains_key(INTERNAL_DISPATCH_ROUTE_HEADER)
    );
    assert!(
        !prepared
            .headers
            .contains_key(INTERNAL_DISPATCH_BACKEND_HEADER)
    );
    assert!(!prepared.headers.contains_key(INTERNAL_RETRY_AWARE_HEADER));
    assert_eq!(
        prepared.headers["x-nemo-relay-session-id"],
        "session-kept-for-correlation"
    );
}

#[test]
fn execution_middleware_is_explicitly_incompatible_with_raw_delivery() {
    let owner = RuntimeRegistrationOwner {
        kind: RuntimeRegistrationOwnerKind::GlobalApi,
        plugin_kind: None,
        component_ordinal: None,
    };
    let registrations = [
        RuntimeRegistrationIdentity {
            kind: RuntimeRegistrationKind::LlmExecutionIntercept,
            local_name: "buffered".into(),
            effective_name: "plugin.buffered".into(),
            owner: owner.clone(),
        },
        RuntimeRegistrationIdentity {
            kind: RuntimeRegistrationKind::LlmStreamExecutionIntercept,
            local_name: "streaming".into(),
            effective_name: "plugin.streaming".into(),
            owner,
        },
        RuntimeRegistrationIdentity {
            kind: RuntimeRegistrationKind::LlmRequestIntercept,
            local_name: "request".into(),
            effective_name: "plugin.request".into(),
            owner: RuntimeRegistrationOwner {
                kind: RuntimeRegistrationOwnerKind::GlobalApi,
                plugin_kind: None,
                component_ordinal: None,
            },
        },
    ];

    assert_eq!(
        incompatible_registration_names(&registrations),
        ["plugin.buffered", "plugin.streaming"]
    );
}

#[test]
fn only_request_middleware_requires_request_body_decoding() {
    let owner = RuntimeRegistrationOwner {
        kind: RuntimeRegistrationOwnerKind::GlobalApi,
        plugin_kind: None,
        component_ordinal: None,
    };
    for kind in [
        RuntimeRegistrationKind::LlmSanitizeRequestGuardrail,
        RuntimeRegistrationKind::LlmConditionalExecutionGuardrail,
        RuntimeRegistrationKind::LlmRequestIntercept,
    ] {
        assert!(registration_reads_request_body(
            &RuntimeRegistrationIdentity {
                kind,
                local_name: "request-reader".into(),
                effective_name: "request-reader".into(),
                owner: owner.clone(),
            }
        ));
    }
    for kind in [
        RuntimeRegistrationKind::Subscriber,
        RuntimeRegistrationKind::LlmSanitizeResponseGuardrail,
    ] {
        assert!(!registration_reads_request_body(
            &RuntimeRegistrationIdentity {
                kind,
                local_name: "response-only".into(),
                effective_name: "response-only".into(),
                owner: owner.clone(),
            }
        ));
    }
}

struct PendingRequestBody {
    polls: Arc<AtomicUsize>,
}

impl HttpBody for PendingRequestBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        self: Pin<&mut Self>,
        _context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        self.polls.fetch_add(1, AtomicOrdering::Relaxed);
        Poll::Pending
    }

    fn is_end_stream(&self) -> bool {
        false
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}

#[tokio::test]
async fn unbuffered_dispatch_returns_response_head_without_collecting_request_body() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind provider");
    let address = listener.local_addr().expect("provider address");
    let app = Router::new().route(
        "/v1/responses",
        post(|| async {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream; charset=utf-8")
                .body(Body::from("data: [DONE]\n\n"))
                .expect("provider response")
        }),
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve provider");
    });
    let polls = Arc::new(AtomicUsize::new(0));
    let request = Request::post("/v1/responses")
        .header(ACCEPT, "text/event-stream")
        .body(Body::new(PendingRequestBody {
            polls: Arc::clone(&polls),
        }))
        .expect("streaming request");
    let config = GatewayConfig {
        openai_base_url: format!("http://{address}"),
        ..GatewayConfig::default()
    };

    let (response, observation, streaming) = tokio::time::timeout(
        Duration::from_secs(1),
        dispatch_unbuffered_observed(
            crate::daemon::common::transport::pooled_client().expect("provider client"),
            request,
            ProviderRoute::OpenAi,
            &config,
            DEFAULT_OBSERVATION_CAPTURE_BYTES,
            OperationalContext::new(),
        ),
    )
    .await
    .expect("response head must not wait for request completion")
    .expect("provider response");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(streaming);
    assert!(polls.load(AtomicOrdering::Relaxed) > 0);
    let delivered = response
        .into_body()
        .collect()
        .await
        .expect("delivered response")
        .to_bytes();
    assert_eq!(delivered, "data: [DONE]\n\n");
    let observed = observation
        .finish(ProviderSurface::OpenAIResponses, true)
        .await;
    assert_eq!(observed.terminal, OBSERVATION_COMPLETE);
    server.abort();
}

#[test]
fn changing_stream_mode_is_a_route_wide_transport_incompatibility() {
    let incompatible = CliError::Flow(FlowError::InvalidArgument(
        STREAM_MODE_MUTATION_ERROR.into(),
    ));
    assert!(requires_route_pass_through(&incompatible));
    assert!(!requires_route_pass_through(&CliError::Flow(
        FlowError::InvalidArgument("some other request error".into())
    )));
}

#[test]
fn chatgpt_shaped_responses_requests_use_the_managed_responses_pipeline() {
    assert_eq!(
        provider_surface("/backend-api/codex/responses"),
        Some(ProviderSurface::OpenAIResponses)
    );
}

#[test]
fn managed_worker_canonicalizes_chatgpt_responses_before_alignment() {
    let _environment = EnvScope::set(&[("OPENAI_API_KEY", None)]);
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_static("Bearer at-managed-chatgpt-token"),
    );
    let prepared = PreparedProviderRequest {
        method: Method::POST,
        version: http::Version::HTTP_11,
        headers,
        path: "/backend-api/codex/responses".into(),
        path_and_query: "/backend-api/codex/responses?client=codex".into(),
        body: Bytes::from_static(br#"{"model":"test","stream":true}"#),
        request_json: json!({"model": "test", "stream": true}),
        streaming: true,
    };
    let destination = effective_destination(
        &prepared,
        ProviderRoute::OpenAi,
        None,
        &GatewayConfig::default(),
    )
    .expect("ChatGPT destination");

    assert_eq!(
        destination,
        "https://chatgpt.com/backend-api/codex/responses?client=codex"
            .parse::<Uri>()
            .unwrap()
    );
}

#[test]
fn managed_worker_does_not_infer_upstream_authority_from_generic_bearer_tokens() {
    let _environment = EnvScope::set(&[("OPENAI_API_KEY", None)]);
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_static("Bearer at-caller-controlled-token"),
    );
    let prepared = PreparedProviderRequest {
        method: Method::POST,
        version: http::Version::HTTP_11,
        headers,
        path: "/responses".into(),
        path_and_query: "/responses?client=pi".into(),
        body: Bytes::from_static(br#"{"model":"test","stream":true}"#),
        request_json: json!({"model": "test", "stream": true}),
        streaming: true,
    };
    let config = GatewayConfig {
        openai_base_url: "https://administrator.example/v1".into(),
        ..GatewayConfig::default()
    };

    let destination = effective_destination(&prepared, ProviderRoute::OpenAi, None, &config)
        .expect("administrator-selected destination");

    assert_eq!(
        destination,
        "https://administrator.example/v1/responses?client=pi"
            .parse::<Uri>()
            .unwrap()
    );
}

#[test]
fn managed_worker_uses_pi_named_provider_endpoint_and_strips_the_routing_header() {
    let mut headers = HeaderMap::new();
    headers.insert(
        crate::agents::pi::alignment::UPSTREAM_BASE_URL_HEADER,
        HeaderValue::from_static("https://custom.example/inference/v1"),
    );
    headers.insert(
        http::header::AUTHORIZATION,
        HeaderValue::from_static("Bearer caller-provider-token"),
    );
    let prepared = PreparedProviderRequest {
        method: Method::POST,
        version: http::Version::HTTP_11,
        headers,
        path: "/chat/completions".into(),
        path_and_query: "/chat/completions?client=pi".into(),
        body: Bytes::from_static(br#"{"model":"custom","stream":true}"#),
        request_json: json!({"model": "custom", "stream": true}),
        streaming: true,
    };

    let destination = effective_destination(
        &prepared,
        ProviderRoute::OpenAi,
        None,
        &GatewayConfig::default(),
    )
    .expect("Pi-selected destination");
    assert_eq!(
        destination,
        "https://custom.example/inference/v1/chat/completions?client=pi"
            .parse::<Uri>()
            .unwrap()
    );

    let (forwarded, _, _) = effective_request(&prepared, None).expect("forwarded request");
    assert!(!forwarded.contains_key(crate::agents::pi::alignment::UPSTREAM_BASE_URL_HEADER));
}

#[test]
fn unchanged_middleware_headers_preserve_credentials_and_duplicate_values() {
    let mut headers = HeaderMap::new();
    headers.append("x-provider-feature", HeaderValue::from_static("first"));
    headers.append("x-provider-feature", HeaderValue::from_static("second"));
    headers.insert(
        http::header::COOKIE,
        HeaderValue::from_static("session=secret"),
    );
    headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer provider"));
    headers.insert("x-api-key", HeaderValue::from_static("provider-key"));
    let prepared = prepared_request(headers);
    let effective = LlmRequest {
        headers: crate::gateway::daemon_observable_headers(&prepared.headers),
        content: prepared.request_json.clone(),
    };

    let (forwarded, body, explicit_target) =
        effective_request(&prepared, Some(&effective)).expect("effective request");

    let values = forwarded
        .get_all("x-provider-feature")
        .iter()
        .map(|value| value.to_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(values, ["first", "second"]);
    assert_eq!(forwarded[http::header::COOKIE], "session=secret");
    assert_eq!(forwarded[AUTHORIZATION], "Bearer provider");
    assert_eq!(forwarded["x-api-key"], "provider-key");
    assert_eq!(body, prepared.body);
    assert!(!explicit_target);
}

#[test]
fn middleware_header_diff_changes_only_the_named_observable_header() {
    let mut headers = HeaderMap::new();
    headers.append("x-unchanged", HeaderValue::from_static("first"));
    headers.append("x-unchanged", HeaderValue::from_static("second"));
    headers.insert("x-changed", HeaderValue::from_static("before"));
    headers.insert(
        http::header::COOKIE,
        HeaderValue::from_static("session=secret"),
    );
    headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer provider"));
    let prepared = prepared_request(headers);
    let mut effective_headers = crate::gateway::daemon_observable_headers(&prepared.headers);
    effective_headers.insert("x-changed".into(), json!("after"));
    let effective = LlmRequest {
        headers: effective_headers,
        content: prepared.request_json.clone(),
    };

    let (forwarded, _, explicit_target) =
        effective_request(&prepared, Some(&effective)).expect("effective request");

    let unchanged = forwarded
        .get_all("x-unchanged")
        .iter()
        .map(|value| value.to_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(unchanged, ["first", "second"]);
    assert_eq!(forwarded["x-changed"], "after");
    assert_eq!(forwarded[http::header::COOKIE], "session=secret");
    assert_eq!(forwarded[AUTHORIZATION], "Bearer provider");
    assert!(!explicit_target);
}

#[test]
fn explicit_target_removes_hidden_provider_credentials() {
    let mut headers = HeaderMap::new();
    headers.insert(
        http::header::COOKIE,
        HeaderValue::from_static("session=secret"),
    );
    headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer provider"));
    headers.insert("x-api-key", HeaderValue::from_static("provider-key"));
    let prepared = prepared_request(headers);
    let mut effective_headers = crate::gateway::daemon_observable_headers(&prepared.headers);
    effective_headers.insert(
        INTERNAL_DISPATCH_URL_HEADER.into(),
        json!("https://selected.example/v1/responses"),
    );
    effective_headers.insert("authorization".into(), json!("Bearer selected-provider"));
    let effective = LlmRequest {
        headers: effective_headers,
        content: prepared.request_json.clone(),
    };

    let (forwarded, _, explicit_target) =
        effective_request(&prepared, Some(&effective)).expect("effective request");

    assert!(explicit_target);
    assert!(!forwarded.contains_key(http::header::COOKIE));
    assert_eq!(forwarded[AUTHORIZATION], "Bearer selected-provider");
    assert!(!forwarded.contains_key("x-api-key"));
}

fn prepared_request(headers: HeaderMap) -> PreparedProviderRequest {
    PreparedProviderRequest {
        method: Method::POST,
        version: http::Version::HTTP_11,
        headers,
        path: "/v1/responses".into(),
        path_and_query: "/v1/responses".into(),
        body: Bytes::from_static(br#"{"model":"test","stream":true}"#),
        request_json: json!({"model": "test", "stream": true}),
        streaming: true,
    }
}

#[test]
fn observation_capture_limit_defaults_and_accepts_a_positive_override() {
    {
        let _environment = EnvScope::set(&[(OBSERVATION_CAPTURE_BYTES_ENV, None)]);
        assert_eq!(
            observation_capture_limit_from_environment().unwrap(),
            DEFAULT_OBSERVATION_CAPTURE_BYTES
        );
    }
    {
        let _environment =
            EnvScope::set(&[(OBSERVATION_CAPTURE_BYTES_ENV, Some(OsStr::new("65536")))]);
        assert_eq!(
            observation_capture_limit_from_environment().unwrap(),
            65_536
        );
    }
}

#[test]
fn observation_capture_limit_rejects_zero_and_invalid_values() {
    for value in ["0", "not-a-number"] {
        let _environment =
            EnvScope::set(&[(OBSERVATION_CAPTURE_BYTES_ENV, Some(OsStr::new(value)))]);
        let error = observation_capture_limit_from_environment()
            .unwrap_err()
            .to_string();
        assert!(error.contains("positive integer"), "{error}");
    }
}

#[tokio::test]
async fn prepared_requests_and_both_dispatch_paths_preserve_provider_contracts() {
    let received: ProviderRequests = Arc::new(std::sync::Mutex::new(Vec::new()));
    let app = Router::new()
        .route(
            "/v1/responses",
            post(
                |State(received): State<ProviderRequests>, request: Request<Body>| async move {
                    let (parts, body) = request.into_parts();
                    let body = axum::body::to_bytes(body, usize::MAX)
                        .await
                        .expect("provider request body");
                    received
                        .lock()
                        .expect("capture provider request")
                        .push((parts.headers, body));
                    Response::builder()
                        .status(StatusCode::CREATED)
                        .header(CONTENT_TYPE, "text/event-stream; charset=utf-8")
                        .header("x-provider-response", "preserved")
                        .body(Body::from("event: message\ndata: raw\n\n"))
                        .expect("provider response")
                },
            ),
        )
        .with_state(Arc::clone(&received));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind provider");
    let address = listener.local_addr().expect("provider address");
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve provider");
    });
    let config = GatewayConfig {
        openai_base_url: format!("http://{address}"),
        openai_auth_header: Some("Bearer configured-secret".into()),
        ..GatewayConfig::default()
    };

    let source = Request::post("/v1/responses?trace=one")
        .header(WORKER_TOKEN_HEADER, "worker-secret")
        .header(CLIENT_TOKEN_HEADER, "client-secret")
        .header(INTERNAL_DISPATCH_URL_HEADER, "https://attacker.invalid")
        .header("x-provider-request", "preserved")
        .body(Body::from(br#"{"model":"test","stream":true}"#.as_slice()))
        .expect("provider request");
    let prepared = PreparedProviderRequest::read(source, &config)
        .await
        .expect("prepared provider request");
    assert_eq!(prepared.path, "/v1/responses");
    assert_eq!(prepared.path_and_query, "/v1/responses?trace=one");
    assert!(prepared.streaming);
    assert!(!prepared.headers.contains_key(WORKER_TOKEN_HEADER));
    assert!(!prepared.headers.contains_key(INTERNAL_DISPATCH_URL_HEADER));

    let response = dispatch_observed(
        pooled_client().expect("provider client"),
        prepared,
        ProviderRoute::OpenAi,
        None,
        &config,
        DEFAULT_OBSERVATION_CAPTURE_BYTES,
        OperationalContext::new(),
    )
    .await
    .expect("observed provider dispatch");
    assert_eq!(response.0.status(), StatusCode::CREATED);
    assert_eq!(response.0.headers()["x-provider-response"], "preserved");
    let delivered = response
        .0
        .into_body()
        .collect()
        .await
        .expect("observed delivery")
        .to_bytes();
    assert_eq!(delivered, "event: message\ndata: raw\n\n");
    let observed = response
        .1
        .finish(ProviderSurface::OpenAIResponses, true)
        .await;
    assert_eq!(observed.terminal, OBSERVATION_COMPLETE);

    let raw = Request::post("/v1/responses")
        .header(WORKER_TOKEN_HEADER, "worker-secret")
        .header(CLIENT_TOKEN_HEADER, "client-secret")
        .body(Body::from("raw provider payload"))
        .expect("unmanaged request");
    let response = dispatch_unmanaged(
        pooled_client().expect("provider client"),
        raw,
        ProviderRoute::OpenAi,
        &config,
    )
    .await
    .expect("unmanaged provider dispatch");
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(
        response
            .into_body()
            .collect()
            .await
            .expect("unmanaged delivery")
            .to_bytes(),
        "event: message\ndata: raw\n\n"
    );

    let captured = received.lock().expect("read provider requests");
    assert_eq!(captured.len(), 2);
    for (headers, _) in captured.iter() {
        assert_eq!(headers[AUTHORIZATION], "Bearer configured-secret");
        assert!(!headers.contains_key(WORKER_TOKEN_HEADER));
        assert!(!headers.contains_key(CLIENT_TOKEN_HEADER));
    }
    assert_eq!(
        captured[0].1,
        br#"{"model":"test","stream":true}"#.as_slice()
    );
    assert_eq!(captured[1].1, "raw provider payload");
    server.abort();
}

#[tokio::test]
async fn prepared_request_enforces_body_limit_and_normalizes_non_json_payloads() {
    let limited = GatewayConfig {
        max_passthrough_body_bytes: 2,
        ..GatewayConfig::default()
    };
    let too_large = PreparedProviderRequest::read(
        Request::post("/v1/responses")
            .body(Body::from("abc"))
            .expect("request"),
        &limited,
    )
    .await;
    let too_large = match too_large {
        Ok(_) => panic!("request must honor configured body limit"),
        Err(error) => error,
    };
    assert!(matches!(too_large, CliError::PayloadTooLarge(_)));

    let prepared = PreparedProviderRequest::read(
        Request::post("/v1/responses")
            .body(Body::from("not-json"))
            .expect("request"),
        &GatewayConfig::default(),
    )
    .await
    .expect("non-json requests remain forwardable");
    assert_eq!(prepared.request_json, Value::Null);
    assert!(!prepared.streaming);
}

#[test]
fn managed_request_helpers_cover_overrides_streaming_and_provider_credentials() {
    let _environment = EnvScope::set(&[
        ("OPENAI_API_KEY", Some(OsStr::new("  openai-env-key  "))),
        ("ANTHROPIC_API_KEY", Some(OsStr::new("anthropic-env-key"))),
    ]);
    let prepared = prepared_request(HeaderMap::from_iter([(
        CONTENT_LENGTH,
        HeaderValue::from_static("123"),
    )]));
    let (headers, body, explicit) = effective_request(&prepared, None).expect("raw request");
    assert!(!explicit);
    assert!(!headers.contains_key(CONTENT_LENGTH));
    assert_eq!(body, prepared.body);

    let mut effective_headers = serde_json::Map::new();
    effective_headers.insert(
        INTERNAL_DISPATCH_URL_HEADER.into(),
        json!("http://127.0.0.1:9999/v1/responses"),
    );
    let effective = LlmRequest {
        headers: effective_headers,
        content: json!({"model":"replacement","stream":false}),
    };
    assert!(has_explicit_target(&effective));
    assert!(!stream_mode(&effective));
    assert!(!prepared_streaming(&effective));
    let destination = effective_destination(
        &prepared,
        ProviderRoute::OpenAi,
        Some(&effective),
        &GatewayConfig::default(),
    )
    .expect("explicit destination");
    assert_eq!(
        destination,
        "http://127.0.0.1:9999/v1/responses".parse::<Uri>().unwrap()
    );
    let (_, body, explicit) =
        effective_request(&prepared, Some(&effective)).expect("rewritten request");
    assert!(explicit);
    assert_eq!(
        body,
        Bytes::from_static(br#"{"model":"replacement","stream":false}"#)
    );

    let mut headers = HeaderMap::new();
    inject_provider_auth(
        &mut headers,
        ProviderRoute::OpenAi,
        &GatewayConfig::default(),
    );
    assert_eq!(headers[AUTHORIZATION], "Bearer openai-env-key");
    headers.clear();
    inject_provider_auth(
        &mut headers,
        ProviderRoute::Anthropic,
        &GatewayConfig::default(),
    );
    assert_eq!(headers["x-api-key"], "anthropic-env-key");
    headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer caller"));
    inject_provider_auth(
        &mut headers,
        ProviderRoute::OpenAi,
        &GatewayConfig::default(),
    );
    assert_eq!(headers[AUTHORIZATION], "Bearer caller");

    let accepts_sse = HeaderMap::from_iter([(
        ACCEPT,
        HeaderValue::from_static("application/json, text/event-stream; charset=utf-8"),
    )]);
    assert!(request_streaming_hint(&accepts_sse));
    let content_type = HeaderMap::from_iter([(
        CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    )]);
    assert!(response_streaming(&content_type));
    assert_eq!(
        provider_surface("/v1/messages"),
        Some(ProviderSurface::AnthropicMessages)
    );
    assert_eq!(provider_surface("/unsupported"), None);
}

#[tokio::test]
async fn observation_reports_provider_body_errors_and_metadata() {
    let frames = futures_util::stream::iter(vec![Err::<Frame<Bytes>, io::Error>(
        io::Error::other("provider body failed"),
    )]);
    let (mut body, observation) = observe_body(
        StreamBody::new(frames),
        StatusCode::BAD_GATEWAY,
        DEFAULT_OBSERVATION_CAPTURE_BYTES,
        OperationalContext::new(),
    );
    assert!(body.frame().await.expect("body frame").is_err());
    let observed = observation
        .finish(ProviderSurface::OpenAIResponses, false)
        .await;
    assert_eq!(observed.terminal, OBSERVATION_BODY_ERROR);
    assert_eq!(observed.status, StatusCode::BAD_GATEWAY);
    assert!(observed.failure.is_some());
    assert_eq!(
        observed.metadata()["daemon_worker_observation"]["terminal"],
        "body_error"
    );
}

#[tokio::test]
async fn managed_runtime_forwards_all_managed_hook_shapes_and_closes_cleanly() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let runtime =
        ManagedRuntime::initialize(GatewayConfig::default(), Vec::new(), "machine-owner".into())
            .await
            .expect("initialize managed runtime");

    for (route, payload, expected) in [
        (HookRoute::Codex, json!({}), json!({})),
        (HookRoute::Claude, json!({}), json!({"continue": true})),
        (HookRoute::Pi, json!({}), json!({})),
    ] {
        let response = runtime
            .handle_hook(
                route,
                Request::post("/hook")
                    .header(WORKER_TOKEN_HEADER, "internal-only")
                    .body(Body::from(payload.to_string()))
                    .expect("hook request"),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response
            .into_body()
            .collect()
            .await
            .expect("hook response")
            .to_bytes();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).expect("hook JSON response"),
            expected
        );
    }

    runtime
        .ensure_streaming_transport_compatible()
        .expect("no incompatible middleware");
    runtime.close().await.expect("close managed runtime");
}

#[tokio::test]
async fn managed_runtime_uses_unmanaged_provider_fallback_without_buffering_response() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind provider");
    let address = listener.local_addr().expect("provider address");
    let app = Router::new().route(
        "/v1/custom",
        post(|| async {
            Response::builder()
                .status(StatusCode::ACCEPTED)
                .header(CONTENT_TYPE, "application/octet-stream")
                .body(Body::from(Bytes::from_static(b"opaque provider bytes")))
                .expect("provider response")
        }),
    );
    let provider = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve provider");
    });
    let config = GatewayConfig {
        openai_base_url: format!("http://{address}"),
        ..GatewayConfig::default()
    };
    let runtime = ManagedRuntime::initialize(config, Vec::new(), "machine-owner".into())
        .await
        .expect("initialize managed runtime");
    let response = runtime
        .proxy_provider(
            pooled_client().expect("provider client"),
            Request::post("/custom")
                .body(Body::from("unbuffered request bytes"))
                .expect("provider request"),
            ProviderRoute::OpenAi,
        )
        .await
        .expect("unmanaged dispatch");
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    assert_eq!(
        response
            .into_body()
            .collect()
            .await
            .expect("provider body")
            .to_bytes(),
        "opaque provider bytes"
    );
    runtime.close().await.expect("close managed runtime");
    provider.abort();
}

#[tokio::test]
async fn cancelled_provider_dispatch_releases_sessions_in_both_worker_paths() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const INTERCEPT: &str = "daemon-cancel-before-head";
    let _registration = RequestInterceptRegistration(INTERCEPT);
    for buffered in [false, true] {
        if buffered {
            register_llm_request_intercept(
                INTERCEPT,
                1,
                false,
                Arc::new(|_, request, annotated| {
                    Box::pin(async move { Ok(LlmRequestInterceptOutcome::new(request, annotated)) })
                }),
            )
            .unwrap();
        }
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/v1/responses",
            post({
                let entered = Arc::clone(&entered);
                let release = Arc::clone(&release);
                move || {
                    let entered = Arc::clone(&entered);
                    let release = Arc::clone(&release);
                    async move {
                        entered.notify_one();
                        release.notified().await;
                        StatusCode::OK
                    }
                }
            }),
        );
        let provider = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let runtime = Arc::new(
            ManagedRuntime::initialize(
                GatewayConfig {
                    openai_base_url: format!("http://{address}"),
                    ..GatewayConfig::default()
                },
                Vec::new(),
                "cancel-owner".into(),
            )
            .await
            .unwrap(),
        );
        let dispatch_runtime = Arc::clone(&runtime);
        let dispatch = tokio::spawn(async move {
            dispatch_runtime
                .proxy_provider(
                    pooled_client().unwrap(),
                    Request::post("/v1/responses")
                        .body(Body::from(
                            r#"{"model":"test","input":"test","stream":true}"#,
                        ))
                        .unwrap(),
                    ProviderRoute::OpenAi,
                )
                .await
        });
        tokio::time::timeout(Duration::from_secs(5), entered.notified())
            .await
            .unwrap();
        assert!(runtime.sessions.has_open_sessions().await);
        dispatch.abort();
        assert!(dispatch.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(5), async {
            while runtime.sessions.has_open_sessions().await {
                runtime
                    .sessions
                    .close_idle_sessions_at(
                        std::time::Instant::now(),
                        Duration::ZERO,
                        "cancelled_test_call",
                    )
                    .await
                    .unwrap();
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancelled request retained its gateway session");
        release.notify_one();
        runtime.close().await.unwrap();
        provider.abort();
    }
}

#[tokio::test]
async fn cancelled_request_middleware_releases_its_gateway_session() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const INTERCEPT: &str = "daemon-cancel-in-middleware";
    let _registration = RequestInterceptRegistration(INTERCEPT);
    let entered = Arc::new(Notify::new());
    register_llm_request_intercept(
        INTERCEPT,
        1,
        false,
        Arc::new({
            let entered = Arc::clone(&entered);
            move |_, _, _| {
                let entered = Arc::clone(&entered);
                Box::pin(async move {
                    entered.notify_one();
                    std::future::pending().await
                })
            }
        }),
    )
    .unwrap();
    let runtime = Arc::new(
        ManagedRuntime::initialize(
            GatewayConfig::default(),
            Vec::new(),
            "cancel-middleware-owner".into(),
        )
        .await
        .unwrap(),
    );
    let dispatch_runtime = Arc::clone(&runtime);
    let dispatch = tokio::spawn(async move {
        dispatch_runtime
            .proxy_provider(
                pooled_client().unwrap(),
                Request::post("/v1/responses")
                    .body(Body::from(
                        r#"{"model":"test","input":"test","stream":true}"#,
                    ))
                    .unwrap(),
                ProviderRoute::OpenAi,
            )
            .await
    });
    tokio::time::timeout(Duration::from_secs(5), entered.notified())
        .await
        .unwrap();
    assert!(runtime.sessions.has_open_sessions().await);
    dispatch.abort();
    assert!(dispatch.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(5), async {
        while runtime.sessions.has_open_sessions().await {
            runtime
                .sessions
                .close_idle_sessions_at(
                    std::time::Instant::now(),
                    Duration::ZERO,
                    "cancelled_test_call",
                )
                .await
                .unwrap();
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("cancelled middleware retained its gateway session");
    runtime.close().await.unwrap();
}

#[tokio::test]
async fn managed_runtime_bypasses_middleware_for_a_claude_startup_probe() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const INTERCEPT: &str = "daemon-worker-startup-probe-bypass-coverage";
    let _ = deregister_llm_request_intercept(INTERCEPT);
    let _registration = RequestInterceptRegistration(INTERCEPT);
    register_llm_request_intercept(
        INTERCEPT,
        1,
        false,
        Arc::new(|_name, request, annotated| {
            Box::pin(async move { Ok(LlmRequestInterceptOutcome::new(request, annotated)) })
        }),
    )
    .unwrap();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let provider = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().route(
                "/v1/messages",
                post(|| async { (StatusCode::ACCEPTED, "startup probe forwarded") }),
            ),
        )
        .await
        .unwrap();
    });
    let runtime = ManagedRuntime::initialize(
        GatewayConfig {
            anthropic_base_url: format!("http://{address}"),
            ..GatewayConfig::default()
        },
        Vec::new(),
        "machine-owner".into(),
    )
    .await
    .unwrap();
    let response = runtime
        .proxy_provider(
            pooled_client().unwrap(),
            Request::post("/v1/messages")
                .header("x-claude-code-session-id", "startup-probe")
                .body(Body::from(
                    r#"{"model":"claude-opus-4-8[1m]","max_tokens":1,"messages":[{"role":"user","content":"test"}]}"#,
                ))
                .unwrap(),
            ProviderRoute::Anthropic,
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    assert_eq!(
        response.into_body().collect().await.unwrap().to_bytes(),
        "startup probe forwarded"
    );
    drop(_registration);
    let response = runtime
        .proxy_provider(
            pooled_client().unwrap(),
            Request::post("/v1/messages")
                .header("x-claude-code-session-id", "unbuffered-startup-probe")
                .body(Body::from(
                    r#"{"model":"claude-opus-4-8[1m]","max_tokens":1,"messages":[{"role":"user","content":"test"}]}"#,
                ))
                .unwrap(),
            ProviderRoute::Anthropic,
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    assert_eq!(
        response.into_body().collect().await.unwrap().to_bytes(),
        "startup probe forwarded"
    );
    runtime.close().await.unwrap();
    provider.abort();
}

#[tokio::test]
async fn managed_runtime_observes_a_supported_stream_without_rewriting_its_bytes() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind provider");
    let address = listener.local_addr().expect("provider address");
    let app = Router::new().route(
        "/v1/responses",
        post(|| async {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(
                    "event: response.output_text.delta\ndata: hello\n\ndata: [DONE]\n\n",
                ))
                .expect("provider response")
        }),
    );
    let provider = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve provider");
    });
    let runtime = ManagedRuntime::initialize(
        GatewayConfig {
            openai_base_url: format!("http://{address}"),
            ..GatewayConfig::default()
        },
        Vec::new(),
        "machine-owner".into(),
    )
    .await
    .expect("initialize managed runtime");
    let response = runtime
        .proxy_provider(
            pooled_client().expect("provider client"),
            Request::post("/v1/responses")
                .header(ACCEPT, "text/event-stream")
                .body(Body::from(
                    r#"{"model":"test","input":"hello","stream":true}"#,
                ))
                .expect("provider request"),
            ProviderRoute::OpenAi,
        )
        .await
        .expect("managed stream dispatch");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .into_body()
            .collect()
            .await
            .expect("managed stream body")
            .to_bytes(),
        "event: response.output_text.delta\ndata: hello\n\ndata: [DONE]\n\n"
    );
    tokio::time::sleep(Duration::from_millis(10)).await;
    runtime.close().await.expect("close managed runtime");
    provider.abort();
}

#[tokio::test]
async fn managed_runtime_closes_the_llm_lifecycle_when_provider_dispatch_fails() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let runtime = ManagedRuntime::initialize(
        GatewayConfig {
            openai_base_url: "http://127.0.0.1:9".into(),
            ..GatewayConfig::default()
        },
        Vec::new(),
        "machine-owner".into(),
    )
    .await
    .expect("initialize managed runtime");
    let error = runtime
        .proxy_provider(
            pooled_client().expect("provider client"),
            Request::post("/v1/responses")
                .body(Body::from(r#"{"model":"test","input":"hello"}"#))
                .expect("provider request"),
            ProviderRoute::OpenAi,
        )
        .await
        .expect_err("unreachable provider must fail");
    assert!(
        matches!(error, CliError::Launch(_)),
        "unexpected dispatch error: {error:?}"
    );
    runtime.close().await.expect("close managed runtime");
}

struct RequestInterceptRegistration(&'static str);

impl Drop for RequestInterceptRegistration {
    fn drop(&mut self) {
        let _ = deregister_llm_request_intercept(self.0);
    }
}

struct ExecutionInterceptRegistration(&'static str);

impl Drop for ExecutionInterceptRegistration {
    fn drop(&mut self) {
        let _ = deregister_llm_execution_intercept(self.0);
    }
}

#[tokio::test]
async fn managed_runtime_rejects_response_mutating_execution_middleware() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const INTERCEPT: &str = "daemon-worker-incompatible-execution-coverage";
    let _ = deregister_llm_execution_intercept(INTERCEPT);
    let _registration = ExecutionInterceptRegistration(INTERCEPT);
    register_llm_execution_intercept(
        INTERCEPT,
        1,
        Arc::new(|_name, _request, _next| Box::pin(async { Ok(json!({})) })),
    )
    .expect("register execution middleware");

    let result =
        ManagedRuntime::initialize(GatewayConfig::default(), Vec::new(), "machine-owner".into())
            .await;
    let error = match result {
        Ok(_) => panic!("raw worker delivery must reject execution middleware"),
        Err(error) => error,
    };
    assert!(error.to_string().contains(INTERCEPT));
}

#[tokio::test]
async fn managed_runtime_applies_request_middleware_before_raw_provider_delivery() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const INTERCEPT: &str = "daemon-worker-managed-request-coverage";
    let _ = deregister_llm_request_intercept(INTERCEPT);
    let _registration = RequestInterceptRegistration(INTERCEPT);
    register_llm_request_intercept(
        INTERCEPT,
        1,
        false,
        Arc::new(|_name, mut request, annotated| {
            request
                .headers
                .insert("x-worker-intercept".into(), json!("applied"));
            request.content["input"] = json!("rewritten by middleware");
            Box::pin(async move { Ok(LlmRequestInterceptOutcome::new(request, annotated)) })
        }),
    )
    .expect("register request middleware");

    let captured: CapturedProviderRequest = Arc::new(std::sync::Mutex::new(None));
    let app = Router::new()
        .route(
            "/v1/responses",
            post(
                |State(captured): State<CapturedProviderRequest>,
                 request: Request<Body>| async move {
                    let (parts, body) = request.into_parts();
                    let body = axum::body::to_bytes(body, usize::MAX)
                        .await
                        .expect("provider body");
                    *captured.lock().expect("capture request") = Some((parts.headers, body));
                    Response::builder()
                        .status(StatusCode::OK)
                        .header(CONTENT_TYPE, "text/event-stream")
                        .body(Body::from("data: raw provider result\n\n"))
                        .expect("provider response")
                },
            ),
        )
        .with_state(Arc::clone(&captured));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind provider");
    let address = listener.local_addr().expect("provider address");
    let provider = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve provider");
    });
    let runtime = ManagedRuntime::initialize(
        GatewayConfig {
            openai_base_url: format!("http://{address}"),
            ..GatewayConfig::default()
        },
        Vec::new(),
        "machine-owner".into(),
    )
    .await
    .expect("initialize managed runtime");
    let response = runtime
        .proxy_provider(
            pooled_client().expect("provider client"),
            Request::post("/v1/responses")
                .header(ACCEPT, "text/event-stream")
                .body(Body::from(
                    r#"{"model":"test","input":"original","stream":true}"#,
                ))
                .expect("provider request"),
            ProviderRoute::OpenAi,
        )
        .await
        .expect("managed provider dispatch");
    assert_eq!(
        response
            .into_body()
            .collect()
            .await
            .expect("provider response body")
            .to_bytes(),
        "data: raw provider result\n\n"
    );
    let (headers, body) = captured
        .lock()
        .expect("captured request")
        .take()
        .expect("provider saw request");
    assert_eq!(headers["x-worker-intercept"], "applied");
    assert_eq!(
        serde_json::from_slice::<Value>(&body).expect("rewritten request JSON")["input"],
        "rewritten by middleware"
    );
    runtime.close().await.expect("close managed runtime");
    provider.abort();
}

#[tokio::test]
async fn managed_runtime_rejects_request_middleware_that_changes_stream_mode() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const INTERCEPT: &str = "daemon-worker-stream-mode-mutation-coverage";
    let _ = deregister_llm_request_intercept(INTERCEPT);
    let _registration = RequestInterceptRegistration(INTERCEPT);
    register_llm_request_intercept(
        INTERCEPT,
        1,
        false,
        Arc::new(|_name, mut request, annotated| {
            request.content["stream"] = json!(false);
            Box::pin(async move { Ok(LlmRequestInterceptOutcome::new(request, annotated)) })
        }),
    )
    .expect("register request middleware");

    let runtime =
        ManagedRuntime::initialize(GatewayConfig::default(), Vec::new(), "machine-owner".into())
            .await
            .expect("initialize managed runtime");
    let error = runtime
        .proxy_provider(
            pooled_client().expect("provider client"),
            Request::post("/v1/responses")
                .header(ACCEPT, "text/event-stream")
                .body(Body::from(
                    r#"{"model":"test","input":"value","stream":true}"#,
                ))
                .expect("provider request"),
            ProviderRoute::OpenAi,
        )
        .await
        .expect_err("stream-mode mutation must fail closed");
    assert!(error.to_string().contains(STREAM_MODE_MUTATION_ERROR));
    runtime.close().await.expect("close managed runtime");
}

#[tokio::test]
async fn managed_helper_error_and_metadata_paths_are_explicit() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let prepared = prepared_request(HeaderMap::new());
    let invalid_destination = LlmRequest {
        headers: serde_json::Map::from_iter([(
            INTERNAL_DISPATCH_URL_HEADER.into(),
            json!("http://[invalid"),
        )]),
        content: Value::Null,
    };
    assert!(
        effective_destination(
            &prepared,
            ProviderRoute::OpenAi,
            Some(&invalid_destination),
            &GatewayConfig::default(),
        )
        .is_err()
    );
    assert!(!has_explicit_target(&LlmRequest {
        headers: serde_json::Map::from_iter([(
            INTERNAL_DISPATCH_ROUTE_HEADER.into(),
            json!("   "),
        )]),
        content: Value::Null,
    }));
    assert_eq!(json_header_value(&json!(42)).unwrap(), "42");
    assert!(json_header_value(&json!("\n")).is_none());
    assert_eq!(
        merge_object(json!("not-an-object"), json!({"added": true})),
        json!({"added": true})
    );
    let mut metadata = Value::Null;
    insert_metadata(&mut metadata, "key", json!("value"));
    assert_eq!(metadata, json!({"key": "value"}));
    assert!(!request_body_decode_required().expect("empty middleware registry"));
    assert!(reject_incompatible_execution_middleware().is_ok());
}

#[test]
fn managed_header_and_observation_helpers_cover_ignored_and_invalid_values() {
    let mut source = HeaderMap::new();
    source.insert("x-remove-me", HeaderValue::from_static("before"));
    source.insert(CONTENT_ENCODING, HeaderValue::from_static("gzip"));
    source.insert(
        INTERNAL_DISPATCH_BACKEND_HEADER,
        HeaderValue::from_static("untrusted"),
    );
    let prepared = prepared_request(source);
    let effective = LlmRequest {
        headers: serde_json::Map::from_iter([
            ("bad header".into(), json!("ignored")),
            (INTERNAL_RETRY_AWARE_HEADER.into(), json!(true)),
            ("x-json".into(), json!({"nested": true})),
        ]),
        content: json!({"model": "rewritten", "stream": true}),
    };
    let (headers, body, explicit) = effective_request(&prepared, Some(&effective)).unwrap();
    assert!(!explicit);
    assert!(!headers.contains_key("x-remove-me"));
    assert!(!headers.contains_key(CONTENT_ENCODING));
    assert!(!headers.contains_key(INTERNAL_DISPATCH_BACKEND_HEADER));
    assert!(!headers.contains_key(INTERNAL_RETRY_AWARE_HEADER));
    assert_eq!(headers["x-json"], r#"{"nested":true}"#);
    assert_ne!(body, prepared.body);

    let null_content = LlmRequest {
        headers: serde_json::Map::new(),
        content: Value::Null,
    };
    assert_eq!(
        effective_request(&prepared, Some(&null_content)).unwrap().1,
        prepared.body
    );
    assert!(!request_streaming_hint(&HeaderMap::from_iter([(
        ACCEPT,
        HeaderValue::from_static("application/json"),
    )])));
    assert!(!response_streaming(&HeaderMap::from_iter([(
        CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    )])));
    assert_eq!(
        provider_surface("/responses"),
        Some(ProviderSurface::OpenAIResponses)
    );
    assert_eq!(
        provider_surface("/chat/completions"),
        Some(ProviderSurface::OpenAIChat)
    );
    assert_eq!(
        provider_surface("/v1/chat/completions"),
        Some(ProviderSurface::OpenAIChat)
    );
    assert_eq!(
        provider_surface("/backend-api/codex/responses"),
        Some(ProviderSurface::OpenAIResponses)
    );

    let unknown = ObservedResponse {
        value: None,
        truncated: false,
        terminal: 99,
        status: StatusCode::OK,
        failure: Some("unexpected".into()),
    };
    assert_eq!(
        unknown.metadata()["daemon_worker_observation"]["terminal"],
        "unknown"
    );
}

#[cfg(unix)]
#[test]
fn managed_observation_and_provider_environment_values_reject_invalid_inputs() {
    use std::os::unix::ffi::OsStrExt;

    let _environment = EnvScope::set(&[
        (
            OBSERVATION_CAPTURE_BYTES_ENV,
            Some(std::ffi::OsStr::from_bytes(b"\xff")),
        ),
        ("OPENAI_API_KEY", Some(OsStr::new("   "))),
        ("ANTHROPIC_API_KEY", None),
    ]);
    assert!(observation_capture_limit_from_environment().is_err());
    assert!(environment_value("OPENAI_API_KEY").is_none());
    assert!(environment_value("ANTHROPIC_API_KEY").is_none());
    let mut headers = HeaderMap::new();
    inject_provider_auth(
        &mut headers,
        ProviderRoute::OpenAi,
        &GatewayConfig {
            openai_auth_header: Some("bad\nvalue".into()),
            ..GatewayConfig::default()
        },
    );
    assert!(headers.is_empty());
}

#[tokio::test]
async fn observation_signal_waits_for_terminal_result_and_marks_truncation() {
    let signal = Arc::new(ObservationSignal::new());
    signal.truncate();
    let waiter = {
        let signal = Arc::clone(&signal);
        tokio::spawn(async move { signal.wait().await })
    };
    tokio::task::yield_now().await;
    signal.finish(OBSERVATION_COMPLETE);
    assert_eq!(waiter.await.expect("waiter task"), OBSERVATION_COMPLETE);
    assert!(signal.truncated.load(AtomicOrdering::Acquire));
    signal.finish(OBSERVATION_CANCELLED);
    assert_eq!(
        signal.terminal.load(AtomicOrdering::Acquire),
        OBSERVATION_COMPLETE
    );
}

#[tokio::test]
async fn managed_hook_rejects_invalid_and_oversized_json_before_adapter_dispatch() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let runtime = ManagedRuntime::initialize(
        GatewayConfig {
            max_hook_payload_bytes: 8,
            ..GatewayConfig::default()
        },
        Vec::new(),
        "machine-owner".into(),
    )
    .await
    .expect("initialize managed runtime");

    for body in ["not-json", r#"{"payload":"too-large"}"#] {
        let response = runtime
            .handle_hook(
                HookRoute::Pi,
                Request::post("/hook")
                    .body(Body::from(body))
                    .expect("hook request"),
            )
            .await;
        assert!(response.status().is_client_error());
    }
    runtime.close().await.expect("close managed runtime");
}

#[tokio::test]
async fn observation_handles_empty_invalid_and_unsuccessful_provider_responses() {
    let (body, observation) = observe_body(
        http_body_util::Empty::<Bytes>::new(),
        StatusCode::BAD_REQUEST,
        DEFAULT_OBSERVATION_CAPTURE_BYTES,
        OperationalContext::new(),
    );
    assert!(body.is_end_stream());
    let observed = observation
        .finish(ProviderSurface::OpenAIResponses, false)
        .await;
    assert_eq!(observed.terminal, OBSERVATION_COMPLETE);
    assert!(observed.value.is_none());
    assert!(observed.failure.unwrap().contains("HTTP 400"));

    let malformed = Bytes::from_static(b"data: {not-json}\n\n");
    let (body, observation) = observe_body(
        Full::new(malformed.clone()),
        StatusCode::OK,
        DEFAULT_OBSERVATION_CAPTURE_BYTES,
        OperationalContext::new(),
    );
    assert_eq!(body.collect().await.unwrap().to_bytes(), malformed);
    let observed = observation
        .finish(ProviderSurface::OpenAIResponses, true)
        .await;
    assert!(observed.truncated);
    assert!(observed.value.is_none());
    assert!(observed.failure.unwrap().contains("truncated"));
}

#[tokio::test]
async fn malformed_permission_hooks_fail_closed_in_each_native_response_shape() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let runtime =
        ManagedRuntime::initialize(GatewayConfig::default(), Vec::new(), "machine-owner".into())
            .await
            .expect("initialize managed runtime");

    let codex = runtime
        .handle_hook(
            HookRoute::Codex,
            Request::post("/hook")
                .body(Body::from(
                    json!({
                        "session_id": "codex-session",
                        "hook_event_name": "PermissionRequest",
                        "tool_name": "shell",
                        "arguments": {"cmd": "pwd"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await;
    assert_eq!(codex.status(), StatusCode::OK);
    let codex: Value =
        serde_json::from_slice(&codex.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(codex["decision"], "deny");
    assert!(
        codex["reason"]
            .as_str()
            .is_some_and(|reason| !reason.is_empty())
    );

    let claude = runtime
        .handle_hook(
            HookRoute::Claude,
            Request::post("/hook")
                .body(Body::from(
                    json!({
                        "session_id": "claude-session",
                        "hook_event_name": "PermissionRequest",
                        "tool_name": "Write",
                        "tool_input": {"file_path": "README.md"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await;
    assert_eq!(claude.status(), StatusCode::OK);
    let claude: Value =
        serde_json::from_slice(&claude.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(claude["hookSpecificOutput"]["decision"]["behavior"], "deny");
    assert!(
        claude["hookSpecificOutput"]["decision"]["message"]
            .as_str()
            .is_some_and(|message| !message.is_empty())
    );

    runtime.close().await.expect("close managed runtime");
}
