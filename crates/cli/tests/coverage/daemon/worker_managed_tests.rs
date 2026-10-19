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
use nemo_relay::api::registry::{
    deregister_llm_execution_intercept, deregister_llm_request_intercept,
    deregister_llm_stream_execution_intercept, register_llm_execution_intercept,
    register_llm_request_intercept, register_llm_stream_execution_intercept,
};

use crate::daemon::common::transport::pooled_client;
use crate::test_support::{EnvScope, PLUGIN_CONFIG_TEST_LOCK};

type CapturedProviderRequest = Arc<std::sync::Mutex<Option<(HeaderMap, Bytes)>>>;
type ProviderRequests = Arc<std::sync::Mutex<Vec<(HeaderMap, Bytes)>>>;

#[tokio::test]
async fn managed_provider_wait_has_no_default_deadline_and_honors_configuration() {
    for (streaming, response_timeout_secs) in [(false, 0), (true, 0), (false, 90), (true, 90)] {
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let app = Router::new().route(
            "/",
            post({
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
        let destination = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let task = tokio::spawn(async move {
            request_worker_upstream(
                pooled_client().unwrap(),
                false,
                Request::post(destination)
                    .body(box_body(Body::empty()))
                    .unwrap(),
                &OperationalContext::new(),
                streaming,
                &GatewayConfig {
                    response_timeout_secs,
                    ..GatewayConfig::default()
                },
            )
            .await
        });
        tokio::time::timeout(Duration::from_secs(5), started.notified())
            .await
            .unwrap();
        tokio::time::pause();
        tokio::time::advance(Duration::from_secs(61)).await;
        tokio::task::yield_now().await;
        assert!(!task.is_finished());
        tokio::time::advance(Duration::from_secs(3600)).await;
        tokio::task::yield_now().await;
        tokio::time::resume();
        if response_timeout_secs == 0 {
            assert!(!task.is_finished());
            release.notify_one();
            let response = task.await.unwrap().unwrap();
            assert_eq!(
                response.into_body().collect().await.unwrap().to_bytes(),
                "compaction complete"
            );
        } else {
            let error = task.await.unwrap().unwrap_err();
            assert!(matches!(&error, CliError::ProviderFailure(failure)
                if failure.class == UpstreamFailureClass::Timeout));
            assert!(error.to_string().contains("response-head timeout"));
            assert_eq!(error.into_response().status(), StatusCode::BAD_GATEWAY);
        }
        server.abort();
    }
}

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
async fn stream_observation_rearms_stall_warning_after_each_silent_interval() {
    let (sender, receiver) = tokio::sync::mpsc::channel(2);
    let operational = OperationalContext::new();
    let mut observation = ObservationReceiver {
        receiver,
        signal: Arc::new(ObservationSignal::new()),
        status: StatusCode::OK,
        operational: operational.clone(),
    };
    let mut first_event_warned = false;
    let mut last_event_at = tokio::time::Instant::now();
    let threshold =
        Duration::from_millis(crate::operational::UPSTREAM_STREAM_STALL_THRESHOLD_MILLIS);

    let delayed_sender = sender.clone();
    let send = tokio::spawn(async move {
        tokio::time::sleep(threshold.saturating_mul(2) + Duration::from_millis(1)).await;
        delayed_sender
            .send(Bytes::from_static(b"after-stall"))
            .await
            .unwrap();
    });
    assert_eq!(
        observation
            .next_stream_chunk(false, &mut first_event_warned, &mut last_event_at)
            .await,
        Some(Bytes::from_static(b"after-stall"))
    );
    send.await.unwrap();

    assert_eq!(
        crate::operational::test_delayed_event_count(&operational, "upstream_stream_stalled"),
        2
    );
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
    tokio::time::advance(Duration::from_secs(61)).await;
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

#[tokio::test]
async fn worker_requirements_follow_registration_and_removal_without_running_gates() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    use nemo_relay::api::registry::{
        deregister_conditional_middleware_guardrail, deregister_llm_sanitize_request_guardrail,
        register_conditional_middleware_guardrail, register_llm_sanitize_request_guardrail,
    };
    struct PresenceRegistrations;

    impl Drop for PresenceRegistrations {
        fn drop(&mut self) {
            let _ = deregister_conditional_middleware_guardrail("presence-gate");
            let _ = deregister_llm_execution_intercept("presence-execution");
            let _ = deregister_llm_sanitize_request_guardrail("presence-sanitize");
        }
    }

    let _registrations = PresenceRegistrations;
    let requirements = provider_middleware_requirements().unwrap();
    assert!(!requirements.request_body_decode_required);
    assert!(!requirements.managed_execution_required);
    register_llm_sanitize_request_guardrail(
        "presence-sanitize",
        1,
        Arc::new(|_, _| Box::pin(async move { Ok(None) })),
    )
    .unwrap();
    let requirements = provider_middleware_requirements().unwrap();
    assert!(requirements.request_body_decode_required);
    assert!(!requirements.managed_execution_required);
    register_llm_execution_intercept(
        "presence-execution",
        1,
        Arc::new(|_, request, _, next| next(request)),
    )
    .unwrap();
    register_conditional_middleware_guardrail(
        "presence-gate",
        std::collections::BTreeSet::from([RuntimeRegistrationKind::LlmExecutionIntercept]),
        "presence-execution",
        Arc::new(|_, _| panic!("presence queries must not run gates")),
    )
    .unwrap();
    let requirements = provider_middleware_requirements().unwrap();
    assert!(requirements.request_body_decode_required);
    assert!(requirements.managed_execution_required);
    assert!(deregister_llm_execution_intercept("presence-execution").unwrap());
    assert!(deregister_llm_sanitize_request_guardrail("presence-sanitize").unwrap());
    assert!(deregister_conditional_middleware_guardrail("presence-gate").unwrap());
    let requirements = provider_middleware_requirements().unwrap();
    assert!(!requirements.request_body_decode_required);
    assert!(!requirements.managed_execution_required);
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

#[tokio::test]
async fn unbuffered_dispatch_uses_request_streaming_hint_for_upstream_telemetry() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind provider");
    let address = listener.local_addr().expect("provider address");
    let app = Router::new().route(
        "/v1/responses",
        post(|| async {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from("{}"))
                .expect("provider response")
        }),
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve provider");
    });
    let request = Request::post("/v1/responses")
        .body(Body::empty())
        .expect("buffered request");
    let config = GatewayConfig {
        openai_base_url: format!("http://{address}"),
        ..GatewayConfig::default()
    };
    let operational = OperationalContext::new();

    let (response, observation, streaming) = dispatch_unbuffered_observed(
        crate::daemon::common::transport::pooled_client().expect("provider client"),
        request,
        ProviderRoute::OpenAi,
        &config,
        DEFAULT_OBSERVATION_CAPTURE_BYTES,
        operational.clone(),
    )
    .await
    .expect("provider response");

    assert!(!streaming);
    assert_eq!(
        crate::operational::test_upstream_started_outcomes(&operational),
        ["buffered"]
    );
    let _ = response
        .into_body()
        .collect()
        .await
        .expect("delivered response");
    let observed = observation
        .finish(ProviderSurface::OpenAIResponses, false)
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
        request_json: Some(json!({"model": "test", "stream": true})),
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
fn managed_worker_routes_chatgpt_responses_to_the_configured_chatgpt_base() {
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
        request_json: Some(json!({"model": "test", "stream": true})),
        streaming: true,
    };
    let config = GatewayConfig {
        chatgpt_codex_base_url: Some("https://llm-proxy.example/chatgpt/".into()),
        ..GatewayConfig::default()
    };

    let destination = effective_destination(&prepared, ProviderRoute::OpenAi, None, &config)
        .expect("ChatGPT destination");

    assert_eq!(
        destination,
        "https://llm-proxy.example/chatgpt/responses?client=codex"
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
        HeaderValue::from_static("Bearer opaque-caller-controlled-token"),
    );
    let prepared = PreparedProviderRequest {
        method: Method::POST,
        version: http::Version::HTTP_11,
        headers,
        path: "/responses".into(),
        path_and_query: "/responses?client=pi".into(),
        body: Bytes::from_static(br#"{"model":"test","stream":true}"#),
        request_json: Some(json!({"model": "test", "stream": true})),
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
        request_json: Some(json!({"model": "custom", "stream": true})),
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
        content: prepared.request_json.clone().unwrap(),
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
        content: prepared.request_json.clone().unwrap(),
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
        content: prepared.request_json.clone().unwrap(),
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
        request_json: Some(json!({"model": "test", "stream": true})),
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
    assert_eq!(prepared.request_json, Some(Value::Null));
    assert!(!prepared.streaming);
}

#[tokio::test]
async fn managed_request_helpers_cover_overrides_streaming_and_provider_credentials() {
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
    )
    .await
    .unwrap();
    assert_eq!(headers[AUTHORIZATION], "Bearer openai-env-key");
    headers.clear();
    inject_provider_auth(
        &mut headers,
        ProviderRoute::Anthropic,
        &GatewayConfig::default(),
    )
    .await
    .unwrap();
    assert_eq!(headers["x-api-key"], "anthropic-env-key");
    headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer caller"));
    inject_provider_auth(
        &mut headers,
        ProviderRoute::OpenAi,
        &GatewayConfig::default(),
    )
    .await
    .unwrap();
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

    assert!(
        !runtime
            .provider_middleware_requirements()
            .unwrap()
            .managed_execution_required
    );
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
        matches!(&error, CliError::ProviderFailure(failure)
            if failure.class == UpstreamFailureClass::Connection),
        "unexpected dispatch error: {error:?}"
    );
    runtime.close().await.expect("close managed runtime");
}

#[test]
fn managed_request_body_preserves_null_edits_and_unchanged_wire_bytes() {
    let mut prepared = prepared_request(HeaderMap::from_iter([
        (CONTENT_ENCODING, HeaderValue::from_static("gzip")),
        (CONTENT_LENGTH, HeaderValue::from_static("123")),
    ]));
    let effective = LlmRequest {
        headers: crate::gateway::daemon_observable_headers(&prepared.headers),
        content: Value::Null,
    };
    let (headers, body, _) = effective_request(&prepared, Some(&effective)).unwrap();
    assert_eq!(body, Bytes::from_static(b"null"));
    assert!(!headers.contains_key(CONTENT_ENCODING));
    assert!(!headers.contains_key(CONTENT_LENGTH));

    prepared.request_json = Some(Value::Null);
    prepared.body = Bytes::from_static(b" \n null \t");
    let (headers, body, _) = effective_request(&prepared, Some(&effective)).unwrap();
    assert_eq!(body, prepared.body);
    assert_eq!(headers[CONTENT_ENCODING], "gzip");
    assert!(!headers.contains_key(CONTENT_LENGTH));

    prepared.request_json = None;
    prepared.body = Bytes::from_static(b"opaque unparsed bytes");
    let (headers, body, _) = effective_request(&prepared, Some(&effective)).unwrap();
    assert_eq!(body, prepared.body);
    assert_eq!(headers[CONTENT_ENCODING], "gzip");
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
        let _ = deregister_llm_stream_execution_intercept(self.0);
    }
}

struct LlmGuardrailRegistration(&'static str);

impl Drop for LlmGuardrailRegistration {
    fn drop(&mut self) {
        use nemo_relay::api::registry::*;
        let _ = deregister_llm_conditional_execution_guardrail(self.0);
        let _ = deregister_llm_sanitize_request_guardrail(self.0);
        let _ = deregister_llm_sanitize_response_guardrail(self.0);
        let _ = nemo_relay::api::subscriber::deregister_subscriber(self.0);
    }
}

#[tokio::test]
async fn observed_responses_preserve_sanitized_payloads_and_usage_without_a_fallback_codec() {
    use nemo_relay::api::event::{Event, ScopeCategory};
    use nemo_relay::api::subscriber::{flush_subscribers, register_subscriber};

    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const SUBSCRIBER: &str = "daemon-worker-responses-pii";
    let _registration = LlmGuardrailRegistration(SUBSCRIBER);
    let recorded = Arc::new(std::sync::Mutex::new(Vec::<Event>::new()));
    register_subscriber(
        SUBSCRIBER,
        Arc::new({
            let recorded = recorded.clone();
            move |event| recorded.lock().unwrap().push(event.clone())
        }),
    )
    .unwrap();

    let provider_response = json!({
        "id": "resp_pii", "model": "test", "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "sk-response-secret"}
        ]}],
        "usage": {"input_tokens": 17, "output_tokens": 5, "total_tokens": 22}
    });
    let app = Router::new().route(
        "/v1/responses",
        post({
            let provider_response = provider_response.clone();
            move |request: Request<Body>| {
                let provider_response = provider_response.clone();
                async move {
                    let body = axum::body::to_bytes(request.into_body(), usize::MAX)
                        .await
                        .unwrap();
                    let request: Value = serde_json::from_slice(&body).unwrap();
                    assert_eq!(request["input"], "sk-request-secret");
                    let expected = format!(r#" {{ "model": "test", "input": "sk-request-\u0073ecret", "stream": {} }} "#, request["stream"]);
                    assert_eq!(body.as_ref(), expected.as_bytes(), "unchanged bodies must retain their wire bytes");
                    let (content_type, body) = if request["stream"] == true {
                        (
                            "text/event-stream",
                            format!(
                                "data: {}\n\n",
                                json!({
                                    "type": "response.completed", "response": provider_response
                                })
                            ),
                        )
                    } else {
                        ("application/json", provider_response.to_string())
                    };
                    Response::builder()
                        .header(CONTENT_TYPE, content_type)
                        .body(Body::from(body))
                        .unwrap()
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let runtime = ManagedRuntime::initialize(
        GatewayConfig {
            openai_base_url: format!("http://{address}"),
            plugin_config: Some(json!({
                "version": 1,
                "components": [{
                    "kind": "pii_redaction", "enabled": true,
                    "config": {
                        "mode": "builtin", "tool_input": false, "tool_output": false,
                        "builtin": {
                            "action": "regex_replace", "pattern": "sk-[A-Za-z0-9_-]+",
                            "replacement": "[REDACTED]",
                            "target_paths": ["/messages/0/content", "/message"]
                        }
                    }
                }]
            })),
            ..GatewayConfig::default()
        },
        Vec::new(),
        "machine-owner".into(),
    )
    .await
    .unwrap();
    let requirements = runtime.provider_middleware_requirements().unwrap();
    assert!(requirements.request_body_decode_required);
    assert!(!requirements.managed_execution_required);
    for streaming in [false, true] {
        let response = runtime
            .proxy_provider(
                pooled_client().unwrap(),
                Request::post("/v1/responses")
                    .body(Body::from(format!(
                        r#" {{ "model": "test", "input": "sk-request-\u0073ecret", "stream": {streaming} }} "#
                    )))
                    .unwrap(),
                ProviderRoute::OpenAi,
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert!(String::from_utf8_lossy(&body).contains("sk-response-secret"));
    }
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            flush_subscribers().unwrap();
            let end_count = recorded
                .lock()
                .unwrap()
                .iter()
                .filter(|event| {
                    event.scope_category() == Some(ScopeCategory::End)
                        && event.annotated_response().is_some()
                })
                .count();
            if end_count == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("both observed responses must finish");
    {
        let events = recorded.lock().unwrap();
        let starts = events
            .iter()
            .filter(|event| {
                event.scope_category() == Some(ScopeCategory::Start)
                    && event.annotated_request().is_some()
            })
            .collect::<Vec<_>>();
        assert_eq!(starts.len(), 2);
        for start in starts {
            assert_eq!(start.input().unwrap()["content"]["input"], "[REDACTED]");
            assert!(
                !start
                    .to_json_string()
                    .unwrap()
                    .contains("sk-request-secret")
            );
        }
        for end in events
            .iter()
            .filter(|event| event.annotated_response().is_some())
        {
            let annotated = end.annotated_response().unwrap();
            assert_eq!(annotated.response_text(), Some("[REDACTED]"));
            let usage = serde_json::to_value(annotated.usage.as_ref().unwrap()).unwrap();
            assert_eq!(usage["prompt_tokens"], 17);
            assert_eq!(usage["completion_tokens"], 5);
            assert_eq!(usage["total_tokens"], 22);
            assert!(!end.to_json_string().unwrap().contains("sk-response-secret"));
        }
    }
    runtime.close().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn managed_runtime_accepts_execution_middleware_and_replaces_response() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const INTERCEPT: &str = "daemon-worker-execution-replacement";
    let _registration = ExecutionInterceptRegistration(INTERCEPT);
    register_llm_execution_intercept(
        INTERCEPT,
        1,
        Arc::new(|_name, _request, context, _next| {
            assert!(context.response_codec().is_some());
            Box::pin(async { Ok(json!({"id": "replacement", "output": []})) })
        }),
    )
    .unwrap();
    // No provider is listening: an intercept can complete the call without dispatch.
    let runtime = ManagedRuntime::initialize(
        GatewayConfig {
            openai_base_url: "http://127.0.0.1:1".into(),
            ..GatewayConfig::default()
        },
        Vec::new(),
        "machine-owner".into(),
    )
    .await
    .unwrap();
    let requirements = runtime.provider_middleware_requirements().unwrap();
    assert!(requirements.request_body_decode_required);
    assert!(requirements.managed_execution_required);
    let response = runtime
        .proxy_provider(
            pooled_client().unwrap(),
            Request::post("/v1/responses")
                .body(Body::from(r#"{"model":"test","input":"hello"}"#))
                .unwrap(),
            ProviderRoute::OpenAi,
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        serde_json::from_slice::<Value>(&body).unwrap()["id"],
        "replacement"
    );
    runtime.close().await.unwrap();
}

#[tokio::test]
async fn managed_runtime_executes_unary_and_stream_middleware_before_delivery() {
    use futures_util::StreamExt;
    use nemo_relay::api::event::PendingMarkSpec;
    use nemo_relay::api::registry::{
        register_llm_conditional_execution_guardrail, register_llm_sanitize_request_guardrail,
        register_llm_sanitize_response_guardrail,
    };
    use nemo_relay::api::runtime::LlmJsonStream;
    use nemo_relay::api::subscriber::{flush_subscribers, register_subscriber};

    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const INTERCEPT: &str = "daemon-worker-execution-transform";
    const REQUEST: &str = "daemon-worker-execution-request";
    let _registration = ExecutionInterceptRegistration(INTERCEPT);
    let _request_registration = RequestInterceptRegistration(REQUEST);
    let _guardrail_registration = LlmGuardrailRegistration(INTERCEPT);
    let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
    register_subscriber(
        INTERCEPT,
        Arc::new({
            let recorded = recorded.clone();
            move |event| recorded.lock().unwrap().push(event.clone())
        }),
    )
    .unwrap();
    let policy_calls = Arc::new(AtomicUsize::new(0));
    register_llm_conditional_execution_guardrail(
        INTERCEPT,
        1,
        Arc::new({
            let calls = policy_calls.clone();
            move |_| {
                calls.fetch_add(1, AtomicOrdering::Relaxed);
                Box::pin(async { Ok(None) })
            }
        }),
    )
    .unwrap();
    register_llm_sanitize_request_guardrail(
        INTERCEPT,
        1,
        Arc::new(|mut request, _| {
            Box::pin(async move {
                request.content["messages"][0]["content"] = json!("recorded request");
                Ok(Some(request))
            })
        }),
    )
    .unwrap();
    register_llm_sanitize_response_guardrail(
        INTERCEPT,
        1,
        Arc::new(|mut response, _| {
            Box::pin(async move {
                response["choices"][0]["message"]["content"] = json!("recorded response");
                Ok(Some(response))
            })
        }),
    )
    .unwrap();
    register_llm_request_intercept(
        REQUEST,
        1,
        false,
        Arc::new(|_, mut request, annotated| {
            request
                .headers
                .insert("x-request-intercept".into(), json!("applied"));
            Box::pin(async move {
                Ok(
                    LlmRequestInterceptOutcome::new(request, annotated).with_pending_mark(
                        PendingMarkSpec::builder()
                            .name("daemon.request.intercept.mark")
                            .build(),
                    ),
                )
            })
        }),
    )
    .unwrap();
    register_llm_execution_intercept(
        INTERCEPT,
        1,
        Arc::new(|_, request, _, next| {
            Box::pin(async move {
                let mut result = next(request).await?;
                result["choices"][0]["message"]["content"] = json!("unary middleware");
                Ok(result)
            })
        }),
    )
    .unwrap();
    register_llm_stream_execution_intercept(
        INTERCEPT,
        1,
        Arc::new(|_, request, context, next| {
            assert!(context.response_codec().is_none());
            Box::pin(async move {
                let stream = next(request).await?;
                Ok(LlmJsonStream::new(stream.map(|item| {
                    item.map(|mut chunk| {
                        chunk["choices"][0]["delta"]["content"] = json!("stream middleware");
                        chunk
                    })
                })))
            })
        }),
    )
    .unwrap();
    let release = Arc::new(Notify::new());
    let app = Router::new().route("/v1/chat/completions", post({
        let release = release.clone();
        move |request: Request<Body>| {
            let release = release.clone();
            async move {
                assert_eq!(request.headers()["x-request-intercept"], "applied");
                assert!(!request.headers().contains_key(WORKER_TOKEN_HEADER));
                assert!(!request.headers().contains_key(INTERNAL_DISPATCH_URL_HEADER));
                let body = axum::body::to_bytes(request.into_body(), usize::MAX).await.unwrap();
                let request: Value = serde_json::from_slice(&body).unwrap();
                // Observability sanitizers must not rewrite the provider's input.
                assert_eq!(request["messages"][0]["content"], "hello");
                if request["stream"] == true {
                    let stream = async_stream::stream! {
                        yield Ok::<_, Infallible>(Bytes::from_static(
                            b"data: {\"id\":\"test\",\"object\":\"chat.completion.chunk\",\"model\":\"test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"provider\"}}]}\n\n"));
                        release.notified().await;
                        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                    };
                    Response::builder().header(CONTENT_TYPE, "text/event-stream")
                        .body(Body::from_stream(stream)).unwrap()
                } else {
                    Response::builder().header(CONTENT_TYPE, "application/json")
                        .body(Body::from(json!({"id":"test", "object":"chat.completion", "model":"test",
                            "choices":[{"index":0,"message":{"role":"assistant","content":"provider"},"finish_reason":"stop"}]
                        }).to_string())).unwrap()
                }
            }
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let runtime = ManagedRuntime::initialize(
        GatewayConfig {
            openai_base_url: format!("http://{address}"),
            response_timeout_secs: 1,
            ..GatewayConfig::default()
        },
        Vec::new(),
        "machine-owner".into(),
    )
    .await
    .unwrap();
    for streaming in [false, true] {
        let response = runtime.proxy_provider(
            pooled_client().unwrap(),
            Request::post("/v1/chat/completions")
                .header(WORKER_TOKEN_HEADER, "hop-secret")
                .header(INTERNAL_DISPATCH_URL_HEADER, "http://untrusted.invalid")
                .body(Body::from(json!({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":streaming}).to_string())).unwrap(),
            ProviderRoute::OpenAi,
        ).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        if streaming {
            let mut body = response.into_body();
            // The first transformed chunk arrives while the provider is still streaming.
            let first = tokio::time::timeout(Duration::from_secs(5), body.frame())
                .await
                .unwrap()
                .unwrap()
                .unwrap()
                .into_data()
                .unwrap();
            assert!(String::from_utf8_lossy(&first).contains("stream middleware"));
            tokio::time::pause();
            tokio::time::advance(Duration::from_secs(2)).await;
            release.notify_one();
            let rest = body.collect().await.unwrap().to_bytes();
            assert!(String::from_utf8_lossy(&rest).contains("data: [DONE]"));
            tokio::time::resume();
        } else {
            let body = response.into_body().collect().await.unwrap().to_bytes();
            let result: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                result["choices"][0]["message"]["content"],
                "unary middleware"
            );
        }
    }
    assert_eq!(policy_calls.load(AtomicOrdering::Relaxed), 2);
    flush_subscribers().unwrap();
    {
        let events = recorded.lock().unwrap();
        assert!(events.iter().any(|event| {
            event
                .data()
                .is_some_and(|data| data["content"]["messages"][0]["content"] == "recorded request")
        }));
        assert!(events.iter().any(|event| {
            event
                .data()
                .is_some_and(|data| data["choices"][0]["message"]["content"] == "recorded response")
        }));
        assert_eq!(
            events
                .iter()
                .filter(|event| event.name() == "daemon.request.intercept.mark")
                .count(),
            2
        );
    }
    runtime.close().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn managed_runtime_conditional_execution_blocks_before_intercepts_or_provider() {
    use nemo_relay::api::registry::register_llm_conditional_execution_guardrail;

    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const GUARDRAIL: &str = "daemon-worker-conditional-block";
    let _registration = LlmGuardrailRegistration(GUARDRAIL);
    let _request_registration = RequestInterceptRegistration(GUARDRAIL);
    let intercept_calls = Arc::new(AtomicUsize::new(0));
    register_llm_request_intercept(
        GUARDRAIL,
        1,
        false,
        Arc::new({
            let calls = intercept_calls.clone();
            move |_, request, annotated| {
                calls.fetch_add(1, AtomicOrdering::Relaxed);
                Box::pin(async move { Ok(LlmRequestInterceptOutcome::new(request, annotated)) })
            }
        }),
    )
    .unwrap();
    register_llm_conditional_execution_guardrail(
        GUARDRAIL,
        1,
        Arc::new(|_| Box::pin(async { Ok(Some("blocked by worker policy".into())) })),
    )
    .unwrap();
    let runtime = ManagedRuntime::initialize(
        GatewayConfig {
            openai_base_url: "http://127.0.0.1:1".into(),
            ..GatewayConfig::default()
        },
        Vec::new(),
        "machine-owner".into(),
    )
    .await
    .unwrap();
    for streaming in [false, true] {
        let error = runtime.proxy_provider(pooled_client().unwrap(),
            Request::post("/v1/chat/completions")
                .body(Body::from(json!({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":streaming}).to_string())).unwrap(),
            ProviderRoute::OpenAi,
        ).await.unwrap_err();
        assert!(error.to_string().contains("blocked by worker policy"));
    }
    assert_eq!(intercept_calls.load(AtomicOrdering::Relaxed), 0);
    runtime.close().await.unwrap();
}

#[tokio::test]
async fn managed_runtime_execution_cancellation_releases_session() {
    use nemo_relay::api::runtime::LlmJsonStream;

    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const INTERCEPT: &str = "daemon-worker-execution-cancellation";
    let _registration = ExecutionInterceptRegistration(INTERCEPT);
    let started = Arc::new(Notify::new());
    register_llm_execution_intercept(
        INTERCEPT,
        1,
        Arc::new({
            let started = started.clone();
            move |_, _, _, _| {
                let started = started.clone();
                Box::pin(async move {
                    started.notify_one();
                    std::future::pending().await
                })
            }
        }),
    )
    .unwrap();
    register_llm_stream_execution_intercept(
        INTERCEPT,
        1,
        Arc::new(|_, _, _, _| {
            Box::pin(async { Ok(LlmJsonStream::new(futures_util::stream::pending())) })
        }),
    )
    .unwrap();
    let runtime = Arc::new(
        ManagedRuntime::initialize(GatewayConfig::default(), Vec::new(), "machine-owner".into())
            .await
            .unwrap(),
    );
    let request = |streaming| {
        Request::post("/v1/chat/completions")
        .body(Body::from(json!({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":streaming}).to_string())).unwrap()
    };
    let task = tokio::spawn({
        let runtime = runtime.clone();
        let request = request(false);
        async move {
            runtime
                .proxy_provider(pooled_client().unwrap(), request, ProviderRoute::OpenAi)
                .await
        }
    });
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .unwrap();
    assert!(runtime.sessions.has_open_sessions().await);
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(5), async {
        while runtime.sessions.has_open_sessions().await {
            runtime
                .sessions
                .close_idle_sessions_at(
                    std::time::Instant::now(),
                    Duration::ZERO,
                    "cancelled_execution_test_call",
                )
                .await
                .unwrap();
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    let response = runtime
        .proxy_provider(
            pooled_client().unwrap(),
            request(true),
            ProviderRoute::OpenAi,
        )
        .await
        .unwrap();
    assert!(runtime.sessions.has_open_sessions().await);
    drop(response);
    tokio::time::timeout(Duration::from_secs(5), async {
        while runtime.sessions.has_open_sessions().await {
            runtime
                .sessions
                .close_idle_sessions_at(
                    std::time::Instant::now(),
                    Duration::ZERO,
                    "cancelled_execution_test_call",
                )
                .await
                .unwrap();
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    runtime.close().await.unwrap();
}

#[tokio::test]
async fn managed_runtime_execution_response_setup_honors_deadline() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const INTERCEPT: &str = "daemon-worker-execution-timeout";
    let _registration = ExecutionInterceptRegistration(INTERCEPT);
    register_llm_execution_intercept(
        INTERCEPT,
        1,
        Arc::new(|_, _, _, _| Box::pin(std::future::pending())),
    )
    .unwrap();
    let runtime = ManagedRuntime::initialize(
        GatewayConfig {
            response_timeout_secs: 1,
            ..GatewayConfig::default()
        },
        Vec::new(),
        "machine-owner".into(),
    )
    .await
    .unwrap();
    tokio::time::pause();
    let result = runtime
        .proxy_provider(
            pooled_client().unwrap(),
            Request::post("/v1/responses")
                .body(Body::from(r#"{"model":"test","input":"hello"}"#))
                .unwrap(),
            ProviderRoute::OpenAi,
        )
        .await;
    tokio::time::resume();
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("response-head timeout")
    );
    tokio::time::timeout(Duration::from_secs(5), async {
        while runtime.sessions.has_open_sessions().await {
            runtime
                .sessions
                .close_idle_sessions_at(
                    std::time::Instant::now(),
                    Duration::ZERO,
                    "timed_out_execution_test_call",
                )
                .await
                .unwrap();
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    runtime.close().await.unwrap();
}

#[tokio::test]
async fn managed_runtime_execution_preserves_selected_provider_error() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const INTERCEPT: &str = "daemon-worker-execution-error";
    let _registration = ExecutionInterceptRegistration(INTERCEPT);
    register_llm_execution_intercept(
        INTERCEPT,
        1,
        Arc::new(|_, request, _, next| Box::pin(async move { next(request).await })),
    )
    .unwrap();
    let app = Router::new().route(
        "/v1/responses",
        post(|| async {
            Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .header("retry-after", "3")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from("{\"error\":\"busy\"}\n"))
                .unwrap()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let runtime = ManagedRuntime::initialize(
        GatewayConfig {
            openai_base_url: format!("http://{address}"),
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
            Request::post("/v1/responses")
                .body(Body::from(r#"{"model":"test","input":"hello"}"#))
                .unwrap(),
            ProviderRoute::OpenAi,
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()["retry-after"], "3");
    assert_eq!(
        response.into_body().collect().await.unwrap().to_bytes(),
        "{\"error\":\"busy\"}\n"
    );
    runtime.close().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn managed_runtime_applies_request_middleware_before_provider_delivery() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const INTERCEPT: &str = "daemon-worker-managed-request-coverage";
    let _ = deregister_llm_request_intercept(INTERCEPT);
    let _registration = RequestInterceptRegistration(INTERCEPT);
    register_llm_request_intercept(
        INTERCEPT,
        1,
        false,
        Arc::new(|_name, mut request, mut annotated| {
            request
                .headers
                .insert("x-worker-intercept".into(), json!("applied"));
            annotated.as_mut().unwrap().messages =
                vec![nemo_relay::codec::request::Message::User {
                    content: nemo_relay::codec::request::MessageContent::Text(
                        "rewritten by middleware".into(),
                    ),
                    name: None,
                }];
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
                        .body(Body::from("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"test\",\"output\":[]}}\n\n"))
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
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let mut decoder = SseEventDecoder::new();
    let events = decoder.push_bytes_results(&body);
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0].as_ref().unwrap().data,
        json!({"type":"response.completed","response":{"id":"test","output":[]}})
    );
    let (headers, body) = captured
        .lock()
        .expect("captured request")
        .take()
        .expect("provider saw request");
    assert_eq!(headers["x-worker-intercept"], "applied");
    assert!(
        serde_json::from_slice::<Value>(&body).unwrap()["input"]
            .to_string()
            .contains("rewritten by middleware")
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
        Arc::new(|_name, request, mut annotated| {
            annotated.as_mut().unwrap().stream = Some(false);
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
    assert!(error.to_string().contains("cannot change stream mode"));
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
    assert!(
        !provider_middleware_requirements()
            .unwrap()
            .managed_execution_required
    );
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
        Bytes::from_static(b"null")
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
#[tokio::test]
async fn managed_observation_and_provider_environment_values_reject_invalid_inputs() {
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
    let mut headers = HeaderMap::new();
    inject_provider_auth(
        &mut headers,
        ProviderRoute::OpenAi,
        &GatewayConfig {
            openai_auth_header: Some("bad\nvalue".into()),
            ..GatewayConfig::default()
        },
    )
    .await
    .unwrap_err();
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

#[cfg(unix)]
#[tokio::test]
async fn unmanaged_daemon_dispatch_generates_fresh_helper_tokens() {
    let seen = Arc::new(Mutex::new(Vec::<HeaderMap>::new()));
    let capture = seen.clone();
    let app = Router::new().route(
        "/v1/messages/count_tokens",
        post(move |headers: HeaderMap| {
            let capture = capture.clone();
            async move {
                capture.lock().unwrap().push(headers);
                "{}"
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let config = GatewayConfig {
        anthropic_base_url: format!("http://{}", listener.local_addr().unwrap()),
        anthropic_api_key_helper: Some(crate::upstream_auth::tests::helper("printf 'token-%s' $$")),
        ..GatewayConfig::default()
    };
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    for _ in 0..2 {
        let request = Request::post("/v1/messages/count_tokens")
            .header(AUTHORIZATION, "Bearer caller-token")
            .body(Body::from("{}"))
            .unwrap();
        let response = dispatch_unmanaged(
            pooled_client().unwrap(),
            request,
            ProviderRoute::Anthropic,
            &config,
        )
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        response.into_body().collect().await.unwrap();
    }
    let captured = seen.lock().unwrap();
    assert_eq!(captured.len(), 2);
    assert_ne!(captured[0]["x-api-key"], captured[1]["x-api-key"]);
    assert!(!captured[0].contains_key(AUTHORIZATION));
    server.abort();
}

#[cfg(unix)]
#[tokio::test]
async fn regression_backend_alias_preserves_caller_credentials_with_helper() {
    let seen = Arc::new(Mutex::new(None::<HeaderMap>));
    let capture = seen.clone();
    let app = Router::new().route(
        "/v1/realtime/calls",
        post(move |headers: HeaderMap| {
            let capture = capture.clone();
            async move {
                *capture.lock().unwrap() = Some(headers);
                "{}"
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let config = GatewayConfig {
        openai_base_url: format!("http://{}", listener.local_addr().unwrap()),
        openai_api_key_helper: Some(crate::upstream_auth::tests::helper("exit 9")),
        ..GatewayConfig::default()
    };
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let request = Request::post("/backend-api/codex/realtime/calls")
        .header(AUTHORIZATION, "Bearer stale-client-api-key")
        .body(Body::from("{}"))
        .unwrap();
    let response = dispatch_unmanaged(
        pooled_client().unwrap(),
        request,
        ProviderRoute::OpenAi,
        &config,
    )
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    response.into_body().collect().await.unwrap();
    let actual = seen.lock().unwrap().as_ref().unwrap()[AUTHORIZATION].clone();
    server.abort();
    assert_eq!(
        actual, "Bearer stale-client-api-key",
        "An unmanaged backend alias must retain caller credentials without executing the helper"
    );
}

#[tokio::test]
async fn managed_forwarding_rejects_remote_http_as_an_upstream_error() {
    let config = GatewayConfig {
        openai_base_url: "http://172.16.0.223:30919/v1".into(),
        ..Default::default()
    };
    let error = dispatch_unmanaged(
        pooled_client().unwrap(),
        Request::post("/v1/responses").body(Body::empty()).unwrap(),
        ProviderRoute::OpenAi,
        &config,
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        CliError::UpstreamTransport(TransportError::InsecureDestination)
    ));
    assert_eq!(error.into_response().status(), StatusCode::BAD_GATEWAY);

    let error = dispatch_unbuffered_observed(
        pooled_client().unwrap(),
        Request::post("/v1/responses").body(Body::empty()).unwrap(),
        ProviderRoute::OpenAi,
        &config,
        1024,
        OperationalContext::new(),
    )
    .await
    .err()
    .expect("remote HTTP must be rejected before connecting");
    assert!(matches!(
        error,
        CliError::UpstreamTransport(TransportError::InsecureDestination)
    ));
    assert!(!error.to_string().contains("hook payload"));
}

#[tokio::test]
async fn provider_connection_failures_return_bad_gateway_for_both_forwarding_paths() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let destination = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let config = GatewayConfig {
        openai_base_url: destination.clone(),
        ..Default::default()
    };
    let unmanaged = dispatch_unmanaged(
        pooled_client().unwrap(),
        Request::post("/v1/responses").body(Body::empty()).unwrap(),
        ProviderRoute::OpenAi,
        &config,
    )
    .await
    .unwrap_err();
    let managed = request_worker_upstream(
        pooled_client().unwrap(),
        false,
        Request::post(destination)
            .body(box_body(Body::empty()))
            .unwrap(),
        &OperationalContext::new(),
        false,
        &config,
    )
    .await
    .unwrap_err();
    for error in [unmanaged, managed] {
        assert!(matches!(&error, CliError::ProviderFailure(failure)
            if failure.class == UpstreamFailureClass::Connection && failure.status.is_none()));
        assert_eq!(error.into_response().status(), StatusCode::BAD_GATEWAY);
    }
}

#[tokio::test]
async fn unmanaged_provider_response_head_timeout_returns_bad_gateway() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let config = GatewayConfig {
        openai_base_url: format!("http://{}", listener.local_addr().unwrap()),
        response_timeout_secs: 1,
        ..Default::default()
    };
    let provider = tokio::spawn(async move {
        let (_connection, _) = listener.accept().await.unwrap();
        std::future::pending::<()>().await;
    });
    let error = dispatch_unmanaged(
        pooled_client().unwrap(),
        Request::post("/v1/responses").body(Body::empty()).unwrap(),
        ProviderRoute::OpenAi,
        &config,
    )
    .await
    .unwrap_err();
    assert!(matches!(&error, CliError::ProviderFailure(failure)
        if failure.class == UpstreamFailureClass::Timeout && failure.status.is_none()));
    assert!(error.to_string().contains("response-head timeout"));
    assert_eq!(error.into_response().status(), StatusCode::BAD_GATEWAY);
    provider.abort();
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
                let response = dispatch_unmanaged(
                    pooled_client().unwrap(),
                    request,
                    ProviderRoute::OpenAi,
                    &config,
                )
                .await
                .unwrap();
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
async fn managed_pre_tool_hooks_reject_when_conditional_guardrail_blocks() {
    use nemo_relay::api::registry::{
        deregister_tool_conditional_execution_guardrail,
        register_tool_conditional_execution_guardrail,
    };

    const GUARDRAIL: &str = "managed-pre-tool-blocker";
    const TOOL: &str = "ManagedBlockedTool";
    struct Cleanup;
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = deregister_tool_conditional_execution_guardrail(GUARDRAIL);
        }
    }
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = deregister_tool_conditional_execution_guardrail(GUARDRAIL);
    register_tool_conditional_execution_guardrail(
        GUARDRAIL,
        1,
        Arc::new(|name, _args| {
            Box::pin(async move { Ok((name == TOOL).then(|| "blocked by policy".to_string())) })
        }),
    )
    .unwrap();
    let _cleanup = Cleanup;
    let runtime =
        ManagedRuntime::initialize(GatewayConfig::default(), Vec::new(), "machine-owner".into())
            .await
            .unwrap();
    for route in [HookRoute::Claude, HookRoute::Codex] {
        let response = runtime
            .handle_hook(
                route,
                Request::post("/hook")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "session_id": "managed-pre-tool-blocked",
                            "hook_event_name": "PreToolUse",
                            "tool_use_id": "blocked-tool",
                            "tool_name": TOOL,
                            "tool_input": {"command": "pwd"},
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await;
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body["error"]["type"], "nemo_relay_guardrail_rejected");
        assert_eq!(body["error"]["reason"], "blocked by policy");
    }
    runtime.close().await.unwrap();
}
