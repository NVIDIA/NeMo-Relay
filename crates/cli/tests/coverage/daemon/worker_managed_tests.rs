// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::convert::Infallible;
use std::ffi::OsStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use axum::Router;
use axum::routing::post;
use http_body_util::{BodyExt as _, Full, StreamBody};
use nemo_relay::api::registry::{RuntimeRegistrationOwner, RuntimeRegistrationOwnerKind};

use crate::test_support::EnvScope;

#[tokio::test]
async fn observation_preserves_delivery_while_capturing_json() {
    let expected = Bytes::from_static(br#"{"ok":true}"#);
    let (body, observation) =
        observe_body(Full::new(expected.clone()), StatusCode::OK, expected.len());
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
    let (body, observation) = observe_body(Full::new(expected.clone()), StatusCode::OK, 3);
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
