// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::future::Future;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use futures_util::StreamExt;
use serde_json::{Map, json};

use crate::api::llm::{
    LlmCallExecuteParams, LlmStreamCallExecuteParams, llm_call_execute, llm_stream_call_execute,
};
use crate::api::runtime::{MiddlewareContinuationContext, NemoRelayContextState, global_context};
use crate::error::{FlowError, MAX_UPSTREAM_FAILURE_HEADER_VALUE_BYTES, bounded_utf8};

use super::*;

struct FakeProvider {
    url: String,
    request: mpsc::Receiver<Vec<u8>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl FakeProvider {
    fn spawn(response: Vec<u8>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("fake provider should bind");
        listener
            .set_nonblocking(true)
            .expect("fake provider listener should be nonblocking");
        let address = listener.local_addr().expect("fake provider address");
        let (request_tx, request) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            let (mut socket, _) = loop {
                match listener.accept() {
                    Ok(connection) => break connection,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && std::time::Instant::now() < deadline =>
                    {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("fake provider should accept: {error}"),
                }
            };
            socket
                .set_nonblocking(false)
                .expect("fake provider socket should be blocking");
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("read timeout should configure");
            let request_bytes = read_http_request(&mut socket);
            request_tx
                .send(request_bytes)
                .expect("test should receive provider request");
            if let Err(error) = socket.write_all(&response) {
                assert!(
                    matches!(
                        error.kind(),
                        std::io::ErrorKind::BrokenPipe
                            | std::io::ErrorKind::ConnectionAborted
                            | std::io::ErrorKind::ConnectionReset
                    ),
                    "fake provider should write response: {error}"
                );
            }
        });
        Self {
            url: format!("http://{address}/v1/messages"),
            request,
            thread: Some(thread),
        }
    }

    fn request(&self) -> Vec<u8> {
        self.request
            .recv_timeout(Duration::from_secs(5))
            .expect("fake provider request should arrive")
    }
}

impl Drop for FakeProvider {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            thread.join().expect("fake provider thread should finish");
        }
    }
}

fn read_http_request(socket: &mut std::net::TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let read = socket
            .read(&mut buffer)
            .expect("request read should succeed");
        if read == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..read]);
        let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let content_length = String::from_utf8_lossy(&request[..header_end])
            .lines()
            .find_map(|line| {
                line.split_once(':').and_then(|(name, value)| {
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
            })
            .unwrap_or(0);
        if request.len() >= header_end + 4 + content_length {
            break;
        }
    }
    request
}

fn response(status: &str, headers: &[(&str, &str)], body: &[u8]) -> Vec<u8> {
    let mut response = format!("HTTP/1.1 {status}\r\nConnection: close\r\n");
    for (name, value) in headers {
        response.push_str(name);
        response.push_str(": ");
        response.push_str(value);
        response.push_str("\r\n");
    }
    response.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    let mut response = response.into_bytes();
    response.extend_from_slice(body);
    response
}

fn target(url: String, headers: BTreeMap<String, String>) -> LlmDispatchTargetContext {
    LlmDispatchTargetContext::try_new(url, headers).expect("test target should be valid")
}

fn request() -> LlmRequest {
    LlmRequest {
        headers: Map::from_iter([
            ("authorization".into(), json!("Bearer request-secret")),
            (
                "x-nemo-relay-internal-dispatch-url".into(),
                json!("http://attacker.invalid"),
            ),
        ]),
        content: json!({"model": "selected", "prompt": "hello"}),
    }
}

async fn scope_test_target<F: Future>(target: LlmDispatchTargetContext, future: F) -> F::Output {
    let event_uuid = uuid::Uuid::now_v7();
    crate::api::runtime::with_active_event_uuid(
        event_uuid,
        scope_llm_dispatch_target(Some(event_uuid), target, future),
    )
    .await
}

#[tokio::test]
async fn buffered_target_runs_after_downstream_middleware_and_ignores_host_callback() {
    let provider = FakeProvider::spawn(response(
        "200 OK",
        &[("Content-Type", "application/json")],
        br#"{"id":"selected-response"}"#,
    ));
    let target = target(
        format!("{}?api_key=url-secret", provider.url),
        BTreeMap::from([
            ("authorization".into(), "Bearer target-secret".into()),
            ("x-target".into(), "selected".into()),
        ]),
    );
    let debug = format!("{target:?}");
    assert!(!debug.contains("target-secret"));
    assert!(!debug.contains("url-secret"));

    let fallback_called = Arc::new(AtomicBool::new(false));
    let fallback_called_for_fn = fallback_called.clone();
    let terminal = targeted_llm_execution(Arc::new(move |_| {
        fallback_called_for_fn.store(true, Ordering::SeqCst);
        Box::pin(async { Ok(json!({"id": "wrong-provider"})) })
    }));
    let middleware_ran = Arc::new(AtomicBool::new(false));
    let middleware_ran_for_fn = middleware_ran.clone();
    let downstream = Arc::new(move |mut request: LlmRequest| {
        middleware_ran_for_fn.store(true, Ordering::SeqCst);
        request.content["middleware"] = json!(true);
        terminal(request)
    });

    let result = scope_test_target(target, downstream(request()))
        .await
        .expect("targeted request should succeed");

    assert_eq!(result, json!({"id": "selected-response"}));
    assert!(middleware_ran.load(Ordering::SeqCst));
    assert!(!fallback_called.load(Ordering::SeqCst));
    let captured = String::from_utf8(provider.request()).expect("request should be UTF-8");
    assert!(captured.starts_with("POST /v1/messages?api_key=url-secret HTTP/1.1\r\n"));
    assert!(captured.contains("authorization: Bearer target-secret\r\n"));
    assert!(captured.contains("x-target: selected\r\n"));
    assert!(captured.contains(r#"{"middleware":true,"model":"selected","prompt":"hello"}"#));
    assert!(!captured.contains("request-secret"));
    assert!(!captured.contains("attacker.invalid"));
}

#[tokio::test]
async fn malformed_success_json_is_an_internal_provider_failure() {
    let provider = FakeProvider::spawn(response(
        "200 OK",
        &[("Content-Type", "application/json")],
        b"not-json",
    ));

    let error = dispatch_buffered(&target(provider.url.clone(), BTreeMap::new()), request())
        .await
        .expect_err("malformed successful response should fail");

    assert!(matches!(
        error,
        FlowError::Internal(message)
            if message == "targeted LLM provider returned malformed response JSON"
    ));
    let _ = provider.request();
}

#[tokio::test]
async fn buffered_success_body_is_bounded_with_or_without_content_length() {
    for response in [
        response(
            "200 OK",
            &[("Content-Type", "application/json")],
            b"123456789",
        ),
        b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n9\r\n123456789\r\n0\r\n\r\n".to_vec(),
    ] {
        let provider = FakeProvider::spawn(response);
        let target = target(provider.url.clone(), BTreeMap::new());
        let response = send(&target, request(), Some(HTTP_REQUEST_TIMEOUT))
            .await
            .expect("provider should return a successful HTTP response");
        let error = bounded_success_body(&target, response, 8)
            .await
            .expect_err("oversized successful response should fail");
        assert!(matches!(
            error,
            FlowError::Internal(message)
                if message == "targeted LLM provider response exceeded the 8-byte buffered body limit"
        ));
        let _ = provider.request();
    }
}

#[tokio::test]
async fn buffered_http_failure_is_bounded_and_filters_headers() {
    let body = vec![b'x'; MAX_UPSTREAM_FAILURE_BODY_BYTES + 1024];
    let provider = FakeProvider::spawn(response(
        "429 Too Many Requests",
        &[
            ("Retry-After", "2"),
            ("Set-Cookie", "secret=true"),
            ("Authorization", "Bearer response-secret"),
        ],
        &body,
    ));

    let error = dispatch_buffered(&target(provider.url.clone(), BTreeMap::new()), request())
        .await
        .expect_err("429 should fail");
    let FlowError::Upstream(failure) = error else {
        panic!("expected structured upstream failure");
    };
    assert_eq!(failure.status, Some(429));
    assert_eq!(failure.class, UpstreamFailureClass::RetryableStatus);
    assert_eq!(failure.body.len(), MAX_UPSTREAM_FAILURE_BODY_BYTES);
    assert_eq!(
        failure.headers.get("retry-after").map(String::as_str),
        Some("2")
    );
    assert!(!failure.headers.contains_key("set-cookie"));
    assert!(!failure.headers.contains_key("authorization"));
    let _ = provider.request();
}

#[tokio::test]
async fn streaming_http_failure_is_structured_and_filters_headers() {
    let provider = FakeProvider::spawn(response(
        "503 Service Unavailable",
        &[("Retry-After", "3"), ("Set-Cookie", "secret=true")],
        b"provider unavailable",
    ));

    let error =
        match dispatch_stream(&target(provider.url.clone(), BTreeMap::new()), request()).await {
            Ok(_) => panic!("503 should fail before opening a stream"),
            Err(error) => error,
        };
    let FlowError::Upstream(failure) = error else {
        panic!("expected structured upstream failure");
    };
    assert_eq!(failure.status, Some(503));
    assert_eq!(failure.class, UpstreamFailureClass::RetryableStatus);
    assert_eq!(failure.body, "provider unavailable");
    assert_eq!(
        failure.headers.get("retry-after").map(String::as_str),
        Some("3")
    );
    assert!(!failure.headers.contains_key("set-cookie"));
    let _ = provider.request();
}

#[tokio::test]
async fn redirects_are_returned_without_following() {
    let provider = FakeProvider::spawn(response(
        "302 Found",
        &[("Location", "http://127.0.0.1:9/should-not-run")],
        b"redirect",
    ));

    let error = dispatch_buffered(&target(provider.url.clone(), BTreeMap::new()), request())
        .await
        .expect_err("redirect should fail");
    let FlowError::Upstream(failure) = error else {
        panic!("expected HTTP failure");
    };
    assert_eq!(failure.status, Some(302));
    assert_eq!(failure.class, UpstreamFailureClass::Other);
    let _ = provider.request();
}

#[tokio::test]
async fn streaming_target_decodes_events_empty_streams_and_late_errors() {
    let provider = FakeProvider::spawn(response(
        "200 OK",
        &[("Content-Type", "text/event-stream")],
        b"data: {\"delta\":\"hello\"}\n\ndata: not-json\n\n",
    ));
    let fallback_called = Arc::new(AtomicBool::new(false));
    let fallback_called_for_fn = fallback_called.clone();
    let terminal = targeted_llm_stream_execution(Arc::new(move |_| {
        fallback_called_for_fn.store(true, Ordering::SeqCst);
        Box::pin(async { Ok(LlmJsonStream::new(futures_util::stream::empty())) })
    }));
    let stream_target = target(provider.url.clone(), BTreeMap::new());
    let mut stream = scope_test_target(stream_target, terminal(request()))
        .await
        .expect("stream should open");
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        json!({"delta": "hello"})
    );
    assert!(stream.next().await.unwrap().is_err());
    assert!(!fallback_called.load(Ordering::SeqCst));
    let _ = provider.request();

    let empty_provider = FakeProvider::spawn(response(
        "200 OK",
        &[("Content-Type", "text/event-stream")],
        b"",
    ));
    let mut empty = dispatch_stream(
        &target(empty_provider.url.clone(), BTreeMap::new()),
        request(),
    )
    .await
    .expect("empty stream should open");
    assert!(empty.next().await.is_none());
    let _ = empty_provider.request();

    let cancelled_provider = FakeProvider::spawn(response(
        "200 OK",
        &[("Content-Type", "text/event-stream")],
        b"data: {\"delta\":\"first\"}\n\ndata: {\"delta\":\"second\"}\n\n",
    ));
    let mut cancelled = dispatch_stream(
        &target(cancelled_provider.url.clone(), BTreeMap::new()),
        request(),
    )
    .await
    .expect("cancellable stream should open");
    assert_eq!(
        cancelled.next().await.unwrap().unwrap(),
        json!({"delta": "first"})
    );
    cancelled
        .close()
        .await
        .expect("stream should close cleanly");
    assert!(cancelled.next().await.is_none());
    let _ = cancelled_provider.request();
}

#[test]
fn target_validation_rejects_unsafe_transport_inputs() {
    for (url, headers) in [
        ("ftp://provider.example/v1", BTreeMap::new()),
        ("https://user:secret@provider.example/v1", BTreeMap::new()),
        (
            "https://provider.example/v1",
            BTreeMap::from([("host".into(), "attacker.invalid".into())]),
        ),
        (
            "https://provider.example/v1",
            BTreeMap::from([(
                "x-nemo-relay-internal-dispatch-url".into(),
                "http://attacker.invalid".into(),
            )]),
        ),
        (
            "https://provider.example/v1",
            BTreeMap::from([
                ("Authorization".into(), "Bearer first".into()),
                ("authorization".into(), "Bearer second".into()),
            ]),
        ),
    ] {
        assert!(LlmDispatchTargetContext::try_new(url.into(), headers).is_err());
    }
}

#[test]
fn target_validation_reports_malformed_headers() {
    let error = LlmDispatchTargetContext::try_new(
        "https://provider.example/v1".into(),
        BTreeMap::from([("bad header".into(), "value".into())]),
    )
    .expect_err("header name containing a space should be rejected");
    let FlowError::InvalidArgument(message) = error else {
        panic!("expected invalid header name argument");
    };
    assert_eq!(
        message,
        "LLM continuation contained an invalid target header name"
    );

    let error = LlmDispatchTargetContext::try_new(
        "https://provider.example/v1".into(),
        BTreeMap::from([("x-target".into(), "line one\nline two".into())]),
    )
    .expect_err("header value containing a newline should be rejected");
    let FlowError::InvalidArgument(message) = error else {
        panic!("expected invalid header value argument");
    };
    assert_eq!(
        message,
        "LLM continuation target header x-target had an invalid value"
    );
}

#[test]
fn bounded_utf8_truncates_long_multibyte_value_at_character_boundary() {
    let expected = "a".repeat(MAX_UPSTREAM_FAILURE_HEADER_VALUE_BYTES - 1);
    let value = format!("{expected}\u{e9}");
    assert_eq!(value.len(), MAX_UPSTREAM_FAILURE_HEADER_VALUE_BYTES + 1);

    let bounded = bounded_utf8(value, MAX_UPSTREAM_FAILURE_HEADER_VALUE_BYTES);

    assert_eq!(bounded, expected);
    assert_eq!(bounded.len(), MAX_UPSTREAM_FAILURE_HEADER_VALUE_BYTES - 1);
}

#[tokio::test]
async fn transport_failures_do_not_fall_back_to_the_host_callback() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("temporary listener should bind");
    let address = listener.local_addr().unwrap();
    drop(listener);
    let fallback_called = Arc::new(AtomicBool::new(false));
    let fallback_called_for_fn = fallback_called.clone();
    let terminal = targeted_llm_execution(Arc::new(move |_| {
        fallback_called_for_fn.store(true, Ordering::SeqCst);
        Box::pin(async { Ok(json!({"wrong": true})) })
    }));
    let target = target(
        format!("http://{address}/v1/chat/completions?api_key=transport-secret"),
        BTreeMap::new(),
    );

    let error = scope_test_target(target, terminal(request()))
        .await
        .expect_err("connection should fail");
    let FlowError::Upstream(failure) = error else {
        panic!("expected transport failure");
    };
    assert_eq!(failure.status, None);
    assert_eq!(failure.class, UpstreamFailureClass::Connection);
    assert!(!failure.body.contains("transport-secret"));
    assert!(!fallback_called.load(Ordering::SeqCst));
}

#[test]
fn nested_buffered_managed_call_does_not_inherit_outer_target() {
    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    crate::shared_runtime::reset_runtime_owner_for_tests();
    *global_context()
        .write()
        .unwrap_or_else(|error| error.into_inner()) = NemoRelayContextState::new();

    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let target = target("http://127.0.0.1:9/v1/messages".into(), BTreeMap::new());
        let context = crate::api::runtime::with_active_event_uuid(uuid::Uuid::now_v7(), async {
            MiddlewareContinuationContext::capture()
        })
        .await;
        let fallback_called = Arc::new(AtomicBool::new(false));
        let fallback_called_for_fn = Arc::clone(&fallback_called);

        let response = context
            .invoke_with_llm_dispatch_target(target, || async move {
                assert!(current_llm_dispatch_target().is_some());
                let response = Box::pin(llm_call_execute(
                    LlmCallExecuteParams::builder()
                        .name("nested-buffered")
                        .request(request())
                        .func(Arc::new(move |_| {
                            assert!(current_llm_dispatch_target().is_none());
                            fallback_called_for_fn.store(true, Ordering::SeqCst);
                            Box::pin(async { Ok(json!({"provider": "ordinary"})) })
                        }))
                        .build(),
                ))
                .await?;
                assert!(current_llm_dispatch_target().is_some());
                Ok::<_, FlowError>(response)
            })
            .await
            .expect("nested ordinary call should use its own provider callback");

        assert_eq!(response, json!({"provider": "ordinary"}));
        assert!(fallback_called.load(Ordering::SeqCst));
    });
}

#[test]
fn nested_streaming_managed_call_isolated_during_lazy_polling() {
    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    crate::shared_runtime::reset_runtime_owner_for_tests();
    *global_context()
        .write()
        .unwrap_or_else(|error| error.into_inner()) = NemoRelayContextState::new();

    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let target = target("http://127.0.0.1:9/v1/messages".into(), BTreeMap::new());
        let context = crate::api::runtime::with_active_event_uuid(uuid::Uuid::now_v7(), async {
            MiddlewareContinuationContext::capture()
        })
        .await;
        let provider_opened = Arc::new(AtomicBool::new(false));
        let provider_opened_for_fn = Arc::clone(&provider_opened);
        let provider_polled = Arc::new(AtomicBool::new(false));
        let provider_polled_for_fn = Arc::clone(&provider_polled);

        let chunk = context
            .invoke_with_llm_dispatch_target(target, || async move {
                assert!(current_llm_dispatch_target().is_some());
                let mut stream = Box::pin(llm_stream_call_execute(
                    LlmStreamCallExecuteParams::builder()
                        .name("nested-streaming")
                        .request(request())
                        .func(Arc::new(move |_| {
                            assert!(current_llm_dispatch_target().is_none());
                            provider_opened_for_fn.store(true, Ordering::SeqCst);
                            let provider_polled = Arc::clone(&provider_polled_for_fn);
                            Box::pin(async move {
                                Ok(LlmJsonStream::new(futures_util::stream::once(async move {
                                    assert!(current_llm_dispatch_target().is_none());
                                    provider_polled.store(true, Ordering::SeqCst);
                                    Ok(json!({"delta": "ordinary"}))
                                })))
                            })
                        }))
                        .collector(Box::new(|_| Ok(())))
                        .finalizer(Box::new(|| json!({"done": true})))
                        .build(),
                ))
                .await?;
                assert!(current_llm_dispatch_target().is_some());
                let chunk = stream.next().await.expect("nested stream should emit")?;
                assert!(current_llm_dispatch_target().is_some());
                assert!(stream.next().await.is_none());
                assert!(current_llm_dispatch_target().is_some());
                Ok::<_, FlowError>(chunk)
            })
            .await
            .expect("nested ordinary stream should use its own provider callback");

        assert_eq!(chunk, json!({"delta": "ordinary"}));
        assert!(provider_opened.load(Ordering::SeqCst));
        assert!(provider_polled.load(Ordering::SeqCst));
    });
}
