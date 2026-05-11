// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use axum::response::IntoResponse;
use bytes::Bytes;
use futures_util::stream;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tower::ServiceExt;

use super::*;
use crate::error::CliError;

struct TestServer {
    url: String,
    handle: JoinHandle<()>,
}

impl TestServer {
    fn url(&self) -> String {
        self.url.clone()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

fn test_config() -> GatewayConfig {
    GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        atif_dir: None,
        openinference_endpoint: None,
        metadata: None,
        plugin_config: None,
    }
}

#[tokio::test]
async fn codex_hook_keeps_codex_response_shape() {
    let app = router(test_config());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/hooks/codex")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "session_id": "codex-1",
                        "hook_event_name": "sessionStart"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body, json!({}));
}

#[tokio::test]
async fn healthz_returns_ok() {
    let app = router(test_config());
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body, json!({ "status": "ok" }));
}

#[tokio::test]
async fn gateway_errors_render_structured_json_responses() {
    let response = CliError::InvalidPayload("bad input".into()).into_response();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["error"]["type"], json!("nemo_flow_gateway_error"));
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("bad input")
    );

    let response = CliError::Config("bad config".into()).into_response();

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn claude_code_hook_returns_continue_shape() {
    let app = router(test_config());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/hooks/claude-code")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "session_id": "claude-1",
                        "hook_event_name": "SessionStart"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["continue"], json!(true));
}

#[tokio::test]
async fn cursor_hook_returns_cursor_permission_fields() {
    let app = router(test_config());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/hooks/cursor")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "session_id": "cursor-1",
                        "hook_event_name": "beforeShellExecution",
                        "tool_call_id": "shell-1",
                        "tool_name": "shell",
                        "input": { "command": "pwd" }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["continue"], json!(true));
    assert_eq!(body["permission"], json!("allow"));
}

#[tokio::test]
async fn hermes_hook_keeps_shell_hook_response_shape() {
    let app = router(test_config());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/hooks/hermes")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "session_id": "hermes-1",
                        "hook_event_name": "on_session_start"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body, json!({}));
}

#[tokio::test]
async fn gateway_forwards_openai_json_without_rewriting_payload() {
    let upstream = spawn_upstream(false).await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    let app = router(config);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", "Bearer test")
                .header("connection", "close")
                .body(Body::from(
                    json!({
                        "model": "gpt-test",
                        "messages": [{ "role": "user", "content": "hello" }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["model"], json!("gpt-test"));
    assert_eq!(body["authorization"], json!("Bearer test"));
    assert_eq!(body["connection"], Value::Null);
}

#[tokio::test]
async fn gateway_accepts_codex_responses_path() {
    let upstream = spawn_upstream(false).await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    let app = router(config);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/responses")
                .header("content-type", "application/json")
                .header("authorization", "Bearer test")
                .body(Body::from(
                    json!({
                        "model": "gpt-test",
                        "input": "hello"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["model"], json!("gpt-test"));
    assert_eq!(body["authorization"], json!("Bearer test"));
}

#[tokio::test]
async fn gateway_preserves_streaming_body() {
    let upstream = spawn_upstream(true).await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    let app = router(config);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "gpt-test",
                        "input": "hello",
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/event-stream"
    );
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&bytes).unwrap();
    // Managed execution re-encodes each parsed event with the OpenAI Responses event name on
    // its own `event:` line, so the wire shape is closer to the spec but not byte-identical to
    // the upstream feed. Both event payloads should appear in order.
    assert!(
        body_str.contains("event: response.created"),
        "missing response.created event: {body_str}",
    );
    assert!(
        body_str.contains("event: response.completed"),
        "missing response.completed event: {body_str}",
    );
    let created_idx = body_str.find("response.created").unwrap();
    let completed_idx = body_str.find("response.completed").unwrap();
    assert!(
        created_idx < completed_idx,
        "events out of order: {body_str}"
    );
}

#[tokio::test]
async fn gateway_surfaces_streaming_upstream_errors() {
    let upstream = spawn_failing_stream_upstream().await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    let app = router(config);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "gpt-test",
                        "input": "hello",
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn gateway_rejects_unsupported_paths() {
    let app = router(test_config());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/unsupported")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn gateway_returns_bad_gateway_when_upstream_is_unreachable() {
    let mut config = test_config();
    config.openai_base_url = "http://127.0.0.1:1".into();
    let app = router(config);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(json!({ "model": "gpt-test" }).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn models_route_forwards_get_requests() {
    let upstream = spawn_models_upstream().await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    let app = router(config);
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models?limit=1")
                .header("authorization", "Bearer test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["path"], json!("/v1/models?limit=1"));
    assert_eq!(body["authorization"], json!("Bearer test"));
}

async fn spawn_upstream(streaming: bool) -> TestServer {
    async fn chat(headers: HeaderMap, body: Bytes) -> impl IntoResponse {
        let payload: Value = serde_json::from_slice(&body).unwrap();
        Json(json!({
            "model": payload["model"],
            "authorization": headers
                .get(header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            "connection": headers
                .get(header::CONNECTION)
                .and_then(|value| value.to_str().ok())
        }))
    }

    async fn stream_response() -> impl IntoResponse {
        // OpenAI Responses managed pipeline parses each `data:` payload as JSON; emit minimally
        // valid response.created / response.completed events so the runtime collector + finalizer
        // assemble a well-formed end-event payload.
        let chunks = stream::iter([
            Ok::<_, std::convert::Infallible>(Bytes::from_static(
                b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\"}}\n\n",
            )),
            Ok(Bytes::from_static(
                b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\"}}\n\n",
            )),
        ]);
        (
            [(header::CONTENT_TYPE, "text/event-stream")],
            Body::from_stream(chunks),
        )
    }

    let app = if streaming {
        Router::new().route("/v1/responses", post(stream_response))
    } else {
        Router::new()
            .route("/v1/chat/completions", post(chat))
            .route("/v1/responses", post(chat))
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    TestServer {
        url: format!("http://{address}"),
        handle,
    }
}

async fn spawn_failing_stream_upstream() -> TestServer {
    async fn stream_response() -> impl IntoResponse {
        // First chunk is a valid JSON SSE event so the managed pipeline opens cleanly; the
        // following IO error simulates the upstream socket dropping mid-stream.
        let chunks = stream::iter([
            Ok::<_, std::io::Error>(Bytes::from_static(
                b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\"}}\n\n",
            )),
            Err(std::io::Error::other("stream failed")),
        ]);
        (
            [(header::CONTENT_TYPE, "text/event-stream")],
            Body::from_stream(chunks),
        )
    }

    let app = Router::new().route("/v1/responses", post(stream_response));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    TestServer {
        url: format!("http://{address}"),
        handle,
    }
}

async fn spawn_models_upstream() -> TestServer {
    async fn models(headers: HeaderMap, request: Request<Body>) -> impl IntoResponse {
        Json(json!({
            "path": request.uri().path_and_query().map(|value| value.as_str()),
            "authorization": headers
                .get(header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
        }))
    }

    let app = Router::new().route("/v1/models", get(models));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    TestServer {
        url: format!("http://{address}"),
        handle,
    }
}
