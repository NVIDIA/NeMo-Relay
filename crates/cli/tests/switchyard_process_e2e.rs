// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CI-safe process-boundary coverage for the Switchyard plugin.

use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

fn gateway_bin() -> &'static str {
    env!("CARGO_BIN_EXE_nemo-relay")
}

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[derive(Clone, Default)]
struct DecisionState {
    requests: Arc<Mutex<Vec<(HeaderMap, Value)>>>,
}

async fn decide(
    State(state): State<DecisionState>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> Response {
    let call = {
        let mut requests = state.requests.lock().unwrap();
        requests.push((headers, request));
        requests.len()
    };
    if call == 4 {
        return Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .body(Body::from("decision API unavailable"))
            .unwrap();
    }
    let body = json!({
        "schema_version": "switchyard.routing_decision.v1",
        "decision_id": format!("decision-{call}"),
        "router": {"name": "fake-ci-router", "version": "1"},
        "route": {
            "tier": "strong",
            "target_model": "provider/selected",
            "backend_id": "selected-chat",
            "target_protocol_profile": "openai_chat",
            "target_endpoint": "/v1/chat/completions"
        },
        "confidence": 0.99,
        "reason_code": "ci_fixture",
        "reason_summary": "deterministic process E2E decision"
    });
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn switchyard_health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

#[derive(Clone, Default)]
struct ProviderState {
    requests: Arc<Mutex<Vec<(HeaderMap, Value)>>>,
}

async fn provide(
    State(state): State<ProviderState>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> Response {
    let stream = request["stream"].as_bool().unwrap_or(false);
    let model = request["model"].as_str().unwrap_or("unknown").to_string();
    state.requests.lock().unwrap().push((headers, request));
    if stream {
        let first = json!({
            "id": "chat-ci", "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "streamed"}, "finish_reason": null}]
        });
        let last = json!({
            "id": "chat-ci", "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
        });
        let body = format!("data: {first}\n\ndata: {last}\n\ndata: [DONE]\n\n");
        return Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(Body::from(body))
            .unwrap();
    }
    let body = json!({
        "id": "chat-ci", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": format!("served by {model}")}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
    });
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn start_server(router: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (format!("http://{address}"), task)
}

fn unused_address() -> std::net::SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap()
}

async fn wait_for_gateway(client: &reqwest::Client, url: &str, child: &mut Child) {
    for _ in 0..120 {
        if let Some(status) = child.try_wait().unwrap() {
            panic!("gateway exited before readiness with {status}");
        }
        if client
            .get(format!("{url}/healthz"))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("gateway did not become ready at {url}");
}

#[tokio::test(flavor = "multi_thread")]
async fn switchyard_plugin_routes_buffered_and_streaming_then_fails_open() {
    let decision_state = DecisionState::default();
    let decision_requests = Arc::clone(&decision_state.requests);
    let (decision_url, decision_task) = start_server(
        Router::new()
            .route("/v1/routing/decision", post(decide))
            .route("/health", get(switchyard_health))
            .with_state(decision_state),
    )
    .await;

    let provider_state = ProviderState::default();
    let provider_requests = Arc::clone(&provider_state.requests);
    let (provider_url, provider_task) = start_server(
        Router::new()
            .route("/v1/chat/completions", post(provide))
            .with_state(provider_state),
    )
    .await;

    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("plugins.toml");
    let config = format!(
        r#"version = 1

[[components]]
kind = "switchyard"
enabled = true

[components.config]
mode = "enforce"
decision_api_url = "{decision_url}/v1/routing/decision"
decision_profile_id = "ci-process-e2e"
request_materialization = "full_body"
context_mode = "payload_only"
decision_timeout_millis = 1000
max_retries = 0

[components.config.default_targets]
openai_chat = "fallback-chat"
openai_responses = "fallback-responses"
anthropic_messages = "fallback-anthropic"

[components.config.targets.selected-chat]
model = "provider/selected"
protocol = "openai_chat"
endpoint = "/v1/chat/completions"
base_url = "{provider_url}"

[components.config.targets.fallback-chat]
model = "provider/fallback"
protocol = "openai_chat"
endpoint = "/v1/chat/completions"
base_url = "{provider_url}"

[components.config.targets.fallback-responses]
model = "provider/fallback"
protocol = "openai_responses"
endpoint = "/v1/responses"
base_url = "{provider_url}"

[components.config.targets.fallback-anthropic]
model = "provider/fallback"
protocol = "anthropic_messages"
endpoint = "/v1/messages"
base_url = "{provider_url}"
"#
    );
    std::fs::write(&config_path, config).unwrap();

    let address = unused_address();
    let gateway_url = format!("http://{address}");
    let stderr = std::fs::File::create(temp.path().join("gateway.log")).unwrap();
    let child = Command::new(gateway_bin())
        .arg("--plugin-config-path")
        .arg(&config_path)
        .arg("--bind")
        .arg(address.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr))
        .spawn()
        .unwrap();
    let mut gateway = ChildGuard(child);
    let client = reqwest::Client::new();
    wait_for_gateway(&client, &gateway_url, &mut gateway.0).await;

    let send_chat = |request_id: &'static str, stream: bool| {
        client
            .post(format!("{gateway_url}/v1/chat/completions"))
            .header("x-nemo-relay-session-id", "ci-process-session")
            .header("x-nemo-relay-request-id", request_id)
            .header(
                "x-nemo-relay-internal-dispatch-url",
                "http://attacker.invalid",
            )
            .header("x-nemo-relay-internal-dispatch-route", "attacker-route")
            .json(&json!({
                "model": "client/model",
                "stream": stream,
                "messages": [{"role": "user", "content": "process boundary test"}]
            }))
            .send()
    };

    let buffered = send_chat("buffered-request", false).await.unwrap();
    assert!(buffered.status().is_success());
    let buffered: Value = buffered.json().await.unwrap();
    assert_eq!(buffered["model"], "provider/selected");

    let translated = client
        .post(format!("{gateway_url}/v1/responses"))
        .header("x-nemo-relay-session-id", "ci-process-session")
        .header("x-nemo-relay-request-id", "translated-request")
        .json(&json!({
            "model": "client/model",
            "stream": false,
            "input": "process boundary response translation"
        }))
        .send()
        .await
        .unwrap();
    assert!(translated.status().is_success());
    let translated: Value = translated.json().await.unwrap();
    assert_eq!(translated["object"], "response");

    let streaming = send_chat("stream-request", true).await.unwrap();
    assert!(streaming.status().is_success());
    let streaming = streaming.text().await.unwrap();
    assert!(streaming.contains("streamed"));
    assert!(streaming.contains("[DONE]"));

    let fallback = send_chat("fallback-request", false).await.unwrap();
    assert!(fallback.status().is_success());
    let fallback: Value = fallback.json().await.unwrap();
    assert_eq!(fallback["model"], "provider/fallback");

    let decisions = decision_requests.lock().unwrap();
    assert_eq!(decisions.len(), 4);
    for (headers, body) in decisions.iter() {
        assert!(!headers.contains_key("x-nemo-relay-internal-dispatch-url"));
        assert!(!headers.contains_key("x-nemo-relay-internal-dispatch-route"));
        assert_eq!(
            headers
                .get("x-nemo-relay-session-id")
                .unwrap()
                .to_str()
                .unwrap(),
            "ci-process-session"
        );
        assert_eq!(body["schema_version"], "switchyard.routing_request.v1");
        assert_eq!(body["decision_profile"]["profile_id"], "ci-process-e2e");
    }
    drop(decisions);

    let providers = provider_requests.lock().unwrap();
    let models = providers
        .iter()
        .map(|(_, body)| body["model"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        models,
        vec![
            "provider/selected",
            "provider/selected",
            "provider/selected",
            "provider/fallback"
        ]
    );
    assert!(providers[1].1["messages"].is_array());
    assert!(providers[1].1.get("input").is_none());
    assert_eq!(providers[1].1["messages"][0]["role"], "user");
    assert_eq!(
        providers[1].1["messages"][0]["content"],
        "process boundary response translation"
    );
    for (headers, _) in providers.iter() {
        assert!(!headers.contains_key("x-nemo-relay-internal-dispatch-url"));
        assert!(!headers.contains_key("x-nemo-relay-internal-dispatch-route"));
    }

    decision_task.abort();
    provider_task.abort();
}
