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
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{Value, json};

const CLASSIFIER_VERDICT: &str = r#"{"recommended_route":"efficient","p_solve":0.9,"confidence":0.95,"abstain":false,"capability_boundary":"supported","primary_rule":"SUP-1","crux":"bounded task"}"#;

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
    let malformed_response = !stream
        && model == "provider/selected"
        && request["messages"][0]["content"] == "malformed-response";
    state.requests.lock().unwrap().push((headers, request));
    if malformed_response {
        return Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from("{invalid-provider-json"))
            .unwrap();
    }
    if stream {
        let text = if model == "provider/classifier" {
            CLASSIFIER_VERDICT
        } else {
            "streamed"
        };
        let first = json!({
            "id": "chat-ci", "object": "chat.completion.chunk", "model": model,
            "system_fingerprint": "fp_process_e2e",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": null}]
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
    let text = if model == "provider/classifier" {
        CLASSIFIER_VERDICT.to_string()
    } else {
        format!("served by {model}")
    };
    let body = json!({
        "id": "chat-ci", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
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
async fn switchyard_plugin_routes_buffered_and_streaming_without_a_service() {
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
version = 2
max_retries = 0
enabled_inbound_profiles = ["openai_chat"]

[components.config.algorithm]
kind = "random"
seed = 42

[components.config.default_targets]
openai_chat = "fallback-chat"

[components.config.targets.selected-chat]
model = "provider/selected"
protocol = "openai_chat"
endpoint = "/v1/chat/completions"
base_url = "{provider_url}"
weight = 1

[components.config.targets.fallback-chat]
model = "provider/fallback"
protocol = "openai_chat"
endpoint = "/v1/chat/completions"
base_url = "{provider_url}"
weight = 0
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
            .header("authorization", "Bearer caller-secret")
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
                "messages": [{"role": "user", "content": request_id}]
            }))
            .send()
    };

    let buffered = send_chat("buffered-request", false).await.unwrap();
    assert!(buffered.status().is_success());
    let buffered: Value = buffered.json().await.unwrap();
    assert_eq!(buffered["model"], "provider/selected");

    let streaming = send_chat("stream-request", true).await.unwrap();
    assert!(streaming.status().is_success());
    let streaming = streaming.text().await.unwrap();
    assert!(streaming.contains("streamed"));
    assert!(streaming.contains("fp_process_e2e"));
    assert!(streaming.contains("[DONE]"));

    let malformed = send_chat("malformed-response", false).await.unwrap();
    assert!(malformed.status().is_success());
    let malformed: Value = malformed.json().await.unwrap();
    assert_eq!(malformed["model"], "provider/fallback");

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
    let malformed_models = providers
        .iter()
        .filter(|(_, body)| body["messages"][0]["content"] == "malformed-response")
        .map(|(_, body)| body["model"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        malformed_models,
        vec!["provider/selected", "provider/fallback"]
    );
    for (headers, _) in providers.iter() {
        assert!(!headers.contains_key("x-nemo-relay-internal-dispatch-url"));
        assert!(!headers.contains_key("x-nemo-relay-internal-dispatch-route"));
        assert!(!headers.contains_key("authorization"));
    }

    provider_task.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn switchyard_plugin_runs_classifier_and_routed_calls_without_a_service() {
    let provider_state = ProviderState::default();
    let provider_requests = Arc::clone(&provider_state.requests);
    let (provider_url, provider_task) = start_server(
        Router::new()
            .route("/v1/chat/completions", post(provide))
            .with_state(provider_state),
    )
    .await;

    let temp = tempfile::tempdir().unwrap();
    let atof_dir = temp.path().join("atof");
    std::fs::create_dir_all(&atof_dir).unwrap();
    let config_path = temp.path().join("plugins.toml");
    let config = format!(
        r#"version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 2

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = "{}"
filename = "events.jsonl"
mode = "overwrite"

[[components]]
kind = "switchyard"
enabled = true

[components.config]
version = 2
max_retries = 0
enabled_inbound_profiles = ["openai_chat"]

[components.config.algorithm]
kind = "llm_classifier"
classifier_target = "classifier"
weak_target = "weak"
strong_target = "strong"
base_threshold = 0.5
min_confidence = 0.5

[components.config.default_targets]
openai_chat = "strong"

[components.config.targets.classifier]
model = "provider/classifier"
protocol = "openai_chat"
endpoint = "/v1/chat/completions"
base_url = "{provider_url}"

[components.config.targets.weak]
model = "provider/weak"
protocol = "openai_chat"
endpoint = "/v1/chat/completions"
base_url = "{provider_url}"

[components.config.targets.strong]
model = "provider/strong"
protocol = "openai_chat"
endpoint = "/v1/chat/completions"
base_url = "{provider_url}"
"#,
        atof_dir.display()
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

    let buffered = client
        .post(format!("{gateway_url}/v1/chat/completions"))
        .header("x-nemo-relay-session-id", "classifier-buffered")
        .json(&json!({
            "model": "client/model",
            "messages": [{"role": "user", "content": "classify buffered"}]
        }))
        .send()
        .await
        .unwrap();
    assert!(buffered.status().is_success());
    let buffered: Value = buffered.json().await.unwrap();
    assert_eq!(buffered["model"], "provider/weak");

    let streaming = client
        .post(format!("{gateway_url}/v1/chat/completions"))
        .header("x-nemo-relay-session-id", "classifier-streaming")
        .json(&json!({
            "model": "client/model",
            "stream": true,
            "messages": [{"role": "user", "content": "classify streaming"}]
        }))
        .send()
        .await
        .unwrap();
    assert!(streaming.status().is_success());
    let streaming = streaming.text().await.unwrap();
    assert!(streaming.contains("streamed"));
    assert!(streaming.contains("[DONE]"));

    let providers = provider_requests.lock().unwrap();
    let models = providers
        .iter()
        .map(|(_, body)| body["model"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        models,
        vec![
            "provider/classifier",
            "provider/weak",
            "provider/classifier",
            "provider/weak"
        ]
    );
    for (_, body) in providers
        .iter()
        .filter(|(_, body)| body["model"] == "provider/classifier")
    {
        assert!(body["response_format"].is_object());
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
    }
    drop(providers);

    let events_path = atof_dir.join("events.jsonl");
    let mut events = String::new();
    for _ in 0..40 {
        events = std::fs::read_to_string(&events_path).unwrap_or_default();
        if events.matches("switchyard.routing.call").count() >= 4 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let events = events
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|event| {
            event["name"]
                .as_str()
                .is_some_and(|name| name.starts_with("switchyard.routing."))
        })
        .collect::<Vec<_>>();
    let calls = events
        .iter()
        .filter(|event| event["name"] == "switchyard.routing.call")
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 4, "routing events: {events:?}");
    assert!(calls.iter().all(|event| {
        event["data"]["algorithm"] == "llm_task_classifier"
            && event["data"]["semantic_target"]
                .as_str()
                .is_some_and(|target| matches!(target, "classifier" | "weak"))
    }));
    assert_eq!(
        calls
            .iter()
            .filter(|event| event["data"]["is_routed_call"] == false)
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| {
                event["name"] == "switchyard.routing.decision"
                    && event["data"]["semantic_target"] == "weak"
            })
            .count(),
        2
    );

    provider_task.abort();
}
