// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use axum::response::IntoResponse;
use bytes::Bytes;
use futures_util::stream;
use http_body_util::BodyExt;
use nemo_relay::api::event::ScopeCategory;
use nemo_relay::api::registry::{
    deregister_tool_conditional_execution_guardrail, register_tool_conditional_execution_guardrail,
};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::plugin::{
    ConfigDiagnostic, Plugin, PluginRegistration, PluginRegistrationContext, deregister_plugin,
    register_plugin,
};
use serde_json::{Map, Value, json};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tower::ServiceExt;

use super::*;
use crate::error::CliError;
use crate::test_support::PLUGIN_CONFIG_TEST_LOCK;

const GENERIC_TEST_PLUGIN_KIND: &str = "cli-test-generic-plugin";
static GENERIC_TEST_PLUGIN_REGISTRATIONS: AtomicUsize = AtomicUsize::new(0);
static GENERIC_TEST_PLUGIN_DEREGISTRATIONS: AtomicUsize = AtomicUsize::new(0);

struct ToolGuardrailCleanup(&'static str);

impl Drop for ToolGuardrailCleanup {
    fn drop(&mut self) {
        let _ = deregister_tool_conditional_execution_guardrail(self.0);
    }
}

struct SubscriberCleanup(&'static str);

impl Drop for SubscriberCleanup {
    fn drop(&mut self) {
        let _ = deregister_subscriber(self.0);
    }
}

fn test_http_client() -> reqwest::Client {
    crate::tls::install_rustls_crypto_provider();
    reqwest::Client::new()
}

struct GenericTestPlugin;

impl Plugin for GenericTestPlugin {
    fn plugin_kind(&self) -> &str {
        GENERIC_TEST_PLUGIN_KIND
    }

    fn validate(&self, _plugin_config: &Map<String, Value>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Value>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = nemo_relay::plugin::Result<()>> + Send + 'a>> {
        Box::pin(async move {
            GENERIC_TEST_PLUGIN_REGISTRATIONS.fetch_add(1, Ordering::SeqCst);
            ctx.add_registration(PluginRegistration::new(
                "plugin",
                GENERIC_TEST_PLUGIN_KIND,
                Box::new(|| {
                    GENERIC_TEST_PLUGIN_DEREGISTRATIONS.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }),
            ));
            Ok(())
        })
    }
}

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
        metadata: None,
        plugin_config: None,
    }
}

fn find_scope_event<'a>(
    events: &'a [Value],
    name: &str,
    category: &str,
    scope_category: &str,
) -> &'a Value {
    events
        .iter()
        .find(|event| {
            event["kind"] == "scope"
                && event["name"] == name
                && event["category"] == category
                && event["scope_category"] == scope_category
        })
        .unwrap_or_else(|| {
            panic!(
                "expected {scope_category} {category} scope named {name}, got: {}",
                serde_json::to_string_pretty(events).unwrap()
            )
        })
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
async fn serve_listener_activates_plugin_config_and_clears_on_shutdown() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    let temp = tempfile::tempdir().unwrap();
    let atof_dir = temp.path().join("atof");
    let atif_dir = temp.path().join("atif");
    std::fs::create_dir_all(&atof_dir).unwrap();
    std::fs::create_dir_all(&atif_dir).unwrap();
    let mut config = test_config();
    config.plugin_config = Some(json!({
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 1,
                    "atof": {
                        "enabled": true,
                        "output_directory": atof_dir,
                        "filename": "events.jsonl",
                        "mode": "overwrite"
                    },
                    "atif": {
                        "enabled": true,
                        "output_directory": atif_dir,
                        "filename_template": "trajectory-{session_id}.json"
                    }
                }
            }
        ]
    }));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let handle =
        tokio::spawn(async move { serve_listener(listener, config, Some(shutdown_rx)).await });

    wait_for_gateway(&url).await;
    assert!(nemo_relay::plugin::active_plugin_report().is_some());

    let client = test_http_client();
    for hook_event_name in ["on_session_start", "on_session_finalize"] {
        let response = client
            .post(format!("{url}/hooks/hermes"))
            .json(&json!({
                "session_id": "plugin-bridge-session",
                "hook_event_name": hook_event_name
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();
    assert!(nemo_relay::plugin::active_plugin_report().is_none());

    let events = std::fs::read_to_string(temp.path().join("atof/events.jsonl")).unwrap();
    assert!(
        events.lines().count() >= 2,
        "expected ATOF lifecycle events, got {events:?}"
    );
    let trajectories = std::fs::read_dir(temp.path().join("atif"))
        .unwrap()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            serde_json::from_slice::<Value>(&std::fs::read(entry.path()).ok()?).ok()
        })
        .collect::<Vec<_>>();
    let trajectory = trajectories
        .iter()
        .find(|trajectory| atif_matches_session(trajectory, "plugin-bridge-session"))
        .unwrap_or_else(|| {
            panic!(
                "expected ATIF trajectory for plugin-bridge-session, got {}",
                serde_json::to_string_pretty(&trajectories).unwrap()
            )
        });
    assert!(
        trajectory["extra"]["observed_events"]
            .as_array()
            .is_some_and(|events| events.len() >= 2)
    );
}

fn atif_matches_session(trajectory: &Value, session_id: &str) -> bool {
    trajectory["session_id"] == json!(session_id)
        || trajectory["extra"]["observed_events"]
            .as_array()
            .is_some_and(|events| {
                events
                    .iter()
                    .any(|event| event_has_session_id(event, session_id))
            })
}

fn event_has_session_id(event: &Value, session_id: &str) -> bool {
    event["metadata"]["session_id"] == json!(session_id)
        || event["data"]["session_id"] == json!(session_id)
        || event["data"]["extra"]["session_id"] == json!(session_id)
}

#[tokio::test]
async fn serve_listener_observability_plugin_records_non_hermes_hooks() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    let temp = tempfile::tempdir().unwrap();
    let atof_dir = temp.path().join("atof");
    std::fs::create_dir_all(&atof_dir).unwrap();
    let mut config = test_config();
    config.plugin_config = Some(json!({
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 1,
                    "atof": {
                        "enabled": true,
                        "output_directory": atof_dir,
                        "filename": "events.jsonl",
                        "mode": "overwrite"
                    }
                }
            }
        ]
    }));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let handle =
        tokio::spawn(async move { serve_listener(listener, config, Some(shutdown_rx)).await });

    wait_for_gateway(&url).await;
    let client = test_http_client();
    for (path, session_id, start_event, end_event) in [
        (
            "/hooks/codex",
            "codex-plugin-session",
            "sessionStart",
            "sessionEnd",
        ),
        (
            "/hooks/claude-code",
            "claude-plugin-session",
            "SessionStart",
            "SessionEnd",
        ),
    ] {
        let hook_events = vec![start_event, "UserPromptSubmit", end_event];
        for hook_event_name in hook_events {
            let response = client
                .post(format!("{url}{path}"))
                .json(&json!({
                    "session_id": session_id,
                    "hook_event_name": hook_event_name
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
    }

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();
    assert!(nemo_relay::plugin::active_plugin_report().is_none());

    let events = std::fs::read_to_string(temp.path().join("atof/events.jsonl")).unwrap();
    let agent_starts = events
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| {
            event["kind"] == "scope"
                && event["scope_category"] == "start"
                && event["category"] == "agent"
        })
        .filter_map(|event| event["name"].as_str().map(ToOwned::to_owned))
        .collect::<Vec<_>>();
    assert!(agent_starts.contains(&"codex-turn".to_string()));
    assert!(agent_starts.contains(&"claude-code-turn".to_string()));
    assert!(!agent_starts.contains(&"claude-code".to_string()));
}

#[tokio::test]
async fn serve_listener_hermes_api_hooks_write_atof_category_profile_and_fidelity() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    let temp = tempfile::tempdir().unwrap();
    let atof_dir = temp.path().join("atof");
    std::fs::create_dir_all(&atof_dir).unwrap();
    let mut config = test_config();
    config.plugin_config = Some(json!({
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 1,
                    "atof": {
                        "enabled": true,
                        "output_directory": atof_dir,
                        "filename": "events.jsonl",
                        "mode": "overwrite"
                    }
                }
            }
        ]
    }));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let handle =
        tokio::spawn(async move { serve_listener(listener, config, Some(shutdown_rx)).await });

    wait_for_gateway(&url).await;
    let client = test_http_client();

    let response = client
        .post(format!("{url}/hooks/hermes"))
        .json(&json!({
            "hook_event_name": "pre_api_request",
            "session_id": "hermes-atof-exact",
            "extra": {
                "task_id": "task-1",
                "api_request_id": "turn-1:api:2",
                "api_call_count": 2,
                "model": "qwen",
                "provider": "custom",
                "request": {
                    "method": "POST",
                    "body": {
                        "model": "qwen",
                        "messages": [
                            { "role": "user", "content": "hello" }
                        ],
                        "tools": [
                            { "type": "function", "function": { "name": "search_files" } }
                        ]
                    }
                }
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = client
        .post(format!("{url}/hooks/hermes"))
        .json(&json!({
            "hook_event_name": "post_api_request",
            "session_id": "hermes-atof-exact",
            "extra": {
                "task_id": "task-1",
                "api_request_id": "turn-1:api:2",
                "api_call_count": 2,
                "model": "qwen",
                "response": {
                    "model": "qwen",
                    "finish_reason": "tool_calls",
                    "assistant_message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search_files",
                                    "arguments": "{\"query\":\"needle\"}"
                                }
                            }
                        ]
                    },
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "cost": { "total": 0.0042 }
                    }
                }
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = client
        .post(format!("{url}/hooks/hermes"))
        .json(&json!({
            "hook_event_name": "pre_api_request",
            "session_id": "hermes-atof-lossy",
            "extra": {
                "task_id": "task-2",
                "api_call_count": 4,
                "model": "qwen",
                "provider": "custom",
                "request": null,
                "message_count": 2
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();

    let events = std::fs::read_to_string(temp.path().join("atof/events.jsonl")).unwrap();
    let llm_events = events
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["category"] == "llm")
        .collect::<Vec<_>>();
    assert_eq!(
        llm_events.len(),
        4,
        "expected Hermes LLM exports, got {llm_events:?}"
    );

    let start = llm_events
        .iter()
        .find(|event| {
            event["scope_category"] == "start"
                && event["metadata"]["api_call_id"] == json!("turn-1:api:2")
        })
        .unwrap();
    assert_eq!(start["category_profile"]["model_name"], json!("qwen"));
    assert_eq!(start["metadata"]["provider_payload_exact"], json!(true));
    assert_eq!(
        start["metadata"]["fidelity_source"],
        json!("hermes_api_hooks_sanitized")
    );
    assert_eq!(
        start["data"]["content"]["messages"][0]["content"],
        json!("hello")
    );
    assert_eq!(
        start["data"]["content"]["tools"][0]["function"]["name"],
        json!("search_files")
    );

    let end = llm_events
        .iter()
        .find(|event| {
            event["scope_category"] == "end"
                && event["metadata"]["api_call_id"] == json!("turn-1:api:2")
        })
        .unwrap();
    assert_eq!(end["category_profile"]["model_name"], json!("qwen"));
    assert_eq!(end["metadata"]["provider_payload_exact"], json!(true));
    assert_eq!(end["data"]["tool_calls"][0]["id"], json!("call-1"));
    assert_eq!(
        end["data"]["tool_calls"][0]["function"]["name"],
        json!("search_files")
    );
    assert_eq!(end["data"]["usage"]["prompt_tokens"], json!(10));
    assert_eq!(end["data"]["usage"]["completion_tokens"], json!(5));

    let lossy_start = llm_events
        .iter()
        .find(|event| {
            event["scope_category"] == "start"
                && event["metadata"]["api_call_id"] == json!("hermes-atof-lossy:task-2:4")
        })
        .unwrap();
    assert_eq!(lossy_start["category_profile"]["model_name"], json!("qwen"));
    assert_eq!(
        lossy_start["metadata"]["provider_payload_exact"],
        json!(false)
    );
    assert_eq!(
        lossy_start["data"]["content"]["fidelity"]["provider_payload_exact"],
        json!(false)
    );
    assert_eq!(lossy_start["data"]["content"]["message_count"], json!(2));
}

#[tokio::test]
async fn serve_listener_hermes_api_request_error_writes_lossy_atof_error_event() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    let temp = tempfile::tempdir().unwrap();
    let atof_dir = temp.path().join("atof");
    std::fs::create_dir_all(&atof_dir).unwrap();
    let mut config = test_config();
    config.plugin_config = Some(json!({
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 1,
                    "atof": {
                        "enabled": true,
                        "output_directory": atof_dir,
                        "filename": "events.jsonl",
                        "mode": "overwrite"
                    }
                }
            }
        ]
    }));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let handle =
        tokio::spawn(async move { serve_listener(listener, config, Some(shutdown_rx)).await });

    wait_for_gateway(&url).await;
    let client = test_http_client();

    let response = client
        .post(format!("{url}/hooks/hermes"))
        .json(&json!({
            "hook_event_name": "pre_api_request",
            "session_id": "hermes-atof-error",
            "extra": {
                "task_id": "task-err",
                "api_request_id": "turn-1:api:3",
                "api_call_count": 3,
                "model": "qwen",
                "provider": "custom",
                "request": {
                    "method": "POST",
                    "body": {
                        "model": "qwen",
                        "messages": [
                            { "role": "user", "content": "hello" }
                        ]
                    }
                }
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = client
        .post(format!("{url}/hooks/hermes"))
        .json(&json!({
            "hook_event_name": "api_request_error",
            "session_id": "hermes-atof-error",
            "extra": {
                "task_id": "task-err",
                "api_request_id": "turn-1:api:3",
                "api_call_count": 3,
                "model": "qwen",
                "provider": "custom",
                "status_code": 502,
                "retry_count": 1,
                "max_retries": 2,
                "retryable": true,
                "reason": "upstream",
                "error": {
                    "type": "BadGateway",
                    "message": "gateway upstream error"
                }
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();

    let events = std::fs::read_to_string(temp.path().join("atof/events.jsonl")).unwrap();
    let llm_events = events
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["category"] == "llm")
        .collect::<Vec<_>>();
    assert_eq!(
        llm_events.len(),
        2,
        "expected Hermes error-path LLM exports, got {llm_events:?}"
    );

    let end = llm_events
        .iter()
        .find(|event| {
            event["scope_category"] == "end"
                && event["metadata"]["api_call_id"] == json!("turn-1:api:3")
        })
        .unwrap();
    let start = llm_events
        .iter()
        .find(|event| {
            event["scope_category"] == "start"
                && event["metadata"]["api_call_id"] == json!("turn-1:api:3")
        })
        .unwrap();
    assert_eq!(start["metadata"]["provider_payload_exact"], json!(true));
    assert_eq!(
        start["metadata"]["fidelity_source"],
        json!("hermes_api_hooks_sanitized")
    );
    assert_eq!(
        start["data"]["content"]["messages"][0]["content"],
        json!("hello")
    );
    assert_eq!(end["category_profile"]["model_name"], json!("qwen"));
    assert_eq!(end["metadata"]["provider_payload_exact"], json!(false));
    assert_eq!(
        end["metadata"]["fidelity_source"],
        json!("hermes_api_hooks")
    );
    assert_eq!(end["data"]["status_code"], json!(502));
    assert_eq!(end["data"]["retry_count"], json!(1));
    assert_eq!(end["data"]["retryable"], json!(true));
    assert_eq!(end["data"]["reason"], json!("upstream"));
    assert_eq!(
        end["data"]["error"]["message"],
        json!("gateway upstream error")
    );
}

#[tokio::test]
async fn serve_listener_hermes_post_tool_call_writes_atof_tool_events() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    let temp = tempfile::tempdir().unwrap();
    let atof_dir = temp.path().join("atof");
    std::fs::create_dir_all(&atof_dir).unwrap();
    let mut config = test_config();
    config.plugin_config = Some(json!({
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 1,
                    "atof": {
                        "enabled": true,
                        "output_directory": atof_dir,
                        "filename": "events.jsonl",
                        "mode": "overwrite"
                    }
                }
            }
        ]
    }));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let handle =
        tokio::spawn(async move { serve_listener(listener, config, Some(shutdown_rx)).await });

    wait_for_gateway(&url).await;
    let client = test_http_client();

    for payload in [
        json!({
            "hook_event_name": "on_session_start",
            "session_id": "hermes-tool-atof"
        }),
        json!({
            "hook_event_name": "pre_tool_call",
            "session_id": "hermes-tool-atof",
            "tool_name": "search_files",
            "tool_input": { "query": "needle" },
            "extra": {
                "task_id": "task-1",
                "tool_call_id": "call-search-1"
            }
        }),
        json!({
            "hook_event_name": "post_tool_call",
            "session_id": "hermes-tool-atof",
            "tool_name": "search_files",
            "tool_input": { "query": "needle" },
            "tool_response": { "total_count": 6 },
            "extra": {
                "task_id": "task-1",
                "tool_call_id": "call-search-1"
            }
        }),
        json!({
            "hook_event_name": "on_session_finalize",
            "session_id": "hermes-tool-atof"
        }),
    ] {
        let response = client
            .post(format!("{url}/hooks/hermes"))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();

    let events = std::fs::read_to_string(temp.path().join("atof/events.jsonl")).unwrap();
    let tool_events = events
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["category"] == "tool")
        .collect::<Vec<_>>();
    assert_eq!(
        tool_events.len(),
        2,
        "expected Hermes tool start/end exports, got {tool_events:?}"
    );

    let start = tool_events
        .iter()
        .find(|event| event["scope_category"] == "start")
        .unwrap();
    assert_eq!(start["name"], json!("search_files"));
    assert_eq!(
        start["category_profile"]["tool_call_id"],
        json!("call-search-1")
    );
    assert_eq!(start["data"]["query"], json!("needle"));

    let end = tool_events
        .iter()
        .find(|event| event["scope_category"] == "end")
        .unwrap();
    assert_eq!(end["name"], json!("search_files"));
    assert_eq!(
        end["category_profile"]["tool_call_id"],
        json!("call-search-1")
    );
    assert_eq!(end["data"]["total_count"], json!(6));
}

#[tokio::test]
async fn serve_listener_routed_gateway_wire_formats_write_atof_category_profile_and_usage() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    async fn anthropic_messages() -> TestServer {
        async fn messages(_headers: HeaderMap, _request: Request<Body>) -> impl IntoResponse {
            Json(json!({
                "id": "msg_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4",
                "content": [
                    {"type": "text", "text": "I will search."},
                    {"type": "tool_use", "id": "toolu_01", "name": "search", "input": {"query": "file"}}
                ],
                "stop_reason": "tool_use",
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "cache_read_input_tokens": 3,
                    "cost": {"total": 0.0042}
                }
            }))
        }

        let app = Router::new().route("/v1/messages", post(messages));
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

    async fn openai_routed() -> TestServer {
        async fn chat(_headers: HeaderMap, request: Request<Body>) -> impl IntoResponse {
            let path = request.uri().path().to_string();
            if path == "/v1/responses" {
                Json(json!({
                    "id": "resp_1",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "I will check the weather."}]
                        },
                        {
                            "type": "function_call",
                            "call_id": "call_weather_1",
                            "name": "get_weather",
                            "arguments": "{\"city\":\"SF\"}",
                            "status": "completed"
                        }
                    ],
                    "usage": {
                        "input_tokens": 75,
                        "output_tokens": 20,
                        "total_tokens": 95,
                        "input_tokens_details": {"cached_tokens": 10},
                        "cost_usd": 0.005
                    }
                }))
            } else {
                Json(json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "I will inspect.",
                            "tool_calls": [
                                {
                                    "id": "call_read_1",
                                    "type": "function",
                                    "function": {"name": "read", "arguments": "{\"path\":\"api.py\"}"}
                                }
                            ]
                        },
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 4,
                        "total_tokens": 7,
                        "prompt_tokens_details": {"cached_tokens": 2},
                        "cost_usd": 0.001
                    }
                }))
            }
        }

        let app = Router::new()
            .route("/v1/chat/completions", post(chat))
            .route("/v1/responses", post(chat));
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

    let temp = tempfile::tempdir().unwrap();
    let atof_dir = temp.path().join("atof");
    std::fs::create_dir_all(&atof_dir).unwrap();

    let anthropic_upstream = anthropic_messages().await;
    let openai_upstream = openai_routed().await;

    let mut config = test_config();
    config.anthropic_base_url = anthropic_upstream.url();
    config.openai_base_url = openai_upstream.url();
    config.plugin_config = Some(json!({
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 1,
                    "atof": {
                        "enabled": true,
                        "output_directory": atof_dir,
                        "filename": "events.jsonl",
                        "mode": "overwrite"
                    }
                }
            }
        ]
    }));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let handle =
        tokio::spawn(async move { serve_listener(listener, config, Some(shutdown_rx)).await });

    wait_for_gateway(&url).await;
    let client = test_http_client();

    let response = client
        .post(format!("{url}/v1/messages"))
        .header("content-type", "application/json")
        .header("x-api-key", "sk-ant-test")
        .header("x-nemo-relay-session-id", "hermes-routed-atof")
        .json(&json!({
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "Find the file."}],
            "tools": [{"name": "search", "input_schema": {"type": "object"}}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = client
        .post(format!("{url}/v1/responses"))
        .header("content-type", "application/json")
        .header("authorization", "Bearer test")
        .header("x-nemo-relay-session-id", "hermes-routed-atof")
        .json(&json!({
            "model": "gpt-4o",
            "input": "Find the weather.",
            "tools": [{"type": "function", "name": "get_weather"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = client
        .post(format!("{url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .header("authorization", "Bearer test")
        .header("x-nemo-relay-session-id", "hermes-routed-atof")
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Inspect the files."}],
            "tools": [{"type": "function", "function": {"name": "read"}}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();

    let events = std::fs::read_to_string(temp.path().join("atof/events.jsonl")).unwrap();
    let llm_events = events
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["category"] == "llm")
        .collect::<Vec<_>>();
    assert_eq!(
        llm_events.len(),
        6,
        "expected three routed LLM start/end pairs, got {llm_events:?}"
    );

    let anthropic_start = llm_events
        .iter()
        .find(|event| {
            event["scope_category"] == "start"
                && event["name"] == "anthropic.messages"
                && event["metadata"]["gateway_path"] == "/v1/messages"
        })
        .unwrap();
    assert_eq!(
        anthropic_start["category_profile"]["model_name"],
        json!("claude-sonnet-4")
    );
    assert_eq!(
        anthropic_start["data"]["content"]["messages"][0]["content"],
        json!("Find the file.")
    );

    let anthropic_end = llm_events
        .iter()
        .find(|event| {
            event["scope_category"] == "end"
                && event["name"] == "anthropic.messages"
                && event["metadata"]["gateway_path"] == "/v1/messages"
        })
        .unwrap();
    assert_eq!(
        anthropic_end["category_profile"]["annotated_response"]["tool_calls"][0]["id"],
        json!("toolu_01")
    );
    assert_eq!(anthropic_end["data"]["content"][1]["id"], json!("toolu_01"));
    assert_eq!(anthropic_end["data"]["usage"]["input_tokens"], json!(11));
    assert_eq!(
        anthropic_end["data"]["usage"]["cost"]["total"],
        json!(0.0042)
    );

    let responses_end = llm_events
        .iter()
        .find(|event| {
            event["scope_category"] == "end"
                && event["name"] == "openai.responses"
                && event["metadata"]["gateway_path"] == "/v1/responses"
        })
        .unwrap();
    assert_eq!(
        responses_end["category_profile"]["model_name"],
        json!("gpt-4o")
    );
    assert_eq!(
        responses_end["category_profile"]["annotated_response"]["tool_calls"][0]["id"],
        json!("call_weather_1")
    );
    assert_eq!(
        responses_end["data"]["output"][1]["call_id"],
        json!("call_weather_1")
    );
    assert_eq!(
        responses_end["data"]["usage"]["input_tokens_details"]["cached_tokens"],
        json!(10)
    );
    assert_eq!(responses_end["data"]["usage"]["cost_usd"], json!(0.005));

    let chat_end = llm_events
        .iter()
        .find(|event| {
            event["scope_category"] == "end"
                && event["name"] == "openai.chat_completions"
                && event["metadata"]["gateway_path"] == "/v1/chat/completions"
        })
        .unwrap();
    assert_eq!(chat_end["category_profile"]["model_name"], json!("gpt-4o"));
    assert_eq!(
        chat_end["category_profile"]["annotated_response"]["tool_calls"][0]["id"],
        json!("call_read_1")
    );
    assert_eq!(
        chat_end["data"]["choices"][0]["message"]["tool_calls"][0]["id"],
        json!("call_read_1")
    );
    assert_eq!(
        chat_end["data"]["usage"]["prompt_tokens_details"]["cached_tokens"],
        json!(2)
    );
    assert_eq!(chat_end["data"]["usage"]["cost_usd"], json!(0.001));
}

#[tokio::test]
async fn serve_listener_records_codex_stop_atof_contract() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    let temp = tempfile::tempdir().unwrap();
    let atof_dir = temp.path().join("atof");
    std::fs::create_dir_all(&atof_dir).unwrap();
    let mut config = test_config();
    config.plugin_config = Some(json!({
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 1,
                    "atof": {
                        "enabled": true,
                        "output_directory": atof_dir,
                        "filename": "events.jsonl",
                        "mode": "overwrite"
                    }
                }
            }
        ]
    }));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let handle =
        tokio::spawn(async move { serve_listener(listener, config, Some(shutdown_rx)).await });

    wait_for_gateway(&url).await;
    let client = test_http_client();
    for payload in [
        json!({
            "session_id": "codex-atof-session",
            "hook_event_name": "sessionStart",
            "cwd": "/workspace",
            "model": "gpt-5.1-codex"
        }),
        json!({
            "session_id": "codex-atof-session",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "Inspect the repository."
        }),
        json!({
            "session_id": "codex-atof-session",
            "hook_event_name": "PreToolUse",
            "tool_call_id": "tool-call-1",
            "tool_name": "Read",
            "tool_input": { "file_path": "README.md" }
        }),
        json!({
            "session_id": "codex-atof-session",
            "hook_event_name": "PostToolUse",
            "tool_call_id": "tool-call-1",
            "tool_name": "Read",
            "tool_output": { "bytes": 42 },
            "status": "success"
        }),
        json!({
            "session_id": "codex-atof-session",
            "hook_event_name": "Stop",
            "response": "Done."
        }),
    ] {
        let response = client
            .post(format!("{url}/hooks/codex"))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.json::<Value>().await.unwrap(), json!({}));
    }

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();
    assert!(nemo_relay::plugin::active_plugin_report().is_none());

    let events = std::fs::read_to_string(temp.path().join("atof/events.jsonl")).unwrap();
    let events = events
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();

    assert!(events.iter().all(|event| event["atof_version"] == "0.1"));
    assert!(!events.iter().any(|event| {
        event["kind"] == "scope"
            && event["scope_category"] == "start"
            && event["category"] == "agent"
            && event["name"] == "codex"
    }));

    let turn_start = find_scope_event(&events, "codex-turn", "agent", "start");
    let turn_end = find_scope_event(&events, "codex-turn", "agent", "end");
    assert_eq!(turn_start["uuid"], turn_end["uuid"]);
    assert_eq!(
        turn_start["data"],
        json!({
            "session_id": "codex-atof-session",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "Inspect the repository."
        })
    );
    assert_eq!(turn_start["metadata"]["session_id"], "codex-atof-session");
    assert_eq!(turn_start["metadata"]["agent_kind"], "codex");
    assert_eq!(turn_start["metadata"]["nemo_relay_scope_role"], "turn");
    assert_eq!(turn_start["metadata"]["turn_source"], "user_prompt");
    assert_eq!(turn_end["data"]["hook_event_name"], "Stop");
    assert_eq!(turn_end["data"]["response"], "Done.");

    let tool_start = find_scope_event(&events, "Read", "tool", "start");
    let tool_end = find_scope_event(&events, "Read", "tool", "end");
    assert_eq!(tool_start["uuid"], tool_end["uuid"]);
    assert_eq!(tool_start["parent_uuid"], turn_start["uuid"]);
    assert_eq!(tool_end["parent_uuid"], turn_start["uuid"]);
    assert_eq!(
        tool_start["category_profile"]["tool_call_id"],
        "tool-call-1"
    );
    assert_eq!(tool_end["category_profile"]["tool_call_id"], "tool-call-1");
    assert_eq!(tool_start["data"], json!({ "file_path": "README.md" }));
    assert_eq!(tool_end["data"], json!({ "bytes": 42 }));
    assert_eq!(tool_start["metadata"]["agent_kind"], "codex");
    assert_eq!(tool_end["metadata"]["agent_kind"], "codex");
    assert_eq!(tool_end["metadata"]["status"], "success");
}

#[tokio::test]
async fn serve_listener_activates_any_registered_plugin_kind() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();
    let _ = deregister_plugin(GENERIC_TEST_PLUGIN_KIND);
    GENERIC_TEST_PLUGIN_REGISTRATIONS.store(0, Ordering::SeqCst);
    GENERIC_TEST_PLUGIN_DEREGISTRATIONS.store(0, Ordering::SeqCst);
    register_plugin(Arc::new(GenericTestPlugin)).unwrap();

    let mut config = test_config();
    config.plugin_config = Some(json!({
        "version": 1,
        "components": [
            {
                "kind": GENERIC_TEST_PLUGIN_KIND,
                "enabled": true,
                "config": {}
            }
        ]
    }));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let handle =
        tokio::spawn(async move { serve_listener(listener, config, Some(shutdown_rx)).await });

    wait_for_gateway(&url).await;
    assert_eq!(GENERIC_TEST_PLUGIN_REGISTRATIONS.load(Ordering::SeqCst), 1);

    let response = test_http_client()
        .post(format!("{url}/hooks/codex"))
        .json(&json!({
            "session_id": "generic-plugin-session",
            "hook_event_name": "sessionStart"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();
    assert_eq!(
        GENERIC_TEST_PLUGIN_DEREGISTRATIONS.load(Ordering::SeqCst),
        1
    );
    assert!(nemo_relay::plugin::active_plugin_report().is_none());
    let _ = deregister_plugin(GENERIC_TEST_PLUGIN_KIND);
}

#[tokio::test]
async fn serve_listener_activates_adaptive_plugin_config() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    let mut config = test_config();
    config.plugin_config = Some(json!({
        "version": 1,
        "components": [
            {
                "kind": "adaptive",
                "enabled": true,
                "config": {
                    "version": 1,
                    "agent_id": "cli-test",
                    "state": {
                        "backend": {
                            "kind": "in_memory",
                            "config": {}
                        }
                    }
                }
            }
        ]
    }));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let handle =
        tokio::spawn(async move { serve_listener(listener, config, Some(shutdown_rx)).await });

    wait_for_gateway(&url).await;

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn serve_listener_rejects_invalid_plugin_config() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    let mut config = test_config();
    config.plugin_config = Some(json!({
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 1,
                    "atof": {
                        "enabled": true,
                        "mode": "invalid"
                    }
                }
            }
        ]
    }));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let (_shutdown_tx, shutdown_rx) = oneshot::channel();
    let error = serve_listener(listener, config, Some(shutdown_rx))
        .await
        .unwrap_err();

    assert!(error.to_string().contains("ATOF mode"));
    assert!(nemo_relay::plugin::active_plugin_report().is_none());
}

#[tokio::test]
async fn gateway_errors_render_structured_json_responses() {
    let response = CliError::InvalidPayload("bad input".into()).into_response();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["error"]["type"], json!("nemo_relay_gateway_error"));
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
    assert!(body.get("user_message").is_none());
    assert!(body.get("agent_message").is_none());
}

#[tokio::test]
async fn pre_tool_hook_rejects_when_conditional_guardrail_blocks() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = deregister_tool_conditional_execution_guardrail("cli-pre-tool-blocker");
    const BLOCKED_TEST_TOOL: &str = "Nmf137BlockedTool";
    register_tool_conditional_execution_guardrail(
        "cli-pre-tool-blocker",
        1,
        Arc::new(|name, _args| {
            Ok((name == BLOCKED_TEST_TOOL).then(|| "blocked by policy".to_string()))
        }),
    )
    .unwrap();
    let _cleanup = ToolGuardrailCleanup("cli-pre-tool-blocker");

    let app = router(test_config());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/hooks/claude-code")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "session_id": "guardrail-session",
                        "hook_event_name": "PreToolUse",
                        "tool_use_id": "tool-1",
                        "tool_name": BLOCKED_TEST_TOOL,
                        "tool_input": { "command": "rm -rf /tmp/demo" }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        body["error"]["type"],
        json!("nemo_relay_guardrail_rejected")
    );
    assert_eq!(body["error"]["reason"], json!("blocked by policy"));
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

#[tokio::test]
async fn gateway_forwards_anthropic_count_tokens_without_llm_codec() {
    let upstream = spawn_anthropic_upstream().await;
    let mut config = test_config();
    config.anthropic_base_url = upstream.url();
    let app = router(config);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .header("x-api-key", "sk-ant-test")
                .body(Body::from(
                    json!({
                        "model": "claude-test",
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
    assert_eq!(body["path"], json!("/v1/messages/count_tokens"));
    assert_eq!(body["x_api_key"], json!("sk-ant-test"));
    assert_eq!(body["input_tokens"], json!(12));
}

#[tokio::test]
async fn gateway_forwards_claude_startup_probe_without_llm_observability() {
    let subscriber_name = "server-claude-startup-probe-no-llm-test";
    let _ = deregister_subscriber(subscriber_name);
    let captured_llm_starts = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
    let captured = captured_llm_starts.clone();
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::Start)
                && event.name() == "anthropic.messages"
                && event
                    .metadata()
                    .and_then(|metadata| metadata.get("gateway_path"))
                    .and_then(Value::as_str)
                    == Some("/v1/messages")
                && event
                    .input()
                    .and_then(|input| input.get("model"))
                    .and_then(Value::as_str)
                    == Some("claude-opus-4-8[1m]")
                && event
                    .input()
                    .and_then(|input| input.get("max_tokens"))
                    .and_then(Value::as_u64)
                    == Some(1)
                && event
                    .input()
                    .and_then(|input| input.get("messages"))
                    .and_then(Value::as_array)
                    .and_then(|messages| messages.first())
                    .and_then(|message| message.get("content"))
                    .and_then(Value::as_str)
                    == Some("test")
            {
                captured.lock().unwrap().push(json!({
                    "input": event.input().cloned().unwrap_or(Value::Null),
                    "metadata": event.metadata().cloned().unwrap_or(Value::Null)
                }));
            }
        }),
    )
    .unwrap();
    let _subscriber_cleanup = SubscriberCleanup(subscriber_name);

    let upstream = spawn_anthropic_upstream().await;
    let mut config = test_config();
    config.anthropic_base_url = upstream.url();
    let app = router(config);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .header("x-api-key", "sk-ant-test")
                .header("x-claude-code-session-id", "claude-probe")
                .body(Body::from(
                    json!({
                        "model": "claude-opus-4-8[1m]",
                        "max_tokens": 1,
                        "messages": [
                            {
                                "role": "user",
                                "content": "test"
                            }
                        ]
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
    assert_eq!(body["path"], json!("/v1/messages"));
    assert_eq!(body["model"], json!("claude-opus-4-8[1m]"));
    assert_eq!(body["prompt"], json!("test"));

    flush_subscribers().unwrap();
    assert!(
        captured_llm_starts.lock().unwrap().is_empty(),
        "Claude startup probe must not emit a managed LLM span"
    );
}

async fn wait_for_gateway(url: &str) {
    let client = test_http_client();
    for _ in 0..50 {
        if let Ok(response) = client.get(format!("{url}/healthz")).send().await
            && response.status().is_success()
        {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("gateway did not become healthy at {url}");
}

async fn spawn_upstream(streaming: bool) -> TestServer {
    async fn chat(headers: HeaderMap, body: Bytes) -> impl IntoResponse {
        let payload: Value = serde_json::from_slice(&body).unwrap();
        Json(json!({
            "model": payload["model"],
            "input": payload["input"],
            "authorization": headers
                .get(header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            "x_test_intercept": headers
                .get("x-test-intercept")
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

async fn spawn_anthropic_upstream() -> TestServer {
    async fn messages(headers: HeaderMap, request: Request<Body>) -> impl IntoResponse {
        let body = request.into_body().collect().await.unwrap().to_bytes();
        let payload: Value = serde_json::from_slice(&body).unwrap();
        Json(json!({
            "path": "/v1/messages",
            "x_api_key": headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok()),
            "model": payload["model"],
            "prompt": payload["messages"][0]["content"]
        }))
    }

    async fn count_tokens(headers: HeaderMap, request: Request<Body>) -> impl IntoResponse {
        Json(json!({
            "path": request.uri().path(),
            "x_api_key": headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok()),
            "input_tokens": 12
        }))
    }

    let app = Router::new()
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens));
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
