// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::configuration::GatewayConfig;
use axum::{Json, Router, routing::post};
use nemo_relay::api::event::Event;
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::plugin::dynamic::{
    NativePluginLoadSpec, PluginHostActivation, load_native_plugins,
};
use nemo_relay::plugin::{PluginComponentSpec, PluginConfig};
use serde_json::json;

type Captures = Arc<Mutex<Vec<(String, HeaderMap, Value)>>>;

async fn capture(State(captures): State<Captures>, request: Request<Body>) -> Response<Body> {
    let (parts, body) = request.into_parts();
    let body: Value =
        serde_json::from_slice(&axum::body::to_bytes(body, 1024 * 1024).await.unwrap()).unwrap();
    captures
        .lock()
        .unwrap()
        .push((parts.uri.path().into(), parts.headers.clone(), body.clone()));
    let path = parts.uri.path();
    if path == "/redirect" {
        return Response::builder()
            .status(307)
            .header("location", "/stolen")
            .body(Body::empty())
            .unwrap();
    }
    if path == "/fail" {
        return Response::builder()
            .status(503)
            .body(Body::from(
                parts.headers["authorization"].to_str().unwrap().to_owned(),
            ))
            .unwrap();
    }
    let model = body["model"].as_str().unwrap();
    let content = if path == "/echo" {
        parts.headers["authorization"].to_str().unwrap().to_owned()
    } else {
        "ok".to_owned()
    };
    let response = if path == "/responses" {
        json!({"id":"resp_test","object":"response","status":"completed","model":model,"output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}})
    } else if path == "/messages" {
        json!({"id":"msg_test","type":"message","role":"assistant","model":model,"content":[{"type":"text","text":content}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}})
    } else {
        json!({"id":"chatcmpl_test","object":"chat.completion","created":1,"model":model,"choices":[{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}})
    };
    if body["stream"] == true {
        let events = if path == "/responses" {
            vec![json!({"type":"response.completed","response":response})]
        } else if path == "/messages" {
            vec![
                json!({"type":"message_start","message":response}),
                json!({"type":"message_stop"}),
            ]
        } else {
            vec![
                json!({"id":"chatcmpl_test","object":"chat.completion.chunk","created":1,"model":model,"choices":[{"index":0,"delta":{"role":"assistant","content":content},"finish_reason":"stop"}]}),
            ]
        };
        let text = events
            .into_iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>();
        Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from(text))
            .unwrap()
    } else {
        axum::response::IntoResponse::into_response(Json(response))
    }
}

async fn upstream() -> (String, Captures, tokio::task::JoinHandle<()>) {
    let captures = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/{*path}", post(capture))
        .with_state(captures.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, captures, server)
}

fn config(url: &str) -> GatewayConfig {
    // Ordinary forwarding would fail; success must come from the native capability.
    let mut config = GatewayConfig {
        openai_base_url: "http://127.0.0.1:1".into(),
        anthropic_base_url: "http://127.0.0.1:1".into(),
        ..Default::default()
    };
    for (name, path, format) in [
        ("chat", "chat", LlmProviderFormat::OpenaiChat),
        ("responses", "responses", LlmProviderFormat::OpenaiResponses),
        (
            "anthropic",
            "messages",
            LlmProviderFormat::AnthropicMessages,
        ),
        ("redirect", "redirect", LlmProviderFormat::OpenaiChat),
        ("fail", "fail", LlmProviderFormat::OpenaiChat),
        ("echo", "echo", LlmProviderFormat::OpenaiChat),
    ] {
        config.caller_credential_targets.insert(
            name.into(),
            CallerCredentialTarget {
                url: format!("{url}/{path}"),
                format,
            },
        );
    }
    config
}

async fn gateway_call(
    state: AppState,
    path: &str,
    target: &str,
    streaming: bool,
    caller: usize,
) -> Result<String, String> {
    let payload = json!({"model":format!("caller-{caller}"),"messages":[{"role":"user","content":"hello"}],"input":"hello","max_tokens":8,"stream":streaming,"fixture_provider_targets": [target]});
    gateway_payload(state, path, payload, caller).await
}

async fn gateway_payload(
    state: AppState,
    path: &str,
    payload: Value,
    caller: usize,
) -> Result<String, String> {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .header(
            "authorization",
            format!("Bearer synthetic-caller-secret-{caller}"),
        )
        .header("chatgpt-account-id", format!("account-{caller}"))
        .header("anthropic-version", "2023-06-01")
        .header("cookie", "must-not-forward")
        .body(Body::from(payload.to_string()))
        .unwrap();
    let response = super::super::passthrough(State(state), request)
        .await
        .map_err(|e| e.to_string())?;
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .map_err(|e| e.to_string())?;
    Ok(String::from_utf8(bytes.to_vec()).unwrap())
}

// A real cdylib uses its own SDK Tokio runtime and crosses the C ABI in both directions.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn native_provider_dispatch_gateway_end_to_end() {
    let library = std::env::var_os("NEMO_RELAY_TEST_NATIVE_PLUGIN")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let name = format!(
                "{}nemo_relay_plugin_fixture{}",
                std::env::consts::DLL_PREFIX,
                std::env::consts::DLL_SUFFIX
            );
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/test-plugin-fixtures/debug")
                .join(name)
        });
    assert!(
        library.exists(),
        "run just build-test-plugin-fixtures first: {}",
        library.display()
    );
    let temp = tempfile::tempdir().unwrap();
    let manifest = temp.path().join("relay-plugin.toml");
    let library = serde_json::to_string(&library.to_string_lossy()).unwrap();
    std::fs::write(
        &manifest,
        format!(
            r#"
manifest_version = 1
[plugin]
id = "fixture_native"
kind = "rust_dynamic"
[compat]
relay = ">=0.8.0,<1.0"
native_api = "1"
[capabilities]
items = ["plugin_native"]
[source]
artifact = {library}
[load]
library = {library}
symbol = "nemo_relay_fixture_native_plugin"
"#
        ),
    )
    .unwrap();
    let activation = load_native_plugins([NativePluginLoadSpec {
        plugin_id: "fixture_native".into(),
        manifest_ref: manifest.to_string_lossy().into(),
    }])
    .unwrap();
    let mut plugins = PluginConfig::default();
    plugins.components.push(PluginComponentSpec {
        kind: "fixture_native".into(),
        enabled: true,
        config: serde_json::Map::from_iter([("private_provider".into(), json!(true))]),
    });
    let mut host = PluginHostActivation::initialize_exact(plugins)
        .await
        .unwrap();
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured_events = events.clone();
    register_subscriber(
        "private_provider_events",
        Arc::new(move |event| captured_events.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    let (url, captures, server) = upstream().await;
    let state = AppState::new(config(&url));
    let mut calls = Vec::new();
    for caller in 0..12 {
        let state = state.clone();
        calls.push(tokio::spawn(async move {
            let (path, target) = match caller % 3 {
                0 => ("/v1/chat/completions", "chat"),
                1 => ("/v1/responses", "responses"),
                _ => ("/v1/messages", "anthropic"),
            };
            gateway_call(state, path, target, caller % 2 == 0, caller)
                .await
                .unwrap()
        }));
    }
    for (caller, call) in calls.into_iter().enumerate() {
        let body = call.await.unwrap();
        assert!(body.contains(&format!("caller-{caller}")), "{body}");
        assert!(!body.contains("synthetic-caller-secret"));
        assert!(!body.contains("nemo_relay_gateway_error"), "{body}");
    }
    assert_eq!(captures.lock().unwrap().len(), 12);
    for (_, headers, body) in captures.lock().unwrap().iter() {
        let caller = body["model"]
            .as_str()
            .unwrap()
            .strip_prefix("caller-")
            .unwrap();
        assert_eq!(
            headers["authorization"],
            format!("Bearer synthetic-caller-secret-{caller}")
        );
        assert!(!headers.contains_key("cookie"));
        if caller.parse::<usize>().unwrap() % 3 != 2 {
            assert_eq!(headers["chatgpt-account-id"], format!("account-{caller}"));
        }
    }
    let before = captures.lock().unwrap().len();
    for target in ["unknown", "anthropic"] {
        let error = gateway_call(state.clone(), "/v1/chat/completions", target, false, 50)
            .await
            .unwrap_err();
        assert!(!error.contains("synthetic-caller-secret"));
    }
    assert_eq!(captures.lock().unwrap().len(), before);
    let error = gateway_call(state.clone(), "/v1/chat/completions", "redirect", false, 50)
        .await
        .unwrap_err();
    assert!(error.contains("307"));
    assert!(
        !captures
            .lock()
            .unwrap()
            .iter()
            .any(|(path, _, _)| path == "/stolen")
    );
    // A routing-model call, a repeated attempt, and a fallback all keep the same credential.
    let payload = json!({"model":"caller-51","messages":[{"role":"user","content":"hello"}],"stream":true,"fixture_provider_probe":"chat","fixture_provider_targets":["fail","fail","responses"]});
    gateway_payload(state.clone(), "/v1/responses", payload, 51)
        .await
        .unwrap();
    let rows = captures.lock().unwrap().clone();
    assert_eq!(
        rows.iter()
            .filter(|(_, _, body)| body["model"] == "caller-51")
            .count(),
        4
    );
    for (_, headers, body) in rows {
        if body["model"] == "caller-51" {
            assert_eq!(
                headers["authorization"],
                "Bearer synthetic-caller-secret-51"
            );
        }
    }
    for streaming in [false, true] {
        let body = gateway_call(state.clone(), "/v1/chat/completions", "echo", streaming, 52)
            .await
            .unwrap();
        assert!(!body.contains("synthetic-caller-secret-52"));
        assert!(body.contains("[REDACTED]"));
    }
    flush_subscribers().unwrap();
    assert!(events.lock().unwrap().len() >= 24);
    let event_json = serde_json::to_string(&*events.lock().unwrap()).unwrap();
    assert!(event_json.contains("caller-52"));
    assert!(!event_json.contains("synthetic-caller-secret"));
    assert!(!event_json.contains("must-not-forward"));
    deregister_subscriber("private_provider_events").unwrap();
    host.close().unwrap();
    activation.clear();
    server.abort();
}

#[tokio::test]
async fn provider_dispatch_never_uses_invocation_or_deployment_credentials() {
    let (url, captures, server) = upstream().await;
    let mut cfg = config(&url);
    cfg.openai_auth_header = Some("Bearer deployment-secret".into());
    let state = AppState::new(cfg);
    let token = crate::provider_auth::TransparentProxyCredential::from_static("invocation-secret");
    for provider_present in [false, true] {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_static("Bearer invocation-secret"),
        );
        if provider_present {
            headers.insert("api-key", HeaderValue::from_static("provider-secret"));
        }
        let source = token.consume(&mut headers).unwrap();
        let transport = ProviderTransport {
            client: state.http_no_redirect.clone(),
            targets: state.config.caller_credential_targets.clone(),
            source: ProviderRoute::OpenAiChatCompletions,
            headers,
            credential_present: source.provider_credential_present(),
            response_limit: 4096,
        };
        let result = transport
            .buffered(LlmProviderRequest {
                target: "chat".into(),
                content: json!({"model":"test"}),
            })
            .await;
        assert_eq!(result.is_ok(), provider_present);
    }
    let captures = captures.lock().unwrap();
    assert_eq!(captures.len(), 1);
    assert!(!captures[0].1.contains_key("authorization"));
    assert_eq!(captures[0].1["api-key"], "provider-secret");
    server.abort();
}
