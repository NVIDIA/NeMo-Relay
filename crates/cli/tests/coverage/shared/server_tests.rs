// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsString;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures_util::stream;
use http_body_util::BodyExt;
use nemo_relay::api::event::ScopeCategory;
use nemo_relay::api::llm::LlmRequestInterceptOutcome;
use nemo_relay::api::registry::{
    deregister_llm_execution_intercept, deregister_llm_request_intercept,
    deregister_llm_stream_execution_intercept, deregister_scope_sanitize_end_guardrail,
    deregister_tool_conditional_execution_guardrail, register_llm_execution_intercept,
    register_llm_request_intercept, register_llm_stream_execution_intercept,
    register_scope_sanitize_end_guardrail, register_tool_conditional_execution_guardrail,
};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::plugin::dynamic::DynamicPluginKind;
use nemo_relay::plugin::{
    ConfigDiagnostic, Plugin, PluginRegistration, PluginRegistrationContext, deregister_plugin,
    register_plugin,
};
use serde_json::{Map, Value, json};
use tokio::net::TcpListener;
use tokio::sync::{Semaphore, oneshot};
use tokio::task::JoinHandle;
use tower::ServiceExt;

use super::*;
use crate::configuration::BootstrapChallengeKey;
use crate::error::CliError;
use crate::plugins::lifecycle::ActiveDynamicPluginComponent;
use crate::test_support::PLUGIN_CONFIG_TEST_LOCK;

const GENERIC_TEST_PLUGIN_KIND: &str = "cli-test-generic-plugin";
static GENERIC_TEST_PLUGIN_REGISTRATIONS: AtomicUsize = AtomicUsize::new(0);
static GENERIC_TEST_PLUGIN_DEREGISTRATIONS: AtomicUsize = AtomicUsize::new(0);

struct EnvVarGuard {
    _guard: std::sync::MutexGuard<'static, ()>,
    key: &'static str,
    old: Option<OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let old = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, value);
        }
        Self {
            _guard: guard,
            key,
            old,
        }
    }

    fn remove(key: &'static str) -> Self {
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let old = std::env::var_os(key);
        unsafe {
            std::env::remove_var(key);
        }
        Self {
            _guard: guard,
            key,
            old,
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(old) = self.old.take() {
                std::env::set_var(self.key, old);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }
}

struct ToolGuardrailCleanup(&'static str);

impl Drop for ToolGuardrailCleanup {
    fn drop(&mut self) {
        let _ = deregister_tool_conditional_execution_guardrail(self.0);
    }
}

struct LlmExecutionInterceptCleanup(&'static str);

impl Drop for LlmExecutionInterceptCleanup {
    fn drop(&mut self) {
        let _ = deregister_llm_execution_intercept(self.0);
    }
}

struct LlmStreamExecutionInterceptCleanup(&'static str);

impl Drop for LlmStreamExecutionInterceptCleanup {
    fn drop(&mut self) {
        let _ = deregister_llm_stream_execution_intercept(self.0);
    }
}

struct SubscriberCleanup(&'static str);

impl Drop for SubscriberCleanup {
    fn drop(&mut self) {
        let _ = deregister_subscriber(self.0);
    }
}

struct ScopeEndSanitizerCleanup(&'static str);

impl Drop for ScopeEndSanitizerCleanup {
    fn drop(&mut self) {
        let _ = deregister_scope_sanitize_end_guardrail(self.0);
    }
}

struct RequestInterceptCleanup(&'static str);

impl Drop for RequestInterceptCleanup {
    fn drop(&mut self) {
        let _ = deregister_llm_request_intercept(self.0);
    }
}

fn test_http_client() -> reqwest::Client {
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
    crate::test_support::enable_operational_logs();
    GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        openai_auth_header: None,
        anthropic_base_url: "http://127.0.0.1".into(),
        anthropic_auth_header: None,
        metadata: None,
        plugin_config: None,
        max_hook_payload_bytes: crate::configuration::DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
        max_passthrough_body_bytes: crate::configuration::DEFAULT_MAX_PASSTHROUGH_BODY_BYTES,
    }
}

#[test]
fn startup_status_reports_bound_gateway_and_exporters() {
    let config = GatewayConfig {
        plugin_config: Some(json!({
            "version": 1,
            "components": [{
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 3,
                    "opentelemetry": {
                        "enabled": true,
                        "endpoints": [{
                            "type": "full",
                            "endpoint": "http://127.0.0.1:4318/v1/traces"
                        }]
                    }
                }
            }]
        })),
        ..test_config()
    };

    let output = render_startup_status("127.0.0.1:4567".parse().unwrap(), &config, false);

    assert!(output.contains("NeMo Relay"));
    assert!(output.contains("Gateway        http://127.0.0.1:4567"));
    assert!(output.contains("OpenTelemetry full http://127.0.0.1:4318/v1/traces"));
}

#[tokio::test]
async fn failed_server_result_is_reported_after_successful_teardown() {
    let sessions = SessionManager::new(test_config());
    let error = finish_server_shutdown(
        Err(std::io::Error::other("listener failed")),
        &sessions,
        None,
        "test-instance",
    )
    .await
    .expect_err("server failure should be preserved after teardown");

    assert!(matches!(error, CliError::Io(_)));
}

#[test]
fn startup_status_reports_not_configured_when_no_exporters() {
    let output = render_startup_status("127.0.0.1:4567".parse().unwrap(), &test_config(), false);

    assert!(output.contains("Exporters      not configured"));
}

fn write_missing_native_plugin_manifest(
    dir: &std::path::Path,
    plugin_id: &str,
) -> std::path::PathBuf {
    let missing_library = dir.join("missing-native-plugin");
    let manifest_ref = dir.join("relay-plugin.toml");
    let plugin_id = serde_json::to_string(plugin_id).unwrap();
    let library = serde_json::to_string(&missing_library.to_string_lossy()).unwrap();
    std::fs::write(
        &manifest_ref,
        format!(
            r#"manifest_version = 1

[plugin]
id = {plugin_id}
kind = "rust_dynamic"

[compat]
relay = "={version}"
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[load]
library = {library}
symbol = "nemo_relay_missing_native_plugin"
"#,
            version = env!("CARGO_PKG_VERSION"),
        ),
    )
    .unwrap();
    manifest_ref
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

async fn assert_payload_too_large_response(response: axum::response::Response) {
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["error"]["type"], json!("nemo_relay_gateway_error"));
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("payload too large")),
        "unexpected 413 body: {body}"
    );
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
async fn hook_payload_above_axum_default_succeeds_with_relay_default_limit() {
    let app = router(test_config());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/hooks/codex")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "session_id": "codex-large-hook",
                        "hook_event_name": "sessionStart",
                        "large": "x".repeat(2 * 1024 * 1024 + 1024)
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn hook_payload_limit_returns_structured_413() {
    let mut config = test_config();
    config.max_hook_payload_bytes = 128;
    let app = router(config);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/hooks/codex")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "session_id": "codex-too-large-hook",
                        "hook_event_name": "sessionStart",
                        "large": "x".repeat(1024)
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_payload_too_large_response(response).await;
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
    assert_eq!(body["status"], json!("ok"));
    assert_eq!(body["service"], json!("nemo-relay"));
    assert_eq!(body["version"], json!(env!("CARGO_PKG_VERSION")));
    assert_eq!(
        body["bootstrap_protocol"],
        json!(crate::bootstrap::BOOTSTRAP_PROTOCOL_VERSION)
    );
    assert!(
        body["instance_id"]
            .as_str()
            .is_some_and(|value| !value.is_empty())
    );
}

#[tokio::test]
async fn healthz_rejects_a_different_persistent_gateway_fingerprint() {
    let app = router_with_state(AppState::new_with_bootstrap(
        test_config(),
        Some("expected-fingerprint".into()),
        Some(BootstrapChallengeKey::from_bytes(b"test challenge key")),
        false,
        None,
        None,
    ));
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/healthz")
                .header(
                    "x-nemo-relay-bootstrap-fingerprint",
                    "different-fingerprint",
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CONFLICT);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["status"], json!("incompatible"));
    assert!(body.get("bootstrap_fingerprint").is_none());
}

#[tokio::test]
async fn managed_sidecar_requires_private_client_proof_for_forwarded_credentials() {
    let key = BootstrapChallengeKey::from_bytes(b"test challenge key");
    let state = AppState::new_with_bootstrap(
        test_config(),
        Some("expected-fingerprint".into()),
        Some(key.clone()),
        true,
        None,
        None,
    );
    let mut headers = HeaderMap::new();
    assert!(
        !state
            .authorize_provider_request(&mut headers)
            .unwrap()
            .allow_environment_provider_auth
    );
    headers.insert(
        crate::configuration::BOOTSTRAP_CLIENT_TOKEN_HEADER,
        HeaderValue::from_static("hmac-sha256:wrong"),
    );
    assert!(
        !state
            .authorize_provider_request(&mut headers)
            .unwrap()
            .allow_environment_provider_auth
    );
    headers.insert(
        crate::configuration::BOOTSTRAP_CLIENT_TOKEN_HEADER,
        HeaderValue::from_str(&key.client_token()).unwrap(),
    );
    assert!(
        state
            .authorize_provider_request(&mut headers)
            .unwrap()
            .allow_environment_provider_auth
    );

    let foreground = AppState::new(test_config());
    assert!(
        foreground
            .authorize_provider_request(&mut HeaderMap::new())
            .unwrap()
            .allow_environment_provider_auth
    );

    let transparent = AppState::new_with_bootstrap(
        test_config(),
        Some("transparent-fingerprint".into()),
        Some(BootstrapChallengeKey::from_bytes(b"test challenge key")),
        false,
        None,
        Some(crate::provider_auth::TransparentProxyCredential::from_static("test-proxy-token")),
    );
    assert!(
        transparent
            .authorize_provider_request(&mut HeaderMap::new())
            .is_err()
    );
    let mut transparent_headers = HeaderMap::new();
    transparent_headers.insert(
        crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER,
        HeaderValue::from_static("test-proxy-token"),
    );
    let authorization = transparent
        .authorize_provider_request(&mut transparent_headers)
        .unwrap();
    assert!(authorization.allow_environment_provider_auth);
    assert!(
        !transparent_headers
            .contains_key(crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER)
    );
}

#[tokio::test]
async fn healthz_only_refreshes_idle_activity_for_an_authenticated_heartbeat() {
    let challenge_key = BootstrapChallengeKey::from_bytes(b"test challenge key");
    let state = AppState::new_with_bootstrap(
        test_config(),
        Some("expected-fingerprint".into()),
        Some(challenge_key.clone()),
        true,
        None,
        None,
    );
    let activity = state.last_activity.clone();
    let baseline = std::time::Instant::now() - Duration::from_secs(30);
    *activity.lock().unwrap() = baseline;
    let app = router_with_state(state);

    for fingerprint in [
        None,
        Some("wrong-fingerprint"),
        Some("expected-fingerprint"),
    ] {
        let mut request = Request::builder().method("GET").uri("/healthz");
        if let Some(fingerprint) = fingerprint {
            request = request.header("x-nemo-relay-bootstrap-fingerprint", fingerprint);
        }
        let _ = app
            .clone()
            .oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(*activity.lock().unwrap(), baseline);
    }

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/healthz")
                .header("x-nemo-relay-bootstrap-fingerprint", "expected-fingerprint")
                .header(
                    "x-nemo-relay-bootstrap-nonce",
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-nemo-relay-bootstrap-proof")
            .unwrap(),
        challenge_key
            .proof(
                "expected-fingerprint",
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            )
            .as_str()
    );
    assert!(*activity.lock().unwrap() > baseline);
}

#[tokio::test]
async fn bootstrap_shutdown_requires_the_private_owner_token() {
    let (sender, receiver) = oneshot::channel();
    let app = router_with_state(AppState::new_with_bootstrap(
        test_config(),
        None,
        None,
        false,
        Some(BootstrapShutdown {
            token: "private-token".into(),
            sender: Arc::new(std::sync::Mutex::new(Some(sender))),
        }),
        None,
    ));
    let rejected = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/bootstrap/shutdown")
                .header("x-nemo-relay-bootstrap-token", "wrong-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::FORBIDDEN);

    let accepted = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/bootstrap/shutdown")
                .header("x-nemo-relay-bootstrap-token", "private-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(accepted.status(), StatusCode::NO_CONTENT);
    tokio::time::timeout(std::time::Duration::from_secs(1), receiver)
        .await
        .expect("shutdown signal was not delivered")
        .unwrap();
}

#[test]
fn readiness_file_is_published_atomically_with_gateway_identity() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("gateway.ready.json");
    let address = "127.0.0.1:43123".parse().unwrap();

    write_ready_file(&path, address, "test-instance").unwrap();

    let ready: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(ready["address"], json!(address));
    assert_eq!(ready["service"], json!("nemo-relay"));
    assert_eq!(ready["version"], json!(env!("CARGO_PKG_VERSION")));
    assert_eq!(
        ready["bootstrap_protocol"],
        json!(crate::bootstrap::BOOTSTRAP_PROTOCOL_VERSION)
    );
    assert_eq!(ready["instance_id"], json!("test-instance"));
    assert!(!path.with_extension("json.tmp").exists());
}

#[tokio::test]
async fn bind_listener_reports_an_actionable_address_conflict() {
    let occupied = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = occupied.local_addr().unwrap();
    let error = bind_listener(address).await.unwrap_err();
    let message = error.to_string();
    assert!(message.contains("port is already in use"));
    assert!(message.contains("ephemeral port"));
}

#[test]
fn readiness_file_reports_write_and_publish_failures() {
    let temp = tempfile::tempdir().unwrap();
    let address = "127.0.0.1:4040".parse().unwrap();

    let missing_parent = temp.path().join("missing").join("ready.json");
    let error = write_ready_file(&missing_parent, address, "write-failure").unwrap_err();
    assert!(error.to_string().contains("failed to write readiness file"));

    let directory_target = temp.path().join("ready.json");
    std::fs::create_dir(&directory_target).unwrap();
    let error = write_ready_file(&directory_target, address, "publish-failure").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("failed to publish readiness file")
    );
    assert!(!temp.path().join("ready.json.tmp").exists());
}

#[tokio::test]
async fn serve_listener_honors_plugin_idle_timeout_env() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _env = EnvVarGuard::set("NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS", "1");

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let handle = tokio::spawn(async move { serve_listener(listener, test_config(), None).await });

    wait_for_gateway(&url).await;
    let result = tokio::time::timeout(std::time::Duration::from_secs(3), handle)
        .await
        .expect("plugin idle timeout should stop the sidecar")
        .unwrap();
    result.unwrap();
}

#[tokio::test]
async fn serve_listener_flushes_shutdown_scope_events_without_plugins() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let subscriber_name = "server-shutdown-flush-without-plugins-test";
    let _ = deregister_subscriber(subscriber_name);
    let captured = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let captured_events = captured.clone();
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::End)
                && event
                    .metadata()
                    .and_then(|metadata| metadata.get("session_id"))
                    .and_then(Value::as_str)
                    == Some("shutdown-flush-session")
            {
                captured_events
                    .lock()
                    .unwrap()
                    .push(event.name().to_string());
            }
        }),
    )
    .unwrap();
    let _subscriber_cleanup = SubscriberCleanup(subscriber_name);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let handle =
        tokio::spawn(
            async move { serve_listener(listener, test_config(), Some(shutdown_rx)).await },
        );

    wait_for_gateway(&url).await;
    let client = test_http_client();
    for hook_event_name in ["sessionStart", "UserPromptSubmit"] {
        let response = client
            .post(format!("{url}/hooks/codex"))
            .json(&json!({
                "session_id": "shutdown-flush-session",
                "hook_event_name": hook_event_name
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();

    assert!(
        captured
            .lock()
            .unwrap()
            .iter()
            .any(|name| name == "codex-turn"),
        "expected shutdown scope-end event to be flushed"
    );
}

#[tokio::test]
async fn plugin_idle_timeout_parses_absent_invalid_zero_and_positive_values() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;

    let key = "NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS";
    let removed = EnvVarGuard::remove(key);
    assert_eq!(plugin_idle_timeout().unwrap(), None);
    drop(removed);

    let invalid = EnvVarGuard::set(key, "not-a-number");
    assert!(plugin_idle_timeout().is_err());
    drop(invalid);

    let zero = EnvVarGuard::set(key, "0");
    assert!(plugin_idle_timeout().is_err());
    drop(zero);

    let positive = EnvVarGuard::set(key, "2");
    assert_eq!(
        plugin_idle_timeout().unwrap(),
        Some(std::time::Duration::from_secs(2))
    );
    drop(positive);
}

#[tokio::test]
async fn serve_listener_waits_for_active_turn_before_plugin_idle_shutdown() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _env = EnvVarGuard::set("NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS", "1");

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let handle = tokio::spawn(async move { serve_listener(listener, test_config(), None).await });
    let client = test_http_client();

    wait_for_gateway(&url).await;
    let response = client
        .post(format!("{url}/hooks/codex"))
        .json(&json!({
            "session_id": "plugin-idle-open-session",
            "hook_event_name": "sessionStart"
        }))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let response = client
        .post(format!("{url}/hooks/codex"))
        .json(&json!({
            "session_id": "plugin-idle-open-session",
            "hook_event_name": "UserPromptSubmit"
        }))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());

    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
    assert!(
        !handle.is_finished(),
        "plugin sidecar exited before the active turn ended"
    );

    let response = client
        .post(format!("{url}/hooks/codex"))
        .json(&json!({
            "session_id": "plugin-idle-open-session",
            "hook_event_name": "Stop"
        }))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let result = tokio::time::timeout(std::time::Duration::from_secs(3), handle)
        .await
        .expect("plugin idle timeout should stop after Stop closes the turn")
        .unwrap();
    result.unwrap();
}

#[tokio::test]
async fn idle_shutdown_rechecks_activity_after_session_lookup() {
    let timeout = std::time::Duration::from_secs(1);
    let last_activity = Arc::new(std::sync::Mutex::new(
        std::time::Instant::now() - timeout - std::time::Duration::from_millis(1),
    ));
    let activity_during_lookup = Arc::clone(&last_activity);

    let ready = idle_shutdown_ready(&last_activity, timeout, async move {
        *activity_during_lookup.lock().unwrap() = std::time::Instant::now();
        false
    })
    .await;

    assert!(!ready, "new activity must cancel a stale shutdown decision");
}

#[tokio::test]
async fn idle_shutdown_requires_expiry_and_no_open_session_without_new_activity() {
    let timeout = std::time::Duration::from_secs(1);
    let recent = Arc::new(std::sync::Mutex::new(std::time::Instant::now()));
    assert!(!idle_shutdown_ready(&recent, timeout, async { false }).await);

    let expired = Arc::new(std::sync::Mutex::new(
        std::time::Instant::now() - timeout - std::time::Duration::from_millis(1),
    ));
    assert!(!idle_shutdown_ready(&expired, timeout, async { true }).await);
    assert!(idle_shutdown_ready(&expired, timeout, async { false }).await);
}

#[tokio::test]
async fn serve_listener_exits_after_codex_stop_without_session_end() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _env = EnvVarGuard::set("NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS", "1");

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let handle = tokio::spawn(async move { serve_listener(listener, test_config(), None).await });
    let client = test_http_client();

    wait_for_gateway(&url).await;
    for hook_event_name in ["sessionStart", "Stop"] {
        let response = client
            .post(format!("{url}/hooks/codex"))
            .json(&json!({
                "session_id": "plugin-idle-metadata-only-session",
                "hook_event_name": hook_event_name
            }))
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());
    }

    let result = tokio::time::timeout(std::time::Duration::from_secs(3), handle)
        .await
        .expect("plugin idle timeout should stop without a Codex SessionEnd")
        .unwrap();
    result.unwrap();
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
                    "version": 3,
                    "atof": {
                        "enabled": true,
                        "sinks": [{
                            "type": "file",
                            "output_directory": atof_dir,
                            "filename": "events.jsonl",
                            "mode": "overwrite"
                        }]
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
    for hook_event_name in ["SessionStart", "Stop"] {
        let response = client
            .post(format!("{url}/hooks/codex"))
            .json(&json!({
                "session_id": "plugin-bridge-session",
                "hook_event_name": hook_event_name
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    for hook_event_name in ["sessionStart", "UserPromptSubmit"] {
        let response = client
            .post(format!("{url}/hooks/codex"))
            .json(&json!({
                "session_id": "plugin-shutdown-open-session",
                "hook_event_name": hook_event_name,
                "prompt": "Leave this turn open until Relay shuts down."
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
        events.lines().count() >= 1,
        "expected an ATOF lifecycle event, got {events:?}"
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
    assert!(
        trajectories.iter().any(|trajectory| {
            atif_matches_session(trajectory, "plugin-shutdown-open-session")
                && trajectory["extra"]["observed_events"]
                    .as_array()
                    .is_some_and(|events| {
                        events.iter().any(|event| {
                            event["name"] == json!("codex-turn")
                                && event["scope_category"] == json!("end")
                        })
                    })
        }),
        "full server teardown must flush an open session's terminal ATIF snapshot before clearing plugins: {}",
        serde_json::to_string_pretty(&trajectories).unwrap()
    );
}

#[tokio::test]
async fn terminal_hook_responses_wait_for_their_atif_snapshot() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    std::fs::create_dir_all(&atif_dir).unwrap();
    let mut config = test_config();
    config.plugin_config = Some(json!({
        "version": 1,
        "components": [{
            "kind": "observability",
            "enabled": true,
            "config": {
                "version": 3,
                "atif": {
                    "enabled": true,
                    "output_directory": atif_dir,
                    "filename_template": "trajectory-{session_id}.json"
                }
            }
        }]
    }));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let handle =
        tokio::spawn(async move { serve_listener(listener, config, Some(shutdown_rx)).await });
    wait_for_gateway(&url).await;
    let client = test_http_client();

    for (path, session_id, turn_name, terminal_event, sanitizer_name) in [
        (
            "/hooks/codex",
            "codex-atif-response-boundary",
            "codex-turn",
            "Stop",
            "codex-atif-response-boundary-sanitizer",
        ),
        (
            "/hooks/claude-code",
            "claude-atif-response-boundary",
            "claude-code-turn",
            "SessionEnd",
            "claude-atif-response-boundary-sanitizer",
        ),
    ] {
        for hook_event_name in ["sessionStart", "UserPromptSubmit"] {
            let response = client
                .post(format!("{url}{path}"))
                .json(&json!({
                    "session_id": session_id,
                    "hook_event_name": hook_event_name,
                    "prompt": "Return one short answer."
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        let _ = deregister_scope_sanitize_end_guardrail(sanitizer_name);
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let started_tx = Arc::new(Mutex::new(Some(started_tx)));
        let release_rx = Arc::new(Mutex::new(Some(release_rx)));
        let expected_session_id = session_id.to_string();
        let expected_turn_name = turn_name.to_string();
        register_scope_sanitize_end_guardrail(
            sanitizer_name,
            0,
            Arc::new(move |event, fields| {
                let should_block = event.scope_category() == Some(ScopeCategory::End)
                    && event.name() == expected_turn_name
                    && event
                        .metadata()
                        .and_then(|metadata| metadata.get("session_id"))
                        .and_then(Value::as_str)
                        == Some(expected_session_id.as_str());
                let started = should_block
                    .then(|| started_tx.lock().unwrap().take())
                    .flatten();
                let release = should_block
                    .then(|| release_rx.lock().unwrap().take())
                    .flatten();
                Box::pin(async move {
                    if let Some(started) = started {
                        let _ = started.send(());
                        if let Some(release) = release {
                            let _ = release.await;
                        }
                    }
                    Ok(fields)
                })
            }),
        )
        .unwrap();
        let sanitizer_cleanup = ScopeEndSanitizerCleanup(sanitizer_name);

        let terminal_client = client.clone();
        let terminal_url = format!("{url}{path}");
        let terminal_session_id = session_id.to_string();
        let mut terminal = tokio::spawn(async move {
            terminal_client
                .post(terminal_url)
                .json(&json!({
                    "session_id": terminal_session_id,
                    "hook_event_name": terminal_event,
                    "response": "Done."
                }))
                .send()
                .await
                .unwrap()
        });
        tokio::time::timeout(std::time::Duration::from_secs(10), started_rx)
            .await
            .expect("terminal scope sanitizer should start")
            .unwrap();

        let early_response =
            tokio::time::timeout(std::time::Duration::from_millis(100), &mut terminal)
                .await
                .ok();
        let returned_early = early_response.is_some();
        let _ = release_tx.send(());
        let response = match early_response {
            Some(response) => response.unwrap(),
            None => terminal.await.unwrap(),
        };
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            !returned_early,
            "{terminal_event} returned before its terminal subscribers completed"
        );

        let trajectories = std::fs::read_dir(temp.path().join("atif"))
            .unwrap()
            .filter_map(Result::ok)
            .filter_map(|entry| {
                serde_json::from_slice::<Value>(&std::fs::read(entry.path()).ok()?).ok()
            })
            .collect::<Vec<_>>();
        assert!(
            trajectories
                .iter()
                .any(|trajectory| atif_matches_session(trajectory, session_id)),
            "terminal hook response must not precede the ATIF snapshot for {session_id}: {}",
            serde_json::to_string_pretty(&trajectories).unwrap()
        );
        drop(sanitizer_cleanup);
    }

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn serve_listener_observability_plugin_records_supported_agent_hooks() {
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
                    "version": 3,
                    "atof": {
                        "enabled": true,
                        "sinks": [{
                            "type": "file",
                            "output_directory": atof_dir,
                            "filename": "events.jsonl",
                            "mode": "overwrite"
                        }]
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
        for hook_event_name in [start_event, "UserPromptSubmit", end_event] {
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
    let turn_starts = events
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| {
            event["kind"] == "scope"
                && event["scope_category"] == "start"
                && event["metadata"]["nemo_relay_scope_role"] == "turn"
        })
        .filter_map(|event| event["name"].as_str().map(ToOwned::to_owned))
        .collect::<Vec<_>>();
    assert!(turn_starts.contains(&"codex-turn".to_string()));
    assert!(turn_starts.contains(&"claude-code-turn".to_string()));
    assert!(!turn_starts.contains(&"claude-code".to_string()));
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
                    "version": 3,
                    "atof": {
                        "enabled": true,
                        "sinks": [{
                            "type": "file",
                            "output_directory": atof_dir,
                            "filename": "events.jsonl",
                            "mode": "overwrite"
                        }]
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
        .header("x-nemo-relay-session-id", "gateway-routed-atof")
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
        .header("x-nemo-relay-session-id", "gateway-routed-atof")
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
        .header("x-nemo-relay-session-id", "gateway-routed-atof")
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
                    "version": 3,
                    "atof": {
                        "enabled": true,
                        "sinks": [{
                            "type": "file",
                            "output_directory": atof_dir,
                            "filename": "events.jsonl",
                            "mode": "overwrite"
                        }]
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

    let turn_start = find_scope_event(&events, "codex-turn", "custom", "start");
    let turn_end = find_scope_event(&events, "codex-turn", "custom", "end");
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
    assert_eq!(turn_end["metadata"]["hook_event_name"], "Stop");
    assert_eq!(turn_end["metadata"]["session_id"], "codex-atof-session");

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
async fn static_only_cli_configuration_keeps_the_legacy_lifecycle() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();
    let _ = deregister_plugin(GENERIC_TEST_PLUGIN_KIND);
    register_plugin(Arc::new(GenericTestPlugin)).unwrap();

    let activation = initialize_plugin_host(
        Some(json!({
            "version": 1,
            "components": [{
                "kind": GENERIC_TEST_PLUGIN_KIND,
                "enabled": true,
                "config": {}
            }]
        })),
        Vec::new(),
    )
    .await
    .expect("static CLI config should initialize")
    .expect("static CLI config should return a teardown guard");
    assert!(matches!(&activation, ServerPluginActivation::Static));

    nemo_relay::plugin::clear_plugin_configuration()
        .expect("legacy clear should remain available for a static-only CLI config");
    activation
        .clear()
        .expect("the static teardown guard should tolerate prior clear");
    let _ = deregister_plugin(GENERIC_TEST_PLUGIN_KIND);
}

#[test]
fn plugin_component_setup_errors_render_every_diagnostic_variant() {
    let adaptive = PluginComponentSetupError::Adaptive("adaptive failure".into());
    assert_eq!(adaptive.check_name(), "Adaptive plugin");
    assert_eq!(
        adaptive.diagnostic_details(),
        "registration failed: adaptive failure"
    );
    assert_eq!(
        adaptive.to_string(),
        "adaptive plugin registration failed: adaptive failure"
    );

    let pii = PluginComponentSetupError::PiiRedaction("pii failure".into());
    assert_eq!(pii.check_name(), "PII redaction plugin");
    assert_eq!(pii.diagnostic_details(), "registration failed: pii failure");
    assert_eq!(
        pii.to_string(),
        "PII redaction plugin registration failed: pii failure"
    );

    #[cfg(feature = "switchyard")]
    {
        let switchyard = PluginComponentSetupError::Switchyard("registration".into());
        assert_eq!(switchyard.check_name(), "Switchyard plugin");
        assert!(switchyard.to_string().contains("registration failed"));

        let atof = PluginComponentSetupError::SwitchyardAtof("atof ordering".into());
        assert_eq!(atof.check_name(), "Switchyard ATOF");
        assert_eq!(atof.diagnostic_details(), "atof ordering");
        assert!(atof.to_string().contains("ATOF validation failed"));

        let cache = PluginComponentSetupError::SwitchyardResponseCache("cache ordering".into());
        assert_eq!(cache.check_name(), "Switchyard response cache");
        assert_eq!(cache.diagnostic_details(), "cache ordering");
        assert!(
            cache
                .to_string()
                .contains("response-cache validation failed")
        );
    }
}

fn dynamic_component_without_manifest(
    plugin_id: &str,
    kind: DynamicPluginKind,
) -> ActiveDynamicPluginComponent {
    ActiveDynamicPluginComponent {
        plugin_id: plugin_id.into(),
        kind,
        lifecycle_generation: 1,
        manifest_ref: None,
        environment_ref: None,
        config: Map::new(),
        activation_snapshot: None,
    }
}

#[tokio::test]
async fn plugin_activation_covers_empty_invalid_and_missing_manifest_paths() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let inactive = PluginActivation::initialize(None, Vec::new())
        .await
        .unwrap();
    assert!(!inactive.active);
    inactive.clear().unwrap();

    let invalid = PluginActivation::initialize(
        Some(json!("not a plugin config")),
        vec![dynamic_component_without_manifest(
            "acme.invalid-config",
            DynamicPluginKind::Worker,
        )],
    )
    .await
    .err()
    .expect("invalid config should fail activation");
    assert!(invalid.to_string().contains("invalid plugin config"));

    let native = PluginActivation::initialize(
        None,
        vec![dynamic_component_without_manifest(
            "acme.native-missing",
            DynamicPluginKind::RustDynamic,
        )],
    )
    .await
    .err()
    .expect("native plugin without a manifest should fail activation");
    assert!(native.to_string().contains("native dynamic plugin"));

    let worker = PluginActivation::initialize(
        None,
        vec![dynamic_component_without_manifest(
            "acme.worker-missing",
            DynamicPluginKind::Worker,
        )],
    )
    .await
    .err()
    .expect("worker plugin without a manifest should fail activation");
    assert!(worker.to_string().contains("worker dynamic plugin"));
}

#[tokio::test]
async fn shutdown_future_helpers_cover_receiver_combinations() {
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let shutdown = server_shutdown_future(Some(ShutdownMode::Receiver(shutdown_rx)), None).unwrap();
    shutdown_tx.send(()).unwrap();
    shutdown.await;

    let (bootstrap_tx, bootstrap_rx) = oneshot::channel();
    let shutdown = combine_shutdown_futures(None, Some(bootstrap_rx)).unwrap();
    bootstrap_tx.send(()).unwrap();
    shutdown.await;

    let ready: ShutdownFuture = Box::pin(async {});
    combine_shutdown_futures(Some(ready), None).unwrap().await;
}

#[cfg(feature = "switchyard")]
#[test]
fn switchyard_must_run_before_response_cache() {
    let build = |switchyard_priority, cache_priority| {
        let switchyard = nemo_relay_switchyard::SwitchyardConfig {
            priority: switchyard_priority,
            ..nemo_relay_switchyard::SwitchyardConfig::default()
        };
        let adaptive = nemo_relay_adaptive::AdaptiveConfig {
            response_cache: Some(nemo_relay_adaptive::ResponseCacheConfig {
                namespace: "switchyard-order-test".into(),
                priority: cache_priority,
                ..nemo_relay_adaptive::ResponseCacheConfig::default()
            }),
            ..nemo_relay_adaptive::AdaptiveConfig::default()
        };
        PluginConfig {
            components: vec![
                switchyard.into(),
                nemo_relay_adaptive::plugin_component::ComponentSpec::new(adaptive).into(),
            ],
            ..PluginConfig::default()
        }
    };

    assert!(validate_switchyard_response_cache_order(&build(0, 50)).is_ok());
    for (switchyard_priority, cache_priority) in [(50, 50), (51, 50)] {
        let error =
            validate_switchyard_response_cache_order(&build(switchyard_priority, cache_priority))
                .unwrap_err();
        assert!(
            error.contains("must be lower"),
            "unexpected ordering error: {error}"
        );
    }

    let mut disabled = build(50, 50);
    disabled.components[0].enabled = false;
    assert!(
        validate_switchyard_response_cache_order(&disabled).is_ok(),
        "disabled Switchyard components do not participate in ordering"
    );
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
async fn serve_listener_activates_pii_redaction_plugin_config() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    let mut config = test_config();
    config.plugin_config = Some(json!({
        "version": 1,
        "components": [
            {
                "kind": "pii_redaction",
                "enabled": true,
                "config": {
                    "version": 1,
                    "mode": "builtin",
                    "codec": "openai_chat",
                    "input": true,
                    "output": true,
                    "builtin": {
                        "action": "redact",
                        "detector": "email"
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

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();
    assert!(nemo_relay::plugin::active_plugin_report().is_none());
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
                    "version": 3,
                    "atof": {
                        "enabled": true,
                        "sinks": [{
                            "type": "file",
                            "mode": "invalid"
                        }]
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

    assert!(error.to_string().contains("ATOF sinks[0].mode"));
    assert!(nemo_relay::plugin::active_plugin_report().is_none());
}

#[tokio::test]
async fn serve_listener_with_dynamic_reports_native_load_errors() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    let temp = tempfile::tempdir().unwrap();
    let manifest_ref = write_missing_native_plugin_manifest(temp.path(), "cli.missing-native");
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    drop(shutdown_tx);
    let error = serve_listener_with_dynamic(
        listener,
        test_config(),
        vec![ActiveDynamicPluginComponent {
            plugin_id: "cli.missing-native".into(),
            kind: DynamicPluginKind::RustDynamic,
            lifecycle_generation: 0,
            manifest_ref: Some(manifest_ref.to_string_lossy().into_owned()),
            environment_ref: None,
            config: Map::new(),
            activation_snapshot: None,
        }],
        Some(shutdown_rx),
    )
    .await
    .unwrap_err();

    let error = error.to_string();
    assert!(error.contains("native plugin load failed"), "{error}");
    assert!(error.contains("does not exist"), "{error}");
    assert!(nemo_relay::plugin::active_plugin_report().is_none());
}

#[tokio::test]
async fn serve_listener_rejects_invalid_pii_redaction_plugin_config() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    let mut config = test_config();
    config.plugin_config = Some(json!({
        "version": 1,
        "components": [
            {
                "kind": "pii_redaction",
                "enabled": true,
                "config": {
                    "version": 2,
                    "mode": "builtin",
                    "builtin": {
                        "action": "remove"
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

    assert!(error.to_string().contains("unsupported"));
    assert!(error.to_string().contains("version"));
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

    let response = CliError::PayloadTooLarge("too much".into()).into_response();

    assert_payload_too_large_response(response).await;
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
async fn pre_tool_hook_rejects_when_conditional_guardrail_blocks() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = deregister_tool_conditional_execution_guardrail("cli-pre-tool-blocker");
    const BLOCKED_TEST_TOOL: &str = "Nmf137BlockedTool";
    register_tool_conditional_execution_guardrail(
        "cli-pre-tool-blocker",
        1,
        Arc::new(|name, _args| {
            Box::pin(async move {
                Ok((name == BLOCKED_TEST_TOOL).then(|| "blocked by policy".to_string()))
            })
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
async fn transparent_gateway_requires_and_consumes_its_invocation_token() {
    let upstream = spawn_upstream(false).await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    let app = router_with_state(AppState::new_with_bootstrap(
        config,
        Some("transparent-fingerprint".into()),
        None,
        false,
        None,
        Some(
            crate::provider_auth::TransparentProxyCredential::from_static(
                "current-invocation-token",
            ),
        ),
    ));
    let body = || {
        Body::from(
            json!({
                "model": "gpt-test",
                "messages": [{ "role": "user", "content": "hello" }]
            })
            .to_string(),
        )
    };

    let missing = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(body())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);

    let foreign = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(
                    crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER,
                    "different-invocation-token",
                )
                .body(body())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(foreign.status(), StatusCode::UNAUTHORIZED);

    let accepted = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(
                    crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER,
                    "current-invocation-token",
                )
                .header("authorization", "Bearer upstream-provider-key")
                .body(body())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(accepted.status(), StatusCode::OK);
    let bytes = accepted.into_body().collect().await.unwrap().to_bytes();
    let payload: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        payload["authorization"],
        json!("Bearer upstream-provider-key")
    );
    assert_eq!(payload["transparent_proxy_token"], Value::Null);
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
async fn gateway_request_codec_exposes_annotations_and_applies_buffered_edits() {
    let intercept_name = "server-gateway-request-codec-buffered-edit";
    let _ = deregister_llm_request_intercept(intercept_name);
    let _cleanup = RequestInterceptCleanup(intercept_name);
    let observed = Arc::new(Mutex::new(None));
    let captured = observed.clone();
    register_llm_request_intercept(
        intercept_name,
        1,
        false,
        Arc::new(move |_name, mut request, annotated| {
            let captured = captured.clone();
            Box::pin(async move {
                if request.headers.get("x-codec-test").and_then(Value::as_str) != Some("buffered") {
                    return Ok(LlmRequestInterceptOutcome::new(request, annotated));
                }
                let mut annotated = annotated.expect("gateway generation route must have a codec");
                *captured.lock().unwrap() = Some(serde_json::to_value(&annotated).unwrap());
                let nemo_relay::codec::request::Message::User { content, .. } =
                    &mut annotated.messages[0]
                else {
                    panic!("expected portable Responses string input");
                };
                *content = nemo_relay::codec::request::MessageContent::Text("edited".into());
                request
                    .headers
                    .insert("x-test-intercept".into(), json!("visible"));
                Ok(LlmRequestInterceptOutcome::new(request, Some(annotated)))
            })
        }),
    )
    .unwrap();

    let upstream = spawn_upstream(false).await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    let response = router(config)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .header("x-codec-test", "buffered")
                .body(Body::from(
                    json!({"model": "gpt-test", "input": "original"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["input"], json!("edited"));
    assert_eq!(body["x_test_intercept"], json!("visible"));
    let annotation = observed.lock().unwrap().clone().unwrap();
    assert_eq!(annotation["messages"][0]["content"], json!("original"));
    assert_eq!(annotation["api_specific"]["api"], json!("openai_responses"));
}

#[tokio::test]
async fn gateway_request_codec_rejects_raw_body_mutation_before_upstream() {
    let intercept_name = "server-gateway-request-codec-raw-rejection";
    let _ = deregister_llm_request_intercept(intercept_name);
    let _cleanup = RequestInterceptCleanup(intercept_name);
    register_llm_request_intercept(
        intercept_name,
        1,
        false,
        Arc::new(|_name, mut request, annotated| {
            Box::pin(async move {
                if request.headers.get("x-codec-test").and_then(Value::as_str) == Some("raw") {
                    request.content["input"] = json!("forbidden raw edit");
                }
                Ok(LlmRequestInterceptOutcome::new(request, annotated))
            })
        }),
    )
    .unwrap();

    let upstream = spawn_upstream(false).await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    let response = router(config)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .header("x-codec-test", "raw")
                .body(Body::from(
                    json!({"model": "gpt-test", "input": "original"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn gateway_request_codec_rejects_malformed_modeled_structure_before_upstream() {
    let (upstream, captured_requests) = spawn_request_codec_matrix_upstream().await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    let response = router(config)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "gpt-test",
                        "messages": [],
                        "response_format": "json_object",
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(captured_requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn gateway_request_codec_rejects_stream_mode_changes_before_upstream() {
    let intercept_name = "server-gateway-request-codec-stream-mode-rejection";
    let _ = deregister_llm_request_intercept(intercept_name);
    let _cleanup = RequestInterceptCleanup(intercept_name);
    register_llm_request_intercept(
        intercept_name,
        1,
        false,
        Arc::new(|_name, request, annotated| {
            Box::pin(async move {
                if request
                    .headers
                    .get("x-codec-stream-toggle")
                    .and_then(Value::as_str)
                    == Some("true")
                {
                    let mut annotated =
                        annotated.expect("generation route must expose an annotation");
                    annotated.stream = Some(!annotated.stream.unwrap_or(false));
                    return Ok(LlmRequestInterceptOutcome::new(request, Some(annotated)));
                }
                Ok(LlmRequestInterceptOutcome::new(request, annotated))
            })
        }),
    )
    .unwrap();

    let (upstream, captured_requests) = spawn_request_codec_matrix_upstream().await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    let app = router(config);

    for streaming in [false, true] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
                    .header("content-type", "application/json")
                    .header("x-codec-stream-toggle", "true")
                    .body(Body::from(
                        json!({
                            "model": "gpt-test",
                            "input": "original",
                            "stream": streaming,
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
    assert!(captured_requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn gateway_request_codecs_apply_buffered_and_streaming_edits_on_all_generation_routes() {
    let intercept_name = "server-gateway-request-codec-all-generation-routes";
    let _ = deregister_llm_request_intercept(intercept_name);
    let _cleanup = RequestInterceptCleanup(intercept_name);
    let observed = Arc::new(Mutex::new(Vec::new()));
    let captured_annotations = observed.clone();
    register_llm_request_intercept(
        intercept_name,
        1,
        false,
        Arc::new(move |_name, mut request, annotated| {
            let captured_annotations = captured_annotations.clone();
            Box::pin(async move {
                let Some(marker) = request
                    .headers
                    .get("x-codec-matrix")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                else {
                    return Ok(LlmRequestInterceptOutcome::new(request, annotated));
                };
                let mut annotated = annotated.expect("generation route must expose an annotation");
                captured_annotations.lock().unwrap().push(json!({
                    "marker": marker,
                    "annotation": annotated,
                }));
                let nemo_relay::codec::request::Message::User { content, .. } =
                    &mut annotated.messages[0]
                else {
                    panic!("expected the first request item to be a portable user message");
                };
                *content =
                    nemo_relay::codec::request::MessageContent::Text(format!("edited-{marker}"));
                request
                    .headers
                    .insert("x-codec-edited".into(), json!(marker));
                Ok(LlmRequestInterceptOutcome::new(request, Some(annotated)))
            })
        }),
    )
    .unwrap();

    let (upstream, captured_requests) = spawn_request_codec_matrix_upstream().await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    config.anthropic_base_url = upstream.url();
    let app = router(config);

    for (surface, uri, mut payload) in [
        (
            "anthropic",
            "/v1/messages",
            json!({
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "original"}]
            }),
        ),
        (
            "chat",
            "/v1/chat/completions",
            json!({
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "original"}]
            }),
        ),
        (
            "responses",
            "/v1/responses",
            json!({"model": "gpt-5", "input": "original"}),
        ),
    ] {
        for streaming in [false, true] {
            payload["stream"] = json!(streaming);
            let marker = format!(
                "{surface}-{}",
                if streaming { "streaming" } else { "buffered" }
            );
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(uri)
                        .header("content-type", "application/json")
                        .header("x-codec-matrix", &marker)
                        .body(Body::from(payload.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{marker}");
            response.into_body().collect().await.unwrap();
        }
    }

    let captured = captured_requests.lock().unwrap();
    assert_eq!(captured.len(), 6);
    for request in captured.iter() {
        let marker = request["marker"].as_str().unwrap();
        assert_eq!(request["edited_header"], json!(marker));
        let edited = format!("edited-{marker}");
        match request["path"].as_str().unwrap() {
            "/v1/messages" | "/v1/chat/completions" => {
                assert_eq!(request["body"]["messages"][0]["content"], json!(edited));
            }
            "/v1/responses" => assert_eq!(request["body"]["input"], json!(edited)),
            path => panic!("unexpected captured path {path}"),
        }
    }

    let annotations = observed.lock().unwrap();
    assert_eq!(annotations.len(), 6);
    for observation in annotations.iter() {
        let marker = observation["marker"].as_str().unwrap();
        let expected_api = if marker.starts_with("anthropic-") {
            "anthropic_messages"
        } else if marker.starts_with("chat-") {
            "openai_chat"
        } else {
            "openai_responses"
        };
        assert_eq!(
            observation["annotation"]["api_specific"]["api"],
            json!(expected_api)
        );
    }
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
async fn gateway_preserves_streaming_provider_error_response() {
    async fn rate_limited() -> impl IntoResponse {
        (
            StatusCode::TOO_MANY_REQUESTS,
            [
                (header::CONTENT_TYPE, "application/json"),
                (header::RETRY_AFTER, "7"),
            ],
            r#"{"error":{"type":"rate_limit_error"}}"#,
        )
    }

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().route("/v1/responses", post(rate_limited)),
        )
        .await
        .unwrap();
    });
    let mut config = test_config();
    config.openai_base_url = format!("http://{address}");

    let response = router(config)
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

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/json"
    );
    assert_eq!(
        response.headers().get(header::RETRY_AFTER).unwrap(),
        "7",
        "the provider retry hint must reach the client"
    );
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        body.as_ref(),
        br#"{"error":{"type":"rate_limit_error"}}"#,
        "the provider error body must reach the client unchanged"
    );
    handle.abort();
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
async fn gateway_upstream_transport_error_url_is_opaque_in_events() {
    const SECRET_USERNAME: &str = "secret-upstream-user";
    const MODEL: &str = "gpt-opaque-transport-error-test";
    let subscriber_name = "server-gateway-opaque-transport-error-test";
    let _ = deregister_subscriber(subscriber_name);
    let captured_events = Arc::new(Mutex::new(Vec::<Value>::new()));
    let captured = captured_events.clone();
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::End)
                && event.name() == "openai.chat_completions"
                && event.model_name() == Some(MODEL)
            {
                captured.lock().unwrap().push(event.to_json_value());
            }
        }),
    )
    .unwrap();
    let _subscriber_cleanup = SubscriberCleanup(subscriber_name);

    let mut config = test_config();
    config.openai_base_url = format!("http://{SECRET_USERNAME}:password@127.0.0.1:1");
    let app = router(config);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": MODEL,
                        "messages": [{ "role": "user", "content": "hello" }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    flush_subscribers().unwrap();

    let events = captured_events.lock().unwrap();
    assert_eq!(events.len(), 1, "{events:?}");
    let event = &events[0];
    assert!(
        !event.to_string().contains(SECRET_USERNAME),
        "upstream credentials leaked into the event: {event}"
    );
    let description = event["metadata"]["otel.status_description"]
        .as_str()
        .expect("failed call should have an error status description");
    let token = description
        .strip_prefix("internal error: nemo-relay-gateway-upstream-attempt:")
        .expect("event status should contain only the captured failure token");
    assert!(uuid::Uuid::parse_str(token).is_ok(), "{description}");
}

#[tokio::test]
async fn passthrough_body_limit_returns_structured_413() {
    let upstream = spawn_upstream(false).await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    config.max_passthrough_body_bytes = 32;
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
                        "input": "x".repeat(1024)
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_payload_too_large_response(response).await;
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

#[tokio::test]
async fn gateway_suppresses_claude_startup_probe_without_native_session_header() {
    let subscriber_name = "server-claude-startup-probe-no-native-header-test";
    let _ = deregister_subscriber(subscriber_name);
    let captured_events = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
    let captured = captured_events.clone();
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| {
            let session_id = event
                .metadata()
                .and_then(|metadata| metadata.get("session_id"))
                .and_then(Value::as_str);
            let is_probe_llm = event.scope_category() == Some(ScopeCategory::Start)
                && event.name() == "anthropic.messages"
                && event
                    .input()
                    .and_then(|input| input.get("model"))
                    .and_then(Value::as_str)
                    == Some("claude-opus-4-8[1m]");
            let is_probe_turn = event.scope_category() == Some(ScopeCategory::Start)
                && event.name() == "claude-code-turn"
                && session_id == Some("claude-probe-no-native-header");
            if is_probe_llm || is_probe_turn {
                captured.lock().unwrap().push(json!({
                    "name": event.name(),
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
                .header("x-nemo-relay-session-id", "claude-probe-no-native-header")
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
    flush_subscribers().unwrap();
    assert!(
        captured_events.lock().unwrap().is_empty(),
        "no-native-header Claude startup probe must not emit managed LLM or null-input turn events"
    );
}

#[tokio::test]
async fn direct_claude_gateway_request_before_prompt_hook_uses_request_turn_input() {
    let subscriber_name = "server-claude-direct-gateway-turn-input-test";
    let _ = deregister_subscriber(subscriber_name);
    let captured_turn_starts = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
    let captured = captured_turn_starts.clone();
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::Start)
                && event.name() == "claude-code-turn"
                && event
                    .metadata()
                    .and_then(|metadata| metadata.get("session_id"))
                    .and_then(Value::as_str)
                    == Some("claude-direct-installed")
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
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .header("x-api-key", "sk-ant-test")
                .header("x-claude-code-session-id", "claude-direct-installed")
                .body(Body::from(
                    json!({
                        "model": "claude-sonnet-4-5",
                        "messages": [
                            {
                                "role": "user",
                                "content": "inspect direct installed mode"
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

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/hooks/claude-code")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "session_id": "claude-direct-installed",
                        "hook_event_name": "UserPromptSubmit",
                        "prompt": "inspect direct installed mode"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    flush_subscribers().unwrap();
    let starts = captured_turn_starts.lock().unwrap().clone();
    assert_eq!(
        starts.len(),
        1,
        "later UserPromptSubmit must not open a duplicate Claude turn: {starts:#?}"
    );
    assert_eq!(
        starts[0]["input"],
        json!({ "prompt": "inspect direct installed mode" })
    );
    assert_eq!(
        starts[0]["metadata"]["turn_source"],
        json!("gateway_request")
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
            "transparent_proxy_token": headers
                .get(crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER)
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

async fn spawn_request_codec_matrix_upstream() -> (TestServer, Arc<Mutex<Vec<Value>>>) {
    async fn provider(
        State(captured): State<Arc<Mutex<Vec<Value>>>>,
        request: Request<Body>,
    ) -> Response {
        let path = request.uri().path().to_string();
        let marker = request
            .headers()
            .get("x-codec-matrix")
            .and_then(|value| value.to_str().ok())
            .unwrap()
            .to_string();
        let edited_header = request
            .headers()
            .get("x-codec-edited")
            .and_then(|value| value.to_str().ok())
            .unwrap()
            .to_string();
        let body = request.into_body().collect().await.unwrap().to_bytes();
        let payload: Value = serde_json::from_slice(&body).unwrap();
        captured.lock().unwrap().push(json!({
            "path": path,
            "marker": marker,
            "edited_header": edited_header,
            "body": payload,
        }));

        if !payload["stream"].as_bool().unwrap_or(false) {
            return match path.as_str() {
                "/v1/messages" => Json(json!({
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1}
                }))
                .into_response(),
                "/v1/chat/completions" => Json(json!({
                    "id": "chatcmpl_1",
                    "model": "gpt-4.1",
                    "choices": [{
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop"
                    }]
                }))
                .into_response(),
                "/v1/responses" => Json(json!({
                    "id": "resp_1",
                    "model": "gpt-5",
                    "status": "completed",
                    "output": []
                }))
                .into_response(),
                _ => StatusCode::NOT_FOUND.into_response(),
            };
        }

        let events: &[&[u8]] = match path.as_str() {
            "/v1/messages" => &[
                b"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-20250514\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
                b"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}\n\n",
                b"data: {\"type\":\"message_stop\"}\n\n",
            ],
            "/v1/chat/completions" => &[
                b"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
                b"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
                b"data: [DONE]\n\n",
            ],
            "/v1/responses" => &[
                b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\"}}\n\n",
                b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\",\"status\":\"completed\",\"output\":[]}}\n\n",
            ],
            _ => return StatusCode::NOT_FOUND.into_response(),
        };
        let chunks = stream::iter(
            events
                .iter()
                .map(|event| Ok::<_, std::convert::Infallible>(Bytes::copy_from_slice(event))),
        );
        (
            [(header::CONTENT_TYPE, "text/event-stream")],
            Body::from_stream(chunks),
        )
            .into_response()
    }

    let captured = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/messages", post(provider))
        .route("/v1/chat/completions", post(provider))
        .route("/v1/responses", post(provider))
        .with_state(captured.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (
        TestServer {
            url: format!("http://{address}"),
            handle,
        },
        captured,
    )
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

/// Spawns a minimal mock upstream that counts calls and returns a fixed
/// (status, content-type, body) for every POST. Returns its base URL and the
/// call counter.
async fn spawn_mock_upstream(
    status: StatusCode,
    content_type: &'static str,
    body: &'static str,
) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    use axum::routing::any;
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = calls.clone();
    let app = axum::Router::new().route(
        "/{*path}",
        any(move || {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                axum::response::Response::builder()
                    .status(status)
                    .header(http::header::CONTENT_TYPE, content_type)
                    .header("x-upstream-marker", "mock")
                    .body(axum::body::Body::from(body))
                    .unwrap()
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{address}"), calls)
}

fn concurrent_selected_chat_response() -> Value {
    json!({
        "id": "chatcmpl-selected",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "selected attempt"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12
        }
    })
}

const CONCURRENT_SELECTED_CHAT_SSE: &str = concat!(
    "data: {\"id\":\"chatcmpl-selected\",\"object\":\"chat.completion.chunk\",",
    "\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,",
    "\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-selected\",\"object\":\"chat.completion.chunk\",",
    "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"selected attempt\"},",
    "\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-selected\",\"object\":\"chat.completion.chunk\",",
    "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    "data: [DONE]\n\n",
);

const CONCURRENT_LOSING_CHAT_SSE: &str =
    "data: {\"losing\":\"attempt metadata must not escape\"}\n\ndata: [DONE]\n\n";

async fn spawn_concurrent_attempt_upstream(release_losing: Arc<Semaphore>) -> TestServer {
    async fn provider(State(release_losing): State<Arc<Semaphore>>, body: Bytes) -> Response {
        let request: Value = serde_json::from_slice(&body).unwrap();
        let attempt = request["attempt"].as_str().unwrap();
        let streaming = request["stream"].as_bool().unwrap_or(false);
        let both_fail = request["both_fail"].as_bool().unwrap_or(false);
        if attempt == "losing" {
            release_losing.acquire().await.unwrap().forget();
        }

        let (status, content_type, body) = match (attempt, streaming, both_fail) {
            ("selected", false, true) => (
                StatusCode::TOO_MANY_REQUESTS,
                "application/vnd.selected-error+json",
                r#"{"error":"selected buffered failure"}"#.to_string(),
            ),
            ("losing", false, true) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "application/vnd.losing-error+json",
                r#"{"error":"losing buffered failure"}"#.to_string(),
            ),
            ("selected", true, true) => (
                StatusCode::TOO_MANY_REQUESTS,
                "application/vnd.selected-stream-error+json",
                r#"{"error":"selected streaming failure"}"#.to_string(),
            ),
            ("losing", true, true) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "application/vnd.losing-stream-error+json",
                r#"{"error":"losing streaming failure"}"#.to_string(),
            ),
            ("selected", false, false) => (
                StatusCode::CREATED,
                "application/vnd.selected+json",
                serde_json::to_string_pretty(&concurrent_selected_chat_response()).unwrap(),
            ),
            ("losing", false, false) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "application/x-losing-json",
                format!(" \n{}\n ", concurrent_selected_chat_response()),
            ),
            ("selected", true, false) => (
                StatusCode::CREATED,
                "application/vnd.selected.event-stream",
                CONCURRENT_SELECTED_CHAT_SSE.to_string(),
            ),
            ("losing", true, false) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "application/x-losing-event-stream",
                CONCURRENT_LOSING_CHAT_SSE.to_string(),
            ),
            _ => panic!("unexpected concurrent attempt {attempt:?}"),
        };

        Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, content_type)
            .header("x-upstream-attempt", attempt)
            .body(Body::from(body))
            .unwrap()
    }

    let app = Router::new()
        .route("/v1/chat/completions", post(provider))
        .with_state(release_losing);
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

#[tokio::test]
async fn gateway_concurrent_next_uses_canonical_selected_buffered_response() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    const INTERCEPT_NAME: &str = "cli-test-concurrent-buffered-next";
    const MARKER: &str = "select the successful concurrent buffered attempt";
    let _ = deregister_llm_execution_intercept(INTERCEPT_NAME);
    let _cleanup = LlmExecutionInterceptCleanup(INTERCEPT_NAME);
    let release_losing = Arc::new(Semaphore::new(0));
    let release_from_intercept = release_losing.clone();
    register_llm_execution_intercept(
        INTERCEPT_NAME,
        1,
        Arc::new(move |_name, request, next| {
            let release_losing = release_from_intercept.clone();
            Box::pin(async move {
                if request.content["messages"][0]["content"].as_str() != Some(MARKER) {
                    return next(request).await;
                }

                let mut selected_request = request.clone();
                selected_request.content["attempt"] = json!("selected");
                let mut losing_request = request;
                losing_request.content["attempt"] = json!("losing");
                let mut selected = next(selected_request);
                let mut losing = next(losing_request);
                let selected_result = tokio::select! {
                    biased;
                    _ = &mut losing => {
                        panic!("losing attempt completed before release");
                    }
                    selected_result = &mut selected => selected_result,
                };

                release_losing.add_permits(1);
                let losing_result = losing.await;
                assert!(
                    losing_result.is_err(),
                    "the deliberately failed losing attempt unexpectedly succeeded"
                );
                selected_result
            })
        }),
    )
    .unwrap();

    let upstream = spawn_concurrent_attempt_upstream(release_losing).await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    let response = router(config)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": MARKER}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    assert!(
        response.headers().get("x-upstream-attempt").is_none(),
        "managed success must not inherit metadata from an arbitrary upstream attempt"
    );
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        body.as_ref(),
        concurrent_selected_chat_response().to_string().as_bytes(),
        "managed success must serialize the selected final JSON, not losing raw bytes"
    );
}

#[tokio::test]
async fn gateway_concurrent_next_uses_canonical_selected_streaming_response() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    const INTERCEPT_NAME: &str = "cli-test-concurrent-streaming-next";
    const MARKER: &str = "select the successful concurrent streaming attempt";
    let _ = deregister_llm_stream_execution_intercept(INTERCEPT_NAME);
    let _cleanup = LlmStreamExecutionInterceptCleanup(INTERCEPT_NAME);
    let release_losing = Arc::new(Semaphore::new(0));
    let release_from_intercept = release_losing.clone();
    register_llm_stream_execution_intercept(
        INTERCEPT_NAME,
        1,
        Arc::new(move |_name, request, next| {
            let release_losing = release_from_intercept.clone();
            Box::pin(async move {
                if request.content["messages"][0]["content"].as_str() != Some(MARKER) {
                    return next(request).await;
                }

                let mut selected_request = request.clone();
                selected_request.content["attempt"] = json!("selected");
                let mut losing_request = request;
                losing_request.content["attempt"] = json!("losing");
                let mut selected = next(selected_request);
                let mut losing = next(losing_request);
                let selected_result = tokio::select! {
                    biased;
                    _ = &mut losing => {
                        panic!("losing attempt completed before release");
                    }
                    selected_result = &mut selected => selected_result,
                };

                release_losing.add_permits(1);
                drop(losing.await);
                selected_result
            })
        }),
    )
    .unwrap();

    let upstream = spawn_concurrent_attempt_upstream(release_losing).await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    let response = router(config)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": MARKER}],
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
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    assert!(
        response.headers().get("x-upstream-attempt").is_none(),
        "managed stream success must not inherit metadata from an arbitrary upstream attempt"
    );
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let body = std::str::from_utf8(&body).unwrap();
    assert!(body.contains("selected attempt"), "{body}");
    assert!(!body.contains("losing attempt"), "{body}");
}

#[tokio::test]
async fn gateway_concurrent_next_relays_selected_buffered_failure() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    const INTERCEPT_NAME: &str = "cli-test-concurrent-buffered-failures";
    const MARKER: &str = "select one concurrent buffered failure";
    let _ = deregister_llm_execution_intercept(INTERCEPT_NAME);
    let _cleanup = LlmExecutionInterceptCleanup(INTERCEPT_NAME);
    let release_losing = Arc::new(Semaphore::new(0));
    let release_from_intercept = release_losing.clone();
    register_llm_execution_intercept(
        INTERCEPT_NAME,
        1,
        Arc::new(move |_name, request, next| {
            let release_losing = release_from_intercept.clone();
            Box::pin(async move {
                if request.content["messages"][0]["content"].as_str() != Some(MARKER) {
                    return next(request).await;
                }

                let mut selected_request = request.clone();
                selected_request.content["attempt"] = json!("selected");
                selected_request.content["both_fail"] = json!(true);
                let mut losing_request = request;
                losing_request.content["attempt"] = json!("losing");
                losing_request.content["both_fail"] = json!(true);
                let mut selected = next(selected_request);
                let mut losing = next(losing_request);
                let selected_result = tokio::select! {
                    biased;
                    _ = &mut losing => {
                        panic!("losing attempt completed before release");
                    }
                    selected_result = &mut selected => selected_result,
                };

                release_losing.add_permits(1);
                assert!(losing.await.is_err());
                selected_result
            })
        }),
    )
    .unwrap();

    let upstream = spawn_concurrent_attempt_upstream(release_losing).await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    let response = router(config)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": MARKER}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/vnd.selected-error+json"
    );
    assert_eq!(
        response.headers().get("x-upstream-attempt").unwrap(),
        "selected"
    );
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body.as_ref(), br#"{"error":"selected buffered failure"}"#);
}

#[tokio::test]
async fn gateway_concurrent_next_relays_selected_streaming_failure() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    const INTERCEPT_NAME: &str = "cli-test-concurrent-streaming-failures";
    const MARKER: &str = "select one concurrent streaming failure";
    let _ = deregister_llm_stream_execution_intercept(INTERCEPT_NAME);
    let _cleanup = LlmStreamExecutionInterceptCleanup(INTERCEPT_NAME);
    let release_losing = Arc::new(Semaphore::new(0));
    let release_from_intercept = release_losing.clone();
    register_llm_stream_execution_intercept(
        INTERCEPT_NAME,
        1,
        Arc::new(move |_name, request, next| {
            let release_losing = release_from_intercept.clone();
            Box::pin(async move {
                if request.content["messages"][0]["content"].as_str() != Some(MARKER) {
                    return next(request).await;
                }

                let mut selected_request = request.clone();
                selected_request.content["attempt"] = json!("selected");
                selected_request.content["both_fail"] = json!(true);
                let mut losing_request = request;
                losing_request.content["attempt"] = json!("losing");
                losing_request.content["both_fail"] = json!(true);
                let mut selected = next(selected_request);
                let mut losing = next(losing_request);
                let selected_result = tokio::select! {
                    biased;
                    _ = &mut losing => {
                        panic!("losing attempt completed before release");
                    }
                    selected_result = &mut selected => selected_result,
                };

                release_losing.add_permits(1);
                assert!(losing.await.is_err());
                selected_result
            })
        }),
    )
    .unwrap();

    let upstream = spawn_concurrent_attempt_upstream(release_losing).await;
    let mut config = test_config();
    config.openai_base_url = upstream.url();
    let response = router(config)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": MARKER}],
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/vnd.selected-stream-error+json"
    );
    assert_eq!(
        response.headers().get("x-upstream-attempt").unwrap(),
        "selected"
    );
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body.as_ref(), br#"{"error":"selected streaming failure"}"#);
}

/// Gateway config wired to `upstream` with the response cache enabled.
fn cache_gateway_config(upstream: &str) -> GatewayConfig {
    let mut config = test_config();
    config.openai_base_url = upstream.into();
    config.anthropic_base_url = upstream.into();
    config.plugin_config = Some(json!({
        "version": 1,
        "components": [{
            "kind": "adaptive",
            "enabled": true,
            "config": {"response_cache": {
                "ttl_seconds": 3600,
                "bypass_rate": 0.0,
                "namespace": "gateway-test",
                "backend": {"kind": "in_memory"}
            }}
        }]
    }));
    config
}

const MOCK_CHAT_BODY: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"The answer is 42."},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}"#;

#[tokio::test]
async fn gateway_serves_cached_hit_with_full_body_and_json_content_type() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    let (upstream, upstream_calls) =
        spawn_mock_upstream(StatusCode::OK, "application/json", MOCK_CHAT_BODY).await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let config = cache_gateway_config(&upstream);
    let handle =
        tokio::spawn(async move { serve_listener(listener, config, Some(shutdown_rx)).await });
    wait_for_gateway(&url).await;

    let client = test_http_client();
    let request_body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is the answer?"}],
        "temperature": 0.0
    });
    let first = client
        .post(format!("{url}/v1/chat/completions"))
        .json(&request_body)
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    let first_body: Value = first.json().await.unwrap();

    let second = client
        .post(format!("{url}/v1/chat/completions"))
        .json(&request_body)
        .send()
        .await
        .unwrap();
    assert_eq!(second.status(), StatusCode::OK);
    assert_eq!(
        second
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json"),
        "a cached hit must still declare its JSON content type"
    );
    let second_body: Value = second.json().await.unwrap();

    assert_eq!(
        upstream_calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the repeat must be served from cache, not the upstream"
    );
    assert_eq!(
        second_body, first_body,
        "the cached body must be byte-equivalent to the live one"
    );
    assert_eq!(
        second_body["choices"][0]["message"]["content"],
        json!("The answer is 42.")
    );

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();
    let _ = nemo_relay::plugin::clear_plugin_configuration();
}

#[tokio::test]
async fn gateway_preserves_non_2xx_upstream_failure_and_never_caches_it() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    // A 503 whose body has NO top-level `error` key — the shape the
    // response-body error check alone would not catch; only the status gate
    // keeps it out of the cache.
    let (upstream, upstream_calls) = spawn_mock_upstream(
        StatusCode::SERVICE_UNAVAILABLE,
        "application/json",
        r#"{"message":"service temporarily unavailable"}"#,
    )
    .await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let config = cache_gateway_config(&upstream);
    let handle =
        tokio::spawn(async move { serve_listener(listener, config, Some(shutdown_rx)).await });
    wait_for_gateway(&url).await;

    let client = test_http_client();
    let request_body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}]
    });
    for _ in 0..2 {
        let response = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&request_body)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "the typed provider failure must preserve the upstream status"
        );
        assert_eq!(
            response
                .headers()
                .get("x-upstream-marker")
                .and_then(|value| value.to_str().ok()),
            Some("mock"),
            "ordinary upstream failure headers must reach the client"
        );
        assert_eq!(
            response.text().await.unwrap(),
            r#"{"message":"service temporarily unavailable"}"#,
            "ordinary upstream failure bodies must reach the client unchanged"
        );
    }
    assert_eq!(
        upstream_calls.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "an upstream failure must never be cached"
    );

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();
    let _ = nemo_relay::plugin::clear_plugin_configuration();
}

#[tokio::test]
async fn gateway_preserves_2xx_non_json_and_never_caches_it() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    let (upstream, upstream_calls) = spawn_mock_upstream(
        StatusCode::OK,
        "text/plain",
        "provider returned malformed JSON",
    )
    .await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let config = cache_gateway_config(&upstream);
    let handle =
        tokio::spawn(async move { serve_listener(listener, config, Some(shutdown_rx)).await });
    wait_for_gateway(&url).await;

    let client = test_http_client();
    let request_body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.0
    });
    for _ in 0..2 {
        let response = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&request_body)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "an ordinary upstream response keeps its provider status"
        );
        assert_eq!(
            response
                .headers()
                .get(http::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("text/plain"),
            "an ordinary upstream response keeps its content type"
        );
        assert_eq!(
            response
                .headers()
                .get("x-upstream-marker")
                .and_then(|value| value.to_str().ok()),
            Some("mock"),
            "an ordinary upstream response keeps its provider headers"
        );
        assert_eq!(
            response.text().await.unwrap(),
            "provider returned malformed JSON",
            "an undecodable provider body must reach the client unchanged"
        );
    }
    assert_eq!(
        upstream_calls.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "an unparseable upstream response must never be cached"
    );

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();
    let _ = nemo_relay::plugin::clear_plugin_configuration();
}

#[tokio::test]
async fn gateway_surfaces_post_upstream_intercept_rejection_instead_of_relaying_body() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    const INTERCEPT_NAME: &str = "cli-test-post-upstream-reject";
    const MARKER: &str = "cli-test reject the response after upstream success";
    let _ = deregister_llm_execution_intercept(INTERCEPT_NAME);
    register_llm_execution_intercept(
        INTERCEPT_NAME,
        1,
        Arc::new(|_name, request, next| {
            Box::pin(async move {
                let marked = request.content["messages"][0]["content"].as_str() == Some(MARKER);
                let response = next(request).await?;
                if marked {
                    return Err(nemo_relay::error::FlowError::Internal(
                        "response rejected by policy".to_string(),
                    ));
                }
                Ok(response)
            })
        }),
    )
    .unwrap();
    let _cleanup = LlmExecutionInterceptCleanup(INTERCEPT_NAME);

    let (upstream, upstream_calls) =
        spawn_mock_upstream(StatusCode::OK, "application/json", MOCK_CHAT_BODY).await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let config = cache_gateway_config(&upstream);
    let handle =
        tokio::spawn(async move { serve_listener(listener, config, Some(shutdown_rx)).await });
    wait_for_gateway(&url).await;

    let client = test_http_client();
    let response = client
        .post(format!("{url}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": MARKER}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        upstream_calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the intercept must reject only after a completed upstream exchange"
    );
    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "a rejection after upstream success must surface as the translated \
         runtime error, not a relay of the upstream body"
    );
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["type"], json!("nemo_relay_gateway_error"));

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();
    let _ = nemo_relay::plugin::clear_plugin_configuration();
}

const MOCK_CHAT_SSE: &str = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"The answer is 42.\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

#[tokio::test]
async fn gateway_streaming_hit_carries_event_stream_content_type() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _ = nemo_relay::plugin::clear_plugin_configuration();

    let (upstream, upstream_calls) =
        spawn_mock_upstream(StatusCode::OK, "text/event-stream", MOCK_CHAT_SSE).await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let config = cache_gateway_config(&upstream);
    let handle =
        tokio::spawn(async move { serve_listener(listener, config, Some(shutdown_rx)).await });
    wait_for_gateway(&url).await;

    let client = test_http_client();
    let request_body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "stream it"}],
        "temperature": 0.0,
        "stream": true
    });
    let first = client
        .post(format!("{url}/v1/chat/completions"))
        .json(&request_body)
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    let _ = first.text().await.unwrap(); // drain so the tee stores the aggregate

    let second = client
        .post(format!("{url}/v1/chat/completions"))
        .json(&request_body)
        .send()
        .await
        .unwrap();
    assert_eq!(second.status(), StatusCode::OK);
    assert_eq!(
        second
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream"),
        "a replayed stream must declare the SSE content type"
    );
    let replayed = second.text().await.unwrap();
    assert!(
        replayed.contains("The answer is 42."),
        "the replay must carry the stored answer: {replayed}"
    );
    assert!(
        replayed.trim_end().ends_with("data: [DONE]"),
        "a chat replay must terminate the SSE stream: {replayed}"
    );
    assert_eq!(
        upstream_calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the streaming repeat must be served from cache"
    );

    shutdown_tx.send(()).unwrap();
    handle.await.unwrap().unwrap();
    let _ = nemo_relay::plugin::clear_plugin_configuration();
}
