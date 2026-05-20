// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::config::GatewayConfig;
use crate::server::AppState;
use crate::session::SessionManager;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Method, Request, StatusCode, header};
use http_body_util::BodyExt;
use reqwest::Client;

fn test_http_client() -> Client {
    crate::tls::install_rustls_crypto_provider();
    Client::new()
}

#[test]
fn removes_hop_by_hop_headers() {
    assert!(!should_forward_request_header(&HeaderName::from_static(
        "connection"
    )));
    assert!(!should_forward_request_header(&HeaderName::from_static(
        "host"
    )));
    assert!(should_forward_request_header(&HeaderName::from_static(
        "authorization"
    )));
    assert!(!should_record_header(&HeaderName::from_static(
        "authorization"
    )));
    assert!(!should_record_header(&HeaderName::from_static("x-api-key")));
    assert!(!should_record_header(&HeaderName::from_static(
        "anthropic-api-key"
    )));
    // Additional credential aliases must not appear in observability metadata:
    // `cookie` carries session credentials; `api-key` is the generic alias used by some providers
    // (e.g., Azure OpenAI). Without these, secrets would leak into `LlmRequest.headers` and any
    // downstream exporter that mirrors them (ATIF, OpenInference span attributes).
    assert!(!should_record_header(&HeaderName::from_static("cookie")));
    assert!(!should_record_header(&HeaderName::from_static("api-key")));
    assert!(should_record_header(&HeaderName::from_static(
        "x-request-id"
    )));
}

#[test]
fn selects_provider_routes() {
    assert_eq!(
        ProviderRoute::from_path("/responses"),
        Some(ProviderRoute::OpenAiResponses)
    );
    assert_eq!(
        ProviderRoute::from_path("/v1/responses"),
        Some(ProviderRoute::OpenAiResponses)
    );
    assert_eq!(
        ProviderRoute::from_path("/v1/messages/count_tokens"),
        Some(ProviderRoute::AnthropicCountTokens)
    );
    assert_eq!(
        ProviderRoute::from_path("/v1/chat/completions")
            .unwrap()
            .name(),
        "openai.chat_completions"
    );
    assert_eq!(
        ProviderRoute::from_path("/models"),
        Some(ProviderRoute::OpenAiModels)
    );
    assert_eq!(ProviderRoute::OpenAiModels.name(), "openai.models");
    assert_eq!(
        ProviderRoute::AnthropicMessages.name(),
        "anthropic.messages"
    );
    assert_eq!(
        ProviderRoute::AnthropicCountTokens.name(),
        "anthropic.count_tokens"
    );
    assert_eq!(ProviderRoute::from_path("/unsupported"), None);
}

#[test]
fn provider_routes_preserve_path_query_and_choose_upstream() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai/v1/".into(),

        anthropic_base_url: "http://anthropic/".into(),
        metadata: None,
        plugin_config: None,
    };

    assert_eq!(
        ProviderRoute::OpenAiResponses.upstream_url(&config, "/v1/responses?x=1"),
        "http://openai/v1/responses?x=1"
    );
    assert_eq!(
        ProviderRoute::OpenAiResponses.upstream_url(&config, "/responses?x=1"),
        "http://openai/v1/responses?x=1"
    );
    assert_eq!(
        ProviderRoute::OpenAiModels.upstream_url(&config, "/models"),
        "http://openai/v1/models"
    );
    assert_eq!(
        ProviderRoute::AnthropicMessages.upstream_url(&config, "/v1/messages"),
        "http://anthropic/v1/messages"
    );
}

#[test]
fn openai_upstream_url_accepts_origin_or_v1_base() {
    let mut config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai".into(),
        anthropic_base_url: "http://anthropic".into(),
        metadata: None,
        plugin_config: None,
    };

    assert_eq!(
        ProviderRoute::OpenAiResponses.upstream_url(&config, "/responses"),
        "http://openai/v1/responses"
    );
    assert_eq!(
        ProviderRoute::OpenAiResponses.upstream_url(&config, "/v1/responses"),
        "http://openai/v1/responses"
    );

    config.openai_base_url = "http://openai/v1".into();
    assert_eq!(
        ProviderRoute::OpenAiResponses.upstream_url(&config, "/responses"),
        "http://openai/v1/responses"
    );
    assert_eq!(
        ProviderRoute::OpenAiResponses.upstream_url(&config, "/v1/responses"),
        "http://openai/v1/responses"
    );
}

#[test]
fn effective_upstream_request_overlays_runtime_body_and_headers() {
    let original_body = Bytes::from_static(br#"{"model":"original"}"#);
    let mut original_headers = HeaderMap::new();
    original_headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer original"),
    );
    let request = LlmRequest {
        headers: Map::from_iter([
            ("x-runtime".to_string(), json!("enabled")),
            ("x-runtime-json".to_string(), json!({ "enabled": true })),
        ]),
        content: json!({
            "model": "rewritten",
            "nvext": { "agent_hints": { "priority": 1 } }
        }),
    };

    let (body, headers) =
        effective_upstream_request(&original_body, &original_headers, Some(&request));
    let body: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(body["model"], json!("rewritten"));
    assert_eq!(body["nvext"]["agent_hints"]["priority"], json!(1));
    assert_eq!(
        headers.get(header::AUTHORIZATION).unwrap(),
        "Bearer original"
    );
    assert_eq!(headers.get("x-runtime").unwrap(), "enabled");
    assert_eq!(
        headers.get("x-runtime-json").unwrap(),
        r#"{"enabled":true}"#
    );
}

#[test]
fn effective_upstream_request_returns_original_without_runtime_request() {
    let original_body = Bytes::from_static(br#"{"model":"original"}"#);
    let mut original_headers = HeaderMap::new();
    original_headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer original"),
    );
    original_headers.insert("x-request-id", HeaderValue::from_static("request-1"));

    let (body, headers) = effective_upstream_request(&original_body, &original_headers, None);

    assert_eq!(body, original_body);
    assert_eq!(
        headers.get(header::AUTHORIZATION).unwrap(),
        "Bearer original"
    );
    assert_eq!(headers.get("x-request-id").unwrap(), "request-1");
}

#[test]
fn effective_upstream_request_preserves_original_body_for_null_runtime_content() {
    let original_body = Bytes::from_static(b"not-json-but-still-upstream-body");
    let mut original_headers = HeaderMap::new();
    original_headers.insert("x-original", HeaderValue::from_static("kept"));
    let request = LlmRequest {
        headers: Map::from_iter([("x-runtime".to_string(), json!("enabled"))]),
        content: Value::Null,
    };

    let (body, headers) =
        effective_upstream_request(&original_body, &original_headers, Some(&request));

    assert_eq!(body, original_body);
    assert_eq!(headers.get("x-original").unwrap(), "kept");
    assert_eq!(headers.get("x-runtime").unwrap(), "enabled");
}

#[test]
fn effective_upstream_request_skips_invalid_runtime_headers() {
    let original_body = Bytes::from_static(br#"{"model":"original"}"#);
    let mut original_headers = HeaderMap::new();
    original_headers.insert("x-original", HeaderValue::from_static("kept"));
    let request = LlmRequest {
        headers: Map::from_iter([
            ("bad header".to_string(), json!("skip")),
            ("x-invalid-value".to_string(), json!("line\nbreak")),
            ("x-good".to_string(), json!("ok")),
        ]),
        content: json!({ "model": "rewritten" }),
    };

    let (body, headers) =
        effective_upstream_request(&original_body, &original_headers, Some(&request));
    let body: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(body["model"], json!("rewritten"));
    assert_eq!(headers.get("x-original").unwrap(), "kept");
    assert_eq!(headers.get("x-good").unwrap(), "ok");
    assert!(headers.get("x-invalid-value").is_none());
    assert!(headers.keys().all(|name| name.as_str() != "bad header"));
}

#[test]
fn gateway_session_id_prefers_headers_and_has_fallbacks() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "anthropic-beta",
        HeaderValue::from_static("prompt-caching-2024-07-31"),
    );
    assert_eq!(gateway_session_id(&headers), None);

    headers.insert(
        "x-claude-code-session-id",
        HeaderValue::from_static("claude-session"),
    );
    assert_eq!(
        gateway_session_id(&headers).as_deref(),
        Some("claude-session")
    );

    headers.insert(
        "x-nemo-flow-session-id",
        HeaderValue::from_static("explicit-session"),
    );
    assert_eq!(
        gateway_session_id(&headers).as_deref(),
        Some("explicit-session")
    );

    assert_eq!(gateway_session_id(&HeaderMap::new()), None);
}

#[test]
fn gateway_identifiers_accept_headers_and_scalar_body_values() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-nemo-flow-request-id",
        HeaderValue::from_static("req-header"),
    );
    let body = json!({
        "conversation": { "id": 42 },
        "generation": { "id": true },
        "request": { "id": "req-body" },
        "object": { "id": { "nested": true } }
    });

    assert_eq!(
        gateway_identifier(
            &headers,
            &body,
            "x-nemo-flow-request-id",
            &[&["request", "id"]]
        )
        .as_deref(),
        Some("req-header")
    );
    assert_eq!(
        gateway_identifier(
            &HeaderMap::new(),
            &body,
            "missing",
            &[&["conversation", "id"]]
        )
        .as_deref(),
        Some("42")
    );
    assert_eq!(
        gateway_identifier(
            &HeaderMap::new(),
            &body,
            "missing",
            &[&["generation", "id"]]
        )
        .as_deref(),
        Some("true")
    );
    assert_eq!(
        gateway_identifier(&HeaderMap::new(), &body, "missing", &[&["object", "id"]]),
        None
    );
}

#[test]
fn observable_headers_omit_secrets_and_transport_headers() {
    let mut headers = HeaderMap::new();
    headers.insert("authorization", HeaderValue::from_static("Bearer secret"));
    headers.insert("x-api-key", HeaderValue::from_static("secret"));
    headers.insert("connection", HeaderValue::from_static("close"));
    headers.insert("x-request-id", HeaderValue::from_static("req-1"));

    let observed = observable_headers(&headers);

    assert_eq!(observed.get("x-request-id"), Some(&json!("req-1")));
    assert!(!observed.contains_key("authorization"));
    assert!(!observed.contains_key("x-api-key"));
    assert!(!observed.contains_key("connection"));
}

#[test]
fn strips_chatgpt_plus_jwt_from_openai_route_inbound() {
    // When OPENAI_API_KEY is set the gateway strips JWT-shaped (`Bearer eyJ...`) Authorization
    // from inbound OpenAI-route requests so the auth-injection path substitutes the env key
    // instead of forwarding the ChatGPT-Plus OAuth JWT.
    let mut inbound = HeaderMap::new();
    inbound.insert(
        "authorization",
        HeaderValue::from_static("Bearer eyJhbGciOiJIUzI1NiJ9.deadbeef.signature"),
    );
    let sanitized =
        strip_chatgpt_oauth_for_openai_route(&inbound, ProviderRoute::OpenAiResponses, true);
    assert!(sanitized.get("authorization").is_none());
}

#[test]
fn preserves_real_bearer_keys_on_openai_route() {
    // Real provider keys (Hermes's `sk-...` against NVIDIA, an actual OpenAI dev key, etc.)
    // must pass through untouched — only the consumer JWT shape (`Bearer eyJ...`) is stripped.
    let mut inbound = HeaderMap::new();
    inbound.insert(
        "authorization",
        HeaderValue::from_static("Bearer sk-real-provider-key"),
    );
    let sanitized =
        strip_chatgpt_oauth_for_openai_route(&inbound, ProviderRoute::OpenAiResponses, true);
    assert_eq!(
        sanitized.get("authorization").unwrap(),
        "Bearer sk-real-provider-key"
    );
}

#[test]
fn does_not_touch_anthropic_route_authorization() {
    // Defensive — the JWT shape only conflicts with OpenAI routes; Anthropic routes use
    // `x-api-key` anyway. Leaving Anthropic's Authorization alone avoids any cross-provider
    // edge cases.
    let mut inbound = HeaderMap::new();
    inbound.insert(
        "authorization",
        HeaderValue::from_static("Bearer eyJ.anthropic.case"),
    );
    let sanitized =
        strip_chatgpt_oauth_for_openai_route(&inbound, ProviderRoute::AnthropicMessages, true);
    assert!(sanitized.get("authorization").is_some());
}

#[test]
fn preserves_jwt_when_no_replacement_key_available() {
    // If OPENAI_API_KEY isn't set the gateway has nothing to inject after stripping, so leave
    // the inbound bearer in place. Stripping would silently de-auth setups that point at an
    // upstream which happens to accept the ChatGPT-Plus token.
    let mut inbound = HeaderMap::new();
    inbound.insert(
        "authorization",
        HeaderValue::from_static("Bearer eyJhbGciOiJIUzI1NiJ9.deadbeef.signature"),
    );
    let sanitized =
        strip_chatgpt_oauth_for_openai_route(&inbound, ProviderRoute::OpenAiResponses, false);
    assert!(sanitized.get("authorization").is_some());
}

#[test]
fn injects_openai_bearer_when_inbound_has_no_auth() {
    // NMF-86 mitigation: codex now sends no credentials, so the gateway must inject
    // `Authorization: Bearer ${OPENAI_API_KEY}` on outbound forwards to api.openai.com.
    let http = test_http_client();
    let inbound = HeaderMap::new();
    let env = |k: &str| match k {
        "OPENAI_API_KEY" => Some("sk-test-123".into()),
        _ => None,
    };
    let builder = http.get("http://upstream/v1/responses");
    let built =
        inject_provider_auth_with_env(builder, ProviderRoute::OpenAiResponses, &inbound, env)
            .build()
            .unwrap();
    assert_eq!(
        built.headers().get("authorization").unwrap(),
        "Bearer sk-test-123"
    );
}

#[test]
fn injects_anthropic_x_api_key_for_anthropic_routes() {
    let http = test_http_client();
    let inbound = HeaderMap::new();
    let env = |k: &str| match k {
        "ANTHROPIC_API_KEY" => Some("sk-ant-test".into()),
        _ => None,
    };
    let builder = http.post("http://upstream/v1/messages");
    let built =
        inject_provider_auth_with_env(builder, ProviderRoute::AnthropicMessages, &inbound, env)
            .build()
            .unwrap();
    assert_eq!(built.headers().get("x-api-key").unwrap(), "sk-ant-test");
    // Anthropic uses `x-api-key`, not Authorization. The gateway must not duplicate the secret
    // into a Bearer header — that would defeat the purpose of using the provider's standard
    // auth scheme and might trigger upstream-side rejection of the conflicting auth.
    assert!(built.headers().get("authorization").is_none());
}

#[test]
fn skips_injection_when_inbound_already_has_authorization() {
    // If the agent (e.g., a future codex version, or anyone using the gateway directly) sends
    // its own auth, we must not stomp on it.
    let http = test_http_client();
    let mut inbound = HeaderMap::new();
    inbound.insert(
        "authorization",
        HeaderValue::from_static("Bearer agent-supplied"),
    );
    let env = |_: &str| Some("sk-test-from-env".into());
    let builder = http.post("http://upstream/v1/responses");
    let built =
        inject_provider_auth_with_env(builder, ProviderRoute::OpenAiResponses, &inbound, env)
            .build()
            .unwrap();
    // The builder doesn't carry inbound headers itself (forward_upstream_request adds them in a
    // separate loop), so the only header on `built` would be the env-injected one. Since the
    // inbound had auth, we expect no injection at all.
    assert!(built.headers().get("authorization").is_none());
}

#[test]
fn skips_injection_when_env_var_unset() {
    let http = test_http_client();
    let inbound = HeaderMap::new();
    let env = |_: &str| None;
    let builder = http.post("http://upstream/v1/responses");
    let built =
        inject_provider_auth_with_env(builder, ProviderRoute::OpenAiResponses, &inbound, env)
            .build()
            .unwrap();
    assert!(built.headers().get("authorization").is_none());
}

// --- ChatGPT backend routing tests ---

#[test]
fn chatgpt_jwt_routes_to_chatgpt_backend_when_no_api_key() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer eyJhbGciOiJIUzI1NiJ9.deadbeef.signature"),
    );
    // With no OPENAI_API_KEY and a JWT, should_use_chatgpt_backend returns true and the URL is
    // built against the ChatGPT backend (no /v1 prefix — ChatGPT backend doesn't use it).
    let result = chatgpt_upstream_url("/responses");
    assert_eq!(result, "https://chatgpt.com/backend-api/codex/responses");

    // has_chatgpt_jwt should detect the JWT
    assert!(has_chatgpt_jwt(&headers));
    assert!(ProviderRoute::OpenAiResponses.is_openai());
}

#[test]
fn no_jwt_does_not_trigger_chatgpt_backend() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer sk-real-api-key"),
    );
    assert!(!has_chatgpt_jwt(&headers));

    // Empty headers also should not trigger
    assert!(!has_chatgpt_jwt(&HeaderMap::new()));
}

#[test]
fn anthropic_route_never_triggers_chatgpt_backend() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer eyJhbGciOiJIUzI1NiJ9.deadbeef.signature"),
    );
    assert!(!ProviderRoute::AnthropicMessages.is_openai());
}

#[test]
fn chatgpt_backend_url_omits_v1_prefix() {
    // The ChatGPT backend expects paths directly under the base, not /v1-prefixed.
    assert_eq!(
        chatgpt_upstream_url("/responses"),
        "https://chatgpt.com/backend-api/codex/responses"
    );
    assert_eq!(
        chatgpt_upstream_url("/models"),
        "https://chatgpt.com/backend-api/codex/models"
    );
    // /v1-prefixed inbound paths are stripped
    assert_eq!(
        chatgpt_upstream_url("/v1/responses"),
        "https://chatgpt.com/backend-api/codex/responses"
    );
}

#[tokio::test]
async fn passthrough_rejects_unsupported_provider_path_directly() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai".into(),

        anthropic_base_url: "http://anthropic".into(),
        metadata: None,
        plugin_config: None,
    };
    let state = AppState {
        config: config.clone(),
        http: test_http_client(),
        sessions: SessionManager::new(config),
    };
    let request = Request::builder()
        .method(Method::POST)
        .uri("/unsupported")
        .body(Body::empty())
        .unwrap();

    let error = passthrough(State(state), request).await.unwrap_err();

    assert!(error.to_string().contains("unsupported gateway path"));
}

#[tokio::test]
async fn models_rejects_non_get_requests_directly() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai".into(),

        anthropic_base_url: "http://anthropic".into(),
        metadata: None,
        plugin_config: None,
    };
    let state = AppState {
        config: config.clone(),
        http: test_http_client(),
        sessions: SessionManager::new(config),
    };
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();

    let response = models(State(state), request).await.unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert!(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .is_empty()
    );
}

#[test]
fn response_headers_preserve_duplicates() {
    let mut headers = HeaderMap::new();
    headers.append("set-cookie", HeaderValue::from_static("a=1"));
    headers.append("set-cookie", HeaderValue::from_static("b=2"));

    let copied = response_headers(&headers);

    assert_eq!(copied.get_all("set-cookie").iter().count(), 2);
}

// `stream_response_records_preview_and_truncation` and `streaming_llm_guard_closes_on_drop` were
// removed when the gateway moved to `llm_stream_call_execute`. The runtime now owns stream-end
// lifecycle (start/end events emitted by `LlmStreamWrapper`); core tests cover that contract,
// and the gateway no longer carries a stream preview/truncation helper or a separate guard struct.
