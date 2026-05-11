// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::config::SidecarConfig;
use crate::server::AppState;
use crate::session::SessionManager;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Method, Request, StatusCode};
use http_body_util::BodyExt;
use reqwest::Client;

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
    let config = SidecarConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai/".into(),
        anthropic_base_url: "http://anthropic/".into(),
        atif_dir: None,
        openinference_endpoint: None,
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

#[tokio::test]
async fn passthrough_rejects_unsupported_provider_path_directly() {
    let config = SidecarConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai".into(),
        anthropic_base_url: "http://anthropic".into(),
        atif_dir: None,
        openinference_endpoint: None,
        metadata: None,
        plugin_config: None,
    };
    let state = AppState {
        config: config.clone(),
        http: Client::new(),
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
    let config = SidecarConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai".into(),
        anthropic_base_url: "http://anthropic".into(),
        atif_dir: None,
        openinference_endpoint: None,
        metadata: None,
        plugin_config: None,
    };
    let state = AppState {
        config: config.clone(),
        http: Client::new(),
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
