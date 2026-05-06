// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::config::SidecarConfig;
use axum::http::{HeaderMap, HeaderValue};

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
    assert!(should_record_header(&HeaderName::from_static(
        "x-request-id"
    )));
}

#[test]
fn selects_provider_routes() {
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
fn stream_response_records_preview_and_truncation() {
    assert_eq!(
        stream_response_json(b"data: done", false),
        json!({ "stream": "data: done" })
    );
    assert_eq!(
        stream_response_json(b"partial", true),
        json!({ "stream_preview": "partial", "stream_truncated": true })
    );
}
