// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::agents::shared::alignment::GatewayRouteKind;
use crate::configuration::GatewayConfig;
use crate::server::AppState;
use crate::sessions::{LlmGatewayStart, SessionManager};
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Method, Request, StatusCode, header};
use axum::response::IntoResponse;
use http_body_util::BodyExt;
use reqwest::Client;
use serde_json::Map;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn test_http_client() -> Client {
    Client::new()
}

#[test]
fn removes_hop_by_hop_headers() {
    let headers = HeaderMap::new();
    assert!(!should_forward_request_header(
        &HeaderName::from_static("connection"),
        &headers
    ));
    assert!(!should_forward_request_header(
        &HeaderName::from_static("host"),
        &headers
    ));
    assert!(!should_forward_request_header(
        &HeaderName::from_static(crate::configuration::BOOTSTRAP_CLIENT_TOKEN_HEADER),
        &headers
    ));
    assert!(should_forward_request_header(
        &HeaderName::from_static("authorization"),
        &headers
    ));
    assert!(!should_record_header(
        &HeaderName::from_static("authorization"),
        &headers
    ));
    assert!(!should_record_header(
        &HeaderName::from_static("x-api-key"),
        &headers
    ));
    assert!(!should_record_header(
        &HeaderName::from_static("anthropic-api-key"),
        &headers
    ));
    // Additional credential aliases must not appear in observability metadata:
    // `cookie` carries session credentials; `api-key` is the generic alias used by some providers
    // (e.g., Azure OpenAI). Without these, secrets would leak into `LlmRequest.headers` and any
    // downstream exporter that mirrors them (ATIF, OpenInference span attributes).
    assert!(!should_record_header(
        &HeaderName::from_static("cookie"),
        &headers
    ));
    assert!(!should_record_header(
        &HeaderName::from_static("api-key"),
        &headers
    ));
    assert!(should_record_header(
        &HeaderName::from_static("x-request-id"),
        &headers
    ));

    let mut connection_headers = HeaderMap::new();
    connection_headers.insert(
        header::CONNECTION,
        HeaderValue::from_static("x-private, upgrade"),
    );
    connection_headers.insert("x-private", HeaderValue::from_static("secret"));
    assert!(!should_forward_request_header(
        &HeaderName::from_static("x-private"),
        &connection_headers
    ));
    assert!(!response_headers(&connection_headers).contains_key("x-private"));
}

#[tokio::test]
async fn prepared_gateway_request_consumes_private_client_proof() {
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(
            crate::configuration::BOOTSTRAP_CLIENT_TOKEN_HEADER,
            "hmac-sha256:private-proof",
        )
        .body(Body::from(r#"{"model":"gpt-test"}"#))
        .unwrap();

    let prepared = prepare_gateway_request(
        &GatewayConfig::default(),
        request,
        crate::provider_auth::ProviderRequestAuthorization {
            source_credential:
                crate::provider_auth::SourceCredentialDisposition::ProviderCredential,
            allow_environment_provider_auth: true,
        },
    )
    .await
    .unwrap();

    assert!(prepared.authorization.allow_environment_provider_auth);
    assert!(
        !prepared
            .headers
            .contains_key(crate::configuration::BOOTSTRAP_CLIENT_TOKEN_HEADER)
    );
}

#[tokio::test]
async fn prepared_gateway_request_decodes_zstd_for_observability() {
    let body = br#"{"model":"gpt-test","stream":true}"#;
    let compressed = zstd::stream::encode_all(body.as_slice(), 0).unwrap();
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(header::CONTENT_ENCODING, "zstd")
        .body(Body::from(compressed.clone()))
        .unwrap();

    let prepared = prepare_gateway_request(&GatewayConfig::default(), request, true)
        .await
        .unwrap();

    assert_eq!(prepared.body_bytes.as_ref(), compressed);
    assert_eq!(
        prepared.request_json,
        json!({
            "model": "gpt-test",
            "stream": true,
        })
    );
    assert!(prepared.streaming);
    assert_eq!(
        prepared.headers.get(header::CONTENT_ENCODING).unwrap(),
        "zstd"
    );
}

#[tokio::test]
async fn prepared_gateway_request_decodes_chained_zstd_for_observability() {
    let body = br#"{"model":"gpt-test","stream":true}"#;
    let compressed_once = zstd::stream::encode_all(body.as_slice(), 0).unwrap();
    let compressed_twice = zstd::stream::encode_all(compressed_once.as_slice(), 0).unwrap();
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(header::CONTENT_ENCODING, "zstd, zstd")
        .body(Body::from(compressed_twice.clone()))
        .unwrap();

    let prepared = prepare_gateway_request(&GatewayConfig::default(), request, true)
        .await
        .unwrap();

    assert_eq!(prepared.body_bytes.as_ref(), compressed_twice);
    assert_eq!(
        prepared.request_json,
        json!({
            "model": "gpt-test",
            "stream": true,
        })
    );
    assert_eq!(
        prepared.headers.get(header::CONTENT_ENCODING).unwrap(),
        "zstd, zstd"
    );
}

#[test]
fn zstd_decoder_window_tracks_the_managed_body_limit() {
    assert_eq!(zstd_window_log_max(0), 10);
    assert_eq!(zstd_window_log_max(1 << 10), 10);
    assert_eq!(zstd_window_log_max((1 << 10) + 1), 11);
    assert_eq!(
        zstd_window_log_max(usize::MAX),
        if usize::BITS == 32 { 30 } else { 31 }
    );
}

#[tokio::test]
async fn request_observability_decode_is_bounded_and_encoding_aware() {
    let oversized = vec![b'x'; 256];
    let compressed = zstd::stream::encode_all(oversized.as_slice(), 0).unwrap();
    let config = GatewayConfig {
        max_passthrough_body_bytes: 32,
        ..GatewayConfig::default()
    };
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(header::CONTENT_ENCODING, "zstd")
        .body(Body::from(compressed))
        .unwrap();
    let prepared = prepare_gateway_request(&config, request, true)
        .await
        .unwrap();
    assert!(prepared.request_json.is_null());

    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(header::CONTENT_ENCODING, "gzip")
        .body(Body::from(r#"{"model":"opaque"}"#))
        .unwrap();
    let prepared = prepare_gateway_request(&config, request, true)
        .await
        .unwrap();
    assert!(prepared.request_json.is_null());

    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(header::CONTENT_ENCODING, "identity")
        .body(Body::from(r#"{"model":"gpt-test"}"#))
        .unwrap();
    let prepared = prepare_gateway_request(&config, request, true)
        .await
        .unwrap();
    assert_eq!(
        prepared.request_json,
        json!({
            "model": "gpt-test",
        })
    );
}

#[tokio::test]
async fn malformed_encoded_request_remains_a_raw_passthrough() {
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(header::CONTENT_ENCODING, "zstd")
        .body(Body::from("not-a-zstd-frame"))
        .unwrap();
    let prepared = prepare_gateway_request(&GatewayConfig::default(), request, true)
        .await
        .unwrap();
    let managed = build_llm_gateway_start(&prepared).request;

    let (body, headers) =
        effective_upstream_request(&prepared.body_bytes, &prepared.headers, Some(&managed));

    assert!(managed.content.is_null());
    assert_eq!(body, prepared.body_bytes);
    assert_eq!(headers.get(header::CONTENT_ENCODING).unwrap(), "zstd");
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
    assert_eq!(
        ProviderRoute::OpenAiResponses.alignment_route(),
        GatewayRouteKind::OpenAiResponses
    );
    assert_eq!(
        ProviderRoute::OpenAiChatCompletions.alignment_route(),
        GatewayRouteKind::OpenAiChatCompletions
    );
    assert_eq!(
        ProviderRoute::OpenAiModels.alignment_route(),
        GatewayRouteKind::OpenAiModels
    );
    assert_eq!(
        ProviderRoute::AnthropicMessages.alignment_route(),
        GatewayRouteKind::AnthropicMessages
    );
    assert_eq!(
        ProviderRoute::AnthropicCountTokens.alignment_route(),
        GatewayRouteKind::AnthropicCountTokens
    );
    assert_eq!(ProviderRoute::from_path("/unsupported"), None);
}

#[test]
fn dispatch_override_routes_cover_models_and_count_tokens() {
    for alias in ["openai_models", "openai.models", "/models", "/v1/models"] {
        assert_eq!(
            ProviderRoute::from_dispatch_override(alias),
            Some(ProviderRoute::OpenAiModels),
            "alias {alias}"
        );
    }
    for alias in [
        "anthropic_count_tokens",
        "anthropic.count_tokens",
        "/v1/messages/count_tokens",
    ] {
        assert_eq!(
            ProviderRoute::from_dispatch_override(alias),
            Some(ProviderRoute::AnthropicCountTokens),
            "alias {alias}"
        );
    }
}

#[test]
fn provider_route_names_round_trip_through_alignment_routes() {
    for route in [
        ProviderRoute::OpenAiResponses,
        ProviderRoute::OpenAiChatCompletions,
        ProviderRoute::OpenAiModels,
        ProviderRoute::AnthropicMessages,
        ProviderRoute::AnthropicCountTokens,
    ] {
        assert_eq!(
            GatewayRouteKind::from_provider_name(route.name()),
            Some(route.alignment_route())
        );
    }
}

#[test]
fn provider_routes_preserve_path_query_and_choose_upstream() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai/v1/".into(),

        anthropic_base_url: "http://anthropic/".into(),
        metadata: None,
        plugin_config: None,
        max_hook_payload_bytes: crate::configuration::DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
        max_passthrough_body_bytes: crate::configuration::DEFAULT_MAX_PASSTHROUGH_BODY_BYTES,
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
        max_hook_payload_bytes: crate::configuration::DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
        max_passthrough_body_bytes: crate::configuration::DEFAULT_MAX_PASSTHROUGH_BODY_BYTES,
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
fn effective_upstream_request_removes_content_encoding_after_reencoding() {
    let original_body = Bytes::from_static(b"compressed bytes");
    let mut original_headers = HeaderMap::new();
    original_headers.insert(header::CONTENT_ENCODING, HeaderValue::from_static("zstd"));
    let request = LlmRequest {
        headers: Map::from_iter([("content-encoding".to_string(), json!("zstd"))]),
        content: json!({ "model": "rewritten" }),
    };

    let (body, headers) =
        effective_upstream_request(&original_body, &original_headers, Some(&request));

    assert_eq!(
        serde_json::from_slice::<Value>(&body).unwrap(),
        json!({ "model": "rewritten" })
    );
    assert!(!headers.contains_key(header::CONTENT_ENCODING));
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
fn internal_dispatch_controls_are_consumed_and_never_forwarded() {
    let original_body = Bytes::from_static(br#"{"model":"original"}"#);
    let mut original_headers = HeaderMap::new();
    original_headers.insert(
        INTERNAL_DISPATCH_URL_HEADER,
        HeaderValue::from_static("http://attacker.invalid"),
    );
    original_headers.insert(
        INTERNAL_RETRY_AWARE_HEADER,
        HeaderValue::from_static("true"),
    );
    let request = LlmRequest {
        headers: Map::from_iter([
            (
                INTERNAL_DISPATCH_URL_HEADER.to_string(),
                json!("http://127.0.0.1:9000/v1/responses"),
            ),
            (
                INTERNAL_DISPATCH_ROUTE_HEADER.to_string(),
                json!("openai_responses"),
            ),
            (INTERNAL_RETRY_AWARE_HEADER.to_string(), json!("true")),
            ("x-backend".to_string(), json!("selected")),
        ]),
        content: json!({"model": "selected"}),
    };

    let effective = effective_dispatch_request(
        &original_body,
        &original_headers,
        Some(&request),
        "http://default.invalid/v1/chat/completions",
        ProviderRoute::OpenAiChatCompletions,
    );
    assert_eq!(effective.url, "http://127.0.0.1:9000/v1/responses");
    assert_eq!(effective.target_route, ProviderRoute::OpenAiResponses);
    assert_eq!(effective.headers.get("x-backend").unwrap(), "selected");
    assert!(
        effective
            .headers
            .get(INTERNAL_DISPATCH_URL_HEADER)
            .is_none()
    );
    assert!(
        effective
            .headers
            .get(INTERNAL_DISPATCH_ROUTE_HEADER)
            .is_none()
    );
    assert!(effective.headers.get(INTERNAL_RETRY_AWARE_HEADER).is_none());
    assert!(retry_aware_dispatch(&request));
}

#[test]
fn explicit_keyless_target_drops_source_credentials() {
    let mut source_headers = HeaderMap::new();
    source_headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer source-provider-key"),
    );
    source_headers.insert("x-api-key", HeaderValue::from_static("source-api-key"));
    let request = LlmRequest {
        headers: Map::from_iter([
            (
                INTERNAL_DISPATCH_URL_HEADER.to_string(),
                json!("http://keyless.invalid/v1/chat/completions"),
            ),
            (
                INTERNAL_DISPATCH_ROUTE_HEADER.to_string(),
                json!("openai_chat"),
            ),
        ]),
        content: Value::Null,
    };

    let effective = effective_dispatch_request(
        &Bytes::from_static(br#"{"model":"source"}"#),
        &source_headers,
        Some(&request),
        "http://source.invalid/v1/responses",
        ProviderRoute::OpenAiResponses,
    );

    assert_eq!(
        effective.credential_policy,
        TargetCredentialPolicy::ExplicitTarget
    );
    assert!(!crate::provider_auth::has_provider_credential(
        &effective.headers
    ));
}

#[test]
fn cross_protocol_target_uses_only_binding_owned_authentication() {
    let mut source_headers = HeaderMap::new();
    source_headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer openai-source-key"),
    );
    let request = LlmRequest {
        headers: Map::from_iter([
            (
                INTERNAL_DISPATCH_URL_HEADER.to_string(),
                json!("http://anthropic-target.invalid/v1/messages"),
            ),
            (
                INTERNAL_DISPATCH_ROUTE_HEADER.to_string(),
                json!("anthropic_messages"),
            ),
            ("x-api-key".to_string(), json!("anthropic-binding-key")),
        ]),
        content: Value::Null,
    };

    let effective = effective_dispatch_request(
        &Bytes::from_static(br#"{"model":"source"}"#),
        &source_headers,
        Some(&request),
        "http://source.invalid/v1/responses",
        ProviderRoute::OpenAiResponses,
    );

    assert_eq!(effective.target_route, ProviderRoute::AnthropicMessages);
    assert_eq!(
        effective.credential_policy,
        TargetCredentialPolicy::ExplicitTarget
    );
    assert!(effective.headers.get(header::AUTHORIZATION).is_none());
    assert_eq!(
        effective.headers.get("x-api-key").unwrap(),
        "anthropic-binding-key"
    );

    let built = inject_provider_auth_with_env(
        test_http_client().post("http://anthropic-target.invalid/v1/messages"),
        effective.target_route,
        &effective.headers,
        false,
        |_: &str| Some("ambient-key-must-not-win".into()),
    )
    .build()
    .unwrap();
    assert!(built.headers().get(header::AUTHORIZATION).is_none());
}

#[test]
fn same_route_rewrite_without_explicit_dispatch_preserves_source_auth_policy() {
    let mut source_headers = HeaderMap::new();
    source_headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer source-provider-key"),
    );
    let request = LlmRequest {
        headers: Map::from_iter([("x-plugin-metadata".to_string(), json!("present"))]),
        content: json!({"model": "rewritten"}),
    };

    let effective = effective_dispatch_request(
        &Bytes::from_static(br#"{"model":"source"}"#),
        &source_headers,
        Some(&request),
        "http://source.invalid/v1/responses",
        ProviderRoute::OpenAiResponses,
    );

    assert_eq!(
        effective.credential_policy,
        TargetCredentialPolicy::SourceOrEnvironment
    );
    assert_eq!(
        effective.headers.get(header::AUTHORIZATION).unwrap(),
        "Bearer source-provider-key"
    );
}

#[test]
fn malformed_dispatch_route_discards_the_override_url() {
    let original_body = Bytes::from_static(br#"{"model":"original"}"#);
    let request = LlmRequest {
        headers: Map::from_iter([
            (
                INTERNAL_DISPATCH_URL_HEADER.to_string(),
                json!("http://127.0.0.1:9000/v1/models"),
            ),
            (
                INTERNAL_DISPATCH_ROUTE_HEADER.to_string(),
                json!("not-a-provider-route"),
            ),
        ]),
        content: Value::Null,
    };

    let effective = effective_dispatch_request(
        &original_body,
        &HeaderMap::new(),
        Some(&request),
        "http://default.invalid/v1/chat/completions",
        ProviderRoute::OpenAiChatCompletions,
    );

    assert_eq!(effective.url, "http://default.invalid/v1/chat/completions");
    assert_eq!(effective.target_route, ProviderRoute::OpenAiChatCompletions);
    assert!(
        effective
            .headers
            .get(INTERNAL_DISPATCH_URL_HEADER)
            .is_none()
    );
    assert!(
        effective
            .headers
            .get(INTERNAL_DISPATCH_ROUTE_HEADER)
            .is_none()
    );
}

#[test]
fn dispatch_url_without_a_route_remains_supported() {
    let original_body = Bytes::from_static(br#"{"model":"original"}"#);
    let request = LlmRequest {
        headers: Map::from_iter([(
            INTERNAL_DISPATCH_URL_HEADER.to_string(),
            json!("http://127.0.0.1:9000/v1/chat/completions"),
        )]),
        content: Value::Null,
    };

    let effective = effective_dispatch_request(
        &original_body,
        &HeaderMap::new(),
        Some(&request),
        "http://default.invalid/v1/chat/completions",
        ProviderRoute::OpenAiChatCompletions,
    );

    assert_eq!(effective.url, "http://127.0.0.1:9000/v1/chat/completions");
    assert_eq!(effective.target_route, ProviderRoute::OpenAiChatCompletions);
}

#[test]
fn structured_upstream_failure_classification_matches_retry_policy() {
    let mut headers = HeaderMap::new();
    headers.insert("set-cookie", HeaderValue::from_static("session=secret"));
    headers.insert(
        "www-authenticate",
        HeaderValue::from_static("Bearer realm=provider"),
    );
    headers.insert(
        "proxy-authenticate",
        HeaderValue::from_static("Basic realm=proxy"),
    );
    headers.insert(
        "proxy-authorization",
        HeaderValue::from_static("Basic secret"),
    );
    headers.insert("authorization", HeaderValue::from_static("Bearer secret"));
    headers.insert("cookie", HeaderValue::from_static("session=secret"));
    headers.insert("x-api-key", HeaderValue::from_static("secret"));
    headers.insert("api-key", HeaderValue::from_static("secret"));
    headers.insert("anthropic-api-key", HeaderValue::from_static("secret"));
    headers.insert("connection", HeaderValue::from_static("close"));
    headers.insert("content-length", HeaderValue::from_static("12"));
    headers.insert("retry-after", HeaderValue::from_static("3"));
    headers.insert("x-request-id", HeaderValue::from_static("request-123"));
    for status in [408, 429, 500, 502, 503, 504] {
        let failure = http_failure(
            StatusCode::from_u16(status).unwrap(),
            &headers,
            b"temporary",
        );
        assert!(failure.is_retryable(), "status={status}");
    }
    let failure = http_failure(StatusCode::BAD_GATEWAY, &headers, b"temporary");
    for name in [
        "set-cookie",
        "www-authenticate",
        "proxy-authenticate",
        "proxy-authorization",
        "authorization",
        "cookie",
        "x-api-key",
        "api-key",
        "anthropic-api-key",
        "connection",
        "content-length",
    ] {
        assert!(
            !failure.headers.contains_key(name),
            "sensitive header: {name}"
        );
    }
    assert_eq!(failure.headers.get("retry-after"), Some(&"3".to_string()));
    assert_eq!(
        failure.headers.get("x-request-id"),
        Some(&"request-123".to_string())
    );
    assert_eq!(
        http_failure(
            StatusCode::BAD_REQUEST,
            &headers,
            b"context_length_exceeded"
        )
        .class,
        UpstreamFailureClass::ContextWindow
    );
    assert_eq!(
        http_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            &headers,
            b"model unavailable"
        )
        .class,
        UpstreamFailureClass::ModelUnavailable
    );
    assert!(!http_failure(StatusCode::UNAUTHORIZED, &headers, b"bad token").is_retryable());
    assert!(!http_failure(StatusCode::BAD_REQUEST, &headers, b"invalid request").is_retryable());
    assert_eq!(
        bounded_error_body(&vec![b'x'; MAX_UPSTREAM_ERROR_BODY_BYTES + 10]).len(),
        MAX_UPSTREAM_ERROR_BODY_BYTES
    );
}

#[tokio::test]
async fn retry_aware_buffered_body_read_failure_stays_structured() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0_u8; 1024];
        let _ = socket.read(&mut request).await.unwrap();
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 64\r\nconnection: close\r\n\r\n{}",
            )
            .await
            .unwrap();
    });

    let config = GatewayConfig::default();
    let state = AppState::new(config);
    let prepared = PreparedGatewayRequest {
        method: Method::POST,
        headers: HeaderMap::new(),
        path: "/v1/chat/completions".into(),
        provider: ProviderRoute::OpenAiChatCompletions,
        upstream_url: format!("http://{address}/v1/chat/completions"),
        body_bytes: Bytes::from_static(b"{}"),
        request_json: json!({}),
        streaming: false,
        authorization: crate::provider_auth::ProviderRequestAuthorization {
            source_credential: crate::provider_auth::SourceCredentialDisposition::Absent,
            allow_environment_provider_auth: false,
        },
    };
    let upstream_info = Arc::new(Mutex::new(None));
    let upstream_error = Arc::new(Mutex::new(None));
    let response_bytes = Arc::new(Mutex::new(None));
    let func = build_buffered_func(
        state,
        &prepared,
        upstream_info,
        upstream_error.clone(),
        response_bytes,
    );
    let error = func(LlmRequest {
        headers: Map::from_iter([(INTERNAL_RETRY_AWARE_HEADER.into(), json!("true"))]),
        content: json!({}),
    })
    .await
    .unwrap_err();

    let FlowError::Upstream(failure) = error else {
        panic!("expected structured upstream failure, got {error:?}");
    };
    assert_eq!(failure.class, UpstreamFailureClass::Connection);
    assert!(upstream_error.lock().unwrap().is_none());
    server.await.unwrap();
}

#[test]
fn gateway_session_id_prefers_headers_and_has_fallbacks() {
    let mut headers = HeaderMap::new();
    let codex_body = json!({
        "prompt_cache_key": "codex-thread",
        "client_metadata": {
            "x-codex-installation-id": "install-1",
            "session_id": "codex-session"
        },
        "session_id": "body-session"
    });
    headers.insert(
        "anthropic-beta",
        HeaderValue::from_static("prompt-caching-2024-07-31"),
    );
    assert_eq!(
        gateway_session_id(&headers, &Value::Null, ProviderRoute::AnthropicMessages),
        None
    );

    headers.insert(
        "x-claude-code-session-id",
        HeaderValue::from_static("claude-session"),
    );
    assert_eq!(
        gateway_session_id(&headers, &codex_body, ProviderRoute::OpenAiResponses).as_deref(),
        Some("claude-session")
    );

    headers.insert(
        "x-nemo-relay-session-id",
        HeaderValue::from_static("explicit-session"),
    );
    assert_eq!(
        gateway_session_id(&headers, &codex_body, ProviderRoute::OpenAiResponses).as_deref(),
        Some("explicit-session")
    );

    assert_eq!(
        gateway_session_id(
            &HeaderMap::new(),
            &codex_body,
            ProviderRoute::OpenAiResponses
        )
        .as_deref(),
        Some("codex-session")
    );
    assert_eq!(
        gateway_session_id(
            &HeaderMap::new(),
            &json!({ "prompt_cache_key": "plain-cache-key" }),
            ProviderRoute::OpenAiResponses,
        ),
        None
    );
    assert_eq!(
        gateway_session_id(
            &HeaderMap::new(),
            &codex_body,
            ProviderRoute::OpenAiChatCompletions,
        )
        .as_deref(),
        Some("body-session")
    );
    assert_eq!(
        gateway_session_id(
            &HeaderMap::new(),
            &json!({ "session_id": " body-session " }),
            ProviderRoute::OpenAiResponses,
        )
        .as_deref(),
        Some("body-session")
    );
    assert_eq!(
        gateway_session_id(
            &HeaderMap::new(),
            &json!({ "session_id": "body-session" }),
            ProviderRoute::AnthropicMessages,
        ),
        None
    );
}

#[test]
fn gateway_identifiers_accept_headers_and_scalar_body_values() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-nemo-relay-request-id",
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
            "x-nemo-relay-request-id",
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
fn build_llm_gateway_start_uses_alignment_identifiers_and_metadata() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-nemo-relay-subagent-id",
        HeaderValue::from_static("worker-1"),
    );
    headers.insert("x-request-id", HeaderValue::from_static("transport-req"));
    headers.insert("authorization", HeaderValue::from_static("Bearer secret"));
    let request_json = json!({
        "model": "gpt-test",
        "stream": true,
        "prompt_cache_key": "codex-thread",
        "client_metadata": {
            "x-codex-installation-id": "install-1",
            "x-openai-subagent": "collab_spawn",
            "session_id": "codex-session",
            "thread_id": "child-thread"
        },
        "conversation_id": "conversation-1",
        "generation": { "id": "generation-1" }
    });
    let prepared = PreparedGatewayRequest {
        method: Method::POST,
        headers,
        path: "/responses".into(),
        provider: ProviderRoute::OpenAiResponses,
        upstream_url: "http://openai/v1/responses".into(),
        body_bytes: axum::body::Bytes::new(),
        request_json: request_json.clone(),
        streaming: true,
        authorization: crate::provider_auth::ProviderRequestAuthorization {
            source_credential: crate::provider_auth::SourceCredentialDisposition::Absent,
            allow_environment_provider_auth: true,
        },
    };

    let start = build_llm_gateway_start(&prepared);

    assert_eq!(start.session_id.as_deref(), Some("codex-session"));
    assert_eq!(start.provider, "openai.responses");
    assert_eq!(start.model_name.as_deref(), Some("gpt-test"));
    assert_eq!(start.subagent_id.as_deref(), Some("worker-1"));
    assert_eq!(start.conversation_id.as_deref(), Some("conversation-1"));
    assert_eq!(start.generation_id.as_deref(), Some("generation-1"));
    assert_eq!(start.request_id.as_deref(), Some("transport-req"));
    assert!(start.streaming);
    assert_eq!(start.metadata["gateway_path"], json!("/responses"));
    assert_eq!(start.request.content, request_json);
    assert!(
        !start.request.headers.contains_key("authorization"),
        "observable headers should not leak auth secrets"
    );

    let mut metadata_owned = prepared;
    metadata_owned.headers.remove("x-nemo-relay-subagent-id");
    let start = build_llm_gateway_start(&metadata_owned);
    assert_eq!(start.subagent_id.as_deref(), Some("child-thread"));
}

#[test]
fn observable_headers_omit_secrets_and_transport_headers() {
    let mut headers = HeaderMap::new();
    headers.insert("authorization", HeaderValue::from_static("Bearer secret"));
    headers.insert("x-api-key", HeaderValue::from_static("secret"));
    headers.insert(
        crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER,
        HeaderValue::from_static("proxy-secret"),
    );
    headers.insert("connection", HeaderValue::from_static("close"));
    headers.insert("x-request-id", HeaderValue::from_static("req-1"));

    let observed = observable_headers(&headers);

    assert_eq!(observed.get("x-request-id"), Some(&json!("req-1")));
    assert!(!observed.contains_key("authorization"));
    assert!(!observed.contains_key("x-api-key"));
    assert!(!observed.contains_key(crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER));
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
    let sanitized = strip_replaceable_agent_auth_headers_with_openai_key_state(
        &inbound,
        ProviderRoute::OpenAiResponses,
        true,
    );
    assert!(sanitized.get("authorization").is_none());
}

#[test]
fn preserves_real_bearer_keys_on_openai_route() {
    // Real provider keys (Hermes's `sk-...` against NVIDIA, an actual OpenAI dev key, etc.)
    // must pass through untouched — only recognized ChatGPT auth tokens are stripped.
    let mut inbound = HeaderMap::new();
    inbound.insert(
        "authorization",
        HeaderValue::from_static("Bearer sk-real-provider-key"),
    );
    let sanitized = strip_replaceable_agent_auth_headers_with_openai_key_state(
        &inbound,
        ProviderRoute::OpenAiResponses,
        true,
    );
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
    let sanitized = strip_replaceable_agent_auth_headers_with_openai_key_state(
        &inbound,
        ProviderRoute::AnthropicMessages,
        true,
    );
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
    let sanitized = strip_replaceable_agent_auth_headers_with_openai_key_state(
        &inbound,
        ProviderRoute::OpenAiResponses,
        false,
    );
    assert!(sanitized.get("authorization").is_some());
}

#[test]
fn foreground_gateway_does_not_interpret_agent_specific_placeholder_credentials() {
    let mut inbound = HeaderMap::new();
    inbound.insert(
        "authorization",
        HeaderValue::from_static("Bearer no-key-required"),
    );
    let sanitized = strip_replaceable_agent_auth_headers_with_openai_key_state(
        &inbound,
        ProviderRoute::OpenAiChatCompletions,
        true,
    );
    assert_eq!(
        sanitized.get("authorization").unwrap(),
        "Bearer no-key-required"
    );
}

#[test]
fn generated_transparent_proxy_credentials_are_random_and_high_entropy() {
    let first = crate::provider_auth::TransparentProxyCredential::generate().unwrap();
    let second = crate::provider_auth::TransparentProxyCredential::generate().unwrap();

    assert!(first.expose().starts_with("nrp_"));
    assert_eq!(first.expose().len(), 68);
    assert_ne!(first.expose(), second.expose());
}

#[test]
fn transparent_proxy_dedicated_header_is_consumed_before_plugins() {
    let credential =
        crate::provider_auth::TransparentProxyCredential::from_static("test-proxy-token");
    let mut inbound = HeaderMap::new();
    inbound.insert(
        "authorization",
        HeaderValue::from_static("Bearer real-provider-key"),
    );
    inbound.insert(
        crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER,
        HeaderValue::from_static("test-proxy-token"),
    );

    let disposition = credential.consume(&mut inbound).unwrap();

    assert_eq!(
        disposition,
        crate::provider_auth::SourceCredentialDisposition::RelayProxyCredential {
            provider_credential_present: true
        }
    );
    assert_eq!(
        inbound.get("authorization").unwrap(),
        "Bearer real-provider-key"
    );
    assert!(
        inbound
            .get(crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER)
            .is_none()
    );
}

#[test]
fn transparent_proxy_accepts_standard_openai_and_anthropic_auth_shapes() {
    let credential =
        crate::provider_auth::TransparentProxyCredential::from_static("test-proxy-token");
    for (name, value) in [
        ("authorization", "Bearer test-proxy-token"),
        ("x-api-key", "test-proxy-token"),
        ("api-key", "test-proxy-token"),
        ("anthropic-api-key", "test-proxy-token"),
    ] {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_bytes(name.as_bytes()).unwrap(),
            HeaderValue::from_str(value).unwrap(),
        );

        let disposition = credential.consume(&mut headers).unwrap();

        assert_eq!(
            disposition,
            crate::provider_auth::SourceCredentialDisposition::RelayProxyCredential {
                provider_credential_present: false
            },
            "header={name}"
        );
        assert!(headers.is_empty(), "header={name}");
    }
}

#[test]
fn transparent_proxy_rejects_missing_or_foreign_credentials() {
    let credential =
        crate::provider_auth::TransparentProxyCredential::from_static("test-proxy-token");
    let missing = credential.consume(&mut HeaderMap::new()).unwrap_err();
    assert!(matches!(&missing, CliError::Unauthorized(_)));
    assert_eq!(missing.into_response().status(), StatusCode::UNAUTHORIZED);

    let mut foreign = HeaderMap::new();
    foreign.insert(
        crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER,
        HeaderValue::from_static("different-run-token"),
    );
    let foreign_error = credential.consume(&mut foreign).unwrap_err();
    assert!(matches!(&foreign_error, CliError::Unauthorized(_)));
    assert_eq!(
        foreign_error.into_response().status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        foreign
            .get(crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER)
            .unwrap(),
        "different-run-token"
    );
}

#[test]
fn injects_openai_bearer_when_inbound_has_no_auth() {
    // Foreground gateway mode retains the convenience of supplying its own provider key.
    let http = test_http_client();
    let inbound = HeaderMap::new();
    let env = |k: &str| match k {
        "OPENAI_API_KEY" => Some("sk-test-123".into()),
        _ => None,
    };
    let builder = http.get("http://upstream/v1/responses");
    let built =
        inject_provider_auth_with_env(builder, ProviderRoute::OpenAiResponses, &inbound, true, env)
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
    let built = inject_provider_auth_with_env(
        builder,
        ProviderRoute::AnthropicMessages,
        &inbound,
        true,
        env,
    )
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
        inject_provider_auth_with_env(builder, ProviderRoute::OpenAiResponses, &inbound, true, env)
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
        inject_provider_auth_with_env(builder, ProviderRoute::OpenAiResponses, &inbound, true, env)
            .build()
            .unwrap();
    assert!(built.headers().get("authorization").is_none());
}

#[test]
fn managed_sidecar_never_injects_forwarded_provider_credentials() {
    let http = test_http_client();
    let inbound = HeaderMap::new();
    let env = |_: &str| Some("forwarded-secret".into());
    let builder = http.post("http://upstream/v1/responses");
    let built = inject_provider_auth_with_env(
        builder,
        ProviderRoute::OpenAiResponses,
        &inbound,
        false,
        env,
    )
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
    // With no OPENAI_API_KEY and a JWT, alignment returns the ChatGPT backend override.
    assert_eq!(
        gateway_upstream_url_override_with_openai_key_state(
            ProviderRoute::OpenAiResponses,
            &headers,
            "/responses",
            false,
        )
        .as_deref(),
        Some("https://chatgpt.com/backend-api/codex/responses")
    );
}

#[test]
fn provider_key_does_not_trigger_chatgpt_backend() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer sk-real-api-key"),
    );
    assert_eq!(
        gateway_upstream_url_override_with_openai_key_state(
            ProviderRoute::OpenAiResponses,
            &headers,
            "/responses",
            false,
        ),
        None
    );

    // Empty headers also should not trigger.
    assert_eq!(
        gateway_upstream_url_override_with_openai_key_state(
            ProviderRoute::OpenAiResponses,
            &HeaderMap::new(),
            "/responses",
            false,
        ),
        None
    );
}

#[test]
fn anthropic_route_never_triggers_chatgpt_backend() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer eyJhbGciOiJIUzI1NiJ9.deadbeef.signature"),
    );
    assert_eq!(
        gateway_upstream_url_override_with_openai_key_state(
            ProviderRoute::AnthropicMessages,
            &headers,
            "/v1/messages",
            false,
        ),
        None
    );
}

#[test]
fn chatgpt_backend_url_omits_v1_prefix() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer eyJhbGciOiJIUzI1NiJ9.deadbeef.signature"),
    );
    // The ChatGPT backend expects paths directly under the base, not /v1-prefixed.
    assert_eq!(
        gateway_upstream_url_override_with_openai_key_state(
            ProviderRoute::OpenAiResponses,
            &headers,
            "/responses",
            false,
        )
        .as_deref(),
        Some("https://chatgpt.com/backend-api/codex/responses")
    );
    assert_eq!(
        gateway_upstream_url_override_with_openai_key_state(
            ProviderRoute::OpenAiModels,
            &headers,
            "/models",
            false,
        )
        .as_deref(),
        Some("https://chatgpt.com/backend-api/codex/models")
    );
    // /v1-prefixed inbound paths are stripped
    assert_eq!(
        gateway_upstream_url_override_with_openai_key_state(
            ProviderRoute::OpenAiResponses,
            &headers,
            "/v1/responses",
            false,
        )
        .as_deref(),
        Some("https://chatgpt.com/backend-api/codex/responses")
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
        max_hook_payload_bytes: crate::configuration::DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
        max_passthrough_body_bytes: crate::configuration::DEFAULT_MAX_PASSTHROUGH_BODY_BYTES,
    };
    let state = AppState {
        config: config.clone(),
        bootstrap_fingerprint: None,
        bootstrap_challenge_key: None,
        require_provider_client_token: false,
        transparent_proxy_credential: None,
        http: test_http_client(),
        sessions: SessionManager::new(config),
        last_activity: std::sync::Arc::new(std::sync::Mutex::new(std::time::Instant::now())),
        bootstrap_shutdown: None,
        instance_id: "test-instance".into(),
        bootstrap_tls: None,
        local_address: None,
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
        max_hook_payload_bytes: crate::configuration::DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
        max_passthrough_body_bytes: crate::configuration::DEFAULT_MAX_PASSTHROUGH_BODY_BYTES,
    };
    let state = AppState {
        config: config.clone(),
        bootstrap_fingerprint: None,
        bootstrap_challenge_key: None,
        require_provider_client_token: false,
        transparent_proxy_credential: None,
        http: test_http_client(),
        sessions: SessionManager::new(config),
        last_activity: std::sync::Arc::new(std::sync::Mutex::new(std::time::Instant::now())),
        bootstrap_shutdown: None,
        instance_id: "test-instance".into(),
        bootstrap_tls: None,
        local_address: None,
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

#[tokio::test]
async fn streaming_gateway_call_guard_finishes_when_body_is_dropped() {
    let manager = SessionManager::new(GatewayConfig::default());
    let prep = manager
        .prepare_gateway_call(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("stream-drop".into()),
                provider: "openai.responses".into(),
                model_name: Some("gpt-test".into()),
                subagent_id: None,
                conversation_id: None,
                generation_id: None,
                request_id: None,
                request: LlmRequest {
                    headers: Map::new(),
                    content: json!({
                        "input": "Analyze enough text to create a stable idle-timeout test."
                    }),
                },
                streaming: true,
                metadata: json!({}),
            },
        )
        .await
        .unwrap();

    let stream: LlmJsonStream = Box::pin(futures_util::stream::pending::<
        std::result::Result<Value, FlowError>,
    >());
    let body = client_sse_body(
        stream,
        ProviderRoute::OpenAiResponses,
        manager.clone(),
        prep.session_id,
        prep.owner_subagent_id,
        Arc::new(Mutex::new(None)),
        prep.session_finish,
    );

    drop(body);
    tokio::task::yield_now().await;

    let closed = manager
        .close_idle_sessions_at(
            std::time::Instant::now() + std::time::Duration::from_secs(1),
            std::time::Duration::from_millis(1),
            "idle_timeout",
        )
        .await
        .unwrap();
    assert_eq!(closed, 1);
}

#[test]
fn streaming_gateway_call_guard_finishes_without_a_current_runtime() {
    let subscriber_name = "gateway-no-runtime-drop-test";
    let _ = nemo_relay::api::subscriber::deregister_subscriber(subscriber_name);
    let captured_output = Arc::new(Mutex::new(None::<Value>));
    let captured = captured_output.clone();
    nemo_relay::api::subscriber::register_subscriber(
        subscriber_name,
        Arc::new(move |event| {
            if event.scope_category() == Some(nemo_relay::api::event::ScopeCategory::End)
                && event.name() == "codex-turn"
                && event
                    .metadata()
                    .and_then(|metadata| metadata.get("session_id"))
                    .and_then(Value::as_str)
                    == Some("stream-no-runtime")
            {
                *captured.lock().unwrap() = event.output().cloned();
            }
        }),
    )
    .unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (manager, prep) = runtime.block_on(async {
        let manager = SessionManager::new(GatewayConfig::default());
        let prep = manager
            .prepare_gateway_call(
                &HeaderMap::new(),
                LlmGatewayStart {
                    session_id: Some("stream-no-runtime".into()),
                    provider: "openai.responses".into(),
                    model_name: Some("gpt-test".into()),
                    subagent_id: None,
                    conversation_id: None,
                    generation_id: None,
                    request_id: None,
                    request: LlmRequest {
                        headers: Map::new(),
                        content: json!({ "input": "Record a final response without a runtime." }),
                    },
                    streaming: true,
                    metadata: json!({}),
                },
            )
            .await
            .unwrap();
        (manager, prep)
    });
    let final_response = json!({ "output_text": "streamed final" });
    let stream: LlmJsonStream = Box::pin(futures_util::stream::pending::<
        std::result::Result<Value, FlowError>,
    >());
    let body = client_sse_body(
        stream,
        ProviderRoute::OpenAiResponses,
        manager.clone(),
        prep.session_id,
        prep.owner_subagent_id,
        Arc::new(Mutex::new(Some(final_response.clone()))),
        prep.session_finish,
    );

    drop(body);

    let closed = runtime
        .block_on(manager.close_idle_sessions_at(
            std::time::Instant::now() + std::time::Duration::from_secs(1),
            std::time::Duration::from_millis(1),
            "idle_timeout",
        ))
        .unwrap();
    assert_eq!(closed, 1);
    nemo_relay::api::subscriber::flush_subscribers().unwrap();
    assert_eq!(*captured_output.lock().unwrap(), Some(final_response));
    nemo_relay::api::subscriber::deregister_subscriber(subscriber_name).unwrap();
}

#[tokio::test]
async fn streaming_body_records_final_response_for_turn_output() {
    let subscriber_name = "gateway-stream-final-response-turn-output-test";
    let _ = nemo_relay::api::subscriber::deregister_subscriber(subscriber_name);
    let captured_output = Arc::new(Mutex::new(None::<Value>));
    let captured = captured_output.clone();
    nemo_relay::api::subscriber::register_subscriber(
        subscriber_name,
        Arc::new(move |event| {
            if event.scope_category() == Some(nemo_relay::api::event::ScopeCategory::End)
                && event.name() == "codex-turn"
                && event
                    .metadata()
                    .and_then(|metadata| metadata.get("session_id"))
                    .and_then(Value::as_str)
                    == Some("stream-final")
            {
                *captured.lock().unwrap() = event.output().cloned();
            }
        }),
    )
    .unwrap();

    let manager = SessionManager::new(GatewayConfig::default());
    let prep = manager
        .prepare_gateway_call(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("stream-final".into()),
                provider: "openai.responses".into(),
                model_name: Some("gpt-test".into()),
                subagent_id: None,
                conversation_id: None,
                generation_id: None,
                request_id: None,
                request: LlmRequest {
                    headers: Map::new(),
                    content: json!({
                        "input": "Stream enough text to create a final response."
                    }),
                },
                streaming: true,
                metadata: json!({}),
            },
        )
        .await
        .unwrap();
    let session_id = prep.session_id.clone();
    let owner_subagent_id = prep.owner_subagent_id.clone();
    let final_response = json!({ "output_text": "streamed final" });
    let stream: LlmJsonStream = Box::pin(futures_util::stream::empty::<
        std::result::Result<Value, FlowError>,
    >());
    let body = client_sse_body(
        stream,
        ProviderRoute::OpenAiResponses,
        manager.clone(),
        session_id,
        owner_subagent_id,
        Arc::new(Mutex::new(Some(final_response.clone()))),
        prep.session_finish,
    );
    let _ = body.collect().await.unwrap();

    manager
        .close_idle_sessions_at(
            std::time::Instant::now() + std::time::Duration::from_secs(1),
            std::time::Duration::from_millis(1),
            "idle_timeout",
        )
        .await
        .unwrap();

    nemo_relay::api::subscriber::flush_subscribers().unwrap();
    assert_eq!(*captured_output.lock().unwrap(), Some(final_response));
    nemo_relay::api::subscriber::deregister_subscriber(subscriber_name).unwrap();
}

// `stream_response_records_preview_and_truncation` was removed when the gateway moved to
// `llm_stream_call_execute`. The runtime now owns stream-end lifecycle (start/end events emitted
// by `LlmStreamWrapper`); core tests cover that contract, and the gateway no longer carries a
// stream preview/truncation helper.
