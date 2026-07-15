// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the Switchyard Relay plugin component.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::{
    Json as AxumJson, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use nemo_relay::api::event::{Event, ScopeCategory};
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmStreamCallExecuteParams, llm_call_execute, llm_stream_call_execute,
};
use nemo_relay::api::runtime::{LlmExecutionNextFn, LlmStreamExecutionNextFn};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::codec::optimization::LlmOptimizationSummaryStatus;
use nemo_relay::error::{UpstreamFailure, UpstreamFailureClass};
use nemo_relay::plugin::rollback_registrations;

use super::*;

fn binding(protocol: WireProtocol, model: &str) -> TargetBinding {
    TargetBinding {
        model: model.into(),
        protocol,
        endpoint: protocol.endpoint().into(),
        base_url: "http://127.0.0.1:9999".into(),
        headers: BTreeMap::new(),
        header_env: BTreeMap::new(),
    }
}

fn config(decision_api_url: String) -> SwitchyardConfig {
    SwitchyardConfig {
        decision_api_url,
        decision_profile_id: "stage_router".into(),
        request_materialization: RequestMaterialization::SummaryOnly,
        context_mode: ContextMode::PayloadOnly,
        decision_timeout_millis: 1_000,
        targets: BTreeMap::from([
            (
                "selected-chat".into(),
                binding(WireProtocol::OpenaiChat, "selected"),
            ),
            (
                "baseline-chat".into(),
                binding(WireProtocol::OpenaiChat, "baseline"),
            ),
            (
                "fallback-chat".into(),
                binding(WireProtocol::OpenaiChat, "fallback"),
            ),
            (
                "fallback-responses".into(),
                binding(WireProtocol::OpenaiResponses, "fallback"),
            ),
            (
                "fallback-anthropic".into(),
                binding(WireProtocol::AnthropicMessages, "fallback"),
            ),
        ]),
        default_targets: ProtocolDefaults {
            openai_chat: "fallback-chat".into(),
            openai_responses: "fallback-responses".into(),
            anthropic_messages: "fallback-anthropic".into(),
        },
        ..SwitchyardConfig::default()
    }
}

fn decision() -> RoutingDecision {
    RoutingDecision {
        schema_version: ROUTING_DECISION_SCHEMA_VERSION.into(),
        decision_id: "decision-1".into(),
        router: crate::contract::DecisionProvider {
            name: "stage_router".into(),
            version: "1".into(),
        },
        route: crate::contract::RoutingTarget {
            tier: "efficient".into(),
            target_model: "selected".into(),
            backend_id: "selected-chat".into(),
            target_protocol_profile: "openai_chat".into(),
            target_endpoint: "/v1/chat/completions".into(),
        },
        baseline_route: Some(crate::contract::RoutingTarget {
            tier: "capable".into(),
            target_model: "baseline".into(),
            backend_id: "baseline-chat".into(),
            target_protocol_profile: "openai_chat".into(),
            target_endpoint: "/v1/chat/completions".into(),
        }),
        confidence: Some(0.9),
        reason_code: Some("test".into()),
        reason_summary: None,
        metadata: BTreeMap::from([
            ("feature_state".into(), json!("fresh")),
            ("snapshot_age_millis".into(), json!(37)),
            ("snapshot_max_age_millis".into(), json!(300_000)),
        ]),
        extra: BTreeMap::new(),
    }
}

fn chat_request() -> LlmRequest {
    LlmRequest {
        headers: Map::new(),
        content: json!({
            "model": "inbound",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "latest"}
            ]
        }),
    }
}

fn request(protocol: WireProtocol) -> LlmRequest {
    let content = match protocol {
        WireProtocol::OpenaiChat => chat_request().content,
        WireProtocol::OpenaiResponses => {
            json!({"model": "inbound", "instructions": "system", "input": "latest"})
        }
        WireProtocol::AnthropicMessages => {
            json!({"model": "inbound", "system": "system", "max_tokens": 32, "messages": [{"role": "user", "content": "latest"}]})
        }
    };
    LlmRequest {
        headers: Map::new(),
        content,
    }
}

fn chat_response() -> Json {
    json!({
        "id": "chat-1", "object": "chat.completion", "model": "selected",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })
}

fn chat_chunk(text: &str, finish_reason: Json) -> Json {
    json!({
        "id": "chat-1", "object": "chat.completion.chunk", "model": "selected",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish_reason}]
    })
}

#[test]
fn wire_protocol_and_plugin_lifecycle_contracts_are_stable() {
    for (protocol, label, endpoint) in [
        (
            WireProtocol::OpenaiChat,
            "openai_chat",
            "/v1/chat/completions",
        ),
        (
            WireProtocol::OpenaiResponses,
            "openai_responses",
            "/v1/responses",
        ),
        (
            WireProtocol::AnthropicMessages,
            "anthropic_messages",
            "/v1/messages",
        ),
    ] {
        assert_eq!(protocol.label(), label);
        assert_eq!(protocol.endpoint(), endpoint);
        assert_eq!(
            WireProtocol::from_call(label, &request(protocol)),
            Some(protocol)
        );
        assert_eq!(
            WireProtocol::from_call("unknown", &request(protocol)),
            Some(protocol)
        );
    }
    assert_eq!(
        WireProtocol::from_call(
            "unknown",
            &LlmRequest {
                headers: Map::new(),
                content: json!({})
            }
        ),
        None
    );
    assert_eq!(RoutingMode::Enforce.label(), "enforce");
    assert_eq!(RoutingMode::ObserveOnly.label(), "observe_only");

    let valid = config("http://127.0.0.1:1/v1/routing/decision".into());
    let component: PluginComponentSpec = valid.clone().into();
    assert_eq!(component.kind, SWITCHYARD_PLUGIN_KIND);
    assert!(component.enabled);
    assert_eq!(component.config["decision_profile_id"], "stage_router");

    let plugin = SwitchyardPlugin;
    assert_eq!(plugin.plugin_kind(), SWITCHYARD_PLUGIN_KIND);
    assert!(!plugin.allows_multiple_components());
    assert!(plugin.validate(&component.config).is_empty());
    let diagnostics = plugin.validate(json!({"version": "invalid"}).as_object().unwrap());
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].level, DiagnosticLevel::Error);
    assert_eq!(diagnostics[0].code, "switchyard.invalid_config");
}

#[tokio::test]
async fn plugin_registration_installs_and_rolls_back_both_execution_intercepts() {
    let plugin = SwitchyardPlugin;
    let decision_api_url = health_server(StatusCode::OK, r#"{"status":"ok"}"#).await;
    for (mode, profile_id) in [
        (RoutingMode::Enforce, "random"),
        (RoutingMode::Enforce, "llm"),
        (RoutingMode::Enforce, "stage_router"),
        (RoutingMode::ObserveOnly, "random"),
        (RoutingMode::ObserveOnly, "llm"),
        (RoutingMode::ObserveOnly, "stage_router"),
    ] {
        let mut candidate = config(decision_api_url.clone());
        candidate.mode = mode;
        candidate.decision_profile_id = profile_id.into();
        let component: PluginComponentSpec = candidate.into();
        let mut context = PluginRegistrationContext::with_namespace(format!(
            "switchyard-test-{}-",
            Uuid::now_v7()
        ));

        plugin
            .register(&component.config, &mut context)
            .await
            .unwrap();

        let mut registrations = context.into_registrations();
        assert_eq!(registrations.len(), 2);
        rollback_registrations(&mut registrations);
        assert!(registrations.is_empty());
    }
}

#[tokio::test]
async fn plugin_registration_retries_transient_sidecar_health_failures() {
    let (decision_api_url, attempts) = recovering_health_server(2).await;
    let plugin = SwitchyardPlugin;
    let component: PluginComponentSpec = config(decision_api_url).into();
    let mut context =
        PluginRegistrationContext::with_namespace(format!("switchyard-test-{}-", Uuid::now_v7()));

    plugin
        .register(&component.config, &mut context)
        .await
        .unwrap();

    assert_eq!(attempts.load(Ordering::SeqCst), 3);
    let mut registrations = context.into_registrations();
    assert_eq!(registrations.len(), 2);
    rollback_registrations(&mut registrations);
    assert!(registrations.is_empty());
}

#[tokio::test]
async fn plugin_registration_returns_the_final_health_failure_after_retry_exhaustion() {
    let (decision_api_url, attempts) = recovering_health_server(usize::MAX).await;
    let plugin = SwitchyardPlugin;
    let component: PluginComponentSpec = config(decision_api_url).into();
    let mut context =
        PluginRegistrationContext::with_namespace(format!("switchyard-test-{}-", Uuid::now_v7()));

    let error = plugin
        .register(&component.config, &mut context)
        .await
        .unwrap_err();

    assert_eq!(attempts.load(Ordering::SeqCst), 3);
    assert!(
        error
            .to_string()
            .contains("returned HTTP 503 Service Unavailable")
    );
    assert!(context.into_registrations().is_empty());
}

#[tokio::test]
async fn plugin_registration_rejects_unhealthy_or_invalid_sidecars() {
    for (status, body, expected) in [
        (
            StatusCode::SERVICE_UNAVAILABLE,
            r#"{"status":"starting"}"#,
            "returned HTTP 503 Service Unavailable",
        ),
        (StatusCode::OK, "not-json", "returned invalid JSON"),
        (
            StatusCode::OK,
            r#"{"status":"starting"}"#,
            "did not report status=ok",
        ),
    ] {
        let plugin = SwitchyardPlugin;
        let component: PluginComponentSpec = config(health_server(status, body).await).into();
        let mut context = PluginRegistrationContext::with_namespace(format!(
            "switchyard-test-{}-",
            Uuid::now_v7()
        ));
        let error = plugin
            .register(&component.config, &mut context)
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "{error:?} did not contain {expected:?}"
        );
        assert!(context.into_registrations().is_empty());
    }
}

#[tokio::test]
async fn plugin_registration_rejects_an_unreachable_sidecar() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);

    let plugin = SwitchyardPlugin;
    let component: PluginComponentSpec =
        config(format!("http://{address}/v1/routing/decision")).into();
    let mut context =
        PluginRegistrationContext::with_namespace(format!("switchyard-test-{}-", Uuid::now_v7()));
    let error = plugin
        .register(&component.config, &mut context)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("Switchyard service is required"));
    assert!(error.to_string().contains("/health"));
    assert!(context.into_registrations().is_empty());
}

#[test]
fn configuration_validation_rejects_unsafe_or_ambiguous_bindings() {
    let url = "http://127.0.0.1:1/v1/routing/decision".to_string();
    let assert_invalid = |config: SwitchyardConfig, expected: &str| {
        let error = SwitchyardRuntime::new(config)
            .err()
            .expect("config must fail");
        assert!(
            error.contains(expected),
            "{error:?} did not contain {expected:?}"
        );
    };

    let mut candidate = config(url.clone());
    candidate.version = 2;
    assert_invalid(candidate, "unsupported Switchyard config version");
    let mut candidate = config(url.clone());
    candidate.decision_profile_id.clear();
    assert_invalid(candidate, "decision_profile_id");
    let mut candidate = config(url.clone());
    candidate.decision_timeout_millis = 0;
    assert_invalid(candidate, "decision_timeout_millis");
    let mut candidate = config(url.clone());
    candidate.max_retries = 11;
    assert_invalid(candidate, "max_retries");
    let mut candidate = config(url.clone());
    candidate.recent_message_count = 0;
    assert_invalid(candidate, "recent_message_count");
    let mut candidate = config(url.clone());
    candidate.context_mode = ContextMode::AtofRequired;
    assert_invalid(candidate, "atof_endpoint_name");
    let mut candidate = config(url.clone());
    candidate.atof_endpoint_name = Some(" switchyard ".into());
    assert_invalid(candidate, "leading or trailing whitespace");
    let candidate = config("file:///tmp/decision".into());
    assert_invalid(candidate, "must use http or https");
    let mut candidate = config(url.clone());
    candidate.targets.clear();
    assert_invalid(candidate, "targets must not be empty");
    let mut candidate = config(url.clone());
    candidate.enabled_inbound_profiles.clear();
    assert_invalid(candidate, "enabled_inbound_profiles");

    let mut candidate = config(url.clone());
    candidate.targets.get_mut("selected-chat").unwrap().endpoint = "/wrong".into();
    assert_invalid(candidate, "endpoint must be");
    let mut candidate = config(url.clone());
    candidate.targets.get_mut("selected-chat").unwrap().base_url = "file:///tmp/provider".into();
    assert_invalid(candidate, "base_url must use http or https");
    let mut candidate = config(url.clone());
    candidate.targets.insert(
        "duplicate-chat".into(),
        candidate.targets["selected-chat"].clone(),
    );
    assert_invalid(candidate, "conflicts with another exact backend binding");
    let mut candidate = config(url.clone());
    candidate.default_targets.openai_chat = "missing".into();
    assert_invalid(candidate, "default target \"missing\" is not configured");
    let mut candidate = config(url.clone());
    candidate.default_targets.openai_chat = "fallback-responses".into();
    assert_invalid(candidate, "must use protocol openai_chat");

    let mut candidate = config(url.clone());
    candidate
        .decision_headers
        .insert("bad header".into(), "value".into());
    assert_invalid(candidate, "invalid header name");
    let mut candidate = config(url.clone());
    candidate
        .decision_headers
        .insert("x-test".into(), "bad\nvalue".into());
    assert_invalid(candidate, "invalid header value");
    let mut candidate = config(url.clone());
    candidate.decision_header_env.insert(
        "authorization".into(),
        "SWITCHYARD_TEST_MISSING_DECISION_SECRET".into(),
    );
    assert_invalid(candidate, "is not set");
    let mut candidate = config(url);
    candidate
        .targets
        .get_mut("selected-chat")
        .unwrap()
        .header_env
        .insert(
            "authorization".into(),
            "SWITCHYARD_TEST_MISSING_TARGET_SECRET".into(),
        );
    assert_invalid(candidate, "is not set");
}

#[test]
fn disabled_protocol_defaults_are_optional() {
    let defaults: ProtocolDefaults = serde_json::from_value(json!({
        "anthropic_messages": "fallback-anthropic"
    }))
    .expect("omitted disabled protocol defaults should deserialize");
    assert!(defaults.openai_chat.is_empty());
    assert!(defaults.openai_responses.is_empty());

    let mut candidate = config("http://127.0.0.1:1/v1/routing/decision".into());
    candidate.enabled_inbound_profiles = BTreeSet::from([WireProtocol::AnthropicMessages]);
    candidate.default_targets.openai_chat.clear();
    candidate.default_targets.openai_responses.clear();

    SwitchyardRuntime::new(candidate).expect("Anthropic-only configuration should validate");
}

#[test]
fn atof_cross_component_validation_rejects_invalid_endpoint_names_before_lookup() {
    for (name, expected) in [
        (" ", "atof_endpoint_name must be non-empty when configured"),
        (
            " switchyard ",
            "atof_endpoint_name must not have leading or trailing whitespace",
        ),
    ] {
        let mut switchyard = config("http://switchyard.test/v1/routing/decision".into());
        switchyard.context_mode = ContextMode::AtofRequired;
        switchyard.atof_endpoint_name = Some(name.into());
        let plugin_config = PluginConfig {
            components: vec![switchyard.into()],
            ..PluginConfig::default()
        };

        assert_eq!(
            validate_switchyard_atof_configuration(&plugin_config).unwrap_err(),
            expected
        );
    }
}

#[test]
fn atof_cross_component_validation_reports_each_activation_mismatch() {
    assert!(validate_switchyard_atof_configuration(&PluginConfig::default()).is_ok());

    let payload = PluginConfig {
        components: vec![config("http://switchyard.test/v1/routing/decision".into()).into()],
        ..PluginConfig::default()
    };
    assert!(validate_switchyard_atof_configuration(&payload).is_ok());

    let mut switchyard = config("http://switchyard.test/v1/routing/decision".into());
    switchyard.context_mode = ContextMode::AtofRequired;
    switchyard.atof_endpoint_name = Some("switchyard".into());
    let mut plugin_config = PluginConfig {
        components: vec![switchyard.into()],
        ..PluginConfig::default()
    };
    plugin_config.components.push(PluginComponentSpec {
        kind: "observability".into(),
        enabled: true,
        config: json!({"atof": {"enabled": true, "sinks": [{
            "type": "stream",
            "name": "other",
            "url": "http://events.test/v1/atof/events",
            "header_env": {"authorization": "TOKEN"}
        }]}})
        .as_object()
        .unwrap()
        .clone(),
    });
    assert!(
        validate_switchyard_atof_configuration(&plugin_config)
            .unwrap_err()
            .contains("requires named ATOF endpoint")
    );

    plugin_config.components[1].config["atof"]["sinks"][0]["name"] = json!("switchyard");
    assert!(validate_switchyard_atof_configuration(&plugin_config).is_ok());
    plugin_config.components[1].config["atof"]["sinks"][0]["transport"] = json!("websocket");
    assert!(
        validate_switchyard_atof_configuration(&plugin_config)
            .unwrap_err()
            .contains("transport = http_post")
    );
    plugin_config.components[1].config["atof"]["sinks"][0]["transport"] = json!("http_post");
    let duplicate = plugin_config.components[1].config["atof"]["sinks"][0].clone();
    plugin_config.components[1].config["atof"]["sinks"]
        .as_array_mut()
        .unwrap()
        .push(duplicate);
    assert!(
        validate_switchyard_atof_configuration(&plugin_config)
            .unwrap_err()
            .contains("exactly one endpoint")
    );
    plugin_config.components[1].config["atof"]["sinks"]
        .as_array_mut()
        .unwrap()
        .pop();
    plugin_config.components[1].config["atof"]["sinks"][0]["field_name_policy"] =
        json!("snake_case");
    assert!(
        validate_switchyard_atof_configuration(&plugin_config)
            .unwrap_err()
            .contains("field_name_policy = preserve")
    );
    let endpoint = &mut plugin_config.components[1].config["atof"]["sinks"][0];
    endpoint["field_name_policy"] = json!("preserve");
    endpoint.as_object_mut().unwrap().remove("header_env");
    assert!(
        validate_switchyard_atof_configuration(&plugin_config)
            .unwrap_err()
            .contains("environment-referenced header")
    );
}

#[tokio::test]
async fn execution_bypasses_inapplicable_calls_and_fails_open_on_extensions() {
    let runtime =
        SwitchyardRuntime::new(config("http://127.0.0.1:1/v1/routing/decision".into())).unwrap();
    let passthrough: LlmExecutionNextFn =
        Arc::new(|request| Box::pin(async move { Ok(request.content) }));
    let unknown = LlmRequest {
        headers: Map::new(),
        content: json!({"opaque": true}),
    };
    assert_eq!(
        runtime
            .execute_buffered("custom.provider", unknown, Arc::clone(&passthrough))
            .await
            .unwrap()["opaque"],
        true
    );

    let mut disabled_config = config("http://127.0.0.1:1/v1/routing/decision".into());
    disabled_config
        .enabled_inbound_profiles
        .remove(&WireProtocol::OpenaiChat);
    let disabled = SwitchyardRuntime::new(disabled_config).unwrap();
    assert_eq!(
        disabled
            .execute_buffered(
                "openai.chat_completions",
                chat_request(),
                Arc::clone(&passthrough),
            )
            .await
            .unwrap()["model"],
        "inbound"
    );

    let mut unsupported = chat_request();
    unsupported.content["thinking"] = json!({"type": "enabled"});
    let response = runtime
        .execute_buffered(
            "openai.chat_completions",
            unsupported,
            Arc::new(|request| {
                Box::pin(async move {
                    assert_eq!(request.content["model"], "fallback");
                    Ok(chat_response())
                })
            }),
        )
        .await
        .unwrap();
    assert_eq!(response["choices"][0]["message"]["content"], "ok");
}

#[tokio::test]
async fn stream_setup_retries_and_empty_streams_have_one_bounded_fallback() {
    let (url, decisions) = decision_server().await;
    let runtime = SwitchyardRuntime::new(config(url)).unwrap();
    let dispatches = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&dispatches);
    let next: LlmStreamExecutionNextFn = Arc::new(move |_| {
        let seen = Arc::clone(&seen);
        Box::pin(async move {
            if seen.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(FlowError::Upstream(UpstreamFailure {
                    status: None,
                    body: "connect".into(),
                    headers: BTreeMap::new(),
                    class: UpstreamFailureClass::Connection,
                }));
            }
            Ok(Box::pin(futures_stream::iter(vec![Ok(chat_chunk(
                "ok",
                json!("stop"),
            ))])) as LlmJsonStream)
        })
    });
    let output = runtime
        .execute_stream("openai.chat_completions", chat_request(), next)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(output.len(), 1);
    assert_eq!(dispatches.load(Ordering::SeqCst), 2);
    assert_eq!(decisions.lock().unwrap().len(), 2);

    let (url, decisions) = decision_server().await;
    let runtime = SwitchyardRuntime::new(config(url)).unwrap();
    let dispatches = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&dispatches);
    let next: LlmStreamExecutionNextFn = Arc::new(move |request| {
        let seen = Arc::clone(&seen);
        Box::pin(async move {
            let attempt = seen.fetch_add(1, Ordering::SeqCst);
            let items = if attempt < 4 {
                Vec::new()
            } else {
                assert_eq!(request.content["model"], "fallback");
                vec![Ok(chat_chunk("fallback", json!("stop")))]
            };
            Ok(Box::pin(futures_stream::iter(items)) as LlmJsonStream)
        })
    });
    let output = runtime
        .execute_stream("openai.chat_completions", chat_request(), next)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(output.len(), 1);
    assert_eq!(dispatches.load(Ordering::SeqCst), 5);
    assert_eq!(decisions.lock().unwrap().len(), 4);
}

#[tokio::test]
async fn fallback_setup_failures_preserve_the_provider_error() {
    let runtime =
        SwitchyardRuntime::new(config("http://127.0.0.1:1/v1/routing/decision".into())).unwrap();
    let buffered: LlmExecutionNextFn = Arc::new(|_| {
        Box::pin(async { Err(FlowError::Internal("buffered fallback failed".into())) })
    });
    let error = runtime
        .dispatch_fallback_buffered(WireProtocol::OpenaiChat, chat_request(), buffered, "test")
        .await
        .unwrap_err();
    assert!(error.to_string().contains("buffered fallback failed"));

    let streaming: LlmStreamExecutionNextFn =
        Arc::new(|_| Box::pin(async { Err(FlowError::Internal("stream fallback failed".into())) }));
    let error = runtime
        .dispatch_fallback_stream(WireProtocol::OpenaiChat, chat_request(), streaming, "test")
        .await
        .err()
        .expect("stream fallback setup must fail");
    assert!(error.to_string().contains("stream fallback failed"));
}

#[tokio::test]
async fn translated_stream_preserves_success_and_propagates_both_error_sources() {
    let source = Box::pin(futures_stream::iter(vec![
        Ok(chat_chunk("hello", Json::Null)),
        Ok(chat_chunk("", json!("stop"))),
    ])) as LlmJsonStream;
    let output = translated_stream(
        WireProtocol::OpenaiChat,
        WireProtocol::AnthropicMessages,
        "selected".into(),
        source,
    )
    .collect::<Vec<_>>()
    .await;
    assert!(output.iter().all(Result::is_ok));
    assert!(output.iter().any(|item| {
        item.as_ref()
            .is_ok_and(|chunk| chunk.to_string().contains("hello"))
    }));

    let upstream_error = FlowError::Internal("upstream stream failed".into());
    let source = Box::pin(futures_stream::iter(vec![Err(upstream_error)])) as LlmJsonStream;
    let output = translated_stream(
        WireProtocol::OpenaiChat,
        WireProtocol::AnthropicMessages,
        "selected".into(),
        source,
    )
    .collect::<Vec<_>>()
    .await;
    assert!(output[0].is_err());

    let malformed = Box::pin(futures_stream::iter(vec![Ok(json!({
        "choices": [{"delta": {"reasoning_content": "private"}}]
    }))])) as LlmJsonStream;
    let output = translated_stream(
        WireProtocol::OpenaiChat,
        WireProtocol::AnthropicMessages,
        "selected".into(),
        malformed,
    )
    .collect::<Vec<_>>()
    .await;
    assert!(output[0].is_err());
}

#[test]
fn provider_failure_reporting_covers_every_retry_class() {
    for (class, label, retryable) in [
        (UpstreamFailureClass::Connection, "connection", true),
        (UpstreamFailureClass::Timeout, "timeout", true),
        (
            UpstreamFailureClass::RetryableStatus,
            "retryable_status",
            true,
        ),
        (UpstreamFailureClass::ContextWindow, "context_window", true),
        (
            UpstreamFailureClass::ModelUnavailable,
            "model_unavailable",
            true,
        ),
        (
            UpstreamFailureClass::Authentication,
            "authentication",
            false,
        ),
        (
            UpstreamFailureClass::InvalidRequest,
            "invalid_request",
            false,
        ),
        (UpstreamFailureClass::Other, "other", false),
    ] {
        let error = FlowError::Upstream(UpstreamFailure {
            status: Some(503),
            body: "failure".into(),
            headers: BTreeMap::new(),
            class,
        });
        assert_eq!(provider_error_class(&error), label);
        assert_eq!(error_is_retryable(&error), retryable);
        assert_eq!(provider_error_summary(&error), format!("{label}:http_503"));
    }
    let relay = FlowError::Internal("failure".into());
    assert_eq!(provider_error_class(&relay), "relay");
    assert_eq!(provider_error_summary(&relay), relay.to_string());
}

#[test]
fn all_materialization_modes_are_bounded_and_provider_valid() {
    for protocol in [
        WireProtocol::OpenaiChat,
        WireProtocol::OpenaiResponses,
        WireProtocol::AnthropicMessages,
    ] {
        for mode in [
            RequestMaterialization::None,
            RequestMaterialization::SummaryOnly,
            RequestMaterialization::LatestUserPrompt,
            RequestMaterialization::RecentMessageWindow,
            RequestMaterialization::AnnotatedRequest,
            RequestMaterialization::FullBody,
        ] {
            let mut config = config("http://127.0.0.1:1/v1/routing/decision".into());
            config.request_materialization = mode;
            config.recent_message_count = 2;
            let runtime = SwitchyardRuntime::new(config).unwrap();
            let routing = runtime
                .routing_request(protocol, &request(protocol), 1, None)
                .unwrap();
            match mode {
                RequestMaterialization::None | RequestMaterialization::SummaryOnly => {
                    assert!(routing.current_request.is_none())
                }
                RequestMaterialization::LatestUserPrompt => {
                    let current = routing.current_request.unwrap();
                    assert_eq!(current["latest_user_prompt"], "latest");
                    let body = current["body"].clone();
                    decode_request(
                        &runtime.translation,
                        protocol,
                        &LlmRequest {
                            headers: Map::new(),
                            content: body,
                        },
                    )
                    .unwrap();
                }
                RequestMaterialization::RecentMessageWindow => {
                    let current = routing.current_request.unwrap();
                    decode_request(
                        &runtime.translation,
                        protocol,
                        &LlmRequest {
                            headers: Map::new(),
                            content: current["body"].clone(),
                        },
                    )
                    .unwrap();
                }
                RequestMaterialization::AnnotatedRequest | RequestMaterialization::FullBody => {
                    assert!(routing.current_request.is_some())
                }
            }
        }
    }
}

#[test]
fn identity_policy_requires_stable_request_scope_only_for_atof_profiles() {
    let mut config = config("http://127.0.0.1:1/v1/routing/decision".into());
    let payload_runtime = SwitchyardRuntime::new(config.clone()).unwrap();
    let synthetic = payload_runtime
        .routing_request(WireProtocol::OpenaiChat, &chat_request(), 1, None)
        .unwrap();
    assert_eq!(synthetic.identity.quality, "synthetic");

    config.context_mode = ContextMode::AtofRequired;
    config.atof_endpoint_name = Some("switchyard".into());
    let atof_runtime = SwitchyardRuntime::new(config).unwrap();
    assert!(
        atof_runtime
            .routing_request(WireProtocol::OpenaiChat, &chat_request(), 1, None)
            .is_err()
    );
    let mut stable = chat_request();
    stable
        .headers
        .insert("x-nemo-relay-session-id".into(), json!("session-1"));
    stable
        .headers
        .insert("x-nemo-relay-request-id".into(), json!("request-1"));
    let routed = atof_runtime
        .routing_request(WireProtocol::OpenaiChat, &stable, 1, None)
        .unwrap();
    assert_eq!(routed.identity.quality, "explicit");
}

#[test]
fn exact_target_validation_rejects_any_switchyard_drift() {
    let runtime =
        SwitchyardRuntime::new(config("http://127.0.0.1:1/v1/routing/decision".into())).unwrap();
    assert!(runtime.validate_decision(&decision()).is_ok());
    let mut drifted = decision();
    drifted.route.target_model = "unbound-model".into();
    assert!(runtime.validate_decision(&drifted).is_err());
    drifted = decision();
    drifted.route.backend_id = "unknown".into();
    assert!(runtime.validate_decision(&drifted).is_err());
}

#[test]
fn routing_contribution_requires_an_exact_independent_baseline_binding() {
    let runtime =
        SwitchyardRuntime::new(config("http://127.0.0.1:1/v1/routing/decision".into())).unwrap();

    let contribution = runtime.routing_contribution(&decision(), 2, true).unwrap();
    assert!(contribution.applied);
    assert_eq!(
        contribution.kind.as_str(),
        LlmOptimizationKind::MODEL_ROUTING
    );
    let transition = contribution.model_transition.unwrap();
    assert_eq!(transition.baseline.unwrap().model, "baseline");
    assert_eq!(transition.effective.unwrap().model, "selected");
    assert_eq!(contribution.payload.as_ref().unwrap()["routing_attempt"], 2);
    assert_eq!(
        contribution.payload.as_ref().unwrap()["router_metadata"]["feature_state"],
        "fresh"
    );
    assert_eq!(
        contribution.payload.as_ref().unwrap()["router_metadata"]["snapshot_age_millis"],
        37
    );
    assert_eq!(
        contribution.payload_schema.as_ref().unwrap().name,
        ROUTING_CONTRIBUTION_SCHEMA
    );

    let observed = runtime.routing_contribution(&decision(), 1, false).unwrap();
    assert!(!observed.applied);

    let mut missing = decision();
    missing.baseline_route = None;
    assert!(runtime.routing_contribution(&missing, 1, true).is_none());

    let mut drifted = decision();
    drifted.baseline_route.as_mut().unwrap().target_model = "drifted".into();
    assert!(runtime.routing_contribution(&drifted, 1, true).is_none());
    assert!(runtime.validate_decision(&drifted).is_ok());
}

#[test]
fn routing_decision_mark_has_canonical_shape_and_mirrored_identity() {
    let subscriber_name = format!("switchyard-mark-shape-{}", uuid::Uuid::now_v7());
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        &subscriber_name,
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let runtime =
        SwitchyardRuntime::new(config("http://127.0.0.1:1/v1/routing/decision".into())).unwrap();
    let routing_request = runtime
        .routing_request(WireProtocol::OpenaiChat, &chat_request(), 1, None)
        .unwrap();
    runtime.emit_decision(&routing_request, &decision(), 1, false, 17);
    flush_subscribers().unwrap();
    deregister_subscriber(&subscriber_name).unwrap();

    let event = events
        .lock()
        .unwrap()
        .iter()
        .map(Event::to_json_value)
        .find(|event| {
            event["name"] == "switchyard.routing.decision"
                && event["metadata"]["session_id"] == routing_request.identity.session_id
                && event["metadata"]["request_id"] == routing_request.identity.request_id
        })
        .expect("decision mark should be captured");
    assert_eq!(event["kind"], "mark");
    assert_eq!(event["category"], "custom");
    assert_eq!(
        event["category_profile"]["subtype"],
        "switchyard.routing.decision"
    );
    assert_eq!(event["data_schema"]["name"], ROUTING_MARK_SCHEMA);
    assert_eq!(event["data_schema"]["version"], "1");
    assert_eq!(event["data"]["profile_id"], "stage_router");
    assert_eq!(event["data"]["selected_model"], "selected");
    assert_eq!(event["data"]["latency_ms"], 17);
    assert_eq!(event["data"]["router_metadata"]["feature_state"], "fresh");
    assert_eq!(event["data"]["router_metadata"]["snapshot_age_millis"], 37);
    assert_eq!(
        event["metadata"]["session_id"],
        routing_request.identity.session_id
    );
    assert_eq!(
        event["metadata"]["request_id"],
        routing_request.identity.request_id
    );
}

#[test]
fn atof_required_cross_component_validation_is_context_sensitive() {
    let mut switchyard = config("http://switchyard.test:8080/v1/routing/decision".into());
    switchyard.context_mode = ContextMode::AtofRequired;
    switchyard.atof_endpoint_name = Some("switchyard".into());
    let mut plugin_config = PluginConfig {
        components: vec![switchyard.into()],
        ..PluginConfig::default()
    };
    assert!(validate_switchyard_atof_configuration(&plugin_config).is_err());
    plugin_config.components.push(PluginComponentSpec {
        kind: "observability".into(),
        enabled: true,
        config: json!({"atof": {
            "enabled": true,
            "sinks": [{
                "type": "stream",
                "name": "switchyard",
                "url": "http://switchyard.test:8080/v1/atof/events",
                "transport": "http_post",
                "field_name_policy": "preserve",
                "header_env": {"authorization": "SWITCHYARD_TOKEN"}
            }]
        }})
        .as_object()
        .unwrap()
        .clone(),
    });
    assert!(validate_switchyard_atof_configuration(&plugin_config).is_ok());
}

#[derive(Clone)]
struct DecisionState {
    requests: Arc<Mutex<Vec<RoutingRequest>>>,
    decision: RoutingDecision,
}

async fn decision_handler(
    State(state): State<DecisionState>,
    AxumJson(request): AxumJson<RoutingRequest>,
) -> AxumJson<RoutingDecision> {
    state.requests.lock().unwrap().push(request);
    AxumJson(state.decision)
}

async fn decision_server() -> (String, Arc<Mutex<Vec<RoutingRequest>>>) {
    decision_server_for(decision()).await
}

async fn decision_server_for(
    decision: RoutingDecision,
) -> (String, Arc<Mutex<Vec<RoutingRequest>>>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let state = DecisionState {
        requests: Arc::clone(&requests),
        decision,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new()
                .route("/v1/routing/decision", post(decision_handler))
                .with_state(state),
        )
        .await
        .unwrap();
    });
    (format!("http://{address}/v1/routing/decision"), requests)
}

async fn health_server(status: StatusCode, body: &'static str) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().route(
                "/health",
                get(move || async move { (status, [("content-type", "application/json")], body) }),
            ),
        )
        .await
        .unwrap();
    });
    format!("http://{address}/nested/v1/routing/decision?source=test#ignored")
}

async fn recovering_health_server(failures_before_ready: usize) -> (String, Arc<AtomicUsize>) {
    let attempts = Arc::new(AtomicUsize::new(0));
    let captured = Arc::clone(&attempts);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().route(
                "/health",
                get(move || {
                    let attempts = Arc::clone(&captured);
                    async move {
                        if attempts.fetch_add(1, Ordering::SeqCst) < failures_before_ready {
                            (
                                StatusCode::SERVICE_UNAVAILABLE,
                                AxumJson(json!({"status": "starting"})),
                            )
                        } else {
                            (StatusCode::OK, AxumJson(json!({"status": "ok"})))
                        }
                    }
                }),
            ),
        )
        .await
        .unwrap();
    });
    (format!("http://{address}/v1/routing/decision"), attempts)
}

async fn managed_buffered_events(
    runtime: SwitchyardRuntime,
    next: LlmExecutionNextFn,
) -> Vec<Event> {
    let subscriber_name = format!("switchyard-accounting-{}", uuid::Uuid::now_v7());
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        &subscriber_name,
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    let runtime = Arc::new(runtime);
    let func: LlmExecutionNextFn = Arc::new(move |request| {
        let runtime = Arc::clone(&runtime);
        let next = Arc::clone(&next);
        Box::pin(async move {
            runtime
                .execute_buffered("openai.chat_completions", request, next)
                .await
        })
    });
    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai.chat_completions")
            .request(chat_request())
            .func(func)
            .build(),
    )
    .await
    .unwrap();
    flush_subscribers().unwrap();
    deregister_subscriber(&subscriber_name).unwrap();
    Arc::try_unwrap(events).unwrap().into_inner().unwrap()
}

async fn managed_stream_events(
    runtime: SwitchyardRuntime,
    next: LlmStreamExecutionNextFn,
) -> Vec<Event> {
    let subscriber_name = format!("switchyard-stream-accounting-{}", uuid::Uuid::now_v7());
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        &subscriber_name,
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    let runtime = Arc::new(runtime);
    let func: LlmStreamExecutionNextFn = Arc::new(move |request| {
        let runtime = Arc::clone(&runtime);
        let next = Arc::clone(&next);
        Box::pin(async move {
            runtime
                .execute_stream("openai.chat_completions", request, next)
                .await
        })
    });
    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("openai.chat_completions")
            .request(chat_request())
            .func(func)
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| json!({"done": true})))
            .build(),
    )
    .await
    .unwrap();
    while stream.next().await.is_some() {}
    drop(stream);
    flush_subscribers().unwrap();
    deregister_subscriber(&subscriber_name).unwrap();
    Arc::try_unwrap(events).unwrap().into_inner().unwrap()
}

#[tokio::test]
async fn buffered_accounting_records_only_the_terminal_committed_route() {
    let (url, _) = decision_server().await;
    let successful = managed_buffered_events(
        SwitchyardRuntime::new(config(url)).unwrap(),
        Arc::new(|_| Box::pin(async { Ok(chat_response()) })),
    )
    .await;
    let marks = successful
        .iter()
        .filter(|event| event.name() == "nemo_relay.llm.optimization")
        .collect::<Vec<_>>();
    assert_eq!(marks.len(), 1);
    assert_eq!(marks[0].data().unwrap()["applied"], true);
    assert_eq!(
        marks[0].data().unwrap()["model_transition"]["baseline"]["model"],
        "baseline"
    );
    let summary = successful
        .iter()
        .find_map(|event| {
            event
                .annotated_response()
                .and_then(|response| response.optimization_summary.as_ref())
        })
        .unwrap();
    assert_eq!(summary.contributions.len(), 1);
    assert!(summary.contributions[0].applied);

    let (url, _) = decision_server().await;
    let dispatches = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&dispatches);
    let fallback = managed_buffered_events(
        SwitchyardRuntime::new(config(url)).unwrap(),
        Arc::new(move |_| {
            let seen = Arc::clone(&seen);
            Box::pin(async move {
                if seen.fetch_add(1, Ordering::SeqCst) == 0 {
                    Err(FlowError::Upstream(UpstreamFailure {
                        status: Some(401),
                        body: "unauthorized".into(),
                        headers: BTreeMap::new(),
                        class: UpstreamFailureClass::Authentication,
                    }))
                } else {
                    Ok(chat_response())
                }
            })
        }),
    )
    .await;
    assert_eq!(dispatches.load(Ordering::SeqCst), 2);
    assert!(
        fallback
            .iter()
            .all(|event| event.name() != "nemo_relay.llm.optimization")
    );
}

#[tokio::test]
async fn retry_then_success_records_one_terminal_route_with_matching_mark_identity() {
    let (url, decisions) = decision_server().await;
    let dispatches = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&dispatches);
    let events = managed_buffered_events(
        SwitchyardRuntime::new(config(url)).unwrap(),
        Arc::new(move |_| {
            let seen = Arc::clone(&seen);
            Box::pin(async move {
                if seen.fetch_add(1, Ordering::SeqCst) == 0 {
                    Err(FlowError::Upstream(UpstreamFailure {
                        status: Some(503),
                        body: "retry once".into(),
                        headers: BTreeMap::new(),
                        class: UpstreamFailureClass::RetryableStatus,
                    }))
                } else {
                    Ok(chat_response())
                }
            })
        }),
    )
    .await;

    assert_eq!(dispatches.load(Ordering::SeqCst), 2);
    let requests = decisions.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].attempt.routing_attempt, 2);
    drop(requests);

    let start = events
        .iter()
        .find(|event| {
            event.name() == "openai.chat_completions"
                && event.scope_category() == Some(ScopeCategory::Start)
        })
        .unwrap();
    let marks = events
        .iter()
        .filter(|event| event.name() == "nemo_relay.llm.optimization")
        .collect::<Vec<_>>();
    assert_eq!(marks.len(), 1);
    assert_eq!(marks[0].parent_uuid(), Some(start.uuid()));
    assert_eq!(marks[0].data().unwrap()["payload"]["routing_attempt"], 2);

    let summary = events
        .iter()
        .find_map(|event| {
            event
                .annotated_response()
                .and_then(|response| response.optimization_summary.as_ref())
        })
        .unwrap();
    assert_eq!(summary.contributions.len(), 1);
    let contribution = &summary.contributions[0];
    assert!(contribution.applied);
    assert_eq!(contribution.payload.as_ref().unwrap()["routing_attempt"], 2);
    assert_eq!(marks[0].data().unwrap()["id"], json!(contribution.id));
    assert_eq!(
        marks[0].data().unwrap()["sequence"],
        json!(contribution.sequence)
    );
}

#[tokio::test]
async fn oversized_decision_metadata_limits_accounting_without_failing_provider_success() {
    let mut oversized = decision();
    oversized
        .metadata
        .insert("oversized_evidence".into(), json!("x".repeat(20_000)));
    let (url, _) = decision_server_for(oversized).await;
    let dispatches = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&dispatches);
    let events = managed_buffered_events(
        SwitchyardRuntime::new(config(url)).unwrap(),
        Arc::new(move |_| {
            let seen = Arc::clone(&seen);
            Box::pin(async move {
                seen.fetch_add(1, Ordering::SeqCst);
                Ok(chat_response())
            })
        }),
    )
    .await;

    assert_eq!(dispatches.load(Ordering::SeqCst), 1);
    assert!(
        events
            .iter()
            .all(|event| event.name() != "nemo_relay.llm.optimization")
    );
    let summary = events
        .iter()
        .find_map(|event| {
            event
                .annotated_response()
                .and_then(|response| response.optimization_summary.as_ref())
        })
        .unwrap();
    assert_eq!(summary.status, LlmOptimizationSummaryStatus::Partial);
    assert!(summary.contributions.is_empty());
    assert!(
        summary
            .limitations
            .iter()
            .any(|limitation| limitation == "contribution_limit_exceeded")
    );
}

#[tokio::test]
async fn malformed_baseline_is_telemetry_only_and_does_not_change_the_selected_route() {
    let mut drifted = decision();
    drifted.baseline_route.as_mut().unwrap().target_model = "drifted".into();
    let (url, _) = decision_server_for(drifted).await;
    let events = managed_buffered_events(
        SwitchyardRuntime::new(config(url)).unwrap(),
        Arc::new(|request| {
            Box::pin(async move {
                assert_eq!(request.content["model"], "selected");
                Ok(chat_response())
            })
        }),
    )
    .await;
    assert!(events.iter().any(|event| {
        event.name() == "switchyard.routing.error"
            && event.data().is_some_and(|data| {
                data["error_class"] == "baseline_binding"
                    && data["error"]
                        .as_str()
                        .is_some_and(|error| error.contains("exact Relay binding"))
            })
    }));
    assert!(
        events
            .iter()
            .all(|event| event.name() != "nemo_relay.llm.optimization")
    );
}

#[tokio::test]
async fn observe_only_accounting_is_visible_but_not_applied() {
    let (url, _) = decision_server().await;
    let mut observe = config(url);
    observe.mode = RoutingMode::ObserveOnly;
    let events = managed_buffered_events(
        SwitchyardRuntime::new(observe).unwrap(),
        Arc::new(|request| {
            Box::pin(async move {
                assert_eq!(request.content["model"], "fallback");
                Ok(chat_response())
            })
        }),
    )
    .await;
    let contribution = events
        .iter()
        .find(|event| event.name() == "nemo_relay.llm.optimization")
        .and_then(Event::data)
        .unwrap();
    assert_eq!(contribution["applied"], false);
    let summary = events
        .iter()
        .find_map(|event| {
            event
                .annotated_response()
                .and_then(|response| response.optimization_summary.as_ref())
        })
        .unwrap();
    assert_eq!(summary.contributions.len(), 1);
    assert!(!summary.contributions[0].applied);
    assert!(summary.tokens_saved.total_tokens.is_none());
}

#[tokio::test]
async fn streaming_accounting_commits_on_the_first_successful_item_only() {
    let (url, _) = decision_server().await;
    let committed = managed_stream_events(
        SwitchyardRuntime::new(config(url)).unwrap(),
        Arc::new(|_| {
            Box::pin(async {
                Ok(Box::pin(futures_stream::iter(vec![Ok(chat_chunk(
                    "ok",
                    json!("stop"),
                ))])) as LlmJsonStream)
            })
        }),
    )
    .await;
    assert_eq!(
        committed
            .iter()
            .filter(|event| event.name() == "nemo_relay.llm.optimization")
            .count(),
        1
    );

    let (url, _) = decision_server().await;
    let dispatches = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&dispatches);
    let fallback = managed_stream_events(
        SwitchyardRuntime::new(config(url)).unwrap(),
        Arc::new(move |_| {
            let seen = Arc::clone(&seen);
            Box::pin(async move {
                let items = if seen.fetch_add(1, Ordering::SeqCst) == 0 {
                    vec![Err(FlowError::Upstream(UpstreamFailure {
                        status: Some(401),
                        body: "unauthorized".into(),
                        headers: BTreeMap::new(),
                        class: UpstreamFailureClass::Authentication,
                    }))]
                } else {
                    vec![Ok(chat_chunk("fallback", json!("stop")))]
                };
                Ok(Box::pin(futures_stream::iter(items)) as LlmJsonStream)
            })
        }),
    )
    .await;
    assert_eq!(dispatches.load(Ordering::SeqCst), 2);
    assert!(
        fallback
            .iter()
            .all(|event| event.name() != "nemo_relay.llm.optimization")
    );
}

#[tokio::test]
async fn committed_stream_error_keeps_one_route_and_never_redispatches() {
    let (url, decisions) = decision_server().await;
    let dispatches = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&dispatches);
    let events = managed_stream_events(
        SwitchyardRuntime::new(config(url)).unwrap(),
        Arc::new(move |_| {
            let seen = Arc::clone(&seen);
            Box::pin(async move {
                seen.fetch_add(1, Ordering::SeqCst);
                Ok(Box::pin(futures_stream::iter(vec![
                    Ok(chat_chunk("partial", Json::Null)),
                    Err(FlowError::Upstream(UpstreamFailure {
                        status: None,
                        body: "connection closed after commit".into(),
                        headers: BTreeMap::new(),
                        class: UpstreamFailureClass::Connection,
                    })),
                ])) as LlmJsonStream)
            })
        }),
    )
    .await;

    assert_eq!(dispatches.load(Ordering::SeqCst), 1);
    assert_eq!(decisions.lock().unwrap().len(), 1);
    assert_eq!(
        events
            .iter()
            .filter(|event| event.name() == "nemo_relay.llm.optimization")
            .count(),
        1
    );
    let summary = events
        .iter()
        .find_map(|event| {
            event
                .annotated_response()
                .and_then(|response| response.optimization_summary.as_ref())
        })
        .unwrap();
    assert_eq!(summary.contributions.len(), 1);
    assert!(summary.contributions[0].applied);
    assert!(
        summary
            .limitations
            .iter()
            .any(|limitation| limitation == "stream_interrupted")
    );
}

#[tokio::test]
async fn retry_exhaustion_redecides_four_times_then_dispatches_fallback_once() {
    let (url, decisions) = decision_server().await;
    let runtime = SwitchyardRuntime::new(config(url)).unwrap();
    let dispatches = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&dispatches);
    let next: LlmExecutionNextFn = Arc::new(move |request| {
        let seen = Arc::clone(&seen);
        Box::pin(async move {
            let attempt = seen.fetch_add(1, Ordering::SeqCst) + 1;
            if attempt <= 4 {
                return Err(FlowError::Upstream(UpstreamFailure {
                    status: Some(503),
                    body: "temporarily unavailable".into(),
                    headers: BTreeMap::new(),
                    class: UpstreamFailureClass::RetryableStatus,
                }));
            }
            assert_eq!(request.content["model"], "fallback");
            Ok(chat_response())
        })
    });
    let response = runtime
        .execute_buffered("openai.chat_completions", chat_request(), next)
        .await
        .unwrap();
    assert_eq!(response["choices"][0]["message"]["content"], "ok");
    assert_eq!(dispatches.load(Ordering::SeqCst), 5);
    let requests = decisions.lock().unwrap();
    assert_eq!(requests.len(), 4);
    assert_eq!(requests[3].attempt.routing_attempt, 4);
    assert_eq!(
        requests[3].attempt.previous_route.as_deref(),
        Some("selected-chat")
    );
}

#[tokio::test]
async fn non_retryable_provider_failure_bypasses_retry_loop() {
    let (url, decisions) = decision_server().await;
    let runtime = SwitchyardRuntime::new(config(url)).unwrap();
    let dispatches = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&dispatches);
    let next: LlmExecutionNextFn = Arc::new(move |_| {
        let seen = Arc::clone(&seen);
        Box::pin(async move {
            let attempt = seen.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                return Err(FlowError::Upstream(UpstreamFailure {
                    status: Some(401),
                    body: "unauthorized".into(),
                    headers: BTreeMap::new(),
                    class: UpstreamFailureClass::Authentication,
                }));
            }
            Ok(chat_response())
        })
    });
    runtime
        .execute_buffered("openai.chat_completions", chat_request(), next)
        .await
        .unwrap();
    assert_eq!(dispatches.load(Ordering::SeqCst), 2);
    assert_eq!(decisions.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn observe_only_records_one_decision_and_dispatches_only_the_trusted_default() {
    let (url, decisions) = decision_server().await;
    let mut config = config(url);
    config.mode = RoutingMode::ObserveOnly;
    let runtime = SwitchyardRuntime::new(config).unwrap();
    let dispatches = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&dispatches);
    let next: LlmExecutionNextFn = Arc::new(move |request| {
        let seen = Arc::clone(&seen);
        Box::pin(async move {
            seen.fetch_add(1, Ordering::SeqCst);
            assert_eq!(request.content["model"], "fallback");
            Ok(chat_response())
        })
    });
    runtime
        .execute_buffered("openai.chat_completions", chat_request(), next)
        .await
        .unwrap();
    assert_eq!(dispatches.load(Ordering::SeqCst), 1);
    assert_eq!(decisions.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn streaming_retries_before_first_item() {
    let (url, decisions) = decision_server().await;
    let runtime = SwitchyardRuntime::new(config(url)).unwrap();
    let dispatches = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&dispatches);
    let next: LlmStreamExecutionNextFn = Arc::new(move |_| {
        let seen = Arc::clone(&seen);
        Box::pin(async move {
            let attempt = seen.fetch_add(1, Ordering::SeqCst);
            let items = if attempt == 0 {
                vec![Err(FlowError::Upstream(UpstreamFailure {
                    status: Some(503),
                    body: "retry".into(),
                    headers: BTreeMap::new(),
                    class: UpstreamFailureClass::RetryableStatus,
                }))]
            } else {
                vec![Ok(chat_chunk("ok", json!("stop")))]
            };
            Ok(Box::pin(futures_stream::iter(items)) as LlmJsonStream)
        })
    });
    let stream = runtime
        .execute_stream("openai.chat_completions", chat_request(), next)
        .await
        .unwrap();
    let output = stream.collect::<Vec<_>>().await;
    assert_eq!(output.len(), 1);
    assert!(output[0].is_ok());
    assert_eq!(dispatches.load(Ordering::SeqCst), 2);
    assert_eq!(decisions.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn streaming_never_retries_after_first_item() {
    let (url, decisions) = decision_server().await;
    let runtime = SwitchyardRuntime::new(config(url)).unwrap();
    let dispatches = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&dispatches);
    let next: LlmStreamExecutionNextFn = Arc::new(move |_| {
        let seen = Arc::clone(&seen);
        Box::pin(async move {
            seen.fetch_add(1, Ordering::SeqCst);
            let items = vec![
                Ok(chat_chunk("partial", Json::Null)),
                Err(FlowError::Upstream(UpstreamFailure {
                    status: None,
                    body: "connection closed".into(),
                    headers: BTreeMap::new(),
                    class: UpstreamFailureClass::Connection,
                })),
            ];
            Ok(Box::pin(futures_stream::iter(items)) as LlmJsonStream)
        })
    });
    let stream = runtime
        .execute_stream("openai.chat_completions", chat_request(), next)
        .await
        .unwrap();
    let output = stream.collect::<Vec<_>>().await;
    assert_eq!(output.len(), 2);
    assert!(output[0].is_ok());
    assert!(output[1].is_err());
    assert_eq!(dispatches.load(Ordering::SeqCst), 1);
    assert_eq!(decisions.lock().unwrap().len(), 1);
}

fn responses_config(decision_api_url: String) -> SwitchyardConfig {
    let targets = BTreeMap::from([
        (
            "resp-a".into(),
            binding(WireProtocol::OpenaiResponses, "model-a"),
        ),
        (
            "resp-b".into(),
            binding(WireProtocol::OpenaiResponses, "model-b"),
        ),
    ]);
    SwitchyardConfig {
        decision_api_url,
        decision_profile_id: "stage_router".into(),
        request_materialization: RequestMaterialization::SummaryOnly,
        context_mode: ContextMode::PayloadOnly,
        decision_timeout_millis: 1_000,
        targets,
        default_targets: ProtocolDefaults {
            openai_chat: String::new(),
            openai_responses: "resp-a".into(),
            anthropic_messages: String::new(),
        },
        enabled_inbound_profiles: BTreeSet::from([WireProtocol::OpenaiResponses]),
        ..SwitchyardConfig::default()
    }
}

fn responses_decision() -> RoutingDecision {
    RoutingDecision {
        route: crate::contract::RoutingTarget {
            tier: "efficient".into(),
            target_model: "model-b".into(),
            backend_id: "resp-b".into(),
            target_protocol_profile: "openai_responses".into(),
            target_endpoint: "/v1/responses".into(),
        },
        baseline_route: None,
        ..decision()
    }
}

// When every configured target shares the inbound protocol, no cross-protocol translation can
// occur, so the portability guard is skipped: a non-portable Responses request (real Codex
// extensions) is routed through the Decision API with its provider-specific fields preserved.
#[tokio::test]
async fn same_protocol_targets_route_nonportable_streaming_requests() {
    let (url, decisions) = decision_server_for(responses_decision()).await;
    let runtime = SwitchyardRuntime::new(responses_config(url)).unwrap();
    let mut nonportable = request(WireProtocol::OpenaiResponses);
    nonportable.content["reasoning"] = json!({"effort": "high"});
    nonportable.content["store"] = json!(true);
    let next: LlmStreamExecutionNextFn = Arc::new(|request| {
        Box::pin(async move {
            assert_eq!(request.content["model"], "model-b");
            assert_eq!(request.content["store"], true);
            assert_eq!(request.content["reasoning"]["effort"], "high");
            Ok(Box::pin(futures_stream::iter(vec![Ok(
                json!({"type": "response.output_text.delta", "delta": "ok"}),
            )])) as LlmJsonStream)
        })
    });
    let output = runtime
        .execute_stream("openai.responses", nonportable, next)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(output.len(), 1);
    assert_eq!(decisions.lock().unwrap().len(), 1);
}
