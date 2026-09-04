// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the PII redaction plugin component contract.
#![allow(clippy::await_holding_lock)]

use super::*;
use crate::api::event::{
    BaseEvent, CategoryProfile, DataSchema, Event, EventCategory, EventSanitizeFields,
    LOG_SEVERITY_METADATA_KEY, LogSeverity, METRIC_DATA_SCHEMA_NAME, METRIC_DATA_SCHEMA_VERSION,
    MarkEvent, MetricEnvelope, MetricKind, MetricMeasurement, MetricValueType, ScopeCategory,
    ScopeEvent,
};
use crate::api::llm::{
    LlmCallExecuteParams, LlmCallParams, LlmRequest, LlmStreamCallExecuteParams, llm_call,
    llm_call_execute, llm_stream_call_execute,
};
use crate::api::runtime::{
    BuiltinLlmCodec, LlmCodecIdentity, LlmExecutionNextFn, LlmJsonStream,
    LlmSanitizeRequestContext, LlmSanitizeResponseContext, LlmStreamExecutionNextFn,
    NemoRelayContextState, create_scope_stack, global_context, set_thread_scope_stack,
};
use crate::api::scope::{
    EmitMarkEventParams, EmitMetricEventParams, PopScopeParams, PushScopeParams, ScopeType, event,
    metric, pop_scope, push_scope,
};
use crate::api::subscriber::{deregister_subscriber, register_subscriber};
use crate::api::tool::{
    ToolCallEndParams, ToolCallExecuteParams, ToolCallParams, ToolExecutionResult, tool_call,
    tool_call_end, tool_call_execute,
};
use crate::codec::openai_chat::OpenAIChatCodec;
use crate::codec::openai_responses::OpenAIResponsesCodec;
use crate::codec::request::AnnotatedLlmRequest;
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use crate::plugin::{
    ConfigPolicy, DiagnosticLevel, PluginComponentSpec, PluginConfig, PluginError,
    PluginRegistrationContext, UnsupportedBehavior, clear_plugin_configuration,
    ensure_builtin_plugins_registered, initialize_plugins_exact as initialize_plugins,
    list_plugin_kinds, rollback_registrations, validate_plugin_config,
};
use futures::StreamExt;
use nemo_relay::observability::OpenTelemetryType;
use nemo_relay::observability::atif::{AtifAgentInfo, AtifExporter};
use nemo_relay::observability::atof::{AtofExporter, AtofExporterConfig};
use nemo_relay::observability::otel::{OpenTelemetryConfig, OpenTelemetrySubscriber};
use nemo_relay::observability::otel_metrics::{
    OpenTelemetryMetricConfig, OpenTelemetryMetricSubscriber,
};
use opentelemetry_proto::tonic::collector::metrics::v1::ExportMetricsServiceRequest;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, SdkTracerProvider};
use prost::Message;
use serde_json::json;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

static TEST_LOGGING: Once = Once::new();

fn enable_operational_logs() {
    TEST_LOGGING.call_once(|| {
        let runtime =
            nemo_relay::logging::init_logging(&nemo_relay::logging::LoggingConfig::default())
                .expect("test logging should initialize");
        Box::leak(Box::new(runtime));
    });
}

fn component(config: Json) -> PluginComponentSpec {
    let Json::Object(config) = config else {
        panic!("component config must be an object");
    };
    PluginComponentSpec {
        kind: PII_REDACTION_PLUGIN_KIND.to_string(),
        enabled: true,
        config,
    }
}

fn plugin_config(config: Json) -> PluginConfig {
    PluginConfig {
        version: 1,
        components: vec![component(config)],
        policy: Default::default(),
    }
}

struct CapturedOtlpRequest {
    path: String,
    body: Vec<u8>,
}

fn capture_one_otlp_request(
    listener: std::net::TcpListener,
) -> std::sync::mpsc::Receiver<CapturedOtlpRequest> {
    capture_otlp_requests(listener, 1)
}

fn capture_otlp_requests(
    listener: std::net::TcpListener,
    request_count: usize,
) -> std::sync::mpsc::Receiver<CapturedOtlpRequest> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for _ in 0..request_count {
            let (mut stream, _) = listener.accept().expect("collector should accept request");
            let mut bytes = Vec::new();
            let mut buffer = [0_u8; 4_096];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut buffer).expect("collector should read");
                assert!(count > 0, "collector closed before request headers");
                bytes.extend_from_slice(&buffer[..count]);
                if let Some(offset) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let header_end = offset + 4;
                    let headers = String::from_utf8_lossy(&bytes[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    break (header_end, content_length);
                }
            };
            while bytes.len() < header_end + content_length {
                let count = stream
                    .read(&mut buffer)
                    .expect("collector should read body");
                assert!(count > 0, "collector closed before request body");
                bytes.extend_from_slice(&buffer[..count]);
            }
            let path = String::from_utf8_lossy(&bytes[..header_end])
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .expect("request path")
                .to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .expect("collector should respond");
            sender
                .send(CapturedOtlpRequest {
                    path,
                    body: bytes[header_end..header_end + content_length].to_vec(),
                })
                .expect("capture receiver should remain available");
        }
    });
    receiver
}

#[test]
fn top_level_policy_controls_component_diagnostics() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let component_config = json!({
        "mode": "builtin",
        "input": false,
        "output": false,
        "tool_output": true,
        "builtin": {
            "action": "INVALID_ACTION",
            "target_paths": ["/secret"]
        }
    });

    let mut warn_config = plugin_config(component_config.clone());
    warn_config.policy = ConfigPolicy {
        unsupported_value: UnsupportedBehavior::Warn,
        ..ConfigPolicy::default()
    };
    let warn_report = validate_plugin_config(&warn_config);
    assert!(warn_report.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "pii_redaction.unsupported_value"
            && diagnostic.field.as_deref() == Some("builtin.action")
            && diagnostic.level == DiagnosticLevel::Warning
    }));

    let mut ignored_config = plugin_config(component_config);
    ignored_config.policy = ConfigPolicy {
        unsupported_value: UnsupportedBehavior::Ignore,
        ..ConfigPolicy::default()
    };
    let ignored_report = validate_plugin_config(&ignored_config);
    assert!(
        !ignored_report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "pii_redaction.unsupported_value")
    );

    let mut unknown_field_config = plugin_config(json!({
        "mode": "builtin",
        "input": false,
        "output": false,
        "tool_output": true,
        "builtin": {"action": "remove"},
        "unexpected": true
    }));
    unknown_field_config.policy = ConfigPolicy {
        unknown_field: UnsupportedBehavior::Error,
        ..ConfigPolicy::default()
    };
    let unknown_field_report = validate_plugin_config(&unknown_field_config);
    assert!(unknown_field_report.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "pii_redaction.unknown_field"
            && diagnostic.field.as_deref() == Some("unexpected")
            && diagnostic.level == DiagnosticLevel::Error
    }));
}

fn reset_runtime() {
    enable_operational_logs();
    let _ = clear_plugin_configuration();
    crate::plugins::pii_redaction::component::clear_local_backend_provider().unwrap();
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
    register_pii_redaction_component().unwrap();
}

fn setup_isolated_thread() {
    let stack = create_scope_stack();
    set_thread_scope_stack(stack);
}

fn capture_events(name: &str) -> Arc<Mutex<Vec<Event>>> {
    let events = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    register_subscriber(
        name,
        Arc::new(move |event| sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    events
}

fn captured_events_snapshot(events: &Arc<Mutex<Vec<Event>>>) -> Vec<Event> {
    crate::api::subscriber::flush_subscribers().unwrap();
    events.lock().unwrap().clone()
}

fn noop_openai_chat_exec_fn(response: Json) -> LlmExecutionNextFn {
    Arc::new(move |_req| {
        let response = response.clone();
        Box::pin(async move { Ok(response) })
    })
}

#[test]
fn builtin_registry_includes_pii_redaction_component() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    ensure_builtin_plugins_registered().unwrap();

    let plugin_kinds = list_plugin_kinds();
    assert!(
        plugin_kinds
            .iter()
            .any(|kind| kind == PII_REDACTION_PLUGIN_KIND)
    );
}

#[test]
fn builtin_backend_config_default_matches_documented_action_default() {
    let config = BuiltinBackendConfig::default();

    assert!(config.preset.is_none());
    assert_eq!(config.action, "remove");
    assert_eq!(config.custom_mark_payload_policy, "preserve");
    assert!(config.metric_string_attribute_allowlist.is_empty());
    assert!(config.target_paths.is_empty());
    assert!(config.target_path_globs.is_empty());
    assert!(config.pattern.is_none());
    assert!(config.detector.is_none());
}

#[test]
fn builtin_backend_editor_exposes_metric_string_allowlist_as_json() {
    let schema = <BuiltinBackendConfig as nemo_relay::config_editor::EditorConfig>::editor_schema();
    let field = schema
        .field("metric_string_attribute_allowlist")
        .expect("metric string allowlist should be editable");
    assert_eq!(field.kind, nemo_relay::config_editor::EditorFieldKind::Json);
}

#[test]
fn component_spec_and_plugin_contract_preserve_the_public_configuration_shape() {
    let config = PiiRedactionConfig::default();
    let component = ComponentSpec::new(config.clone());
    assert!(component.enabled);
    assert_eq!(component.config.mode, "builtin");

    let registered: PluginComponentSpec = component.into();
    assert_eq!(registered.kind, PII_REDACTION_PLUGIN_KIND);
    assert!(registered.enabled);
    assert!(registered.config.get("mode").is_none());

    let plugin = PiiRedactionPlugin;
    assert_eq!(plugin.plugin_kind(), PII_REDACTION_PLUGIN_KIND);
    assert!(!plugin.allows_multiple_components());
    assert_eq!(
        plugin.validate(&Map::from_iter([("input".into(), json!("invalid"))]))[0].code,
        "pii_redaction.invalid_plugin_config"
    );
}

#[test]
fn trajectory_preset_validates_without_an_action_and_rejects_matcher_fields() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    let valid = validate_plugin_config(&plugin_config(json!({
        "codec": "openai_chat",
        "profiles": [{
            "mode": "builtin",
            "builtin": {
                "preset": "trajectory_context",
                "custom_mark_payload_policy": "redact_all_leaves"
            }
        }]
    })));
    assert!(!valid.has_errors(), "{:#?}", valid.diagnostics);

    let invalid = validate_plugin_config(&plugin_config(json!({
        "codec": "openai_chat",
        "profiles": [{
            "mode": "builtin",
            "builtin": {
                "preset": "trajectory_context",
                "action": "redact",
                "detector": "email",
                "target_path_globs": ["/messages/*/content"]
            }
        }]
    })));
    assert!(invalid.has_errors());
    assert!(
        invalid.diagnostics.iter().any(|diagnostic| {
            diagnostic.field.as_deref() == Some("profiles[0].builtin.action")
        })
    );
    assert!(
        invalid.diagnostics.iter().any(|diagnostic| {
            diagnostic.field.as_deref() == Some("profiles[0].builtin.detector")
        })
    );
    assert!(invalid.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("profiles[0].builtin.target_path_globs")
    }));

    let policy_without_preset = validate_plugin_config(&plugin_config(json!({
        "codec": "openai_chat",
        "profiles": [{
            "mode": "builtin",
            "builtin": {"custom_mark_payload_policy": "preserve"}
        }]
    })));
    assert!(policy_without_preset.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("profiles[0].builtin.custom_mark_payload_policy")
    }));
}

#[test]
fn trajectory_metric_string_allowlist_validation_is_fail_closed() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    ensure_builtin_plugins_registered().unwrap();

    let valid = validate_plugin_config(&plugin_config(json!({
        "codec": "openai_chat",
        "profiles": [{
            "mode": "builtin",
            "builtin": {
                "preset": "trajectory_context",
                "metric_string_attribute_allowlist": {
                    "query.source": ["main", "subagent"]
                }
            }
        }]
    })));
    assert!(!valid.has_errors(), "{:#?}", valid.diagnostics);

    let valid_legacy = validate_plugin_config(&plugin_config(json!({
        "codec": "openai_chat",
        "mode": "builtin",
        "builtin": {
            "preset": "trajectory_context",
            "metric_string_attribute_allowlist": {"query.source": ["main"]}
        }
    })));
    assert!(
        !valid_legacy.has_errors(),
        "{:#?}",
        valid_legacy.diagnostics
    );

    let without_preset = validate_plugin_config(&plugin_config(json!({
        "codec": "openai_chat",
        "profiles": [{
            "mode": "builtin",
            "builtin": {
                "metric_string_attribute_allowlist": {"query.source": ["main"]}
            }
        }]
    })));
    assert!(
        without_preset.diagnostics.iter().any(|diagnostic| {
            diagnostic.field.as_deref()
                == Some("profiles[0].builtin.metric_string_attribute_allowlist")
                && diagnostic.message.contains("requires builtin.preset")
        }),
        "{:#?}",
        without_preset.diagnostics
    );

    for (allowlist, expected_message) in [
        (json!({" ": ["main"]}), "keys must not be blank"),
        (
            json!({"query.source": []}),
            "must contain at least one value",
        ),
        (json!({"query.source": [" "]}), "must not be blank"),
        (
            json!({"query.source": ["main", "main"]}),
            "contains duplicate value",
        ),
    ] {
        let report = validate_plugin_config(&plugin_config(json!({
            "codec": "openai_chat",
            "profiles": [{
                "mode": "builtin",
                "builtin": {
                    "preset": "trajectory_context",
                    "metric_string_attribute_allowlist": allowlist
                }
            }]
        })));
        assert!(report.diagnostics.iter().any(|diagnostic| {
            diagnostic.field.as_deref()
                == Some("profiles[0].builtin.metric_string_attribute_allowlist")
                && diagnostic.message.contains(expected_message)
        }));
    }

    let runtime_error = crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            metric_string_attribute_allowlist: BTreeMap::from([(
                "query.source".into(),
                vec!["main".into()],
            )]),
            ..BuiltinBackendConfig::default()
        },
        None,
    )
    .err()
    .expect("runtime construction must reject an allowlist without the trajectory preset");
    assert!(
        runtime_error
            .to_string()
            .contains("requires builtin.preset")
    );

    let runtime_error = crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            preset: Some("trajectory_context".into()),
            metric_string_attribute_allowlist: BTreeMap::from([(
                "query.source".into(),
                vec!["main".into(), "main".into()],
            )]),
            ..BuiltinBackendConfig::default()
        },
        None,
    )
    .err()
    .expect("runtime construction must reject invalid allowlist entries");
    assert!(runtime_error.to_string().contains("duplicate value"));
}

fn trajectory_backend(codec: Option<&str>, policy: &str) -> crate::builtin::CompiledBuiltinBackend {
    crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            preset: Some("trajectory_context".into()),
            custom_mark_payload_policy: policy.into(),
            ..BuiltinBackendConfig::default()
        },
        codec.map(str::to_string),
    )
    .unwrap()
}

fn trajectory_backend_with_metric_allowlist(
    allowlist: BTreeMap<String, Vec<String>>,
) -> crate::builtin::CompiledBuiltinBackend {
    crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            preset: Some("trajectory_context".into()),
            metric_string_attribute_allowlist: allowlist,
            ..BuiltinBackendConfig::default()
        },
        None,
    )
    .unwrap()
}

fn no_codec_context() -> LlmSanitizeResponseContext {
    LlmSanitizeResponseContext::default()
}

fn no_codec_request_context() -> LlmSanitizeRequestContext {
    LlmSanitizeRequestContext::default()
}

struct IdentifiedRequestCodec {
    identity: LlmCodecIdentity,
    inner: OpenAIResponsesCodec,
}

impl LlmCodec for IdentifiedRequestCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        self.identity.clone()
    }

    fn decode(&self, request: &LlmRequest) -> nemo_relay::error::Result<AnnotatedLlmRequest> {
        self.inner.decode(request)
    }

    fn encode(
        &self,
        annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> nemo_relay::error::Result<LlmRequest> {
        self.inner.encode(annotated, original)
    }
}

#[tokio::test]
async fn normalized_llm_paths_use_the_active_codec_and_fail_closed_for_unknown_codecs() {
    let backend = crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "regex_replace".to_string(),
            pattern: Some("sk-[A-Za-z0-9_-]+".to_string()),
            replacement: Some("[REDACTED]".to_string()),
            target_paths: vec![
                "/messages/0/content/0/text".to_string(),
                "/message".to_string(),
            ],
            ..BuiltinBackendConfig::default()
        },
        Some("openai_chat".to_string()),
    )
    .unwrap();
    let sanitize_request = crate::builtin::llm_sanitize_request_callback(backend.clone());
    let sanitize_response = crate::builtin::llm_sanitize_response_callback(backend);
    let active_request = sanitize_request(
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({
                "model": "gpt-4.1-mini",
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "sk-request-secret"}]}]
            }),
        },
        LlmSanitizeRequestContext::for_request_codec(Some(Arc::new(OpenAIResponsesCodec))),
    )
    .await
    .expect("the active OpenAI Responses codec must override the legacy fallback")
    .expect("the active OpenAI Responses codec must retain the payload");
    assert_eq!(
        active_request.content["input"][0]["content"][0]["text"],
        json!("[REDACTED]")
    );

    for identity in [
        LlmCodecIdentity::Runtime("com.example.responses.v1".to_owned()),
        LlmCodecIdentity::Opaque,
    ] {
        let active_request = sanitize_request(
            LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({
                    "model": "gpt-4.1-mini",
                    "input": [{
                        "role": "user",
                        "content": [{"type": "input_text", "text": "sk-request-secret"}]
                    }]
                }),
            },
            LlmSanitizeRequestContext::for_request_codec(Some(Arc::new(IdentifiedRequestCodec {
                identity,
                inner: OpenAIResponsesCodec,
            }))),
        )
        .await
        .expect("an active runtime or opaque request codec must remain usable")
        .expect("an active runtime or opaque request codec must retain the payload");
        assert_eq!(
            active_request.content["input"][0]["content"][0]["text"],
            json!("[REDACTED]")
        );
    }

    let responses_payload = json!({
        "id": "resp_123",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "sk-responses-secret"}]
        }]
    });

    let active_responses = sanitize_response(
        responses_payload.clone(),
        LlmSanitizeResponseContext::with_identity(LlmCodecIdentity::BuiltIn(
            BuiltinLlmCodec::OpenAiResponses,
        )),
    )
    .await
    .expect("the active OpenAI Responses codec must override the legacy fallback")
    .expect("the active OpenAI Responses codec must retain the payload");
    assert_eq!(
        active_responses["output"][0]["content"][0]["text"],
        json!("[REDACTED]")
    );

    assert!(
        sanitize_response(responses_payload.clone(), no_codec_context())
            .await
            .expect("sanitizer callback must succeed")
            .is_none(),
        "an incompatible configured fallback codec must omit a normalized payload"
    );

    assert!(
        sanitize_response(
            responses_payload,
            LlmSanitizeResponseContext::with_identity(LlmCodecIdentity::Opaque),
        )
        .await
        .expect("sanitizer callback must succeed")
        .is_none(),
        "a normalized-path policy must omit an unknown active provider payload"
    );

    assert!(
        sanitize_response(
            json!({
                "id": "resp_123",
                "output": [{"content": [{"type": "output_text", "text": "sk-runtime-secret"}]}]
            }),
            LlmSanitizeResponseContext::with_identity(LlmCodecIdentity::Runtime(
                "com.example.chat.v1".to_owned(),
            )),
        )
        .await
        .expect("sanitizer callback must succeed")
        .is_none(),
        "a normalized-path policy must omit a runtime codec until it has a compatible projection"
    );
}

#[tokio::test]
async fn normalized_llm_target_path_globs_sanitize_variable_message_and_response_paths() {
    let request_backend = crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "regex_replace".to_string(),
            pattern: Some("sk-[A-Za-z0-9_-]+".to_string()),
            replacement: Some("[REDACTED]".to_string()),
            target_path_globs: vec!["/messages/*/content/*/text".to_string()],
            ..BuiltinBackendConfig::default()
        },
        Some("openai_responses".to_string()),
    )
    .unwrap();
    let sanitize_request = crate::builtin::llm_sanitize_request_callback(request_backend);
    let response_backend = crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "regex_replace".to_string(),
            pattern: Some("sk-[A-Za-z0-9_-]+".to_string()),
            replacement: Some("[REDACTED]".to_string()),
            target_path_globs: vec!["/message".to_string()],
            ..BuiltinBackendConfig::default()
        },
        Some("openai_responses".to_string()),
    )
    .unwrap();
    let sanitize_response = crate::builtin::llm_sanitize_response_callback(response_backend);

    let request = sanitize_request(
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({
                "model": "gpt-4.1-mini",
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": "sk-first-secret"}]},
                    {"role": "user", "content": [{"type": "input_text", "text": "sk-second-secret"}]}
                ]
            }),
        },
        LlmSanitizeRequestContext::for_request_codec(Some(Arc::new(OpenAIResponsesCodec))),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        request.content["input"][0]["content"][0]["text"],
        json!("[REDACTED]")
    );
    assert_eq!(
        request.content["input"][1]["content"][0]["text"],
        json!("[REDACTED]")
    );

    let response = sanitize_response(
        json!({
            "id": "resp_123",
            "model": "gpt-4.1-mini",
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "sk-response-secret"}]
            }]
        }),
        LlmSanitizeResponseContext::with_identity(LlmCodecIdentity::BuiltIn(
            BuiltinLlmCodec::OpenAiResponses,
        )),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        response["output"][0]["content"][0]["text"],
        json!("[REDACTED]")
    );
}

#[tokio::test]
async fn normalized_llm_response_paths_use_the_active_oci_genai_codec_identity() {
    let backend = crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "regex_replace".to_string(),
            pattern: Some("sk-[A-Za-z0-9_-]+".to_string()),
            replacement: Some("[REDACTED]".to_string()),
            target_paths: vec!["/message".to_string()],
            ..BuiltinBackendConfig::default()
        },
        None,
    )
    .unwrap();
    let sanitize_response = crate::builtin::llm_sanitize_response_callback(backend);

    let sanitized = sanitize_response(
        json!({
            "modelId": "meta.llama-3.3-70b-instruct",
            "chatResponse": {
                "apiFormat": "GENERIC",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "ASSISTANT",
                        "content": [{"type": "TEXT", "text": "sk-oci-secret"}]
                    },
                    "finishReason": "stop"
                }]
            }
        }),
        LlmSanitizeResponseContext::with_identity(LlmCodecIdentity::BuiltIn(
            BuiltinLlmCodec::OCIGenAI,
        )),
    )
    .await
    .expect("sanitizer callback must succeed")
    .expect("the active OCI GenAI codec identity must retain the payload");
    assert_eq!(
        sanitized["chatResponse"]["choices"][0]["message"]["content"][0]["text"],
        json!("[REDACTED]")
    );
}

#[tokio::test]
async fn normalized_llm_paths_omit_payloads_when_legacy_codec_decode_fails() {
    let backend = crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "regex_replace".to_string(),
            pattern: Some("sk-[A-Za-z0-9_-]+".to_string()),
            replacement: Some("[REDACTED]".to_string()),
            target_paths: vec!["/messages/0/content".to_string()],
            ..BuiltinBackendConfig::default()
        },
        Some("openai_chat".to_string()),
    )
    .unwrap();
    let sanitize_request = crate::builtin::llm_sanitize_request_callback(backend.clone());
    let sanitize_response = crate::builtin::llm_sanitize_response_callback(backend);

    assert!(
        sanitize_request(
            LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"messages": "sk-request-secret"}),
            },
            no_codec_request_context(),
        )
        .await
        .expect("sanitizer callback must succeed")
        .is_none(),
        "a shallow legacy surface match must not enable a raw-payload fallback"
    );
    assert!(
        sanitize_response(json!({"choices": "sk-response-secret"}), no_codec_context())
            .await
            .expect("sanitizer callback must succeed")
            .is_none(),
        "a legacy response codec failure must omit the payload instead of emitting raw content"
    );
}

#[tokio::test]
async fn normalized_openai_chat_api_specific_policy_omits_multiple_choices() {
    let backend = crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "remove".to_string(),
            target_paths: vec!["/api_specific".to_string()],
            ..BuiltinBackendConfig::default()
        },
        None,
    )
    .unwrap();
    let sanitize_response = crate::builtin::llm_sanitize_response_callback(backend);

    assert!(
        sanitize_response(
            json!({
                "id": "chatcmpl-multi",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "first"},
                        "logprobs": {"content": [{"token": "SECRET-FIRST"}]}
                    },
                    {
                        "index": 1,
                        "message": {"role": "assistant", "content": "second"},
                        "logprobs": {"content": [{"token": "SECRET-SECOND"}]}
                    }
                ]
            }),
            LlmSanitizeResponseContext::with_identity(LlmCodecIdentity::BuiltIn(
                BuiltinLlmCodec::OpenAiChat,
            )),
        )
        .await
        .expect("sanitizer callback must succeed")
        .is_none()
    );
}

#[tokio::test]
async fn normalized_llm_paths_use_configured_anthropic_codec_without_a_system_message() {
    let backend = crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "regex_replace".to_string(),
            pattern: Some("sk-[A-Za-z0-9_-]+".to_string()),
            replacement: Some("[REDACTED]".to_string()),
            target_paths: vec!["/messages/0/content".to_string()],
            ..BuiltinBackendConfig::default()
        },
        Some("anthropic_messages".to_string()),
    )
    .unwrap();
    let sanitize_request = crate::builtin::llm_sanitize_request_callback(backend);

    let sanitized = sanitize_request(
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "sk-anthropic-secret"}],
            }),
        },
        no_codec_request_context(),
    )
    .await
    .expect("the configured Anthropic codec must sanitize a valid message-only request")
    .expect("the configured Anthropic codec must retain the payload");

    assert_eq!(
        sanitized.content["messages"][0]["content"],
        json!("[REDACTED]")
    );
}

#[tokio::test]
async fn trajectory_preset_redacts_chat_content_without_erasing_request_structure() {
    let callback = crate::builtin::llm_sanitize_request_callback(trajectory_backend(
        Some("openai_chat"),
        "preserve",
    ));
    let request = callback(LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "system", "content": "private system prompt"},
                {"role": "user", "content": [
                    {"type": "text", "text": "private user prompt"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,secret", "detail": "high"}}
                ]},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{\"query\":\"private query\",\"limit\":5}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "private result"}
            ],
            "tools": [{"type": "function", "function": {
                "name": "search",
                "description": "private description",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string", "description": "private schema text", "default": "private default"}
                }, "required": ["query"]}
            }}],
            "temperature": 0.2,
            "stop": ["private stop sequence"],
            "participant": {"name": "Alice Example", "username": "alice"},
            "person_name": "Alice Example"
        }),
    }, no_codec_request_context())
    .await
    .unwrap()
    .unwrap();

    assert_eq!(request.content["model"], "claude-sonnet-4-6");
    assert_eq!(request.content["temperature"], 0.2);
    assert_eq!(request.content["stop"][0], "[REDACTED]");
    assert_eq!(request.content["participant"]["name"], "[REDACTED]");
    assert_eq!(request.content["participant"]["username"], "[REDACTED]");
    assert_eq!(request.content["person_name"], "[REDACTED]");
    assert_eq!(request.content["messages"][0]["role"], "system");
    assert_eq!(request.content["messages"][0]["content"], "[REDACTED]");
    assert_eq!(request.content["messages"][1]["content"][0]["type"], "text");
    assert_eq!(
        request.content["messages"][1]["content"][0]["text"],
        "[REDACTED]"
    );
    assert_eq!(
        request.content["messages"][1]["content"][1]["image_url"]["url"],
        "[REDACTED]"
    );
    assert_eq!(
        request.content["messages"][2]["tool_calls"][0]["id"],
        "call_1"
    );
    assert_eq!(
        request.content["messages"][2]["tool_calls"][0]["function"]["name"],
        "search"
    );
    let arguments: Json = serde_json::from_str(
        request.content["messages"][2]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(arguments, json!({"query": "[REDACTED]", "limit": 0}));
    assert_eq!(request.content["messages"][3]["tool_call_id"], "call_1");
    assert_eq!(request.content["messages"][3]["content"], "[REDACTED]");
    assert_eq!(request.content["tools"][0]["function"]["name"], "search");
    assert_eq!(
        request.content["tools"][0]["function"]["description"],
        "[REDACTED]"
    );
    assert_eq!(
        request.content["tools"][0]["function"]["parameters"]["required"][0],
        "query"
    );
}

#[tokio::test]
async fn trajectory_preset_preserves_response_analytics_and_redacts_response_content() {
    let callback = crate::builtin::llm_sanitize_response_callback(trajectory_backend(
        Some("openai_chat"),
        "preserve",
    ));
    let sanitized = callback(
        json!({
            "id": "chatcmpl_1",
            "model": "claude-opus-4-6",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant",
                "content": "private answer",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": "terminal", "arguments": "{\"command\":\"cat secret.txt\"}"
                }}]
            }, "logprobs": {"content": [{"token": "secret", "logprob": -0.5}]}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            "cost": {"total": 1.25}
        }),
        no_codec_context(),
    )
    .await
    .unwrap()
    .unwrap();

    assert_eq!(sanitized["id"], "chatcmpl_1");
    assert_eq!(sanitized["model"], "claude-opus-4-6");
    assert_eq!(sanitized["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(sanitized["choices"][0]["message"]["role"], "assistant");
    assert_eq!(sanitized["choices"][0]["message"]["content"], "[REDACTED]");
    assert_eq!(
        sanitized["choices"][0]["message"]["tool_calls"][0]["id"],
        "call_1"
    );
    assert_eq!(
        sanitized["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "terminal"
    );
    let arguments: Json = serde_json::from_str(
        sanitized["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(arguments, json!({"command": "[REDACTED]"}));
    assert_eq!(sanitized["usage"]["total_tokens"], 25);
    assert_eq!(sanitized["cost"]["total"], 1.25);
}

#[tokio::test]
async fn trajectory_preset_covers_responses_and_anthropic_provider_shapes() {
    let responses_request = crate::builtin::llm_sanitize_request_callback(trajectory_backend(
        Some("openai_responses"),
        "preserve",
    ))(LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-5",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "private input"}]}],
            "reasoning": {"effort": "high", "summary": "private reasoning"},
            "max_output_tokens": 100
        }),
    }, no_codec_request_context())
    .await
    .unwrap()
    .unwrap();
    assert_eq!(responses_request.content["model"], "gpt-5");
    assert_eq!(responses_request.content["input"][0]["role"], "user");
    assert_eq!(
        responses_request.content["input"][0]["content"][0]["text"],
        "[REDACTED]"
    );
    assert_eq!(responses_request.content["max_output_tokens"], 100);

    let responses_response = crate::builtin::llm_sanitize_response_callback(trajectory_backend(
        Some("openai_responses"),
        "preserve",
    ))(
        json!({
            "id": "resp_1",
            "model": "gpt-5",
            "status": "completed",
            "output": [{"id": "msg_1", "type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "private output"}
            ]}],
            "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
        }),
        no_codec_context(),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(responses_response["id"], "resp_1");
    assert_eq!(responses_response["status"], "completed");
    assert_eq!(responses_response["output"][0]["id"], "msg_1");
    assert_eq!(
        responses_response["output"][0]["content"][0]["text"],
        "[REDACTED]"
    );
    assert_eq!(responses_response["usage"]["total_tokens"], 14);

    let anthropic_request = crate::builtin::llm_sanitize_request_callback(trajectory_backend(
        Some("anthropic_messages"),
        "preserve",
    ))(LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "claude-sonnet-4-6",
            "system": "private system",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "private user"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "private-base64"}}
            ]}],
            "max_tokens": 128
        }),
    }, no_codec_request_context())
    .await
    .unwrap()
    .unwrap();
    assert_eq!(anthropic_request.content["model"], "claude-sonnet-4-6");
    assert_eq!(anthropic_request.content["system"], "[REDACTED]");
    assert_eq!(
        anthropic_request.content["messages"][0]["content"][0]["text"], "[REDACTED]",
        "{}",
        anthropic_request.content
    );
    assert_eq!(
        anthropic_request.content["messages"][0]["content"][1]["source"]["data"],
        "[REDACTED]"
    );
    assert_eq!(anthropic_request.content["max_tokens"], 128);

    let anthropic_response = crate::builtin::llm_sanitize_response_callback(trajectory_backend(
        Some("anthropic_messages"),
        "preserve",
    ))(json!({
        "id": "msg_1",
        "model": "claude-sonnet-4-6",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "private chain of thought", "signature": "private-signature"},
            {"type": "text", "text": "private answer"}
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 6, "cache_read_input_tokens": 8}
    }), no_codec_context())
    .await
    .unwrap()
    .unwrap();
    assert_eq!(anthropic_response["id"], "msg_1");
    assert_eq!(anthropic_response["role"], "assistant");
    assert_eq!(anthropic_response["content"][0]["type"], "thinking");
    assert_eq!(anthropic_response["content"][0]["thinking"], "[REDACTED]");
    assert_eq!(anthropic_response["content"][1]["text"], "[REDACTED]");
    assert_eq!(anthropic_response["usage"]["input_tokens"], 12);
    assert_eq!(anthropic_response["usage"]["cache_read_input_tokens"], 8);
}

#[tokio::test]
async fn trajectory_preset_redacts_known_marks_and_nested_scope_content() {
    let callback = crate::builtin::event_sanitize_callback(trajectory_backend(None, "preserve"));
    let chunk = Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("llm.chunk").build(),
        Some(EventCategory::custom()),
        Some(CategoryProfile::builder().subtype("llm.chunk").build()),
    ));
    let sanitized = callback(
        Arc::new(chunk.clone()),
        EventSanitizeFields {
            data: Some(json!({
                "chunk_index": 2,
                "event_type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "private delta"}
            })),
            category_profile: chunk.category_profile().cloned(),
            metadata: None,
        },
    )
    .await
    .unwrap();
    assert_eq!(sanitized.data.as_ref().unwrap()["chunk_index"], 2);
    assert_eq!(
        sanitized.data.as_ref().unwrap()["event_type"],
        "content_block_delta"
    );
    assert_eq!(
        sanitized.data.as_ref().unwrap()["delta"]["text"],
        "[REDACTED]"
    );

    let optimization = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("nemo_relay.llm.optimization")
            .build(),
        Some(EventCategory::custom()),
        Some(
            CategoryProfile::builder()
                .subtype("nemo_relay.llm.optimization")
                .build(),
        ),
    ));
    let sanitized = callback(
        Arc::new(optimization.clone()),
        EventSanitizeFields {
            data: Some(json!({
                "producer": "neutral.router",
                "kind": "model_routing",
                "applied": true,
                "model_transition": {
                    "baseline": {"model": "claude-opus-4-6"},
                    "effective": {"model": "claude-sonnet-4-6"}
                },
                "token_impact": {"saved": {"prompt_tokens": 40, "total_tokens": 40}},
                "payload": {"private_excerpt": "private content"}
            })),
            category_profile: optimization.category_profile().cloned(),
            metadata: None,
        },
    )
    .await
    .unwrap();
    assert_eq!(
        sanitized.data.as_ref().unwrap()["producer"],
        "neutral.router"
    );
    assert_eq!(
        sanitized.data.as_ref().unwrap()["model_transition"]["baseline"]["model"],
        "claude-opus-4-6"
    );
    assert_eq!(
        sanitized.data.as_ref().unwrap()["token_impact"]["saved"]["total_tokens"],
        40
    );
    assert_eq!(
        sanitized.data.as_ref().unwrap()["payload"]["private_excerpt"],
        "[REDACTED]"
    );

    let nested_agent = Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("worker-agent").build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::agent(),
        None,
    ));
    let sanitized = callback(
        Arc::new(nested_agent),
        EventSanitizeFields {
            data: Some(json!({
                "request_id": "request-1",
                "instruction": "private delegated task",
                "history": [{"role": "user", "content": "private history"}]
            })),
            category_profile: None,
            metadata: Some(json!({"parent_scope_id": "scope-1", "note": "private note"})),
        },
    )
    .await
    .unwrap();
    assert_eq!(sanitized.data.as_ref().unwrap()["request_id"], "request-1");
    assert_eq!(
        sanitized.data.as_ref().unwrap()["instruction"],
        "[REDACTED]"
    );
    assert_eq!(
        sanitized.data.as_ref().unwrap()["history"][0]["role"],
        "user"
    );
    assert_eq!(
        sanitized.data.as_ref().unwrap()["history"][0]["content"],
        "[REDACTED]"
    );
    assert_eq!(
        sanitized.metadata.as_ref().unwrap()["parent_scope_id"],
        "scope-1"
    );
    assert_eq!(sanitized.metadata.as_ref().unwrap()["note"], "[REDACTED]");
}

#[tokio::test]
async fn trajectory_preset_preserves_trusted_scope_metadata_only() {
    let callback = crate::builtin::event_sanitize_callback(trajectory_backend(None, "preserve"));
    let metadata = json!({
        "nemo_relay_scope_role": "turn",
        "agent_kind": "codex",
        "hook_event_name": "UserPromptSubmit",
        "gateway_config_profile": "development",
        "gateway_mode": "passthrough",
        "turn_source": "user_prompt",
        "harness": "codex",
        "source": "hook",
        "identity_quality": "native",
        "gateway_path": "responses",
        "llm_correlation_status": "matched",
        "llm_correlation_source": "provider",
        "tool_correlation_status": "matched",
        "tool_correlation_source": "provider",
        "otel.status_code": "OK",
        "fidelity_source": "provider",
        "provider_payload_exact": true,
        "private_note": "private context",
        "nested": {"harness": "private nested context"}
    });
    let expected_metadata = json!({
        "nemo_relay_scope_role": "turn",
        "agent_kind": "codex",
        "hook_event_name": "UserPromptSubmit",
        "gateway_config_profile": "development",
        "gateway_mode": "passthrough",
        "turn_source": "user_prompt",
        "harness": "codex",
        "source": "hook",
        "identity_quality": "native",
        "gateway_path": "responses",
        "llm_correlation_status": "matched",
        "llm_correlation_source": "provider",
        "tool_correlation_status": "matched",
        "tool_correlation_source": "provider",
        "otel.status_code": "OK",
        "fidelity_source": "provider",
        "provider_payload_exact": true,
        "private_note": "[REDACTED]",
        "nested": {"harness": "[REDACTED]"}
    });

    for (scope_category, category) in [
        (ScopeCategory::Start, EventCategory::agent()),
        (ScopeCategory::End, EventCategory::agent()),
        (ScopeCategory::Start, EventCategory::llm()),
        (ScopeCategory::End, EventCategory::llm()),
        (ScopeCategory::Start, EventCategory::tool()),
        (ScopeCategory::End, EventCategory::tool()),
    ] {
        let event = Event::Scope(ScopeEvent::new(
            BaseEvent::builder().name("trusted-scope").build(),
            scope_category,
            Vec::new(),
            category,
            None,
        ));
        let sanitized = callback(
            Arc::new(event),
            EventSanitizeFields {
                data: None,
                category_profile: None,
                metadata: Some(metadata.clone()),
            },
        )
        .await
        .unwrap();
        assert_eq!(sanitized.metadata, Some(expected_metadata.clone()));
    }

    let malformed = Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("malformed-trusted-scope").build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::agent(),
        None,
    ));
    let sanitized = callback(
        Arc::new(malformed),
        EventSanitizeFields {
            data: None,
            category_profile: None,
            metadata: Some(json!({
                "harness": {"private": "private context"},
                "source": 42,
                "identity_quality": ["private context"],
                "provider_payload_exact": "private context"
            })),
        },
    )
    .await
    .unwrap();
    assert_eq!(
        sanitized.metadata,
        Some(json!({
            "harness": {"private": "[REDACTED]"},
            "source": 0,
            "identity_quality": ["[REDACTED]"],
            "provider_payload_exact": "[REDACTED]"
        }))
    );

    let mark = Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("llm.chunk").build(),
        Some(EventCategory::custom()),
        Some(CategoryProfile::builder().subtype("llm.chunk").build()),
    ));
    let sanitized = callback(
        Arc::new(mark.clone()),
        EventSanitizeFields {
            data: None,
            category_profile: mark.category_profile().cloned(),
            metadata: Some(json!({"harness": "codex", "source": "hook"})),
        },
    )
    .await
    .unwrap();
    assert_eq!(
        sanitized.metadata,
        Some(json!({"harness": "[REDACTED]", "source": "[REDACTED]"}))
    );
}

#[tokio::test]
async fn trajectory_custom_mark_policy_is_explicit_and_shape_preserving() {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("neutral.plugin.evidence").build(),
        Some(EventCategory::custom()),
        Some(CategoryProfile {
            subtype: Some("neutral.plugin".into()),
            extra: BTreeMap::from([("opaque".into(), json!({"label": "private"}))]),
            ..CategoryProfile::default()
        }),
    ));
    let fields = EventSanitizeFields {
        data: Some(json!({"text": "private", "score": 0.75, "nested": [true, null]})),
        category_profile: event.category_profile().cloned(),
        metadata: Some(json!({"owner": "private owner"})),
    };

    let preserve = crate::builtin::event_sanitize_callback(trajectory_backend(None, "preserve"));
    assert_eq!(
        preserve(Arc::new(event.clone()), fields.clone())
            .await
            .unwrap(),
        fields
    );

    let redact =
        crate::builtin::event_sanitize_callback(trajectory_backend(None, "redact_all_leaves"));
    let sanitized = redact(Arc::new(event), fields).await.unwrap();
    assert_eq!(
        sanitized.data.unwrap(),
        json!({
            "text": "[REDACTED]", "score": 0, "nested": [false, null]
        })
    );
    assert_eq!(sanitized.metadata.unwrap(), json!({"owner": "[REDACTED]"}));
    let profile = sanitized.category_profile.unwrap();
    assert_eq!(profile.subtype.as_deref(), Some("neutral.plugin"));
    assert_eq!(profile.extra["opaque"]["label"], "[REDACTED]");
}

#[tokio::test]
async fn trajectory_custom_mark_preserves_only_valid_log_severity() {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("neutral.plugin.log").build(),
        Some(EventCategory::custom()),
        None,
    ));
    let callback =
        crate::builtin::event_sanitize_callback(trajectory_backend(None, "redact_all_leaves"));

    let sanitized = callback(
        Arc::new(event.clone()),
        EventSanitizeFields {
            data: Some(json!({"message": "private"})),
            category_profile: None,
            metadata: Some(json!({
                LOG_SEVERITY_METADATA_KEY: "warning",
                "owner": "private owner"
            })),
        },
    )
    .await
    .unwrap();
    assert_eq!(sanitized.data, Some(json!({"message": "[REDACTED]"})));
    assert_eq!(
        sanitized.metadata,
        Some(json!({
            LOG_SEVERITY_METADATA_KEY: "warn",
            "owner": "[REDACTED]"
        }))
    );

    let sanitized = callback(
        Arc::new(event),
        EventSanitizeFields {
            data: None,
            category_profile: None,
            metadata: Some(json!({LOG_SEVERITY_METADATA_KEY: "private severity"})),
        },
    )
    .await
    .unwrap();
    assert_eq!(
        sanitized.metadata,
        Some(json!({LOG_SEVERITY_METADATA_KEY: "[REDACTED]"}))
    );
}

#[tokio::test]
async fn trajectory_metric_marks_preserve_typed_measurements_and_redact_text() {
    let data = json!({
        "measurements": [
            {
                "name": "example.request_count",
                "kind": "counter",
                "value_type": "u64",
                "value": 1,
                "description": "private request count",
                "attributes": {
                    "owner": "person@example.com",
                    "regions": ["us-east", "us-west"],
                    "attempt": 2,
                    "sampled": true
                }
            },
            {
                "name": "example.queue_delta",
                "kind": "up_down_counter",
                "value_type": "i64",
                "value": -2
            },
            {
                "name": "example.latency",
                "kind": "histogram",
                "value_type": "f64",
                "value": 3.5,
                "boundaries": [1.0, 5.0]
            }
        ]
    });
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("example.metrics")
            .data(data.clone())
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version(METRIC_DATA_SCHEMA_VERSION)
                    .build(),
            )
            .build(),
        None,
        None,
    ));
    let callback = crate::builtin::event_sanitize_callback(trajectory_backend(None, "preserve"));
    let sanitized = callback(
        Arc::new(event),
        EventSanitizeFields {
            data: Some(data),
            category_profile: Some(CategoryProfile {
                extra: BTreeMap::from([("owner".into(), json!("person@example.com"))]),
                ..CategoryProfile::default()
            }),
            metadata: Some(json!({"owner": "person@example.com"})),
        },
    )
    .await
    .unwrap();

    let data = sanitized.data.unwrap();
    let envelope = serde_json::from_value::<MetricEnvelope>(data.clone()).unwrap();
    envelope.validate().unwrap();
    assert_eq!(data["measurements"][0]["value_type"], "u64");
    assert_eq!(data["measurements"][0]["value"], 1);
    assert_eq!(data["measurements"][1]["value_type"], "i64");
    assert_eq!(data["measurements"][1]["value"], -2);
    assert_eq!(data["measurements"][2]["value_type"], "f64");
    assert_eq!(data["measurements"][2]["value"], 3.5);
    assert_eq!(data["measurements"][2]["boundaries"], json!([1.0, 5.0]));
    assert_eq!(data["measurements"][0]["description"], "[REDACTED]");
    assert_eq!(data["measurements"][0]["attributes"]["owner"], "[REDACTED]");
    assert_eq!(
        data["measurements"][0]["attributes"]["regions"],
        json!(["[REDACTED]", "[REDACTED]"])
    );
    assert_eq!(data["measurements"][0]["attributes"]["attempt"], 2);
    assert_eq!(data["measurements"][0]["attributes"]["sampled"], true);
    assert_eq!(sanitized.metadata.unwrap()["owner"], "[REDACTED]");
    assert_eq!(
        sanitized.category_profile.unwrap().extra["owner"],
        "[REDACTED]"
    );
}

#[tokio::test]
async fn trajectory_metric_string_allowlist_requires_exact_key_and_value() {
    let data = json!({
        "measurements": [{
            "name": "example.request_count",
            "kind": "counter",
            "value_type": "u64",
            "value": 1,
            "description": "private request count",
            "attributes": {
                "query.source": "main",
                "operation": "private operation",
                "unapproved.key": "main",
                "execution.modes": ["main", "private mode"],
                "attempt": 2,
                "sampled": true
            }
        }]
    });
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("example.metrics")
            .data(data.clone())
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version(METRIC_DATA_SCHEMA_VERSION)
                    .build(),
            )
            .build(),
        None,
        None,
    ));
    let callback = crate::builtin::event_sanitize_callback(
        trajectory_backend_with_metric_allowlist(BTreeMap::from([
            ("query.source".into(), vec!["main".into()]),
            ("operation".into(), vec!["chat".into()]),
            ("execution.modes".into(), vec!["main".into()]),
        ])),
    );
    let sanitized = callback(
        Arc::new(event),
        EventSanitizeFields {
            data: Some(data),
            category_profile: Some(CategoryProfile {
                extra: BTreeMap::from([("query.source".into(), json!("main"))]),
                ..CategoryProfile::default()
            }),
            metadata: Some(json!({"query.source": "main"})),
        },
    )
    .await
    .unwrap();

    let data = sanitized.data.unwrap();
    let envelope = serde_json::from_value::<MetricEnvelope>(data.clone()).unwrap();
    envelope.validate().unwrap();
    let attributes = &data["measurements"][0]["attributes"];
    assert_eq!(attributes["query.source"], "main");
    assert_eq!(attributes["operation"], "[REDACTED]");
    assert_eq!(attributes["unapproved.key"], "[REDACTED]");
    assert_eq!(attributes["execution.modes"], json!(["main", "[REDACTED]"]));
    assert_eq!(attributes["attempt"], 2);
    assert_eq!(attributes["sampled"], true);
    assert_eq!(data["measurements"][0]["description"], "[REDACTED]");
    assert_eq!(sanitized.metadata.unwrap()["query.source"], "[REDACTED]");
    assert_eq!(
        sanitized.category_profile.unwrap().extra["query.source"],
        "[REDACTED]"
    );
}

#[tokio::test]
async fn builtin_metric_marks_preserve_typed_measurements() {
    let data = json!({
        "measurements": [{
            "name": "example.request_count",
            "kind": "counter",
            "value_type": "u64",
            "value": 1,
            "description": "private request count",
            "attributes": {"owner": "person@example.com"}
        }]
    });
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("example.metrics")
            .data(data.clone())
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version(METRIC_DATA_SCHEMA_VERSION)
                    .build(),
            )
            .build(),
        None,
        None,
    ));
    let callback = crate::builtin::event_sanitize_callback(
        crate::builtin::CompiledBuiltinBackend::new(
            BuiltinBackendConfig {
                target_paths: vec!["/owner".to_string()],
                target_path_globs: vec![
                    "/measurements/*/description".to_string(),
                    "/measurements/*/attributes/*".to_string(),
                ],
                ..BuiltinBackendConfig::default()
            },
            None,
        )
        .unwrap(),
    );
    let sanitized = callback(
        Arc::new(event),
        EventSanitizeFields {
            data: Some(data),
            category_profile: Some(CategoryProfile {
                extra: BTreeMap::from([("owner".into(), json!("person@example.com"))]),
                ..CategoryProfile::default()
            }),
            metadata: None,
        },
    )
    .await
    .unwrap();

    let data = sanitized.data.unwrap();
    serde_json::from_value::<MetricEnvelope>(data.clone())
        .unwrap()
        .validate()
        .unwrap();
    assert_eq!(data["measurements"][0]["value_type"], "u64");
    assert_eq!(data["measurements"][0]["value"], 1);
    assert!(data["measurements"][0]["description"].is_null());
    assert_eq!(data["measurements"][0]["attributes"], json!({}));
    assert!(
        !sanitized
            .category_profile
            .unwrap()
            .extra
            .contains_key("owner")
    );
}

#[tokio::test]
async fn builtin_metric_marks_remove_string_array_attributes_at_target_paths() {
    let data = json!({
        "measurements": [{
            "name": "example.request_count",
            "kind": "counter",
            "value_type": "u64",
            "value": 1,
            "attributes": {"regions": ["us-east", "us-west"]}
        }]
    });
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("example.metrics")
            .data(data.clone())
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version(METRIC_DATA_SCHEMA_VERSION)
                    .build(),
            )
            .build(),
        None,
        None,
    ));
    let callback = crate::builtin::event_sanitize_callback(
        crate::builtin::CompiledBuiltinBackend::new(
            BuiltinBackendConfig {
                target_paths: vec!["/measurements/0/attributes/regions/0".to_string()],
                ..BuiltinBackendConfig::default()
            },
            None,
        )
        .unwrap(),
    );
    let sanitized = callback(
        Arc::new(event.clone()),
        EventSanitizeFields {
            data: Some(data),
            category_profile: None,
            metadata: None,
        },
    )
    .await
    .unwrap();

    let data = sanitized.data.unwrap();
    serde_json::from_value::<MetricEnvelope>(data.clone())
        .unwrap()
        .validate()
        .unwrap();
    assert_eq!(
        data["measurements"][0]["attributes"]["regions"],
        json!(["us-west"])
    );

    let callback = crate::builtin::event_sanitize_callback(
        crate::builtin::CompiledBuiltinBackend::new(
            BuiltinBackendConfig {
                target_paths: vec!["/measurements/0/attributes/regions".to_string()],
                ..BuiltinBackendConfig::default()
            },
            None,
        )
        .unwrap(),
    );
    let sanitized = callback(
        Arc::new(event),
        EventSanitizeFields {
            data: Some(json!({
                "measurements": [{
                    "name": "example.request_count",
                    "kind": "counter",
                    "value_type": "u64",
                    "value": 1,
                    "attributes": {"regions": ["us-east", "us-west"]}
                }]
            })),
            category_profile: None,
            metadata: None,
        },
    )
    .await
    .unwrap();

    let data = sanitized.data.unwrap();
    serde_json::from_value::<MetricEnvelope>(data.clone())
        .unwrap()
        .validate()
        .unwrap();
    assert_eq!(data["measurements"][0]["attributes"], json!({}));
}

#[tokio::test]
async fn builtin_target_path_globs_sanitize_nested_mark_payloads() {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("example.mark").build(),
        None,
        None,
    ));
    let callback = crate::builtin::event_sanitize_callback(
        crate::builtin::CompiledBuiltinBackend::new(
            BuiltinBackendConfig {
                action: "redact".to_string(),
                pattern: Some("secret".to_string()),
                replacement: Some("[REDACTED]".to_string()),
                target_paths: vec!["/prompt".to_string()],
                target_path_globs: vec![
                    "/messages/*/content/*/text".to_string(),
                    "/output/*/content/*/text".to_string(),
                ],
                ..BuiltinBackendConfig::default()
            },
            None,
        )
        .unwrap(),
    );
    let sanitized = callback(
        Arc::new(event),
        EventSanitizeFields {
            data: Some(json!({
                "prompt": "secret prompt",
                "messages": [{"content": [{"text": "secret message"}]}],
                "output": [{
                    "content": [{"text": "secret output"}],
                    "id": "secret identifier"
                }],
            })),
            category_profile: None,
            metadata: Some(json!({
                "messages": [{"content": [{"text": "secret metadata"}]}],
            })),
        },
    )
    .await
    .unwrap();

    let data = sanitized.data.unwrap();
    assert_eq!(data["prompt"], "[REDACTED] prompt");
    assert_eq!(
        data["messages"][0]["content"][0]["text"],
        "[REDACTED] message"
    );
    assert_eq!(data["output"][0]["content"][0]["text"], "[REDACTED] output");
    assert_eq!(data["output"][0]["id"], "secret identifier");
    assert_eq!(
        sanitized.metadata.unwrap()["messages"][0]["content"][0]["text"],
        "[REDACTED] metadata"
    );
}

#[tokio::test]
async fn builtin_target_path_globs_treat_partial_asterisks_as_literal_segments() {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("example.mark").build(),
        None,
        None,
    ));
    let callback = crate::builtin::event_sanitize_callback(
        crate::builtin::CompiledBuiltinBackend::new(
            BuiltinBackendConfig {
                action: "redact".to_string(),
                pattern: Some("sensitive".to_string()),
                replacement: Some("[REDACTED]".to_string()),
                target_path_globs: vec!["/content*".to_string()],
                ..BuiltinBackendConfig::default()
            },
            None,
        )
        .unwrap(),
    );
    let sanitized = callback(
        Arc::new(event),
        EventSanitizeFields {
            data: Some(json!({
                "content*": "sensitive literal key",
                "content": "sensitive nonmatching key",
            })),
            category_profile: None,
            metadata: None,
        },
    )
    .await
    .unwrap();

    let data = sanitized.data.unwrap();
    assert_eq!(data["content*"], "[REDACTED] literal key");
    assert_eq!(data["content"], "sensitive nonmatching key");
}

#[tokio::test]
async fn builtin_target_path_globs_sanitize_tool_payloads() {
    let callback = crate::builtin::tool_sanitize_callback(
        crate::builtin::CompiledBuiltinBackend::new(
            BuiltinBackendConfig {
                action: "redact".to_string(),
                pattern: Some("secret".to_string()),
                replacement: Some("[REDACTED]".to_string()),
                target_path_globs: vec!["/output/*/content/*/text".to_string()],
                ..BuiltinBackendConfig::default()
            },
            None,
        )
        .unwrap(),
    );

    let sanitized = callback(
        "example.tool".to_string(),
        json!({
            "output": [{
                "content": [{"text": "secret tool result"}],
                "id": "secret identifier"
            }]
        }),
    )
    .await
    .unwrap();

    assert_eq!(
        sanitized["output"][0]["content"][0]["text"],
        "[REDACTED] tool result"
    );
    assert_eq!(sanitized["output"][0]["id"], "secret identifier");
}

#[tokio::test]
async fn builtin_target_path_globs_sanitize_typed_metric_attributes() {
    let data = json!({
        "measurements": [{
            "name": "example.request_count",
            "kind": "counter",
            "value_type": "u64",
            "value": 1,
            "attributes": {"regions": ["us-east", "us-west"]}
        }]
    });
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("example.metrics")
            .data(data.clone())
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version(METRIC_DATA_SCHEMA_VERSION)
                    .build(),
            )
            .build(),
        None,
        None,
    ));
    let callback = crate::builtin::event_sanitize_callback(
        crate::builtin::CompiledBuiltinBackend::new(
            BuiltinBackendConfig {
                target_path_globs: vec!["/measurements/*/attributes/regions".to_string()],
                ..BuiltinBackendConfig::default()
            },
            None,
        )
        .unwrap(),
    );
    let sanitized = callback(
        Arc::new(event),
        EventSanitizeFields {
            data: Some(data),
            category_profile: None,
            metadata: None,
        },
    )
    .await
    .unwrap();

    let data = sanitized.data.unwrap();
    serde_json::from_value::<MetricEnvelope>(data.clone())
        .unwrap()
        .validate()
        .unwrap();
    assert_eq!(data["measurements"][0]["attributes"], json!({}));
}

#[tokio::test]
async fn configured_target_path_globs_redact_otlp_content_and_preserve_typed_metrics() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "codec": "openai_chat",
        "profiles": [{
            "mode": "builtin",
            "priority": 90,
            "builtin": {
                "action": "redact",
                "pattern": "synthetic-glob-secret",
                "replacement": "[REDACTED]",
                "target_paths": [
                    "/prompt", "/response", "/message", "/content", "/input", "/output",
                    "/result", "/stdout", "/stderr", "/command", "/description", "/arguments"
                ],
                "target_path_globs": [
                    "/content/*/text",
                    "/output/*/text",
                    "/output/*/content/*/text",
                    "/output/*/input",
                    "/messages/*/content",
                    "/messages/*/content/*/text",
                    "/messages/*/content/*/text"
                ]
            }
        }]
    })))
    .await
    .unwrap();

    let trace_listener =
        std::net::TcpListener::bind("127.0.0.1:0").expect("trace capture listener should bind");
    let trace_endpoint = format!("http://{}", trace_listener.local_addr().unwrap());
    let trace_request = capture_one_otlp_request(trace_listener);
    let trace_subscriber = OpenTelemetrySubscriber::new(OpenTelemetryConfig::new(
        OpenTelemetryType::Full,
        trace_endpoint,
    ))
    .unwrap();
    trace_subscriber
        .register("pii-target-glob-e2e-traces")
        .unwrap();

    let metric_listener =
        std::net::TcpListener::bind("127.0.0.1:0").expect("metric capture listener should bind");
    let metric_endpoint = format!("http://{}", metric_listener.local_addr().unwrap());
    let metric_request = capture_otlp_requests(metric_listener, 2);
    let metric_subscriber =
        OpenTelemetryMetricSubscriber::new(OpenTelemetryMetricConfig::new(metric_endpoint))
            .unwrap();
    metric_subscriber
        .register("pii-target-glob-e2e-metrics")
        .unwrap();

    let agent = push_scope(
        PushScopeParams::builder()
            .name("target-glob-agent")
            .scope_type(ScopeType::Agent)
            .input(json!({
                "prompt": "synthetic-glob-secret prompt",
                "messages": [
                    {"content": "synthetic-glob-secret message"},
                    {"content": [{"text": "synthetic-glob-secret nested message"}]}
                ]
            }))
            .metadata(json!({"nv.user.id": "opaque-user-id"}))
            .build(),
    )
    .unwrap();
    tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("target-glob-tool")
            .args(json!({
                "arguments": "synthetic-glob-secret arguments",
                "messages": [{"content": [{"text": "synthetic-glob-secret tool input"}]}]
            }))
            .func(Arc::new(|_| {
                Box::pin(async move {
                    Ok(ToolExecutionResult::new(json!({
                        "output": [{
                            "content": [{"text": "synthetic-glob-secret tool output"}]
                        }]
                    })))
                })
            }))
            .build(),
    )
    .await
    .unwrap();
    event(
        EmitMarkEventParams::builder()
            .name("target-glob-mark")
            .data(json!({
                "output": [{"content": [{"text": "synthetic-glob-secret mark output"}]}]
            }))
            .build(),
    )
    .unwrap();
    pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&agent.uuid)
            .output(json!({
                "response": "synthetic-glob-secret response",
                "content": [{"text": "synthetic-glob-secret final content"}]
            }))
            .build(),
    )
    .unwrap();
    metric(
        EmitMetricEventParams::builder()
            .name("target_glob_cache_efficiency")
            .measurements(vec![
                MetricMeasurement::builder()
                    .name("example.cache_efficiency")
                    .kind(MetricKind::Gauge)
                    .value_type(MetricValueType::U64)
                    .value(json!(42_u64))
                    .attributes(json!({"lane": "metrics"}))
                    .build(),
            ])
            .build(),
    )
    .unwrap();

    trace_subscriber.force_flush().unwrap();
    trace_subscriber
        .deregister("pii-target-glob-e2e-traces")
        .unwrap();
    metric_subscriber.force_flush().unwrap();
    metric_subscriber
        .deregister("pii-target-glob-e2e-metrics")
        .unwrap();
    trace_subscriber.shutdown().unwrap();
    metric_subscriber.shutdown().unwrap();
    clear_plugin_configuration().unwrap();

    let trace_request = trace_request
        .recv_timeout(Duration::from_secs(5))
        .expect("sanitized trace should reach the local OTLP collector");
    assert_eq!(trace_request.path, "/v1/traces");
    let trace = ExportTraceServiceRequest::decode(trace_request.body.as_slice()).unwrap();
    let trace_debug = format!("{trace:#?}");
    assert!(
        !trace_debug.contains("synthetic-glob-secret"),
        "{trace_debug}"
    );
    assert!(trace_debug.contains("[REDACTED]"), "{trace_debug}");
    assert!(trace_debug.contains("opaque-user-id"), "{trace_debug}");

    let metric_request = metric_request
        .recv_timeout(Duration::from_secs(5))
        .expect("typed metric should reach the local OTLP collector");
    assert_eq!(metric_request.path, "/v1/metrics");
    let metrics = ExportMetricsServiceRequest::decode(metric_request.body.as_slice()).unwrap();
    let metric = metrics
        .resource_metrics
        .iter()
        .flat_map(|resource| &resource.scope_metrics)
        .flat_map(|scope| &scope.metrics)
        .find(|metric| metric.name == "example.cache_efficiency")
        .expect("typed metric should retain its name");
    let Some(opentelemetry_proto::tonic::metrics::v1::metric::Data::Gauge(gauge)) = &metric.data
    else {
        panic!("typed metric should remain a gauge: {metric:#?}");
    };
    let Some(opentelemetry_proto::tonic::metrics::v1::number_data_point::Value::AsInt(value)) =
        gauge
            .data_points
            .first()
            .and_then(|point| point.value.as_ref())
    else {
        panic!("typed metric should retain its integer value: {metric:#?}");
    };
    assert_eq!(*value, 42);
}

#[tokio::test]
async fn builtin_target_paths_keep_literal_asterisk_semantics() {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("example.mark").build(),
        None,
        None,
    ));
    let callback = crate::builtin::event_sanitize_callback(
        crate::builtin::CompiledBuiltinBackend::new(
            BuiltinBackendConfig {
                action: "redact".to_string(),
                pattern: Some("secret".to_string()),
                replacement: Some("[REDACTED]".to_string()),
                target_paths: vec!["/*".to_string(), "/a~1b".to_string()],
                ..BuiltinBackendConfig::default()
            },
            None,
        )
        .unwrap(),
    );
    let sanitized = callback(
        Arc::new(event),
        EventSanitizeFields {
            data: Some(json!({
                "*": "secret literal",
                "a/b": "secret escaped",
                "other": "secret other",
            })),
            category_profile: None,
            metadata: None,
        },
    )
    .await
    .unwrap();

    let data = sanitized.data.unwrap();
    assert_eq!(data["*"], "[REDACTED] literal");
    assert_eq!(data["a/b"], "[REDACTED] escaped");
    assert_eq!(data["other"], "secret other");
}

#[tokio::test]
async fn trajectory_metric_marks_drop_invalid_envelopes() {
    let data = json!({
        "measurements": [{
            "name": "example.invalid",
            "kind": "counter",
            "value_type": "invalid",
            "value": 1
        }]
    });
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("example.metrics")
            .data(data.clone())
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version(METRIC_DATA_SCHEMA_VERSION)
                    .build(),
            )
            .build(),
        None,
        None,
    ));
    let callback = crate::builtin::event_sanitize_callback(trajectory_backend(None, "preserve"));
    let sanitized = callback(
        Arc::new(event),
        EventSanitizeFields {
            data: Some(data),
            category_profile: None,
            metadata: None,
        },
    )
    .await
    .unwrap();

    assert!(sanitized.data.is_none());
}

#[tokio::test]
async fn builtin_metric_marks_drop_invalid_envelopes() {
    let data = json!({
        "measurements": [{
            "name": "example.invalid",
            "kind": "counter",
            "value_type": "invalid",
            "value": 1
        }]
    });
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("example.metrics")
            .data(data.clone())
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version(METRIC_DATA_SCHEMA_VERSION)
                    .build(),
            )
            .build(),
        None,
        None,
    ));
    let callback = crate::builtin::event_sanitize_callback(
        crate::builtin::CompiledBuiltinBackend::new(
            BuiltinBackendConfig {
                target_paths: vec!["/unused".to_string()],
                ..BuiltinBackendConfig::default()
            },
            None,
        )
        .unwrap(),
    );
    let sanitized = callback(
        Arc::new(event),
        EventSanitizeFields {
            data: Some(data),
            category_profile: None,
            metadata: None,
        },
    )
    .await
    .unwrap();

    assert!(sanitized.data.is_none());
}

#[tokio::test]
async fn trajectory_profile_preserves_typed_llm_accounting_while_redacting_annotations() {
    let callback = crate::builtin::event_sanitize_callback(trajectory_backend(None, "preserve"));
    let annotated_response: nemo_relay::codec::response::AnnotatedLlmResponse =
        serde_json::from_value(json!({
            "model": "claude-sonnet-4-6",
            "message": "private answer",
            "finish_reason": "complete",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cost": {
                    "total": 0.42,
                    "currency": "USD",
                    "source": "provider_reported"
                }
            },
            "optimization_summary": {
                "schema_version": "1",
                "calculation_version": "1",
                "status": "complete",
                "baseline_model": {"model": "claude-opus-4-6"},
                "effective_model": {"model": "claude-sonnet-4-6"},
                "effective_usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                "baseline_usage": {"prompt_tokens": 140, "completion_tokens": 20, "total_tokens": 160},
                "tokens_saved": {"prompt_tokens": 40, "total_tokens": 40},
                "estimated_cost_saved": 0.8,
                "currency": "USD",
                "contributions": [{
                    "producer": "neutral.optimizer",
                    "kind": "input_compression",
                    "applied": true,
                    "token_impact": {
                        "saved": {"prompt_tokens": 40, "total_tokens": 40},
                        "quality": "estimated",
                        "estimation_method": "neutral_counter"
                    },
                    "payload_schema": {"name": "neutral.evidence", "version": "1"},
                    "payload": {"private_excerpt": "private tool output", "strategy": "head_tail"}
                }]
            }
        }))
        .unwrap();
    let event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("llm").build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::llm(),
        None,
    ));
    let sanitized = callback(
        Arc::new(event),
        EventSanitizeFields {
            data: Some(json!({"already": "sanitized by the response callback"})),
            category_profile: Some(
                CategoryProfile::builder()
                    .model_name("claude-sonnet-4-6")
                    .annotated_response(Arc::new(annotated_response))
                    .build(),
            ),
            metadata: None,
        },
    )
    .await
    .unwrap();

    let profile = sanitized.category_profile.unwrap();
    assert_eq!(profile.model_name.as_deref(), Some("claude-sonnet-4-6"));
    let response = profile.annotated_response.unwrap();
    assert_eq!(response.response_text(), Some("[REDACTED]"));
    assert_eq!(response.usage.as_ref().unwrap().total_tokens, Some(120));
    assert_eq!(
        response
            .usage
            .as_ref()
            .unwrap()
            .cost
            .as_ref()
            .unwrap()
            .total,
        Some(0.42)
    );
    let summary = response.optimization_summary.as_ref().unwrap();
    assert_eq!(summary.tokens_saved.prompt_tokens, Some(40));
    assert_eq!(summary.estimated_cost_saved, Some(0.8));
    assert_eq!(summary.contributions[0].producer, "neutral.optimizer");
    assert_eq!(
        summary.contributions[0].payload.as_ref().unwrap()["private_excerpt"],
        "[REDACTED]"
    );
    assert_eq!(
        summary.contributions[0].payload.as_ref().unwrap()["strategy"],
        "[REDACTED]"
    );
    assert_eq!(
        sanitized.data.unwrap()["already"],
        "sanitized by the response callback",
        "specialized LLM data must not be processed twice"
    );
}

#[tokio::test]
async fn preserved_custom_marks_remain_eligible_for_a_later_email_profile() {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("neutral.plugin.evidence").build(),
        Some(EventCategory::custom()),
        Some(CategoryProfile::builder().subtype("neutral.plugin").build()),
    ));
    let fields = EventSanitizeFields {
        data: Some(json!({"owner": "alice@example.com", "score": 0.9})),
        category_profile: event.category_profile().cloned(),
        metadata: Some(json!({"contact": "bob@example.com"})),
    };
    let trajectory = crate::builtin::event_sanitize_callback(trajectory_backend(None, "preserve"));
    let email = crate::builtin::event_sanitize_callback(
        crate::builtin::CompiledBuiltinBackend::new(
            BuiltinBackendConfig {
                action: "redact".into(),
                detector: Some("email".into()),
                ..BuiltinBackendConfig::default()
            },
            None,
        )
        .unwrap(),
    );

    let fields = trajectory(Arc::new(event.clone()), fields).await.unwrap();
    let sanitized = email(Arc::new(event), fields).await.unwrap();
    assert_eq!(sanitized.data.as_ref().unwrap()["owner"], "[REDACTED]");
    assert_eq!(sanitized.data.as_ref().unwrap()["score"], 0.9);
    assert_eq!(
        sanitized.metadata.as_ref().unwrap()["contact"],
        "[REDACTED]"
    );
}

#[test]
fn pii_redaction_defaults_enable_mark_sanitization() {
    let config = PiiRedactionConfig::default();
    assert!(config.mark);
    assert!(config.profiles.is_empty());
}

#[test]
fn generated_registration_names_sort_in_profile_array_order() {
    assert!(
        registration_name(Some("profile_2"), "mark")
            < registration_name(Some("profile_10"), "mark")
    );
}

#[test]
fn typed_profile_config_serializes_without_conflicting_legacy_defaults() {
    let config = PiiRedactionConfig {
        codec: Some("openai_chat".into()),
        profiles: vec![PiiRedactionProfile {
            builtin: Some(BuiltinBackendConfig {
                action: "redact".into(),
                detector: Some("email".into()),
                ..BuiltinBackendConfig::default()
            }),
            ..PiiRedactionProfile::default()
        }],
        ..PiiRedactionConfig::default()
    };
    let serialized = serde_json::to_value(&config).unwrap();
    let object = serialized.as_object().unwrap();
    for legacy_field in [
        "mode",
        "input",
        "output",
        "mark",
        "tool_input",
        "tool_output",
        "priority",
        "builtin",
        "local",
    ] {
        assert!(!object.contains_key(legacy_field), "{legacy_field}");
    }

    let report = validate_plugin_config(&plugin_config(serialized));
    assert!(!report.has_errors(), "{:?}", report.diagnostics);
}

#[test]
fn typed_trajectory_preset_omits_the_legacy_default_action() {
    let config = BuiltinBackendConfig {
        preset: Some("trajectory_context".into()),
        ..BuiltinBackendConfig::default()
    };
    let serialized = serde_json::to_value(config).unwrap();
    assert_eq!(serialized["preset"], "trajectory_context");
    assert!(serialized.get("action").is_none());
    assert!(serialized.get("custom_mark_payload_policy").is_none());
    assert!(
        serialized
            .get("metric_string_attribute_allowlist")
            .is_none()
    );
}

#[test]
fn profile_array_executes_every_profile_in_stable_array_order() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "codec": "openai_chat",
        "profiles": [
            {
                "mode": "builtin",
                "priority": 100,
                "builtin": {
                    "action": "regex_replace",
                    "pattern": "alpha",
                    "replacement": "beta"
                }
            },
            {
                "mode": "builtin",
                "priority": 100,
                "builtin": {
                    "action": "regex_replace",
                    "pattern": "beta",
                    "replacement": "gamma"
                }
            }
        ]
    }))))
    .unwrap();

    let events = capture_events("pii-profile-order");
    event(
        EmitMarkEventParams::builder()
            .name("ordered-profile-mark")
            .data(json!({"value": "alpha"}))
            .build(),
    )
    .unwrap();
    let captured = captured_events_snapshot(&events);
    assert_eq!(captured[0].data().unwrap()["value"], "gamma");

    deregister_subscriber("pii-profile-order").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn profile_array_rejects_legacy_fields_and_reports_profile_paths() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "codec": "openai_chat",
        "mark": true,
        "profiles": [{
            "mode": "builtin",
            "builtin": {
                "action": "regex_replace"
            },
            "unexpected": true
        }]
    })));

    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("mark")
            && diagnostic
                .message
                .contains("cannot be combined with profiles")
    }));
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| { diagnostic.field.as_deref() == Some("profiles[0].unexpected") })
    );
    assert!(
        report.diagnostics.iter().any(|diagnostic| {
            diagnostic.field.as_deref() == Some("profiles[0].builtin.pattern")
        })
    );
}

#[test]
fn profile_array_requires_at_least_one_profile_and_matching_local_settings() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let empty = validate_plugin_config(&plugin_config(json!({
        "codec": "openai_chat",
        "profiles": []
    })));
    assert!(empty.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("profiles")
            && diagnostic.message.contains("at least one")
    }));

    let all_disabled = validate_plugin_config(&plugin_config(json!({
        "codec": "openai_chat",
        "profiles": [{
            "enabled": false,
            "mode": "builtin",
            "builtin": {"action": "remove"}
        }]
    })));
    assert!(all_disabled.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "pii_redaction.unsupported_value"
            && diagnostic.field.as_deref() == Some("profiles")
            && diagnostic.message.contains("at least one enabled")
    }));

    let missing_local = validate_plugin_config(&plugin_config(json!({
        "codec": "openai_chat",
        "profiles": [{"mode": "local_model"}]
    })));
    assert!(missing_local.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("profiles[0].local")
            && diagnostic.message.contains("required")
    }));
}

#[test]
fn disabled_profiles_are_validated_when_an_enabled_profile_exists() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "codec": "openai_chat",
        "profiles": [
            {
                "enabled": false,
                "mode": "builtin",
                "builtin": {
                    "action": "not-an-action"
                }
            },
            {
                "mode": "builtin",
                "builtin": {"action": "remove"}
            }
        ]
    })));
    assert!(
        report.diagnostics.iter().any(|diagnostic| {
            diagnostic.field.as_deref() == Some("profiles[0].builtin.action")
        })
    );
}

#[test]
fn local_profile_registrations_receive_generated_namespaces() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    register_local_backend_provider(Arc::new(|_, ctx| {
        ctx.register_mark_sanitize_guardrail(
            "shared",
            100,
            Arc::new(|_, fields| Box::pin(async move { Ok(fields) })),
        )
    }))
    .unwrap();

    let plugin = PiiRedactionPlugin;
    let config = json!({
        "codec": "openai_chat",
        "profiles": [
            {"mode": "local_model", "local": {"backend": "one"}},
            {"mode": "local_model", "local": {"backend": "two"}}
        ]
    });
    let Json::Object(config) = config else {
        panic!("component config must be object");
    };
    let mut ctx = PluginRegistrationContext::with_namespace("profiles::");
    futures::executor::block_on(plugin.register(&config, &mut ctx)).unwrap();
    let mut registrations = ctx.into_registrations();
    let registrations_debug = format!("{registrations:?}");
    assert!(registrations_debug.contains("profiles::profile_00000000000000000000/shared"));
    assert!(registrations_debug.contains("profiles::profile_00000000000000000001/shared"));
    rollback_registrations(&mut registrations);
    assert!(registrations.is_empty());
}

#[test]
fn failed_later_profile_rolls_back_earlier_profile_registrations() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    register_local_backend_provider(Arc::new(|_, _| {
        Err(PluginError::RegistrationFailed(
            "intentional profile failure".into(),
        ))
    }))
    .unwrap();
    let activation = futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "codec": "openai_chat",
        "profiles": [
            {
                "mode": "builtin",
                "builtin": {"action": "redact", "detector": "email"}
            },
            {
                "mode": "local_model",
                "local": {"backend": "failing"}
            }
        ]
    }))));
    assert!(activation.is_err());

    let events = capture_events("pii-profile-rollback");
    event(
        EmitMarkEventParams::builder()
            .name("raw-after-rollback")
            .data(json!({"email": "person@example.com"}))
            .build(),
    )
    .unwrap();
    let captured = captured_events_snapshot(&events);
    assert_eq!(captured[0].data().unwrap()["email"], "person@example.com");
    deregister_subscriber("pii-profile-rollback").unwrap();
}

#[tokio::test]
async fn event_sanitizer_transforms_data_category_profile_and_metadata_independently() {
    let backend = crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "regex_replace".into(),
            pattern: Some("person@example\\.com".into()),
            replacement: Some("[REDACTED]".into()),
            ..BuiltinBackendConfig::default()
        },
        None,
    )
    .unwrap();
    let callback = crate::builtin::event_sanitize_callback(backend);
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("mark").build(),
        None,
        None,
    ));
    let sanitized = callback(
        Arc::new(event),
        EventSanitizeFields {
            data: Some(json!({"email": "person@example.com"})),
            category_profile: Some(
                CategoryProfile::builder()
                    .subtype("person@example.com")
                    .build(),
            ),
            metadata: Some(json!({"owner": "person@example.com"})),
        },
    )
    .await
    .unwrap();
    assert_eq!(sanitized.data.unwrap()["email"], "[REDACTED]");
    assert_eq!(
        sanitized.category_profile.unwrap().subtype.as_deref(),
        Some("[REDACTED]")
    );
    assert_eq!(sanitized.metadata.unwrap()["owner"], "[REDACTED]");
}

#[tokio::test]
async fn llm_and_tool_scope_metadata_is_sanitized_without_reprocessing_typed_fields() {
    let backend = crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "redact".into(),
            detector: Some("email".into()),
            ..BuiltinBackendConfig::default()
        },
        None,
    )
    .unwrap();
    let callback = crate::builtin::event_sanitize_callback(backend);

    for category in [EventCategory::llm(), EventCategory::tool()] {
        let event = Event::Scope(ScopeEvent::new(
            BaseEvent::builder().name("typed-scope").build(),
            ScopeCategory::Start,
            Vec::new(),
            category,
            None,
        ));
        let original_profile = CategoryProfile::builder()
            .subtype("person@example.com")
            .build();
        let sanitized = callback(
            Arc::new(event),
            EventSanitizeFields {
                data: Some(json!({"content": "person@example.com"})),
                category_profile: Some(original_profile.clone()),
                metadata: Some(json!({"owner": "person@example.com"})),
            },
        )
        .await
        .unwrap();

        assert_eq!(
            sanitized.data.unwrap()["content"],
            "person@example.com",
            "specialized scope data must not be generically sanitized twice"
        );
        assert_eq!(sanitized.category_profile.unwrap(), original_profile);
        assert_eq!(sanitized.metadata.unwrap()["owner"], "[REDACTED]");
    }
}

#[tokio::test]
async fn scope_event_sanitizer_respects_enabled_llm_and_tool_surfaces() {
    let backend = crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "redact".into(),
            detector: Some("email".into()),
            ..BuiltinBackendConfig::default()
        },
        None,
    )
    .unwrap();

    for (sanitize_llm, sanitize_tool, expected_llm, expected_tool) in [
        (true, false, "[REDACTED]", "person@example.com"),
        (false, true, "person@example.com", "[REDACTED]"),
    ] {
        let callback = crate::builtin::scope_event_sanitize_callback(
            backend.clone(),
            sanitize_llm,
            sanitize_tool,
        );
        for (category, expected_owner) in [
            (EventCategory::llm(), expected_llm),
            (EventCategory::tool(), expected_tool),
        ] {
            let event = Event::Scope(ScopeEvent::new(
                BaseEvent::builder().name("typed-scope").build(),
                ScopeCategory::Start,
                Vec::new(),
                category,
                None,
            ));
            let original_profile = CategoryProfile::builder()
                .subtype("person@example.com")
                .build();
            let sanitized = callback(
                Arc::new(event),
                EventSanitizeFields {
                    data: Some(json!({"content": "person@example.com"})),
                    category_profile: Some(original_profile.clone()),
                    metadata: Some(json!({"owner": "person@example.com"})),
                },
            )
            .await
            .unwrap();

            assert_eq!(sanitized.data.unwrap()["content"], "person@example.com");
            assert_eq!(sanitized.category_profile.unwrap(), original_profile);
            assert_eq!(sanitized.metadata.unwrap()["owner"], expected_owner);
        }
    }
}

#[tokio::test]
async fn event_sanitizer_discards_category_profile_when_sanitization_fails() {
    let backend = crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "regex_replace".into(),
            pattern: Some("person@example\\.com".into()),
            replacement: Some("[REDACTED]".into()),
            ..BuiltinBackendConfig::default()
        },
        None,
    )
    .unwrap();
    let callback = crate::builtin::event_sanitize_callback(backend);
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("mark").build(),
        None,
        None,
    ));
    let sanitized = callback(
        Arc::new(event),
        EventSanitizeFields {
            data: None,
            category_profile: Some(CategoryProfile {
                extra: BTreeMap::from([(
                    "annotated_request".to_string(),
                    json!("invalid annotation"),
                )]),
                ..CategoryProfile::default()
            }),
            metadata: None,
        },
    )
    .await
    .unwrap();

    assert!(sanitized.category_profile.is_none());
}

#[test]
fn validate_rejects_config_with_no_enabled_surfaces() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "builtin": {
            "action": "remove"
        },
        "input": false,
        "output": false,
        "mark": false,
        "tool_input": false,
        "tool_output": false,
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.code == "pii_redaction.unsupported_value"
            && diag
                .message
                .contains("at least one redaction surface must be enabled")
    }));
}

#[test]
fn validate_allows_documented_policy_unknown_component_field() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "tool_input": true,
        "tool_output": false,
        "input": false,
        "output": false,
        "builtin": {
            "action": "remove"
        },
        "policy": {
            "unknown_component": "warn",
            "unknown_field": "warn",
            "unsupported_value": "error"
        }
    })));

    assert!(!report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("policy.unknown_component")
            && diag.code == "pii_redaction.unknown_field"
    }));
}

#[test]
fn validate_rejects_unsupported_config_version() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "version": 2,
        "mode": "builtin",
        "tool_input": true,
        "input": false,
        "output": false,
        "tool_output": false,
        "builtin": {
            "action": "remove"
        }
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("version")
            && diag.code == "pii_redaction.unsupported_config_version"
            && diag.message.contains("version 2 is unsupported")
    }));
}

#[test]
fn validate_rejects_local_section_outside_local_mode() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "builtin": {
            "action": "remove"
        },
        "local": {
            "backend": "future-local-model"
        }
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("local") && diag.message.contains("mode = 'local_model'")
    }));
}

#[test]
fn validate_rejects_builtin_mode_without_builtin_section() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin"
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("builtin")
            && diag.message.contains("required when mode = 'builtin'")
    }));
}

#[test]
fn validate_allows_llm_surfaces_without_codec() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "builtin": {
            "action": "remove",
            "target_paths": ["/message"]
        },
        "input": true,
        "output": false,
    })));

    assert!(report.diagnostics.is_empty(), "{report:?}");
}

#[test]
fn validate_rejects_ambiguous_gemini_codec_name() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "codec": "gemini",
        "input": true,
        "output": true,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/messages/0/content"]
        }
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("codec")
            && diag.message.contains("gemini_generate_content")
            && !diag.message.contains("gemini,")
            && !diag.message.ends_with("gemini")
    }));
}

#[test]
fn validate_rejects_regex_replace_without_pattern() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "builtin": {
            "action": "regex_replace"
        }
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("builtin.pattern")
            && diag
                .message
                .contains("required when builtin.action = 'regex_replace'")
    }));
}

#[test]
fn validate_rejects_invalid_builtin_pattern_regex() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "builtin": {
            "action": "regex_replace",
            "pattern": "[unterminated"
        }
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("builtin.pattern")
            && diag.message.contains("invalid builtin matcher regex")
    }));
}

#[test]
fn validate_rejects_malformed_builtin_target_paths() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "builtin": {
            "action": "remove",
            "target_paths": [
                "messages/0/content",
                "/messages/~2content",
                "/messages/trailing~"
            ]
        }
    })));

    for index in 0..3 {
        let expected_field = format!("builtin.target_paths[{index}]");
        assert!(report.diagnostics.iter().any(|diagnostic| {
            diagnostic.field.as_deref() == Some(expected_field.as_str())
                && diagnostic.message.contains("RFC 6901")
        }));
    }
}

#[test]
fn validate_accepts_valid_builtin_target_paths() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "builtin": {
            "action": "remove",
            "target_paths": [
                "",
                "/",
                "//empty-segment",
                "/messages/0/content",
                "/metadata/~0owner",
                "/metadata/a~1b"
            ]
        }
    })));

    assert!(!report.has_errors(), "{:#?}", report.diagnostics);
}

#[test]
fn validate_builtin_target_path_globs_as_json_pointers() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "builtin": {
            "action": "remove",
            "target_path_globs": [
                "/messages/*/content",
                "/output/*/content/*/text",
                "/metadata/a~1b"
            ]
        }
    })));
    assert!(!report.has_errors(), "{:#?}", report.diagnostics);

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "builtin": {
            "action": "remove",
            "target_path_globs": ["messages/*/content", "/messages/~2content"]
        }
    })));
    for index in 0..2 {
        let expected_field = format!("builtin.target_path_globs[{index}]");
        assert!(report.diagnostics.iter().any(|diagnostic| {
            diagnostic.field.as_deref() == Some(expected_field.as_str())
                && diagnostic.message.contains("RFC 6901")
        }));
    }
}

#[test]
fn profile_validation_reports_malformed_target_path_index() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "codec": "openai_chat",
        "profiles": [{
            "mode": "builtin",
            "builtin": {
                "action": "remove",
                "target_paths": ["/messages/0/content", "message"]
            }
        }]
    })));

    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("profiles[0].builtin.target_paths[1]")
            && diagnostic.message.contains("RFC 6901")
    }));
}

#[test]
fn activation_rejects_malformed_target_path_when_diagnostic_is_downgraded() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    let activation = futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "builtin": {
            "action": "remove",
            "target_paths": ["messages/0/content"]
        },
        "policy": {
            "unsupported_value": "warn"
        }
    }))));

    let error = activation.expect_err("malformed target path must fail plugin activation");
    assert!(error.to_string().contains("builtin.target_paths[0]"));
}

#[test]
fn preset_does_not_bypass_malformed_target_path_validation() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let config = json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "builtin": {
            "preset": "trajectory_context",
            "target_paths": ["messages/0/content"]
        },
        "policy": {
            "unsupported_value": "warn"
        }
    });
    let report = validate_plugin_config(&plugin_config(config.clone()));
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("builtin.target_paths[0]")
            && diagnostic.message.contains("RFC 6901")
    }));

    setup_isolated_thread();
    let activation = futures::executor::block_on(initialize_plugins(plugin_config(config)));
    let error = activation.expect_err("preset must not bypass malformed target path validation");
    assert!(error.to_string().contains("builtin.target_paths[0]"));
}

#[test]
fn validate_rejects_mask_with_empty_mask_char() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "builtin": {
            "action": "mask",
            "mask_char": ""
        }
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("builtin.mask_char")
            && diag.message.contains("must not be empty")
    }));
}

#[test]
fn validate_rejects_builtin_detector_and_pattern_together() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "builtin": {
            "action": "mask",
            "pattern": "secret",
            "detector": "email"
        }
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("builtin.detector")
            && diag.message.contains("cannot both be set")
    }));
}

#[test]
fn validate_rejects_unknown_builtin_detector() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "builtin": {
            "action": "mask",
            "detector": "ssn-ish"
        }
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("builtin.detector")
            && diag.message.contains("supported built-in detector presets")
    }));
}

#[test]
fn local_backend_provider_is_invoked_for_local_model_mode() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let called = Arc::new(AtomicBool::new(false));
    let called_inner = Arc::clone(&called);
    register_local_backend_provider(Arc::new(
        move |config, _ctx: &mut PluginRegistrationContext| {
            called_inner.store(true, Ordering::SeqCst);
            assert_eq!(config.mode, "local_model");
            Ok(())
        },
    ))
    .unwrap();

    let plugin = PiiRedactionPlugin;
    let mut ctx = PluginRegistrationContext::with_namespace("test::");
    let config = json!({
        "mode": "local_model",
        "tool_input": true,
    });
    let Json::Object(config) = config else {
        panic!("component config must be object");
    };

    futures::executor::block_on(plugin.register(&config, &mut ctx)).unwrap();

    assert!(called.load(Ordering::SeqCst));
}

#[test]
fn local_backend_reports_missing_and_failed_provider_initialization() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let plugin = PiiRedactionPlugin;
    let config = json!({"mode": "local_model"});
    let Json::Object(config) = config else {
        panic!("component config must be object");
    };
    let mut ctx = PluginRegistrationContext::with_namespace("missing::");
    let missing = futures::executor::block_on(plugin.register(&config, &mut ctx))
        .expect_err("missing local provider should fail registration");
    assert!(missing.to_string().contains("unavailable"));

    register_local_backend_provider(Arc::new(|_, _| {
        Err(PluginError::RegistrationFailed(
            "provider initialization failed".into(),
        ))
    }))
    .unwrap();
    let mut ctx = PluginRegistrationContext::with_namespace("failed::");
    let failed = futures::executor::block_on(plugin.register(&config, &mut ctx))
        .expect_err("failed local provider should fail registration");
    assert!(
        failed
            .to_string()
            .contains("provider initialization failed")
    );
}

#[test]
fn builtin_backend_sanitizes_mark_and_generic_scope_observability_fields() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": true,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "person@example\\.com",
            "replacement": "[REDACTED]"
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-event-fields");
    event(
        EmitMarkEventParams::builder()
            .name("sensitive-mark")
            .data(json!({"email": "person@example.com"}))
            .metadata(json!({"owner": "person@example.com"}))
            .build(),
    )
    .unwrap();
    let scope = push_scope(
        PushScopeParams::builder()
            .name("sensitive-scope")
            .scope_type(ScopeType::Custom)
            .input(json!({"email": "person@example.com"}))
            .metadata(json!({"owner": "person@example.com"}))
            .build(),
    )
    .unwrap();
    pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&scope.uuid)
            .output(json!({"email": "person@example.com"}))
            .metadata(json!({"reviewer": "person@example.com"}))
            .build(),
    )
    .unwrap();

    let captured = captured_events_snapshot(&events);
    let mark = captured
        .iter()
        .find(|event| event.name() == "sensitive-mark")
        .unwrap();
    assert_eq!(mark.data().unwrap()["email"], "[REDACTED]");
    assert_eq!(mark.metadata().unwrap()["owner"], "[REDACTED]");
    let start = captured
        .iter()
        .find(|event| {
            event.name() == "sensitive-scope"
                && event.scope_category() == Some(ScopeCategory::Start)
        })
        .unwrap();
    assert_eq!(start.data().unwrap()["email"], "[REDACTED]");
    assert_eq!(start.metadata().unwrap()["owner"], "[REDACTED]");
    let end = captured
        .iter()
        .find(|event| {
            event.name() == "sensitive-scope" && event.scope_category() == Some(ScopeCategory::End)
        })
        .unwrap();
    assert_eq!(end.data().unwrap()["email"], "[REDACTED]");
    assert_eq!(end.metadata().unwrap()["owner"], "[REDACTED]");
    assert_eq!(end.metadata().unwrap()["reviewer"], "[REDACTED]");

    deregister_subscriber("pii-redaction-event-fields").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn mark_false_preserves_mark_fields() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": true,
        "output": false,
        "mark": false,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "person@example\\.com",
            "replacement": "[REDACTED]"
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-mark-opt-out");
    event(
        EmitMarkEventParams::builder()
            .name("raw-mark")
            .data(json!({"email": "person@example.com"}))
            .build(),
    )
    .unwrap();
    let captured = captured_events_snapshot(&events);
    assert_eq!(captured[0].data().unwrap()["email"], "person@example.com");

    deregister_subscriber("pii-redaction-mark-opt-out").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn sanitized_trajectory_content_never_reaches_subscribers_or_exporters() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "codec": "openai_chat",
        "profiles": [
            {
                "mode": "builtin",
                "priority": 80,
                "builtin": {
                    "preset": "trajectory_context",
                    "custom_mark_payload_policy": "redact_all_leaves"
                }
            },
            {
                "mode": "builtin",
                "priority": 90,
                "builtin": {"action": "redact", "detector": "email"}
            }
        ]
    }))))
    .unwrap();

    let captured = capture_events("pii-regression-subscriber");
    let output = tempfile::tempdir().unwrap();
    let atof = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(output.path())
            .with_filename("events.jsonl"),
    )
    .unwrap();
    atof.register("pii-regression-atof").unwrap();
    let atif = AtifExporter::new(
        "pii-regression".into(),
        AtifAgentInfo {
            name: "test-agent".into(),
            version: "1".into(),
            model_name: None,
            tool_definitions: None,
            extra: None,
        },
    );
    register_subscriber("pii-regression-atif", atif.subscriber()).unwrap();

    let otel_exporter = InMemorySpanExporterBuilder::new().build();
    let otel_provider = SdkTracerProvider::builder()
        .with_simple_exporter(otel_exporter.clone())
        .build();
    let otel = OpenTelemetrySubscriber::from_tracer_provider(otel_provider, "pii-regression");
    otel.register("pii-regression-otel").unwrap();

    let openinference_exporter = InMemorySpanExporterBuilder::new().build();
    let openinference_provider = SdkTracerProvider::builder()
        .with_simple_exporter(openinference_exporter.clone())
        .build();
    let openinference = OpenTelemetrySubscriber::from_tracer_provider_with_type(
        openinference_provider,
        "pii-regression",
        OpenTelemetryType::OpenInference,
    );
    openinference
        .register("pii-regression-openinference")
        .unwrap();

    let raw_pii = "person@example.com";
    let raw_context = "private trajectory context canary";
    let trusted_scope_metadata = json!({
        "nemo_relay_scope_role": "session",
        "agent_kind": "hermes",
        "hook_event_name": "SessionStart",
        "gateway_config_profile": "development",
        "gateway_mode": "passthrough",
        "turn_source": "user_prompt",
        "harness": "hermes",
        "source": "hook",
        "identity_quality": "native",
        "gateway_path": "responses",
        "llm_correlation_status": "matched",
        "llm_correlation_source": "provider",
        "tool_correlation_status": "matched",
        "tool_correlation_source": "provider",
        "otel.status_code": "OK",
        "fidelity_source": "provider",
        "provider_payload_exact": true,
        "session_owner": raw_pii
    });
    let agent = push_scope(
        PushScopeParams::builder()
            .name("hermes-agent")
            .scope_type(ScopeType::Agent)
            .input(json!({"prompt": raw_context, "request_id": "request-1"}))
            .metadata(trusted_scope_metadata.clone())
            .build(),
    )
    .unwrap();
    event(
        EmitMarkEventParams::builder()
            .name("hermes.checkpoint")
            .data(json!({"content": raw_context, "email": raw_pii, "score": 0.95}))
            .metadata(json!({"reviewer": raw_pii}))
            .severity(LogSeverity::Warn)
            .build(),
    )
    .unwrap();
    pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&agent.uuid)
            .output(json!({"answer": raw_context}))
            .metadata(json!({"approver": raw_pii}))
            .build(),
    )
    .unwrap();

    crate::api::subscriber::flush_subscribers().unwrap();
    atof.force_flush().unwrap();
    let trajectory = atif.export().unwrap();
    otel.force_flush().unwrap();
    openinference.force_flush().unwrap();

    let subscriber_json = serde_json::to_string(&captured_events_snapshot(&captured)).unwrap();
    let atof_json = std::fs::read_to_string(atof.path().expect("file sink path")).unwrap();
    let atif_json = serde_json::to_string(&trajectory).unwrap();
    let otel_debug = format!("{:?}", otel_exporter.get_finished_spans().unwrap());
    let openinference_debug = format!("{:?}", openinference_exporter.get_finished_spans().unwrap());
    for (surface, output, retains_scope_metadata) in [
        ("subscriber", subscriber_json, true),
        ("ATOF", atof_json, true),
        ("ATIF", atif_json, false),
        ("OpenTelemetry", otel_debug, true),
        ("OpenInference", openinference_debug, true),
    ] {
        assert!(
            !output.contains(raw_pii),
            "raw PII leaked through {surface}: {output}"
        );
        assert!(
            !output.contains(raw_context),
            "trajectory context leaked through {surface}: {output}"
        );
        if retains_scope_metadata {
            for (key, value) in trusted_scope_metadata
                .as_object()
                .unwrap()
                .iter()
                .filter(|(key, _)| *key != "session_owner")
            {
                assert!(
                    output.contains(key) && output.contains(value.to_string().trim_matches('"')),
                    "trusted scope metadata {key} was not retained in {surface}: {output}"
                );
            }
        }
    }

    let captured = captured_events_snapshot(&captured);
    let agent_start = captured
        .iter()
        .find(|event| {
            event.name() == "hermes-agent" && event.scope_category() == Some(ScopeCategory::Start)
        })
        .unwrap();
    assert_eq!(agent_start.data().unwrap()["request_id"], "request-1");
    assert_eq!(agent_start.data().unwrap()["prompt"], "[REDACTED]");
    let agent_metadata = agent_start.metadata().unwrap();
    for (key, value) in trusted_scope_metadata
        .as_object()
        .unwrap()
        .iter()
        .filter(|(key, _)| *key != "session_owner")
    {
        assert_eq!(agent_metadata[key], *value, "{key}");
    }
    assert_eq!(agent_metadata["session_owner"], "[REDACTED]");
    let custom_mark = captured
        .iter()
        .find(|event| event.name() == "hermes.checkpoint")
        .unwrap();
    assert_eq!(custom_mark.data().unwrap()["content"], "[REDACTED]");
    assert_eq!(custom_mark.data().unwrap()["score"], 0);
    assert_eq!(
        custom_mark.metadata().unwrap(),
        &json!({
            LOG_SEVERITY_METADATA_KEY: "warn",
            "reviewer": "[REDACTED]"
        })
    );

    deregister_subscriber("pii-regression-subscriber").unwrap();
    atof.deregister("pii-regression-atof").unwrap();
    deregister_subscriber("pii-regression-atif").unwrap();
    otel.deregister("pii-regression-otel").unwrap();
    openinference
        .deregister("pii-regression-openinference")
        .unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn trajectory_preset_sanitizes_stream_finalization_without_changing_client_chunks() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "codec": "openai_chat",
        "profiles": [{
            "mode": "builtin",
            "builtin": {"preset": "trajectory_context"}
        }]
    })))
    .await
    .unwrap();

    let events = capture_events("pii-trajectory-stream");
    let raw_delta = "private streaming delta";
    let provider: LlmStreamExecutionNextFn = Arc::new(move |_request| {
        Box::pin(async move {
            Ok(LlmJsonStream::new(futures::stream::iter(vec![Ok(json!({
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": {"content": raw_delta}, "finish_reason": null}]
            }))])))
        })
    });
    let request_codec: Arc<dyn LlmCodec> = Arc::new(OpenAIChatCodec);
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);
    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("openai")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "private stream prompt"}]
                }),
            })
            .func(provider)
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| {
                json!({
                    "id": "chatcmpl-stream",
                    "model": "gpt-4o-mini",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "private final answer"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
                })
            }))
            .codec(request_codec)
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let chunk = stream.next().await.unwrap().unwrap();
    assert_eq!(chunk["choices"][0]["delta"]["content"], raw_delta);
    assert!(stream.next().await.is_none());

    let captured = captured_events_snapshot(&events);
    let start = captured
        .iter()
        .find(|event| event.scope_category() == Some(ScopeCategory::Start))
        .unwrap();
    assert_eq!(
        start.input().unwrap()["content"]["messages"][0]["content"],
        "[REDACTED]"
    );
    let chunk_mark = captured
        .iter()
        .find(|event| event.name() == "llm.chunk")
        .unwrap();
    assert_eq!(chunk_mark.data().unwrap()["chunk_index"], 0);
    assert!(!chunk_mark.to_json_string().unwrap().contains(raw_delta));
    let end = captured
        .iter()
        .find(|event| event.scope_category() == Some(ScopeCategory::End))
        .unwrap();
    assert_eq!(
        end.output().unwrap()["choices"][0]["message"]["content"],
        "[REDACTED]"
    );
    assert_eq!(end.output().unwrap()["usage"]["total_tokens"], 11);

    deregister_subscriber("pii-trajectory-stream").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn normalized_paths_use_the_active_codec_for_stream_finalization() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_responses",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/message"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-active-codec-stream-finalization");
    let provider: LlmStreamExecutionNextFn = Arc::new(move |_request| {
        Box::pin(async move {
            Ok(LlmJsonStream::new(futures::stream::iter(vec![Ok(json!({
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "sk-client-visible"}}]
            }))])))
        })
    });
    let request_codec: Arc<dyn LlmCodec> = Arc::new(OpenAIChatCodec);
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);
    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("openai")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}),
            })
            .func(provider)
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| {
                json!({
                    "id": "chatcmpl-stream",
                    "model": "gpt-4o-mini",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "sk-stream-secret"},
                        "finish_reason": "stop"
                    }]
                })
            }))
            .codec(request_codec)
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(
        stream.next().await.unwrap().unwrap()["choices"][0]["delta"]["content"],
        json!("sk-client-visible")
    );
    assert!(stream.next().await.is_none());

    let captured = captured_events_snapshot(&events);
    let end = captured
        .iter()
        .find(|event| event.scope_category() == Some(ScopeCategory::End))
        .unwrap();
    assert_eq!(
        end.output().unwrap()["choices"][0]["message"]["content"],
        json!("[REDACTED]")
    );
    assert_eq!(
        end.annotated_response()
            .and_then(|response| response.response_text()),
        Some("[REDACTED]")
    );

    deregister_subscriber("pii-active-codec-stream-finalization").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_backend_sanitizes_tool_start_and_end_payloads_with_preorder_targets() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": true,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/api_key", "/nested/token", "/result/secret", "/owner", "/reviewer"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-tool-events");
    let handle = tool_call(
        ToolCallParams::builder()
            .name("search")
            .args(json!({
                "api_key": "sk-abc123",
                "nested": {
                    "token": "sk-secret",
                    "note": "leave me"
                }
            }))
            .metadata(json!({"owner": "sk-universal-metadata"}))
            .build(),
    )
    .unwrap();
    tool_call_end(
        ToolCallEndParams::builder()
            .handle(&handle)
            .execution_result(
                json!({
                    "result": {
                        "secret": "sk-final",
                        "public": "ok"
                    }
                })
                .into(),
            )
            .metadata(json!({"reviewer": "sk-universal-metadata"}))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "api_key": "[REDACTED]",
            "nested": {
                "token": "[REDACTED]",
                "note": "leave me"
            }
        }))
    );
    assert_eq!(
        captured_events[1].output(),
        Some(&json!({
            "result": {
                "secret": "[REDACTED]",
                "public": "ok"
            }
        }))
    );
    assert_eq!(
        captured_events[0].metadata().unwrap()["owner"],
        "[REDACTED]"
    );
    assert_eq!(
        captured_events[1].metadata().unwrap()["reviewer"],
        "[REDACTED]"
    );

    deregister_subscriber("pii-redaction-tool-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_remove_deletes_object_fields_and_nulls_array_or_root_targets() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": true,
        "builtin": {
            "action": "remove",
            "target_paths": ["/secret", "/nested/remove_me", "/items/1", "/result/token"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-remove-events");
    let handle = tool_call(
        ToolCallParams::builder()
            .name("search")
            .args(json!({
                "secret": "abc",
                "nested": {
                    "keep": "yes",
                    "remove_me": "gone"
                },
                "items": ["a", "b", "c"]
            }))
            .build(),
    )
    .unwrap();
    tool_call_end(
        ToolCallEndParams::builder()
            .handle(&handle)
            .execution_result(
                json!({
                    "result": {
                        "token": "drop-me",
                        "public": "ok"
                    }
                })
                .into(),
            )
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "nested": {
                "keep": "yes"
            },
            "items": ["a", null, "c"]
        }))
    );
    assert_eq!(
        captured_events[1].output(),
        Some(&json!({
            "result": {
                "public": "ok"
            }
        }))
    );

    deregister_subscriber("pii-redaction-remove-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_remove_without_target_paths_is_rejected() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    let config = plugin_config(json!({
        "mode": "builtin",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "remove"
        }
    }));
    let report = validate_plugin_config(&config);
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("builtin.target_paths")
            && diagnostic.message.contains("requires at least one")
    }));

    let activation = futures::executor::block_on(initialize_plugins(config));
    let error = activation.expect_err("unscoped remove must fail plugin activation");
    assert!(error.to_string().contains("requires at least one"));
}

#[test]
fn builtin_remove_deletes_targeted_object_and_array_container_fields() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "remove",
            "target_paths": ["/nested", "/items"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-remove-container-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("search")
            .args(json!({
                "nested": {
                    "keep": "yes",
                    "remove_me": "gone"
                },
                "items": ["a", "b", "c"],
                "public": "ok"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "public": "ok"
        }))
    );

    deregister_subscriber("pii-redaction-remove-container-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_redact_replaces_matching_tool_payload_substrings_with_default_token() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "tool_input": true,
        "tool_output": true,
        "input": false,
        "output": false,
        "builtin": {
            "action": "redact",
            "detector": "bearer_token",
            "target_paths": []
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-redact-tool-events");
    let secret = "Bearer sk-demo-secret-123456";
    let handle = tool_call(
        ToolCallParams::builder()
            .name("redact_tool")
            .args(json!({
                "auth": secret,
                "message": format!("primary auth={secret}")
            }))
            .build(),
    )
    .unwrap();
    tool_call_end(
        ToolCallEndParams::builder()
            .handle(&handle)
            .execution_result(
                json!({
                    "result": secret,
                    "nested": {"token": secret}
                })
                .into(),
            )
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(
        captured_events[0].input().unwrap()["auth"],
        json!("[REDACTED]")
    );
    assert_eq!(
        captured_events[0].input().unwrap()["message"],
        json!("primary auth=[REDACTED]")
    );
    assert_eq!(
        captured_events[1].output().unwrap()["result"],
        json!("[REDACTED]")
    );
    assert_eq!(
        captured_events[1].output().unwrap()["nested"]["token"],
        json!("[REDACTED]")
    );

    deregister_subscriber("pii-redaction-redact-tool-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_preserves_configured_prefix_and_suffix() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": true,
        "builtin": {
            "action": "mask",
            "mask_char": "*",
            "unmasked_prefix": 2,
            "unmasked_suffix": 2,
            "target_paths": ["/account", "/result/token"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-mask-events");
    let handle = tool_call(
        ToolCallParams::builder()
            .name("lookup")
            .args(json!({
                "account": "abcdef1234",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();
    tool_call_end(
        ToolCallEndParams::builder()
            .handle(&handle)
            .execution_result(
                json!({
                    "result": {
                        "token": "9876543210",
                        "public": "ok"
                    }
                })
                .into(),
            )
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "account": "ab******34",
            "keep": "unchanged"
        }))
    );
    assert_eq!(
        captured_events[1].output(),
        Some(&json!({
            "result": {
                "token": "98******10",
                "public": "ok"
            }
        }))
    );

    deregister_subscriber("pii-redaction-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_detector_masks_only_matching_substrings() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "email",
            "mask_char": "*",
            "target_paths": ["/message"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-detector-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "message": "Email alice@example.com or bob@example.com",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "message": "Email a****@example.com or b**@example.com",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-detector-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_email_detector_preserves_domain_by_default() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "email",
            "target_paths": ["/contact"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-email-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "contact": "alice@example.com",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "contact": "a****@example.com",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-email-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_phone_detector_preserves_last_four_digits_by_default() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "phone",
            "target_paths": ["/phone"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-phone-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "phone": "+1 (555) 123-4567",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "phone": "+* (***) ***-4567",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-phone-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_api_key_detector_preserves_prefix_and_last_four_by_default() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "api_key",
            "target_paths": ["/api_key"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-api-key-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "api_key": "sk-abcdef123456",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "api_key": "sk-********3456",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-api-key-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_detector_uses_explicit_prefix_suffix_over_defaults() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "email",
            "unmasked_prefix": 2,
            "unmasked_suffix": 2,
            "target_paths": ["/contact"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-detector-explicit-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "contact": "alice@example.com",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "contact": "al*************om",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-detector-explicit-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_ip_address_detector_preserves_last_octet_by_default() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "ip_address",
            "target_paths": ["/ip"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-ip-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "ip": "192.168.10.42",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "ip": "***.***.***.42",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-ip-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_url_detector_preserves_scheme_and_host_by_default() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "url",
            "target_paths": ["/url"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-url-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "url": "https://example.com/path?q=1",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "url": "https://example.com/*",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-url-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_ipv6_detector_preserves_last_segment_by_default() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "ipv6",
            "target_paths": ["/ip"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-ipv6-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "ip": "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "ip": "****:****:****:****:****:****:****:7334",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-ipv6-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_ipv6_detector_supports_compressed_addresses() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "ipv6",
            "target_paths": ["/ip"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-ipv6-compressed-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "ip": "2001:db8::1",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "ip": "****:****::1",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-ipv6-compressed-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn mask_text_handles_extreme_unmasked_bounds_without_overflow() {
    let masked = mask_text("secret", "*", usize::MAX, 4);
    assert_eq!(masked, "secret");
}

#[test]
fn builtin_mask_with_bearer_token_detector_preserves_scheme_and_last_four() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "bearer_token",
            "target_paths": ["/auth"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-bearer-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "auth": "Bearer token-value-1234",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "auth": "Bearer ************1234",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-bearer-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_bearer_token_detector_ignores_short_benign_values() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "redact",
            "detector": "bearer_token",
            "target_paths": ["/auth"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-bearer-short-benign-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "auth": "Bearer token",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "auth": "Bearer token",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-bearer-short-benign-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_credit_card_detector_preserves_last_four_digits() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "credit_card",
            "target_paths": ["/card"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-credit-card-default-mask-events");
    let credit_card = ["4111", "1111", "1111", "1234"].join(" ");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "card": credit_card,
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "card": "**** **** **** 1234",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-credit-card-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_ip_detector_honors_custom_mask_char() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "ip_address",
            "mask_char": "#",
            "target_paths": ["/ip"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-ip-custom-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "ip": "10.20.30.40"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input().unwrap()["ip"],
        json!("###.###.###.40")
    );

    deregister_subscriber("pii-redaction-ip-custom-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_jwt_detector_preserves_header_and_signature_tail() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    let jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.cGF5bG9hZA.signaturetail";
    let expected_jwt = {
        let parts = jwt.split('.').collect::<Vec<_>>();
        format!(
            "{}.{}.{}",
            parts[0],
            mask_text(parts[1], "*", 0, 0),
            mask_text(parts[2], "*", 0, 6)
        )
    };
    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "jwt",
            "target_paths": ["/token"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-jwt-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "token": jwt,
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "token": expected_jwt,
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-jwt-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_cloud_key_detectors_preserves_expected_segments() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "aws_access_key_id",
            "target_paths": ["/key"]
        }
    }))))
    .unwrap();
    let events = capture_events("pii-redaction-aws-access-key-mask-events");
    let aws_access_key = "AKIAIOSFODNN7EXAMPLE";
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({"key": aws_access_key}))
            .build(),
    )
    .unwrap();
    assert_eq!(
        captured_events_snapshot(&events)[0].input(),
        Some(&json!({"key": mask_text(aws_access_key, "*", 4, 4)}))
    );
    deregister_subscriber("pii-redaction-aws-access-key-mask-events").unwrap();
    clear_plugin_configuration().unwrap();

    reset_runtime();
    setup_isolated_thread();
    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "gcp_api_key",
            "target_paths": ["/key"]
        }
    }))))
    .unwrap();
    let events = capture_events("pii-redaction-gcp-key-mask-events");
    let gcp_key = format!("AIza{}", "A".repeat(35));
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({"key": gcp_key}))
            .build(),
    )
    .unwrap();
    assert_eq!(
        captured_events_snapshot(&events)[0].input(),
        Some(&json!({"key": mask_text(&gcp_key, "*", 6, 4)}))
    );
    deregister_subscriber("pii-redaction-gcp-key-mask-events").unwrap();
    clear_plugin_configuration().unwrap();

    reset_runtime();
    setup_isolated_thread();
    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "azure_storage_account_key",
            "target_paths": ["/key"]
        }
    }))))
    .unwrap();
    let events = capture_events("pii-redaction-azure-storage-key-mask-events");
    let azure_key = format!("{}==", "A".repeat(86));
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({"key": azure_key}))
            .build(),
    )
    .unwrap();
    assert_eq!(
        captured_events_snapshot(&events)[0].input(),
        Some(&json!({"key": mask_text(&azure_key, "*", 0, 4)}))
    );
    deregister_subscriber("pii-redaction-azure-storage-key-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_hash_with_detector_hashes_only_matching_substrings() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "hash",
            "detector": "email",
            "target_paths": ["/message"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-detector-hash-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "message": "Email alice@example.com please",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "message": format!(
                "Email {} please",
                hex_sha256("alice@example.com")
            ),
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-detector-hash-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_short_detector_match_leaves_value_unchanged() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "email",
            "target_paths": ["/contact"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-short-detector-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "contact": "a@example.com",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "contact": "a@example.com",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-short-detector-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_empty_target_paths_sanitizes_all_matching_string_leaves() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "email"
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-empty-target-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "primary": "alice@example.com",
                "nested": {
                    "secondary": "bob@example.com",
                    "note": "no pii here"
                },
                "items": ["carol@example.com", "safe text"]
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "primary": "a****@example.com",
            "nested": {
                "secondary": "b**@example.com",
                "note": "no pii here"
            },
            "items": ["c****@example.com", "safe text"]
        }))
    );

    deregister_subscriber("pii-redaction-empty-target-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_malformed_ip_or_url_detector_input_leaves_value_unchanged() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "ip_address",
            "target_paths": ["/ip", "/url"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-malformed-detector-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "ip": "not-an-ip",
                "url": "mailto:alice@example.com",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "ip": "not-an-ip",
            "url": "mailto:alice@example.com",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-malformed-detector-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_mask_with_detector_sanitizes_llm_response_from_normalized_message_path() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "email",
            "target_paths": ["/message"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-detector-llm-response-events");
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let _ = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}),
            })
            .func(noop_openai_chat_exec_fn(json!({
                "id": "chatcmpl-123",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Reach me at alice@example.com"},
                        "finish_reason": "stop"
                    }
                ]
            })))
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(
        captured_events[1].output().unwrap()["choices"][0]["message"]["content"],
        json!("Reach me at a****@example.com")
    );
    assert_eq!(
        captured_events[1]
            .annotated_response()
            .and_then(|response| response.response_text()),
        Some("Reach me at a****@example.com")
    );

    deregister_subscriber("pii-redaction-detector-llm-response-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_hash_with_detector_hashes_multiple_matches_in_one_string() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "hash",
            "detector": "email",
            "target_paths": ["/message"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-multi-detector-hash-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "message": "alice@example.com and bob@example.com",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "message": format!(
                "{} and {}",
                hex_sha256("alice@example.com"),
                hex_sha256("bob@example.com")
            ),
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-multi-detector-hash-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_empty_target_paths_handles_arrays_and_multiple_detector_types() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "url"
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-array-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "items": [
                    "https://example.com/a",
                    "safe text",
                    {"nested": "http://nvidia.com/private/path"},
                    42
                ],
                "keep": "mailto:alice@example.com"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "items": [
                "https://example.com/*",
                "safe text",
                {"nested": "http://nvidia.com/*"},
                42
            ],
            "keep": "mailto:alice@example.com"
        }))
    );

    deregister_subscriber("pii-redaction-array-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_detector_sanitizes_tool_output_payloads() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": false,
        "tool_output": true,
        "builtin": {
            "action": "mask",
            "detector": "email",
            "target_paths": ["/result/contact"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-tool-output-mask-events");
    let handle = tool_call(
        ToolCallParams::builder()
            .name("lookup")
            .args(json!({"query": "alice"}))
            .build(),
    )
    .unwrap();
    tool_call_end(
        ToolCallEndParams::builder()
            .handle(&handle)
            .execution_result(
                json!({
                    "result": {
                        "contact": "alice@example.com",
                        "public": "ok"
                    }
                })
                .into(),
            )
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert_eq!(
        captured_events[1].output(),
        Some(&json!({
            "result": {
                "contact": "a****@example.com",
                "public": "ok"
            }
        }))
    );

    deregister_subscriber("pii-redaction-tool-output-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_tool_output_sanitizes_annotation_as_an_independent_json_boundary() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "input": false,
        "output": false,
        "tool_input": false,
        "tool_output": true,
        "builtin": {
            "action": "mask",
            "detector": "email",
            "target_paths": ["/contact"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-tool-annotation-mask-events");
    let execution_result = futures::executor::block_on(tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("lookup")
            .args(json!({"query": "alice"}))
            .func(Arc::new(|_args| {
                Box::pin(async {
                    Ok(ToolExecutionResult::annotated(
                        json!({
                            "contact": "alice@example.com",
                            "outside": "alice@example.com"
                        }),
                        json!({
                            "contact": "alice@example.com",
                            "outside": "alice@example.com"
                        }),
                    ))
                })
            }))
            .build(),
    ))
    .unwrap();
    assert_eq!(
        execution_result,
        ToolExecutionResult::annotated(
            json!({
                "contact": "alice@example.com",
                "outside": "alice@example.com"
            }),
            json!({
                "contact": "alice@example.com",
                "outside": "alice@example.com"
            }),
        )
    );

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert_eq!(
        captured_events[1].output(),
        Some(&json!({
            "contact": "a****@example.com",
            "outside": "alice@example.com"
        }))
    );
    assert_eq!(
        captured_events[1].tool_result_annotation().unwrap(),
        json!({
            "contact": "a****@example.com",
            "outside": "alice@example.com"
        })
    );

    deregister_subscriber("pii-redaction-tool-annotation-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_disabled_tool_output_preserves_tool_result_annotation() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "email",
            "target_paths": ["/contact"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-disabled-tool-annotation-events");
    let handle = tool_call(
        ToolCallParams::builder()
            .name("lookup")
            .args(json!({"query": "alice"}))
            .build(),
    )
    .unwrap();
    let annotation = json!({"contact": "alice@example.com"});
    tool_call_end(
        ToolCallEndParams::builder()
            .handle(&handle)
            .execution_result(ToolExecutionResult::annotated(
                json!({"contact": "alice@example.com"}),
                annotation.clone(),
            ))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert_eq!(
        captured_events[1].tool_result_annotation().unwrap(),
        annotation
    );

    deregister_subscriber("pii-redaction-disabled-tool-annotation-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_root_remove_drops_tool_result_annotation() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "input": false,
        "output": false,
        "tool_input": false,
        "tool_output": true,
        "builtin": {
            "action": "remove",
            "target_paths": [""]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-remove-tool-annotation-events");
    let handle = tool_call(
        ToolCallParams::builder()
            .name("lookup")
            .args(json!({"query": "alice"}))
            .tool_call_id("provider-call-123")
            .build(),
    )
    .unwrap();
    tool_call_end(
        ToolCallEndParams::builder()
            .handle(&handle)
            .execution_result(ToolExecutionResult::annotated(
                json!({"contact": "alice@example.com"}),
                json!({"contact": "alice@example.com"}),
            ))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert_eq!(captured_events[1].tool_call_id(), Some("provider-call-123"));
    assert!(captured_events[1].tool_result_annotation().is_none());

    deregister_subscriber("pii-redaction-remove-tool-annotation-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_phone_detector_ignores_non_matching_digit_shapes() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "phone",
            "target_paths": ["/value"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-phone-false-positive-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "value": "Order 12345 is ready",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "value": "Order 12345 is ready",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-phone-false-positive-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_backend_sanitizes_llm_start_payload_via_codec_and_reencodes_provider_shape() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": true,
        "output": false,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/messages/0/content", "/messages/1/content", "/audit_owner"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-llm-events");
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "sk-system-secret"},
                {"role": "user", "content": "sk-user-secret"}
            ],
            "temperature": 0.2
        }),
    };

    let _handle = llm_call(
        LlmCallParams::builder()
            .name("openai")
            .request(&request)
            .metadata(json!({"audit_owner": "sk-universal-metadata"}))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "headers": {},
            "content": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "[REDACTED]"},
                    {"role": "user", "content": "[REDACTED]"}
                ],
                "temperature": 0.2
            }
        }))
    );
    assert_eq!(
        captured_events[0].metadata().unwrap()["audit_owner"],
        "[REDACTED]"
    );

    deregister_subscriber("pii-redaction-llm-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_removes_targeted_message_names_and_ignores_missing_normalized_paths() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": true,
        "output": false,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "remove",
            "target_paths": [
                "/messages/0/missing",
                "/messages/0/name",
                "/messages/1/name"
            ]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-remove-message-names");
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "name": "SECRET-SYSTEM", "content": "instructions"},
                {"role": "user", "name": "SECRET-USER", "content": "hello"}
            ]
        }),
    };

    llm_call(
        LlmCallParams::builder()
            .name("openai")
            .request(&request)
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "headers": {},
            "content": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "instructions"},
                    {"role": "user", "content": "hello"}
                ]
            }
        }))
    );
    assert!(
        !serde_json::to_string(&captured_events[0])
            .unwrap()
            .contains("SECRET-")
    );

    deregister_subscriber("pii-redaction-remove-message-names").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_gemini_generate_content_via_codec() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "gemini_generate_content",
        "input": true,
        "output": false,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/messages/0/content"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-gemini-generate-content");
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "contents": [{"role": "user", "parts": [{"text": "sk-gemini-secret"}]}],
        }),
    };

    let _handle = llm_call(
        LlmCallParams::builder()
            .name("gemini-generate-content")
            .request(&request)
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "headers": {},
            "content": {
                "contents": [{"role": "user", "parts": [{"text": "[REDACTED]"}]}]
            }
        }))
    );

    deregister_subscriber("pii-redaction-gemini-generate-content").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_gemini_provider_native_tools_via_codec() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "gemini_generate_content",
        "input": true,
        "output": false,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/tools/1/value/googleSearch/apiKey"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-gemini-native-tools");
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "tools": [{
                "functionDeclarations": [{"name": "lookup"}],
                "googleSearch": {"apiKey": "sk-tool-secret"}
            }]
        }),
    };

    let _handle = llm_call(
        LlmCallParams::builder()
            .name("gemini_generate_content")
            .request(&request)
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "headers": {},
            "content": {
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "tools": [{
                    "functionDeclarations": [{"name": "lookup"}],
                    "googleSearch": {"apiKey": "[REDACTED]"}
                }]
            }
        }))
    );

    deregister_subscriber("pii-redaction-gemini-native-tools").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_gemini_provider_native_request_content_via_codec() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "gemini_generate_content",
        "input": true,
        "output": false,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/messages/0/content/1/value/fileData/fileUri"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-gemini-native-request-content");
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "summarize this file"},
                    {
                        "fileData": {
                            "mimeType": "text/plain",
                            "fileUri": "https://example.test/sk-file-secret"
                        }
                    }
                ]
            }]
        }),
    };

    let _handle = llm_call(
        LlmCallParams::builder()
            .name("gemini_generate_content")
            .request(&request)
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input().unwrap()["content"]["contents"][0]["parts"][1]["fileData"]["fileUri"],
        json!("https://example.test/[REDACTED]")
    );
    assert!(
        !serde_json::to_string(&captured_events[0])
            .unwrap()
            .contains("sk-file-secret")
    );

    deregister_subscriber("pii-redaction-gemini-native-request-content").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_gemini_function_response_nested_parts_via_codec() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "gemini_generate_content",
        "input": true,
        "output": false,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/messages/2/content/1/value/inlineData/data"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-gemini-function-response-parts");
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "contents": [
                {"role": "user", "parts": [{"text": "show my ordered instrument"}]},
                {"role": "model", "parts": [{
                    "functionCall": {"id": "call_img", "name": "get_image", "args": {}},
                    "thoughtSignature": "sig_CALL=="
                }]},
                {"role": "user", "parts": [{
                    "functionResponse": {
                        "id": "call_img",
                        "name": "get_image",
                        "response": {"image_ref": {"$ref": "instrument.jpg"}},
                        "parts": [{
                            "inlineData": {
                                "displayName": "instrument.jpg",
                                "mimeType": "image/jpeg",
                                "data": "sk-image-secret"
                            }
                        }]
                    }
                }]}
            ]
        }),
    };

    let _handle = llm_call(
        LlmCallParams::builder()
            .name("gemini_generate_content")
            .request(&request)
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input().unwrap()["content"]["contents"][2]["parts"][0]["functionResponse"]
            ["parts"][0]["inlineData"]["data"],
        json!("[REDACTED]")
    );
    assert!(
        !serde_json::to_string(&captured_events[0])
            .unwrap()
            .contains("sk-image-secret")
    );

    deregister_subscriber("pii-redaction-gemini-function-response-parts").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_gemini_provider_native_response_content_via_codec() {
    use crate::codec::gemini_generate_content::GeminiGenerateContentCodec;
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "gemini_generate_content",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/message/1/value/codeExecutionResult/output"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-gemini-native-response-content");
    let response = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"text": "ran code"},
                    {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "sk-code-secret"}}
                ]
            },
            "finishReason": "STOP",
            "index": 0
        }]
    });

    let _ = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("gemini_generate_content")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({
                    "contents": [{"role": "user", "parts": [{"text": "run code"}]}]
                }),
            })
            .func(noop_openai_chat_exec_fn(response.clone()))
            .response_codec(Arc::new(GeminiGenerateContentCodec))
            .build(),
    )
    .await
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert_eq!(
        captured_events[1].output().unwrap()["candidates"][0]["content"]["parts"][1]["codeExecutionResult"]
            ["output"],
        json!("[REDACTED]")
    );
    assert!(
        !serde_json::to_string(&captured_events[1])
            .unwrap()
            .contains("sk-code-secret")
    );

    deregister_subscriber("pii-redaction-gemini-native-response-content").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_omits_request_and_annotation_for_unsafe_normalized_array_removal() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_responses",
        "input": true,
        "output": false,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "remove",
            "target_paths": ["/messages/0/content/0"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-unsafe-normalized-array-removal");
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4.1-mini",
            "input": [{
                "role": "user",
                "content": [{"type": "input_text", "text": "SECRET-REQUEST"}]
            }]
        }),
    };
    let annotated_request = Arc::new(OpenAIResponsesCodec.decode(&request).unwrap());

    llm_call(
        LlmCallParams::builder()
            .name("openai")
            .request(&request)
            .annotated_request(annotated_request)
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert!(captured_events[0].input().is_none());
    assert!(captured_events[0].annotated_request().is_none());
    assert!(
        !serde_json::to_string(&captured_events[0])
            .unwrap()
            .contains("SECRET-REQUEST")
    );

    deregister_subscriber("pii-redaction-unsafe-normalized-array-removal").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_observable_llm_request_headers() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "input": true,
        "output": false,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "redact",
            "detector": "email"
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-llm-request-headers");
    let request = LlmRequest {
        headers: [("x-user-email".to_string(), json!("alice@example.com"))]
            .into_iter()
            .collect(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    };

    llm_call(
        LlmCallParams::builder()
            .name("openai")
            .request(&request)
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input().unwrap()["headers"]["x-user-email"],
        json!("[REDACTED]")
    );
    assert!(
        !serde_json::to_string(&captured_events[0])
            .unwrap()
            .contains("alice@example.com")
    );

    deregister_subscriber("pii-redaction-llm-request-headers").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn trajectory_preset_redacts_opaque_llm_request_header_values() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "input": true,
        "output": false,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "preset": "trajectory_context"
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-trajectory-llm-request-headers");
    let request = LlmRequest {
        headers: [("model".to_string(), json!("SECRET-HEADER"))]
            .into_iter()
            .collect(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    };

    llm_call(
        LlmCallParams::builder()
            .name("openai")
            .request(&request)
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input().unwrap()["headers"]["model"],
        json!("[REDACTED]")
    );
    assert!(
        !serde_json::to_string(&captured_events[0])
            .unwrap()
            .contains("SECRET-HEADER")
    );

    deregister_subscriber("pii-trajectory-llm-request-headers").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_llm_end_payload_and_response_codec_decodes_sanitized_output() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/choices/0/message/content", "/audit_owner"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-llm-end-events");
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "hello"}
            ]
        }),
    };
    let response = json!({
        "id": "chatcmpl-123",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "sk-response-secret"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5
        }
    });
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(request)
            .func(noop_openai_chat_exec_fn(response.clone()))
            .response_codec(response_codec)
            .metadata(json!({"audit_owner": "sk-universal-metadata"}))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(result, response);

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert_eq!(
        captured_events[1].output(),
        Some(&json!({
            "id": "chatcmpl-123",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "[REDACTED]"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5
            }
        }))
    );

    let annotated = captured_events[1]
        .annotated_response()
        .expect("annotated_response should be present");
    assert_eq!(annotated.response_text(), Some("[REDACTED]"));
    assert_eq!(annotated.model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(
        captured_events[1].metadata().unwrap()["audit_owner"],
        "[REDACTED]"
    );

    deregister_subscriber("pii-redaction-llm-end-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_openai_chat_response_from_normalized_message_path() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/message"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-openai-chat-normalized-response");
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let _ = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}),
            })
            .func(noop_openai_chat_exec_fn(json!({
                "id": "chatcmpl-123",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "sk-chat-secret"},
                        "finish_reason": "stop"
                    }
                ]
            })))
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(
        captured_events[1].output().unwrap()["choices"][0]["message"]["content"],
        json!("[REDACTED]")
    );
    assert_eq!(
        captured_events[1]
            .annotated_response()
            .and_then(|response| response.response_text()),
        Some("[REDACTED]")
    );

    deregister_subscriber("pii-redaction-openai-chat-normalized-response").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_omits_unprojectable_openai_chat_api_specific_response() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "remove",
            "target_paths": ["/api_specific/system_fingerprint"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-openai-chat-api-specific-response");
    let response = json!({
        "id": "chatcmpl-api-specific",
        "model": "gpt-4o-mini",
        "system_fingerprint": "SECRET-FINGERPRINT",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"
        }]
    });

    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hello"}]
                }),
            })
            .func(noop_openai_chat_exec_fn(response.clone()))
            .response_codec(Arc::new(OpenAIChatCodec))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(result, response);
    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert!(captured_events[1].output().is_none());
    assert!(captured_events[1].annotated_response().is_none());
    assert!(
        !serde_json::to_string(&captured_events[1])
            .unwrap()
            .contains("SECRET-FINGERPRINT")
    );

    deregister_subscriber("pii-redaction-openai-chat-api-specific-response").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_omits_multi_choice_openai_chat_normalized_response() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/message"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-openai-chat-multi-choice-response");
    let response = json!({
        "id": "chatcmpl-multi",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "sk-first-secret"},
                "finish_reason": "stop"
            },
            {
                "index": 1,
                "message": {"role": "assistant", "content": "sk-second-secret"},
                "finish_reason": "stop"
            }
        ]
    });

    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hello"}]
                }),
            })
            .func(noop_openai_chat_exec_fn(response.clone()))
            .response_codec(Arc::new(OpenAIChatCodec))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(result, response);
    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert!(captured_events[1].output().is_none());
    assert!(captured_events[1].annotated_response().is_none());
    let serialized_end = serde_json::to_string(&captured_events[1]).unwrap();
    assert!(!serialized_end.contains("sk-first-secret"));
    assert!(!serialized_end.contains("sk-second-secret"));

    deregister_subscriber("pii-redaction-openai-chat-multi-choice-response").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_redact_sanitizes_openai_chat_response_from_detector_path() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "redact",
            "detector": "email",
            "target_paths": ["/message"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-openai-chat-redact-response");
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let _ = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}),
            })
            .func(noop_openai_chat_exec_fn(json!({
                "id": "chatcmpl-redact-123",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "alice@example.com"},
                        "finish_reason": "stop"
                    }
                ]
            })))
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(
        captured_events[1].output().unwrap()["choices"][0]["message"]["content"],
        json!("[REDACTED]")
    );
    assert_eq!(
        captured_events[1]
            .annotated_response()
            .and_then(|response| response.response_text()),
        Some("[REDACTED]")
    );

    deregister_subscriber("pii-redaction-openai-chat-redact-response").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_anthropic_response_from_normalized_message_path() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "anthropic_messages",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/message"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-anthropic-normalized-response");
    let response_codec: Arc<dyn LlmResponseCodec> =
        Arc::new(crate::codec::anthropic::AnthropicMessagesCodec);

    let _ = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("anthropic")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "hello"}]}),
            })
            .func(noop_openai_chat_exec_fn(json!({
                "id": "msg_123",
                "model": "claude-sonnet-4-20250514",
                "role": "assistant",
                "type": "message",
                "content": [{"type": "text", "text": "sk-anthropic-secret"}],
                "stop_reason": "end_turn"
            })))
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(
        captured_events[1].output().unwrap()["content"][0]["text"],
        json!("[REDACTED]")
    );
    assert_eq!(
        captured_events[1]
            .annotated_response()
            .and_then(|response| response.response_text()),
        Some("[REDACTED]")
    );

    deregister_subscriber("pii-redaction-anthropic-normalized-response").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_omits_unprojectable_anthropic_normalized_usage_response() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "anthropic_messages",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "remove",
            "target_paths": ["/usage/prompt_tokens"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-anthropic-normalized-usage-response");
    let response = json!({
        "id": "msg_usage",
        "model": "claude-sonnet-4-20250514",
        "role": "assistant",
        "type": "message",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 123, "output_tokens": 4}
    });

    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("anthropic")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({
                    "model": "claude-sonnet-4-20250514",
                    "messages": [{"role": "user", "content": "hello"}]
                }),
            })
            .func(noop_openai_chat_exec_fn(response.clone()))
            .response_codec(Arc::new(crate::codec::anthropic::AnthropicMessagesCodec))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(result, response);
    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert!(captured_events[1].output().is_none());
    assert!(captured_events[1].annotated_response().is_none());

    deregister_subscriber("pii-redaction-anthropic-normalized-usage-response").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_openai_responses_response_from_normalized_message_path() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/message"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-openai-responses-normalized-response");
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIResponsesCodec);

    let _ = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"model": "gpt-4.1-mini", "input": "hello"}),
            })
            .func(noop_openai_chat_exec_fn(json!({
                "id": "resp_123",
                "model": "gpt-4.1-mini",
                "status": "completed",
                "output_text": "sk-responses-secret",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "sk-responses-secret"}
                        ]
                    }
                ]
            })))
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(
        captured_events[1].output().unwrap()["output"][0]["content"][0]["text"],
        json!("[REDACTED]")
    );
    assert_eq!(
        captured_events[1].output().unwrap()["output_text"],
        json!("[REDACTED]")
    );
    assert_eq!(
        captured_events[1]
            .annotated_response()
            .and_then(|response| response.response_text()),
        Some("[REDACTED]")
    );
    assert!(
        !captured_events[1]
            .to_json_string()
            .unwrap()
            .contains("sk-responses-secret")
    );

    deregister_subscriber("pii-redaction-openai-responses-normalized-response").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_openai_responses_output_text_alias_on_stream_finalization() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/message"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-openai-responses-output-text-stream");
    let provider: LlmStreamExecutionNextFn = Arc::new(move |_request| {
        Box::pin(async move {
            Ok(LlmJsonStream::new(futures::stream::iter(vec![Ok(json!({
                "type": "response.output_text.delta",
                "delta": "sk-client-visible"
            }))])))
        })
    });
    let request_codec: Arc<dyn LlmCodec> = Arc::new(OpenAIResponsesCodec);
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIResponsesCodec);
    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("openai")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({
                    "model": "gpt-4.1-mini",
                    "input": "hello"
                }),
            })
            .func(provider)
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| {
                json!({
                    "id": "resp_stream",
                    "model": "gpt-4.1-mini",
                    "status": "completed",
                    "output_text": "sk-stream-secret",
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": "sk-stream-secret"
                        }]
                    }]
                })
            }))
            .codec(request_codec)
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(
        stream.next().await.unwrap().unwrap()["delta"],
        json!("sk-client-visible")
    );
    assert!(stream.next().await.is_none());

    let captured_events = captured_events_snapshot(&events);
    let end = captured_events
        .iter()
        .find(|event| event.scope_category() == Some(ScopeCategory::End))
        .unwrap();
    assert_eq!(
        end.output().unwrap()["output"][0]["content"][0]["text"],
        json!("[REDACTED]")
    );
    assert_eq!(end.output().unwrap()["output_text"], json!("[REDACTED]"));
    assert_eq!(
        end.annotated_response()
            .and_then(|response| response.response_text()),
        Some("[REDACTED]")
    );
    assert!(!end.to_json_string().unwrap().contains("sk-stream-secret"));

    deregister_subscriber("pii-redaction-openai-responses-output-text-stream").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_omits_multi_candidate_gemini_normalized_response() {
    use crate::codec::gemini_generate_content::GeminiGenerateContentCodec;
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "gemini_generate_content",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/message"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-gemini-multi-candidate-response");
    let response = json!({
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "sk-first-secret"}]},
                "finishReason": "STOP",
                "index": 0
            },
            {
                "content": {"role": "model", "parts": [{"text": "sk-second-secret"}]},
                "finishReason": "STOP",
                "index": 1
            }
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10}
    });

    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("gemini-multi-candidate")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({
                    "contents": [{"role": "user", "parts": [{"text": "hello"}]}]
                }),
            })
            // noop_openai_chat_exec_fn is codec-agnostic: it returns whatever JSON is passed.
            .func(noop_openai_chat_exec_fn(response.clone()))
            .response_codec(Arc::new(GeminiGenerateContentCodec))
            .build(),
    )
    .await
    .unwrap();

    // The raw response is returned unchanged (fail-closed: no partial redaction).
    assert_eq!(result, response);
    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    // The end event must carry no annotated output (omitted due to multi-candidate).
    assert!(captured_events[1].output().is_none());
    assert!(captured_events[1].annotated_response().is_none());
    // Neither secret must appear in the serialized event.
    let serialized_end = serde_json::to_string(&captured_events[1]).unwrap();
    assert!(!serialized_end.contains("sk-first-secret"));
    assert!(!serialized_end.contains("sk-second-secret"));

    deregister_subscriber("pii-redaction-gemini-multi-candidate-response").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_raw_multi_candidate_gemini_response() {
    use crate::codec::gemini_generate_content::GeminiGenerateContentCodec;
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "gemini_generate_content",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/candidates/1/content/parts/0/text"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-gemini-raw-multi-candidate-response");
    let response = json!({
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "sk-first-secret"}]},
                "finishReason": "STOP",
                "index": 0
            },
            {
                "content": {"role": "model", "parts": [{"text": "sk-second-secret"}]},
                "finishReason": "STOP",
                "index": 1
            }
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10}
    });

    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("gemini-raw-multi-candidate")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({
                    "contents": [{"role": "user", "parts": [{"text": "hello"}]}]
                }),
            })
            .func(noop_openai_chat_exec_fn(response.clone()))
            .response_codec(Arc::new(GeminiGenerateContentCodec))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(
        result, response,
        "PII sanitization must not mutate the caller result"
    );
    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    let output = captured_events[1].output().expect("sanitized output event");
    assert_eq!(
        output["candidates"][0]["content"]["parts"][0]["text"],
        json!("sk-first-secret"),
        "raw targeting must not trigger normalized multi-candidate omission"
    );
    assert_eq!(
        output["candidates"][1]["content"]["parts"][0]["text"],
        json!("[REDACTED]")
    );
    assert!(captured_events[1].annotated_response().is_some());

    deregister_subscriber("pii-redaction-gemini-raw-multi-candidate-response").unwrap();
    clear_plugin_configuration().unwrap();
}
