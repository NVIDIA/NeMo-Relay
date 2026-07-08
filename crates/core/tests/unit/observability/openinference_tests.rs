// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for openinference in the NeMo Relay core crate.

use super::*;
use crate::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
    tool_attributes_to_strings,
};
use crate::api::runtime::NemoRelayContextState;
use crate::api::runtime::global_context;
use crate::api::scope::ScopeType;
use crate::api::scope::{event, pop_scope, push_scope};
use crate::api::tool::ToolAttributes;
use crate::codec::model_pricing::pricing_test_mutex;
use crate::codec::request::{
    AnnotatedLlmRequest, FunctionDefinition, GenerationParams, Message, MessageContent,
    ToolDefinition,
};
use crate::codec::response::{
    AnnotatedLlmResponse, CostEstimate, CostSource, FinishReason, PricingCatalog, PricingResolver,
    ResponseToolCall, Usage, reset_active_pricing_resolver, set_active_pricing_resolver,
};
use crate::json::Json;
use crate::observability::atif::{AtifAgentInfo, AtifExporter, AtifStepExtra};
use opentelemetry::trace::Status;
use opentelemetry_sdk::trace::InMemorySpanExporterBuilder;
use serde_json::json;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::thread;
use uuid::Uuid;

struct ResetPricingResolverGuard;

impl Drop for ResetPricingResolverGuard {
    fn drop(&mut self) {
        let _ = reset_active_pricing_resolver();
    }
}

fn reset_global() {
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

fn make_provider() -> (
    SdkTracerProvider,
    opentelemetry_sdk::trace::InMemorySpanExporter,
) {
    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    (provider, exporter)
}

fn attr_map(attributes: &[KeyValue]) -> HashMap<String, String> {
    attributes
        .iter()
        .map(|attribute| {
            (
                attribute.key.as_str().to_string(),
                attribute.value.to_string(),
            )
        })
        .collect()
}

fn assert_attr(attributes: &HashMap<String, String>, key: &str, value: &str) {
    assert_eq!(attributes.get(key).map(String::as_str), Some(value));
}

fn assert_attr_contains(attributes: &HashMap<String, String>, key: &str, expected: &str) {
    let value = attributes
        .get(key)
        .unwrap_or_else(|| panic!("missing attribute {key}"));
    assert!(
        value.contains(expected),
        "attribute {key} value {value:?} did not contain {expected:?}"
    );
}

fn assert_no_attr_contains(attributes: &HashMap<String, String>, expected: &str) {
    assert!(
        !attributes
            .iter()
            .any(|(key, value)| key.contains(expected) || value.contains(expected)),
        "attribute map unexpectedly contained {expected:?}"
    );
}

fn empty_annotated_request() -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
        messages: Vec::new(),
        model: None,
        params: None,
        tools: None,
        tool_choice: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: serde_json::Map::new(),
    }
}

fn empty_annotated_response() -> AnnotatedLlmResponse {
    AnnotatedLlmResponse {
        id: None,
        model: None,
        message: None,
        tool_calls: None,
        finish_reason: None,
        usage: None,
        optimization_summary: None,
        api_specific: None,
        extra: serde_json::Map::new(),
    }
}

#[test]
fn optimization_summary_emits_namespaced_openinference_attributes() {
    let summary: crate::codec::optimization::LlmOptimizationSummary =
        serde_json::from_value(json!({
            "schema_version":"1", "calculation_version":"1", "status":"complete",
            "baseline_model":{"model":"baseline"}, "effective_model":{"model":"effective"},
            "tokens_saved":{"prompt_tokens":12,"total_tokens":12},
            "baseline_cost":{"total":0.02,"currency":"USD","source":"model_pricing","pricing_as_of":"2026-07-08","pricing_source":"test"},
            "actual_cost":{"total":0.01,"currency":"USD","source":"model_pricing"},
            "estimated_cost_saved":0.01, "currency":"USD", "contributions":[]
        }))
        .unwrap();
    let mut attributes = Vec::new();
    push_optimization_attributes(&mut attributes, &summary);
    let attributes = attr_map(&attributes);
    assert_eq!(
        attributes["nemo_relay.llm.optimization.effective_model"],
        "effective"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.total_tokens_saved"],
        "12"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_cost"],
        "0.01"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.pricing_source"],
        "test"
    );
}

fn install_test_pricing(model_id: &str) {
    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [
                {
                    "provider": "test",
                    "model_id": model_id,
                    "pricing_as_of": "2026-06-05",
                    "pricing_source": "test",
                    "rates": {
                        "input_per_million": 0.15,
                        "output_per_million": 0.60,
                        "cache_read_per_million": 0.075
                    },
                    "prompt_cache": {
                        "read_accounting": "included_in_prompt_tokens"
                    }
                }
            ]
        })
        .to_string(),
    )
    .unwrap();
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();
}

fn install_openai_disambiguation_pricing(model_id: &str) {
    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [
                {
                    "provider": "other",
                    "model_id": model_id,
                    "pricing_as_of": "2026-06-05",
                    "pricing_source": "test",
                    "rates": {
                        "input_per_million": 1000.0,
                        "output_per_million": 1000.0
                    },
                    "prompt_cache": {
                        "read_accounting": "included_in_prompt_tokens"
                    }
                },
                {
                    "provider": "openai",
                    "model_id": model_id,
                    "pricing_as_of": "2026-06-05",
                    "pricing_source": "test",
                    "rates": {
                        "input_per_million": 0.15,
                        "output_per_million": 0.60,
                        "cache_read_per_million": 0.075
                    },
                    "prompt_cache": {
                        "read_accounting": "included_in_prompt_tokens"
                    }
                }
            ]
        })
        .to_string(),
    )
    .unwrap();
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();
}

fn openai_chat_provider_response(model_id: &str) -> Json {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 1_000,
            "completion_tokens": 500,
            "total_tokens": 1_500,
            "prompt_tokens_details": {"cached_tokens": 200}
        }
    })
}

fn sample_openinference_annotated_request() -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
        messages: vec![
            Message::System {
                content: MessageContent::Text("Use concise answers.".to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text("Search docs.".to_string()),
                name: None,
            },
        ],
        model: Some("gpt-4o".to_string()),
        params: Some(GenerationParams {
            temperature: Some(0.2),
            max_tokens: Some(128),
            top_p: None,
            stop: None,
        }),
        tools: Some(vec![ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "search_docs".to_string(),
                description: Some("Search the docs corpus.".to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"query": {"type": "string"}}
                })),
            },
        }]),
        ..empty_annotated_request()
    }
}

fn sample_openinference_annotated_response() -> AnnotatedLlmResponse {
    AnnotatedLlmResponse {
        message: Some(MessageContent::Text("I will search docs.".to_string())),
        tool_calls: Some(vec![ResponseToolCall {
            id: "call-search-docs".to_string(),
            name: "search_docs".to_string(),
            arguments: json!({"query": "docs"}),
        }]),
        finish_reason: Some(FinishReason::ToolUse),
        ..empty_annotated_response()
    }
}

fn make_start_event(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    input: Option<Json>,
) -> Event {
    make_scope_event(
        ScopeCategory::Start,
        uuid,
        parent_uuid,
        name,
        scope_type,
        input,
    )
}

fn make_end_event(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    output: Option<Json>,
) -> Event {
    make_scope_event(
        ScopeCategory::End,
        uuid,
        parent_uuid,
        name,
        scope_type,
        output,
    )
}

fn make_scope_event(
    scope_category: ScopeCategory,
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    data: Option<Json>,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(uuid)
            .name(name)
            .data_opt(data)
            .build(),
        scope_category,
        Vec::new(),
        EventCategory::from(scope_type),
        None,
    ))
}

fn make_scope_event_with_profile(
    scope_category: ScopeCategory,
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    data: Option<Json>,
    category_profile: Option<CategoryProfile>,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(uuid)
            .name(name)
            .data_opt(data)
            .build(),
        scope_category,
        Vec::new(),
        EventCategory::from(scope_type),
        category_profile,
    ))
}

fn make_scope_event_with_attributes(
    scope_category: ScopeCategory,
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    data: Option<Json>,
    attributes: Vec<String>,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(uuid)
            .name(name)
            .data_opt(data)
            .build(),
        scope_category,
        attributes,
        EventCategory::from(scope_type),
        None,
    ))
}

fn make_mark_event(parent_uuid: Option<Uuid>, name: &str, data: Option<Json>) -> Event {
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .name(name)
            .data_opt(data)
            .build(),
        None,
        None,
    ))
}

struct CapturedHttpRequest {
    path: String,
    content_type: String,
    body: Vec<u8>,
}

fn spawn_http_collector(listener: TcpListener, request_tx: mpsc::Sender<CapturedHttpRequest>) {
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let request = read_http_request(&mut stream);
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
        request_tx.send(request).unwrap();
    });
}

fn read_http_request(stream: &mut impl Read) -> CapturedHttpRequest {
    let mut bytes = Vec::new();
    let mut buf = [0_u8; 4096];
    let (header_end, content_length) = read_http_headers(stream, &mut bytes, &mut buf);
    read_http_body(stream, &mut bytes, &mut buf, header_end + content_length);

    let headers_text = String::from_utf8_lossy(&bytes[..header_end]);
    let request_line = headers_text.lines().next().unwrap();
    CapturedHttpRequest {
        path: request_line.split_whitespace().nth(1).unwrap().to_string(),
        content_type: header_value(&headers_text, "content-type").unwrap_or_default(),
        body: bytes[header_end..header_end + content_length].to_vec(),
    }
}

fn read_http_headers(
    stream: &mut impl Read,
    bytes: &mut Vec<u8>,
    buf: &mut [u8; 4096],
) -> (usize, usize) {
    loop {
        let read = stream.read(buf).unwrap();
        if read == 0 {
            panic!("collector closed before receiving an OTLP request");
        }
        bytes.extend_from_slice(&buf[..read]);

        if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            let header_end = header_end + 4;
            let headers_text = String::from_utf8_lossy(&bytes[..header_end]);
            let content_length = header_value(&headers_text, "content-length")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            return (header_end, content_length);
        }
    }
}

fn read_http_body(
    stream: &mut impl Read,
    bytes: &mut Vec<u8>,
    buf: &mut [u8; 4096],
    expected_len: usize,
) {
    while bytes.len() < expected_len {
        let read = stream.read(buf).unwrap();
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&buf[..read]);
    }
}

fn header_value(headers_text: &str, header_name: &str) -> Option<String> {
    headers_text.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case(header_name)
            .then(|| value.trim().to_string())
    })
}

#[test]
fn config_defaults_and_builder_overrides_are_applied() {
    let config = OpenInferenceConfig::new()
        .with_service_name("demo-agent")
        .with_endpoint("http://localhost:4318/v1/traces")
        .with_header("authorization", "Bearer token")
        .with_resource_attribute("deployment.environment", "test")
        .with_service_namespace("agents")
        .with_service_version("1.2.3")
        .with_instrumentation_scope("demo-scope")
        .with_mark_projection(MarkProjection::Tool)
        .with_mark_exclude_names(["notification", "hook_mark"])
        .with_timeout(Duration::from_millis(1250));

    assert_eq!(config.transport, OtlpTransport::HttpBinary);
    assert_eq!(
        config.endpoint.as_deref(),
        Some("http://localhost:4318/v1/traces")
    );
    assert_eq!(
        config.headers.get("authorization"),
        Some(&"Bearer token".into())
    );
    assert_eq!(
        config.resource_attributes.get("deployment.environment"),
        Some(&"test".into())
    );
    assert_eq!(config.service_name, "demo-agent");
    assert_eq!(config.service_namespace.as_deref(), Some("agents"));
    assert_eq!(config.service_version.as_deref(), Some("1.2.3"));
    assert_eq!(config.instrumentation_scope, "demo-scope");
    assert_eq!(config.mark_projection, MarkProjection::Tool);
    assert_eq!(config.mark_exclude_names, vec!["notification", "hook_mark"]);
    assert_eq!(config.timeout, Duration::from_millis(1250));

    let defaults = OpenInferenceConfig::default();
    assert_eq!(defaults.transport, OtlpTransport::HttpBinary);
    assert_eq!(defaults.service_name, "nemo-relay");
    assert_eq!(defaults.instrumentation_scope, "nemo-relay-openinference");
    assert_eq!(defaults.mark_projection, MarkProjection::Inherit);
    assert_eq!(defaults.mark_exclude_names, vec!["llm.chunk"]);
    assert_eq!(defaults.timeout, Duration::from_secs(3));
    assert!(defaults.headers.is_empty());
    assert!(defaults.resource_attributes.is_empty());
}

#[test]
fn grpc_config_requires_a_tokio_runtime() {
    let err = match OpenInferenceSubscriber::new(
        OpenInferenceConfig::new()
            .with_service_name("demo-agent")
            .with_transport(OtlpTransport::Grpc),
    ) {
        Ok(_) => panic!("gRPC construction should require a Tokio runtime"),
        Err(err) => err,
    };
    assert!(matches!(err, OpenInferenceError::MissingTokioRuntime));
}

#[test]
fn invalid_grpc_headers_are_rejected() {
    let err = build_grpc_metadata(&HashMap::from([(
        "bad key".to_string(),
        "value".to_string(),
    )]))
    .expect_err("invalid metadata key should fail");
    assert!(matches!(err, OpenInferenceError::InvalidGrpcHeader { .. }));
}

#[test]
fn subscriber_registration_and_provider_lifecycle_methods_work() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_global();

    let (provider, _exporter) = make_provider();
    let subscriber = OpenInferenceSubscriber::from_tracer_provider(provider, "test-scope");
    let name = format!("otel_test_{}", Uuid::now_v7().simple());

    subscriber.register(&name).unwrap();
    assert!(subscriber.deregister(&name).unwrap());
    assert!(!subscriber.deregister(&name).unwrap());
    subscriber.force_flush().unwrap();
    subscriber.shutdown().unwrap();
}

#[test]
fn registered_subscriber_emits_spans_for_scope_push_pop_and_marks() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_global();

    let (provider, exporter) = make_provider();
    let subscriber = OpenInferenceSubscriber::from_tracer_provider(provider, "e2e-scope");
    let name = format!("otel_e2e_{}", Uuid::now_v7().simple());

    subscriber.register(&name).unwrap();
    let handle = push_scope(
        crate::api::scope::PushScopeParams::builder()
            .name("otel_scope")
            .scope_type(ScopeType::Agent)
            .data(json!({"scope": true}))
            .metadata(json!({"phase": "start"}))
            .input(json!({"task": "scope-start"}))
            .build(),
    )
    .unwrap();
    event(
        crate::api::scope::EmitMarkEventParams::builder()
            .name("otel_mark")
            .parent(&handle)
            .data(json!({"step": 1}))
            .metadata(json!({"source": "rust-test"}))
            .build(),
    )
    .unwrap();
    pop_scope(
        crate::api::scope::PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .output(json!({"status": "done"}))
            .build(),
    )
    .unwrap();

    assert!(subscriber.deregister(&name).unwrap());
    subscriber.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);

    let span = &spans[0];
    assert_eq!(span.name.as_ref(), "otel_scope");
    assert_eq!(span.events.events.len(), 1);
    assert_eq!(span.events.events[0].name.as_ref(), "otel_mark");

    let attributes = attr_map(&span.attributes);
    assert_eq!(
        attributes.get("openinference.span.kind"),
        Some(&"AGENT".to_string())
    );
    assert!(!attributes.contains_key("nemo_relay.start.data_json"));
    assert!(!attributes.contains_key("nemo_relay.start.metadata_json"));
    assert_eq!(
        attributes.get("nemo_relay.start.input_json"),
        Some(&"{\"task\":\"scope-start\"}".to_string())
    );
    assert_eq!(
        attributes.get("input.value"),
        Some(&"{\"task\":\"scope-start\"}".to_string())
    );
    assert_eq!(
        attributes.get("output.value"),
        Some(&"{\"status\":\"done\"}".to_string())
    );
    assert_eq!(
        attributes.get("metadata"),
        Some(&"{\"phase\":\"start\"}".to_string())
    );

    let event_attributes = attr_map(&span.events.events[0].attributes);
    assert_eq!(
        event_attributes.get("nemo_relay.mark.data_json"),
        Some(&"{\"step\":1}".to_string())
    );
    assert_eq!(
        event_attributes.get("nemo_relay.mark.metadata_json"),
        Some(&"{\"source\":\"rust-test\"}".to_string())
    );
}

#[test]
fn http_config_exports_scope_push_pop_and_marks_without_tokio_runtime() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_global();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1/traces", listener.local_addr().unwrap());
    let (request_tx, request_rx) = mpsc::channel();
    spawn_http_collector(listener, request_tx);

    let config = OpenInferenceConfig::new()
        .with_service_name("demo-agent")
        .with_endpoint(endpoint);
    let subscriber = OpenInferenceSubscriber::new(config).unwrap();
    let name = format!("otel_http_{}", Uuid::now_v7().simple());

    subscriber.register(&name).unwrap();
    let handle = push_scope(
        crate::api::scope::PushScopeParams::builder()
            .name("otel_scope")
            .scope_type(ScopeType::Agent)
            .data(json!({"scope": true}))
            .input(json!({"task": "http-start"}))
            .build(),
    )
    .unwrap();
    event(
        crate::api::scope::EmitMarkEventParams::builder()
            .name("otel_mark")
            .parent(&handle)
            .data(json!({"step": 1}))
            .metadata(json!({"source": "rust-http"}))
            .build(),
    )
    .unwrap();
    pop_scope(
        crate::api::scope::PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .output(json!({"status": "http-done"}))
            .build(),
    )
    .unwrap();

    assert!(subscriber.deregister(&name).unwrap());
    subscriber.force_flush().unwrap();

    let request = request_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("expected an OTLP request");
    assert_eq!(request.path, "/v1/traces");
    assert_eq!(request.content_type, "application/x-protobuf");
    assert!(!request.body.is_empty());
}

#[test]
fn records_span_start_mark_and_end() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let root_uuid = Uuid::now_v7();

    let start = make_start_event(
        root_uuid,
        None,
        "search",
        ScopeType::Tool,
        Some(json!({"query": "hello"})),
    );
    processor.process(&start);

    let mark = make_mark_event(Some(root_uuid), "checkpoint", Some(json!({"step": 1})));
    processor.process(&mark);

    let end = make_end_event(
        root_uuid,
        None,
        "search",
        ScopeType::Tool,
        Some(json!({"result": "ok"})),
    );
    processor.process(&end);

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let span = &spans[0];
    assert_eq!(span.name.as_ref(), "search");
    assert_eq!(span.events.events.len(), 1);
    assert_eq!(span.events.events[0].name.as_ref(), "checkpoint");

    let attributes = attr_map(&span.attributes);
    assert_eq!(
        attributes.get("nemo_relay.uuid"),
        Some(&root_uuid.to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.start.input_json"),
        Some(&"{\"query\":\"hello\"}".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.end.output_json"),
        Some(&"{\"result\":\"ok\"}".to_string())
    );
}

#[test]
fn openclaw_model_timing_marks_attach_to_parent_spans() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let root_uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        root_uuid,
        None,
        "openclaw.session",
        ScopeType::Agent,
        Some(json!({"sessionId": "session-1"})),
    ));
    processor.process(&make_mark_event(
        Some(root_uuid),
        "openclaw.model_call_timing_ambiguous",
        Some(json!({
            "runId": "run-1",
            "sessionId": "session-1",
            "provider": "openai",
            "model": "gpt-4",
            "candidateCount": 2
        })),
    ));
    processor.process(&make_mark_event(
        Some(root_uuid),
        "openclaw.model_call_timing_unpaired",
        Some(json!({
            "runId": "run-1",
            "callId": "call-1",
            "provider": "openai",
            "model": "gpt-4",
            "durationMs": 42,
            "outcome": "completed"
        })),
    ));
    processor.process(&make_end_event(
        root_uuid,
        None,
        "openclaw.session",
        ScopeType::Agent,
        Some(json!({"status": "closed"})),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let span = &spans[0];
    assert_eq!(span.name.as_ref(), "openclaw.session");
    assert_eq!(span.events.events.len(), 2);
    assert_eq!(
        span.events.events[0].name.as_ref(),
        "openclaw.model_call_timing_ambiguous"
    );
    assert_eq!(
        span.events.events[1].name.as_ref(),
        "openclaw.model_call_timing_unpaired"
    );

    let ambiguous_attributes = attr_map(&span.events.events[0].attributes);
    assert_eq!(
        ambiguous_attributes.get("nemo_relay.mark.parent_uuid"),
        Some(&root_uuid.to_string())
    );
    let ambiguous_data: serde_json::Value = serde_json::from_str(
        ambiguous_attributes
            .get("nemo_relay.mark.data_json")
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        ambiguous_data,
        json!({
            "runId": "run-1",
            "sessionId": "session-1",
            "provider": "openai",
            "model": "gpt-4",
            "candidateCount": 2
        })
    );
    assert!(!ambiguous_attributes.contains_key("nemo_relay.mark.metadata_json"));

    let unpaired_attributes = attr_map(&span.events.events[1].attributes);
    assert_eq!(
        unpaired_attributes.get("nemo_relay.mark.parent_uuid"),
        Some(&root_uuid.to_string())
    );
    let unpaired_data: serde_json::Value = serde_json::from_str(
        unpaired_attributes
            .get("nemo_relay.mark.data_json")
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        unpaired_data,
        json!({
            "runId": "run-1",
            "callId": "call-1",
            "provider": "openai",
            "model": "gpt-4",
            "durationMs": 42,
            "outcome": "completed"
        })
    );
    assert!(!unpaired_attributes.contains_key("nemo_relay.mark.metadata_json"));
}

#[test]
fn openclaw_hook_only_fallbacks_preserve_stripped_content_and_explicit_usage() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let stripped_uuid = Uuid::now_v7();
    let partial_uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        stripped_uuid,
        None,
        "openclaw-model-call",
        ScopeType::Llm,
        Some(json!({
            "headers": {"authorization": "Bearer secret-token"},
            "content": {
                "provider": "openai",
                "model": "gpt-4",
                "messages": [],
                "imagesCount": 1,
                "source": "openclaw.llm_output"
            }
        })),
    ));
    processor.process(&make_end_event(
        stripped_uuid,
        None,
        "openclaw-model-call",
        ScopeType::Llm,
        Some(json!({
            "role": "assistant",
            "assistant_texts_count": 1,
            "usage": {
                "cost_usd": 0.001
            },
            "openclaw": {
                "assistant_tool_call_names": []
            }
        })),
    ));

    processor.process(&make_start_event(
        partial_uuid,
        None,
        "openclaw-model-call",
        ScopeType::Llm,
        Some(json!({
            "headers": {},
            "content": {
                "provider": "openai",
                "model": "gpt-4",
                "prompt": "visible prompt",
                "messages": [{"role": "user", "content": "visible prompt"}],
                "imagesCount": 0,
                "source": "openclaw.llm_output"
            }
        })),
    ));
    processor.process(&make_end_event(
        partial_uuid,
        None,
        "openclaw-model-call",
        ScopeType::Llm,
        Some(json!({
            "role": "assistant",
            "content": "visible answer",
            "usage": {
                "prompt_tokens": 42
            },
            "openclaw": {
                "assistant_tool_call_names": []
            }
        })),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 2);

    let stripped_span = spans
        .iter()
        .find(|span| {
            let attributes = attr_map(&span.attributes);
            attributes.get("llm.cost.total") == Some(&"0.001".to_string())
        })
        .expect("missing stripped OpenClaw fallback span");
    let stripped_attributes = attr_map(&stripped_span.attributes);
    assert_eq!(
        stripped_attributes.get("input.mime_type"),
        Some(&"application/json".to_string())
    );
    let stripped_input = stripped_attributes
        .get("input.value")
        .expect("missing stripped input.value");
    let parsed_input: serde_json::Value = serde_json::from_str(stripped_input).unwrap();
    assert_eq!(parsed_input["content"]["messages"], json!([]));
    assert!(parsed_input["headers"].is_null() || parsed_input.get("headers").is_none());
    assert_eq!(
        stripped_attributes.get("output.mime_type"),
        Some(&"application/json".to_string())
    );
    let stripped_output = stripped_attributes
        .get("output.value")
        .expect("missing stripped output.value");
    let parsed_output: serde_json::Value = serde_json::from_str(stripped_output).unwrap();
    assert!(parsed_output.get("content").is_none());
    assert_eq!(parsed_output["assistant_texts_count"], json!(1));
    assert_eq!(
        stripped_attributes.get("llm.cost.total"),
        Some(&"0.001".to_string())
    );
    assert!(!stripped_attributes.contains_key("llm.token_count.prompt"));
    assert!(!stripped_attributes.contains_key("llm.output_messages.0.message.content"));
    assert!(!stripped_attributes.contains_key("llm.input_messages.0.message.role"));
    assert_no_attr_contains(&stripped_attributes, "secret-token");

    let partial_span = spans
        .iter()
        .find(|span| {
            let attributes = attr_map(&span.attributes);
            attributes.get("llm.token_count.prompt") == Some(&"42".to_string())
        })
        .expect("missing partial-usage OpenClaw fallback span");
    let partial_attributes = attr_map(&partial_span.attributes);
    assert_eq!(
        partial_attributes.get("input.value"),
        Some(&"user: visible prompt".to_string())
    );
    assert_eq!(
        partial_attributes.get("output.value"),
        Some(&"visible answer".to_string())
    );
    assert_eq!(
        partial_attributes.get("llm.token_count.prompt"),
        Some(&"42".to_string())
    );
    assert!(!partial_attributes.contains_key("llm.token_count.completion"));
    assert!(!partial_attributes.contains_key("llm.token_count.total"));
    assert!(!partial_attributes.contains_key("llm.cost.total"));
}

#[test]
fn llm_input_value_omits_request_headers() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let root_uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        root_uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({
            "headers": {"authorization": "Bearer secret-token"},
            "content": {"messages": [{"role": "user", "content": "hi"}], "model": "demo-model"}
        })),
    ));
    processor.process(&make_end_event(
        root_uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"message": "hello"})),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_attr(&attributes, "input.value", "user: hi");
    assert_attr(&attributes, "input.mime_type", "text/plain");
    assert!(!attributes.contains_key("nemo_relay.start.input_json"));
    assert!(!attributes["input.value"].contains("authorization"));
    assert!(!attributes["input.value"].contains("secret-token"));
    // The provider-shaped request is decoded through the codec layer, so
    // structured messages are emitted — without leaking transport headers.
    assert_attr(&attributes, "llm.input_messages.0.message.role", "user");
    assert_attr(&attributes, "llm.input_messages.0.message.content", "hi");
    assert_no_attr_contains(&attributes, "headers");
    assert_no_attr_contains(&attributes, "secret-token");
}

#[test]
fn un_annotated_provider_response_decoded_through_codec() {
    // No annotation and no OpenClaw envelope: the raw provider response is
    // detected and decoded through the codec layer (tier 3), so OpenInference
    // emits structured output messages instead of nothing.
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({
            "headers": {},
            "content": {"messages": [{"role": "user", "content": "hi"}], "model": "demo-model"}
        })),
    ));
    processor.process(&make_end_event(
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({
            "choices": [{
                "message": {"role": "assistant", "content": "hello there"},
                "finish_reason": "stop"
            }]
        })),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.role",
        "assistant",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.content",
        "hello there",
    );
}

#[test]
fn un_annotated_anthropic_response_emits_codec_computed_total_tokens() {
    // Anthropic raw usage carries no total; the codec computes input + output.
    // The un-annotated path must surface that codec total rather than dropping
    // it the way the manual scraper does.
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        uuid,
        None,
        "anthropic",
        ScopeType::Llm,
        Some(json!({
            "headers": {},
            "content": {"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "hi"}]}
        })),
    ));
    processor.process(&make_end_event(
        uuid,
        None,
        "anthropic",
        ScopeType::Llm,
        Some(json!({
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        })),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_attr(&attributes, "llm.token_count.prompt", "10");
    assert_attr(&attributes, "llm.token_count.completion", "20");
    assert_attr(&attributes, "llm.token_count.total", "30");
}

#[test]
fn provider_shaped_empty_usage_falls_back_to_manual_token_usage() {
    // A provider-shaped response with an empty `usage` object yields an empty
    // codec usage; that must not mask the manual scraper's `token_usage`.
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({
            "headers": {},
            "content": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        })),
    ));
    processor.process(&make_end_event(
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({
            "model": "gpt-4o",
            "choices": [{
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {},
            "token_usage": {"prompt_tokens": 5, "completion_tokens": 7}
        })),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_attr(&attributes, "llm.token_count.prompt", "5");
    assert_attr(&attributes, "llm.token_count.completion", "7");
}

#[test]
fn provider_shaped_partial_usage_merges_with_manual_token_usage() {
    // Codec usage covers only prompt; `token_usage` covers completion/total. The
    // per-field merge must keep all three rather than letting partial codec usage
    // mask the scraper's fields.
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({
            "headers": {},
            "content": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        })),
    ));
    processor.process(&make_end_event(
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({
            "model": "gpt-4o",
            "choices": [{
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5},
            "token_usage": {"completion_tokens": 7, "total_tokens": 12}
        })),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_attr(&attributes, "llm.token_count.prompt", "5");
    assert_attr(&attributes, "llm.token_count.completion", "7");
    assert_attr(&attributes, "llm.token_count.total", "12");
}

#[test]
fn openclaw_replay_payloads_emit_flattened_openinference_llm_attributes() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        uuid,
        None,
        "openclaw-model-call",
        ScopeType::Llm,
        Some(json!({
            "headers": {"authorization": "Bearer secret-token"},
            "content": {
                "provider": "nvidia-inference",
                "model": "claude-sonnet-4",
                "systemPrompt": "Use reliable sources.",
                "prompt": "Find the answer.",
                "messages": [
                    {"role": "user", "content": "Find the answer."}
                ],
                "placeholderRequest": false,
                "source": "openclaw.llm_output"
            }
        })),
    ));
    processor.process(&make_end_event(
        uuid,
        None,
        "openclaw-model-call",
        ScopeType::Llm,
        Some(json!({
            "role": "assistant",
            "content": "I will search.",
            "tool_calls": [{
                "id": "toolu_search",
                "name": "tavily_search",
                "input": {"query": "answer"}
            }],
            "openclaw": {
                "duration_ms": 42,
                "assistant_tool_call_names": ["tavily_search"]
            }
        })),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_attr(&attributes, "llm.provider", "nvidia-inference");
    assert_attr(&attributes, "llm.system", "Use reliable sources.");
    assert_attr(&attributes, "llm.input_messages.0.message.role", "user");
    assert_attr(
        &attributes,
        "llm.input_messages.0.message.content",
        "Find the answer.",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.role",
        "assistant",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.content",
        "I will search.",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.tool_calls.0.tool_call.id",
        "toolu_search",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name",
        "tavily_search",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments",
        "{\"query\":\"answer\"}",
    );
    assert!(!attributes.contains_key("llm.invocation_parameters"));
    assert!(!attributes.contains_key("llm.finish_reason"));
    assert_no_attr_contains(&attributes, "headers");
    assert_no_attr_contains(&attributes, "secret-token");
}

#[test]
fn openclaw_replay_tool_call_alias_fields_emit_openinference_attributes() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        uuid,
        None,
        "openclaw-model-call",
        ScopeType::Llm,
        Some(json!({
            "content": {
                "messages": [{"role": "user", "content": "Find docs."}],
                "source": "openclaw.llm_output"
            }
        })),
    ));
    processor.process(&make_end_event(
        uuid,
        None,
        "openclaw-model-call",
        ScopeType::Llm,
        Some(json!({
            "role": "assistant",
            "tool_calls": [
                {
                    "call_id": "call-search-docs",
                    "name": "top_level_search",
                    "toolName": "camel_search",
                    "function": {
                        "name": "nested_search",
                        "arguments": {"query": "docs"}
                    }
                },
                {
                    "call_id": "call-read-docs",
                    "toolName": "read_docs",
                    "args": {"path": "README.md"}
                }
            ],
            "openclaw": {
                "assistant_tool_call_names": ["top_level_search", "read_docs"]
            }
        })),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.tool_calls.0.tool_call.id",
        "call-search-docs",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name",
        "top_level_search",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments",
        "{\"query\":\"docs\"}",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.tool_calls.1.tool_call.id",
        "call-read-docs",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.tool_calls.1.tool_call.function.name",
        "read_docs",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.tool_calls.1.tool_call.function.arguments",
        "{\"path\":\"README.md\"}",
    );
}

#[test]
fn openclaw_subagent_scopes_preserve_nested_and_fallback_parent_linkage() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let parent_uuid = Uuid::now_v7();
    let nested_child_uuid = Uuid::now_v7();
    let fallback_child_uuid = Uuid::now_v7();

    let nested_parent_start = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(parent_uuid)
            .name("requester-agent")
            .metadata(json!({
                "source": "openclaw.session_start",
                "hook_event_name": "session_start",
                "session_id": "parent-session"
            }))
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::agent(),
        None,
    ));
    let nested_child_start = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(nested_child_uuid)
            .parent_uuid(parent_uuid)
            .name("nested-worker")
            .metadata(json!({
                "source": "openclaw.session_start",
                "hook_event_name": "session_start",
                "session_id": "nested-child-session",
                "nemo_relay_scope_role": "subagent"
            }))
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::agent(),
        None,
    ));
    let nested_child_end = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(nested_child_uuid)
            .parent_uuid(parent_uuid)
            .name("nested-worker")
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::agent(),
        None,
    ));
    let nested_parent_end = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(parent_uuid)
            .name("requester-agent")
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::agent(),
        None,
    ));
    let fallback_child_start = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(fallback_child_uuid)
            .name("fallback-worker")
            .metadata(json!({
                "source": "openclaw.session_start",
                "hook_event_name": "session_start",
                "session_id": "fallback-child-session",
                "nemo_relay_scope_role": "subagent"
            }))
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::agent(),
        None,
    ));
    let fallback_child_end = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(fallback_child_uuid)
            .name("fallback-worker")
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::agent(),
        None,
    ));

    for event in [
        nested_parent_start,
        nested_child_start,
        nested_child_end,
        nested_parent_end,
        fallback_child_start,
        fallback_child_end,
    ] {
        processor.process(&event);
    }
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let nested_child_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "nested-worker")
        .unwrap();
    let fallback_child_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "fallback-worker")
        .unwrap();
    let nested_child_attributes = attr_map(&nested_child_span.attributes);
    let fallback_child_attributes = attr_map(&fallback_child_span.attributes);

    assert_eq!(
        nested_child_attributes.get("nemo_relay.parent_uuid"),
        Some(&parent_uuid.to_string())
    );
    assert_eq!(
        fallback_child_attributes.get("nemo_relay.parent_uuid"),
        Some(&String::new())
    );
}

#[test]
fn openclaw_placeholder_replay_falls_back_to_sanitized_json_input_value() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        uuid,
        None,
        "openclaw-model-call",
        ScopeType::Llm,
        Some(json!({
            "headers": {"authorization": "Bearer secret-token"},
            "content": {
                "provider": "nvidia-inference",
                "model": "claude-sonnet-4",
                "prompt": "",
                "messages": [],
                "imagesCount": 0,
                "placeholderRequest": true,
                "source": "openclaw.llm_output"
            }
        })),
    ));
    processor.process(&make_end_event(
        uuid,
        None,
        "openclaw-model-call",
        ScopeType::Llm,
        Some(json!({
            "role": "assistant",
            "content": "I will search.",
            "assistant_texts_count": 1,
            "openclaw": {
                "assistant_tool_call_names": []
            }
        })),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_attr(&attributes, "llm.provider", "nvidia-inference");
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.role",
        "assistant",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.content",
        "I will search.",
    );
    assert!(!attributes.contains_key("llm.input_messages.0.message.role"));
    assert!(!attributes.contains_key("llm.input_messages.0.message.content"));
    assert_attr(&attributes, "input.mime_type", "application/json");

    let input_value = attributes.get("input.value").expect("missing input.value");
    let parsed_input: serde_json::Value = serde_json::from_str(input_value).unwrap();
    assert!(parsed_input.get("headers").is_none());
    assert_eq!(parsed_input["content"]["placeholderRequest"], json!(true));
    assert_eq!(parsed_input["content"]["messages"], json!([]));
    assert_eq!(parsed_input["content"]["prompt"], json!(""));
    assert_eq!(
        parsed_input["content"]["source"],
        json!("openclaw.llm_output")
    );
    assert_no_attr_contains(&attributes, "authorization");
    assert_no_attr_contains(&attributes, "secret-token");
}

#[test]
fn generic_unannotated_llm_output_does_not_emit_flattened_output_message_attrs() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        uuid,
        None,
        "generic-model-call",
        ScopeType::Llm,
        Some(json!({
            "headers": {},
            "content": {"messages": [{"role": "user", "content": "hi"}], "model": "demo-model"}
        })),
    ));
    processor.process(&make_end_event(
        uuid,
        None,
        "generic-model-call",
        ScopeType::Llm,
        Some(json!({
            "role": "assistant",
            "content": "hello",
            "tool_calls": [{
                "id": "tool-call-1",
                "name": "demo_tool",
                "input": {"query": "hi"}
            }]
        })),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert!(!attributes.contains_key("llm.output_messages.0.message.role"));
    assert!(!attributes.contains_key("llm.output_messages.0.message.content"));
    assert!(
        !attributes
            .contains_key("llm.output_messages.0.message.tool_calls.0.tool_call.function.name")
    );
}

#[test]
fn llm_input_value_summarizes_tool_call_messages() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let root_uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        root_uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({
            "content": {
                "messages": [
                    {"role": "user", "content": "Inspect the files."},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "stripped": true},
                            {"type": "text", "text": "I will inspect the files."},
                            {"type": "toolCall", "name": "read", "arguments": {"stripped": true}}
                        ]
                    },
                    {"role": "tool", "content": {"stripped": true, "reason": "tool result"}}
                ]
            }
        })),
    ));
    processor.process(&make_end_event(
        root_uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"message": "done"})),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(
        attributes.get("input.value"),
        Some(
            &"user: Inspect the files.\n\nassistant: I will inspect the files.\nRequested tools: read\n\ntool: Tool result omitted"
                .to_string()
        )
    );
    assert!(!attributes["input.value"].contains("thinking"));
    assert!(!attributes["input.value"].contains("arguments"));
    assert!(!attributes["input.value"].contains("tool result"));
}

#[test]
fn output_value_prefers_display_content() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let root_uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        root_uuid,
        None,
        "edit",
        ScopeType::Tool,
        Some(json!({"path": "api.py"})),
    ));
    processor.process(&make_end_event(
        root_uuid,
        None,
        "edit",
        ScopeType::Tool,
        Some(json!({
            "content": "Tool edit completed.",
            "details": {"diff": "-old\n+new"}
        })),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(
        attributes.get("output.value"),
        Some(&"Tool edit completed.".to_string())
    );
    assert_eq!(
        attributes.get("output.mime_type"),
        Some(&"text/plain".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.end.output_json"),
        Some(
            &"{\"content\":\"Tool edit completed.\",\"details\":{\"diff\":\"-old\\n+new\"}}"
                .to_string()
        )
    );
    assert!(!attributes.contains_key("nemo_relay.end.data_json"));
}

#[test]
fn output_value_extracts_chat_completion_display_text() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let root_uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        root_uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"content": {"messages": [{"role": "user", "content": "hi"}]}})),
    ));
    processor.process(&make_end_event(
        root_uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I will inspect the files.",
                    "tool_calls": [
                        {"id": "call-1", "function": {"name": "read", "arguments": "{\"path\":\"api.py\"}"}}
                    ]
                }
            }],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 2},
                "cost_usd": 0.001
            }
        })),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(
        attributes.get("output.value"),
        Some(&"I will inspect the files.\nRequested tools: read".to_string())
    );
    assert_eq!(
        attributes.get("output.mime_type"),
        Some(&"text/plain".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.prompt"),
        Some(&"3".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.prompt_details.cache_read"),
        Some(&"2".to_string())
    );
    assert_eq!(attributes.get("llm.cost.total"), Some(&"0.001".to_string()));
}

#[test]
fn display_text_from_tool_calls_preserves_legacy_name_precedence() {
    assert_eq!(
        display_text_from_tool_calls(&json!([
            {
                "name": "top_level",
                "toolName": "camel",
                "function": {"name": "nested"}
            },
            {
                "toolName": "camel_without_top_level",
                "function": {"name": "nested_without_top_level"}
            }
        ])),
        Some("Requested tools: top_level, camel_without_top_level".to_string())
    );
}

#[test]
fn output_value_extracts_openai_responses_display_text_and_usage() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let root_uuid = Uuid::now_v7();

    processor.process(&make_scope_event_with_profile(
        ScopeCategory::Start,
        root_uuid,
        None,
        "openai.responses",
        ScopeType::Llm,
        Some(json!({
            "input": "Find the weather.",
            "model": "gpt-4o"
        })),
        Some(CategoryProfile::builder().model_name("gpt-4o").build()),
    ));
    processor.process(&make_end_event(
        root_uuid,
        None,
        "openai.responses",
        ScopeType::Llm,
        Some(json!({
            "id": "resp_1",
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "I will check the weather."}
                    ]
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
        })),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(
        attributes.get("llm.model_name"),
        Some(&"gpt-4o".to_string())
    );
    assert_eq!(
        attributes.get("output.value"),
        Some(&"I will check the weather.\nRequested tools: get_weather".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.prompt"),
        Some(&"75".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.completion"),
        Some(&"20".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.total"),
        Some(&"95".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.prompt_details.cache_read"),
        Some(&"10".to_string())
    );
    assert_eq!(attributes.get("llm.cost.total"), Some(&"0.005".to_string()));
}

#[test]
fn tool_semantic_names_exist_without_input_payload() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let root_uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        root_uuid,
        None,
        "ping",
        ScopeType::Tool,
        None,
    ));
    processor.process(&make_end_event(
        root_uuid,
        None,
        "ping",
        ScopeType::Tool,
        Some(json!({"ok": true})),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(attributes.get("tool.name"), Some(&"ping".to_string()));
    assert_eq!(
        attributes.get("tool_call.function.name"),
        Some(&"ping".to_string())
    );
    assert!(!attributes.contains_key("tool.parameters"));
    assert!(!attributes.contains_key("tool_call.function.arguments"));
}

#[test]
fn preserves_parent_child_relationships() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());

    let root_uuid = Uuid::now_v7();
    let child_uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        root_uuid,
        None,
        "agent",
        ScopeType::Agent,
        None,
    ));
    processor.process(&make_start_event(
        child_uuid,
        Some(root_uuid),
        "model-call",
        ScopeType::Llm,
        None,
    ));
    processor.process(&make_end_event(
        child_uuid,
        Some(root_uuid),
        "model-call",
        ScopeType::Llm,
        None,
    ));
    processor.process(&make_end_event(
        root_uuid,
        None,
        "agent",
        ScopeType::Agent,
        None,
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 2);
    let parent = spans
        .iter()
        .find(|span| span.name.as_ref() == "agent")
        .unwrap();
    let child = spans
        .iter()
        .find(|span| span.name.as_ref() == "model-call")
        .unwrap();

    assert_eq!(
        child.span_context.trace_id(),
        parent.span_context.trace_id()
    );
    assert_eq!(child.parent_span_id, parent.span_context.span_id());
    assert!(!child.parent_span_is_remote);
}

#[test]
fn atif_lineage_correlates_with_openinference_span_attributes() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());

    let agent_uuid = Uuid::now_v7();
    let llm_uuid = Uuid::now_v7();
    let atif_exporter = AtifExporter::new(
        agent_uuid.to_string(),
        AtifAgentInfo {
            name: "test-agent".to_string(),
            version: "1.0.0".to_string(),
            model_name: None,
            tool_definitions: None,
            extra: None,
        },
    );
    let atif_subscriber = atif_exporter.subscriber();

    let events = vec![
        make_start_event(agent_uuid, None, "agent", ScopeType::Agent, None),
        make_start_event(
            llm_uuid,
            Some(agent_uuid),
            "model-call",
            ScopeType::Llm,
            Some(json!({"messages": [{"role": "user", "content": "hello"}]})),
        ),
        make_end_event(
            llm_uuid,
            Some(agent_uuid),
            "model-call",
            ScopeType::Llm,
            Some(json!({"content": "hi", "role": "assistant"})),
        ),
        make_end_event(agent_uuid, None, "agent", ScopeType::Agent, None),
    ];

    for event in &events {
        processor.process(event);
        atif_subscriber(event);
    }
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let agent_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "agent")
        .unwrap();
    let llm_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "model-call")
        .unwrap();
    let agent_attributes = attr_map(&agent_span.attributes);
    let llm_attributes = attr_map(&llm_span.attributes);

    assert_eq!(
        agent_attributes.get("nemo_relay.uuid"),
        Some(&agent_uuid.to_string())
    );
    assert_eq!(
        llm_attributes.get("nemo_relay.uuid"),
        Some(&llm_uuid.to_string())
    );
    assert_eq!(
        llm_attributes.get("nemo_relay.parent_uuid"),
        Some(&agent_uuid.to_string())
    );

    let trajectory = atif_exporter.export().unwrap();
    assert_eq!(trajectory.session_id, agent_uuid.to_string());
    let agent_step = trajectory
        .steps
        .iter()
        .find(|step| step.source == "agent")
        .unwrap();
    let extra: AtifStepExtra = serde_json::from_value(agent_step.extra.clone().unwrap()).unwrap();

    assert_eq!(
        llm_attributes.get("nemo_relay.uuid"),
        Some(&extra.ancestry.function_id)
    );
    assert_eq!(extra.ancestry.parent_id, Some(trajectory.session_id));
}

#[test]
fn orphan_marks_become_zero_duration_spans() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let mark = make_mark_event(None, "detached", Some(json!({"kind": "standalone"})));

    processor.process(&mark);
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let span = &spans[0];
    assert_eq!(span.name.as_ref(), "mark:detached");
    assert_eq!(span.start_time, span.end_time);

    let attributes = attr_map(&span.attributes);
    assert_eq!(
        attributes.get("nemo_relay.mark.orphan"),
        Some(&"true".to_string())
    );
    assert_eq!(
        attributes.get("openinference.span.kind"),
        Some(&"CHAIN".to_string())
    );
}

#[test]
fn tool_projection_emits_generic_mark_as_parented_openinference_tool_span() {
    let (provider, exporter) = make_provider();
    let mut processor = OpenInferenceEventProcessor::new_with_mark_projection(
        provider.clone(),
        "test-scope".to_string(),
        MarkProjection::Tool,
    );
    let parent_uuid = Uuid::now_v7();
    let start = make_start_event(parent_uuid, None, "agent-turn", ScopeType::Agent, None);
    let mark = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid(parent_uuid)
            .name("plugin.output_compacted")
            .data(json!({"count": 3}))
            .build(),
        Some(EventCategory::custom()),
        Some(
            CategoryProfile::builder()
                .subtype("example.compaction")
                .build(),
        ),
    ));
    let end = make_end_event(parent_uuid, None, "agent-turn", ScopeType::Agent, None);

    processor.process(&start);
    processor.process(&mark);
    processor.process(&end);
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 2);
    let parent = spans
        .iter()
        .find(|span| span.name.as_ref() == "agent-turn")
        .unwrap();
    let projected = spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:plugin.output_compacted")
        .unwrap();
    assert!(parent.events.events.is_empty());
    assert_eq!(projected.parent_span_id, parent.span_context.span_id());
    assert_eq!(projected.start_time, projected.end_time);

    let attributes = attr_map(&projected.attributes);
    assert_eq!(
        attributes.get("openinference.span.kind"),
        Some(&"TOOL".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.mark.projection"),
        Some(&"tool".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.mark.data_json"),
        Some(&"{\"count\":3}".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.mark.category"),
        Some(&"custom".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.mark.category_profile_json"),
        Some(&"{\"subtype\":\"example.compaction\"}".to_string())
    );
    assert!(!attributes.contains_key("nemo_relay.mark.orphan"));
}

#[test]
fn tool_projection_exclusion_keeps_custom_mark_as_native_event() {
    let (provider, exporter) = make_provider();
    let mut processor = OpenInferenceEventProcessor::new_with_mark_projection_and_exclusions(
        provider.clone(),
        "test-scope".to_string(),
        MarkProjection::Tool,
        vec!["plugin.excluded".to_string()],
    );
    processor.process(&make_mark_event(
        None,
        "plugin.excluded",
        Some(json!({"count": 3})),
    ));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert!(!attributes.contains_key("nemo_relay.mark.projection"));
    assert_eq!(
        attributes.get("openinference.span.kind"),
        Some(&"CHAIN".to_string())
    );
}

#[test]
fn tool_projection_reuses_completed_parent_context_for_late_marks() {
    let (provider, exporter) = make_provider();
    let mut processor = OpenInferenceEventProcessor::new_with_mark_projection(
        provider.clone(),
        "test-scope".to_string(),
        MarkProjection::Tool,
    );
    let parent_uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        parent_uuid,
        None,
        "completed-tool",
        ScopeType::Tool,
        None,
    ));
    processor.process(&make_end_event(
        parent_uuid,
        None,
        "completed-tool",
        ScopeType::Tool,
        None,
    ));
    processor.process(&make_mark_event(
        Some(parent_uuid),
        "plugin.late_checkpoint",
        Some(json!({"status": "complete"})),
    ));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let parent = spans
        .iter()
        .find(|span| span.name.as_ref() == "completed-tool")
        .unwrap();
    let projected = spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:plugin.late_checkpoint")
        .unwrap();
    assert_eq!(projected.parent_span_id, parent.span_context.span_id());
    assert_eq!(
        projected.span_context.trace_id(),
        parent.span_context.trace_id()
    );
    let attributes = attr_map(&projected.attributes);
    assert_eq!(
        attributes.get("nemo_relay.mark.orphan"),
        Some(&"true".to_string())
    );
    assert_eq!(
        attributes.get("openinference.span.kind"),
        Some(&"TOOL".to_string())
    );
}

#[test]
fn tool_projection_keeps_llm_chunks_as_parent_span_events() {
    let (provider, exporter) = make_provider();
    let mut processor = OpenInferenceEventProcessor::new_with_mark_projection(
        provider.clone(),
        "test-scope".to_string(),
        MarkProjection::Tool,
    );
    let parent_uuid = Uuid::now_v7();
    let start = make_start_event(parent_uuid, None, "llm", ScopeType::Llm, None);
    let chunk = make_mark_event(Some(parent_uuid), "llm.chunk", Some(json!({"delta": "x"})));
    let aliased_chunk = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid(parent_uuid)
            .name("hook_mark")
            .metadata(json!({"hook_event_name": "llm.chunk"}))
            .data(json!({"delta": "y"}))
            .build(),
        None,
        None,
    ));
    let end = make_end_event(parent_uuid, None, "llm", ScopeType::Llm, None);

    processor.process(&start);
    processor.process(&chunk);
    processor.process(&aliased_chunk);
    processor.process(&end);
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].events.events.len(), 2);
    assert_eq!(spans[0].events.events[0].name.as_ref(), "llm.chunk");
    assert_eq!(spans[0].events.events[1].name.as_ref(), "hook_mark");
}

#[test]
fn late_parented_marks_reuse_completed_parent_trace_context() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let tool_uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        tool_uuid,
        None,
        "terminal",
        ScopeType::Tool,
        None,
    ));
    processor.process(&make_end_event(
        tool_uuid,
        None,
        "terminal",
        ScopeType::Tool,
        Some(json!({"status": "done"})),
    ));
    processor.process(&make_mark_event(
        Some(tool_uuid),
        "plugin.output_compacted",
        Some(json!({"estimated_tokens_saved": 42})),
    ));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 2);
    let tool_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "terminal")
        .unwrap();
    let mark_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:plugin.output_compacted")
        .unwrap();

    assert_eq!(
        mark_span.span_context.trace_id(),
        tool_span.span_context.trace_id()
    );
    assert_eq!(mark_span.parent_span_id, tool_span.span_context.span_id());
    assert!(!mark_span.parent_span_is_remote);

    let attributes = attr_map(&mark_span.attributes);
    assert_eq!(
        attributes.get("nemo_relay.mark.orphan"),
        Some(&"true".to_string())
    );
    assert_eq!(
        attributes.get("openinference.span.kind"),
        Some(&"CHAIN".to_string())
    );
}

#[test]
fn completed_span_context_cache_evicts_oldest_parent_contexts() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let span_count = COMPLETED_SPAN_CONTEXT_LIMIT + 2;
    let mut completed_uuids = Vec::with_capacity(span_count);

    for index in 0..span_count {
        let uuid = Uuid::now_v7();
        completed_uuids.push(uuid);
        let name = format!("completed-{index}");
        processor.process(&make_start_event(uuid, None, &name, ScopeType::Tool, None));
        processor.process(&make_end_event(
            uuid,
            None,
            &name,
            ScopeType::Tool,
            Some(json!({"status": "done"})),
        ));
    }

    let oldest_uuid = completed_uuids[0];
    let recent_uuid = completed_uuids[span_count - 1];
    assert!(!processor.completed_span_contexts.contains_key(&oldest_uuid));
    assert!(processor.completed_span_contexts.contains_key(&recent_uuid));

    processor.process(&make_mark_event(
        Some(oldest_uuid),
        "oldest-after-eviction",
        Some(json!({"case": "oldest"})),
    ));
    processor.process(&make_mark_event(
        Some(recent_uuid),
        "recent-after-eviction",
        Some(json!({"case": "recent"})),
    ));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), span_count + 2);

    let oldest_parent = spans
        .iter()
        .find(|span| span.name.as_ref() == "completed-0")
        .unwrap();
    let recent_parent_name = format!("completed-{}", span_count - 1);
    let recent_parent = spans
        .iter()
        .find(|span| span.name.as_ref() == recent_parent_name.as_str())
        .unwrap();
    let oldest_mark = spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:oldest-after-eviction")
        .unwrap();
    let recent_mark = spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:recent-after-eviction")
        .unwrap();

    assert_ne!(
        oldest_mark.parent_span_id,
        oldest_parent.span_context.span_id()
    );
    assert_ne!(
        oldest_mark.span_context.trace_id(),
        oldest_parent.span_context.trace_id()
    );
    assert_eq!(
        recent_mark.span_context.trace_id(),
        recent_parent.span_context.trace_id()
    );
    assert_eq!(
        recent_mark.parent_span_id,
        recent_parent.span_context.span_id()
    );
    assert!(!recent_mark.parent_span_is_remote);
}

#[test]
fn process_start_removes_completed_span_order_entry() {
    let (provider, _exporter) = make_provider();
    let mut processor = OpenInferenceEventProcessor::new(provider, "test-scope".to_string());
    let tool_uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        tool_uuid,
        None,
        "terminal",
        ScopeType::Tool,
        None,
    ));
    processor.process(&make_end_event(
        tool_uuid,
        None,
        "terminal",
        ScopeType::Tool,
        Some(json!({"status": "done"})),
    ));
    assert!(processor.completed_span_contexts.contains_key(&tool_uuid));
    assert_eq!(
        processor
            .completed_span_order
            .iter()
            .filter(|uuid| **uuid == tool_uuid)
            .count(),
        1
    );

    processor.process(&make_start_event(
        tool_uuid,
        None,
        "terminal",
        ScopeType::Tool,
        None,
    ));
    assert!(!processor.completed_span_contexts.contains_key(&tool_uuid));
    assert!(!processor.completed_span_order.contains(&tool_uuid));

    processor.process(&make_end_event(
        tool_uuid,
        None,
        "terminal",
        ScopeType::Tool,
        Some(json!({"status": "done"})),
    ));
    assert!(processor.completed_span_contexts.contains_key(&tool_uuid));
    assert_eq!(
        processor
            .completed_span_order
            .iter()
            .filter(|uuid| **uuid == tool_uuid)
            .count(),
        1
    );
}

#[test]
fn semantic_scope_type_and_input_value_follow_event_variants() {
    let llm_with_content = make_start_event(
        Uuid::now_v7(),
        None,
        "model-call",
        ScopeType::Llm,
        Some(json!({
            "headers": {"authorization": "Bearer token"},
            "content": {"messages": [{"role": "user", "content": "hello"}]},
        })),
    );
    assert_eq!(semantic_scope_type(&llm_with_content), Some(ScopeType::Llm));
    assert_eq!(span_kind(&llm_with_content), SpanKind::Client);
    assert_eq!(
        openinference_input_value(&llm_with_content),
        Some(("user: hello".to_string(), "text/plain"))
    );

    let llm_without_content = make_start_event(
        Uuid::now_v7(),
        None,
        "model-call",
        ScopeType::Llm,
        Some(json!({
            "headers": {"authorization": "Bearer token"},
            "prompt": "hello",
        })),
    );
    assert_eq!(
        openinference_input_value(&llm_without_content),
        Some(("hello".to_string(), "text/plain"))
    );

    let remote_tool = make_scope_event_with_attributes(
        ScopeCategory::Start,
        Uuid::now_v7(),
        None,
        "search",
        ScopeType::Tool,
        Some(json!({"query": "hello"})),
        tool_attributes_to_strings(ToolAttributes::REMOTE),
    );
    assert_eq!(semantic_scope_type(&remote_tool), Some(ScopeType::Tool));
    assert_eq!(span_kind(&remote_tool), SpanKind::Client);
    let (remote_tool_input, remote_tool_mime_type) =
        openinference_input_value(&remote_tool).unwrap();
    assert_eq!(remote_tool_mime_type, "application/json");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&remote_tool_input).unwrap(),
        json!({"query": "hello"})
    );
}

#[test]
fn scope_end_output_payload_is_exported_to_openinference_attributes() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let scope_uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        scope_uuid,
        None,
        "agent",
        ScopeType::Agent,
        Some(json!({"task": "summarize"})),
    ));
    processor.process(&make_end_event(
        scope_uuid,
        None,
        "agent",
        ScopeType::Agent,
        Some(json!({"status": "done", "metrics": {"tokens": 42}})),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(attributes.get("output.value").unwrap()).unwrap(),
        json!({"status": "done", "metrics": {"tokens": 42}})
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(
            attributes.get("nemo_relay.end.output_json").unwrap(),
        )
        .unwrap(),
        json!({"status": "done", "metrics": {"tokens": 42}})
    );
}

#[test]
fn scope_end_metadata_sets_openinference_span_status() {
    let cases = [
        (
            json!({"otel.status_code": "ERROR", "otel.status_description": "failed"}),
            Status::error("failed".to_string()),
        ),
        (json!({"otel.status_code": "OK"}), Status::Ok),
        (json!({}), Status::Unset),
    ];

    for (metadata, expected_status) in cases {
        let (provider, exporter) = make_provider();
        let mut processor =
            OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
        let scope_uuid = Uuid::now_v7();

        processor.process(&make_start_event(
            scope_uuid,
            None,
            "agent",
            ScopeType::Agent,
            Some(json!({"task": "summarize"})),
        ));
        processor.process(&Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .uuid(scope_uuid)
                .name("agent")
                .metadata(metadata)
                .data(json!({"status": "done"}))
                .build(),
            ScopeCategory::End,
            Vec::new(),
            EventCategory::agent(),
            None,
        )));

        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].status, expected_status);
    }
}

#[test]
fn pre_epoch_timestamps_round_trip_through_system_time() {
    let timestamp = DateTime::parse_from_rfc3339("1969-12-31T23:59:58.500000000Z")
        .unwrap()
        .with_timezone(&Utc);

    assert_eq!(
        to_system_time(timestamp),
        UNIX_EPOCH - Duration::new(1, 500_000_000)
    );
}

#[test]
fn helper_functions_cover_additional_openinference_branches() {
    let function_end = make_end_event(Uuid::now_v7(), None, "fn-scope", ScopeType::Function, None);
    assert_eq!(span_name(&function_end), "fn-scope");
    assert_eq!(
        semantic_scope_type(&function_end),
        Some(ScopeType::Function)
    );

    assert_eq!(scope_type_name(Some(ScopeType::Retriever)), "retriever");
    assert_eq!(scope_type_name(Some(ScopeType::Embedder)), "embedder");
    assert_eq!(scope_type_name(Some(ScopeType::Reranker)), "reranker");
    assert_eq!(scope_type_name(Some(ScopeType::Guardrail)), "guardrail");
    assert_eq!(scope_type_name(Some(ScopeType::Evaluator)), "evaluator");
    assert_eq!(scope_type_name(Some(ScopeType::Custom)), "custom");
    assert_eq!(scope_type_name(Some(ScopeType::Unknown)), "unknown");
    assert_eq!(scope_type_name(None), "unknown");

    assert_eq!(
        openinference_span_kind(Some(ScopeType::Embedder)),
        OpenInferenceSpanKind::Embedding
    );
    assert_eq!(
        openinference_span_kind(Some(ScopeType::Reranker)),
        OpenInferenceSpanKind::Reranker
    );
    assert_eq!(
        openinference_span_kind(Some(ScopeType::Guardrail)),
        OpenInferenceSpanKind::Guardrail
    );
    assert_eq!(
        openinference_span_kind(Some(ScopeType::Evaluator)),
        OpenInferenceSpanKind::Evaluator
    );
    assert_eq!(
        openinference_span_kind(Some(ScopeType::Custom)),
        OpenInferenceSpanKind::Chain
    );
    assert_eq!(
        openinference_span_kind(Some(ScopeType::Unknown)),
        OpenInferenceSpanKind::Chain
    );
    assert_eq!(openinference_span_kind(None), OpenInferenceSpanKind::Chain);

    let llm_end = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("chat")
            .metadata(json!({"phase": "done"}))
            .data(json!({"answer": "ok"}))
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::llm(),
        Some(CategoryProfile::builder().model_name("demo-model").build()),
    ));
    let llm_attributes = attr_map(&common_attributes(&llm_end));
    assert!(!llm_attributes.contains_key("nemo_relay.model_name"));
    assert_eq!(
        llm_attributes.get(oi::llm::MODEL_NAME.as_str()),
        Some(&"demo-model".to_string())
    );
    let raw_model_end = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("raw-llm")
            .data(json!({"model": "raw-model", "answer": "ok"}))
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::llm(),
        None,
    ));
    let raw_model_attributes = attr_map(&common_attributes(&raw_model_end));
    assert_eq!(
        raw_model_attributes.get(oi::llm::MODEL_NAME.as_str()),
        Some(&"raw-model".to_string())
    );
    assert_eq!(
        llm_attributes.get(oi::METADATA.as_str()),
        Some(&"{\"phase\":\"done\"}".to_string())
    );

    let tool_start = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("lookup")
            .metadata(json!({"meta": true}))
            .data(json!({"query": "hello"}))
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::tool(),
        Some(CategoryProfile::builder().tool_call_id("call-123").build()),
    ));
    let tool_start_attributes = attr_map(&start_attributes(&tool_start));
    assert_eq!(
        tool_start_attributes.get(oi::tool::NAME.as_str()),
        Some(&"lookup".to_string())
    );
    assert_eq!(
        tool_start_attributes.get(oi::tool_call::function::NAME.as_str()),
        Some(&"lookup".to_string())
    );
    assert_eq!(
        tool_start_attributes.get(oi::tool::PARAMETERS.as_str()),
        Some(&"{\"query\":\"hello\"}".to_string())
    );
    assert_eq!(
        tool_start_attributes.get(oi::tool_call::function::ARGUMENTS.as_str()),
        Some(&"{\"query\":\"hello\"}".to_string())
    );
    assert_eq!(
        tool_start_attributes.get(oi::tool_call::ID.as_str()),
        Some(&"call-123".to_string())
    );

    let tool_end = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("lookup")
            .metadata(json!({"phase": "complete"}))
            .data(json!({"result": true}))
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::tool(),
        Some(CategoryProfile::builder().tool_call_id("call-456").build()),
    ));
    let tool_end_attributes = attr_map(&end_attributes(&tool_end));
    assert_eq!(
        tool_end_attributes.get(oi::output::VALUE.as_str()),
        Some(&"{\"result\":true}".to_string())
    );
    assert_eq!(
        tool_end_attributes.get(oi::output::MIME_TYPE.as_str()),
        Some(&"application/json".to_string())
    );

    let mark = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid(Uuid::now_v7())
            .name("checkpoint")
            .data(json!({"kind": "aux"}))
            .metadata(json!({"source": "unit"}))
            .build(),
        None,
        None,
    ));
    let mark_attributes = attr_map(&mark_attributes(&mark));
    assert_eq!(
        mark_attributes.get("nemo_relay.mark.data_json"),
        Some(&"{\"kind\":\"aux\"}".to_string())
    );
    assert_eq!(
        mark_attributes.get("nemo_relay.mark.metadata_json"),
        Some(&"{\"source\":\"unit\"}".to_string())
    );

    let llm_with_scalar_input = make_start_event(
        Uuid::now_v7(),
        None,
        "raw-llm",
        ScopeType::Llm,
        Some(json!("hello")),
    );
    assert_eq!(
        openinference_input_value(&llm_with_scalar_input),
        Some(("hello".to_string(), "text/plain"))
    );

    let opaque_input = openinference_input_value(&make_start_event(
        Uuid::now_v7(),
        None,
        "opaque-llm",
        ScopeType::Llm,
        Some(json!({"headers": {"authorization": "Bearer token"}, "opaque": true})),
    ))
    .unwrap();
    assert_eq!(
        opaque_input,
        ("{\"opaque\":true}".to_string(), "application/json")
    );

    assert_eq!(
        display_text_from_string(r#"{"content":"json text"}"#),
        Some("json text".to_string())
    );
    assert_eq!(
        display_text_from_chat_choices(
            &json!([{"message": {"tool_calls": [{"toolName": "read"}]}}])
        ),
        Some("Requested tools: read".to_string())
    );
    let alias_usage = crate::observability::manual::usage_from_manual_llm_output(Some(&json!({
        "usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18, "cacheReadInputTokens": 5}
    })))
    .unwrap();
    assert_eq!(alias_usage.prompt_tokens, Some(11));
    assert_eq!(alias_usage.completion_tokens, Some(7));
    assert_eq!(alias_usage.total_tokens, Some(18));
    assert_eq!(alias_usage.cache_read_tokens, Some(5));

    let mut processor = OpenInferenceEventProcessor::new(make_provider().0, "test".into());
    processor.process(&make_end_event(
        Uuid::now_v7(),
        None,
        "missing",
        ScopeType::Agent,
        None,
    ));
    assert!(processor.active_spans.is_empty());

    let local_context = local_parent_span_context(&SpanContext::empty_context());
    assert!(!local_context.is_remote());

    let whole_second_pre_epoch = DateTime::parse_from_rfc3339("1969-12-31T23:59:58Z")
        .unwrap()
        .with_timezone(&Utc);
    assert_eq!(
        to_system_time(whole_second_pre_epoch),
        UNIX_EPOCH - Duration::from_secs(2)
    );
}

#[test]
fn provider_builders_cover_success_paths() {
    let http_provider = build_tracer_provider(
        &OpenInferenceConfig::new()
            .with_service_name("demo-agent")
            .with_header("authorization", "Bearer token")
            .with_resource_attribute("deployment.environment", "test")
            .with_service_namespace("agents")
            .with_service_version("1.2.3"),
    )
    .unwrap();
    http_provider.force_flush().unwrap();
    http_provider.shutdown().unwrap();

    let subscriber =
        OpenInferenceSubscriber::new(OpenInferenceConfig::new().with_service_name("http-success"))
            .unwrap();
    subscriber.force_flush().unwrap();
    subscriber.shutdown().unwrap();
}

#[test]
fn grpc_metadata_and_runtime_builder_paths_succeed() {
    let metadata = build_grpc_metadata(&HashMap::from([(
        "authorization".to_string(),
        "Bearer token".to_string(),
    )]))
    .unwrap();
    assert_eq!(
        metadata.get("authorization").unwrap().to_str().unwrap(),
        "Bearer token"
    );

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let provider = build_tracer_provider(
            &OpenInferenceConfig::new()
                .with_service_name("grpc-demo")
                .with_transport(OtlpTransport::Grpc)
                .with_endpoint("http://127.0.0.1:4317")
                .with_header("authorization", "Bearer token"),
        )
        .unwrap();
        provider.force_flush().ok();
        provider.shutdown().ok();
    });
}

#[test]
fn llm_end_with_usage_emits_token_count_attributes() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(uuid, None, "chat", ScopeType::Llm, None));
    processor.process(&make_scope_event_with_profile(
        ScopeCategory::End,
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"message": "hello"})),
        Some(
            CategoryProfile::builder()
                .model_name("gpt-4")
                .annotated_response(Arc::new(AnnotatedLlmResponse {
                    id: None,
                    model: None,
                    message: None,
                    tool_calls: None,
                    finish_reason: None,
                    usage: Some(Usage {
                        prompt_tokens: Some(100),
                        completion_tokens: Some(50),
                        total_tokens: Some(150),
                        cache_read_tokens: Some(25),
                        cache_write_tokens: Some(10),
                        cost: None,
                    }),
                    optimization_summary: None,
                    api_specific: None,
                    extra: serde_json::Map::new(),
                }))
                .build(),
        ),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(
        attributes.get("llm.token_count.prompt"),
        Some(&"100".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.completion"),
        Some(&"50".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.total"),
        Some(&"150".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.prompt_details.cache_read"),
        Some(&"25".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.prompt_details.cache_write"),
        Some(&"10".to_string())
    );
    assert!(!attributes.contains_key("llm.output_messages.0.message.role"));
}

#[test]
fn llm_end_with_known_model_usage_emits_derived_cost_attribute() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    install_test_pricing("priced-model");
    let _reset_guard = ResetPricingResolverGuard;
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(uuid, None, "chat", ScopeType::Llm, None));
    processor.process(&make_scope_event_with_profile(
        ScopeCategory::End,
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"message": "hello"})),
        Some(
            CategoryProfile::builder()
                .model_name("priced-model")
                .annotated_response(Arc::new(AnnotatedLlmResponse {
                    usage: Some(Usage {
                        prompt_tokens: Some(1_000),
                        completion_tokens: Some(500),
                        total_tokens: Some(1_500),
                        cache_read_tokens: Some(200),
                        cache_write_tokens: None,
                        cost: None,
                    }),
                    ..empty_annotated_response()
                }))
                .build(),
        ),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(
        attributes.get("llm.cost.total"),
        Some(&"0.000435".to_string())
    );
}

#[test]
fn llm_end_with_unannotated_openai_response_uses_codec_cost_attribute() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    install_openai_disambiguation_pricing("priced-model");
    let _reset_guard = ResetPricingResolverGuard;

    let event = make_end_event(
        Uuid::now_v7(),
        None,
        "other",
        ScopeType::Llm,
        Some(openai_chat_provider_response("priced-model")),
    );

    assert!(event.annotated_response().is_none());
    assert!(event.normalized_llm_response().is_some());

    let attributes = attr_map(&end_attributes(&event));
    assert_eq!(
        attributes.get("llm.cost.total"),
        Some(&"0.000435".to_string())
    );
}

#[test]
fn llm_end_with_unpriced_response_model_uses_requested_model_cost_attribute() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    install_openai_disambiguation_pricing("priced-model");
    let _reset_guard = ResetPricingResolverGuard;

    let event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "openai",
        ScopeType::Llm,
        Some(openai_chat_provider_response("api-echoed-model")),
        Some(
            CategoryProfile::builder()
                .model_name("priced-model")
                .build(),
        ),
    );

    assert!(event.annotated_response().is_none());
    let normalized = event.normalized_llm_response().unwrap();
    assert_eq!(normalized.model.as_deref(), Some("api-echoed-model"));

    let attributes = attr_map(&end_attributes(&event));
    assert_eq!(
        attributes.get("llm.cost.total"),
        Some(&"0.000435".to_string())
    );
}

#[test]
fn llm_end_with_unannotated_openai_response_without_usage_omits_cost_attribute() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    reset_active_pricing_resolver().unwrap();
    let _reset_guard = ResetPricingResolverGuard;

    let mut output = openai_chat_provider_response("priced-model");
    output.as_object_mut().unwrap().remove("usage");
    let event = make_end_event(Uuid::now_v7(), None, "openai", ScopeType::Llm, Some(output));

    assert!(event.annotated_response().is_none());
    let normalized = event.normalized_llm_response().unwrap();
    assert!(normalized.usage.is_none());

    let attributes = attr_map(&end_attributes(&event));
    assert!(!attributes.contains_key("llm.cost.total"));
}

#[test]
fn llm_end_with_manual_usage_and_output_model_emits_derived_cost_attribute() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    install_test_pricing("priced-model");
    let _reset_guard = ResetPricingResolverGuard;
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(uuid, None, "chat", ScopeType::Llm, None));
    processor.process(&make_end_event(
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({
            "model": "priced-model",
            "usage": {
                "prompt_tokens": 1_000,
                "completion_tokens": 500,
                "total_tokens": 1_500,
                "prompt_tokens_details": {"cached_tokens": 200}
            }
        })),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(
        attributes.get("llm.cost.total"),
        Some(&"0.000435".to_string())
    );
}

#[test]
fn llm_end_with_manual_component_cost_emits_cost_attribute() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(uuid, None, "chat", ScopeType::Llm, None));
    processor.process(&make_end_event(
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({
            "model": "unknown-model",
            "usage": {
                "prompt_tokens": 1_000,
                "completion_tokens": 500,
                "cost": {
                    "currency": "usd",
                    "input": 0.25,
                    "output": 0.5,
                    "cache_read": 0.125
                }
            }
        })),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(attributes.get("llm.cost.total"), Some(&"0.875".to_string()));
}

#[test]
fn llm_end_with_normalized_usage_cost_emits_cost_attribute() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(uuid, None, "chat", ScopeType::Llm, None));
    processor.process(&make_scope_event_with_profile(
        ScopeCategory::End,
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"message": "hello"})),
        Some(
            CategoryProfile::builder()
                .model_name("unknown-model")
                .annotated_response(Arc::new(AnnotatedLlmResponse {
                    usage: Some(Usage {
                        prompt_tokens: Some(1_000),
                        completion_tokens: Some(500),
                        cost: Some(CostEstimate {
                            total: Some(0.42),
                            currency: "USD".into(),
                            input: None,
                            output: None,
                            cache_read: None,
                            cache_write: None,
                            source: CostSource::ProviderReported,
                            pricing_provider: Some("external".to_string()),
                            pricing_model: Some("external-model".to_string()),
                            pricing_as_of: Some("2026-06-04".to_string()),
                            pricing_source: None,
                        }),
                        ..Usage::default()
                    }),
                    ..empty_annotated_response()
                }))
                .build(),
        ),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(attributes.get("llm.cost.total"), Some(&"0.42".to_string()));
}

#[test]
fn llm_end_with_component_only_usd_usage_cost_emits_cost_attribute() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(uuid, None, "chat", ScopeType::Llm, None));
    processor.process(&make_scope_event_with_profile(
        ScopeCategory::End,
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"message": "hello"})),
        Some(
            CategoryProfile::builder()
                .model_name("unknown-model")
                .annotated_response(Arc::new(AnnotatedLlmResponse {
                    usage: Some(Usage {
                        prompt_tokens: Some(1_000),
                        completion_tokens: Some(500),
                        cost: Some(CostEstimate {
                            total: None,
                            currency: "usd".into(),
                            input: Some(0.25),
                            output: Some(0.5),
                            cache_read: Some(0.125),
                            cache_write: None,
                            source: CostSource::ProviderReported,
                            pricing_provider: Some("external".to_string()),
                            pricing_model: Some("external-model".to_string()),
                            pricing_as_of: Some("2026-06-04".to_string()),
                            pricing_source: None,
                        }),
                        ..Usage::default()
                    }),
                    ..empty_annotated_response()
                }))
                .build(),
        ),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(attributes.get("llm.cost.total"), Some(&"0.875".to_string()));
}

#[test]
fn llm_end_with_non_usd_normalized_usage_cost_blocks_model_pricing_estimate() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    install_test_pricing("priced-model");
    let _reset_guard = ResetPricingResolverGuard;
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(uuid, None, "test", ScopeType::Llm, None));
    processor.process(&make_scope_event_with_profile(
        ScopeCategory::End,
        uuid,
        None,
        "test",
        ScopeType::Llm,
        Some(json!({"message": "hello"})),
        Some(
            CategoryProfile::builder()
                .model_name("priced-model")
                .annotated_response(Arc::new(AnnotatedLlmResponse {
                    usage: Some(Usage {
                        prompt_tokens: Some(1_000),
                        completion_tokens: Some(500),
                        cost: Some(CostEstimate {
                            total: Some(0.42),
                            currency: "EUR".into(),
                            input: None,
                            output: None,
                            cache_read: None,
                            cache_write: None,
                            source: CostSource::ProviderReported,
                            pricing_provider: Some("external".to_string()),
                            pricing_model: Some("external-model".to_string()),
                            pricing_as_of: Some("2026-06-04".to_string()),
                            pricing_source: None,
                        }),
                        ..Usage::default()
                    }),
                    ..empty_annotated_response()
                }))
                .build(),
        ),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let attributes = attr_map(&spans[0].attributes);
    assert!(!attributes.contains_key("llm.cost.total"));
}

#[test]
fn llm_end_with_unknown_model_usage_omits_derived_cost_attribute() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    reset_active_pricing_resolver().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(uuid, None, "chat", ScopeType::Llm, None));
    processor.process(&make_scope_event_with_profile(
        ScopeCategory::End,
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"message": "hello"})),
        Some(
            CategoryProfile::builder()
                .model_name("unknown-model")
                .annotated_response(Arc::new(AnnotatedLlmResponse {
                    usage: Some(Usage {
                        prompt_tokens: Some(1_000),
                        completion_tokens: Some(500),
                        ..Usage::default()
                    }),
                    ..empty_annotated_response()
                }))
                .build(),
        ),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let attributes = attr_map(&spans[0].attributes);
    assert!(!attributes.contains_key("llm.cost.total"));
}

#[test]
fn llm_end_with_manual_usage_payload_emits_token_count_attributes() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(uuid, None, "chat", ScopeType::Llm, None));
    processor.process(&make_scope_event_with_profile(
        ScopeCategory::End,
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({
            "content": "hello",
            "usage": {
                "prompt_tokens": 100
            },
            "token_usage": {
                "completion_tokens": 50,
                "total_tokens": 150,
                "cached_tokens": 25,
                "cache_write_tokens": 10
            }
        })),
        Some(CategoryProfile::builder().model_name("gpt-4").build()),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(
        attributes.get("llm.token_count.prompt"),
        Some(&"100".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.completion"),
        Some(&"50".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.total"),
        Some(&"150".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.prompt_details.cache_read"),
        Some(&"25".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.prompt_details.cache_write"),
        Some(&"10".to_string())
    );
    assert!(!attributes.contains_key("llm.cost.total"));
}

#[test]
fn anthropic_messages_output_emits_openinference_text_tool_and_usage_attributes() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_scope_event_with_profile(
        ScopeCategory::Start,
        uuid,
        None,
        "claude-sonnet-4",
        ScopeType::Llm,
        Some(json!({
            "messages": [{"role": "user", "content": "Find the file."}],
            "model": "claude-sonnet-4"
        })),
        Some(
            CategoryProfile::builder()
                .model_name("claude-sonnet-4")
                .build(),
        ),
    ));
    processor.process(&make_scope_event_with_profile(
        ScopeCategory::End,
        uuid,
        None,
        "claude-sonnet-4",
        ScopeType::Llm,
        Some(json!({
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4",
            "content": [
                {"type": "text", "text": "I will search for it."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "search",
                    "input": {"query": "file"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 5,
                "cost": {"total": 0.0042}
            }
        })),
        Some(
            CategoryProfile::builder()
                .model_name("claude-sonnet-4")
                .build(),
        ),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(
        attributes.get("openinference.span.kind"),
        Some(&"LLM".to_string())
    );
    assert_eq!(
        attributes.get("llm.model_name"),
        Some(&"claude-sonnet-4".to_string())
    );
    assert_eq!(
        attributes.get("input.value"),
        Some(&"user: Find the file.".to_string())
    );
    assert_eq!(
        attributes.get("output.value"),
        Some(&"I will search for it.\nRequested tools: search".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.prompt"),
        Some(&"11".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.completion"),
        Some(&"7".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.prompt_details.cache_read"),
        Some(&"3".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.prompt_details.cache_write"),
        Some(&"5".to_string())
    );
    assert_eq!(
        attributes.get("llm.cost.total"),
        Some(&"0.0042".to_string())
    );
}

#[test]
fn annotated_llm_payloads_emit_flattened_openinference_message_and_tool_attributes() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_scope_event_with_profile(
        ScopeCategory::Start,
        uuid,
        None,
        "annotated-chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_request(Arc::new(sample_openinference_annotated_request()))
                .build(),
        ),
    ));
    processor.process(&make_scope_event_with_profile(
        ScopeCategory::End,
        uuid,
        None,
        "annotated-chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_response(Arc::new(sample_openinference_annotated_response()))
                .build(),
        ),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_attr(&attributes, "llm.system", "Use concise answers.");
    assert_attr(&attributes, "llm.input_messages.0.message.role", "system");
    assert_attr(&attributes, "llm.input_messages.1.message.role", "user");
    assert_attr(
        &attributes,
        "llm.input_messages.1.message.content",
        "Search docs.",
    );
    assert_attr_contains(
        &attributes,
        "llm.invocation_parameters",
        "\"temperature\":0.2",
    );
    assert_attr_contains(
        &attributes,
        "llm.tools.0.tool.json_schema",
        "\"name\":\"search_docs\"",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.role",
        "assistant",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.content",
        "I will search docs.",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.tool_calls.0.tool_call.id",
        "call-search-docs",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name",
        "search_docs",
    );
    assert_attr(
        &attributes,
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments",
        "{\"query\":\"docs\"}",
    );
    assert_attr(&attributes, "llm.finish_reason", "tool_use");
}

#[test]
fn hermes_exact_api_payloads_emit_openinference_text_usage_and_metadata() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();
    let metadata = json!({
        "provider_payload_exact": true,
        "fidelity_source": "hermes_api_hooks_sanitized"
    });

    processor.process(&Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(uuid)
            .name("custom")
            .data(json!({
                "model": "qwen",
                "messages": [{ "role": "user", "content": "hello" }],
                "tools": [
                    { "type": "function", "function": { "name": "search_files" } }
                ]
            }))
            .metadata(metadata.clone())
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::llm(),
        Some(CategoryProfile::builder().model_name("qwen").build()),
    )));
    processor.process(&Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(uuid)
            .name("custom")
            .data(json!({
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
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cost": { "total": 0.0042 }
                },
                "model": "qwen",
                "finish_reason": "tool_calls"
            }))
            .metadata(metadata)
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::llm(),
        Some(CategoryProfile::builder().model_name("qwen").build()),
    )));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(
        attributes.get("openinference.span.kind"),
        Some(&"LLM".to_string())
    );
    assert_eq!(attributes.get("llm.model_name"), Some(&"qwen".to_string()));
    assert_eq!(
        attributes.get("input.value"),
        Some(&"user: hello".to_string())
    );
    assert_eq!(
        attributes.get("output.value"),
        Some(&"Requested tools: search_files".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.prompt"),
        Some(&"10".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.completion"),
        Some(&"5".to_string())
    );
    assert_eq!(
        attributes.get("llm.cost.total"),
        Some(&"0.0042".to_string())
    );
    assert_attr_contains(&attributes, "metadata", "\"provider_payload_exact\":true");
    assert_attr_contains(
        &attributes,
        "metadata",
        "\"fidelity_source\":\"hermes_api_hooks_sanitized\"",
    );
}

#[test]
fn hermes_api_request_error_emits_openinference_json_output_and_metadata() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();
    let start_metadata = json!({
        "provider_payload_exact": true,
        "fidelity_source": "hermes_api_hooks_sanitized"
    });
    let end_metadata = json!({
        "provider_payload_exact": false,
        "fidelity_source": "hermes_api_hooks"
    });

    processor.process(&Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(uuid)
            .name("custom")
            .data(json!({
                "model": "qwen",
                "messages": [{ "role": "user", "content": "hello" }]
            }))
            .metadata(start_metadata)
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::llm(),
        Some(CategoryProfile::builder().model_name("qwen").build()),
    )));
    processor.process(&Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(uuid)
            .name("custom")
            .data(json!({
                "status_code": 502,
                "retry_count": 1,
                "max_retries": 2,
                "retryable": true,
                "reason": "upstream",
                "error": {
                    "type": "BadGateway",
                    "message": "gateway upstream error"
                }
            }))
            .metadata(end_metadata)
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::llm(),
        Some(CategoryProfile::builder().model_name("qwen").build()),
    )));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(
        attributes.get("openinference.span.kind"),
        Some(&"LLM".to_string())
    );
    assert_eq!(attributes.get("llm.model_name"), Some(&"qwen".to_string()));
    assert_eq!(
        attributes.get("input.value"),
        Some(&"user: hello".to_string())
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(attributes.get("output.value").unwrap()).unwrap(),
        json!({
            "status_code": 502,
            "retry_count": 1,
            "max_retries": 2,
            "retryable": true,
            "reason": "upstream",
            "error": {
                "type": "BadGateway",
                "message": "gateway upstream error"
            }
        })
    );
    assert_eq!(
        attributes.get("output.mime_type"),
        Some(&"application/json".to_string())
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(
            attributes.get("nemo_relay.end.output_json").unwrap(),
        )
        .unwrap(),
        json!({
            "status_code": 502,
            "retry_count": 1,
            "max_retries": 2,
            "retryable": true,
            "reason": "upstream",
            "error": {
                "type": "BadGateway",
                "message": "gateway upstream error"
            }
        })
    );
    assert_attr_contains(&attributes, "metadata", "\"provider_payload_exact\":false");
    assert_attr_contains(
        &attributes,
        "metadata",
        "\"fidelity_source\":\"hermes_api_hooks\"",
    );
}

#[test]
fn llm_end_with_inconsistent_manual_usage_omits_invalid_total_tokens() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(uuid, None, "chat", ScopeType::Llm, None));
    processor.process(&make_scope_event_with_profile(
        ScopeCategory::End,
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({
            "content": "hello",
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 10,
                "total_tokens": 5
            }
        })),
        Some(CategoryProfile::builder().model_name("gpt-4").build()),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(
        attributes.get("llm.token_count.prompt"),
        Some(&"3".to_string())
    );
    assert_eq!(
        attributes.get("llm.token_count.completion"),
        Some(&"10".to_string())
    );
    assert!(!attributes.contains_key("llm.token_count.total"));
}

#[test]
fn llm_end_without_usage_omits_token_count_attributes() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(uuid, None, "chat", ScopeType::Llm, None));
    processor.process(&make_end_event(
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"message": "hello"})),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert!(!attributes.contains_key("llm.token_count.prompt"));
    assert!(!attributes.contains_key("llm.token_count.completion"));
    assert!(!attributes.contains_key("llm.token_count.total"));
}

#[test]
fn llm_end_with_partial_usage_emits_only_present_fields() {
    let (provider, exporter) = make_provider();
    let mut processor =
        OpenInferenceEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(uuid, None, "chat", ScopeType::Llm, None));
    processor.process(&make_scope_event_with_profile(
        ScopeCategory::End,
        uuid,
        None,
        "chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_response(Arc::new(AnnotatedLlmResponse {
                    id: None,
                    model: None,
                    message: None,
                    tool_calls: None,
                    finish_reason: None,
                    usage: Some(Usage {
                        prompt_tokens: Some(42),
                        completion_tokens: None,
                        total_tokens: None,
                        cache_read_tokens: None,
                        cache_write_tokens: None,
                        cost: None,
                    }),
                    optimization_summary: None,
                    api_specific: None,
                    extra: serde_json::Map::new(),
                }))
                .build(),
        ),
    ));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_eq!(
        attributes.get("llm.token_count.prompt"),
        Some(&"42".to_string())
    );
    assert!(!attributes.contains_key("llm.token_count.completion"));
    assert!(!attributes.contains_key("llm.token_count.total"));
    assert!(!attributes.contains_key("llm.token_count.prompt_details.cache_read"));
}
