// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for otel in the NeMo Relay core crate.

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
use crate::codec::pricing::pricing_test_mutex;
use crate::codec::response::{
    AnnotatedLlmResponse, CostEstimate, CostSource, PricingCatalog, PricingResolver, Usage,
    reset_active_pricing_resolver, set_active_pricing_resolver,
};
use crate::json::Json;
use crate::observability::atif::{AtifAgentInfo, AtifExporter, AtifStepExtra};
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

fn empty_annotated_response() -> AnnotatedLlmResponse {
    AnnotatedLlmResponse {
        id: None,
        model: None,
        message: None,
        tool_calls: None,
        finish_reason: None,
        usage: None,
        api_specific: None,
        extra: serde_json::Map::new(),
    }
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

fn install_provider_disambiguation_pricing(model_id: &str) {
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
    let config = OpenTelemetryConfig::http_binary("demo-agent")
        .with_endpoint("http://localhost:4318/v1/traces")
        .with_header("authorization", "Bearer token")
        .with_resource_attribute("deployment.environment", "test")
        .with_service_namespace("agents")
        .with_service_version("1.2.3")
        .with_instrumentation_scope("demo-scope")
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
    assert_eq!(config.timeout, Duration::from_millis(1250));

    let defaults = OpenTelemetryConfig::default();
    assert_eq!(defaults.transport, OtlpTransport::HttpBinary);
    assert_eq!(defaults.service_name, "nemo-relay");
    assert_eq!(defaults.instrumentation_scope, "nemo-relay-otel");
    assert_eq!(defaults.timeout, Duration::from_secs(3));
    assert!(defaults.headers.is_empty());
    assert!(defaults.resource_attributes.is_empty());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn grpc_config_requires_a_tokio_runtime() {
    let err = match OpenTelemetrySubscriber::new(OpenTelemetryConfig::grpc("demo-agent")) {
        Ok(_) => panic!("gRPC construction should require a Tokio runtime"),
        Err(err) => err,
    };
    assert!(matches!(err, OpenTelemetryError::MissingTokioRuntime));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn invalid_grpc_headers_are_rejected() {
    let err = build_grpc_metadata(&HashMap::from([(
        "bad key".to_string(),
        "value".to_string(),
    )]))
    .expect_err("invalid metadata key should fail");
    assert!(matches!(err, OpenTelemetryError::InvalidGrpcHeader { .. }));
}

#[test]
fn subscriber_registration_and_provider_lifecycle_methods_work() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_global();

    let (provider, _exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "test-scope");
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
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "e2e-scope");
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
        attributes.get("nemo_relay.start.data_json"),
        Some(&"{\"task\":\"scope-start\"}".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.start.metadata_json"),
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

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn http_config_exports_scope_push_pop_and_marks_without_tokio_runtime() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_global();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1/traces", listener.local_addr().unwrap());
    let (request_tx, request_rx) = mpsc::channel();
    spawn_http_collector(listener, request_tx);

    let config = OpenTelemetryConfig::http_binary("demo-agent").with_endpoint(endpoint);
    let subscriber = OpenTelemetrySubscriber::new(config).unwrap();
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
    let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());
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
fn preserves_parent_child_relationships() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());

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
fn atif_lineage_correlates_with_otel_span_attributes() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());

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
    let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());
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
}

#[test]
fn semantic_scope_type_and_span_kind_follow_event_variants() {
    let scope_event = make_start_event(
        Uuid::now_v7(),
        None,
        "guardrail",
        ScopeType::Guardrail,
        Some(json!({"input": true})),
    );
    assert_eq!(
        semantic_scope_type(&scope_event),
        Some(ScopeType::Guardrail)
    );
    assert_eq!(span_kind(&scope_event), SpanKind::Internal);

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

    let llm_event = make_end_event(
        Uuid::now_v7(),
        None,
        "model-call",
        ScopeType::Llm,
        Some(json!({"result": "hello"})),
    );
    assert_eq!(semantic_scope_type(&llm_event), Some(ScopeType::Llm));
    assert_eq!(span_kind(&llm_event), SpanKind::Client);

    let mark = make_mark_event(None, "checkpoint", None);
    assert_eq!(semantic_scope_type(&mark), None);
    assert_eq!(span_kind(&mark), SpanKind::Internal);
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
fn helper_functions_cover_additional_otel_branches() {
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

    let llm_event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"answer": "ok"})),
        Some(CategoryProfile::builder().model_name("demo-model").build()),
    );
    let llm_attributes = attr_map(&common_attributes(&llm_event));
    assert_eq!(
        llm_attributes.get("nemo_relay.model_name"),
        Some(&"demo-model".to_string())
    );

    let tool_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("lookup")
            .data(json!({"query": "hello"}))
            .metadata(json!({"meta": true}))
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::tool(),
        Some(CategoryProfile::builder().tool_call_id("call-123").build()),
    ));
    let tool_attributes = attr_map(&common_attributes(&tool_event));
    assert_eq!(
        tool_attributes.get("nemo_relay.tool_call_id"),
        Some(&"call-123".to_string())
    );

    let start_attributes = attr_map(&start_attributes(&tool_event));
    assert_eq!(
        start_attributes.get("nemo_relay.start.input_json"),
        Some(&"{\"query\":\"hello\"}".to_string())
    );
    assert_eq!(
        start_attributes.get("nemo_relay.start.metadata_json"),
        Some(&"{\"meta\":true}".to_string())
    );

    let tool_end_attributes = attr_map(&end_attributes(&Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("lookup")
            .metadata(json!({"phase": "complete"}))
            .data(json!({"result": true}))
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::tool(),
        Some(CategoryProfile::builder().tool_call_id("call-456").build()),
    ))));
    assert_eq!(
        tool_end_attributes.get("nemo_relay.end.output_json"),
        Some(&"{\"result\":true}".to_string())
    );

    {
        let _pricing_guard = pricing_test_mutex().lock().unwrap();
        install_test_pricing("priced-model");
        let _reset_guard = ResetPricingResolverGuard;
        let llm_cost_event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
            None,
            "chat",
            ScopeType::Llm,
            Some(json!({"answer": "ok"})),
            Some(
                CategoryProfile::builder()
                    .model_name("priced-model")
                    .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
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
        );
        let llm_cost_attributes = attr_map(&end_attributes(&llm_cost_event));
        assert_eq!(
            llm_cost_attributes.get("nemo_relay.llm.cost.total"),
            Some(&"0.000435".to_string())
        );
        assert_eq!(
            llm_cost_attributes.get("nemo_relay.llm.cost.currency"),
            Some(&"USD".to_string())
        );
    }

    {
        let _pricing_guard = pricing_test_mutex().lock().unwrap();
        install_provider_disambiguation_pricing("priced-model");
        let _reset_guard = ResetPricingResolverGuard;
        let provider_qualified_cost_event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
            None,
            "test",
            ScopeType::Llm,
            Some(json!({"answer": "ok"})),
            Some(
                CategoryProfile::builder()
                    .model_name("priced-model")
                    .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
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
        );
        let provider_qualified_cost_attributes =
            attr_map(&end_attributes(&provider_qualified_cost_event));
        assert_eq!(
            provider_qualified_cost_attributes.get("nemo_relay.llm.cost.total"),
            Some(&"0.000435".to_string())
        );
        assert_eq!(
            provider_qualified_cost_attributes.get("nemo_relay.llm.cost.currency"),
            Some(&"USD".to_string())
        );
    }

    let normalized_cost_event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"answer": "ok"})),
        Some(
            CategoryProfile::builder()
                .model_name("unknown-model")
                .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
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
    );
    let normalized_cost_attributes = attr_map(&end_attributes(&normalized_cost_event));
    assert_eq!(
        normalized_cost_attributes.get("nemo_relay.llm.cost.total"),
        Some(&"0.42".to_string())
    );
    assert_eq!(
        normalized_cost_attributes.get("nemo_relay.llm.cost.currency"),
        Some(&"USD".to_string())
    );

    {
        let _pricing_guard = pricing_test_mutex().lock().unwrap();
        install_test_pricing("priced-model");
        let _reset_guard = ResetPricingResolverGuard;
        let reported_cost_without_total_event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
            None,
            "test",
            ScopeType::Llm,
            Some(json!({"answer": "ok"})),
            Some(
                CategoryProfile::builder()
                    .model_name("priced-model")
                    .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                        usage: Some(Usage {
                            prompt_tokens: Some(1_000),
                            completion_tokens: Some(500),
                            cost: Some(CostEstimate {
                                total: None,
                                currency: "EUR".into(),
                                input: Some(0.10),
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
        );
        let reported_cost_without_total_attributes =
            attr_map(&end_attributes(&reported_cost_without_total_event));
        assert_eq!(
            reported_cost_without_total_attributes.get("nemo_relay.llm.cost.total"),
            Some(&"0.1".to_string())
        );
        assert_eq!(
            reported_cost_without_total_attributes.get("nemo_relay.llm.cost.currency"),
            Some(&"EUR".to_string())
        );
    }

    {
        let _pricing_guard = pricing_test_mutex().lock().unwrap();
        install_test_pricing("priced-model");
        let _reset_guard = ResetPricingResolverGuard;
        let manual_cost_event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
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
            None,
        );
        let manual_cost_attributes = attr_map(&end_attributes(&manual_cost_event));
        assert_eq!(
            manual_cost_attributes.get("nemo_relay.llm.cost.total"),
            Some(&"0.000435".to_string())
        );
        assert_eq!(
            manual_cost_attributes.get("nemo_relay.llm.cost.currency"),
            Some(&"USD".to_string())
        );

        let manual_component_cost_event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
            None,
            "chat",
            ScopeType::Llm,
            Some(json!({
                "model": "unknown-model",
                "usage": {
                    "prompt_tokens": 1_000,
                    "completion_tokens": 500,
                    "cost": {
                        "currency": "EUR",
                        "input": 0.25,
                        "output": 0.5,
                        "cache_read": 0.125
                    }
                }
            })),
            None,
        );
        let manual_component_cost_attributes =
            attr_map(&end_attributes(&manual_component_cost_event));
        assert_eq!(
            manual_component_cost_attributes.get("nemo_relay.llm.cost.total"),
            Some(&"0.875".to_string())
        );
        assert_eq!(
            manual_component_cost_attributes.get("nemo_relay.llm.cost.currency"),
            Some(&"EUR".to_string())
        );

        let annotated_without_model_event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
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
            Some(
                CategoryProfile::builder()
                    .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                        usage: Some(Usage {
                            prompt_tokens: Some(1_000),
                            completion_tokens: Some(500),
                            total_tokens: Some(1_500),
                            cache_read_tokens: Some(200),
                            ..Usage::default()
                        }),
                        ..empty_annotated_response()
                    }))
                    .build(),
            ),
        );
        let annotated_without_model_attributes =
            attr_map(&end_attributes(&annotated_without_model_event));
        assert_eq!(
            annotated_without_model_attributes.get("nemo_relay.llm.cost.total"),
            Some(&"0.000435".to_string())
        );
        assert_eq!(
            annotated_without_model_attributes.get("nemo_relay.llm.cost.currency"),
            Some(&"USD".to_string())
        );
    }

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

    let mut processor = OtelEventProcessor::new(make_provider().0, "test".into());
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
        &OpenTelemetryConfig::http_binary("demo-agent")
            .with_header("authorization", "Bearer token")
            .with_resource_attribute("deployment.environment", "test")
            .with_service_namespace("agents")
            .with_service_version("1.2.3"),
    )
    .unwrap();
    http_provider.force_flush().unwrap();
    http_provider.shutdown().unwrap();

    let subscriber =
        OpenTelemetrySubscriber::new(OpenTelemetryConfig::http_binary("http-success")).unwrap();
    subscriber.force_flush().unwrap();
    subscriber.shutdown().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
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
            &OpenTelemetryConfig::grpc("grpc-demo")
                .with_endpoint("http://127.0.0.1:4317")
                .with_header("authorization", "Bearer token"),
        )
        .unwrap();
        provider.force_flush().ok();
        provider.shutdown().ok();
    });
}
