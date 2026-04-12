// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::api::scope::{event, pop_scope, push_scope};
use crate::context::global::global_context;
use crate::context::state::NemoFlowContextState;
use crate::json::Json;
use crate::types::llm::LLMAttributes;
use crate::types::scope::{ScopeAttributes, ScopeType};
use crate::types::tool::ToolAttributes;
use opentelemetry_sdk::trace::InMemorySpanExporterBuilder;
use serde_json::json;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::thread;
use uuid::Uuid;

fn reset_global() {
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoFlowContextState::new();
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
    match scope_type {
        ScopeType::Tool => Event::tool_start(
            parent_uuid,
            uuid,
            name,
            None,
            None,
            ToolAttributes::empty(),
            input,
            None,
        ),
        ScopeType::Llm => Event::llm_start(
            parent_uuid,
            uuid,
            name,
            None,
            None,
            LLMAttributes::empty(),
            input,
            None,
            None,
        ),
        _ => Event::scope_start(
            parent_uuid,
            uuid,
            name,
            None,
            None,
            ScopeAttributes::empty(),
            scope_type,
        ),
    }
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
    assert_eq!(defaults.service_name, "nemo-flow");
    assert_eq!(defaults.instrumentation_scope, "nemo-flow-otel");
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
        "otel_scope",
        ScopeType::Agent,
        None,
        ScopeAttributes::empty(),
        Some(json!({"scope": true})),
        Some(json!({"phase": "start"})),
    )
    .unwrap();
    event(
        "otel_mark",
        Some(&handle),
        Some(json!({"step": 1})),
        Some(json!({"source": "rust-test"})),
    )
    .unwrap();
    pop_scope(&handle.uuid).unwrap();

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
        attributes.get("nemo_flow.start.data_json"),
        Some(&"{\"scope\":true}".to_string())
    );
    assert_eq!(
        attributes.get("nemo_flow.start.metadata_json"),
        Some(&"{\"phase\":\"start\"}".to_string())
    );

    let event_attributes = attr_map(&span.events.events[0].attributes);
    assert_eq!(
        event_attributes.get("nemo_flow.mark.data_json"),
        Some(&"{\"step\":1}".to_string())
    );
    assert_eq!(
        event_attributes.get("nemo_flow.mark.metadata_json"),
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

    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut bytes = Vec::new();
        let mut buf = [0_u8; 4096];

        let (header_end, content_length) = loop {
            let read = stream.read(&mut buf).unwrap();
            if read == 0 {
                panic!("collector closed before receiving an OTLP request");
            }
            bytes.extend_from_slice(&buf[..read]);

            if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                let header_end = header_end + 4;
                let headers_text = String::from_utf8_lossy(&bytes[..header_end]);
                let content_length = headers_text
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        if name.eq_ignore_ascii_case("content-length") {
                            value.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                break (header_end, content_length);
            }
        };

        while bytes.len() < header_end + content_length {
            let read = stream.read(&mut buf).unwrap();
            if read == 0 {
                break;
            }
            bytes.extend_from_slice(&buf[..read]);
        }

        let headers_text = String::from_utf8_lossy(&bytes[..header_end]);
        let request_line = headers_text.lines().next().unwrap();
        let path = request_line.split_whitespace().nth(1).unwrap().to_string();
        let content_type = headers_text
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.eq_ignore_ascii_case("content-type") {
                    Some(value.trim().to_string())
                } else {
                    None
                }
            })
            .unwrap_or_default();
        let body = bytes[header_end..header_end + content_length].to_vec();

        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
        request_tx.send((path, content_type, body)).unwrap();
    });

    let config = OpenTelemetryConfig::http_binary("demo-agent").with_endpoint(endpoint);
    let subscriber = OpenTelemetrySubscriber::new(config).unwrap();
    let name = format!("otel_http_{}", Uuid::now_v7().simple());

    subscriber.register(&name).unwrap();
    let handle = push_scope(
        "otel_scope",
        ScopeType::Agent,
        None,
        ScopeAttributes::empty(),
        Some(json!({"scope": true})),
        None,
    )
    .unwrap();
    event(
        "otel_mark",
        Some(&handle),
        Some(json!({"step": 1})),
        Some(json!({"source": "rust-http"})),
    )
    .unwrap();
    pop_scope(&handle.uuid).unwrap();

    assert!(subscriber.deregister(&name).unwrap());
    subscriber.force_flush().unwrap();

    let (path, content_type, body) = request_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("expected an OTLP request");
    assert_eq!(path, "/v1/traces");
    assert_eq!(content_type, "application/x-protobuf");
    assert!(!body.is_empty());
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

    let mark = Event::mark(
        Some(root_uuid),
        Uuid::now_v7(),
        "checkpoint",
        Some(json!({"step": 1})),
        None,
    );
    processor.process(&mark);

    let end = Event::tool_end(
        None,
        root_uuid,
        "search",
        None,
        None,
        ToolAttributes::empty(),
        Some(json!({"result": "ok"})),
        None,
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
        attributes.get("nemo_flow.uuid"),
        Some(&root_uuid.to_string())
    );
    assert_eq!(
        attributes.get("nemo_flow.start.input_json"),
        Some(&"{\"query\":\"hello\"}".to_string())
    );
    assert_eq!(
        attributes.get("nemo_flow.end.output_json"),
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
    processor.process(&Event::llm_end(
        Some(root_uuid),
        child_uuid,
        "model-call",
        None,
        None,
        LLMAttributes::empty(),
        None,
        None,
        None,
    ));
    processor.process(&Event::scope_end(
        None,
        root_uuid,
        "agent",
        None,
        None,
        ScopeAttributes::empty(),
        ScopeType::Agent,
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
fn orphan_marks_become_zero_duration_spans() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());
    let mark = Event::mark(
        None,
        Uuid::now_v7(),
        "detached",
        Some(json!({"kind": "standalone"})),
        None,
    );

    processor.process(&mark);
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let span = &spans[0];
    assert_eq!(span.name.as_ref(), "mark:detached");
    assert_eq!(span.start_time, span.end_time);

    let attributes = attr_map(&span.attributes);
    assert_eq!(
        attributes.get("nemo_flow.mark.orphan"),
        Some(&"true".to_string())
    );
}

#[test]
fn semantic_scope_type_and_span_kind_follow_event_variants() {
    let scope_event = Event::scope_start(
        None,
        Uuid::now_v7(),
        "guardrail",
        None,
        None,
        ScopeAttributes::empty(),
        ScopeType::Guardrail,
    );
    assert_eq!(
        semantic_scope_type(&scope_event),
        Some(ScopeType::Guardrail)
    );
    assert_eq!(span_kind(&scope_event), SpanKind::Internal);

    let local_tool = Event::tool_start(
        None,
        Uuid::now_v7(),
        "search",
        None,
        None,
        ToolAttributes::LOCAL,
        Some(json!({"query": "hello"})),
        None,
    );
    assert_eq!(semantic_scope_type(&local_tool), Some(ScopeType::Tool));
    assert_eq!(span_kind(&local_tool), SpanKind::Internal);

    let llm_event = Event::llm_end(
        None,
        Uuid::now_v7(),
        "model-call",
        None,
        None,
        LLMAttributes::empty(),
        Some(json!({"result": "hello"})),
        None,
        None,
    );
    assert_eq!(semantic_scope_type(&llm_event), Some(ScopeType::Llm));
    assert_eq!(span_kind(&llm_event), SpanKind::Client);

    let mark = Event::mark(None, Uuid::now_v7(), "checkpoint", None, None);
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
    let function_end = Event::scope_end(
        None,
        Uuid::now_v7(),
        "fn-scope",
        None,
        None,
        ScopeAttributes::empty(),
        ScopeType::Function,
    );
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

    let llm_event = Event::llm_end(
        None,
        Uuid::now_v7(),
        "chat",
        None,
        None,
        LLMAttributes::empty(),
        Some(json!({"answer": "ok"})),
        Some("demo-model".into()),
        None,
    );
    let llm_attributes = attr_map(&common_attributes(&llm_event));
    assert_eq!(
        llm_attributes.get("nemo_flow.model_name"),
        Some(&"demo-model".to_string())
    );

    let tool_event = Event::tool_start(
        None,
        Uuid::now_v7(),
        "lookup",
        Some(json!({"step": 1})),
        Some(json!({"meta": true})),
        ToolAttributes::empty(),
        Some(json!({"query": "hello"})),
        Some("call-123".into()),
    );
    let tool_attributes = attr_map(&common_attributes(&tool_event));
    assert_eq!(
        tool_attributes.get("nemo_flow.tool_call_id"),
        Some(&"call-123".to_string())
    );

    let start_attributes = attr_map(&start_attributes(&tool_event));
    assert_eq!(
        start_attributes.get("nemo_flow.start.input_json"),
        Some(&"{\"query\":\"hello\"}".to_string())
    );
    assert_eq!(
        start_attributes.get("nemo_flow.start.metadata_json"),
        Some(&"{\"meta\":true}".to_string())
    );

    let end_attributes = attr_map(&end_attributes(&Event::tool_end(
        None,
        Uuid::now_v7(),
        "lookup",
        None,
        Some(json!({"phase": "complete"})),
        ToolAttributes::empty(),
        Some(json!({"result": true})),
        Some("call-456".into()),
    )));
    assert_eq!(
        end_attributes.get("nemo_flow.end.output_json"),
        Some(&"{\"result\":true}".to_string())
    );

    let mark = Event::mark(
        Some(Uuid::now_v7()),
        Uuid::now_v7(),
        "checkpoint",
        Some(json!({"kind": "aux"})),
        Some(json!({"source": "unit"})),
    );
    let mark_attributes = attr_map(&mark_attributes(&mark));
    assert_eq!(
        mark_attributes.get("nemo_flow.mark.data_json"),
        Some(&"{\"kind\":\"aux\"}".to_string())
    );
    assert_eq!(
        mark_attributes.get("nemo_flow.mark.metadata_json"),
        Some(&"{\"source\":\"unit\"}".to_string())
    );

    let mut processor = OtelEventProcessor::new(make_provider().0, "test".into());
    processor.process(&Event::scope_end(
        None,
        Uuid::now_v7(),
        "missing",
        None,
        None,
        ScopeAttributes::empty(),
        ScopeType::Agent,
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
