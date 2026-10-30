// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core provider normalization and scope integration with OpenInference.

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
use crate::json::Json;
use crate::observability_test_support::MarkProjection;
use crate::observability_test_support::atif::{AtifAgentInfo, AtifExporter, AtifStepExtra};
use crate::observability_test_support::otel::RelayIdGenerator;
use nemo_relay_model_pricing::pricing_test_mutex;
use nemo_relay_model_pricing::{
    PricingCatalog, PricingResolver, reset_active_pricing_resolver, set_active_pricing_resolver,
};
use nemo_relay_types::codec::request::{
    AnnotatedLlmRequest, FunctionDefinition, GenerationParams, Message, MessageContent,
    ToolDefinition,
};
use nemo_relay_types::codec::response::{
    AnnotatedLlmResponse, CostEstimate, CostSource, FinishReason, ResponseToolCall, Usage,
};
use opentelemetry::trace::SpanContext;
use opentelemetry::trace::Status;
use opentelemetry_sdk::trace::InMemorySpanExporterBuilder;
use opentelemetry_sdk::trace::SdkTracerProvider;
use serde_json::json;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
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
    let _ = spdlog::init_log_crate_proxy();
    log::set_max_level(log::LevelFilter::Info);
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
        .with_id_generator(RelayIdGenerator)
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

fn assert_no_attr_contains(attributes: &HashMap<String, String>, expected: &str) {
    assert!(
        !attributes
            .iter()
            .any(|(key, value)| key.contains(expected) || value.contains(expected)),
        "attribute map unexpectedly contained {expected:?}"
    );
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
fn subscriber_registration_and_provider_lifecycle_methods_work() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_global();

    let (provider, _exporter) = make_provider();
    let subscriber =
        crate::observability_test_support::otel::OpenTelemetrySubscriber::from_tracer_provider_with_type(
            provider,
            "test-scope",
            crate::observability_test_support::OpenTelemetryType::OpenInference,
        )
        .with_host(crate::api::subscriber::observability_host());
    let name = format!("otel_test_{}", Uuid::now_v7().simple());

    subscriber.register(&name).unwrap();
    assert!(subscriber.deregister(&name).unwrap());
    assert!(!subscriber.deregister(&name).unwrap());
    subscriber.force_flush().unwrap();
    subscriber.shutdown().unwrap();
}

#[test]
fn registered_subscriber_emits_spans_for_scope_push_pop_and_marks() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_global();

    let (provider, exporter) = make_provider();
    let subscriber =
        crate::observability_test_support::otel::OpenTelemetrySubscriber::from_tracer_provider_with_type(
            provider,
            "e2e-scope",
            crate::observability_test_support::OpenTelemetryType::OpenInference,
        )
        .with_host(crate::api::subscriber::observability_host());
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
    assert_registered_scope_span(&spans[0]);
}

fn assert_registered_scope_span(span: &opentelemetry_sdk::trace::SpanData) {
    assert_eq!(span.name.as_ref(), "otel_scope");
    assert_eq!(span.events.events.len(), 1);
    assert_eq!(span.events.events[0].name.as_ref(), "otel_mark");
    assert_registered_scope_attributes(span);
    assert_registered_scope_event_attributes(span);
}

fn assert_registered_scope_attributes(span: &opentelemetry_sdk::trace::SpanData) {
    let attributes = attr_map(&span.attributes);
    assert_eq!(
        attributes.get("openinference.span.kind"),
        Some(&"AGENT".to_string())
    );
    assert!(!attributes.contains_key("nemo_relay.start.data_json"));
    assert!(!attributes.contains_key("nemo_relay.start.metadata_json"));
    assert_eq!(
        attributes.get("nemo_relay.start.input.task"),
        Some(&"scope-start".to_string())
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
        attributes.get("openinference.metadata.phase"),
        Some(&"start".to_string())
    );
    assert_eq!(
        attributes.get("metadata"),
        Some(&"{\"phase\":\"start\"}".to_string())
    );
}

fn assert_registered_scope_event_attributes(span: &opentelemetry_sdk::trace::SpanData) {
    let event_attributes = attr_map(&span.events.events[0].attributes);
    assert_eq!(
        event_attributes.get("nemo_relay.mark.data.step"),
        Some(&"1".to_string())
    );
    assert_eq!(
        event_attributes.get("nemo_relay.mark.metadata.source"),
        Some(&"rust-test".to_string())
    );
    assert!(!event_attributes.contains_key("tool.name"));
    assert!(!event_attributes.contains_key("tool_call.function.name"));
    assert!(!event_attributes.contains_key("output.value"));
    assert!(!event_attributes.contains_key("output.mime_type"));
    assert!(!event_attributes.contains_key("metadata"));
}

#[test]
fn http_config_exports_scope_push_pop_and_marks_without_tokio_runtime() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_global();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1/traces", listener.local_addr().unwrap());
    let (request_tx, request_rx) = mpsc::channel();
    spawn_http_collector(listener, request_tx);

    let config = crate::observability_test_support::otel::OpenTelemetryConfig::new(
        crate::observability_test_support::OpenTelemetryType::OpenInference,
        endpoint,
    )
    .with_service_name("demo-agent");
    let subscriber = crate::observability_test_support::otel::OpenTelemetrySubscriber::new(config)
        .map(|exporter| exporter.with_host(crate::api::subscriber::observability_host()))
        .unwrap();
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
fn llm_input_value_omits_request_headers() {
    let (provider, exporter) = make_provider();
    let mut processor =
        crate::observability_test_support::otel::OtelEventProcessor::new_openinference(
            provider.clone(),
            "test-scope".to_string(),
        );
    let root_uuid = Uuid::now_v7();

    processor.process(&crate::observability_runtime::event_view(
        &make_start_event(
            root_uuid,
            None,
            "chat",
            ScopeType::Llm,
            Some(json!({
                "headers": {"authorization": "Bearer secret-token"},
                "content": {"messages": [{"role": "user", "content": "hi"}], "model": "demo-model"}
            })),
        ),
    ));
    processor.process(&crate::observability_runtime::event_view(&make_end_event(
        root_uuid,
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"message": "hello"})),
    )));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert_attr(&attributes, "input.value", "user: hi");
    assert_attr(&attributes, "input.mime_type", "text/plain");
    assert!(!attributes.contains_key("nemo_relay.start.input.content"));
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
        crate::observability_test_support::otel::OtelEventProcessor::new_openinference(
            provider.clone(),
            "test-scope".to_string(),
        );
    let uuid = Uuid::now_v7();

    processor.process(&crate::observability_runtime::event_view(
        &make_start_event(
            uuid,
            None,
            "chat",
            ScopeType::Llm,
            Some(json!({
                "headers": {},
                "content": {"messages": [{"role": "user", "content": "hi"}], "model": "demo-model"}
            })),
        ),
    ));
    processor.process(&crate::observability_runtime::event_view(&make_end_event(
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
    )));

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
