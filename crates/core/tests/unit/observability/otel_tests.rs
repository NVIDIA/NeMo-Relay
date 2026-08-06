// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for otel in the NeMo Relay core crate.

use super::*;
use crate::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
    tool_attributes_to_strings,
};
use crate::api::runtime::{
    NemoRelayContextState, PropagationContext, ThreadScopeStackBinding,
    capture_propagation_context, capture_thread_scope_stack, create_scope_stack_from_propagation,
    fork_scope_stack, global_context, restore_thread_scope_stack, set_thread_scope_stack,
};
use crate::api::scope::ScopeType;
use crate::api::scope::{event, pop_scope, push_scope};
use crate::api::tool::ToolAttributes;
use crate::codec::model_pricing::pricing_test_mutex;
use crate::codec::response::{
    AnnotatedLlmResponse, CostEstimate, CostSource, FinishReason, PricingCatalog, PricingResolver,
    Usage, reset_active_pricing_resolver, set_active_pricing_resolver,
};
use crate::json::Json;
use crate::observability::atif::{AtifAgentInfo, AtifExporter, AtifStepExtra};
use crate::observability::{relay_span_id, relay_trace_id};
use opentelemetry::trace::TraceContextExt;
use opentelemetry_sdk::trace::{BatchConfigBuilder, InMemorySpanExporterBuilder};
use serde_json::json;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use uuid::Uuid;

struct ResetPricingResolverGuard;

impl Drop for ResetPricingResolverGuard {
    fn drop(&mut self) {
        let _ = reset_active_pricing_resolver();
    }
}

struct ClearPluginConfigurationGuard;

impl Drop for ClearPluginConfigurationGuard {
    fn drop(&mut self) {
        let _ = crate::plugin::clear_plugin_configuration();
    }
}

struct RestoreThreadScopeStackGuard(ThreadScopeStackBinding);

impl Drop for RestoreThreadScopeStackGuard {
    fn drop(&mut self) {
        restore_thread_scope_stack(self.0.clone());
    }
}

#[derive(Clone, Debug, Default)]
struct BlockingSpanExporter {
    state: Arc<(Mutex<BlockingExporterState>, Condvar)>,
}

#[derive(Debug, Default)]
struct BlockingExporterState {
    export_started: bool,
    release_export: bool,
}

impl BlockingSpanExporter {
    fn wait_until_export_starts(&self) {
        let (state, changed) = &*self.state;
        let guard = state.lock().unwrap();
        let (guard, timeout) = changed
            .wait_timeout_while(guard, Duration::from_secs(5), |state| !state.export_started)
            .unwrap();
        assert!(guard.export_started && !timeout.timed_out());
    }

    fn release(&self) {
        let (state, changed) = &*self.state;
        state.lock().unwrap().release_export = true;
        changed.notify_all();
    }
}

impl Drop for BlockingSpanExporter {
    fn drop(&mut self) {
        self.release();
    }
}

impl SpanExporter for BlockingSpanExporter {
    async fn export(&self, _batch: Vec<SpanData>) -> OTelSdkResult {
        let (state, changed) = &*self.state;
        let mut state = state.lock().unwrap();
        state.export_started = true;
        changed.notify_all();
        while !state.release_export {
            state = changed.wait(state).unwrap();
        }
        Ok(())
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
fn optimization_summary_emits_namespaced_otel_attributes() {
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
        attributes["nemo_relay.llm.optimization.baseline_model"],
        "baseline"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.prompt_tokens_saved"],
        "12"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.estimated_cost_saved"],
        "0.01"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.pricing_as_of"],
        "2026-07-08"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_cost_currency"],
        "USD"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_cost_currency"],
        "USD"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.estimated_cost_saved_currency"],
        "USD"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_pricing_source"],
        "test"
    );
}

#[test]
fn optimization_cost_attributes_preserve_independent_currency_and_provenance() {
    let summary: crate::codec::optimization::LlmOptimizationSummary =
        serde_json::from_value(json!({
            "schema_version":"1", "calculation_version":"1", "status":"partial",
            "limitations":["cost_currency_mismatch"],
            "tokens_saved":{},
            "baseline_cost":{"total":2.0,"currency":"EUR","source":"model_pricing","pricing_as_of":"2026-01-01"},
            "actual_cost":{"total":1.0,"currency":"GBP","source":"provider_reported","pricing_as_of":"2026-02-02","pricing_source":"provider"},
            "contributions":[]
        }))
        .unwrap();
    let mut attributes = Vec::new();
    push_optimization_attributes(&mut attributes, &summary);
    let attributes = attr_map(&attributes);

    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_cost_currency"],
        "EUR"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_cost_currency"],
        "GBP"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_pricing_source"],
        "provider"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_pricing_as_of"],
        "2026-02-02"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.pricing_source"],
        "provider"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.pricing_as_of"],
        "2026-01-01"
    );
    assert!(!attributes.contains_key("nemo_relay.llm.optimization.estimated_cost_saved"));
    assert!(!attributes.contains_key("nemo_relay.llm.optimization.estimated_cost_saved_currency"));
}

#[test]
fn complete_non_usd_optimization_costs_keep_currency_and_independent_provenance() {
    let summary: crate::codec::optimization::LlmOptimizationSummary =
        serde_json::from_value(json!({
            "schema_version":"1", "calculation_version":"1", "status":"complete",
            "tokens_saved":{},
            "baseline_cost":{"total":2.0,"currency":"EUR","source":"model_pricing","pricing_as_of":"2026-01-01","pricing_source":"baseline-catalog"},
            "actual_cost":{"total":1.0,"currency":"EUR","source":"provider_reported","pricing_as_of":"2026-02-02","pricing_source":"provider-meter"},
            "estimated_cost_saved":1.0, "currency":"EUR", "contributions":[]
        }))
        .unwrap();
    let mut attributes = Vec::new();
    push_optimization_attributes(&mut attributes, &summary);
    let attributes = attr_map(&attributes);

    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_cost_currency"],
        "EUR"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_cost_currency"],
        "EUR"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.estimated_cost_saved_currency"],
        "EUR"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_pricing_source"],
        "baseline-catalog"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_pricing_source"],
        "provider-meter"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_pricing_as_of"],
        "2026-01-01"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_pricing_as_of"],
        "2026-02-02"
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

fn install_provider_disambiguation_pricing(model_id: &str) {
    install_disambiguation_pricing(model_id, "test");
}

fn install_openai_disambiguation_pricing(model_id: &str) {
    install_disambiguation_pricing(model_id, "openai");
}

fn install_disambiguation_pricing(model_id: &str, preferred_provider: &str) {
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
                    "provider": preferred_provider,
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

fn finished_span_named<'a>(
    spans: &'a [opentelemetry_sdk::trace::SpanData],
    name: &str,
) -> &'a opentelemetry_sdk::trace::SpanData {
    spans
        .iter()
        .find(|span| span.name.as_ref() == name)
        .unwrap_or_else(|| panic!("missing span {name}"))
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

#[test]
fn propagated_root_parent_projects_as_a_remote_otel_parent() {
    let root_uuid = Uuid::now_v7();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    let imported_stack = create_scope_stack_from_propagation(&PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid: Some(root_uuid),
        parent_uuid: root_uuid,
    })
    .unwrap();
    set_thread_scope_stack(imported_stack);

    let processor = OtelEventProcessor::new(make_provider().0, "test".into());
    let parent_context = processor.parent_context(&make_start_event(
        Uuid::now_v7(),
        Some(root_uuid),
        "receiver-tool",
        ScopeType::Tool,
        None,
    ));
    let parent_span = parent_context.span();
    let span_context = parent_span.span_context();
    assert!(span_context.is_remote());
    assert_eq!(span_context.trace_id(), relay_trace_id(root_uuid));
    assert_eq!(span_context.span_id(), relay_span_id(root_uuid));
}

#[test]
fn rootless_propagation_starts_a_new_otel_trace() {
    let parent_uuid = Uuid::now_v7();
    let local_uuid = Uuid::now_v7();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    let imported_stack = create_scope_stack_from_propagation(&PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid: None,
        parent_uuid,
    })
    .unwrap();
    set_thread_scope_stack(imported_stack);

    let captured = capture_propagation_context().unwrap();
    assert_eq!(captured.root_uuid, None);
    assert_eq!(captured.parent_uuid, parent_uuid);

    let forked_stack = fork_scope_stack().unwrap();
    {
        let forked_stack = forked_stack.read().unwrap();
        assert_eq!(forked_stack.root_uuid(), parent_uuid);
        assert_eq!(forked_stack.top().uuid, parent_uuid);
    }
    set_thread_scope_stack(forked_stack);

    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider, "test".into());
    processor.process(&make_start_event(
        local_uuid,
        Some(parent_uuid),
        "receiver-tool",
        ScopeType::Tool,
        None,
    ));
    processor.process(&make_end_event(
        local_uuid,
        Some(parent_uuid),
        "receiver-tool",
        ScopeType::Tool,
        None,
    ));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let span = &spans[0];
    assert_eq!(span.span_context.trace_id(), relay_trace_id(local_uuid));
    assert_eq!(span.span_context.span_id(), relay_span_id(local_uuid));
    assert_eq!(span.parent_span_id, SpanId::INVALID);
    assert!(!span.parent_span_is_remote);
}

fn make_start_event_with_metadata(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    metadata: Json,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(uuid)
            .name(name)
            .metadata(metadata)
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::agent(),
        None,
    ))
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

fn make_end_event_with_metadata(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    metadata: Json,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(uuid)
            .name(name)
            .metadata(metadata)
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::from(scope_type),
        None,
    ))
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

fn make_mark_event_with_metadata(parent_uuid: Option<Uuid>, metadata: Json) -> Event {
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .name("session.start")
            .metadata(metadata)
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
    let config =
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_service_name("demo-agent")
            .with_header("authorization", "Bearer token")
            .with_resource_attribute("deployment.environment", "test")
            .with_service_namespace("agents")
            .with_service_version("1.2.3")
            .with_instrumentation_scope("demo-scope")
            .with_mark_projection(MarkProjection::Tool)
            .with_mark_exclude_names(["notification"])
            .with_attribute_mapping("nemo_relay.model_name", "model.alias")
            .with_timeout(Duration::from_millis(1250));
    assert_config_builder_overrides(&config);
    assert_config_defaults(&OpenTelemetryConfig::default());
}

fn assert_config_builder_overrides(config: &OpenTelemetryConfig) {
    assert_eq!(config.transport, OtlpTransport::HttpBinary);
    assert_eq!(config.endpoint, "http://localhost:4318/v1/traces");
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
    assert_eq!(config.mark_exclude_names, vec!["notification"]);
    assert_eq!(config.attribute_mappings.len(), 1);
    assert_eq!(config.timeout, Duration::from_millis(1250));
}

fn assert_config_defaults(defaults: &OpenTelemetryConfig) {
    assert_eq!(defaults.transport, OtlpTransport::HttpBinary);
    assert_eq!(defaults.service_name, "unknown_service");
    assert_eq!(defaults.instrumentation_scope, "opentelemetry");
    assert_eq!(defaults.mark_projection, MarkProjection::Inherit);
    assert_eq!(defaults.mark_exclude_names, vec!["llm.chunk"]);
    assert_eq!(defaults.timeout, Duration::from_secs(3));
    assert!(defaults.headers.is_empty());
    assert!(defaults.resource_attributes.is_empty());
}

#[test]
fn http_trace_endpoint_resolution_completes_only_bare_http_urls() {
    for (endpoint, expected) in [
        ("http://localhost:4318", "http://localhost:4318/v1/traces"),
        ("http://localhost:4318/", "http://localhost:4318/"),
        (
            "https://collector.example?tenant=one",
            "https://collector.example/v1/traces?tenant=one",
        ),
        (
            "https://collector.example/?tenant=one",
            "https://collector.example/?tenant=one",
        ),
        (
            "http://localhost:4318/v1/traces",
            "http://localhost:4318/v1/traces",
        ),
        (
            "http://collector.example/custom-ingest",
            "http://collector.example/custom-ingest",
        ),
        ("not a URL", "not a URL"),
    ] {
        assert_eq!(resolve_http_trace_endpoint(endpoint), expected);
    }
}

#[test]
fn grpc_config_owns_its_tokio_runtime() {
    let subscriber = OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4317")
            .with_transport(OtlpTransport::Grpc)
            .with_service_name("demo-agent"),
    )
    .expect("gRPC construction should not require an ambient Tokio runtime");
    subscriber.shutdown().unwrap();
}

#[test]
fn direct_config_rejects_a_blank_endpoint_before_exporter_construction() {
    let error = match OpenTelemetrySubscriber::new(OpenTelemetryConfig::new(
        OpenTelemetryType::GenAi,
        "  ",
    )) {
        Ok(_) => panic!("blank endpoints must be rejected"),
        Err(error) => error,
    };
    assert!(
        matches!(error, OpenTelemetryError::ExporterBuild(message) if message.contains("nonblank"))
    );
}

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
fn direct_config_rejects_invalid_and_case_duplicate_headers() {
    for config in [
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_header("bad header", "value"),
        OpenTelemetryConfig::new(OpenTelemetryType::GenAi, "http://localhost:4318/v1/traces")
            .with_header("x-control", "line one\nline two"),
        OpenTelemetryConfig::new(
            OpenTelemetryType::OpenInference,
            "http://localhost:4318/v1/traces",
        )
        .with_header("Authorization", "first")
        .with_header("authorization", "second"),
    ] {
        let error = match OpenTelemetrySubscriber::new(config) {
            Ok(_) => panic!("invalid direct exporter headers should fail construction"),
            Err(error) => error,
        };
        assert!(matches!(error, OpenTelemetryError::InvalidHeader { .. }));
    }
}

#[test]
fn direct_config_rejects_process_global_otel_headers() {
    const CHILD_MARKER: &str = "NEMO_RELAY_TEST_GLOBAL_OTEL_HEADER_CHILD";
    if let Ok(variable) = std::env::var(CHILD_MARKER) {
        assert!(matches!(
            reject_global_header_environment(),
            Err(OpenTelemetryError::GlobalHeaderEnvironmentUnsupported {
                variable: rejected
            }) if rejected == variable
        ));
        return;
    }

    for variable in [
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    ] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("observability::otel::tests::direct_config_rejects_process_global_otel_headers")
            .env(CHILD_MARKER, variable)
            .env(variable, "authorization=secret")
            .env_remove(if variable == "OTEL_EXPORTER_OTLP_HEADERS" {
                "OTEL_EXPORTER_OTLP_TRACES_HEADERS"
            } else {
                "OTEL_EXPORTER_OTLP_HEADERS"
            })
            .output()
            .unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("test result: ok. 1 passed"),
            "child test filter did not execute exactly one test: {stdout}"
        );
    }
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
fn mapped_aliases_are_typed_and_cannot_replace_projected_span_fields() {
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider_with_options(
        provider,
        "mapping-scope",
        OpenTelemetrySubscriberOptions {
            mark_projection: MarkProjection::Tool,
            mark_exclude_names: vec!["custom.mark".to_string()],
            attribute_mappings: vec![
                crate::observability::OtlpAttributeMapping::new(
                    "nemo_relay.start.data.tenant",
                    "tenant.id",
                ),
                crate::observability::OtlpAttributeMapping::new(
                    "nemo_relay.end.data.tenant",
                    "nemo_relay.start.data.tenant",
                ),
                crate::observability::OtlpAttributeMapping::new(
                    "nemo_relay.start.data.tenant",
                    "nemo_relay.start.data.existing",
                ),
                crate::observability::OtlpAttributeMapping::new("missing.source", "ignored.alias"),
            ],
        },
    )
    .unwrap();
    let callback = subscriber.subscriber();
    let uuid = Uuid::now_v7();
    callback(&make_start_event(
        uuid,
        None,
        "mapped-scope",
        ScopeType::Agent,
        Some(json!({"tenant": 7, "existing": 9})),
    ));
    callback(&make_end_event(
        uuid,
        None,
        "mapped-scope",
        ScopeType::Agent,
        Some(json!({"tenant": 8})),
    ));
    subscriber.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let span = spans
        .iter()
        .find(|span| span.name.as_ref() == "mapped-scope")
        .unwrap();
    assert_eq!(
        span.attributes
            .iter()
            .filter(|attribute| attribute.key.as_str() == "nemo_relay.start.data.tenant")
            .count(),
        1
    );
    assert_eq!(
        span.attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == "tenant.id")
            .map(|attribute| &attribute.value),
        Some(&opentelemetry::Value::I64(7))
    );
    assert_eq!(
        span.attributes
            .iter()
            .filter(|attribute| attribute.key.as_str() == "nemo_relay.start.data.existing")
            .count(),
        1
    );
    assert_eq!(
        span.attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == "nemo_relay.start.data.existing")
            .map(|attribute| &attribute.value),
        Some(&opentelemetry::Value::I64(9))
    );
    assert!(
        !span
            .attributes
            .iter()
            .any(|attribute| attribute.key.as_str() == "ignored.alias")
    );
}

#[test]
fn session_identity_is_projected_on_trace_roots_and_marks_only() {
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "session-identity");
    let callback = subscriber.subscriber();
    let root_uuid = Uuid::now_v7();
    let child_uuid = Uuid::now_v7();
    let second_root_uuid = Uuid::now_v7();
    let instance_id = crate::api::runtime::current_scope_stack()
        .read()
        .unwrap()
        .root_uuid()
        .to_string();
    let identity = json!({
        "session_id": "logical-session",
        "user_id": "alice",
        "agent_kind": "claude-code"
    });

    callback(&make_start_event_with_metadata(
        root_uuid,
        None,
        "identity-root",
        identity.clone(),
    ));
    callback(&make_start_event_with_metadata(
        child_uuid,
        Some(root_uuid),
        "identity-child",
        identity.clone(),
    ));
    callback(&make_mark_event_with_metadata(
        Some(root_uuid),
        identity.clone(),
    ));
    callback(&make_end_event(
        child_uuid,
        Some(root_uuid),
        "identity-child",
        ScopeType::Agent,
        None,
    ));
    callback(&make_end_event(
        root_uuid,
        None,
        "identity-root",
        ScopeType::Agent,
        None,
    ));
    callback(&make_start_event_with_metadata(
        second_root_uuid,
        None,
        "identity-second-root",
        json!({"session_id": "logical-session", "user_id": 42}),
    ));
    callback(&make_end_event(
        second_root_uuid,
        None,
        "identity-second-root",
        ScopeType::Agent,
        None,
    ));
    callback(&make_mark_event_with_metadata(None, identity));
    subscriber.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_session_root_and_child_identity(&spans, &instance_id);
    assert_session_mark_identity(&spans);
}

fn assert_session_root_and_child_identity(
    spans: &[opentelemetry_sdk::trace::SpanData],
    instance_id: &str,
) {
    let root = finished_span_named(spans, "identity-root");
    let child = finished_span_named(spans, "identity-child");
    let second_root = finished_span_named(spans, "identity-second-root");

    let root_attributes = attr_map(&root.attributes);
    assert_eq!(root_attributes["session.id"], "logical-session");
    assert_eq!(root_attributes["user.id"], "alice");
    assert_eq!(root_attributes["nemo_relay.agent.kind"], "claude-code");
    assert_eq!(
        root_attributes["nemo_relay.session.instance_id"],
        instance_id
    );
    assert_eq!(
        root.attributes
            .iter()
            .filter(|attribute| attribute.key.as_str() == "session.id")
            .count(),
        1
    );
    let child_attributes = attr_map(&child.attributes);
    assert!(!child_attributes.contains_key("session.id"));
    assert!(!child_attributes.contains_key("user.id"));
    assert!(!child_attributes.contains_key("nemo_relay.session.instance_id"));
    assert!(!child_attributes.contains_key("nemo_relay.agent.kind"));
    assert_eq!(
        child_attributes["nemo_relay.start.metadata.user_id"],
        "alice"
    );
    let second_root_attributes = attr_map(&second_root.attributes);
    assert_eq!(second_root_attributes["session.id"], "logical-session");
    assert!(!second_root_attributes.contains_key("user.id"));
    assert_eq!(
        second_root_attributes["nemo_relay.start.metadata.user_id"],
        "42"
    );
    assert_eq!(
        second_root_attributes["nemo_relay.session.instance_id"],
        root_attributes["nemo_relay.session.instance_id"]
    );
    assert_ne!(
        root.span_context.trace_id(),
        second_root.span_context.trace_id()
    );
}

fn assert_session_mark_identity(spans: &[opentelemetry_sdk::trace::SpanData]) {
    let root = finished_span_named(spans, "identity-root");
    let orphan_mark = finished_span_named(spans, "mark:session.start");
    let root_attributes = attr_map(&root.attributes);
    let mark_attributes = attr_map(&root.events.events[0].attributes);
    assert_eq!(mark_attributes["session.id"], "logical-session");
    assert_eq!(mark_attributes["user.id"], "alice");
    assert_eq!(mark_attributes["nemo_relay.agent.kind"], "claude-code");
    assert_eq!(
        mark_attributes["nemo_relay.session.instance_id"],
        root_attributes["nemo_relay.session.instance_id"]
    );
    let orphan_attributes = attr_map(&orphan_mark.attributes);
    assert_eq!(orphan_attributes["session.id"], "logical-session");
    assert_eq!(orphan_attributes["user.id"], "alice");
    assert_eq!(orphan_attributes["nemo_relay.agent.kind"], "claude-code");
    assert_eq!(
        orphan_attributes["nemo_relay.session.instance_id"],
        root_attributes["nemo_relay.session.instance_id"]
    );
}

#[test]
fn mapped_orphan_mark_alias_cannot_replace_intrinsic_mark_fields() {
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider_with_attribute_mappings(
        provider,
        "mapping-scope",
        [crate::observability::OtlpAttributeMapping::new(
            "nemo_relay.mark.data.value",
            "nemo_relay.mark.orphan",
        )],
    )
    .unwrap();
    let callback = subscriber.subscriber();
    callback(&make_mark_event(
        None,
        "mapped-orphan-mark",
        Some(json!({"value": "not-an-orphan-flag"})),
    ));
    subscriber.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let span = spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:mapped-orphan-mark")
        .unwrap();
    assert_eq!(
        span.attributes
            .iter()
            .filter(|attribute| attribute.key.as_str() == "nemo_relay.mark.orphan")
            .count(),
        1
    );
    assert_eq!(
        span.attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == "nemo_relay.mark.orphan")
            .map(|attribute| &attribute.value),
        Some(&opentelemetry::Value::Bool(true))
    );
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
        attributes.get("nemo_relay.start.input.task"),
        Some(&"scope-start".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.start.metadata.phase"),
        Some(&"start".to_string())
    );

    let event_attributes = attr_map(&span.events.events[0].attributes);
    assert_eq!(
        event_attributes.get("nemo_relay.mark.data.step"),
        Some(&"1".to_string())
    );
    assert_eq!(
        event_attributes.get("nemo_relay.mark.metadata.source"),
        Some(&"rust-test".to_string())
    );
}

#[test]
fn gen_ai_projection_is_fixed_and_preserves_ancestry_through_omitted_scopes() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings(
        provider,
        "gen-ai-test".to_string(),
        OpenTelemetryType::GenAi,
        MarkProjection::Tool,
        Vec::new(),
        Vec::new(),
    );
    assert!(
        processor
            .mark_attributes(&make_mark_event(None, "omitted-mark", None))
            .is_empty()
    );
    let agent_uuid = Uuid::now_v7();
    let reranker_uuid = Uuid::now_v7();
    let tool_uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        agent_uuid,
        None,
        "research-agent",
        ScopeType::Agent,
        Some(json!({"secret": "must-not-export"})),
    ));
    processor.process(&make_start_event(
        reranker_uuid,
        Some(agent_uuid),
        "rerank",
        ScopeType::Reranker,
        None,
    ));
    processor.process(&make_mark_event(
        Some(agent_uuid),
        "checkpoint",
        Some(json!({"secret": "must-not-export"})),
    ));
    processor.process(&make_start_event(
        tool_uuid,
        Some(reranker_uuid),
        "web-search",
        ScopeType::Tool,
        Some(json!({"query": "must-not-export"})),
    ));
    processor.process(&make_end_event(
        tool_uuid,
        Some(reranker_uuid),
        "web-search",
        ScopeType::Tool,
        Some(json!({"result": "must-not-export"})),
    ));
    processor.process(&make_end_event(
        reranker_uuid,
        Some(agent_uuid),
        "rerank",
        ScopeType::Reranker,
        None,
    ));
    processor.process(&make_end_event(
        agent_uuid,
        None,
        "research-agent",
        ScopeType::Agent,
        None,
    ));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 2);
    let agent = spans
        .iter()
        .find(|span| span.name.as_ref() == "invoke_agent research-agent")
        .unwrap();
    let tool = spans
        .iter()
        .find(|span| span.name.as_ref() == "execute_tool web-search")
        .unwrap();
    assert_eq!(agent.span_kind, SpanKind::Internal);
    assert_eq!(tool.span_kind, SpanKind::Internal);
    assert_eq!(tool.parent_span_id, agent.span_context.span_id());
    assert!(agent.events.events.is_empty());
    assert!(spans.iter().all(|span| {
        span.attributes.iter().all(|attribute| {
            !attribute.key.as_str().starts_with("nemo_relay.")
                && !attribute.value.to_string().contains("must-not-export")
        })
    }));
    let agent_attributes = attr_map(&agent.attributes);
    let tool_attributes = attr_map(&tool.attributes);
    assert_eq!(
        agent_attributes.get("gen_ai.operation.name"),
        Some(&"invoke_agent".to_string())
    );
    assert_eq!(
        tool_attributes.get("gen_ai.operation.name"),
        Some(&"execute_tool".to_string())
    );
}

#[test]
fn gen_ai_projection_uses_standard_operation_names_and_span_kinds() {
    for (scope_type, name, expected_name, expected_kind) in [
        (
            ScopeType::Agent,
            "planner",
            "invoke_agent planner",
            SpanKind::Internal,
        ),
        (ScopeType::Llm, "chat", "chat", SpanKind::Client),
        (
            ScopeType::Llm,
            "generate_content",
            "generate_content",
            SpanKind::Client,
        ),
        (
            ScopeType::Llm,
            "text_completion",
            "text_completion",
            SpanKind::Client,
        ),
        (
            ScopeType::Tool,
            "search",
            "execute_tool search",
            SpanKind::Internal,
        ),
        (ScopeType::Embedder, "embed", "embeddings", SpanKind::Client),
        (
            ScopeType::Retriever,
            "retrieve",
            "retrieval",
            SpanKind::Client,
        ),
    ] {
        let event = make_start_event(Uuid::now_v7(), None, name, scope_type, None);
        assert!(crate::observability::otel_genai::supports(&event));
        assert_eq!(
            crate::observability::otel_genai::span_name(&event),
            expected_name
        );
        assert_eq!(
            crate::observability::otel_genai::span_kind(&event),
            expected_kind
        );
    }

    for unsupported in [ScopeType::Reranker, ScopeType::Guardrail] {
        let event = make_start_event(Uuid::now_v7(), None, "unsupported", unsupported, None);
        assert!(!crate::observability::otel_genai::supports(&event));
    }
}

#[test]
fn gen_ai_projection_emits_only_span_specific_attributes() {
    let common = json!({
        "provider": "openai",
        "conversation_id": "conversation-1",
        "server_address": "api.example.test",
        "server_port": 443,
        "model": "model-1",
        "agent_description": "helpful",
        "tool_type": "function",
        "tool_call_id": "call-1",
        "tool_description": "searches",
        "data_source_id": "docs",
        "top_k": 5
    });
    let cases = [
        (
            ScopeType::Agent,
            "agent",
            [
                "gen_ai.agent.description",
                "gen_ai.agent.name",
                "gen_ai.conversation.id",
                "gen_ai.operation.name",
                "gen_ai.request.model",
            ]
            .as_slice(),
        ),
        (
            ScopeType::Llm,
            "chat",
            [
                "gen_ai.conversation.id",
                "gen_ai.operation.name",
                "gen_ai.provider.name",
                "gen_ai.request.model",
                "server.address",
                "server.port",
            ]
            .as_slice(),
        ),
        (
            ScopeType::Tool,
            "search",
            [
                "gen_ai.operation.name",
                "gen_ai.tool.call.id",
                "gen_ai.tool.description",
                "gen_ai.tool.name",
                "gen_ai.tool.type",
            ]
            .as_slice(),
        ),
        (
            ScopeType::Embedder,
            "embed",
            [
                "gen_ai.operation.name",
                "gen_ai.provider.name",
                "gen_ai.request.model",
                "server.address",
                "server.port",
            ]
            .as_slice(),
        ),
        (
            ScopeType::Retriever,
            "retrieve",
            [
                "gen_ai.data_source.id",
                "gen_ai.operation.name",
                "gen_ai.provider.name",
                "gen_ai.request.model",
                "gen_ai.retrieval.top_k",
                "server.address",
                "server.port",
            ]
            .as_slice(),
        ),
    ];
    for (scope_type, name, expected) in cases {
        let event = make_start_event(Uuid::now_v7(), None, name, scope_type, Some(common.clone()));
        let attributes = crate::observability::otel_genai::start_attributes(&event);
        let actual = attributes
            .iter()
            .map(|attribute| attribute.key.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(actual, expected.iter().copied().collect());
        assert!(attributes.iter().all(|attribute| {
            attribute.key.as_str() != "gen_ai.conversation.id"
                || matches!(scope_type, ScopeType::Agent | ScopeType::Llm)
        }));
        if scope_type == ScopeType::Retriever {
            let top_k = attributes
                .iter()
                .find(|attribute| attribute.key.as_str() == "gen_ai.retrieval.top_k")
                .unwrap();
            assert_eq!(top_k.value, opentelemetry::Value::I64(5));
        }
    }

    let embed_end = make_end_event(
        Uuid::now_v7(),
        None,
        "embed",
        ScopeType::Embedder,
        Some(json!({"input_tokens": 7, "output_tokens": 11})),
    );
    let embed_attributes = attr_map(&crate::observability::otel_genai::end_attributes(
        &embed_end,
    ));
    assert_eq!(
        embed_attributes.get("gen_ai.usage.input_tokens"),
        Some(&"7".to_string())
    );
    assert!(!embed_attributes.contains_key("gen_ai.usage.output_tokens"));
}

#[test]
fn gen_ai_projection_emits_normalized_response_attributes() {
    let event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"answer": "ok"})),
        Some(
            CategoryProfile::builder()
                .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                    id: Some("response-1".to_string()),
                    model: Some("model-1".to_string()),
                    finish_reason: Some(FinishReason::ToolUse),
                    usage: Some(Usage {
                        prompt_tokens: Some(13),
                        completion_tokens: Some(8),
                        total_tokens: Some(21),
                        cache_read_tokens: Some(5),
                        cache_write_tokens: Some(3),
                        cost: None,
                    }),
                    ..empty_annotated_response()
                }))
                .build(),
        ),
    );

    let attributes = attr_map(&crate::observability::otel_genai::end_attributes(&event));
    assert_eq!(
        attributes.get("gen_ai.response.id"),
        Some(&"response-1".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.response.model"),
        Some(&"model-1".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.response.finish_reasons"),
        Some(&"[\"tool_calls\"]".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.usage.input_tokens"),
        Some(&"13".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.usage.output_tokens"),
        Some(&"8".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.usage.cache_read.input_tokens"),
        Some(&"5".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.usage.cache_creation.input_tokens"),
        Some(&"3".to_string())
    );
}

#[test]
fn gen_ai_end_projection_preserves_explicit_error_type() {
    let event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name("chat")
            .metadata(json!({
                "otel.status_code": "ERROR",
                "otel.status_description": "invalid argument: invalid value",
                "error.type": "invalid_argument",
            }))
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::from(ScopeType::Llm),
        None,
    ));

    let attributes = attr_map(&crate::observability::otel_genai::end_attributes(&event));
    assert_eq!(
        attributes.get("error.type"),
        Some(&"invalid_argument".to_string())
    );
}

#[test]
fn failed_descendant_classification_and_exception_propagate_to_agent_span() {
    for otel_type in [OpenTelemetryType::Full, OpenTelemetryType::GenAi] {
        let (provider, exporter) = make_provider();
        let mut processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings(
                provider,
                "error-propagation-test".to_string(),
                otel_type,
                MarkProjection::default(),
                default_mark_exclude_names(),
                Vec::new(),
            );
        let agent_uuid = Uuid::now_v7();
        let function_uuid = Uuid::now_v7();
        let llm_uuid = Uuid::now_v7();
        processor.process(&make_start_event(
            agent_uuid,
            None,
            "agent",
            ScopeType::Agent,
            None,
        ));
        let llm_parent_uuid = if otel_type == OpenTelemetryType::GenAi {
            processor.process(&make_start_event(
                function_uuid,
                Some(agent_uuid),
                "function",
                ScopeType::Function,
                None,
            ));
            function_uuid
        } else {
            agent_uuid
        };
        processor.process(&make_start_event(
            llm_uuid,
            Some(llm_parent_uuid),
            "chat",
            ScopeType::Llm,
            None,
        ));
        processor.process(&make_end_event_with_metadata(
            llm_uuid,
            Some(llm_parent_uuid),
            "chat",
            ScopeType::Llm,
            json!({
                "otel.status_code": "ERROR",
                "otel.status_description": "internal error: ValueError: boom",
                "error.type": "internal_error",
                "exception.type": "ValueError",
            }),
        ));
        if otel_type == OpenTelemetryType::GenAi {
            processor.process(&make_end_event_with_metadata(
                function_uuid,
                Some(agent_uuid),
                "function",
                ScopeType::Function,
                json!({
                    "otel.status_code": "ERROR",
                    "otel.status_description": "internal error: ValueError: boom",
                }),
            ));
        }
        processor.process(&make_end_event_with_metadata(
            agent_uuid,
            None,
            "agent",
            ScopeType::Agent,
            json!({
                "otel.status_code": "ERROR",
                "otel.status_description": "internal error: ValueError: boom",
            }),
        ));
        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 2);
        for span in &spans {
            assert_eq!(
                attr_map(&span.attributes).get("error.type"),
                Some(&"internal_error".to_string())
            );
            let exception = span
                .events
                .events
                .iter()
                .find(|event| event.name.as_ref() == "exception")
                .expect("expected exception event");
            assert_eq!(
                attr_map(&exception.attributes).get("exception.type"),
                Some(&"ValueError".to_string())
            );
        }
    }
}

#[test]
fn suppressed_function_error_propagates_to_agent_span() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings(
        provider,
        "suppressed-error-propagation-test".to_string(),
        OpenTelemetryType::GenAi,
        MarkProjection::default(),
        default_mark_exclude_names(),
        Vec::new(),
    );
    let agent_uuid = Uuid::now_v7();
    let function_uuid = Uuid::now_v7();
    processor.process(&make_start_event(
        agent_uuid,
        None,
        "agent",
        ScopeType::Agent,
        None,
    ));
    processor.process(&make_start_event(
        function_uuid,
        Some(agent_uuid),
        "function",
        ScopeType::Function,
        None,
    ));
    processor.process(&make_end_event_with_metadata(
        function_uuid,
        Some(agent_uuid),
        "function",
        ScopeType::Function,
        json!({
            "otel.status_code": "ERROR",
            "error.type": "internal_error",
            "exception.type": "ValueError",
        }),
    ));
    processor.process(&make_end_event_with_metadata(
        agent_uuid,
        None,
        "agent",
        ScopeType::Agent,
        json!({"otel.status_code": "ERROR"}),
    ));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let agent_span = &spans[0];
    assert_eq!(
        attr_map(&agent_span.attributes).get("error.type"),
        Some(&"internal_error".to_string())
    );
    let exception = agent_span
        .events
        .events
        .iter()
        .find(|event| event.name.as_ref() == "exception")
        .expect("expected propagated exception event");
    assert_eq!(
        attr_map(&exception.attributes).get("exception.type"),
        Some(&"ValueError".to_string())
    );
}

#[test]
fn suppressed_parent_error_propagation_isolated_by_trace_id() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings(
        provider,
        "error-trace-isolation-test".to_string(),
        OpenTelemetryType::GenAi,
        MarkProjection::default(),
        default_mark_exclude_names(),
        Vec::new(),
    );
    let shared_span_id = [0xAB; 8];
    let mut first_agent_bytes = [0x11; 16];
    first_agent_bytes[8..].copy_from_slice(&shared_span_id);
    let mut second_agent_bytes = [0x22; 16];
    second_agent_bytes[8..].copy_from_slice(&shared_span_id);
    let cases = [
        (
            Uuid::from_bytes(first_agent_bytes),
            Uuid::now_v7(),
            Uuid::now_v7(),
            "first_error",
            "FirstException",
        ),
        (
            Uuid::from_bytes(second_agent_bytes),
            Uuid::now_v7(),
            Uuid::now_v7(),
            "second_error",
            "SecondException",
        ),
    ];
    assert_eq!(
        relay_span_id(cases[0].0),
        relay_span_id(cases[1].0),
        "fixture agents must share a span ID"
    );
    assert_ne!(
        relay_trace_id(cases[0].0),
        relay_trace_id(cases[1].0),
        "fixture agents must belong to different traces"
    );

    for (agent_uuid, function_uuid, llm_uuid, _, _) in cases {
        processor.process(&make_start_event(
            agent_uuid,
            None,
            "agent",
            ScopeType::Agent,
            None,
        ));
        processor.process(&make_start_event(
            function_uuid,
            Some(agent_uuid),
            "function",
            ScopeType::Function,
            None,
        ));
        processor.process(&make_start_event(
            llm_uuid,
            Some(function_uuid),
            "chat",
            ScopeType::Llm,
            None,
        ));
    }
    for (agent_uuid, function_uuid, llm_uuid, error_type, exception_type) in cases {
        processor.process(&make_end_event_with_metadata(
            llm_uuid,
            Some(function_uuid),
            "chat",
            ScopeType::Llm,
            json!({
                "otel.status_code": "ERROR",
                "error.type": error_type,
                "exception.type": exception_type,
            }),
        ));
        processor.process(&make_end_event_with_metadata(
            function_uuid,
            Some(agent_uuid),
            "function",
            ScopeType::Function,
            json!({"otel.status_code": "ERROR"}),
        ));
    }
    for (agent_uuid, _, _, _, _) in cases {
        processor.process(&make_end_event_with_metadata(
            agent_uuid,
            None,
            "agent",
            ScopeType::Agent,
            json!({"otel.status_code": "ERROR"}),
        ));
    }
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 4);
    for (agent_uuid, _, _, error_type, exception_type) in cases {
        let agent_span = spans
            .iter()
            .find(|span| {
                span.span_context.trace_id() == relay_trace_id(agent_uuid)
                    && span.parent_span_id == SpanId::INVALID
            })
            .expect("expected agent span for trace");
        assert_eq!(
            attr_map(&agent_span.attributes).get("error.type"),
            Some(&error_type.to_string())
        );
        let exception = agent_span
            .events
            .events
            .iter()
            .find(|event| event.name.as_ref() == "exception")
            .expect("expected propagated exception event");
        assert_eq!(
            attr_map(&exception.attributes).get("exception.type"),
            Some(&exception_type.to_string())
        );
    }
}

#[test]
fn gen_ai_projection_prefers_standard_names_and_normalized_provider_details() {
    let agent = make_start_event(
        Uuid::now_v7(),
        None,
        "fallback-agent",
        ScopeType::Agent,
        Some(json!({"gen_ai.agent.name": "semantic-agent"})),
    );
    assert_eq!(
        crate::observability::otel_genai::span_name(&agent),
        "invoke_agent semantic-agent"
    );

    let tool = make_start_event(
        Uuid::now_v7(),
        None,
        "fallback-tool",
        ScopeType::Tool,
        Some(json!({"gen_ai.tool.name": "semantic-tool"})),
    );
    assert_eq!(
        crate::observability::otel_genai::span_name(&tool),
        "execute_tool semantic-tool"
    );

    let llm = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name("openai.chat")
            .data(json!({
                "headers": {},
                "content": {
                    "model": "gpt-5",
                    "messages": [{"role": "user", "content": "hello"}],
                    "frequency_penalty": 0.25,
                    "n": 2,
                    "presence_penalty": -0.5,
                    "seed": 42,
                    "stream": true
                }
            }))
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::llm(),
        None,
    ));
    let attributes = attr_map(&crate::observability::otel_genai::start_attributes(&llm));
    assert_eq!(
        attributes.get("gen_ai.provider.name"),
        Some(&"openai".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.request.stream"),
        Some(&"true".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.request.frequency_penalty"),
        Some(&"0.25".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.request.choice.count"),
        Some(&"2".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.request.presence_penalty"),
        Some(&"-0.5".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.request.seed"),
        Some(&"42".to_string())
    );

    let anthropic = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name("anthropic.messages")
            .data(json!({
                "headers": {},
                "content": {
                    "model": "claude-sonnet",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hello"}],
                    "top_k": 12
                }
            }))
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::llm(),
        None,
    ));
    let attributes = attr_map(&crate::observability::otel_genai::start_attributes(
        &anthropic,
    ));
    assert_eq!(
        attributes.get("gen_ai.request.top_k"),
        Some(&"12".to_string())
    );

    for (name, expected) in [
        ("azure_openai.chat", "azure.ai.openai"),
        ("azure_ai_inference.chat", "azure.ai.inference"),
    ] {
        let event = Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .uuid(Uuid::now_v7())
                .name(name)
                .data(json!({
                    "headers": {},
                    "content": {
                        "model": "gpt-5",
                        "messages": [{"role": "user", "content": "hello"}]
                    }
                }))
                .build(),
            ScopeCategory::Start,
            Vec::new(),
            EventCategory::llm(),
            None,
        ));
        let attributes = attr_map(&crate::observability::otel_genai::start_attributes(&event));
        assert_eq!(
            attributes.get("gen_ai.provider.name"),
            Some(&expected.to_string())
        );
    }
}

#[test]
fn gen_ai_projection_covers_optional_request_controls_and_finish_reasons() {
    let request = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name("chat")
            .data(json!({
                "headers": {},
                "content": {
                    "model": "gpt-5",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 512,
                    "temperature": 0.4,
                    "top_p": 0.8,
                    "stop": ["done"],
                    "n": 1
                }
            }))
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::llm(),
        None,
    ));
    let attributes = attr_map(&crate::observability::otel_genai::start_attributes(
        &request,
    ));
    for (key, expected) in [
        ("gen_ai.provider.name", "openai"),
        ("gen_ai.request.max_tokens", "512"),
        ("gen_ai.request.temperature", "0.4"),
        ("gen_ai.request.top_p", "0.8"),
        ("gen_ai.request.stop_sequences", "[\"done\"]"),
    ] {
        assert_eq!(attributes.get(key), Some(&expected.to_string()));
    }
    assert!(!attributes.contains_key("gen_ai.request.choice.count"));

    for (reason, expected) in [
        (FinishReason::Complete, "stop"),
        (FinishReason::Length, "length"),
        (FinishReason::ContentFilter, "content_filter"),
        (
            FinishReason::Unknown("provider_reason".to_string()),
            "provider_reason",
        ),
    ] {
        let event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
            None,
            "chat",
            ScopeType::Llm,
            None,
            Some(
                CategoryProfile::builder()
                    .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                        finish_reason: Some(reason),
                        ..empty_annotated_response()
                    }))
                    .build(),
            ),
        );
        let attributes = attr_map(&crate::observability::otel_genai::end_attributes(&event));
        assert_eq!(
            attributes.get("gen_ai.response.finish_reasons"),
            Some(&format!("[\"{expected}\"]"))
        );
    }
}

#[test]
fn gen_ai_projection_reads_nested_scalar_fallbacks() {
    let event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name("embed")
            .metadata(json!({
                "request": {"provider": true, "server_port": 8443},
                "response": {"response_model": 17},
                "usage": {"input_tokens": u64::MAX}
            }))
            .data(json!({"usage": {"prompt_tokens": 23}}))
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::from(ScopeType::Embedder),
        None,
    ));
    let attributes = attr_map(&crate::observability::otel_genai::end_attributes(&event));
    assert_eq!(
        attributes.get("gen_ai.response.model"),
        Some(&"17".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.usage.input_tokens"),
        Some(&"23".to_string())
    );
}

#[test]
fn http_config_exports_scope_push_pop_and_marks_without_tokio_runtime() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_global();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
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
fn direct_gen_ai_and_openinference_configs_export_typed_otlp_payloads() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_global();

    for (otel_type, expected_key) in [
        (
            OpenTelemetryType::GenAi,
            b"gen_ai.operation.name".as_slice(),
        ),
        (
            OpenTelemetryType::OpenInference,
            b"openinference.span.kind".as_slice(),
        ),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1/traces", listener.local_addr().unwrap());
        let (request_tx, request_rx) = mpsc::channel();
        spawn_http_collector(listener, request_tx);
        let subscriber = OpenTelemetrySubscriber::new(
            OpenTelemetryConfig::new(otel_type, endpoint)
                .with_service_name("typed-direct-service")
                .with_instrumentation_scope("typed-direct-scope"),
        )
        .unwrap();
        let callback = subscriber.subscriber();
        let uuid = Uuid::now_v7();
        callback(&make_start_event(
            uuid,
            None,
            "typed-agent",
            ScopeType::Agent,
            None,
        ));
        callback(&make_end_event(
            uuid,
            None,
            "typed-agent",
            ScopeType::Agent,
            None,
        ));
        subscriber.force_flush().unwrap();

        let request = request_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("typed direct config should export an OTLP request");
        assert_eq!(request.path, "/v1/traces");
        assert_eq!(request.content_type, "application/x-protobuf");
        assert!(
            request
                .body
                .windows(expected_key.len())
                .any(|window| window == expected_key),
            "typed projection key should be encoded in the OTLP payload"
        );
        for expected_metadata in [
            b"typed-direct-service".as_slice(),
            b"typed-direct-scope".as_slice(),
        ] {
            assert!(
                request
                    .body
                    .windows(expected_metadata.len())
                    .any(|window| window == expected_metadata)
            );
        }
        subscriber.shutdown().unwrap();
    }
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
        attributes.get("nemo_relay.start.input.query"),
        Some(&"hello".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.end.output.result"),
        Some(&"ok".to_string())
    );
}

#[test]
fn derives_span_ids_from_relay_uuids() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection(
        provider.clone(),
        "test-scope".to_string(),
        MarkProjection::Tool,
    );

    let root_uuid = Uuid::from_u128(0x018f_0f0f_0f0f_7000_8123_4567_89ab_cdef);
    let child_uuid = Uuid::from_u128(0x018f_0f0f_0f10_7000_8fed_cba9_8765_4321);
    let orphan_uuid = Uuid::from_u128(0x018f_0f0f_0f11_7000_8011_2233_4455_6677);

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
    processor.process(&Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .uuid(orphan_uuid)
            .name("checkpoint")
            .build(),
        None,
        None,
    )));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 3);
    let parent = spans
        .iter()
        .find(|span| span.name.as_ref() == "agent")
        .unwrap();
    let child = spans
        .iter()
        .find(|span| span.name.as_ref() == "model-call")
        .unwrap();
    let orphan = spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:checkpoint")
        .unwrap();

    assert_eq!(
        parent.span_context.trace_id().to_bytes(),
        *root_uuid.as_bytes()
    );
    assert_eq!(
        parent.span_context.span_id().to_bytes(),
        root_uuid.as_bytes()[8..]
    );
    assert_eq!(
        child.span_context.trace_id(),
        parent.span_context.trace_id()
    );
    assert_eq!(
        child.span_context.span_id().to_bytes(),
        child_uuid.as_bytes()[8..]
    );
    assert_eq!(child.parent_span_id, parent.span_context.span_id());
    assert!(!child.parent_span_is_remote);
    assert_eq!(
        orphan.span_context.trace_id().to_bytes(),
        *orphan_uuid.as_bytes()
    );
    assert_eq!(
        orphan.span_context.span_id().to_bytes(),
        orphan_uuid.as_bytes()[8..]
    );
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
fn tool_projection_emits_generic_mark_as_parented_zero_duration_span() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection(
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
    assert!(parent.start_time <= projected.start_time);
    assert!(projected.end_time <= parent.end_time);

    let attributes = attr_map(&projected.attributes);
    assert_eq!(
        attributes.get("nemo_relay.mark.projection"),
        Some(&"tool".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.scope_type"),
        Some(&"tool".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.mark.data.count"),
        Some(&"3".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.mark.category"),
        Some(&"custom".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.mark.category_profile.subtype"),
        Some(&"example.compaction".to_string())
    );
    assert!(!attributes.contains_key("nemo_relay.mark.orphan"));
}

#[test]
fn tool_projection_exclusion_keeps_custom_mark_as_native_event() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection_and_exclusions(
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
        attributes.get("nemo_relay.mark.orphan"),
        Some(&"true".to_string())
    );
}

#[test]
fn tool_projection_reuses_completed_parent_context_for_late_marks() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection(
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
    assert_eq!(
        attr_map(&projected.attributes).get("nemo_relay.mark.orphan"),
        Some(&"true".to_string())
    );
}

#[test]
fn tool_projection_keeps_llm_chunks_as_parent_span_events() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection(
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
    let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());
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
}

#[test]
fn process_start_removes_completed_span_order_entry() {
    let (provider, _exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider, "test-scope".to_string());
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
    let remote_tool_attributes = attr_map(&start_attributes(&remote_tool));
    assert_eq!(
        remote_tool_attributes.get("nemo_relay.handle_attributes"),
        Some(&"[\"remote\"]".to_string())
    );
    assert!(!remote_tool_attributes.contains_key("nemo_relay.handle_attributes_json"));

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
fn llm_end_with_unannotated_openai_response_uses_codec_cost() {
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
        attributes.get("nemo_relay.llm.cost.total"),
        Some(&"0.000435".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.llm.cost.currency"),
        Some(&"USD".to_string())
    );
}

#[test]
fn llm_end_emits_cost_only_no_token_or_gen_ai_attributes() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    install_openai_disambiguation_pricing("priced-model");
    let _reset_guard = ResetPricingResolverGuard;

    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&make_start_event(uuid, None, "other", ScopeType::Llm, None));
    processor.process(&make_end_event(
        uuid,
        None,
        "other",
        ScopeType::Llm,
        Some(openai_chat_provider_response("priced-model")),
    ));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let keys: Vec<String> = spans[0]
        .attributes
        .iter()
        .map(|kv| kv.key.as_str().to_string())
        .collect();

    assert!(keys.iter().any(|k| k == "nemo_relay.llm.cost.total"));
    assert!(keys.iter().any(|k| k == "nemo_relay.llm.cost.currency"));
    assert!(
        keys.iter()
            .all(|k| !k.starts_with("llm.token") && !k.starts_with("gen_ai")),
        "no token attributes expected on the LLM span: {keys:?}"
    );
}

#[test]
fn llm_end_with_unpriced_response_model_uses_requested_model_cost() {
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
        attributes.get("nemo_relay.llm.cost.total"),
        Some(&"0.000435".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.llm.cost.currency"),
        Some(&"USD".to_string())
    );
}

#[test]
fn llm_end_with_unannotated_openai_response_without_usage_omits_cost() {
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
    assert!(!attributes.contains_key("nemo_relay.llm.cost.total"));
    assert!(!attributes.contains_key("nemo_relay.llm.cost.currency"));
}

#[test]
fn helper_functions_cover_additional_otel_branches() {
    assert_otel_scope_and_model_attribute_branches();
    assert_otel_tool_attribute_branches();
    assert_otel_catalog_cost_branches();
    assert_otel_normalized_cost_branches();
    assert_otel_manual_cost_branches();
}

fn assert_otel_scope_and_model_attribute_branches() {
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
    let raw_model_event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"model": "raw-model", "answer": "ok"})),
        None,
    );
    let raw_model_attributes = attr_map(&common_attributes(&raw_model_event));
    assert_eq!(
        raw_model_attributes.get("nemo_relay.model_name"),
        Some(&"raw-model".to_string())
    );
    let response_model_event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({
            "id": "chatcmpl-response-model",
            "model": "response-model",
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }]
        })),
        Some(
            CategoryProfile::builder()
                .model_name("requested-model")
                .build(),
        ),
    );
    let response_model_attributes = attr_map(&common_attributes(&response_model_event));
    assert_eq!(
        response_model_attributes.get("nemo_relay.model_name"),
        Some(&"response-model".to_string())
    );
}

fn assert_otel_tool_attribute_branches() {
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
        start_attributes.get("nemo_relay.start.data.query"),
        Some(&"hello".to_string())
    );
    assert_eq!(
        start_attributes.get("nemo_relay.start.metadata.meta"),
        Some(&"true".to_string())
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
        tool_end_attributes.get("nemo_relay.end.data.result"),
        Some(&"true".to_string())
    );
}

fn assert_otel_catalog_cost_branches() {
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
}

fn assert_otel_normalized_cost_branches() {
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
}

fn assert_otel_manual_cost_branches() {
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
        mark_attributes.get("nemo_relay.mark.data.kind"),
        Some(&"aux".to_string())
    );
    assert_eq!(
        mark_attributes.get("nemo_relay.mark.metadata.source"),
        Some(&"unit".to_string())
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
        &OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_service_name("demo-agent")
            .with_header("authorization", "Bearer token")
            .with_resource_attribute("deployment.environment", "test")
            .with_service_namespace("agents")
            .with_service_version("1.2.3")
            .with_max_queue_size(16)
            .with_max_export_batch_size(4)
            .with_scheduled_delay(Duration::from_millis(10)),
        None,
    )
    .unwrap();
    http_provider.force_flush().unwrap();
    http_provider.shutdown().unwrap();

    let subscriber = OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_service_name("http-success"),
    )
    .unwrap();
    subscriber.force_flush().unwrap();
    subscriber.shutdown().unwrap();
}

#[test]
fn dropped_spans_are_recorded_in_the_active_plugin_report() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let _ = crate::plugin::clear_plugin_configuration();
    let _clear_guard = ClearPluginConfigurationGuard;
    futures::executor::block_on(crate::plugin::initialize_plugins_exact(
        crate::plugin::PluginConfig::default(),
    ))
    .unwrap();

    let exporter = BlockingSpanExporter::default();
    let processor = DiagnosticBatchSpanProcessor::new_with_batch_config(
        exporter.clone(),
        "https://collector.example/v1/traces".to_string(),
        Some("opentelemetry.endpoints[2].endpoint".to_string()),
        BatchConfigBuilder::default()
            .with_max_queue_size(1)
            .with_max_export_batch_size(1)
            .with_scheduled_delay(Duration::from_secs(60))
            .build(),
    );
    let provider = SdkTracerProvider::builder()
        .with_span_processor(processor)
        .build();
    let tracer = provider.tracer("dropped-span-diagnostic-test");

    tracer.start("export-in-progress").end();
    exporter.wait_until_export_starts();
    tracer.start("queued").end();
    tracer.start("dropped-1").end();
    tracer.start("dropped-2").end();
    exporter.release();
    let shutdown = provider.shutdown().unwrap_err();
    assert!(
        shutdown
            .to_string()
            .contains(OTEL_RUNTIME_DELIVERY_FAILURE_MARKER)
    );

    let report = crate::plugin::active_plugin_report().unwrap();
    let diagnostic = report
        .runtime_diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "otel.spans_dropped")
        .unwrap();
    assert_eq!(diagnostic.count, 2);
    assert_eq!(
        diagnostic.field.as_deref(),
        Some("opentelemetry.endpoints[2].endpoint")
    );
    assert!(
        diagnostic
            .message
            .contains("https://collector.example/v1/traces")
    );
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
            &OpenTelemetryConfig::grpc("grpc-demo")
                .with_endpoint("http://127.0.0.1:4317")
                .with_header("authorization", "Bearer token"),
            None,
        )
        .unwrap();
        provider.force_flush().ok();
        provider.shutdown().ok();
    });
}
