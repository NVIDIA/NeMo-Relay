// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for otel in the NeMo Relay core crate.

use super::*;
use crate::api::event::{
    BaseEvent, CategoryProfile, DataSchema, Event, EventCategory, METRIC_DATA_SCHEMA_NAME,
    METRIC_DATA_SCHEMA_VERSION, MarkEvent, ScopeCategory, ScopeEvent, tool_attributes_to_strings,
};
use crate::api::runtime::{
    NemoRelayContextState, PropagationContext, ThreadScopeStackBinding,
    capture_propagation_context, capture_rootless_propagation_context, capture_thread_scope_stack,
    create_scope_stack_from_propagation, fork_scope_stack, global_context,
    restore_thread_scope_stack, set_thread_scope_stack,
};
use crate::api::scope::ScopeType;
use crate::api::scope::{event, pop_scope, push_scope};
use crate::api::tool::ToolAttributes;
use crate::codec::model_pricing::pricing_test_mutex;
use crate::codec::request::{AnnotatedLlmRequest, MessageContent};
use crate::codec::response::{
    AnnotatedLlmResponse, CostEstimate, CostSource, FinishReason, PricingCatalog, PricingResolver,
    ResponseToolCall, Usage, reset_active_pricing_resolver, set_active_pricing_resolver,
};
use crate::json::Json;
use crate::observability::atif::{AtifAgentInfo, AtifExporter, AtifStepExtra};
use crate::observability::otel_logs::{OpenTelemetryLogConfig, OpenTelemetryLogSubscriber};
use crate::observability::otel_metrics::{
    OpenTelemetryMetricConfig, OpenTelemetryMetricSubscriber,
};
use crate::observability::{relay_span_id, relay_trace_id};
use opentelemetry::trace::TraceContextExt;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use opentelemetry_proto::tonic::common::v1::{KeyValue as OtlpKeyValue, any_value};
use opentelemetry_sdk::trace::{BatchConfigBuilder, InMemorySpanExporterBuilder};
use prost::Message;
use serde_json::json;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::AtomicUsize;
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

#[test]
fn provider_errors_identify_their_telemetry_signal() {
    for (error, expected) in [
        (
            OpenTelemetryError::TraceProvider("trace failure".to_string()),
            "OpenTelemetry tracer provider error: trace failure",
        ),
        (
            OpenTelemetryError::LogProvider("log failure".to_string()),
            "OpenTelemetry log provider error: log failure",
        ),
        (
            OpenTelemetryError::MetricProvider("metric failure".to_string()),
            "OpenTelemetry metric provider error: metric failure",
        ),
    ] {
        assert_eq!(error.to_string(), expected);
    }
}

#[test]
fn shutdown_is_idempotent_for_all_otlp_subscribers() {
    let _guard = crate::observability::test_mutex().lock().unwrap();

    let trace = OpenTelemetrySubscriber::from_tracer_provider(
        SdkTracerProvider::builder().build(),
        "shutdown-idempotency-test",
    );
    trace.shutdown().unwrap();
    trace.shutdown().unwrap();

    let logs = OpenTelemetryLogSubscriber::new(OpenTelemetryLogConfig::new(
        "http://127.0.0.1:4318/v1/logs",
    ))
    .unwrap();
    logs.shutdown().unwrap();
    logs.shutdown().unwrap();
    assert!(logs.force_flush().is_err());

    let metrics = OpenTelemetryMetricSubscriber::new(OpenTelemetryMetricConfig::new(
        "http://127.0.0.1:4318/v1/metrics",
    ))
    .unwrap();
    metrics.shutdown().unwrap();
    metrics.shutdown().unwrap();
    assert!(metrics.force_flush().is_err());
}

#[test]
fn shutdown_normalization_preserves_provider_failures() {
    let _guard = crate::observability::test_mutex().lock().unwrap();

    assert!(normalize_shutdown_result(Err(OTelSdkError::AlreadyShutdown)).is_ok());

    let error = normalize_shutdown_result(Err(OTelSdkError::InternalFailure(
        "collector unavailable".to_string(),
    )))
    .unwrap_err();
    assert!(matches!(
        error,
        OTelSdkError::InternalFailure(message) if message == "collector unavailable"
    ));

    let processor = DiagnosticBatchSpanProcessor::new_with_batch_config(
        AlwaysFailingSpanExporter,
        "https://collector.example/v1/traces".to_string(),
        SignalRuntimeDiagnostics::new(Some("opentelemetry.traces[0].endpoint".to_string())),
        BatchConfigBuilder::default()
            .with_max_export_batch_size(1)
            .build(),
    );
    let provider = SdkTracerProvider::builder()
        .with_span_processor(processor)
        .build();
    provider
        .tracer("first-shutdown-failure-test")
        .start("export-fails")
        .end();
    provider.force_flush().unwrap_err();
    let subscriber =
        OpenTelemetrySubscriber::from_tracer_provider(provider, "first-shutdown-failure-test");

    let error = subscriber.shutdown().unwrap_err();
    assert!(matches!(
        error,
        OpenTelemetryError::TraceProvider(message)
            if message.contains(OTEL_RUNTIME_DELIVERY_FAILURE_MARKER)
    ));
    subscriber.shutdown().unwrap();
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

#[test]
fn slow_trace_flush_does_not_block_other_subscribers_or_lifecycle_barriers() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_global();

    let exporter = BlockingSpanExporter::default();
    let processor = DiagnosticBatchSpanProcessor::new_with_batch_config(
        exporter.clone(),
        "https://collector.example/v1/traces".to_string(),
        SignalRuntimeDiagnostics::new(None),
        BatchConfigBuilder::default()
            .with_max_queue_size(1)
            .with_max_export_batch_size(1)
            .with_scheduled_delay(Duration::from_secs(60))
            .build(),
    );
    let provider = SdkTracerProvider::builder()
        .with_span_processor(processor)
        .build();
    let trace_subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "trace");
    let trace_name = format!("slow-trace-{}", Uuid::now_v7().simple());
    trace_subscriber.register(&trace_name).unwrap();

    let (healthy_tx, healthy_rx) = mpsc::channel();
    let healthy_name = format!("healthy-{}", Uuid::now_v7().simple());
    crate::api::subscriber::register_subscriber(
        &healthy_name,
        Arc::new(move |event| {
            if event.name() == "healthy-event" {
                healthy_tx.send(()).unwrap();
            }
        }),
    )
    .unwrap();

    let trace_callback = trace_subscriber.subscriber();
    let trace_uuid = Uuid::now_v7();
    trace_callback(&make_start_event(
        trace_uuid,
        None,
        "slow-trace",
        ScopeType::Agent,
        None,
    ));
    trace_callback(&make_end_event(
        trace_uuid,
        None,
        "slow-trace",
        ScopeType::Agent,
        None,
    ));

    let flush_trace = trace_subscriber.clone();
    let trace_flush = thread::spawn(move || flush_trace.force_flush());
    exporter.wait_until_export_starts();

    crate::api::scope::event(
        crate::api::scope::EmitMarkEventParams::builder()
            .name("healthy-event")
            .build(),
    )
    .unwrap();

    let (barrier_tx, barrier_rx) = mpsc::channel();
    let barrier = thread::spawn(move || {
        barrier_tx
            .send(crate::api::subscriber::flush_subscribers())
            .unwrap();
    });

    healthy_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("a slow trace flush must not delay another subscriber");
    barrier_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("a slow trace flush must not delay the subscriber lifecycle barrier")
        .unwrap();

    exporter.release();
    trace_flush.join().unwrap().unwrap();
    barrier.join().unwrap();
    trace_subscriber.deregister(&trace_name).unwrap();
    crate::api::subscriber::deregister_subscriber(&healthy_name).unwrap();
    trace_subscriber.shutdown().unwrap();
}

#[derive(Clone, Debug, Default)]
struct FailingThenRecoveringSpanExporter {
    attempts: Arc<AtomicUsize>,
}

impl SpanExporter for FailingThenRecoveringSpanExporter {
    async fn export(&self, _batch: Vec<SpanData>) -> OTelSdkResult {
        if self.attempts.fetch_add(1, Ordering::Relaxed) == 0 {
            Err(OTelSdkError::InternalFailure(
                "collector unavailable".to_string(),
            ))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug, Default)]
struct SensitiveFailingSpanExporter;

impl SpanExporter for SensitiveFailingSpanExporter {
    async fn export(&self, _batch: Vec<SpanData>) -> OTelSdkResult {
        Err(OTelSdkError::InternalFailure(
            "Authorization: Bearer exporter-secret".to_string(),
        ))
    }
}

#[derive(Clone, Debug)]
struct AlwaysFailingSpanExporter;

impl SpanExporter for AlwaysFailingSpanExporter {
    async fn export(&self, _batch: Vec<SpanData>) -> OTelSdkResult {
        Err(OTelSdkError::InternalFailure(
            "collector unavailable".to_string(),
        ))
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
    let parent_uuid = Uuid::now_v7();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    let imported_stack = create_scope_stack_from_propagation(&PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid: Some(root_uuid),
        parent_uuid,
    })
    .unwrap();
    set_thread_scope_stack(imported_stack);

    let processor = OtelEventProcessor::new(make_provider().0, "test".into());
    let mut event = make_start_event(
        Uuid::now_v7(),
        Some(parent_uuid),
        "receiver-tool",
        ScopeType::Tool,
        None,
    );
    event.set_propagation_root_uuid(Some(root_uuid));
    let parent_context = processor.parent_context(&event);
    let parent_span = parent_context.span();
    let span_context = parent_span.span_context();
    assert!(span_context.is_remote());
    assert_eq!(span_context.trace_id(), relay_trace_id(root_uuid));
    assert_eq!(span_context.span_id(), relay_span_id(parent_uuid));
}

#[test]
fn rootless_propagation_remains_rootless_when_forked() {
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

    let captured = capture_rootless_propagation_context().unwrap();
    assert_eq!(captured.root_uuid, None);
    assert_eq!(captured.parent_uuid, parent_uuid);

    let forked_stack = fork_scope_stack().unwrap();
    {
        let forked_stack = forked_stack.read().unwrap();
        assert_eq!(forked_stack.root_uuid(), parent_uuid);
        assert_eq!(forked_stack.top().uuid, parent_uuid);
    }
    let propagation_root_uuid = forked_stack.read().unwrap().event_propagation_root_uuid();
    set_thread_scope_stack(forked_stack.clone());
    let forked_context = capture_propagation_context().unwrap();
    assert_eq!(forked_context.root_uuid, None);

    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider, "test".into());
    let mut start = make_start_event(
        local_uuid,
        Some(parent_uuid),
        "receiver-tool",
        ScopeType::Tool,
        None,
    );
    start.set_propagation_root_uuid(propagation_root_uuid);
    assert_eq!(start.propagation_root_uuid(), None);
    processor.process(&start);
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

#[test]
fn default_propagation_context_preserves_the_imported_root() {
    let root_uuid = Uuid::now_v7();
    let parent_uuid = Uuid::now_v7();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    let imported_stack = create_scope_stack_from_propagation(&PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid: Some(root_uuid),
        parent_uuid,
    })
    .unwrap();
    set_thread_scope_stack(imported_stack);

    let captured = capture_propagation_context().unwrap();
    assert_eq!(captured.root_uuid, Some(root_uuid));
    assert_eq!(captured.parent_uuid, parent_uuid);
}

#[test]
fn promotes_final_scope_and_mark_metadata_without_duplicate_span_keys() {
    for otel_type in [OpenTelemetryType::Full, OpenTelemetryType::OpenInference] {
        let (provider, exporter) = make_provider();
        let mut processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
                provider,
                "metadata-promotion-test".into(),
                otel_type,
                MarkProjection::default(),
                default_mark_exclude_names(),
                Vec::new(),
                vec!["nv.".to_string(), "nemo_relay.".to_string()],
                SignalRuntimeDiagnostics::new(None),
            );
        let uuid = Uuid::now_v7();
        processor.process(&make_start_event_with_metadata(
            uuid,
            None,
            "metadata-promotion-scope",
            json!({
                "nv.source": "start",
                "nemo_relay.scope_type": "attempted-overwrite"
            }),
        ));
        processor.process(&make_mark_event_with_metadata(
            Some(uuid),
            json!({"nv.source": "mark"}),
        ));
        processor.process(&make_end_event_with_metadata(
            uuid,
            None,
            "metadata-promotion-scope",
            ScopeType::Agent,
            json!({
                "nv.source": "end",
                "nv.completed": true,
                "nemo_relay.scope_type": "attempted-overwrite"
            }),
        ));
        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(
            span.attributes
                .iter()
                .filter(|attribute| attribute.key.as_str() == "nv.source")
                .count(),
            1
        );
        let attributes = attr_map(&span.attributes);
        assert_eq!(attributes.get("nv.source"), Some(&"end".to_string()));
        assert_eq!(attributes.get("nv.completed"), Some(&"true".to_string()));
        assert_eq!(
            attributes.get("nemo_relay.scope_type"),
            Some(&"agent".to_string())
        );
        assert!(
            !attributes
                .keys()
                .any(|key| key.ends_with(".metadata.nv.source"))
        );
        let mark_attributes = attr_map(&span.events.events[0].attributes);
        assert_eq!(mark_attributes.get("nv.source"), Some(&"mark".to_string()));
        assert!(
            !mark_attributes
                .keys()
                .any(|key| key.ends_with(".metadata.nv.source"))
        );
    }
}

#[test]
fn removes_resource_promoted_metadata_from_all_otlp_event_projections() {
    for otel_type in [OpenTelemetryType::Full, OpenTelemetryType::OpenInference] {
        let (provider, exporter) = make_provider();
        let mut processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
                provider,
                "resource-metadata-filter-test".into(),
                otel_type,
                MarkProjection::default(),
                default_mark_exclude_names(),
                Vec::new(),
                Vec::new(),
                SignalRuntimeDiagnostics::new(None),
        );
        processor.resource_metadata_prefixes = vec!["tenant.".to_string()];
        processor.resource_metadata_protected_keys = HashSet::from(["tenant.region".to_string()]);
        let uuid = Uuid::now_v7();
        processor.process(&make_start_event_with_metadata(
            uuid,
            None,
            "resource-metadata-filter-scope",
            json!({"tenant.id": "root", "tenant.region": "metadata", "keep": "start"}),
        ));
        processor.process(&make_mark_event_with_metadata(
            Some(uuid),
            json!({"tenant.id": "mark", "tenant.region": "metadata", "keep": "mark"}),
        ));
        processor.process(&make_end_event_with_metadata(
            uuid,
            None,
            "resource-metadata-filter-scope",
            ScopeType::Agent,
            json!({"tenant.id": "end", "tenant.region": "metadata", "keep": "end"}),
        ));
        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        let span = finished_span_named(&spans, "resource-metadata-filter-scope");
        let serialized_attributes = span
            .attributes
            .iter()
            .chain(
                span.events
                    .events
                    .iter()
                    .flat_map(|event| event.attributes.iter()),
            )
            .map(|attribute| format!("{}={}", attribute.key, attribute.value))
            .collect::<Vec<_>>();
        assert!(
            serialized_attributes
                .iter()
                .any(|attribute| attribute.contains("keep"))
        );
        assert!(
            serialized_attributes
                .iter()
                .any(|attribute| attribute.contains("tenant.region"))
        );
        assert!(serialized_attributes.iter().all(
            |attribute| !attribute.contains("tenant.id") && !attribute.contains("tenant\":\"")
        ));
    }
}

#[test]
fn promotes_orphan_and_tool_projection_mark_metadata() {
    for otel_type in [OpenTelemetryType::Full, OpenTelemetryType::OpenInference] {
        let (provider, exporter) = make_provider();
        let mut processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
                provider,
                "metadata-promotion-mark-test".into(),
                otel_type,
                MarkProjection::Tool,
                default_mark_exclude_names(),
                Vec::new(),
                vec!["nv.".to_string()],
                SignalRuntimeDiagnostics::new(None),
            );
        let parent_uuid = Uuid::now_v7();
        processor.process(&Event::Mark(MarkEvent::new(
            BaseEvent::builder()
                .name("metadata.orphan")
                .metadata(json!({"nv.source": "orphan"}))
                .build(),
            None,
            None,
        )));
        processor.process(&make_start_event(
            parent_uuid,
            None,
            "metadata-promotion-parent",
            ScopeType::Agent,
            None,
        ));
        processor.process(&Event::Mark(MarkEvent::new(
            BaseEvent::builder()
                .parent_uuid(parent_uuid)
                .name("metadata.projected")
                .metadata(json!({"nv.source": "projected"}))
                .build(),
            None,
            None,
        )));
        processor.process(&make_end_event(
            parent_uuid,
            None,
            "metadata-promotion-parent",
            ScopeType::Agent,
            None,
        ));
        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 3);
        let parent = finished_span_named(&spans, "metadata-promotion-parent");
        let orphan = finished_span_named(&spans, "mark:metadata.orphan");
        let projected = finished_span_named(&spans, "mark:metadata.projected");
        assert!(!attr_map(&parent.attributes).contains_key("nv.source"));
        assert_eq!(
            attr_map(&orphan.attributes).get("nv.source"),
            Some(&"orphan".to_string())
        );
        assert_eq!(
            attr_map(&projected.attributes).get("nv.source"),
            Some(&"projected".to_string())
        );
        assert_eq!(projected.parent_span_id, parent.span_context.span_id());
    }
}

#[test]
fn promotes_final_scope_metadata_across_trace_projections() {
    for otel_type in [
        OpenTelemetryType::Full,
        OpenTelemetryType::GenAi,
        OpenTelemetryType::OpenInference,
    ] {
        let (provider, exporter) = make_provider();
        let mut processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
                provider,
                "metadata-promotion-projection-test".into(),
                otel_type,
                MarkProjection::default(),
                default_mark_exclude_names(),
                Vec::new(),
                vec!["nv.".to_string()],
                SignalRuntimeDiagnostics::new(None),
            );
        let uuid = Uuid::now_v7();
        processor.process(&make_start_event_with_metadata(
            uuid,
            None,
            "metadata-promotion-projection-scope",
            json!({"nv.source": "start"}),
        ));
        processor.process(&make_end_event_with_metadata(
            uuid,
            None,
            "metadata-promotion-projection-scope",
            ScopeType::Agent,
            json!({"nv.source": "end"}),
        ));
        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(
            attr_map(&spans[0].attributes).get("nv.source"),
            Some(&"end".to_string())
        );
    }
}

#[test]
fn omits_scope_metadata_when_final_value_is_unsupported_across_trace_projections() {
    for otel_type in [
        OpenTelemetryType::Full,
        OpenTelemetryType::GenAi,
        OpenTelemetryType::OpenInference,
    ] {
        let (provider, exporter) = make_provider();
        let runtime_diagnostics = SignalRuntimeDiagnostics::new(None);
        let mut processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
                provider,
                "unsupported-final-metadata-promotion-test".into(),
                otel_type,
                MarkProjection::default(),
                default_mark_exclude_names(),
                Vec::new(),
                vec!["nv.".to_string()],
                runtime_diagnostics.clone(),
            );
        let uuid = Uuid::now_v7();
        processor.process(&make_start_event_with_metadata(
            uuid,
            None,
            "unsupported-final-metadata-promotion-scope",
            json!({"nv.source": "start"}),
        ));
        processor.process(&make_end_event_with_metadata(
            uuid,
            None,
            "unsupported-final-metadata-promotion-scope",
            ScopeType::Agent,
            json!({"nv.source": {"unsupported": true}}),
        ));
        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert!(!attr_map(&spans[0].attributes).contains_key("nv.source"));

        let diagnostics = runtime_diagnostics.snapshot();
        let diagnostic = diagnostics
            .get("otel.metadata_promotion_value_unsupported.nv.source")
            .expect("unsupported final metadata diagnostic");
        assert_eq!(diagnostic.count, 1);
        assert!(diagnostic.message.contains("nv.source"));
    }
}

#[test]
fn reports_each_unsupported_metadata_key_with_a_deterministic_diagnostic_code() {
    let (provider, exporter) = make_provider();
    let runtime_diagnostics = SignalRuntimeDiagnostics::new(None);
    let mut processor =
        OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
            provider,
            "unsupported-metadata-promotion-keys-test".into(),
            OpenTelemetryType::Full,
            MarkProjection::default(),
            default_mark_exclude_names(),
            Vec::new(),
            vec!["tenant.".to_string()],
            runtime_diagnostics.clone(),
        );
    let uuid = Uuid::now_v7();
    let unsupported_metadata = json!({
        "tenant.plan": {"name": "enterprise"},
        "tenant.flags": null,
        "tenant.tags": [["nested"]],
        "tenant.mixed": [1, "string"],
    });

    processor.process(&make_start_event_with_metadata(
        uuid,
        None,
        "unsupported-metadata-promotion-keys-scope",
        unsupported_metadata.clone(),
    ));
    processor.process(&make_end_event_with_metadata(
        uuid,
        None,
        "unsupported-metadata-promotion-keys-scope",
        ScopeType::Agent,
        unsupported_metadata,
    ));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let span_attributes = attr_map(&spans[0].attributes);
    for key in ["tenant.flags", "tenant.mixed", "tenant.plan", "tenant.tags"] {
        assert!(!span_attributes.contains_key(key));
    }

    let diagnostics = runtime_diagnostics.snapshot();
    let expected_keys = ["tenant.flags", "tenant.mixed", "tenant.plan", "tenant.tags"];
    assert_eq!(
        diagnostics
            .entries()
            .iter()
            .map(|diagnostic| diagnostic.code.as_str())
            .collect::<Vec<_>>(),
        expected_keys
            .iter()
            .map(|key| format!("otel.metadata_promotion_value_unsupported.{key}"))
            .collect::<Vec<_>>()
    );
    for key in expected_keys {
        let code = format!("otel.metadata_promotion_value_unsupported.{key}");
        let diagnostic = diagnostics
            .get(&code)
            .expect("metadata-key-specific promotion diagnostic");
        assert_eq!(diagnostic.count, 2);
        assert!(diagnostic.message.contains(key));
    }
}

#[test]
fn promotes_start_only_scope_metadata_across_trace_projections() {
    for otel_type in [
        OpenTelemetryType::Full,
        OpenTelemetryType::GenAi,
        OpenTelemetryType::OpenInference,
    ] {
        let (provider, exporter) = make_provider();
        let mut processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
                provider,
                "start-metadata-promotion-projection-test".into(),
                otel_type,
                MarkProjection::default(),
                default_mark_exclude_names(),
                Vec::new(),
                vec!["nv.".to_string()],
                SignalRuntimeDiagnostics::new(None),
            );
        let uuid = Uuid::now_v7();
        processor.process(&make_start_event_with_metadata(
            uuid,
            None,
            "start-metadata-promotion-projection-scope",
            json!({"nv.start_only": "configured"}),
        ));
        processor.process(&make_end_event_with_metadata(
            uuid,
            None,
            "start-metadata-promotion-projection-scope",
            ScopeType::Agent,
            json!({}),
        ));
        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(
            attr_map(&spans[0].attributes).get("nv.start_only"),
            Some(&"configured".to_string())
        );
    }
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
    authorization: Option<String>,
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
        authorization: header_value(&headers_text, "authorization"),
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
            .with_header_env("x-api-key", "NEMO_RELAY_TEST_API_KEY")
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
        config.header_env.get("x-api-key"),
        Some(&"NEMO_RELAY_TEST_API_KEY".into())
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
fn http_trace_endpoint_resolution_preserves_an_explicit_root_path() {
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
            "https://collector.example/#root",
            "https://collector.example/#root",
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
fn trace_subscriber_constructors_reject_zero_completed_context_ttl() {
    let config = OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://127.0.0.1:4318")
        .with_completed_span_context_ttl(std::time::Duration::ZERO);
    let error = match OpenTelemetrySubscriber::new(config) {
        Ok(_) => panic!("zero TTL must be rejected"),
        Err(error) => error,
    };
    assert!(
        matches!(error, OpenTelemetryError::ExporterBuild(message) if message.contains("completed_span_context_ttl must be greater than 0"))
    );

    for typed in [false, true] {
        let (provider, _exporter) = make_provider();
        let options = OpenTelemetrySubscriberOptions {
            completed_span_context_ttl: std::time::Duration::ZERO,
            ..Default::default()
        };
        let result = if typed {
            OpenTelemetrySubscriber::from_tracer_provider_with_type_and_options(
                provider,
                "test-scope",
                OpenTelemetryType::OpenInference,
                options,
            )
        } else {
            OpenTelemetrySubscriber::from_tracer_provider_with_options(
                provider,
                "test-scope",
                options,
            )
        };
        let error = match result {
            Ok(_) => panic!("zero TTL must be rejected"),
            Err(error) => error,
        };
        assert!(
            matches!(error, OpenTelemetryError::ExporterBuild(message) if message.contains("completed_span_context_ttl must be greater than 0"))
        );
    }
}

#[test]
fn subscriber_options_apply_custom_completed_context_ttl_at_the_boundary() {
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider_with_options(
        provider,
        "test-scope",
        OpenTelemetrySubscriberOptions {
            completed_span_context_ttl: std::time::Duration::from_millis(10),
            ..Default::default()
        },
    )
    .unwrap();
    let callback = subscriber.subscriber();
    let uuid = Uuid::now_v7();
    let closed_at = chrono::Utc::now();
    let scope_event = |category| {
        Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .uuid(uuid)
                .name("completed")
                .timestamp(closed_at)
                .build(),
            category,
            Vec::new(),
            EventCategory::from(ScopeType::Tool),
            None,
        ))
    };
    let mark_event = |name, timestamp| {
        Event::Mark(MarkEvent::new(
            BaseEvent::builder()
                .parent_uuid(uuid)
                .name(name)
                .timestamp(timestamp)
                .build(),
            None,
            None,
        ))
    };

    callback(&scope_event(ScopeCategory::Start));
    callback(&scope_event(ScopeCategory::End));
    callback(&mark_event(
        "at-boundary",
        closed_at + chrono::Duration::milliseconds(10),
    ));
    callback(&mark_event(
        "after-boundary",
        closed_at + chrono::Duration::milliseconds(11),
    ));
    subscriber.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let parent = finished_span_named(&spans, "completed");
    let boundary_mark = finished_span_named(&spans, "mark:at-boundary");
    let expired_mark = finished_span_named(&spans, "mark:after-boundary");
    assert_eq!(
        boundary_mark.span_context.trace_id(),
        parent.span_context.trace_id()
    );
    assert_eq!(boundary_mark.parent_span_id, parent.span_context.span_id());
    assert_ne!(
        expired_mark.span_context.trace_id(),
        parent.span_context.trace_id()
    );
    assert_ne!(expired_mark.parent_span_id, parent.span_context.span_id());
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
fn direct_config_rejects_invalid_header_env_without_exposing_values() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let missing = format!("NEMO_RELAY_TEST_MISSING_HEADER_{}", Uuid::now_v7().simple());

    for (variable, value, expected) in [
        (missing.as_str(), None, "is not set"),
        ("NEMO_RELAY_TEST_BLANK_HEADER", Some("  "), "nonblank value"),
        (
            "NEMO_RELAY_TEST_SECRET_HEADER",
            Some("relay-secret\ninvalid"),
            "valid header value",
        ),
    ] {
        if let Some(value) = value {
            unsafe { std::env::set_var(variable, value) };
        }
        let error = match OpenTelemetrySubscriber::new(
            OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
                .with_header_env("authorization", variable),
        ) {
            Ok(_) => panic!("invalid header_env should fail activation"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(message.contains(expected), "unexpected error: {message}");
        assert!(!message.contains("relay-secret"));
        unsafe { std::env::remove_var(variable) };
    }

    for variable in ["", " padded ", "INVALID=NAME", "INVALID\0NAME"] {
        let error = match OpenTelemetrySubscriber::new(
            OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
                .with_header_env("authorization", variable),
        ) {
            Ok(_) => panic!("invalid environment variable reference should fail activation"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("header_env must name"));
    }

    let error = match OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_header("Authorization", "static")
            .with_header_env("authorization", &missing),
    ) {
        Ok(_) => panic!("case-insensitive duplicate should fail before environment lookup"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("unique across headers and header_env")
    );

    unsafe { std::env::set_var(&missing, "valid") };
    let exact_error = match OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_header("authorization", "static")
            .with_header_env("authorization", &missing),
    ) {
        Ok(_) => panic!("exact duplicate should fail activation"),
        Err(error) => error,
    };
    assert!(
        exact_error
            .to_string()
            .contains("unique across headers and header_env")
    );
    unsafe { std::env::remove_var(&missing) };
}

#[test]
fn direct_config_rejects_case_duplicate_header_env_names() {
    let missing = format!("NEMO_RELAY_TEST_MISSING_HEADER_{}", Uuid::now_v7().simple());
    let error = match OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_header_env("Authorization", &missing)
            .with_header_env("authorization", &missing),
    ) {
        Ok(_) => panic!("case-insensitive header_env duplicate should fail activation"),
        Err(error) => error,
    };

    assert!(
        error
            .to_string()
            .contains("unique across headers and header_env")
    );
}

#[cfg(unix)]
#[test]
fn direct_config_non_unicode_header_env_errors_do_not_expose_values() {
    use std::os::unix::ffi::OsStringExt;

    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let variable = format!(
        "NEMO_RELAY_TEST_NON_UNICODE_DIRECT_HEADER_{}",
        Uuid::now_v7().simple()
    );
    let secret = "relay-direct-secret";
    let mut value = vec![0xff];
    value.extend_from_slice(secret.as_bytes());
    unsafe { std::env::set_var(&variable, std::ffi::OsString::from_vec(value)) };

    let error = match OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_header_env("authorization", &variable),
    ) {
        Ok(_) => panic!("non-Unicode header environment value should fail activation"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(message.contains(&variable));
    assert!(!message.contains(secret));

    unsafe { std::env::remove_var(variable) };
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
            promote_metadata_prefixes: Vec::new(),
            completed_span_context_ttl: DEFAULT_COMPLETED_SPAN_CONTEXT_TTL,
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

fn with_propagation_root(mut event: Event, root_uuid: Uuid) -> Event {
    event.set_propagation_root_uuid(Some(root_uuid));
    event
}

#[test]
fn session_identity_is_projected_on_trace_roots_and_marks_only() {
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "session-identity");
    let callback = subscriber.subscriber();
    let root_uuid = Uuid::now_v7();
    let child_uuid = Uuid::now_v7();
    let second_root_uuid = Uuid::now_v7();
    let instance_id = root_uuid.to_string();
    let second_instance_id = second_root_uuid.to_string();
    let identity = json!({
        "session_id": "logical-session",
        "user_id": "alice",
        "agent_kind": "claude-code"
    });

    callback(&with_propagation_root(
        make_start_event_with_metadata(root_uuid, None, "identity-root", identity.clone()),
        root_uuid,
    ));
    callback(&with_propagation_root(
        make_start_event_with_metadata(
            child_uuid,
            Some(root_uuid),
            "identity-child",
            identity.clone(),
        ),
        root_uuid,
    ));
    callback(&with_propagation_root(
        make_mark_event_with_metadata(Some(root_uuid), identity.clone()),
        root_uuid,
    ));
    callback(&with_propagation_root(
        make_end_event(
            child_uuid,
            Some(root_uuid),
            "identity-child",
            ScopeType::Agent,
            None,
        ),
        root_uuid,
    ));
    callback(&with_propagation_root(
        make_end_event(root_uuid, None, "identity-root", ScopeType::Agent, None),
        root_uuid,
    ));
    callback(&with_propagation_root(
        make_start_event_with_metadata(
            second_root_uuid,
            None,
            "identity-second-root",
            json!({"session_id": "logical-session", "user_id": 42}),
        ),
        second_root_uuid,
    ));
    callback(&with_propagation_root(
        make_end_event(
            second_root_uuid,
            None,
            "identity-second-root",
            ScopeType::Agent,
            None,
        ),
        second_root_uuid,
    ));
    callback(&with_propagation_root(
        make_mark_event_with_metadata(None, identity),
        root_uuid,
    ));
    subscriber.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_session_root_and_child_identity(&spans, &instance_id, &second_instance_id);
    assert_session_mark_identity(&spans);
}

fn assert_session_root_and_child_identity(
    spans: &[opentelemetry_sdk::trace::SpanData],
    instance_id: &str,
    second_instance_id: &str,
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
        second_instance_id
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
fn gen_ai_projection_is_fixed_and_preserves_all_scope_parentage() {
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
    assert_eq!(spans.len(), 3);
    let agent = spans
        .iter()
        .find(|span| span.name.as_ref() == "invoke_agent research-agent")
        .unwrap();
    let tool = spans
        .iter()
        .find(|span| span.name.as_ref() == "execute_tool web-search")
        .unwrap();
    let reranker = spans
        .iter()
        .find(|span| span.name.as_ref() == "rerank")
        .unwrap();
    assert_eq!(agent.span_kind, SpanKind::Internal);
    assert_eq!(reranker.span_kind, SpanKind::Internal);
    assert_eq!(tool.span_kind, SpanKind::Internal);
    assert_eq!(reranker.parent_span_id, agent.span_context.span_id());
    assert_eq!(tool.parent_span_id, reranker.span_context.span_id());
    assert!(agent.events.events.is_empty());
    assert!(spans.iter().all(|span| {
        span.attributes.iter().all(|attribute| {
            !attribute.key.as_str().starts_with("nemo_relay.")
                && !attribute.value.to_string().contains("must-not-export")
        })
    }));
    let agent_attributes = attr_map(&agent.attributes);
    assert!(reranker.attributes.is_empty());
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
        assert_eq!(
            crate::observability::otel_genai::span_name(&event),
            expected_name
        );
        assert_eq!(
            crate::observability::otel_genai::span_kind(&event),
            expected_kind
        );
    }

    for generic in [
        ScopeType::Function,
        ScopeType::Reranker,
        ScopeType::Guardrail,
        ScopeType::Evaluator,
        ScopeType::Custom,
        ScopeType::Unknown,
    ] {
        let event = make_start_event(Uuid::now_v7(), None, "generic", generic, None);
        assert_eq!(
            crate::observability::otel_genai::span_name(&event),
            "generic"
        );
        assert_eq!(
            crate::observability::otel_genai::span_kind(&event),
            SpanKind::Internal
        );
        assert!(crate::observability::otel_genai::start_attributes(&event).is_empty());
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
        Some(&"[\"tool_call\"]".to_string())
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
fn gen_ai_projection_includes_anthropic_cache_tokens_in_input_total() {
    let event = make_end_event(
        Uuid::now_v7(),
        None,
        "anthropic.messages",
        ScopeType::Llm,
        Some(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 2,
                "output_tokens": 1,
                "cache_read_input_tokens": 17_980,
                "cache_creation_input_tokens": 9_421
            }
        })),
    );

    let attributes = attr_map(&crate::observability::otel_genai::end_attributes(&event));
    assert_eq!(
        attributes.get("gen_ai.usage.input_tokens"),
        Some(&"27403".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.usage.cache_read.input_tokens"),
        Some(&"17980".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.usage.cache_creation.input_tokens"),
        Some(&"9421".to_string())
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
        let expected_span_count = if otel_type == OpenTelemetryType::GenAi {
            3
        } else {
            2
        };
        assert_eq!(spans.len(), expected_span_count);
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
fn generic_function_error_propagates_to_agent_span() {
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
    assert_eq!(spans.len(), 2);
    let agent_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "invoke_agent agent")
        .expect("expected agent span");
    let function_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "function")
        .expect("expected generic function span");
    assert_eq!(
        function_span.parent_span_id,
        agent_span.span_context.span_id()
    );
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
fn generic_parent_error_propagation_isolated_by_trace_id() {
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
    assert_eq!(spans.len(), 6);
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
    assert_eq!(
        serde_json::from_str::<Json>(&attributes["gen_ai.input.messages"]).unwrap(),
        json!([{
            "role": "user",
            "parts": [{"type": "text", "content": "hello"}]
        }])
    );

    let response = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                    message: Some(MessageContent::Text("hello back".to_string())),
                    finish_reason: Some(FinishReason::Complete),
                    ..empty_annotated_response()
                }))
                .build(),
        ),
    );
    let response_attributes =
        attr_map(&crate::observability::otel_genai::end_attributes(&response));
    assert_eq!(
        serde_json::from_str::<Json>(&response_attributes["gen_ai.output.messages"]).unwrap(),
        json!([{
            "role": "assistant",
            "parts": [{"type": "text", "content": "hello back"}],
            "finish_reason": "stop"
        }])
    );

    let response_without_finish_reason = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                    message: Some(MessageContent::Text("still working".to_string())),
                    ..empty_annotated_response()
                }))
                .build(),
        ),
    );
    let response_attributes = attr_map(&crate::observability::otel_genai::end_attributes(
        &response_without_finish_reason,
    ));
    assert_eq!(
        serde_json::from_str::<Json>(&response_attributes["gen_ai.output.messages"]).unwrap(),
        json!([{
            "role": "assistant",
            "parts": [{"type": "text", "content": "still working"}],
            "finish_reason": "unknown"
        }])
    );

    for (reason, expected) in [
        (FinishReason::Complete, "stop"),
        (FinishReason::Length, "length"),
        (FinishReason::ToolUse, "tool_call"),
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
fn gen_ai_projection_covers_message_variants_and_empty_input() {
    let annotated_request = serde_json::from_value::<AnnotatedLlmRequest>(json!({
        "instructions": "Be concise.",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{not-json"
                    }
                }]
            },
            {
                "role": "tool",
                "content": "result",
                "tool_call_id": "call-1"
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "call-2",
                    "content": "claude result"
                }]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-3",
                        "content": "mixed result"
                    },
                    {
                        "type": "text",
                        "text": "continue"
                    }
                ]
            },
            {
                "role": "user",
                "content": []
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "provider_native",
                        "provider": "example",
                        "kind": "reasoning",
                        "value": {"content": "provider payload"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"}
                    }
                ]
            }
        ]
    }))
    .unwrap();
    let event = make_scope_event_with_profile(
        ScopeCategory::Start,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_request(std::sync::Arc::new(annotated_request))
                .build(),
        ),
    );
    let attributes = attr_map(&crate::observability::otel_genai::start_attributes(&event));
    assert_eq!(
        serde_json::from_str::<Json>(&attributes["gen_ai.system_instructions"]).unwrap(),
        json!([{"type": "text", "content": "Be concise."}])
    );
    assert_eq!(
        serde_json::from_str::<Json>(&attributes["gen_ai.input.messages"]).unwrap(),
        json!([
            {
                "role": "assistant",
                "parts": [{
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "lookup",
                    "arguments": "{not-json"
                }]
            },
            {
                "role": "tool",
                "parts": [{
                    "type": "tool_call_response",
                    "id": "call-1",
                    "response": "result"
                }]
            },
            {
                "role": "tool",
                "parts": [{
                    "type": "tool_call_response",
                    "id": "call-2",
                    "response": "claude result"
                }]
            },
            {
                "role": "user",
                "parts": [
                    {
                        "type": "tool_call_response",
                        "id": "call-3",
                        "response": "mixed result"
                    },
                    {
                        "type": "text",
                        "content": "continue"
                    }
                ]
            },
            {
                "role": "user",
                "parts": []
            },
            {
                "role": "user",
                "parts": [
                    {"type": "reasoning", "content": "provider payload"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"}
                    }
                ]
            }
        ])
    );

    let empty_event = make_scope_event_with_profile(
        ScopeCategory::Start,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_request(std::sync::Arc::new(AnnotatedLlmRequest::default()))
                .build(),
        ),
    );
    let attributes = attr_map(&crate::observability::otel_genai::start_attributes(
        &empty_event,
    ));
    assert!(!attributes.contains_key("gen_ai.input.messages"));
    assert!(!attributes.contains_key("gen_ai.system_instructions"));
}

#[test]
fn gen_ai_projection_covers_output_tool_calls_and_empty_output() {
    let tool_call_event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                    tool_calls: Some(vec![ResponseToolCall {
                        id: "call-1".to_string(),
                        name: "lookup".to_string(),
                        arguments: json!({"city": "Paris"}),
                    }]),
                    finish_reason: Some(FinishReason::ToolUse),
                    ..empty_annotated_response()
                }))
                .build(),
        ),
    );
    let attributes = attr_map(&crate::observability::otel_genai::end_attributes(
        &tool_call_event,
    ));
    assert_eq!(
        serde_json::from_str::<Json>(&attributes["gen_ai.output.messages"]).unwrap(),
        json!([{
            "role": "assistant",
            "parts": [{
                "type": "tool_call",
                "id": "call-1",
                "name": "lookup",
                "arguments": {"city": "Paris"}
            }],
            "finish_reason": "tool_call"
        }])
    );

    let empty_event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_response(std::sync::Arc::new(empty_annotated_response()))
                .build(),
        ),
    );
    let attributes = attr_map(&crate::observability::otel_genai::end_attributes(
        &empty_event,
    ));
    assert!(!attributes.contains_key("gen_ai.output.messages"));
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

    let variable = format!(
        "NEMO_RELAY_TEST_HEADER_SNAPSHOT_{}",
        Uuid::now_v7().simple()
    );
    let secret = "Bearer activation-secret";
    unsafe { std::env::set_var(&variable, secret) };
    let config = OpenTelemetryConfig::http_binary("demo-agent")
        .with_endpoint(endpoint)
        .with_header_env("authorization", &variable);
    let subscriber = OpenTelemetrySubscriber::new(config).unwrap();
    unsafe { std::env::set_var(&variable, "Bearer changed-secret") };
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
    assert_eq!(request.authorization.as_deref(), Some(secret));
    assert!(!request.body.is_empty());
    assert!(
        !request
            .body
            .windows(secret.len())
            .any(|window| window == secret.as_bytes())
    );
    assert!(
        subscriber
            .runtime_diagnostics()
            .entries()
            .iter()
            .all(|diagnostic| !diagnostic.message.contains(secret))
    );
    unsafe { std::env::remove_var(variable) };
}

#[test]
fn root_metadata_promotes_to_a_shared_otlp_resource() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let (request_tx, request_rx) = mpsc::channel();
    spawn_http_collector(listener, request_tx);

    let subscriber = OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::http_binary("resource-test")
            .with_endpoint(endpoint)
            .with_resource_attribute("tenant.region", "configured")
            .with_promote_resource_metadata_prefixes(["tenant."]),
    )
    .unwrap();
    let callback = subscriber.subscriber();
    let root_uuid = Uuid::now_v7();
    let child_uuid = Uuid::now_v7();
    callback(&make_start_event_with_metadata(
        root_uuid,
        None,
        "resource-root",
        json!({"tenant.id": "root-tenant", "tenant.region": "metadata"}),
    ));
    callback(&make_start_event_with_metadata(
        child_uuid,
        Some(root_uuid),
        "resource-child",
        json!({"tenant.id": "child-tenant"}),
    ));
    callback(&make_end_event(
        child_uuid,
        Some(root_uuid),
        "resource-child",
        ScopeType::Agent,
        None,
    ));
    callback(&make_end_event(
        root_uuid,
        None,
        "resource-root",
        ScopeType::Agent,
        None,
    ));
    {
        let processor = subscriber.inner.processor.lock().unwrap();
        let pipelines = processor.dynamic_pipelines.lock().unwrap();
        assert_eq!(pipelines.len(), 1);
        let key = pipelines.keys().next().unwrap();
        assert!(key.contains("root-tenant"));
        assert!(key.contains("configured"));
        assert!(!key.contains("child-tenant"));
        assert!(!key.contains("metadata"));
    }
    subscriber.force_flush().unwrap();

    let request = request_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("expected an OTLP request for the promoted root resource");
    let request = ExportTraceServiceRequest::decode(request.body.as_slice()).unwrap();
    assert_eq!(request.resource_spans.len(), 1);
    let resource_spans = request.resource_spans.first().unwrap();
    let resource = resource_spans.resource.as_ref().unwrap();
    assert_eq!(
        otlp_string_attribute(&resource.attributes, "tenant.id"),
        Some("root-tenant")
    );
    assert_eq!(
        otlp_string_attribute(&resource.attributes, "tenant.region"),
        Some("configured")
    );

    let spans = resource_spans
        .scope_spans
        .iter()
        .flat_map(|scope_spans| &scope_spans.spans)
        .collect::<Vec<_>>();
    assert_eq!(spans.len(), 2);
    for span in spans {
        assert!(!has_promoted_resource_metadata(&span.attributes));
        assert!(
            span.events
                .iter()
                .all(|event| !has_promoted_resource_metadata(&event.attributes))
        );
    }
}

fn otlp_string_attribute<'a>(attributes: &'a [OtlpKeyValue], key: &str) -> Option<&'a str> {
    attributes
        .iter()
        .find(|attribute| attribute.key == key)
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|value| match value.value.as_ref() {
            Some(any_value::Value::StringValue(value)) => Some(value.as_str()),
            _ => None,
        })
}

fn has_promoted_resource_metadata(attributes: &[OtlpKeyValue]) -> bool {
    attributes.iter().any(|attribute| {
        attribute.key.ends_with(".metadata.tenant.id")
            || attribute.key == "tenant.id"
            || matches!(
                attribute
                    .value
                    .as_ref()
                    .and_then(|value| value.value.as_ref()),
                Some(any_value::Value::StringValue(value)) if value == "child-tenant"
            )
    })
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
fn metric_schema_marks_are_not_projected_to_direct_traces() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());
    let root_uuid = Uuid::now_v7();

    processor.process(&make_start_event(
        root_uuid,
        None,
        "agent",
        ScopeType::Agent,
        None,
    ));
    processor.process(&Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid(root_uuid)
            .name("tokens-saved")
            .data(json!({
                "measurements": [{
                    "name": "example.tokens.saved",
                    "kind": "counter",
                    "value_type": "u64",
                    "value": 42
                }]
            }))
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version(METRIC_DATA_SCHEMA_VERSION)
                    .build(),
            )
            .build(),
        None,
        None,
    )));
    processor.process(&Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid(root_uuid)
            .name("future-metric")
            .data(json!({"measurements": []}))
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version("999")
                    .build(),
            )
            .build(),
        None,
        None,
    )));
    processor.process(&Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid(root_uuid)
            .name("invalid-metric")
            .data(json!({"measurements": []}))
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version(METRIC_DATA_SCHEMA_VERSION)
                    .build(),
            )
            .build(),
        None,
        None,
    )));
    processor.process(&make_mark_event(Some(root_uuid), "routing-decision", None));
    processor.process(&make_end_event(
        root_uuid,
        None,
        "agent",
        ScopeType::Agent,
        None,
    ));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].events.events.len(), 1);
    assert_eq!(spans[0].events.events[0].name.as_ref(), "routing-decision");
    assert_eq!(processor.invalid_metric_count, 2);
}

#[test]
fn direct_trace_subscriber_exposes_runtime_diagnostics() {
    let (provider, _exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "diagnostics");
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("invalid-metric")
            .data(json!({"measurements": []}))
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version("999")
                    .build(),
            )
            .build(),
        None,
        None,
    ));

    for _ in 0..3 {
        (subscriber.subscriber())(&event);
    }

    let diagnostics = subscriber.runtime_diagnostics();
    let diagnostic = diagnostics
        .get("otel.metric_mark_invalid")
        .expect("invalid metric diagnostic");
    assert_eq!(diagnostic.count, 3);
    assert!(
        diagnostic
            .message
            .contains("unsupported metric schema version")
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
fn process_start_removes_completed_span_expiry_index_entry() {
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
            .completed_span_expiry_index
            .values()
            .filter(|uuids| uuids.contains(&tool_uuid))
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
    assert!(
        processor
            .completed_span_expiry_index
            .values()
            .all(|uuids| !uuids.contains(&tool_uuid))
    );

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
            .completed_span_expiry_index
            .values()
            .filter(|uuids| uuids.contains(&tool_uuid))
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

    let tool_end_profile = CategoryProfile::builder()
        .tool_call_id("call-456")
        .tool_result_annotation(json!({"opaque": {"rank": 1}}))
        .build();
    let tool_end_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("lookup")
            .metadata(json!({"phase": "complete"}))
            .data(json!({"result": true}))
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::tool(),
        Some(tool_end_profile),
    ));
    let tool_end_attributes = attr_map(&end_attributes(&tool_end_event));
    assert_eq!(
        tool_end_attributes.get("nemo_relay.end.data.result"),
        Some(&"true".to_string())
    );
    assert_eq!(
        tool_end_attributes.get("nemo_relay.tool.result.annotation"),
        Some(&r#"{"opaque":{"rank":1}}"#.to_string())
    );

    let llm_profile = CategoryProfile::builder()
        .tool_result_annotation(json!({"must_not_project": true}))
        .build();
    let llm_end_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("chat").build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::llm(),
        Some(llm_profile),
    ));
    assert!(
        !attr_map(&end_attributes(&llm_end_event))
            .contains_key("nemo_relay.tool.result.annotation")
    );

    let gen_ai_attributes = attr_map(&crate::observability::otel_genai::end_attributes(
        &tool_end_event,
    ));
    assert!(!gen_ai_attributes.contains_key("nemo_relay.tool.result.annotation"));
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

    {
        let _pricing_guard = pricing_test_mutex().lock().unwrap();
        install_test_pricing("priced-model");
        let _reset_guard = ResetPricingResolverGuard;
        let partial_cost_event = make_scope_event_with_profile(
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
                            cache_write_tokens: Some(10),
                            cost: None,
                        }),
                        ..empty_annotated_response()
                    }))
                    .build(),
            ),
        );
        let partial_cost_attributes = attr_map(&end_attributes(&partial_cost_event));
        assert!(!partial_cost_attributes.contains_key("nemo_relay.llm.cost.total"));
        assert!(!partial_cost_attributes.contains_key("nemo_relay.llm.cost.currency"));
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
        SignalRuntimeDiagnostics::new(None),
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
    let runtime_diagnostics =
        SignalRuntimeDiagnostics::new(Some("opentelemetry.traces[2].endpoint".to_string()));
    let processor = DiagnosticBatchSpanProcessor::new_with_batch_config(
        exporter.clone(),
        "https://collector.example/v1/traces".to_string(),
        runtime_diagnostics.clone(),
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
        Some("opentelemetry.traces[2].endpoint")
    );
    assert!(diagnostic.message.contains("https://collector.example:443"));
    assert_eq!(
        runtime_diagnostics
            .snapshot()
            .get("otel.spans_dropped")
            .map(|diagnostic| diagnostic.count),
        Some(2)
    );
}

#[test]
fn direct_trace_processor_records_cumulative_queue_drops_on_flush_and_shutdown() {
    let runtime_diagnostics = SignalRuntimeDiagnostics::new(None);
    let processor = DiagnosticBatchSpanProcessor::new_with_batch_config(
        InMemorySpanExporterBuilder::new().build(),
        "https://collector.example/v1/traces".to_string(),
        runtime_diagnostics.clone(),
        BatchConfigBuilder::default().build(),
    );
    processor.completed_spans.store(3, Ordering::Relaxed);
    processor.accepted_spans.store(1, Ordering::Relaxed);

    processor.force_flush().unwrap();
    assert_eq!(
        runtime_diagnostics
            .snapshot()
            .get("otel.spans_dropped")
            .map(|diagnostic| diagnostic.count),
        Some(2)
    );

    processor.completed_spans.store(5, Ordering::Relaxed);
    processor
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();
    let diagnostics = runtime_diagnostics.snapshot();
    let diagnostic = diagnostics
        .get("otel.spans_dropped")
        .expect("direct trace drop diagnostic");
    assert_eq!(diagnostic.count, 4);
    assert!(diagnostic.message.contains("https://collector.example:443"));
}

#[test]
fn plugin_trace_subscriber_runtime_diagnostics_use_trace_field() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let _ = crate::plugin::clear_plugin_configuration();
    let _clear_guard = ClearPluginConfigurationGuard;
    futures::executor::block_on(crate::plugin::initialize_plugins_exact(
        crate::plugin::PluginConfig::default(),
    ))
    .unwrap();

    let subscriber = OpenTelemetrySubscriber::new_for_plugin(
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces"),
        2,
    )
    .unwrap();
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("invalid-metric")
            .data(json!({"measurements": []}))
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version("999")
                    .build(),
            )
            .build(),
        None,
        None,
    ));

    (subscriber.subscriber())(&event);

    let report = crate::plugin::active_plugin_report().unwrap();
    let diagnostic = report
        .runtime_diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "otel.metric_mark_invalid")
        .expect("invalid metric diagnostic");
    assert_eq!(
        diagnostic.field.as_deref(),
        Some("opentelemetry.traces[2].endpoint")
    );
}

#[test]
fn trace_export_failures_are_diagnosed_until_a_later_export_recovers() {
    let runtime_diagnostics = SignalRuntimeDiagnostics::new(None);
    let processor = DiagnosticBatchSpanProcessor::new_with_batch_config(
        FailingThenRecoveringSpanExporter::default(),
        "https://collector.example/v1/traces".to_string(),
        runtime_diagnostics.clone(),
        BatchConfigBuilder::default()
            .with_max_export_batch_size(1)
            .build(),
    );
    let provider = SdkTracerProvider::builder()
        .with_span_processor(processor)
        .build();
    let tracer = provider.tracer("trace-export-failure-diagnostics-test");

    tracer.start("first-export-fails").end();
    assert!(provider.force_flush().is_err());

    let diagnostics = runtime_diagnostics.snapshot();
    let diagnostic = diagnostics
        .get("otel.traces_export_failed")
        .expect("trace export failure diagnostic");
    assert_eq!(diagnostic.count, 1);
    assert!(diagnostic.message.contains("https://collector.example:443"));
    assert!(!diagnostic.message.contains("collector unavailable"));

    let repeated_flush = provider.force_flush().unwrap_err();
    assert!(
        repeated_flush
            .to_string()
            .contains("otel.traces_export_failed (1)")
    );

    tracer.start("recovery-export").end();
    provider.force_flush().unwrap();
    provider.shutdown().unwrap();
}

#[test]
fn trace_endpoint_log_identity_redacts_and_validates_urls() {
    for (endpoint, expected) in [
        ("not a URL", "an invalid OTLP endpoint"),
        ("ftp://collector.example/secret", "an invalid OTLP endpoint"),
        (
            "https://[::1]:4318/private?access_token=url-secret",
            "https://[::1]:4318",
        ),
        (
            "https://user:password@collector.example:4318/private?access_token=url-secret#fragment-secret",
            "https://collector.example:4318",
        ),
    ] {
        assert_eq!(trace_endpoint_log_identity(endpoint), expected);
    }
}

#[test]
fn trace_export_failures_use_a_safe_endpoint_identity() {
    let endpoint = "https://user:password@collector.example:4318/private-path?access_token=url-secret#fragment-secret";
    let runtime_diagnostics = SignalRuntimeDiagnostics::new(None);
    let processor = DiagnosticBatchSpanProcessor::new_with_batch_config(
        SensitiveFailingSpanExporter,
        endpoint.to_string(),
        runtime_diagnostics.clone(),
        BatchConfigBuilder::default()
            .with_max_export_batch_size(1)
            .build(),
    );
    let provider = SdkTracerProvider::builder()
        .with_span_processor(processor)
        .build();
    let tracer = provider.tracer("safe-trace-export-failure-test");

    tracer.start("export-fails").end();
    let error = provider.force_flush().unwrap_err().to_string();
    let diagnostic = runtime_diagnostics
        .snapshot()
        .get("otel.traces_export_failed")
        .expect("trace export failure diagnostic")
        .message
        .clone();

    for message in [&error, &diagnostic] {
        assert!(message.contains("https://collector.example:4318"));
        for secret in [
            "user",
            "password",
            "private-path",
            "url-secret",
            "fragment-secret",
            "exporter-secret",
            "Authorization",
        ] {
            assert!(
                !message.contains(secret),
                "trace export failure leaked {secret:?}: {message}"
            );
        }
    }

    provider.shutdown().unwrap();
}

#[test]
fn unrecovered_trace_export_failure_is_retained_in_the_active_plugin_report() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let _ = crate::plugin::clear_plugin_configuration();
    let _clear_guard = ClearPluginConfigurationGuard;
    futures::executor::block_on(crate::plugin::initialize_plugins_exact(
        crate::plugin::PluginConfig::default(),
    ))
    .unwrap();

    let runtime_diagnostics =
        SignalRuntimeDiagnostics::new(Some("opentelemetry.traces[2].endpoint".to_string()));
    let processor = DiagnosticBatchSpanProcessor::new_with_batch_config(
        FailingThenRecoveringSpanExporter::default(),
        "https://collector.example/v1/traces".to_string(),
        runtime_diagnostics.clone(),
        BatchConfigBuilder::default()
            .with_max_export_batch_size(1)
            .build(),
    );
    let provider = SdkTracerProvider::builder()
        .with_span_processor(processor)
        .build();
    let tracer = provider.tracer("unrecovered-trace-export-failure-test");

    tracer.start("export-fails").end();
    assert!(provider.force_flush().is_err());

    let shutdown = provider.shutdown().unwrap_err();
    assert!(
        shutdown
            .to_string()
            .contains(OTEL_RUNTIME_DELIVERY_FAILURE_MARKER)
    );
    assert!(
        shutdown
            .to_string()
            .contains("otel.traces_export_failed (1)")
    );

    let report = crate::plugin::active_plugin_report().unwrap();
    let diagnostic = report
        .runtime_diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "otel.traces_export_failed")
        .expect("trace export failure diagnostic");
    assert_eq!(diagnostic.count, 1);
    assert_eq!(
        diagnostic.field.as_deref(),
        Some("opentelemetry.traces[2].endpoint")
    );
    assert!(diagnostic.message.contains("https://collector.example:443"));
    assert!(!diagnostic.message.contains("collector unavailable"));
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
            SignalRuntimeDiagnostics::new(None),
        )
        .unwrap();
        provider.force_flush().ok();
        provider.shutdown().ok();
    });
}
