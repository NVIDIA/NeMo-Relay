// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core scope, provider normalization, and propagation integration with OTLP.

use super::*;
use crate::observability_test_support::atif::{AtifAgentInfo, AtifExporter, AtifStepExtra};
use crate::observability_test_support::otel_logs::{
    OpenTelemetryLogConfig, OpenTelemetryLogSubscriber,
};
use crate::observability_test_support::{relay_span_id, relay_trace_id};
use crate::test_support::pricing_test_mutex;
use nemo_relay_model_pricing::{
    PricingCatalog, PricingResolver, reset_active_pricing_resolver, set_active_pricing_resolver,
};
use nemo_relay_runtime::api::event::{
    BaseEvent, CategoryProfile, DataSchema, Event, EventCategory, METRIC_DATA_SCHEMA_NAME,
    METRIC_DATA_SCHEMA_VERSION, MarkEvent, ScopeCategory, ScopeEvent, tool_attributes_to_strings,
};
use nemo_relay_runtime::api::llm::{LlmCallExecuteParams, LlmRequest, llm_call_execute};
use nemo_relay_runtime::api::runtime::{
    NemoRelayContextState, PropagationContext, TASK_SCOPE_STACK, ThreadScopeStackBinding,
    capture_propagation_context, capture_propagation_context_with_root,
    capture_rootless_propagation_context, capture_thread_scope_stack, capture_traceparent,
    create_scope_stack, create_scope_stack_from_propagation, fork_scope_stack, global_context,
    restore_thread_scope_stack, set_thread_scope_stack, task_scope_push, task_scope_remove,
};
use nemo_relay_runtime::api::scope::{ScopeHandle, ScopeType};
use nemo_relay_runtime::api::scope::{event, pop_scope, push_scope};
use nemo_relay_runtime::api::tool::ToolAttributes;
use nemo_relay_types::Json;
use nemo_relay_types::codec::request::{AnnotatedLlmRequest, MessageContent};
use nemo_relay_types::codec::response::{
    AnnotatedLlmResponse, ApiSpecificResponse, CostEstimate, CostSource, FinishReason,
    ResponseToolCall, Usage,
};
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

struct ClosePluginHostGuard;

impl Drop for ClosePluginHostGuard {
    fn drop(&mut self) {
        let _ = crate::test_support::plugin_host::test_close_plugin_host();
    }
}

fn test_tokio_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap()
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
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_global();
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();

    let exporter = BlockingSpanExporter::default();
    let processor = DiagnosticBatchSpanProcessor::new_with_batch_config(
        exporter.clone(),
        "https://collector.example/v1/traces".to_string(),
        SignalRuntimeDiagnostics::new(None)
            .with_host(crate::observability_test_support::core_host()),
        BatchConfigBuilder::default()
            .with_max_queue_size(1)
            .with_max_export_batch_size(1)
            .with_scheduled_delay(Duration::from_secs(60))
            .build(),
    );
    let provider = SdkTracerProvider::builder()
        .with_span_processor(processor)
        .build();
    let trace_subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "trace")
        .with_host(crate::observability_runtime::runtime_host());
    let trace_name = format!("slow-trace-{}", Uuid::now_v7().simple());
    trace_subscriber.register(&trace_name).unwrap();

    let (healthy_tx, healthy_rx) = mpsc::channel();
    let healthy_name = format!("healthy-{}", Uuid::now_v7().simple());
    nemo_relay_runtime::api::subscriber::register_subscriber(
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

    nemo_relay_runtime::api::scope::event(
        nemo_relay_runtime::api::scope::EmitMarkEventParams::builder()
            .name("healthy-event")
            .build(),
    )
    .unwrap();

    let (barrier_tx, barrier_rx) = mpsc::channel();
    let barrier = thread::spawn(move || {
        barrier_tx
            .send(nemo_relay_runtime::api::subscriber::flush_subscribers())
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
    nemo_relay_runtime::api::subscriber::deregister_subscriber(&healthy_name).unwrap();
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

// Fail the first request, then allow later requests to succeed.

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
    log::set_max_level(log::LevelFilter::Info);
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

async fn execute_llm_and_capture_trace_context_with_parent(
    name: &str,
    parent: Option<ScopeHandle>,
) -> (String, Option<String>, String) {
    let captured = Arc::new(Mutex::new(None));
    let captured_request = captured.clone();
    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name(name)
            .parent_opt(parent)
            .request(LlmRequest {
                headers: serde_json::Map::from_iter([(
                    "tracestate".to_string(),
                    Json::String("caller=value".to_string()),
                )]),
                content: json!({"prompt": "hello"}),
            })
            .func(Arc::new(move |request| {
                let traceparent = request
                    .headers
                    .get("traceparent")
                    .and_then(Json::as_str)
                    .map(ToOwned::to_owned)
                    .expect("managed LLM request must contain traceparent");
                let tracestate = request
                    .headers
                    .get("tracestate")
                    .and_then(Json::as_str)
                    .map(ToOwned::to_owned);
                let callback_traceparent = capture_traceparent()
                    .expect("managed LLM callback must expose its exact traceparent");
                *captured_request.lock().unwrap() =
                    Some((traceparent, tracestate, callback_traceparent));
                Box::pin(async { Ok(json!({"ok": true})) })
            }))
            .build(),
    )
    .await
    .unwrap();
    captured
        .lock()
        .unwrap()
        .take()
        .expect("managed LLM provider must capture trace context")
}

async fn execute_llm_and_capture_trace_context(name: &str) -> (String, Option<String>, String) {
    execute_llm_and_capture_trace_context_with_parent(name, None).await
}

async fn execute_llm_and_capture_traceparent(name: &str) -> String {
    execute_llm_and_capture_trace_context(name).await.0
}

fn finish_trace_subscriber(
    subscriber: &OpenTelemetrySubscriber,
    name: &str,
    exporter: &opentelemetry_sdk::trace::InMemorySpanExporter,
) -> Vec<opentelemetry_sdk::trace::SpanData> {
    nemo_relay_runtime::api::subscriber::flush_subscribers().unwrap();
    assert!(subscriber.deregister(name).unwrap());
    subscriber.force_flush().unwrap();
    exporter.get_finished_spans().unwrap()
}

fn assert_traceparent_matches_exported_span(
    traceparent: &str,
    span: &opentelemetry_sdk::trace::SpanData,
) {
    assert_eq!(
        traceparent,
        format!(
            "00-{}-{}-01",
            span.span_context.trace_id(),
            span.span_context.span_id()
        )
    );
}

#[test]
fn managed_llm_traceparent_matches_exported_span_directly() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    set_thread_scope_stack(create_scope_stack());
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "direct-llm")
        .with_host(crate::observability_runtime::runtime_host());
    let subscriber_name = format!("direct_llm_{}", Uuid::now_v7().simple());
    subscriber.register(&subscriber_name).unwrap();

    let turn = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("direct-turn")
            .scope_type(ScopeType::Custom)
            .build(),
    )
    .unwrap();
    let traceparent = runtime.block_on(execute_llm_and_capture_traceparent("direct-llm"));
    pop_scope(
        nemo_relay_runtime::api::scope::PopScopeParams::builder()
            .handle_uuid(&turn.uuid)
            .build(),
    )
    .unwrap();

    let spans = finish_trace_subscriber(&subscriber, &subscriber_name, &exporter);
    let turn_span = finished_span_named(&spans, "direct-turn");
    let llm_span = finished_span_named(&spans, "direct-llm");
    assert_traceparent_matches_exported_span(&traceparent, llm_span);
    assert_eq!(
        llm_span.span_context.trace_id(),
        turn_span.span_context.trace_id()
    );
    assert_eq!(llm_span.parent_span_id, turn_span.span_context.span_id());
}

#[test]
fn explicit_capture_root_controls_receiver_trace_from_local_senders() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let root = Uuid::now_v7();
    let mut contexts = Vec::new();
    for _ in 0..2 {
        set_thread_scope_stack(create_scope_stack());
        let sender = push_scope(
            nemo_relay_runtime::api::scope::PushScopeParams::builder()
                .name("explicit-root-sender")
                .scope_type(ScopeType::Agent)
                .build(),
        )
        .unwrap();
        contexts.push(capture_propagation_context_with_root(Some(root)).unwrap());
        pop_scope(
            nemo_relay_runtime::api::scope::PopScopeParams::builder()
                .handle_uuid(&sender.uuid)
                .build(),
        )
        .unwrap();
    }
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "explicit-root")
        .with_host(crate::observability_runtime::runtime_host());
    let subscriber_name = format!("explicit_root_{}", Uuid::now_v7().simple());
    subscriber.register(&subscriber_name).unwrap();
    for context in &contexts {
        let receiver = create_scope_stack_from_propagation(context).unwrap();
        runtime.block_on(TASK_SCOPE_STACK.scope(
            receiver,
            execute_llm_and_capture_traceparent("explicit-root-receiver"),
        ));
    }
    let spans = finish_trace_subscriber(&subscriber, &subscriber_name, &exporter);
    assert_eq!(spans.len(), 2);
    for span in &spans {
        assert_eq!(span.span_context.trace_id(), relay_trace_id(root));
    }
    for context in contexts {
        assert_eq!(context.root_uuid, Some(root));
        assert_eq!(context.traceparent, None);
        assert_eq!(context.tracestate, None);
    }
}

#[test]
fn explicit_popped_parent_traceparent_matches_exported_span() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    set_thread_scope_stack(create_scope_stack());
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "popped-parent")
        .with_host(crate::observability_runtime::runtime_host());
    let subscriber_name = format!("popped_parent_{}", Uuid::now_v7().simple());
    subscriber.register(&subscriber_name).unwrap();
    let agent = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("popped-parent-agent")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    let parent = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("popped-parent")
            .scope_type(ScopeType::Function)
            .build(),
    )
    .unwrap();
    pop_scope(
        nemo_relay_runtime::api::scope::PopScopeParams::builder()
            .handle_uuid(&parent.uuid)
            .build(),
    )
    .unwrap();
    let (header, _, callback_header) =
        runtime.block_on(execute_llm_and_capture_trace_context_with_parent(
            "popped-parent-llm",
            Some(parent.clone()),
        ));
    pop_scope(
        nemo_relay_runtime::api::scope::PopScopeParams::builder()
            .handle_uuid(&agent.uuid)
            .build(),
    )
    .unwrap();
    let spans = finish_trace_subscriber(&subscriber, &subscriber_name, &exporter);
    let llm = finished_span_named(&spans, "popped-parent-llm");
    assert_traceparent_matches_exported_span(&header, llm);
    assert_eq!(callback_header, header);
    assert_eq!(llm.parent_span_id, relay_span_id(parent.uuid));
    assert_eq!(llm.span_context.trace_id(), relay_trace_id(agent.uuid));
}

#[test]
fn stacked_child_of_popped_parent_preserves_trace_for_early_and_late_exporters() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    set_thread_scope_stack(create_scope_stack());
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "popped-ancestor")
        .with_host(crate::observability_runtime::runtime_host());
    let subscriber_name = format!("popped_ancestor_{}", Uuid::now_v7().simple());
    subscriber.register(&subscriber_name).unwrap();
    let agent = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("popped-ancestor-agent")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    let parent = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("popped-ancestor-parent")
            .scope_type(ScopeType::Function)
            .build(),
    )
    .unwrap();
    pop_scope(
        nemo_relay_runtime::api::scope::PopScopeParams::builder()
            .handle_uuid(&parent.uuid)
            .build(),
    )
    .unwrap();

    let (late_provider, late_exporter) = make_provider();
    let late_subscriber =
        OpenTelemetrySubscriber::from_tracer_provider(late_provider, "late-popped-ancestor")
            .with_host(crate::observability_runtime::runtime_host());
    let late_name = format!("late_popped_ancestor_{}", Uuid::now_v7().simple());
    late_subscriber.register(&late_name).unwrap();
    let child = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("popped-ancestor-child")
            .scope_type(ScopeType::Function)
            .parent(&parent)
            .build(),
    )
    .unwrap();
    let child_header = capture_traceparent().unwrap();
    let (header, _, callback_header) = runtime.block_on(
        execute_llm_and_capture_trace_context_with_parent("popped-ancestor-llm", None),
    );
    for handle in [&child, &agent] {
        pop_scope(
            nemo_relay_runtime::api::scope::PopScopeParams::builder()
                .handle_uuid(&handle.uuid)
                .build(),
        )
        .unwrap();
    }
    let early_spans = finish_trace_subscriber(&subscriber, &subscriber_name, &exporter);
    let late_spans = finish_trace_subscriber(&late_subscriber, &late_name, &late_exporter);
    assert!(
        late_spans
            .iter()
            .all(|span| span.name != "popped-ancestor-parent")
    );
    for spans in [&early_spans, &late_spans] {
        let child_span = finished_span_named(spans, "popped-ancestor-child");
        let llm_span = finished_span_named(spans, "popped-ancestor-llm");
        assert_traceparent_matches_exported_span(&child_header, child_span);
        assert_traceparent_matches_exported_span(&header, llm_span);
        assert_eq!(
            child_span.span_context.trace_id(),
            relay_trace_id(agent.uuid)
        );
        assert_eq!(child_span.parent_span_id, relay_span_id(parent.uuid));
        assert_eq!(llm_span.parent_span_id, relay_span_id(child.uuid));
    }
    assert_eq!(callback_header, header);
}

#[test]
fn top_level_callback_context_preserves_relay_root_and_exported_trace() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    let stack = create_scope_stack();
    let relay_root_uuid = stack.read().unwrap().root_uuid();
    set_thread_scope_stack(stack);
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let (upstream_provider, upstream_exporter) = make_provider();
    let upstream_subscriber =
        OpenTelemetrySubscriber::from_tracer_provider(upstream_provider, "top-level-capture")
            .with_host(crate::observability_runtime::runtime_host());
    let upstream_name = format!("top_level_capture_{}", Uuid::now_v7().simple());
    upstream_subscriber.register(&upstream_name).unwrap();
    let observed = Arc::new(Mutex::new(None));
    let observed_request = observed.clone();

    runtime
        .block_on(llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("top-level-capture-llm")
                .request(LlmRequest {
                    headers: serde_json::Map::new(),
                    content: json!({"prompt": "hello"}),
                })
                .func(Arc::new(move |request| {
                    let context = capture_propagation_context().unwrap();
                    let traceparent = request
                        .headers
                        .get("traceparent")
                        .and_then(Json::as_str)
                        .expect("managed LLM request must contain traceparent")
                        .to_owned();
                    *observed_request.lock().unwrap() = Some((context, traceparent));
                    Box::pin(async { Ok(json!({"ok": true})) })
                }))
                .build(),
        ))
        .unwrap();

    let (context, upstream_traceparent) = observed
        .lock()
        .unwrap()
        .take()
        .expect("provider must capture propagation context");
    assert_eq!(context.root_uuid, Some(relay_root_uuid));
    let upstream_spans =
        finish_trace_subscriber(&upstream_subscriber, &upstream_name, &upstream_exporter);
    let upstream_span = finished_span_named(&upstream_spans, "top-level-capture-llm");
    assert_traceparent_matches_exported_span(&upstream_traceparent, upstream_span);
    assert_eq!(
        context.traceparent.as_deref(),
        Some(upstream_traceparent.as_str())
    );

    let imported = create_scope_stack_from_propagation(&context).unwrap();
    let (downstream_provider, downstream_exporter) = make_provider();
    let downstream_subscriber =
        OpenTelemetrySubscriber::from_tracer_provider(downstream_provider, "captured-child")
            .with_host(crate::observability_runtime::runtime_host());
    let downstream_name = format!("captured_child_{}", Uuid::now_v7().simple());
    downstream_subscriber.register(&downstream_name).unwrap();
    let downstream_traceparent = runtime.block_on(TASK_SCOPE_STACK.scope(
        imported,
        execute_llm_and_capture_traceparent("captured-child-llm"),
    ));

    let downstream_spans = finish_trace_subscriber(
        &downstream_subscriber,
        &downstream_name,
        &downstream_exporter,
    );
    let downstream_span = finished_span_named(&downstream_spans, "captured-child-llm");
    assert_traceparent_matches_exported_span(&downstream_traceparent, downstream_span);
    assert_eq!(
        downstream_span.span_context.trace_id(),
        upstream_span.span_context.trace_id()
    );
    assert_eq!(
        downstream_span.parent_span_id,
        upstream_span.span_context.span_id()
    );
    assert!(downstream_span.parent_span_is_remote);
}

#[test]
fn forked_llm_traceparent_matches_exported_span() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    set_thread_scope_stack(create_scope_stack());
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "forked-llm")
        .with_host(crate::observability_runtime::runtime_host());
    let subscriber_name = format!("forked_llm_{}", Uuid::now_v7().simple());
    subscriber.register(&subscriber_name).unwrap();

    let turn = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("forked-turn")
            .scope_type(ScopeType::Custom)
            .build(),
    )
    .unwrap();
    let forked = fork_scope_stack().unwrap();
    let traceparent = runtime.block_on(
        TASK_SCOPE_STACK.scope(forked, execute_llm_and_capture_traceparent("forked-llm")),
    );
    pop_scope(
        nemo_relay_runtime::api::scope::PopScopeParams::builder()
            .handle_uuid(&turn.uuid)
            .build(),
    )
    .unwrap();

    let spans = finish_trace_subscriber(&subscriber, &subscriber_name, &exporter);
    let turn_span = finished_span_named(&spans, "forked-turn");
    let llm_span = finished_span_named(&spans, "forked-llm");
    assert_traceparent_matches_exported_span(&traceparent, llm_span);
    assert_eq!(
        llm_span.span_context.trace_id(),
        turn_span.span_context.trace_id()
    );
    assert_eq!(llm_span.parent_span_id, turn_span.span_context.span_id());
}

#[test]
fn nested_agent_fork_llm_traceparent_matches_exported_span() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    set_thread_scope_stack(create_scope_stack());
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "agent-fork-llm")
        .with_host(crate::observability_runtime::runtime_host());
    let subscriber_name = format!("agent_fork_llm_{}", Uuid::now_v7().simple());
    subscriber.register(&subscriber_name).unwrap();

    let turn = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("agent-fork-turn")
            .scope_type(ScopeType::Custom)
            .build(),
    )
    .unwrap();
    let agent = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("agent-fork-parent")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    let forked = fork_scope_stack().unwrap();
    let traceparent = runtime.block_on(TASK_SCOPE_STACK.scope(
        forked,
        execute_llm_and_capture_traceparent("agent-fork-llm"),
    ));
    for handle in [&agent, &turn] {
        pop_scope(
            nemo_relay_runtime::api::scope::PopScopeParams::builder()
                .handle_uuid(&handle.uuid)
                .build(),
        )
        .unwrap();
    }

    let spans = finish_trace_subscriber(&subscriber, &subscriber_name, &exporter);
    let turn_span = finished_span_named(&spans, "agent-fork-turn");
    let agent_span = finished_span_named(&spans, "agent-fork-parent");
    let llm_span = finished_span_named(&spans, "agent-fork-llm");
    assert_traceparent_matches_exported_span(&traceparent, llm_span);
    assert_eq!(
        llm_span.span_context.trace_id(),
        turn_span.span_context.trace_id()
    );
    assert_eq!(llm_span.parent_span_id, agent_span.span_context.span_id());
}

#[test]
fn rooted_import_llm_traceparent_matches_exported_span() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    set_thread_scope_stack(create_scope_stack());
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "rooted-import-llm")
        .with_host(crate::observability_runtime::runtime_host());
    let subscriber_name = format!("rooted_import_llm_{}", Uuid::now_v7().simple());
    subscriber.register(&subscriber_name).unwrap();
    let root_uuid = Uuid::now_v7();
    let parent_uuid = Uuid::now_v7();
    let context = PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid: Some(root_uuid),
        parent_uuid,
        traceparent: Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_string()),
        tracestate: Some("vendor=value".to_string()),
    };
    let imported = create_scope_stack_from_propagation(&context).unwrap();

    let (traceparent, tracestate, callback_traceparent) = runtime.block_on(TASK_SCOPE_STACK.scope(
        imported,
        execute_llm_and_capture_trace_context("rooted-import-llm"),
    ));

    let spans = finish_trace_subscriber(&subscriber, &subscriber_name, &exporter);
    let llm_span = finished_span_named(&spans, "rooted-import-llm");
    assert_traceparent_matches_exported_span(&traceparent, llm_span);
    assert_eq!(
        llm_span.span_context.trace_id().to_string(),
        "4bf92f3577b34da6a3ce929d0e0e4736"
    );
    assert_eq!(llm_span.parent_span_id.to_string(), "00f067aa0ba902b7");
    assert!(llm_span.parent_span_is_remote);
    assert_eq!(tracestate.as_deref(), Some("vendor=value"));
    assert_eq!(callback_traceparent, traceparent);
}

#[test]
fn invalid_imported_tracestate_is_absent_from_provider_and_exported_span() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    set_thread_scope_stack(create_scope_stack());
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let (provider, exporter) = make_provider();
    let subscriber =
        OpenTelemetrySubscriber::from_tracer_provider(provider, "invalid-tracestate-llm")
            .with_host(crate::observability_runtime::runtime_host());
    let subscriber_name = format!("invalid_tracestate_llm_{}", Uuid::now_v7().simple());
    subscriber.register(&subscriber_name).unwrap();
    let root_uuid = Uuid::now_v7();
    let parent_uuid = Uuid::now_v7();
    let context = PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid: Some(root_uuid),
        parent_uuid,
        traceparent: Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_string()),
        tracestate: Some("vendor=bad\r\nX-Injected: yes".to_string()),
    };
    let imported = create_scope_stack_from_propagation(&context).unwrap();

    let (traceparent, tracestate, callback_traceparent) = runtime.block_on(TASK_SCOPE_STACK.scope(
        imported,
        execute_llm_and_capture_trace_context("invalid-tracestate-llm"),
    ));

    let spans = finish_trace_subscriber(&subscriber, &subscriber_name, &exporter);
    let llm_span = finished_span_named(&spans, "invalid-tracestate-llm");
    assert_traceparent_matches_exported_span(&traceparent, llm_span);
    assert_eq!(
        llm_span.span_context.trace_id().to_string(),
        "4bf92f3577b34da6a3ce929d0e0e4736"
    );
    assert_eq!(llm_span.parent_span_id.to_string(), "00f067aa0ba902b7");
    assert!(llm_span.parent_span_is_remote);
    assert_eq!(tracestate, None);
    assert!(llm_span.span_context.trace_state().header().is_empty());
    assert_eq!(callback_traceparent, traceparent);
}

#[test]
fn imported_w3c_parent_survives_an_additional_fork() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    set_thread_scope_stack(create_scope_stack());
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "reforked-import")
        .with_host(crate::observability_runtime::runtime_host());
    let subscriber_name = format!("reforked_import_{}", Uuid::now_v7().simple());
    subscriber.register(&subscriber_name).unwrap();
    let original_traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let context = PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid: Some(Uuid::now_v7()),
        parent_uuid: Uuid::now_v7(),
        traceparent: Some(original_traceparent.to_string()),
        tracestate: Some("vendor=value".to_string()),
    };
    let imported = create_scope_stack_from_propagation(&context).unwrap();

    let (captured_traceparent, outbound_traceparent) =
        runtime.block_on(TASK_SCOPE_STACK.scope(imported, async {
            let captured_traceparent = capture_propagation_context()
                .unwrap()
                .traceparent
                .expect("captured propagation context must retain traceparent");
            let forked = fork_scope_stack().unwrap();
            let outbound_traceparent = TASK_SCOPE_STACK
                .scope(
                    forked,
                    execute_llm_and_capture_traceparent("reforked-import-llm"),
                )
                .await;
            (captured_traceparent, outbound_traceparent)
        }));

    assert_eq!(captured_traceparent, original_traceparent);
    let spans = finish_trace_subscriber(&subscriber, &subscriber_name, &exporter);
    let llm_span = finished_span_named(&spans, "reforked-import-llm");
    assert_traceparent_matches_exported_span(&outbound_traceparent, llm_span);
    assert_eq!(
        llm_span.span_context.trace_id().to_string(),
        "4bf92f3577b34da6a3ce929d0e0e4736"
    );
    assert_eq!(llm_span.parent_span_id.to_string(), "00f067aa0ba902b7");
    assert!(llm_span.parent_span_is_remote);
}

#[test]
fn rootless_capture_from_local_agent_starts_a_separate_trace() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    set_thread_scope_stack(create_scope_stack());
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let (sender_provider, sender_exporter) = make_provider();
    let sender_subscriber =
        OpenTelemetrySubscriber::from_tracer_provider(sender_provider, "rootless-sender")
            .with_host(crate::observability_runtime::runtime_host());
    let sender_name = format!("rootless_sender_{}", Uuid::now_v7().simple());
    sender_subscriber.register(&sender_name).unwrap();

    let turn = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("rootless-import-turn")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    let context = capture_rootless_propagation_context().unwrap();
    assert_eq!(context.root_uuid, None);
    assert_eq!(context.traceparent, None);
    assert_eq!(context.tracestate, None);
    assert_eq!(
        capture_propagation_context_with_root(None).unwrap(),
        context
    );
    pop_scope(
        nemo_relay_runtime::api::scope::PopScopeParams::builder()
            .handle_uuid(&turn.uuid)
            .build(),
    )
    .unwrap();
    let sender_spans = finish_trace_subscriber(&sender_subscriber, &sender_name, &sender_exporter);
    let turn_span = finished_span_named(&sender_spans, "rootless-import-turn");

    let (receiver_provider, receiver_exporter) = make_provider();
    let receiver_subscriber =
        OpenTelemetrySubscriber::from_tracer_provider(receiver_provider, "rootless-receiver")
            .with_host(crate::observability_runtime::runtime_host());
    let receiver_name = format!("rootless_receiver_{}", Uuid::now_v7().simple());
    receiver_subscriber.register(&receiver_name).unwrap();
    let imported = create_scope_stack_from_propagation(&context).unwrap();
    let traceparent = runtime.block_on(TASK_SCOPE_STACK.scope(
        imported,
        execute_llm_and_capture_traceparent("rootless-import-llm"),
    ));
    let receiver_spans =
        finish_trace_subscriber(&receiver_subscriber, &receiver_name, &receiver_exporter);
    let llm_span = finished_span_named(&receiver_spans, "rootless-import-llm");
    assert_traceparent_matches_exported_span(&traceparent, llm_span);
    assert_ne!(
        llm_span.span_context.trace_id(),
        turn_span.span_context.trace_id()
    );
    assert_eq!(
        llm_span.parent_span_id,
        opentelemetry::trace::SpanId::INVALID
    );
}

#[test]
fn legacy_rootless_import_starts_its_first_local_scope_as_a_trace_root() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    set_thread_scope_stack(create_scope_stack());
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "legacy-rootless")
        .with_host(crate::observability_runtime::runtime_host());
    let subscriber_name = format!("legacy_rootless_{}", Uuid::now_v7().simple());
    subscriber.register(&subscriber_name).unwrap();
    let propagated_parent = Uuid::now_v7();
    let context = PropagationContext::from_json(
        &json!({
            "version": PropagationContext::VERSION,
            "parent_uuid": propagated_parent,
        })
        .to_string(),
    )
    .unwrap();
    let imported = create_scope_stack_from_propagation(&context).unwrap();

    let local_scope = runtime.block_on(TASK_SCOPE_STACK.scope(imported, async {
        let local_scope = push_scope(
            nemo_relay_runtime::api::scope::PushScopeParams::builder()
                .name("legacy-rootless-local")
                .scope_type(ScopeType::Custom)
                .build(),
        )
        .unwrap();
        pop_scope(
            nemo_relay_runtime::api::scope::PopScopeParams::builder()
                .handle_uuid(&local_scope.uuid)
                .build(),
        )
        .unwrap();
        local_scope
    }));

    let spans = finish_trace_subscriber(&subscriber, &subscriber_name, &exporter);
    let span = finished_span_named(&spans, "legacy-rootless-local");
    assert_eq!(
        span.span_context.trace_id(),
        relay_trace_id(local_scope.uuid)
    );
    assert_eq!(span.parent_span_id, opentelemetry::trace::SpanId::INVALID);
    assert!(!span.parent_span_is_remote);
}

#[test]
fn propagation_context_traceparent_agrees_with_active_trace_capture() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    let stack = create_scope_stack();
    let relay_root_uuid = stack.read().unwrap().root_uuid();
    set_thread_scope_stack(stack);
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "capture-agreement")
        .with_host(crate::observability_runtime::runtime_host());
    let subscriber_name = format!("capture_agreement_{}", Uuid::now_v7().simple());
    subscriber.register(&subscriber_name).unwrap();

    let turn = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("capture-agreement-turn")
            .scope_type(ScopeType::Custom)
            .build(),
    )
    .unwrap();
    let context = capture_propagation_context().unwrap();
    let traceparent = capture_traceparent().unwrap();
    assert_eq!(context.root_uuid, Some(relay_root_uuid));
    assert_ne!(context.root_uuid, Some(turn.uuid));
    assert_eq!(context.traceparent.as_deref(), Some(traceparent.as_str()));
    assert_eq!(context.to_traceparent().unwrap(), traceparent);
    pop_scope(
        nemo_relay_runtime::api::scope::PopScopeParams::builder()
            .handle_uuid(&turn.uuid)
            .build(),
    )
    .unwrap();

    let spans = finish_trace_subscriber(&subscriber, &subscriber_name, &exporter);
    let turn_span = finished_span_named(&spans, "capture-agreement-turn");
    assert_traceparent_matches_exported_span(&traceparent, turn_span);
}

#[test]
fn late_subscriber_llm_traceparent_matches_exported_span() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    set_thread_scope_stack(create_scope_stack());
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let turn = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("late-subscriber-turn")
            .scope_type(ScopeType::Custom)
            .build(),
    )
    .unwrap();

    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "late-subscriber-llm")
        .with_host(crate::observability_runtime::runtime_host());
    let subscriber_name = format!("late_subscriber_llm_{}", Uuid::now_v7().simple());
    subscriber.register(&subscriber_name).unwrap();
    let traceparent = runtime.block_on(execute_llm_and_capture_traceparent("late-subscriber-llm"));
    pop_scope(
        nemo_relay_runtime::api::scope::PopScopeParams::builder()
            .handle_uuid(&turn.uuid)
            .build(),
    )
    .unwrap();

    let spans = finish_trace_subscriber(&subscriber, &subscriber_name, &exporter);
    assert!(
        spans
            .iter()
            .all(|span| span.name.as_ref() != "late-subscriber-turn")
    );
    let llm_span = finished_span_named(&spans, "late-subscriber-llm");
    assert_traceparent_matches_exported_span(&traceparent, llm_span);
    assert_eq!(llm_span.span_context.trace_id(), relay_trace_id(turn.uuid));
    assert_eq!(llm_span.parent_span_id, relay_span_id(turn.uuid));
    assert!(llm_span.parent_span_is_remote);
}

#[test]
fn subscriber_registered_inside_managed_llm_preserves_parent_context() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    set_thread_scope_stack(create_scope_stack());
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let (provider, exporter) = make_provider();
    let subscriber =
        OpenTelemetrySubscriber::from_tracer_provider(provider, "callback-late-subscriber")
            .with_host(crate::observability_runtime::runtime_host());
    let subscriber_name = format!("callback_late_subscriber_{}", Uuid::now_v7().simple());
    let subscriber_for_callback = subscriber.clone();
    let subscriber_name_for_callback = subscriber_name.clone();
    let captured = Arc::new(Mutex::new(None));
    let captured_for_callback = Arc::clone(&captured);

    runtime
        .block_on(llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("unobserved-outer-llm")
                .request(LlmRequest {
                    headers: serde_json::Map::new(),
                    content: json!({"prompt": "outer"}),
                })
                .func(Arc::new(move |request| {
                    let subscriber = subscriber_for_callback.clone();
                    let subscriber_name = subscriber_name_for_callback.clone();
                    let captured = Arc::clone(&captured_for_callback);
                    Box::pin(async move {
                        let outer_traceparent = request
                            .headers
                            .get("traceparent")
                            .and_then(Json::as_str)
                            .map(ToOwned::to_owned)
                            .expect("managed outer LLM request must contain traceparent");
                        let propagation = capture_propagation_context().unwrap();
                        assert_eq!(
                            propagation.traceparent.as_deref(),
                            Some(outer_traceparent.as_str())
                        );
                        let managed_parent = ScopeHandle::builder()
                            .uuid(propagation.parent_uuid)
                            .name("unobserved-outer-llm")
                            .scope_type(ScopeType::Llm)
                            .build();
                        subscriber.register(&subscriber_name).unwrap();
                        let nested_traceparent = execute_llm_and_capture_trace_context_with_parent(
                            "observed-nested-llm",
                            Some(managed_parent),
                        )
                        .await
                        .0;
                        *captured.lock().unwrap() = Some((outer_traceparent, nested_traceparent));
                        Ok(json!({"ok": true}))
                    })
                }))
                .build(),
        ))
        .unwrap();

    let (outer_traceparent, nested_traceparent) = captured
        .lock()
        .unwrap()
        .take()
        .expect("managed callbacks must capture both traceparents");
    let spans = finish_trace_subscriber(&subscriber, &subscriber_name, &exporter);
    assert!(
        spans
            .iter()
            .all(|span| span.name.as_ref() != "unobserved-outer-llm")
    );
    let nested_span = finished_span_named(&spans, "observed-nested-llm");
    assert_traceparent_matches_exported_span(&nested_traceparent, nested_span);
    let outer_context = nemo_relay_runtime::api::runtime::scope_stack::w3c_span_context(
        Some(&outer_traceparent),
        None,
    )
    .expect("managed outer LLM traceparent must be valid");
    assert_eq!(
        nested_span.span_context.trace_id(),
        outer_context.trace_id()
    );
    assert_eq!(nested_span.parent_span_id, outer_context.span_id());
    assert!(nested_span.parent_span_is_remote);
}

#[test]
fn invalid_parent_span_id_does_not_split_llm_or_disable_later_exports() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    reset_global();
    set_thread_scope_stack(create_scope_stack());
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "invalid-parent")
        .with_host(crate::observability_runtime::runtime_host());
    let subscriber_name = format!("invalid_parent_{}", Uuid::now_v7().simple());
    subscriber.register(&subscriber_name).unwrap();

    let valid_ancestor = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("valid-ancestor")
            .scope_type(ScopeType::Custom)
            .build(),
    )
    .unwrap();
    let invalid_parent = ScopeHandle::builder()
        .uuid(Uuid::from_u128(0x018f_13f0_7c1a_7a80_0000_0000_0000_0000))
        .name("invalid-span-id-parent")
        .scope_type(ScopeType::Custom)
        .build();
    task_scope_push(invalid_parent.clone());
    let traceparent = runtime.block_on(execute_llm_and_capture_traceparent("invalid-parent-llm"));
    let explicit_child = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("explicit-child-skipping-invalid-scope")
            .scope_type(ScopeType::Function)
            .parent(&valid_ancestor)
            .build(),
    )
    .unwrap();
    let nested_traceparent =
        runtime.block_on(execute_llm_and_capture_traceparent("explicit-child-llm"));
    pop_scope(
        nemo_relay_runtime::api::scope::PopScopeParams::builder()
            .handle_uuid(&explicit_child.uuid)
            .build(),
    )
    .unwrap();
    let explicit_traceparent = runtime.block_on(execute_llm_and_capture_trace_context_with_parent(
        "explicit-valid-ancestor-llm",
        Some(valid_ancestor.clone()),
    ));
    let child = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("invalid-parent-child")
            .scope_type(ScopeType::Function)
            .build(),
    )
    .unwrap();
    pop_scope(
        nemo_relay_runtime::api::scope::PopScopeParams::builder()
            .handle_uuid(&child.uuid)
            .build(),
    )
    .unwrap();
    task_scope_remove(&invalid_parent.uuid).unwrap();
    let invalid_agent = ScopeHandle::builder()
        .uuid(Uuid::from_u128(0x018f_13f0_7c1a_7a81_0000_0000_0000_0000))
        .name("invalid-span-id-agent")
        .scope_type(ScopeType::Agent)
        .build();
    task_scope_push(invalid_agent.clone());
    let invalid_agent_traceparent = runtime.block_on(execute_llm_and_capture_trace_context(
        "invalid-agent-parent-llm",
    ));
    task_scope_remove(&invalid_agent.uuid).unwrap();
    let unpushed_traceparent = runtime.block_on(execute_llm_and_capture_trace_context_with_parent(
        "unpushed-invalid-parent-llm",
        Some(invalid_parent),
    ));

    let later = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("later-valid-scope")
            .scope_type(ScopeType::Custom)
            .build(),
    )
    .unwrap();
    pop_scope(
        nemo_relay_runtime::api::scope::PopScopeParams::builder()
            .handle_uuid(&later.uuid)
            .build(),
    )
    .unwrap();
    pop_scope(
        nemo_relay_runtime::api::scope::PopScopeParams::builder()
            .handle_uuid(&valid_ancestor.uuid)
            .build(),
    )
    .unwrap();

    let spans = finish_trace_subscriber(&subscriber, &subscriber_name, &exporter);
    let llm_span = finished_span_named(&spans, "invalid-parent-llm");
    assert_traceparent_matches_exported_span(&traceparent, llm_span);
    assert_eq!(llm_span.parent_span_id, SpanId::INVALID);
    assert!(!llm_span.parent_span_is_remote);
    let ancestor_span = finished_span_named(&spans, "valid-ancestor");
    let explicit_child_span = finished_span_named(&spans, "explicit-child-skipping-invalid-scope");
    let nested_span = finished_span_named(&spans, "explicit-child-llm");
    assert_traceparent_matches_exported_span(&nested_traceparent, nested_span);
    assert_eq!(
        explicit_child_span.span_context.trace_id(),
        ancestor_span.span_context.trace_id()
    );
    assert_eq!(
        nested_span.span_context.trace_id(),
        ancestor_span.span_context.trace_id()
    );
    assert_eq!(
        nested_span.parent_span_id,
        explicit_child_span.span_context.span_id()
    );
    let explicit_span = finished_span_named(&spans, "explicit-valid-ancestor-llm");
    assert_traceparent_matches_exported_span(&explicit_traceparent.0, explicit_span);
    assert_eq!(explicit_traceparent.2, explicit_traceparent.0);
    assert_eq!(
        explicit_span.parent_span_id,
        ancestor_span.span_context.span_id()
    );
    assert_eq!(
        explicit_span.span_context.trace_id(),
        ancestor_span.span_context.trace_id()
    );
    let unpushed_span = finished_span_named(&spans, "unpushed-invalid-parent-llm");
    assert_traceparent_matches_exported_span(&unpushed_traceparent.0, unpushed_span);
    assert_eq!(unpushed_traceparent.2, unpushed_traceparent.0);
    assert_eq!(unpushed_span.parent_span_id, SpanId::INVALID);
    let invalid_agent_span = finished_span_named(&spans, "invalid-agent-parent-llm");
    assert_traceparent_matches_exported_span(&invalid_agent_traceparent.0, invalid_agent_span);
    assert_eq!(invalid_agent_traceparent.2, invalid_agent_traceparent.0);
    assert_eq!(invalid_agent_span.parent_span_id, SpanId::INVALID);
    finished_span_named(&spans, "invalid-parent-child");
    finished_span_named(&spans, "later-valid-scope");
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
        traceparent: None,
        tracestate: None,
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
    event.set_propagation_parent_uuid(Some(parent_uuid));
    let parent_context =
        processor.parent_context(&crate::observability_runtime::event_view(&event));
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
        traceparent: None,
        tracestate: None,
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
    processor.process(&crate::observability_runtime::event_view(&start));
    processor.process(&crate::observability_runtime::event_view(&make_end_event(
        local_uuid,
        Some(parent_uuid),
        "receiver-tool",
        ScopeType::Tool,
        None,
    )));
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
        traceparent: None,
        tracestate: None,
    })
    .unwrap();
    set_thread_scope_stack(imported_stack);

    let captured = capture_propagation_context().unwrap();
    assert_eq!(captured.root_uuid, Some(root_uuid));
    assert_eq!(captured.parent_uuid, parent_uuid);
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
fn subscriber_registration_and_provider_lifecycle_methods_work() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_global();

    let (provider, _exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "test-scope")
        .with_host(crate::observability_runtime::runtime_host());
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
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "e2e-scope")
        .with_host(crate::observability_runtime::runtime_host());
    let name = format!("otel_e2e_{}", Uuid::now_v7().simple());

    subscriber.register(&name).unwrap();
    let handle = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("otel_scope")
            .scope_type(ScopeType::Agent)
            .data(json!({"scope": true}))
            .metadata(json!({"phase": "start"}))
            .input(json!({"task": "scope-start"}))
            .build(),
    )
    .unwrap();
    event(
        nemo_relay_runtime::api::scope::EmitMarkEventParams::builder()
            .name("otel_mark")
            .parent(&handle)
            .data(json!({"step": 1}))
            .metadata(json!({"source": "rust-test"}))
            .build(),
    )
    .unwrap();
    pop_scope(
        nemo_relay_runtime::api::scope::PopScopeParams::builder()
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
fn http_config_exports_scope_push_pop_and_marks_without_tokio_runtime() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
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
    let subscriber = OpenTelemetrySubscriber::new(config)
        .map(|exporter| exporter.with_host(crate::observability_runtime::runtime_host()))
        .unwrap();
    unsafe { std::env::set_var(&variable, "Bearer changed-secret") };
    let name = format!("otel_http_{}", Uuid::now_v7().simple());

    subscriber.register(&name).unwrap();
    let handle = push_scope(
        nemo_relay_runtime::api::scope::PushScopeParams::builder()
            .name("otel_scope")
            .scope_type(ScopeType::Agent)
            .data(json!({"scope": true}))
            .input(json!({"task": "http-start"}))
            .build(),
    )
    .unwrap();
    event(
        nemo_relay_runtime::api::scope::EmitMarkEventParams::builder()
            .name("otel_mark")
            .parent(&handle)
            .data(json!({"step": 1}))
            .metadata(json!({"source": "rust-http"}))
            .build(),
    )
    .unwrap();
    pop_scope(
        nemo_relay_runtime::api::scope::PopScopeParams::builder()
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
fn exported_logs_join_trace_spans_with_unobserved_local_parents_and_late_marks() {
    use opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest;

    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_global();
    for projection in [
        OpenTelemetryType::Full,
        OpenTelemetryType::GenAi,
        OpenTelemetryType::OpenInference,
    ] {
        for imported in [false, true] {
            let traces_listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let logs_listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let trace_endpoint =
                format!("http://{}/v1/traces", traces_listener.local_addr().unwrap());
            let log_endpoint = format!("http://{}/v1/logs", logs_listener.local_addr().unwrap());
            let (trace_tx, trace_rx) = mpsc::channel();
            let (log_tx, log_rx) = mpsc::channel();
            spawn_http_collector(traces_listener, trace_tx);
            spawn_http_collector(logs_listener, log_tx);
            let traces = OpenTelemetrySubscriber::new(
                OpenTelemetryConfig::new(projection, trace_endpoint)
                    .with_scheduled_delay(Duration::from_secs(60)),
            )
            .map(|exporter| exporter.with_host(crate::observability_runtime::runtime_host()))
            .unwrap();
            let logs = OpenTelemetryLogSubscriber::new(OpenTelemetryLogConfig::new(log_endpoint))
                .map(|exporter| exporter.with_host(crate::observability_runtime::runtime_host()))
                .unwrap();
            let trace_callback = traces.subscriber();
            let log_callback = logs.subscriber();
            let root = Uuid::now_v7();
            let missing_parent = Uuid::now_v7();
            let turn = Uuid::now_v7();
            let tool = Uuid::now_v7();
            let mut start =
                make_start_event(turn, Some(missing_parent), "turn", ScopeType::Custom, None);
            start.set_propagation_root_uuid(Some(root));
            if imported {
                start.set_propagation_parent_uuid(Some(missing_parent));
            }
            let tool_start =
                make_start_event(tool, Some(turn), "ordinary-tool", ScopeType::Tool, None);
            let tool_end = make_end_event(tool, Some(turn), "ordinary-tool", ScopeType::Tool, None);
            let turn_end =
                make_end_event(turn, Some(missing_parent), "turn", ScopeType::Custom, None);
            let events = [
                start,
                tool_start,
                make_mark_event(Some(tool), "nv.agent.tool.start", None),
                make_mark_event(Some(turn), "plugin.custom.mark", None),
                tool_end,
                turn_end,
                make_mark_event(Some(tool), "late.tool.mark", None),
                make_mark_event(Some(turn), "late.turn.mark", None),
            ];
            for event in &events {
                trace_callback(event);
                log_callback(event);
            }
            traces.force_flush().unwrap();
            logs.force_flush().unwrap();
            let trace_request = trace_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let log_request = log_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let trace_export =
                ExportTraceServiceRequest::decode(trace_request.body.as_slice()).unwrap();
            let log_export = ExportLogsServiceRequest::decode(log_request.body.as_slice()).unwrap();
            let spans: Vec<_> = trace_export
                .resource_spans
                .iter()
                .flat_map(|r| &r.scope_spans)
                .flat_map(|s| &s.spans)
                .collect();
            let records: Vec<_> = log_export
                .resource_logs
                .iter()
                .flat_map(|r| &r.scope_logs)
                .flat_map(|s| &s.log_records)
                .collect();
            assert_eq!(records.len(), 4);
            let expected_trace = relay_trace_id(if imported { root } else { turn }).to_bytes();
            for record in records {
                assert_eq!(
                    record.trace_id, expected_trace,
                    "{projection:?}, imported={imported}"
                );
                assert!(
                    spans
                        .iter()
                        .any(|span| span.trace_id == record.trace_id
                            && span.span_id == record.span_id),
                    "every log must join an exported span: {projection:?}, imported={imported}"
                );
            }
            traces.shutdown().unwrap();
            logs.shutdown().unwrap();
        }
    }
}

#[test]
fn http_trace_exports_do_not_follow_redirects() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_global();

    for otel_type in [
        OpenTelemetryType::Full,
        OpenTelemetryType::GenAi,
        OpenTelemetryType::OpenInference,
    ] {
        for status in [307, 308] {
            let destination = TcpListener::bind("127.0.0.1:0").unwrap();
            destination.set_nonblocking(true).unwrap();
            let location = format!("http://{}/leak", destination.local_addr().unwrap());
            let redirector = TcpListener::bind("127.0.0.1:0").unwrap();
            let endpoint = format!("http://{}/v1/traces", redirector.local_addr().unwrap());
            let server = thread::spawn(move || {
                let (mut stream, _) = redirector.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let request = read_http_request(&mut stream);
                write!(stream, "HTTP/1.1 {status} Redirect\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
                request
            });
            let config = OpenTelemetryConfig::new(otel_type, endpoint)
                .with_timeout(Duration::from_secs(1))
                .with_header("x-collector-key", "test-secret");
            let subscriber = OpenTelemetrySubscriber::new(config)
                .map(|exporter| exporter.with_host(crate::observability_runtime::runtime_host()))
                .unwrap();
            let callback = subscriber.subscriber();
            let uuid = Uuid::now_v7();
            callback(&make_start_event(
                uuid,
                None,
                "redirect-test",
                ScopeType::Agent,
                None,
            ));
            callback(&make_end_event(
                uuid,
                None,
                "redirect-test",
                ScopeType::Agent,
                None,
            ));
            assert!(
                subscriber.force_flush().is_err(),
                "redirect must fail export"
            );
            let request = server.join().unwrap();
            assert!(
                !request.body.is_empty(),
                "initial collector must receive the export"
            );
            assert!(
                matches!(destination.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock),
                "redirect destination must receive no connection, payload, or credentials"
            );
            subscriber.shutdown().unwrap();
        }
    }
}

#[test]
fn http_trace_subscriber_drains_accepted_events_on_drop() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_global();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1/traces", listener.local_addr().unwrap());
    let (request_tx, request_rx) = mpsc::channel();
    spawn_http_collector(listener, request_tx);
    let subscriber = OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::new(OpenTelemetryType::Full, endpoint)
            .with_scheduled_delay(Duration::from_secs(60)),
    )
    .map(|exporter| exporter.with_host(crate::observability_runtime::runtime_host()))
    .unwrap();
    let name = format!("drop-drain-{}", Uuid::now_v7().simple());
    subscriber.register(&name).unwrap();
    event(
        nemo_relay_runtime::api::scope::EmitMarkEventParams::builder()
            .name("drop-drain")
            .build(),
    )
    .unwrap();
    subscriber.deregister(&name).unwrap();

    drop(subscriber);

    let request = request_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("dropping the subscriber must wait for its accepted trace export");
    let traces = ExportTraceServiceRequest::decode(request.body.as_slice()).unwrap();
    assert_eq!(traces.resource_spans[0].scope_spans[0].spans.len(), 1);
}

#[test]
fn direct_gen_ai_and_openinference_configs_export_typed_otlp_payloads() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
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
        .map(|exporter| exporter.with_host(crate::observability_runtime::runtime_host()))
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
fn dropped_spans_are_recorded_in_the_active_plugin_report() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let _ = crate::test_support::plugin_host::test_close_plugin_host();
    let _clear_guard = ClosePluginHostGuard;
    futures::executor::block_on(
        crate::test_support::plugin_host::test_initialize_plugin_host_exact(
            nemo_relay_plugin_host::PluginConfig::default(),
        ),
    )
    .unwrap();

    let runtime_diagnostics =
        SignalRuntimeDiagnostics::new(Some("opentelemetry.traces[2].endpoint".to_string()))
            .with_host(crate::observability_test_support::core_host());
    let processor = DiagnosticBatchSpanProcessor::new_with_batch_config(
        InMemorySpanExporterBuilder::new().build(),
        "https://collector.example/v1/traces".to_string(),
        runtime_diagnostics.clone(),
        BatchConfigBuilder::default().build(),
    );
    processor.completed_spans.store(3, Ordering::Relaxed);
    processor.accepted_spans.store(1, Ordering::Relaxed);
    processor.force_flush().unwrap();

    let report = crate::test_support::plugin_host::test_plugin_host_report().unwrap();
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
fn plugin_trace_subscriber_runtime_diagnostics_use_trace_field() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _ = crate::test_support::plugin_host::test_close_plugin_host();
    let _clear_guard = ClosePluginHostGuard;
    futures::executor::block_on(
        crate::test_support::plugin_host::test_initialize_plugin_host_exact(
            nemo_relay_plugin_host::PluginConfig::default(),
        ),
    )
    .unwrap();

    let subscriber = OpenTelemetrySubscriber::new_for_plugin(
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces"),
        2,
    )
    .map(|exporter| exporter.with_host(crate::observability_runtime::runtime_host()))
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

    let report = crate::test_support::plugin_host::test_plugin_host_report().unwrap();
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
fn unrecovered_trace_export_failure_is_retained_in_the_active_plugin_report() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let _ = crate::test_support::plugin_host::test_close_plugin_host();
    let _clear_guard = ClosePluginHostGuard;
    futures::executor::block_on(
        crate::test_support::plugin_host::test_initialize_plugin_host_exact(
            nemo_relay_plugin_host::PluginConfig::default(),
        ),
    )
    .unwrap();

    let runtime_diagnostics =
        SignalRuntimeDiagnostics::new(Some("opentelemetry.traces[2].endpoint".to_string()))
            .with_host(crate::observability_test_support::core_host());
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

    let report = crate::test_support::plugin_host::test_plugin_host_report().unwrap();
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
    let attributes =
        attr_map(&crate::observability_test_support::otel_genai::start_attributes(&request));
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
        attr_map(&crate::observability_test_support::otel_genai::end_attributes(&response));
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
    let response_attributes = attr_map(
        &crate::observability_test_support::otel_genai::end_attributes(
            &response_without_finish_reason,
        ),
    );
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
        let attributes =
            attr_map(&crate::observability_test_support::otel_genai::end_attributes(&event));
        assert_eq!(
            attributes.get("gen_ai.response.finish_reasons"),
            Some(&format!("[\"{expected}\"]"))
        );
    }
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

    let attributes =
        attr_map(&crate::observability_test_support::otel_genai::end_attributes(&event));
    assert_eq!(
        attributes.get("gen_ai.usage.input_tokens"),
        Some(&"27403".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.usage.cache_read.input_tokens"),
        Some(&"17980".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.usage.cache_write.input_tokens"),
        Some(&"9421".to_string())
    );
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
        crate::observability_test_support::otel_genai::span_name(&agent),
        "invoke_agent semantic-agent"
    );

    let mut tool = make_start_event(Uuid::now_v7(), None, "fallback-tool", ScopeType::Tool, None);
    if let Event::Scope(scope) = &mut tool {
        scope.base.metadata = Some(json!({"gen_ai.tool.name": "semantic-tool"}));
    }
    assert_eq!(
        crate::observability_test_support::otel_genai::span_name(&tool),
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
    let attributes =
        attr_map(&crate::observability_test_support::otel_genai::start_attributes(&llm));
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
    let attributes =
        attr_map(&crate::observability_test_support::otel_genai::start_attributes(&anthropic));
    assert_eq!(
        attributes.get("gen_ai.request.top_k"),
        Some(&"12".to_string())
    );

    for (name, expected) in [
        ("azure_openai.chat", "azure.ai.openai"),
        ("azure_ai_inference.chat", "azure.ai.inference"),
        ("azure", "azure.ai.openai"),
        ("azure/openai", "azure.ai.openai"),
        ("azure_ai", "azure.ai.inference"),
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
        let attributes =
            attr_map(&crate::observability_test_support::otel_genai::start_attributes(&event));
        assert_eq!(
            attributes.get("gen_ai.provider.name"),
            Some(&expected.to_string())
        );
    }
}

#[test]
fn llm_end_emits_cost_only_no_token_or_gen_ai_attributes() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    install_openai_disambiguation_pricing("priced-model");
    let _reset_guard = ResetPricingResolverGuard;

    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());
    let uuid = Uuid::now_v7();

    processor.process(&crate::observability_runtime::event_view(
        &make_start_event(uuid, None, "other", ScopeType::Llm, None),
    ));
    processor.process(&crate::observability_runtime::event_view(&make_end_event(
        uuid,
        None,
        "other",
        ScopeType::Llm,
        Some(openai_chat_provider_response("priced-model")),
    )));
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
fn propagated_children_and_marks_preserve_lineage_after_completed_context_expiry() {
    for w3c in [false, true] {
        let (provider, exporter) = make_provider();
        let mut processor = OtelEventProcessor::new(provider, "test-scope".to_string());
        let parent_uuid = Uuid::now_v7();
        let child_uuid = Uuid::now_v7();
        let closed_at = chrono::Utc::now();
        let scope_event = |uuid, parent, name, category, timestamp| {
            Event::Scope(ScopeEvent::new(
                BaseEvent::builder()
                    .uuid(uuid)
                    .parent_uuid_opt(parent)
                    .name(name)
                    .timestamp(timestamp)
                    .build(),
                category,
                Vec::new(),
                ScopeType::Tool.into(),
                None,
            ))
        };
        for category in [ScopeCategory::Start, ScopeCategory::End] {
            processor.process(&crate::observability_runtime::event_view(&scope_event(
                parent_uuid,
                None,
                "parent",
                category,
                closed_at,
            )));
        }
        let context = processor.completed_span_contexts[&parent_uuid]
            .span_context
            .clone();
        let propagate = |mut event: Event| {
            event.set_propagation_parent_uuid(Some(parent_uuid));
            if w3c {
                event.set_propagation_traceparent(Some(format!(
                    "00-{}-{}-01",
                    context.trace_id(),
                    context.span_id()
                )));
            } else {
                event.set_propagation_root_uuid(Some(parent_uuid));
            }
            event
        };
        let after_ttl = closed_at + chrono::Duration::seconds(61);
        processor.process(&crate::observability_runtime::event_view(&propagate(
            Event::Mark(MarkEvent::new(
                BaseEvent::builder()
                    .parent_uuid(parent_uuid)
                    .name("late-mark")
                    .timestamp(after_ttl)
                    .build(),
                None,
                None,
            )),
        )));
        assert!(!processor.completed_span_contexts.contains_key(&parent_uuid));
        for category in [ScopeCategory::Start, ScopeCategory::End] {
            processor.process(&crate::observability_runtime::event_view(&propagate(
                scope_event(
                    child_uuid,
                    Some(parent_uuid),
                    "late-child",
                    category,
                    after_ttl,
                ),
            )));
        }
        processor.force_flush().unwrap();
        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 3);
        let parent = finished_span_named(&spans, "parent");
        for name in ["mark:late-mark", "late-child"] {
            let span = finished_span_named(&spans, name);
            assert_eq!(span.span_context.trace_id(), parent.span_context.trace_id());
            assert_eq!(span.parent_span_id, parent.span_context.span_id());
        }
    }
}

#[test]
fn propagated_w3c_parent_projects_trace_flags_and_tracestate() {
    let parent_uuid = Uuid::now_v7();
    let mut event = make_start_event(
        Uuid::now_v7(),
        Some(parent_uuid),
        "receiver-tool",
        ScopeType::Tool,
        None,
    );
    event.set_propagation_parent_uuid(Some(parent_uuid));
    event.set_propagation_traceparent(Some(
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00".to_string(),
    ));
    event.set_propagation_tracestate(Some("vendor=value".to_string()));

    for otel_type in [
        OpenTelemetryType::Full,
        OpenTelemetryType::GenAi,
        OpenTelemetryType::OpenInference,
    ] {
        let processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
                make_provider().0,
                "test".into(),
                otel_type,
                MarkProjection::default(),
                default_mark_exclude_names(),
                Vec::new(),
                Vec::new(),
                SignalRuntimeDiagnostics::new(None).with_host(crate::observability_test_support::core_host()),
            );
        let span_context = processor
            .parent_context(&crate::observability_runtime::event_view(&event))
            .span()
            .span_context()
            .clone();
        assert!(span_context.is_remote());
        assert_eq!(
            span_context.trace_id().to_string(),
            "4bf92f3577b34da6a3ce929d0e0e4736"
        );
        assert_eq!(span_context.span_id().to_string(), "00f067aa0ba902b7");
        assert!(!span_context.is_sampled());
        assert_eq!(span_context.trace_state().header(), "vendor=value");
    }
}
