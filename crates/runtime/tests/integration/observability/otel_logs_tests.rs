// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for Relay OTLP log projection.

use super::*;
use crate::api::event::{
    BaseEvent, CategoryProfile, DataSchema, EventCategory, METRIC_DATA_SCHEMA_NAME,
    METRIC_DATA_SCHEMA_VERSION, MarkEvent, ScopeCategory, ScopeEvent,
};
use crate::api::scope::ScopeType;
use opentelemetry_sdk::logs::{InMemoryLogExporter, SdkLoggerProvider};
use serde_json::json;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

fn mark(
    parent_uuid: Option<Uuid>,
    name: &str,
    data: Option<Json>,
    data_schema: Option<DataSchema>,
    metadata: Option<Json>,
) -> Event {
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .name(name)
            .data_opt(data)
            .data_schema_opt(data_schema)
            .metadata_opt(metadata)
            .build(),
        None,
        None,
    ))
}

fn scope_with_parent(uuid: Uuid, parent_uuid: Option<Uuid>, category: ScopeCategory) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(uuid)
            .parent_uuid_opt(parent_uuid)
            .name("agent")
            .build(),
        category,
        Vec::new(),
        ScopeType::Agent.into(),
        None,
    ))
}

fn processor(
    minimum_severity: LogSeverity,
) -> (LogEventProcessor, InMemoryLogExporter, SdkLoggerProvider) {
    let exporter = InMemoryLogExporter::default();
    let provider = SdkLoggerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let logger = provider.logger("nemo-relay-test");
    (
        LogEventProcessor::new(logger, minimum_severity, None),
        exporter,
        provider,
    )
}

#[test]
fn tool_lifecycle_subscriber_marks_export_the_tool_span_context() {
    use crate::api::runtime::{create_scope_stack, task_scope_top, with_scope_stack};
    use crate::api::scope::{EmitMarkEventParams, PopScopeParams, PushScopeParams};
    use crate::api::subscriber::{flush_subscribers, scope_register_subscriber};
    use std::sync::{Condvar, Mutex};

    let (processor, exporter, provider) = processor(LogSeverity::Info);
    let processor = Arc::new(Mutex::new(processor));
    let delivery_gate = Arc::new((Mutex::new(false), Condvar::new()));
    let callback_gate = Arc::clone(&delivery_gate);
    let stack = create_scope_stack();
    let (turn, tools) = with_scope_stack(stack, || {
        scope_register_subscriber(
            &task_scope_top().uuid,
            "tool-log-context",
            Arc::new(move |observed| {
                processor
                    .lock()
                    .unwrap()
                    .process(&crate::observability_runtime::event_view(observed));
                if observed.scope_type() == Some(ScopeType::Tool) {
                    let (closed, ready) = &*callback_gate;
                    drop(
                        ready
                            .wait_while(closed.lock().unwrap(), |closed| !*closed)
                            .unwrap(),
                    );
                    let name = match observed.scope_category().unwrap() {
                        ScopeCategory::Start => "nv.agent.tool.start",
                        ScopeCategory::End => "nv.agent.tool.end",
                    };
                    crate::api::scope::event(
                        EmitMarkEventParams::builder()
                            .name(name)
                            .data(json!({"gen_ai.tool.call.id": observed.uuid().to_string()}))
                            .build(),
                    )
                    .unwrap();
                }
            }),
        )
        .unwrap();
        let turn = crate::api::scope::push_scope(
            PushScopeParams::builder()
                .name("turn")
                .scope_type(ScopeType::Agent)
                .build(),
        )
        .unwrap();
        let mut tools = Vec::new();
        for _ in 0..2 {
            let tool = crate::api::tool::tool_call(
                crate::api::tool::ToolCallParams::builder()
                    .name("Bash")
                    .args(json!({"command": "true"}))
                    .build(),
            )
            .unwrap();
            crate::api::tool::tool_call_end(
                crate::api::tool::ToolCallEndParams::builder()
                    .handle(&tool)
                    .execution_result(json!({"exit_code": 0}).into())
                    .build(),
            )
            .unwrap();
            tools.push(tool.uuid);
        }
        crate::api::scope::event(EmitMarkEventParams::builder().name("turn.mark").build()).unwrap();
        crate::api::scope::pop_scope(PopScopeParams::builder().handle_uuid(&turn.uuid).build())
            .unwrap();
        // Release callbacks only after both tools and their turn have closed.
        let (closed, ready) = &*delivery_gate;
        *closed.lock().unwrap() = true;
        ready.notify_all();
        flush_subscribers().unwrap();
        (turn.uuid, tools)
    });
    provider.force_flush().unwrap();
    let logs = exporter.get_emitted_logs().unwrap();
    assert_eq!(logs.len(), 5);
    for tool in tools {
        let matching = logs.iter().filter(|log| {
            matches!(log.record.body(), Some(AnyValue::Map(body))
                if body.get(&Key::new("gen_ai.tool.call.id")) == Some(&AnyValue::from(tool.to_string())))
        }).collect::<Vec<_>>();
        assert_eq!(matching.len(), 2, "each tool has a start and end log");
        for log in matching {
            let context = log.record.trace_context().unwrap();
            assert_eq!(context.span_id, relay_span_id(tool));
            assert_ne!(context.span_id, relay_span_id(turn));
            assert_eq!(context.trace_id, relay_trace_id(turn));
        }
    }
    let turn_log = logs.iter().find(|log| log.record.body().is_none()).unwrap();
    assert_eq!(
        turn_log.record.trace_context().unwrap().span_id,
        relay_span_id(turn)
    );
}

// Serves one OTLP/HTTP request on `listener` and hands back body.

#[test]
fn log_projection_preserves_imported_w3c_trace_identity_and_sampling_flag() {
    let (mut processor, exporter, provider) = processor(LogSeverity::Info);
    let upstream_parent = Uuid::now_v7();
    let local_scope = Uuid::now_v7();
    let mut imported = scope_with_parent(local_scope, Some(upstream_parent), ScopeCategory::Start);
    imported.set_propagation_parent_uuid(Some(upstream_parent));
    imported.set_propagation_traceparent(Some(
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00".to_string(),
    ));
    imported.set_propagation_tracestate(Some("vendor=value".to_string()));
    processor.process(&crate::observability_runtime::event_view(&imported));
    processor.process(&crate::observability_runtime::event_view(&mark(
        Some(local_scope),
        "relay.w3c-context",
        None,
        None,
        None,
    )));
    provider.force_flush().unwrap();

    let logs = exporter.get_emitted_logs().unwrap();
    let context = logs[0]
        .record
        .trace_context()
        .expect("imported W3C context should be attached to the log record");
    assert_eq!(
        context.trace_id.to_string(),
        "4bf92f3577b34da6a3ce929d0e0e4736"
    );
    assert_eq!(context.span_id, relay_span_id(local_scope));
    assert_eq!(context.trace_flags, Some(TraceFlags::default()));
}

#[test]
fn log_projection_uses_event_w3c_context_for_an_unobserved_local_parent() {
    let (mut processor, exporter, provider) = processor(LogSeverity::Info);
    let local_parent = Uuid::now_v7();
    let local_scope = Uuid::now_v7();
    let mut child = scope_with_parent(local_scope, Some(local_parent), ScopeCategory::Start);
    child.set_propagation_traceparent(Some(crate::observability_test_support::format_traceparent(
        local_parent,
        local_parent,
    )));
    processor.process(&crate::observability_runtime::event_view(&child));
    processor.process(&crate::observability_runtime::event_view(&mark(
        Some(local_scope),
        "relay.late-local-parent",
        None,
        None,
        None,
    )));
    provider.force_flush().unwrap();

    let logs = exporter.get_emitted_logs().unwrap();
    let context = logs[0]
        .record
        .trace_context()
        .expect("event-carried W3C context should attach the unobserved local parent");
    assert_eq!(context.trace_id, relay_trace_id(local_parent));
    assert_eq!(context.span_id, relay_span_id(local_scope));
    assert_eq!(context.trace_flags, Some(TraceFlags::SAMPLED));
}
