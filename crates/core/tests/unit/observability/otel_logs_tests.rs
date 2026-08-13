// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for Relay OTLP log projection.

use super::*;
use crate::api::event::{
    BaseEvent, DataSchema, METRIC_DATA_SCHEMA_NAME, MarkEvent, ScopeCategory, ScopeEvent,
};
use crate::api::scope::ScopeType;
use opentelemetry_sdk::logs::{InMemoryLogExporter, SdkLoggerProvider};
use serde_json::json;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

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

fn scope(uuid: Uuid, category: ScopeCategory) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder().uuid(uuid).name("agent").build(),
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
fn direct_log_subscriber_recovers_a_poisoned_processor_lock() {
    let subscriber =
        OpenTelemetryLogSubscriber::new(OpenTelemetryLogConfig::new("http://127.0.0.1:4318"))
            .unwrap();
    let processor = Arc::clone(&subscriber.inner._processor);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = processor.lock().unwrap();
            panic!("poison log processor");
        }))
        .is_err()
    );

    let uuid = Uuid::now_v7();
    (subscriber.subscriber())(&scope(uuid, ScopeCategory::Start));

    let Err(poisoned) = processor.lock() else {
        panic!("log processor lock should remain poisoned after recovery");
    };
    let processor = poisoned.into_inner();
    assert!(processor.lineage.active.contains_key(&uuid));
}

#[test]
fn signal_endpoint_resolution_replaces_trace_suffix_without_nested_paths() {
    for (input, expected) in [
        (
            "https://collector.example",
            "https://collector.example/v1/logs",
        ),
        (
            "https://collector.example/prefix/v1/traces?tenant=one",
            "https://collector.example/prefix/v1/logs?tenant=one",
        ),
        (
            "https://collector.example/custom",
            "https://collector.example/custom",
        ),
        ("https://collector.example/", "https://collector.example/"),
    ] {
        let resolved = resolve_http_log_endpoint(input);
        assert_eq!(resolved, expected);
        assert!(!resolved.contains("/v1/traces/v1/logs"));
    }
}

#[test]
fn log_delivery_state_reports_queue_and_export_failures_independently() {
    let diagnostics = LogDeliveryDiagnostics::new(
        "https://collector.example/v1/logs".to_string(),
        Some("opentelemetry.logs.endpoints[0].endpoint".to_string()),
    );
    diagnostics.emitted.store(3, Ordering::Relaxed);
    diagnostics.accepted.store(2, Ordering::Relaxed);
    diagnostics.export_failures.store(2, Ordering::Relaxed);

    assert_eq!(
        diagnostics.failure_summary().as_deref(),
        Some("otel.logs_dropped (1), otel.logs_export_failed (2)")
    );
}

#[test]
fn non_metric_mark_maps_structured_body_attributes_and_scope_context() {
    let (mut processor, exporter, provider) = processor(LogSeverity::Info);
    let parent_uuid = Uuid::now_v7();
    processor.process(&scope(parent_uuid, ScopeCategory::Start));
    processor.process(&mark(
        Some(parent_uuid),
        "tokens.estimated",
        Some(json!({"value": 42, "nested": null})),
        None,
        Some(json!({
            LOG_SEVERITY_METADATA_KEY: "warning",
            "tenant": "demo"
        })),
    ));
    provider.force_flush().unwrap();

    let logs = exporter.get_emitted_logs().unwrap();
    assert_eq!(logs.len(), 1);
    let record = &logs[0].record;
    assert_eq!(record.event_name(), None);
    assert_eq!(record.severity_number(), Some(Severity::Warn));
    assert_eq!(record.severity_text(), Some("WARN"));
    assert!(record.timestamp().is_some());
    assert!(record.observed_timestamp().is_some());
    let context = record.trace_context().expect("containing scope context");
    assert_eq!(context.span_id, relay_span_id(parent_uuid));
    assert!(matches!(record.body(), Some(AnyValue::Map(_))));
    let attributes = record
        .attributes_iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<HashMap<_, _>>();
    assert_eq!(
        attributes.get(&Key::new("nemo_relay.mark.name")),
        Some(&AnyValue::from("tokens.estimated"))
    );
    assert_eq!(
        attributes.get(&Key::new("nemo_relay.atof.version")),
        Some(&AnyValue::from(ATOF_VERSION))
    );
    assert_eq!(
        attributes.get(&Key::new("nemo_relay.mark.parent_uuid")),
        Some(&AnyValue::from(parent_uuid.to_string()))
    );
}

#[test]
fn routing_and_severity_filtering_drop_without_fallback() {
    let (mut processor, exporter, provider) = processor(LogSeverity::Warn);
    processor.process(&mark(None, "default-info", Some(json!(1)), None, None));
    processor.process(&mark(
        None,
        "invalid-severity",
        None,
        None,
        Some(json!({LOG_SEVERITY_METADATA_KEY: "notice"})),
    ));
    processor.process(&mark(
        None,
        "metric",
        Some(json!({"measurements": []})),
        Some(
            DataSchema::builder()
                .name(METRIC_DATA_SCHEMA_NAME)
                .version("999")
                .build(),
        ),
        Some(json!({LOG_SEVERITY_METADATA_KEY: "error"})),
    ));
    provider.force_flush().unwrap();
    assert!(exporter.get_emitted_logs().unwrap().is_empty());
    assert_eq!(processor.invalid_severity_count, 1);
    assert_eq!(processor.invalid_metric_count, 1);
}

#[test]
fn json_conversion_preserves_top_level_absence_and_nested_null() {
    assert_eq!(json_body(&Json::Null), None);
    let body = json_body(&json!([null, i64::MAX as u64 + 1])).unwrap();
    assert_eq!(
        body,
        AnyValue::ListAny(Box::new(vec![
            AnyValue::from("null"),
            AnyValue::from((i64::MAX as u64 + 1).to_string()),
        ]))
    );
}
