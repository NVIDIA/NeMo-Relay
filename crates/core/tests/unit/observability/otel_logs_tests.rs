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
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
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

fn scope(uuid: Uuid, category: ScopeCategory) -> Event {
    scope_with_parent(uuid, None, category)
}

fn scope_at(
    uuid: Uuid,
    category: ScopeCategory,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(uuid)
            .name("agent")
            .timestamp(timestamp)
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
fn scope_lineage_retains_active_contexts_and_preserves_root_trace_id() {
    let mut lineage = ScopeLineage::new();
    let root = Uuid::now_v7();
    lineage.process_start(&scope(root, ScopeCategory::Start));
    for _ in 0..4_096 {
        lineage.process_start(&scope(Uuid::now_v7(), ScopeCategory::Start));
    }

    assert_eq!(lineage.active.len(), 4_097);
    assert!(lineage.active.contains_key(&root));

    let child = Uuid::now_v7();
    lineage.process_start(&scope_with_parent(child, Some(root), ScopeCategory::Start));
    assert_eq!(lineage.active[&child].trace_id(), relay_trace_id(root));

    lineage.process_end(&scope(child, ScopeCategory::End));
    assert!(!lineage.active.contains_key(&child));
    assert!(lineage.completed.contains_key(&child));
}

#[test]
fn scope_lineage_reuses_completed_parent_within_ttl_and_expires_after_boundary() {
    let mut lineage = ScopeLineage::new();
    let parent = Uuid::now_v7();
    let closed_at = chrono::Utc::now();
    lineage.process_start(&scope_at(parent, ScopeCategory::Start, closed_at));
    lineage.process_end(&scope_at(parent, ScopeCategory::End, closed_at));

    let child = mark(Some(parent), "late.mark", None, None, None);
    let context = lineage
        .parent_context(&child)
        .expect("completed parent context");
    assert_eq!(context.span_id(), relay_span_id(parent));

    for _ in 0..4_098 {
        let uuid = Uuid::now_v7();
        lineage.process_start(&scope_at(uuid, ScopeCategory::Start, closed_at));
        lineage.process_end(&scope_at(uuid, ScopeCategory::End, closed_at));
    }
    assert!(lineage.completed.contains_key(&parent));
    assert_eq!(
        lineage.expire_completed(
            closed_at + chrono::Duration::seconds(60),
            Duration::from_secs(60)
        ),
        0
    );
    assert!(lineage.completed.contains_key(&parent));
    assert_eq!(
        lineage.expire_completed(
            closed_at + chrono::Duration::seconds(61),
            Duration::from_secs(60)
        ),
        4_099
    );
    assert!(lineage.parent_context(&child).is_none());

    lineage.process_end(&scope(Uuid::now_v7(), ScopeCategory::End));
}

#[test]
fn scope_lineage_replaces_completed_context_expiry_index_entries() {
    let mut lineage = ScopeLineage::new();
    let reused = Uuid::now_v7();
    let closed_at = chrono::Utc::now();
    lineage.process_start(&scope_at(reused, ScopeCategory::Start, closed_at));
    lineage.process_end(&scope_at(reused, ScopeCategory::End, closed_at));
    lineage.process_start(&scope_at(
        reused,
        ScopeCategory::Start,
        closed_at + chrono::Duration::seconds(1),
    ));
    lineage.process_end(&scope_at(
        reused,
        ScopeCategory::End,
        closed_at + chrono::Duration::seconds(1),
    ));

    assert!(lineage.completed.contains_key(&reused));
    assert_eq!(lineage.completed_expiry_index.len(), 1);
}

#[test]
fn log_config_rejects_zero_timeout() {
    let error = OpenTelemetryLogConfig::new("https://collector.example/v1/logs")
        .with_timeout(Duration::ZERO)
        .validate()
        .unwrap_err();

    assert!(error.to_string().contains("timeout must be greater than 0"));
}

#[test]
fn log_config_rejects_blank_and_padded_headers() {
    for (key, value) in [
        ("", "token"),
        (" authorization", "token"),
        ("authorization", ""),
        ("authorization", " token "),
    ] {
        let error = OpenTelemetryLogConfig::new("https://collector.example/v1/logs")
            .with_header(key, value)
            .validate()
            .unwrap_err();
        assert!(error.to_string().contains("surrounding whitespace"));
    }
}

#[test]
fn log_config_validates_batch_limits_and_retains_resource_identity() {
    for config in [
        OpenTelemetryLogConfig::new("   "),
        OpenTelemetryLogConfig::new("https://collector.example/v1/logs").with_max_queue_size(0),
        OpenTelemetryLogConfig::new("https://collector.example/v1/logs")
            .with_max_queue_size(1)
            .with_max_export_batch_size(2),
        OpenTelemetryLogConfig::new("https://collector.example/v1/logs")
            .with_scheduled_delay(Duration::ZERO),
    ] {
        assert!(config.validate().is_err());
    }

    let config = OpenTelemetryLogConfig::new("https://collector.example/v1/logs")
        .with_service_namespace("relay")
        .with_service_version("0.8.0")
        .with_resource_attribute("deployment.environment", "test");
    assert_eq!(config.service_namespace.as_deref(), Some("relay"));
    assert_eq!(config.service_version.as_deref(), Some("0.8.0"));
    assert_eq!(
        config.resource_attributes.get("deployment.environment"),
        Some(&"test".to_string())
    );
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
fn direct_log_subscriber_exposes_runtime_diagnostics() {
    let subscriber =
        OpenTelemetryLogSubscriber::new(OpenTelemetryLogConfig::new("http://127.0.0.1:4318"))
            .unwrap();
    let event = mark(
        None,
        "invalid-metric",
        Some(json!({"measurements": []})),
        Some(
            DataSchema::builder()
                .name(METRIC_DATA_SCHEMA_NAME)
                .version("999")
                .build(),
        ),
        None,
    );

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
        (
            "https://collector.example/",
            "https://collector.example/v1/logs",
        ),
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
        SignalRuntimeDiagnostics::new(Some("opentelemetry.logs.endpoints[0].endpoint".to_string())),
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
fn direct_log_processor_records_cumulative_queue_drops_on_flush_and_shutdown() {
    let runtime_diagnostics = SignalRuntimeDiagnostics::new(None);
    let delivery_diagnostics = Arc::new(LogDeliveryDiagnostics::new(
        "https://collector.example/v1/logs".to_string(),
        runtime_diagnostics.clone(),
    ));
    delivery_diagnostics.emitted.store(3, Ordering::Relaxed);
    delivery_diagnostics.accepted.store(1, Ordering::Relaxed);
    let processor = DiagnosticBatchLogProcessor {
        inner: BatchLogProcessor::builder(InMemoryLogExporter::default()).build(),
        diagnostics: Arc::clone(&delivery_diagnostics),
    };

    processor.force_flush().unwrap();

    let diagnostics = runtime_diagnostics.snapshot();
    let diagnostic = diagnostics
        .get("otel.logs_dropped")
        .expect("direct subscriber flush drop diagnostic");
    assert_eq!(diagnostic.count, 2);
    assert!(
        diagnostic
            .message
            .contains("https://collector.example/v1/logs")
    );

    delivery_diagnostics.emitted.store(5, Ordering::Relaxed);
    processor
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    let diagnostics = runtime_diagnostics.snapshot();
    let diagnostic = diagnostics
        .get("otel.logs_dropped")
        .expect("direct subscriber shutdown drop diagnostic");
    assert_eq!(diagnostic.count, 4);
    assert!(
        diagnostic
            .message
            .contains("https://collector.example/v1/logs")
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
    assert_eq!(context.trace_id, relay_trace_id(parent_uuid));
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
    let Some(AnyValue::Map(metadata)) = attributes.get(&Key::new("nemo_relay.mark.metadata"))
    else {
        panic!("metadata attribute must be a map");
    };
    assert_eq!(
        metadata.get(&Key::new("tenant")),
        Some(&AnyValue::from("demo"))
    );
    assert!(!metadata.contains_key(&Key::new(LOG_SEVERITY_METADATA_KEY)));
}

#[test]
fn mark_projection_preserves_category_profile_schema_and_json_scalars() {
    let (mut processor, exporter, provider) = processor(LogSeverity::Trace);
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("relay.checkpoint")
            .data(json!(true))
            .data_schema(
                DataSchema::builder()
                    .name("relay.checkpoint")
                    .version("1")
                    .build(),
            )
            .metadata(json!("opaque-metadata"))
            .build(),
        Some(EventCategory::custom()),
        Some(CategoryProfile::builder().subtype("checkpoint").build()),
    ));
    processor.process(&event);
    provider.force_flush().unwrap();

    let logs = exporter.get_emitted_logs().unwrap();
    let record = &logs[0].record;
    assert_eq!(record.body(), Some(&AnyValue::Boolean(true)));
    let attributes = record
        .attributes_iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<HashMap<_, _>>();
    assert_eq!(
        attributes.get(&Key::new("nemo_relay.mark.category")),
        Some(&AnyValue::from("custom"))
    );
    assert!(attributes.contains_key(&Key::new("nemo_relay.mark.category_profile")));
    assert_eq!(
        attributes.get(&Key::new("nemo_relay.mark.data_schema.name")),
        Some(&AnyValue::from("relay.checkpoint"))
    );
    assert_eq!(
        attributes.get(&Key::new("nemo_relay.mark.metadata")),
        Some(&AnyValue::from("opaque-metadata"))
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
    processor.process(&mark(
        None,
        "valid-metric",
        Some(json!({"measurements": [{
            "name": "example.valid",
            "kind": "counter",
            "value_type": "u64",
            "value": 1
        }]})),
        Some(
            DataSchema::builder()
                .name(METRIC_DATA_SCHEMA_NAME)
                .version(METRIC_DATA_SCHEMA_VERSION)
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

#[test]
fn severity_and_json_conversion_cover_remaining_scalar_variants() {
    assert_eq!(
        otel_severity(LogSeverity::Trace),
        (Severity::Trace, "TRACE")
    );
    assert_eq!(
        otel_severity(LogSeverity::Debug),
        (Severity::Debug, "DEBUG")
    );
    assert_eq!(
        otel_severity(LogSeverity::Error),
        (Severity::Error, "ERROR")
    );
    assert_eq!(
        json_any_value(&json!(true), false),
        Some(AnyValue::Boolean(true))
    );
    assert_eq!(
        json_any_value(&json!(1.25), false),
        Some(AnyValue::Double(1.25))
    );
}
