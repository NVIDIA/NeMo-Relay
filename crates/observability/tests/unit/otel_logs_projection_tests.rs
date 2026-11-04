// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_relay_types::Json;
use nemo_relay_types::api::event::*;
use nemo_relay_types::api::llm::*;
use nemo_relay_types::api::scope::*;
use nemo_relay_types::api::tool::*;
use nemo_relay_types::codec::request::*;
use nemo_relay_types::codec::response::*;
use opentelemetry_sdk::logs::{InMemoryLogExporter, SdkLoggerProvider};
use serde_json::json;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

#[test]
fn scope_lineage_retains_active_contexts_and_preserves_root_trace_id() {
    let mut lineage = ScopeLineage::new();
    let root = Uuid::now_v7();
    lineage.process_start(&crate::test_support::event_view(&scope(
        root,
        ScopeCategory::Start,
    )));
    for _ in 0..4_096 {
        lineage.process_start(&crate::test_support::event_view(&scope(
            Uuid::now_v7(),
            ScopeCategory::Start,
        )));
    }

    assert_eq!(lineage.active.len(), 4_097);
    assert!(lineage.active.contains_key(&root));

    let child = Uuid::now_v7();
    lineage.process_start(&crate::test_support::event_view(&scope_with_parent(
        child,
        Some(root),
        ScopeCategory::Start,
    )));
    assert_eq!(lineage.active[&child].trace_id(), relay_trace_id(root));

    lineage.process_end(&crate::test_support::event_view(&scope(
        child,
        ScopeCategory::End,
    )));
    assert!(!lineage.active.contains_key(&child));
    assert!(lineage.completed.contains_key(&child));
}

#[test]
fn scope_lineage_synthesizes_remote_context_only_for_imported_parents() {
    let lineage = ScopeLineage::new();
    let root_uuid = Uuid::now_v7();
    let imported_parent_uuid = Uuid::now_v7();
    let mut imported = scope_with_parent(
        Uuid::now_v7(),
        Some(imported_parent_uuid),
        ScopeCategory::Start,
    );
    imported.set_propagation_root_uuid(Some(root_uuid));
    imported.set_propagation_parent_uuid(Some(imported_parent_uuid));

    let remote_context = lineage
        .parent_context(&crate::test_support::event_view(&imported))
        .expect("imported parent context");
    assert!(remote_context.is_remote());
    assert_eq!(remote_context.trace_id(), relay_trace_id(root_uuid));
    assert_eq!(
        remote_context.span_id(),
        relay_span_id(imported_parent_uuid)
    );

    let mut local = scope_with_parent(Uuid::now_v7(), Some(Uuid::now_v7()), ScopeCategory::Start);
    local.set_propagation_root_uuid(Some(Uuid::now_v7()));
    assert!(
        lineage
            .parent_context(&crate::test_support::event_view(&local))
            .is_none()
    );
}

#[test]
fn scope_lineage_reuses_completed_parent_within_ttl_and_expires_after_boundary() {
    let mut lineage = ScopeLineage::new();
    let parent = Uuid::now_v7();
    let closed_at = chrono::Utc::now();
    lineage.process_start(&crate::test_support::event_view(&scope_at(
        parent,
        ScopeCategory::Start,
        closed_at,
    )));
    lineage.process_end(&crate::test_support::event_view(&scope_at(
        parent,
        ScopeCategory::End,
        closed_at,
    )));

    let child = mark(Some(parent), "late.mark", None, None, None);
    let context = lineage
        .parent_context(&crate::test_support::event_view(&child))
        .expect("completed parent context");
    assert_eq!(context.span_id(), relay_span_id(parent));

    for _ in 0..4_098 {
        let uuid = Uuid::now_v7();
        lineage.process_start(&crate::test_support::event_view(&scope_at(
            uuid,
            ScopeCategory::Start,
            closed_at,
        )));
        lineage.process_end(&crate::test_support::event_view(&scope_at(
            uuid,
            ScopeCategory::End,
            closed_at,
        )));
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
    assert!(
        lineage
            .parent_context(&crate::test_support::event_view(&child))
            .is_none()
    );

    lineage.process_end(&crate::test_support::event_view(&scope(
        Uuid::now_v7(),
        ScopeCategory::End,
    )));
}

#[test]
fn scope_lineage_replaces_completed_context_expiry_index_entries() {
    let mut lineage = ScopeLineage::new();
    let reused = Uuid::now_v7();
    let closed_at = chrono::Utc::now();
    lineage.process_start(&crate::test_support::event_view(&scope_at(
        reused,
        ScopeCategory::Start,
        closed_at,
    )));
    lineage.process_end(&crate::test_support::event_view(&scope_at(
        reused,
        ScopeCategory::End,
        closed_at,
    )));
    lineage.process_start(&crate::test_support::event_view(&scope_at(
        reused,
        ScopeCategory::Start,
        closed_at + chrono::Duration::seconds(1),
    )));
    lineage.process_end(&crate::test_support::event_view(&scope_at(
        reused,
        ScopeCategory::End,
        closed_at + chrono::Duration::seconds(1),
    )));

    assert!(lineage.completed.contains_key(&reused));
    assert_eq!(lineage.completed_expiry_index.len(), 1);
}

#[test]
fn non_metric_mark_maps_structured_body_attributes_and_scope_context() {
    let (mut processor, exporter, provider) = processor(LogSeverity::Info);
    let parent_uuid = Uuid::now_v7();
    processor.process(&crate::test_support::event_view(&scope(
        parent_uuid,
        ScopeCategory::Start,
    )));
    processor.process(&crate::test_support::event_view(&mark(
        Some(parent_uuid),
        "tokens.estimated",
        Some(json!({"value": 42, "nested": null})),
        None,
        Some(json!({
            LOG_SEVERITY_METADATA_KEY: "warning",
            "tenant": "demo"
        })),
    )));
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
    processor.process(&crate::test_support::event_view(&event));
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
    processor.process(&crate::test_support::event_view(&mark(
        None,
        "default-info",
        Some(json!(1)),
        None,
        None,
    )));
    processor.process(&crate::test_support::event_view(&mark(
        None,
        "invalid-severity",
        None,
        None,
        Some(json!({LOG_SEVERITY_METADATA_KEY: "notice"})),
    )));
    processor.process(&crate::test_support::event_view(&mark(
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
    )));
    processor.process(&crate::test_support::event_view(&mark(
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
    )));
    provider.force_flush().unwrap();
    assert!(exporter.get_emitted_logs().unwrap().is_empty());
    assert_eq!(processor.invalid_severity_count, 1);
    assert_eq!(processor.invalid_metric_count, 1);
}

#[test]
fn otlp_timestamp_range_drops_invalid_logs_and_preserves_boundaries() {
    let (mut processor, exporter, _provider) = processor(LogSeverity::Trace);
    for value in [
        "1969-12-31T23:59:59Z",
        "1969-12-31T23:59:59.999999999Z",
        "2554-07-21T23:34:33.709551616Z",
        "9999-12-31T23:59:59.999999Z",
        "1970-01-01T00:00:00Z",
        "2554-07-21T23:34:33.709551615Z",
    ] {
        let timestamp = chrono::DateTime::parse_from_rfc3339(value)
            .unwrap()
            .with_timezone(&chrono::Utc);
        let event = Event::Mark(MarkEvent::new(
            BaseEvent::builder()
                .name("replay")
                .timestamp(timestamp)
                .build(),
            None,
            None,
        ));
        processor.process(&crate::test_support::event_view(&event));
    }
    assert_eq!(
        processor
            .runtime_diagnostics
            .snapshot()
            .get("otel.timestamp_out_of_range")
            .unwrap()
            .count,
        4
    );
    let logs = exporter.get_emitted_logs().unwrap();
    assert_eq!(logs.len(), 2);
    assert_eq!(logs[0].record.timestamp(), Some(std::time::UNIX_EPOCH));
    // SystemTime uses 100-nanosecond ticks on Windows. The invalid event
    // one nanosecond above the limit must still be dropped before conversion.
    assert_eq!(
        logs[1].record.timestamp(),
        Some(std::time::UNIX_EPOCH + Duration::from_nanos(u64::MAX))
    );
}

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
