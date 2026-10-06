// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for mark and metric API behavior.

use std::sync::{Arc, Mutex};

use chrono::{TimeZone, Utc};
use serde_json::json;

use super::{
    EmitMarkEventParams, EmitMetricEventParams, event, metadata_with_log_severity, metric,
};
use crate::api::event::{
    Event, LOG_SEVERITY_METADATA_KEY, LogSeverity, METRIC_DATA_SCHEMA_NAME,
    METRIC_DATA_SCHEMA_VERSION, MetricKind, MetricMeasurement, MetricValueType,
};
use crate::api::runtime::{NemoRelayContextState, global_context};
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use crate::error::FlowError;

fn reset_global() {
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

fn lock_global_runtime() -> std::sync::MutexGuard<'static, ()> {
    crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

#[test]
fn typed_severity_overrides_metadata_and_requires_an_object() {
    let metadata = metadata_with_log_severity(
        Some(json!({
            (LOG_SEVERITY_METADATA_KEY): "trace",
            "source": "test"
        })),
        Some(LogSeverity::Error),
    )
    .unwrap()
    .unwrap();
    assert_eq!(metadata[LOG_SEVERITY_METADATA_KEY], "error");
    assert_eq!(metadata["source"], "test");

    let metadata = metadata_with_log_severity(None, Some(LogSeverity::Debug))
        .unwrap()
        .unwrap();
    assert_eq!(metadata[LOG_SEVERITY_METADATA_KEY], "debug");

    let error = metadata_with_log_severity(Some(json!("not-an-object")), Some(LogSeverity::Info))
        .unwrap_err();
    assert!(matches!(error, FlowError::InvalidArgument(_)));

    assert_eq!(
        metadata_with_log_severity(Some(json!("unchanged")), None).unwrap(),
        Some(json!("unchanged"))
    );
}

#[test]
fn event_and_metric_emit_canonical_mark_metadata_and_schema() {
    let _guard = lock_global_runtime();
    reset_global();

    let captured = Arc::new(Mutex::new(Vec::<Event>::new()));
    let subscriber_events = captured.clone();
    register_subscriber(
        "scope-api-mark-observer",
        Arc::new(move |event| subscriber_events.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    event(
        EmitMarkEventParams::builder()
            .name("application.warning")
            .metadata(json!({
                (LOG_SEVERITY_METADATA_KEY): "trace",
                "source": "test"
            }))
            .severity(LogSeverity::Warn)
            .build(),
    )
    .unwrap();

    let timestamp = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap();
    metric(
        EmitMetricEventParams::builder()
            .name("tokenomics.sample")
            .measurements(vec![
                MetricMeasurement::builder()
                    .name("example.tokens.saved")
                    .kind(MetricKind::Counter)
                    .value_type(MetricValueType::U64)
                    .value(json!(42u64))
                    .unit("{token}")
                    .attributes(json!({"model": "example-model"}))
                    .build(),
            ])
            .metadata(json!({"source": "test"}))
            .timestamp(timestamp)
            .build(),
    )
    .unwrap();

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("scope-api-mark-observer").unwrap());

    let events = captured.lock().unwrap();
    let warning = events
        .iter()
        .find(|event| event.name() == "application.warning")
        .expect("warning mark should be emitted");
    assert_eq!(
        warning.metadata().unwrap()[LOG_SEVERITY_METADATA_KEY],
        "warn"
    );
    assert_eq!(warning.metadata().unwrap()["source"], "test");

    let metric = events
        .iter()
        .find(|event| event.name() == "tokenomics.sample")
        .expect("metric mark should be emitted");
    assert_eq!(metric.timestamp(), &timestamp);
    assert_eq!(metric.data_schema().unwrap().name, METRIC_DATA_SCHEMA_NAME);
    assert_eq!(
        metric.data_schema().unwrap().version,
        METRIC_DATA_SCHEMA_VERSION
    );
    assert_eq!(metric.data().unwrap()["measurements"][0]["value"], 42);
    assert_eq!(metric.metadata().unwrap()["source"], "test");
}

#[test]
fn metric_rejects_the_entire_invalid_envelope_before_emission() {
    let _guard = lock_global_runtime();
    reset_global();

    let error = metric(
        EmitMetricEventParams::builder()
            .name("invalid.metric")
            .measurements(Vec::new())
            .build(),
    )
    .unwrap_err();
    assert!(matches!(error, FlowError::InvalidArgument(_)));

    let error = event(
        EmitMarkEventParams::builder()
            .name("invalid.severity.metadata")
            .metadata(json!(["not", "an", "object"]))
            .severity(LogSeverity::Info)
            .build(),
    )
    .unwrap_err();
    assert!(matches!(error, FlowError::InvalidArgument(_)));
}

#[test]
fn malformed_reserved_tool_completion_marks_are_rejected_before_publication() {
    let _guard = lock_global_runtime();
    reset_global();
    let captured = Arc::new(Mutex::new(Vec::<Event>::new()));
    let capture = captured.clone();
    register_subscriber(
        "invalid-completion-mark",
        Arc::new(move |event| {
            capture.lock().unwrap().push(event.clone());
        }),
    )
    .unwrap();
    for data in [
        None,
        Some(json!({"arguments": {"secret": "request"}, "result": "secret"})),
        Some(json!({"tool_name": 7, "result": "secret"})),
        Some(json!({"tool_name": "   ", "arguments": "secret"})),
    ] {
        for metadata in [
            None,
            Some(json!({"start_observed": false})),
            Some(json!({"start_observed": true})),
        ] {
            let error = event(
                EmitMarkEventParams::builder()
                    .name("tool_end_without_start")
                    .data_opt(data.clone())
                    .metadata_opt(metadata)
                    .build(),
            )
            .unwrap_err();
            assert!(matches!(error, FlowError::InvalidArgument(_)));
        }
    }
    flush_subscribers().unwrap();
    assert!(captured.lock().unwrap().is_empty());
    deregister_subscriber("invalid-completion-mark").unwrap();
}

#[test]
fn reserved_tool_completion_marks_sanitize_payloads_without_start_metadata() {
    use crate::api::registry::{
        deregister_tool_sanitize_request_guardrail, deregister_tool_sanitize_response_guardrail,
        register_tool_sanitize_request_guardrail, register_tool_sanitize_response_guardrail,
    };
    let _guard = lock_global_runtime();
    reset_global();
    register_tool_sanitize_request_guardrail(
        "completion-request",
        1,
        Arc::new(|_, _| Box::pin(async { Ok(json!({"request": "redacted"})) })),
    )
    .unwrap();
    register_tool_sanitize_response_guardrail(
        "completion-response",
        1,
        Arc::new(|_, _| Box::pin(async { Ok(json!({"response": "redacted"})) })),
    )
    .unwrap();
    let captured = Arc::new(Mutex::new(Vec::<Event>::new()));
    let capture = captured.clone();
    register_subscriber(
        "completion-sanitizer-observer",
        Arc::new(move |event| {
            capture.lock().unwrap().push(event.clone());
        }),
    )
    .unwrap();
    event(EmitMarkEventParams::builder().name("tool_end_without_start")
        .data(json!({"tool_name": "example", "arguments": "request-secret", "result": "response-secret"}))
        .build()).unwrap();
    flush_subscribers().unwrap();
    let events = captured.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0].data().unwrap()["arguments"],
        json!({"request": "redacted"})
    );
    assert_eq!(
        events[0].data().unwrap()["result"],
        json!({"response": "redacted"})
    );
    drop(events);
    deregister_subscriber("completion-sanitizer-observer").unwrap();
    deregister_tool_sanitize_request_guardrail("completion-request").unwrap();
    deregister_tool_sanitize_response_guardrail("completion-response").unwrap();
}
