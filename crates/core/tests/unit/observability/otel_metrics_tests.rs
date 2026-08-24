// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for Relay OTLP metric routing and recording.

use super::*;
use crate::api::event::{
    BaseEvent, DataSchema, METRIC_DATA_SCHEMA_NAME, METRIC_DATA_SCHEMA_VERSION, MarkEvent,
};
use crate::logging::{
    FileLogSinkConfig, LogFormat, LogLevel, LogSinkConfig, LoggingConfig, init_logging,
};
use crate::observability::otel_logs::{OpenTelemetryLogConfig, OpenTelemetryLogSubscriber};
use opentelemetry_proto::tonic::collector::logs::v1::logs_service_server::{
    LogsService, LogsServiceServer,
};
use opentelemetry_proto::tonic::collector::logs::v1::{
    ExportLogsServiceRequest, ExportLogsServiceResponse,
};
use opentelemetry_proto::tonic::collector::metrics::v1::metrics_service_server::{
    MetricsService, MetricsServiceServer,
};
use opentelemetry_proto::tonic::collector::metrics::v1::{
    ExportMetricsServiceRequest, ExportMetricsServiceResponse,
};
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader};
use prost::Message;
use serde_json::json;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use tokio_stream::wrappers::TcpListenerStream;

fn measurement(
    name: &str,
    kind: MetricKind,
    value_type: MetricValueType,
    value: Json,
) -> MetricMeasurement {
    MetricMeasurement {
        name: name.to_string(),
        kind,
        value_type,
        value,
        unit: None,
        description: None,
        attributes: None,
        boundaries: None,
    }
}

fn metric_event(version: &str, data: Json) -> Event {
    metric_event_named("metric.record", version, data)
}

fn metric_event_named(name: &str, version: &str, data: Json) -> Event {
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name(name)
            .data(data)
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version(version)
                    .build(),
            )
            .build(),
        None,
        None,
    ))
}

fn processor() -> (
    MetricEventProcessor,
    InMemoryMetricExporter,
    SdkMeterProvider,
) {
    let exporter = InMemoryMetricExporter::default();
    let reader = PeriodicReader::builder(exporter.clone()).build();
    let provider = SdkMeterProvider::builder().with_reader(reader).build();
    let meter = provider.meter("nemo-relay-test");
    (
        MetricEventProcessor::new(meter, 8, None, 2_000),
        exporter,
        provider,
    )
}

#[derive(Debug)]
struct FailingMetricExporter;

impl PushMetricExporter for FailingMetricExporter {
    async fn export(&self, _metrics: &ResourceMetrics) -> OTelSdkResult {
        Err(opentelemetry_sdk::error::OTelSdkError::InternalFailure(
            "collector unavailable".to_string(),
        ))
    }

    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }

    fn shutdown_with_timeout(&self, _timeout: Duration) -> OTelSdkResult {
        Ok(())
    }

    fn temporality(&self) -> Temporality {
        Temporality::Cumulative
    }
}

#[test]
fn direct_metric_subscriber_recovers_a_poisoned_processor_lock() {
    let subscriber =
        OpenTelemetryMetricSubscriber::new(OpenTelemetryMetricConfig::new("http://127.0.0.1:4318"))
            .unwrap();
    let processor = Arc::clone(&subscriber.inner._processor);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = processor.lock().unwrap();
            panic!("poison metric processor");
        }))
        .is_err()
    );

    (subscriber.subscriber())(&metric_event(
        "1",
        json!({"measurements": [{
            "name": "example.recovered",
            "kind": "counter",
            "value_type": "u64",
            "value": 1
        }]}),
    ));

    let Err(poisoned) = processor.lock() else {
        panic!("metric processor lock should remain poisoned after recovery");
    };
    let processor = poisoned.into_inner();
    assert!(processor.instruments.contains_key("example.recovered"));
}

#[test]
fn direct_metric_subscriber_exposes_runtime_diagnostics() {
    let subscriber =
        OpenTelemetryMetricSubscriber::new(OpenTelemetryMetricConfig::new("http://127.0.0.1:4318"))
            .unwrap();
    let event = metric_event("999", json!({"measurements": []}));

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
fn direct_metric_subscriber_retains_distinct_invalid_mark_causes() {
    let subscriber =
        OpenTelemetryMetricSubscriber::new(OpenTelemetryMetricConfig::new("http://127.0.0.1:4318"))
            .unwrap();
    let events = [
        metric_event_named("wrong-schema", "999", json!({"measurements": []})),
        metric_event_named(
            "empty-measurements",
            METRIC_DATA_SCHEMA_VERSION,
            json!({"measurements": []}),
        ),
        metric_event_named(
            "invalid-counter-type",
            METRIC_DATA_SCHEMA_VERSION,
            json!({"measurements": [{
                "name": "example.requests",
                "kind": "counter",
                "value_type": "i64",
                "value": -1
            }]}),
        ),
        metric_event_named(
            "invalid-instrument-name",
            METRIC_DATA_SCHEMA_VERSION,
            json!({"measurements": [{
                "name": "9bad-name",
                "kind": "counter",
                "value_type": "u64",
                "value": 1
            }]}),
        ),
    ];

    for event in &events {
        (subscriber.subscriber())(event);
    }

    let diagnostics = subscriber.runtime_diagnostics();
    let diagnostic = diagnostics
        .get("otel.metric_mark_invalid")
        .expect("invalid metric diagnostic");
    assert_eq!(diagnostic.count, 4);
    for mark_name in [
        "wrong-schema",
        "empty-measurements",
        "invalid-counter-type",
        "invalid-instrument-name",
    ] {
        assert!(diagnostic.message.contains(mark_name));
    }
}

#[test]
fn metric_endpoint_resolution_replaces_terminal_trace_path() {
    for (input, expected) in [
        (
            "https://collector.example",
            "https://collector.example/v1/metrics",
        ),
        (
            "https://collector.example/team/v1/traces?project=dev",
            "https://collector.example/team/v1/metrics?project=dev",
        ),
        (
            "https://collector.example/",
            "https://collector.example/v1/metrics",
        ),
        (
            "https://collector.example/custom",
            "https://collector.example/custom",
        ),
    ] {
        let resolved = resolve_http_metric_endpoint(input);
        assert_eq!(resolved, expected);
        assert!(!resolved.contains("/v1/traces/v1/metrics"));
    }
}

#[test]
fn metric_temporality_accepts_all_supported_spellings() {
    for (value, expected, sdk) in [
        (
            "cumulative",
            MetricTemporality::Cumulative,
            Temporality::Cumulative,
        ),
        ("delta", MetricTemporality::Delta, Temporality::Delta),
        (
            "low_memory",
            MetricTemporality::LowMemory,
            Temporality::LowMemory,
        ),
        (
            "lowmemory",
            MetricTemporality::LowMemory,
            Temporality::LowMemory,
        ),
    ] {
        let parsed = value.parse::<MetricTemporality>().unwrap();
        assert_eq!(parsed, expected);
        assert_eq!(parsed.sdk(), sdk);
    }
    assert_eq!(MetricTemporality::Delta.as_str(), "delta");
    assert!("instantaneous".parse::<MetricTemporality>().is_err());
}

#[test]
fn metric_config_validates_limits_and_retains_resource_identity() {
    for config in [
        OpenTelemetryMetricConfig::new("   "),
        OpenTelemetryMetricConfig::new("https://collector.example/v1/metrics")
            .with_export_interval(Duration::ZERO),
        OpenTelemetryMetricConfig::new("https://collector.example/v1/metrics")
            .with_max_instruments(0),
        OpenTelemetryMetricConfig::new("https://collector.example/v1/metrics")
            .with_cardinality_limit(0),
    ] {
        assert!(config.validate().is_err());
    }

    let config = OpenTelemetryMetricConfig::new("https://collector.example/v1/metrics")
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
fn metric_config_rejects_cardinality_limit_that_the_sdk_cannot_build() {
    let error = OpenTelemetryMetricConfig::new("https://collector.example/v1/metrics")
        .with_cardinality_limit(usize::MAX)
        .validate()
        .unwrap_err();

    assert!(
        error.to_string().contains("less than usize::MAX"),
        "{error}"
    );
}

#[test]
fn metric_config_rejects_zero_timeout() {
    let error = OpenTelemetryMetricConfig::new("https://collector.example/v1/metrics")
        .with_timeout(Duration::ZERO)
        .validate()
        .unwrap_err();

    assert!(error.to_string().contains("timeout must be greater than 0"));
}

#[test]
fn metric_config_rejects_blank_and_padded_headers() {
    for (key, value) in [
        ("", "token"),
        (" authorization", "token"),
        ("authorization", ""),
        ("authorization", " token "),
    ] {
        let error = OpenTelemetryMetricConfig::new("https://collector.example/v1/metrics")
            .with_header(key, value)
            .validate()
            .unwrap_err();
        assert!(error.to_string().contains("surrounding whitespace"));
    }
}

#[test]
fn metric_delivery_state_survives_exporter_error_wrapping() {
    let diagnostics = MetricDeliveryDiagnostics::new(
        "https://collector.example/v1/metrics".to_string(),
        SignalRuntimeDiagnostics::new(Some(
            "opentelemetry.metrics.endpoints[0].endpoint".to_string(),
        )),
    );
    diagnostics.export_failures.store(2, Ordering::Relaxed);

    assert_eq!(
        diagnostics.failure_summary().as_deref(),
        Some("otel.metrics_export_failed (2)")
    );
}

#[test]
fn metric_export_failure_logging_is_independent_of_diagnostic_capacity() {
    let _logging_lock = crate::logging::lock_test_logging();
    let temporary_directory = tempfile::tempdir().unwrap();
    let log_path = temporary_directory.path().join("metric-export.log.jsonl");
    let runtime = init_logging(&LoggingConfig {
        level: LogLevel::Error,
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
            path: log_path.clone(),
            level: LogLevel::Error,
            format: LogFormat::Jsonl,
            ..FileLogSinkConfig::default()
        })],
        ..LoggingConfig::default()
    })
    .unwrap();
    let runtime_diagnostics = SignalRuntimeDiagnostics::new(None);
    for index in 0..32 {
        runtime_diagnostics.record(
            format!("test.diagnostic.{index}"),
            "diagnostic capacity filler".to_string(),
            1,
        );
    }
    let exporter = DiagnosticMetricExporter {
        inner: FailingMetricExporter,
        diagnostics: Arc::new(MetricDeliveryDiagnostics::new(
            "https://collector.example/v1/metrics".to_string(),
            runtime_diagnostics.clone(),
        )),
    };

    let result = futures::executor::block_on(exporter.export(&ResourceMetrics::default()));
    assert!(result.is_err());
    runtime.logger.flush();
    let mut contents = String::new();
    for _ in 0..50 {
        contents = std::fs::read_to_string(&log_path).unwrap_or_default();
        if contents.contains("otel_metrics_export_failed") {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    runtime.shutdown();

    let record: serde_json::Value = contents
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .find(|record: &serde_json::Value| record["event"] == "otel_metrics_export_failed")
        .expect("metric export failure error record");
    assert_eq!(record["level"], "error");
    assert_eq!(record["target"], "nemo_relay.observability");
    assert!(
        runtime_diagnostics
            .snapshot()
            .get("otel.metrics_export_failed")
            .is_none()
    );
}

#[test]
fn valid_envelope_records_counter_gauge_and_negative_histogram() {
    let (mut processor, exporter, provider) = processor();
    let mut counter = measurement(
        "example.tokens.saved",
        MetricKind::Counter,
        MetricValueType::U64,
        json!(42),
    );
    counter.unit = Some("{token}".into());
    counter.attributes = Some(json!({"model": "example-model"}));
    let histogram = measurement(
        "example.temperature",
        MetricKind::Histogram,
        MetricValueType::F64,
        json!(-1.25),
    );
    let gauge = measurement(
        "example.active",
        MetricKind::Gauge,
        MetricValueType::I64,
        json!(-2),
    );
    processor.process(&metric_event(
        METRIC_DATA_SCHEMA_VERSION,
        serde_json::to_value(MetricEnvelope {
            measurements: vec![counter, histogram, gauge],
        })
        .unwrap(),
    ));
    provider.force_flush().unwrap();

    let batches = exporter.get_finished_metrics().unwrap();
    assert_eq!(processor.rejected_marks, 0);
    let names = batches
        .iter()
        .flat_map(|batch| batch.scope_metrics())
        .flat_map(|scope| scope.metrics())
        .map(|metric| metric.name())
        .collect::<Vec<_>>();
    assert!(names.contains(&"example.tokens.saved"));
    assert!(names.contains(&"example.temperature"));
    assert!(names.contains(&"example.active"));

    let metrics = batches
        .iter()
        .flat_map(|batch| batch.scope_metrics())
        .flat_map(|scope| scope.metrics())
        .collect::<Vec<_>>();
    let counter = metrics
        .iter()
        .find(|metric| metric.name() == "example.tokens.saved")
        .unwrap();
    assert_eq!(counter.unit(), "{token}");
    let AggregatedMetrics::U64(MetricData::Sum(sum)) = counter.data() else {
        panic!("counter must export as a u64 sum");
    };
    let point = sum.data_points().next().unwrap();
    assert_eq!(point.value(), 42);
    assert!(
        point
            .attributes()
            .any(|attribute| attribute == &KeyValue::new("model", "example-model"))
    );

    let gauge = metrics
        .iter()
        .find(|metric| metric.name() == "example.active")
        .unwrap();
    let AggregatedMetrics::I64(MetricData::Gauge(gauge)) = gauge.data() else {
        panic!("gauge must export as an i64 gauge");
    };
    assert_eq!(gauge.data_points().next().unwrap().value(), -2);

    let histogram = metrics
        .iter()
        .find(|metric| metric.name() == "example.temperature")
        .unwrap();
    let AggregatedMetrics::F64(MetricData::Histogram(histogram)) = histogram.data() else {
        panic!("histogram must export as an f64 histogram");
    };
    let point = histogram.data_points().next().unwrap();
    assert_eq!(point.count(), 1);
    assert_eq!(point.sum(), -1.25);
}

#[test]
fn metric_processor_records_every_supported_instrument_type() {
    let (mut processor, exporter, provider) = processor();
    let mut u64_histogram = measurement(
        "example.latency",
        MetricKind::Histogram,
        MetricValueType::U64,
        json!(8),
    );
    u64_histogram.boundaries = Some(vec![1.0, 5.0, 10.0]);
    let measurements = vec![
        measurement(
            "example.f64.counter",
            MetricKind::Counter,
            MetricValueType::F64,
            json!(1.25),
        ),
        measurement(
            "example.i64.updown",
            MetricKind::UpDownCounter,
            MetricValueType::I64,
            json!(-3),
        ),
        measurement(
            "example.f64.updown",
            MetricKind::UpDownCounter,
            MetricValueType::F64,
            json!(2.5),
        ),
        measurement(
            "example.u64.gauge",
            MetricKind::Gauge,
            MetricValueType::U64,
            json!(4),
        ),
        measurement(
            "example.f64.gauge",
            MetricKind::Gauge,
            MetricValueType::F64,
            json!(5.5),
        ),
        u64_histogram,
    ];
    processor.process(&metric_event(
        METRIC_DATA_SCHEMA_VERSION,
        serde_json::to_value(MetricEnvelope { measurements }).unwrap(),
    ));
    provider.force_flush().unwrap();

    let names = exporter
        .get_finished_metrics()
        .unwrap()
        .iter()
        .flat_map(|batch| batch.scope_metrics())
        .flat_map(|scope| scope.metrics())
        .map(|metric| metric.name().to_string())
        .collect::<Vec<_>>();
    assert_eq!(processor.rejected_marks, 0);
    assert_eq!(names.len(), 6);
    assert!(names.contains(&"example.latency".to_string()));
}

#[test]
fn case_equivalent_measurements_use_a_deterministic_lowercase_instrument_name() {
    for names in [
        ["Example.Tokens", "example.tokens"],
        ["example.tokens", "Example.Tokens"],
    ] {
        let (mut processor, exporter, provider) = processor();
        for (value, name) in names.into_iter().enumerate() {
            processor.process(&metric_event(
                METRIC_DATA_SCHEMA_VERSION,
                serde_json::to_value(MetricEnvelope {
                    measurements: vec![measurement(
                        name,
                        MetricKind::Counter,
                        MetricValueType::U64,
                        json!(value + 1),
                    )],
                })
                .unwrap(),
            ));
        }
        provider.force_flush().unwrap();

        let batches = exporter.get_finished_metrics().unwrap();
        let names = batches
            .iter()
            .flat_map(|batch| batch.scope_metrics())
            .flat_map(|scope| scope.metrics())
            .map(|metric| metric.name().to_string())
            .collect::<Vec<_>>();
        assert_eq!(processor.instruments.len(), 1);
        assert_eq!(names, vec!["example.tokens"]);
    }
}

#[test]
fn invalid_envelopes_are_atomic_and_reserved_versions_do_not_reinterpret() {
    let (mut processor, exporter, provider) = processor();
    let good = measurement(
        "example.good",
        MetricKind::Gauge,
        MetricValueType::I64,
        json!(1),
    );
    let bad = measurement(
        "example.bad",
        MetricKind::Counter,
        MetricValueType::F64,
        json!(-1.0),
    );
    processor.process(&metric_event(
        METRIC_DATA_SCHEMA_VERSION,
        serde_json::to_value(MetricEnvelope {
            measurements: vec![good, bad],
        })
        .unwrap(),
    ));
    processor.process(&metric_event(
        "999",
        json!({"measurements": [{"name": "ignored"}]}),
    ));
    provider.force_flush().unwrap();
    assert_eq!(processor.rejected_marks, 2);
    assert!(exporter.get_finished_metrics().unwrap().is_empty());
}

#[test]
fn descriptor_conflicts_and_instrument_limit_reject_whole_mark() {
    let exporter = InMemoryMetricExporter::default();
    let reader = PeriodicReader::builder(exporter.clone()).build();
    let provider = SdkMeterProvider::builder().with_reader(reader).build();
    let mut processor = MetricEventProcessor::new(provider.meter("limit"), 1, None, 2_000);

    let first = measurement(
        "example.one",
        MetricKind::Gauge,
        MetricValueType::I64,
        json!(1),
    );
    processor.process(&metric_event(
        METRIC_DATA_SCHEMA_VERSION,
        serde_json::to_value(MetricEnvelope {
            measurements: vec![first],
        })
        .unwrap(),
    ));
    let conflicting = measurement(
        "EXAMPLE.ONE",
        MetricKind::Gauge,
        MetricValueType::F64,
        json!(2.0),
    );
    processor.process(&metric_event(
        METRIC_DATA_SCHEMA_VERSION,
        serde_json::to_value(MetricEnvelope {
            measurements: vec![conflicting],
        })
        .unwrap(),
    ));
    let over_limit = measurement(
        "example.two",
        MetricKind::Gauge,
        MetricValueType::I64,
        json!(2),
    );
    processor.process(&metric_event(
        METRIC_DATA_SCHEMA_VERSION,
        serde_json::to_value(MetricEnvelope {
            measurements: vec![over_limit],
        })
        .unwrap(),
    ));
    provider.force_flush().unwrap();
    assert_eq!(processor.rejected_marks, 2);
    assert_eq!(processor.instruments.len(), 1);
}

#[test]
fn metric_attribute_arrays_preserve_supported_primitive_types() {
    for value in [
        json!(["a", "b"]),
        json!([true, false]),
        json!([1, 2]),
        json!([1.0, 2.5]),
    ] {
        let attributes = MetricAttributes::try_from(Some(&json!({"value": value}))).unwrap();
        let (_, value) = attributes.iter().next().unwrap();
        assert!(matches!(metric_attribute_value(value), Value::Array(_)));
    }
    assert!(MetricAttributes::try_from(Some(&json!({"value": []}))).is_err());
    assert!(MetricAttributes::try_from(Some(&json!({"value": {"nested": true}}))).is_err());
}

#[test]
fn metric_attribute_scalars_preserve_supported_primitive_types() {
    for (json_value, expected) in [
        (json!(true), Value::Bool(true)),
        (json!(-3), Value::I64(-3)),
        (json!(1.25), Value::F64(1.25)),
    ] {
        let attributes = MetricAttributes::try_from(Some(&json!({"value": json_value}))).unwrap();
        let (_, value) = attributes.iter().next().unwrap();
        assert_eq!(metric_attribute_value(value), expected);
    }
}

#[test]
fn metric_attribute_fingerprints_are_typed_and_fixed_size() {
    let string = MetricAttributes::try_from(Some(&json!({"value": "1"}))).unwrap();
    let integer = MetricAttributes::try_from(Some(&json!({"value": 1}))).unwrap();
    let reordered = MetricAttributes::try_from(Some(&json!({"b": true, "a": 1}))).unwrap();
    let ordered = MetricAttributes::try_from(Some(&json!({"a": 1, "b": true}))).unwrap();

    let string_fingerprint = metric_attribute_set_fingerprint(&string).unwrap();
    assert_ne!(
        string_fingerprint,
        metric_attribute_set_fingerprint(&integer).unwrap()
    );
    assert_eq!(
        metric_attribute_set_fingerprint(&reordered),
        metric_attribute_set_fingerprint(&ordered)
    );
    assert_eq!(std::mem::size_of_val(&string_fingerprint), 32);
}

struct CapturedRequest {
    path: String,
    body: Vec<u8>,
}

fn capture_requests(
    listener: TcpListener,
    expected_requests: usize,
) -> mpsc::Receiver<CapturedRequest> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        for _ in 0..expected_requests {
            let (mut stream, _) = listener.accept().unwrap();
            let mut bytes = Vec::new();
            let mut buffer = [0_u8; 4_096];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut buffer).unwrap();
                assert!(count > 0, "collector closed before request headers");
                bytes.extend_from_slice(&buffer[..count]);
                if let Some(offset) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let header_end = offset + 4;
                    let headers = String::from_utf8_lossy(&bytes[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    break (header_end, content_length);
                }
            };
            while bytes.len() < header_end + content_length {
                let count = stream.read(&mut buffer).unwrap();
                assert!(count > 0, "collector closed before request body");
                bytes.extend_from_slice(&buffer[..count]);
            }
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let path = headers
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap()
                .to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .unwrap();
            sender
                .send(CapturedRequest {
                    path,
                    body: bytes[header_end..header_end + content_length].to_vec(),
                })
                .unwrap();
        }
    });
    receiver
}

fn capture_one_request(listener: TcpListener) -> mpsc::Receiver<CapturedRequest> {
    capture_requests(listener, 1)
}

#[test]
fn direct_http_subscribers_emit_decodable_signal_payloads() {
    let log_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let log_receiver = capture_one_request(log_listener.try_clone().unwrap());
    let log_endpoint = format!(
        "http://{}/tenant/v1/traces?project=dev",
        log_listener.local_addr().unwrap()
    );
    drop(log_listener);
    let log_subscriber = OpenTelemetryLogSubscriber::new(
        OpenTelemetryLogConfig::new(log_endpoint)
            .with_instrumentation_scope("relay-log-test")
            .with_resource_attribute("nv.project", "observability-dev"),
    )
    .unwrap();
    log_subscriber.subscriber()(&Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("log.mark")
            .data(json!({"message": "hello"}))
            .build(),
        None,
        None,
    )));
    log_subscriber.force_flush().unwrap();
    let log_request = log_receiver.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(log_request.path, "/tenant/v1/logs?project=dev");
    let logs = ExportLogsServiceRequest::decode(log_request.body.as_slice()).unwrap();
    let records = &logs.resource_logs[0].scope_logs[0].log_records;
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].severity_text, "INFO");
    assert_ne!(records[0].time_unix_nano, 0);
    assert_ne!(records[0].observed_time_unix_nano, 0);

    let metric_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let metric_receiver = capture_requests(metric_listener.try_clone().unwrap(), 2);
    let metric_endpoint = format!("http://{}", metric_listener.local_addr().unwrap());
    drop(metric_listener);
    let metric_subscriber = OpenTelemetryMetricSubscriber::new(
        OpenTelemetryMetricConfig::new(metric_endpoint)
            .with_instrumentation_scope("relay-metric-test")
            .with_resource_attribute("nv.project", "observability-dev"),
    )
    .unwrap();
    metric_subscriber.subscriber()(&metric_event(
        METRIC_DATA_SCHEMA_VERSION,
        serde_json::to_value(MetricEnvelope {
            measurements: vec![measurement(
                "example.tokens.saved",
                MetricKind::Counter,
                MetricValueType::U64,
                json!(42),
            )],
        })
        .unwrap(),
    ));
    metric_subscriber.force_flush().unwrap();
    let metric_request = metric_receiver
        .recv_timeout(Duration::from_secs(5))
        .unwrap();
    assert_eq!(metric_request.path, "/v1/metrics");
    let metrics = ExportMetricsServiceRequest::decode(metric_request.body.as_slice()).unwrap();
    assert_eq!(
        metrics.resource_metrics[0].scope_metrics[0].metrics[0].name,
        "example.tokens.saved"
    );

    log_subscriber.shutdown().unwrap();
    metric_subscriber.shutdown().unwrap();
    metric_receiver
        .recv_timeout(Duration::from_secs(5))
        .unwrap();
}

#[derive(Clone)]
struct GrpcLogsCollector {
    sender: tokio::sync::mpsc::UnboundedSender<(ExportLogsServiceRequest, Option<String>)>,
}

#[tonic::async_trait]
impl LogsService for GrpcLogsCollector {
    async fn export(
        &self,
        request: tonic::Request<ExportLogsServiceRequest>,
    ) -> std::result::Result<tonic::Response<ExportLogsServiceResponse>, tonic::Status> {
        let project = request
            .metadata()
            .get("x-nv-project")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        self.sender.send((request.into_inner(), project)).unwrap();
        Ok(tonic::Response::new(ExportLogsServiceResponse::default()))
    }
}

#[derive(Clone)]
struct GrpcMetricsCollector {
    sender: tokio::sync::mpsc::UnboundedSender<(ExportMetricsServiceRequest, Option<String>)>,
}

#[tonic::async_trait]
impl MetricsService for GrpcMetricsCollector {
    async fn export(
        &self,
        request: tonic::Request<ExportMetricsServiceRequest>,
    ) -> std::result::Result<tonic::Response<ExportMetricsServiceResponse>, tonic::Status> {
        let project = request
            .metadata()
            .get("x-nv-project")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        self.sender.send((request.into_inner(), project)).unwrap();
        Ok(tonic::Response::new(ExportMetricsServiceResponse::default()))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn direct_grpc_subscribers_export_both_services_and_metadata() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (logs_sender, mut logs_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (metrics_sender, mut metrics_receiver) = tokio::sync::mpsc::unbounded_channel();
    let server = tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(LogsServiceServer::new(GrpcLogsCollector {
                sender: logs_sender,
            }))
            .add_service(MetricsServiceServer::new(GrpcMetricsCollector {
                sender: metrics_sender,
            }))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );

    let endpoint = format!("http://{address}");
    let log_subscriber = OpenTelemetryLogSubscriber::new(
        OpenTelemetryLogConfig::new(endpoint.clone())
            .with_transport(OtlpTransport::Grpc)
            .with_header("x-nv-project", "observability-dev"),
    )
    .unwrap();
    let metric_subscriber = OpenTelemetryMetricSubscriber::new(
        OpenTelemetryMetricConfig::new(endpoint)
            .with_transport(OtlpTransport::Grpc)
            .with_header("x-nv-project", "observability-dev"),
    )
    .unwrap();

    log_subscriber.subscriber()(&Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("grpc.log").build(),
        None,
        None,
    )));
    metric_subscriber.subscriber()(&metric_event(
        METRIC_DATA_SCHEMA_VERSION,
        serde_json::to_value(MetricEnvelope {
            measurements: vec![measurement(
                "grpc.metric",
                MetricKind::Gauge,
                MetricValueType::I64,
                json!(7),
            )],
        })
        .unwrap(),
    ));
    let flush_log = log_subscriber.clone();
    let flush_metric = metric_subscriber.clone();
    tokio::task::spawn_blocking(move || {
        flush_log.force_flush().unwrap();
        flush_metric.force_flush().unwrap();
    })
    .await
    .unwrap();

    let (logs, log_project) = tokio::time::timeout(Duration::from_secs(5), logs_receiver.recv())
        .await
        .unwrap()
        .unwrap();
    let (metrics, metric_project) =
        tokio::time::timeout(Duration::from_secs(5), metrics_receiver.recv())
            .await
            .unwrap()
            .unwrap();
    assert_eq!(log_project.as_deref(), Some("observability-dev"));
    assert_eq!(metric_project.as_deref(), Some("observability-dev"));
    assert_eq!(
        logs.resource_logs[0].scope_logs[0].log_records[0].severity_text,
        "INFO"
    );
    assert_eq!(
        metrics.resource_metrics[0].scope_metrics[0].metrics[0].name,
        "grpc.metric"
    );

    tokio::task::spawn_blocking(move || {
        log_subscriber.shutdown().unwrap();
        metric_subscriber.shutdown().unwrap();
    })
    .await
    .unwrap();
    server.abort();
}
