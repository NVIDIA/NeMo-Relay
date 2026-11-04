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
use opentelemetry_proto::tonic::collector::logs::v1::logs_service_server::LogsService;
use opentelemetry_proto::tonic::collector::logs::v1::{
    ExportLogsServiceRequest, ExportLogsServiceResponse,
};
use opentelemetry_proto::tonic::collector::metrics::v1::metrics_service_server::MetricsService;
use opentelemetry_proto::tonic::collector::metrics::v1::{
    ExportMetricsServiceRequest, ExportMetricsServiceResponse,
};
use opentelemetry_proto::tonic::common::v1::{KeyValue as OtlpKeyValue, any_value};
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader};
use serde_json::json;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

#[test]
fn metric_measurements_aggregate_across_sessions() {
    let (mut processor, exporter, provider) = processor();
    for _ in 0..2 {
        let mut event = metric_event(
            METRIC_DATA_SCHEMA_VERSION,
            json!({"measurements": [{
                "name": "example.session_tokens",
                "kind": "counter",
                "value_type": "u64",
                "value": 3,
                "attributes": {"model": "example-model"}
            }]}),
        );
        event.set_propagation_root_uuid(Some(uuid::Uuid::now_v7()));
        processor.process(&crate::test_support::event_view(&event));
    }
    provider.force_flush().unwrap();
    assert_eq!(processor.rejected_marks, 0);
    let batches = exporter.get_finished_metrics().unwrap();
    let metric = batches
        .iter()
        .flat_map(|batch| batch.scope_metrics())
        .flat_map(|scope| scope.metrics())
        .find(|metric| metric.name() == "example.session_tokens")
        .unwrap();
    let AggregatedMetrics::U64(MetricData::Sum(sum)) = metric.data() else {
        panic!("counter must export as a u64 sum");
    };
    let points = sum.data_points().collect::<Vec<_>>();
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].value(), 6);
    assert_eq!(
        points[0].attributes().collect::<Vec<_>>(),
        vec![&KeyValue::new("model", "example-model")]
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
    processor.process(&crate::test_support::event_view(&metric_event(
        METRIC_DATA_SCHEMA_VERSION,
        serde_json::to_value(MetricEnvelope {
            measurements: vec![counter, histogram, gauge],
        })
        .unwrap(),
    )));
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
fn metric_measurements_aggregate_across_propagation_roots() {
    let (mut processor, exporter, provider) = processor();
    for root_uuid in [uuid::Uuid::from_u128(1), uuid::Uuid::from_u128(2)] {
        let mut counter = measurement(
            "example.requests",
            MetricKind::Counter,
            MetricValueType::U64,
            json!(1),
        );
        counter.attributes = Some(json!({"model": "example-model"}));
        let mut event = metric_event(
            METRIC_DATA_SCHEMA_VERSION,
            serde_json::to_value(MetricEnvelope {
                measurements: vec![counter],
            })
            .unwrap(),
        );
        event.set_propagation_root_uuid(Some(root_uuid));
        processor.process(&crate::test_support::event_view(&event));
    }
    provider.force_flush().unwrap();

    let batches = exporter.get_finished_metrics().unwrap();
    let metric = batches
        .iter()
        .flat_map(|batch| batch.scope_metrics())
        .flat_map(|scope| scope.metrics())
        .find(|metric| metric.name() == "example.requests")
        .expect("request counter");
    let AggregatedMetrics::U64(MetricData::Sum(sum)) = metric.data() else {
        panic!("request counter must export as a u64 sum");
    };
    let points = sum.data_points().collect::<Vec<_>>();
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].value(), 2);
    assert!(
        points[0]
            .attributes()
            .all(|attribute| attribute.key.as_str() != "nemo_relay.session.instance_id")
    );
}

#[test]
fn records_gen_ai_stream_time_to_first_chunk_as_standard_histogram() {
    let (mut processor, exporter, provider) = processor();
    let event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("openai.chat.completions").build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::llm(),
        Some(
            CategoryProfile::builder()
                .model_name("gpt-5")
                .time_to_first_chunk(0.125)
                .build(),
        ),
    ));

    processor.process(&crate::test_support::event_view(&event));
    provider.force_flush().unwrap();

    let batches = exporter.get_finished_metrics().unwrap();
    let metric = batches
        .iter()
        .flat_map(|batch| batch.scope_metrics())
        .flat_map(|scope| scope.metrics())
        .find(|metric| metric.name() == "gen_ai.client.operation.time_to_first_chunk")
        .expect("GenAI time-to-first-chunk histogram");
    assert_eq!(metric.unit(), "s");
    let AggregatedMetrics::F64(MetricData::Histogram(histogram)) = metric.data() else {
        panic!("time-to-first-chunk must export as an f64 histogram");
    };
    let point = histogram.data_points().next().unwrap();
    assert_eq!(point.count(), 1);
    assert_eq!(point.sum(), 0.125);
    assert!(
        point
            .attributes()
            .any(|attribute| { attribute == &KeyValue::new("gen_ai.operation.name", "chat") })
    );
    assert!(
        point
            .attributes()
            .any(|attribute| { attribute == &KeyValue::new("gen_ai.provider.name", "openai") })
    );
    assert!(
        point
            .attributes()
            .any(|attribute| { attribute == &KeyValue::new("gen_ai.request.model", "gpt-5") })
    );
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
    processor.process(&crate::test_support::event_view(&metric_event(
        METRIC_DATA_SCHEMA_VERSION,
        serde_json::to_value(MetricEnvelope { measurements }).unwrap(),
    )));
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
            processor.process(&crate::test_support::event_view(&metric_event(
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
            )));
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
    processor.process(&crate::test_support::event_view(&metric_event(
        METRIC_DATA_SCHEMA_VERSION,
        serde_json::to_value(MetricEnvelope {
            measurements: vec![good, bad],
        })
        .unwrap(),
    )));
    processor.process(&crate::test_support::event_view(&metric_event(
        "999",
        json!({"measurements": [{"name": "ignored"}]}),
    )));
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
    processor.process(&crate::test_support::event_view(&metric_event(
        METRIC_DATA_SCHEMA_VERSION,
        serde_json::to_value(MetricEnvelope {
            measurements: vec![first],
        })
        .unwrap(),
    )));
    let conflicting = measurement(
        "EXAMPLE.ONE",
        MetricKind::Gauge,
        MetricValueType::F64,
        json!(2.0),
    );
    processor.process(&crate::test_support::event_view(&metric_event(
        METRIC_DATA_SCHEMA_VERSION,
        serde_json::to_value(MetricEnvelope {
            measurements: vec![conflicting],
        })
        .unwrap(),
    )));
    let over_limit = measurement(
        "example.two",
        MetricKind::Gauge,
        MetricValueType::I64,
        json!(2),
    );
    processor.process(&crate::test_support::event_view(&metric_event(
        METRIC_DATA_SCHEMA_VERSION,
        serde_json::to_value(MetricEnvelope {
            measurements: vec![over_limit],
        })
        .unwrap(),
    )));
    provider.force_flush().unwrap();
    assert_eq!(processor.rejected_marks, 2);
    assert_eq!(processor.instruments.len(), 1);
}

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
