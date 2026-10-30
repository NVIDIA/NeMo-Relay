// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for Relay OTLP metric routing and recording.

use super::*;
use crate::api::event::{
    BaseEvent, CategoryProfile, DataSchema, EventCategory, METRIC_DATA_SCHEMA_NAME,
    METRIC_DATA_SCHEMA_VERSION, MarkEvent, ScopeCategory, ScopeEvent,
};
use crate::json::Json;
use nemo_relay_logging::{
    FileLogSinkConfig, LogFormat, LogLevel, LogSinkConfig, LoggingConfig, init_logging,
};
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
fn metric_export_failure_logging_is_independent_of_diagnostic_capacity() {
    let _logging_lock = crate::test_support::lock_test_logging();
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
    let runtime_diagnostics = SignalRuntimeDiagnostics::new(None)
        .with_host(crate::observability_test_support::core_host());
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
    runtime.flush();
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
