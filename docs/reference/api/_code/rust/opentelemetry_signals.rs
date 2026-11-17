// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::exporter::{
    otel_logs::{OpenTelemetryLogConfig, OpenTelemetryLogSubscriber},
    otel_metrics::{OpenTelemetryMetricConfig, OpenTelemetryMetricSubscriber},
};

pub fn run_example() -> Result<(), Box<dyn std::error::Error>> {
    let logs = nemo_relay::exporter::opentelemetry_logs(OpenTelemetryLogConfig::new(
        "http://localhost:4318/v1/logs",
    ))?;
    let metrics = nemo_relay::exporter::opentelemetry_metrics(OpenTelemetryMetricConfig::new(
        "http://localhost:4318/v1/metrics",
    ))?;
    logs.register("otlp-logs")?;
    metrics.register("otlp-metrics")?;

    // Run instrumented work here.

    complete_subscribers(&logs, &metrics)
}

fn complete_subscribers(
    logs: &OpenTelemetryLogSubscriber,
    metrics: &OpenTelemetryMetricSubscriber,
) -> Result<(), Box<dyn std::error::Error>> {
    let deregistration_results = [
        logs.deregister("otlp-logs"),
        metrics.deregister("otlp-metrics"),
    ];
    let flush_results = [logs.force_flush(), metrics.force_flush()];
    for diagnostic in logs.runtime_diagnostics().entries() {
        eprintln!("{}: {}", diagnostic.code, diagnostic.message);
    }
    let shutdown_results = [logs.shutdown(), metrics.shutdown()];
    for result in deregistration_results {
        result?;
    }
    for result in flush_results.into_iter().chain(shutdown_results) {
        result?;
    }
    Ok(())
}
