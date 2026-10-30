// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exporter constructors bound to the core runtime's scopes, codecs, and diagnostics.
//!
//! Configuration and exporter types live in `nemo-relay-observability`. These
//! constructors supply the runtime host internally; exporters are ready to
//! register or receive events immediately.

use nemo_relay_observability::{atif, atof};

/// Create an ATIF exporter using core's provider codecs and scope context.
pub fn atif(session_id: String, agent_info: atif::AtifAgentInfo) -> atif::AtifExporter {
    atif::AtifExporter::new(session_id, agent_info)
        .with_host(crate::observability_runtime::core_host())
}

/// Create an ATOF exporter ready for core subscriber registration.
pub fn atof(config: atof::AtofExporterConfig) -> atof::Result<atof::AtofExporter> {
    atof::AtofExporter::new(config)
        .map(|exporter| exporter.with_host(crate::observability_runtime::core_host()))
}

#[allow(dead_code)]
pub fn opentelemetry_from_provider(
    provider: opentelemetry_sdk::trace::SdkTracerProvider,
    scope: impl Into<String>,
) -> nemo_relay_observability::otel::OpenTelemetrySubscriber {
    opentelemetry_from_provider_with_type(
        provider,
        scope,
        nemo_relay_observability::OpenTelemetryType::Full,
    )
}
pub fn opentelemetry_from_provider_with_type(
    provider: opentelemetry_sdk::trace::SdkTracerProvider,
    scope: impl Into<String>,
    projection: nemo_relay_observability::OpenTelemetryType,
) -> nemo_relay_observability::otel::OpenTelemetrySubscriber {
    nemo_relay_observability::otel::OpenTelemetrySubscriber::from_tracer_provider_with_type(
        provider, scope, projection,
    )
    .with_host(crate::observability_runtime::core_host())
}
pub fn opentelemetry_logs(
    config: nemo_relay_observability::otel_logs::OpenTelemetryLogConfig,
) -> nemo_relay_observability::otel::Result<
    nemo_relay_observability::otel_logs::OpenTelemetryLogSubscriber,
> {
    nemo_relay_observability::otel_logs::OpenTelemetryLogSubscriber::new(config)
        .map(|exporter| exporter.with_host(crate::observability_runtime::core_host()))
}
pub fn opentelemetry_metrics(
    config: nemo_relay_observability::otel_metrics::OpenTelemetryMetricConfig,
) -> nemo_relay_observability::otel::Result<
    nemo_relay_observability::otel_metrics::OpenTelemetryMetricSubscriber,
> {
    nemo_relay_observability::otel_metrics::OpenTelemetryMetricSubscriber::new(config)
        .map(|exporter| exporter.with_host(crate::observability_runtime::core_host()))
}
