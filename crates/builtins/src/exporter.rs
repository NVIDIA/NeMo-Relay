// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exporter constructors bound to the execution runtime's scopes, codecs, and diagnostics.
//!
//! Configuration and exporter types live in `nemo-relay-observability`. These
//! constructors supply the runtime host internally; exporters are ready to
//! register or receive events immediately.

use nemo_relay_observability::{atif, atof, otel, otel_logs, otel_metrics};

/// Create an ATIF exporter using the runtime's provider codecs and scope context.
pub fn atif(session_id: String, agent_info: atif::AtifAgentInfo) -> atif::AtifExporter {
    crate::initialize_runtime();
    nemo_relay_plugin_host::exporter::atif(session_id, agent_info)
}

/// Create an ATOF exporter ready for core subscriber registration.
pub fn atof(config: atof::AtofExporterConfig) -> atof::Result<atof::AtofExporter> {
    crate::initialize_runtime();
    nemo_relay_plugin_host::exporter::atof(config)
}

/// Create an OpenTelemetry or OpenInference trace subscriber for core events.
pub fn opentelemetry(
    config: otel::OpenTelemetryConfig,
) -> otel::Result<otel::OpenTelemetrySubscriber> {
    crate::initialize_runtime();
    nemo_relay_plugin_host::exporter::opentelemetry(config)
}

/// Create an OTLP file subscriber using the runtime's provider codecs and scope context.
pub fn opentelemetry_file(
    config: otel::OpenTelemetryFileSinkConfig,
) -> otel::Result<otel::OpenTelemetrySubscriber> {
    crate::initialize_runtime();
    nemo_relay_plugin_host::exporter::opentelemetry_file(config)
}

/// Create an OpenTelemetry log subscriber for core events.
pub fn opentelemetry_logs(
    config: otel_logs::OpenTelemetryLogConfig,
) -> otel::Result<otel_logs::OpenTelemetryLogSubscriber> {
    crate::initialize_runtime();
    nemo_relay_plugin_host::exporter::opentelemetry_logs(config)
}

/// Create an OpenTelemetry metric subscriber for core events.
pub fn opentelemetry_metrics(
    config: otel_metrics::OpenTelemetryMetricConfig,
) -> otel::Result<otel_metrics::OpenTelemetryMetricSubscriber> {
    crate::initialize_runtime();
    nemo_relay_plugin_host::exporter::opentelemetry_metrics(config)
}

/// Create a runtime-bound subscriber from a configured tracer provider.
pub fn opentelemetry_from_provider(
    provider: opentelemetry_sdk::trace::SdkTracerProvider,
    instrumentation_scope: impl Into<String>,
) -> otel::OpenTelemetrySubscriber {
    crate::initialize_runtime();
    nemo_relay_plugin_host::exporter::opentelemetry_from_provider(provider, instrumentation_scope)
}

/// Create a runtime-bound subscriber with a selected trace projection.
pub fn opentelemetry_from_provider_with_type(
    provider: opentelemetry_sdk::trace::SdkTracerProvider,
    instrumentation_scope: impl Into<String>,
    projection: nemo_relay_observability::OpenTelemetryType,
) -> otel::OpenTelemetrySubscriber {
    crate::initialize_runtime();
    nemo_relay_plugin_host::exporter::opentelemetry_from_provider_with_type(
        provider,
        instrumentation_scope,
        projection,
    )
}
