// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unused_imports)]

//! Core integration test views for the extracted observability implementation.

use nemo_relay_observability::otel_metrics as implementation;

pub(crate) use implementation::*;

use super::OpenTelemetryTlsConfig;

use std::borrow::Cow;

use std::collections::{HashMap, HashSet};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use std::sync::{Arc, Mutex, MutexGuard};

#[cfg(test)]
use nemo_relay_runtime::api::event::MetricEnvelope;
use std::time::Duration;

use nemo_relay_runtime::api::event::{
    AttributeValue, Event, MetricAttributes, MetricKind, MetricMeasurement, MetricValue,
    MetricValueType, ValidatedMetricMeasurement,
};

use nemo_relay_runtime::api::runtime::EventSubscriberFn;

use nemo_relay_runtime::api::scope::ScopeType;

use nemo_relay_runtime::api::subscriber::{
    deregister_subscriber, flush_subscribers, register_subscriber,
};

use opentelemetry::metrics::{Counter, Gauge, Histogram, Meter, MeterProvider as _, UpDownCounter};

use opentelemetry::{Array, InstrumentationScope, KeyValue, Value};

use opentelemetry_otlp::{
    MetricExporter as OtlpMetricExporter, Protocol, WithExportConfig, WithHttpConfig,
    WithTonicConfig,
};

use opentelemetry_sdk::error::OTelSdkResult;

use opentelemetry_sdk::metrics::data::ResourceMetrics;

use opentelemetry_sdk::metrics::exporter::PushMetricExporter;

use opentelemetry_sdk::metrics::periodic_reader_with_async_runtime::PeriodicReader;

use opentelemetry_sdk::metrics::{SdkMeterProvider, Stream, Temporality};

use opentelemetry_sdk::runtime;

use serde::{Deserialize, Serialize};

use serde_json::json;

use sha2::{Digest, Sha256};

use super::header_file::{
    HeaderFileHttpClient, HeaderFileInterceptor, HeaderFileResolver, HeaderFiles, PlaintextPolicy,
    has_configured_headers, validate_header_files, validate_header_http_endpoint_typed,
    validate_otlp_transport_typed,
};

use super::otel::{
    DEFAULT_COMPLETED_SPAN_CONTEXT_TTL, ExporterInitializationError, ExporterInitializationResult,
    OpenTelemetryError, OtlpTransport, Result, normalize_shutdown_result,
};

use super::otel_signal::{
    MAX_DYNAMIC_SIGNAL_PIPELINES, MetricMarkClassification, SignalExporterRuntime,
    SignalResourceLineage, SignalRuntimeDiagnostics, automatic_otlp_http_client,
    automatic_protocol_is_grpc, automatic_protocol_is_unset, automatic_signal_endpoint,
    build_grpc_metadata, build_in_owned_runtime, classify_metric_mark,
    promoted_signal_resource_attributes, record_resource_pipeline_limit,
    reject_signal_header_environment, resolve_header_env, resolve_http_signal_endpoint,
    should_relog_runtime_diagnostic, signal_resource, telemetry_resource, validate_signal_headers,
    validate_telemetry_sdk_resource_attributes,
};

use super::{ExporterRuntimeDiagnostics, validate_metadata_promotion_prefixes};

#[cfg(test)]
#[path = "../../integration/observability/otel_metrics_tests.rs"]
mod tests;
