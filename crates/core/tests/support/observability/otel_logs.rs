// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unused_imports)]

//! Core integration test views for the extracted observability implementation.

use nemo_relay_observability::otel_logs as implementation;

pub(crate) use implementation::*;

use std::borrow::Cow;

use std::collections::{BTreeMap, HashMap, HashSet};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use std::sync::{Arc, Mutex};

use std::time::{Duration, SystemTime};

use chrono::{DateTime, Utc};

use opentelemetry::logs::{AnyValue, LogRecord as _, Logger as _, LoggerProvider as _, Severity};

use opentelemetry::trace::{SpanContext, TraceFlags, TraceState};

use opentelemetry::{InstrumentationScope, Key, KeyValue};

use opentelemetry_otlp::{
    LogExporter as OtlpLogExporter, Protocol, WithExportConfig, WithHttpConfig, WithTonicConfig,
};

use opentelemetry_sdk::logs::log_processor_with_async_runtime::BatchLogProcessor as AsyncBatchLogProcessor;

use opentelemetry_sdk::logs::{
    BatchConfigBuilder, LogBatch, LogExporter, LogProcessor, SdkLogRecord, SdkLogger,
    SdkLoggerProvider,
};

use opentelemetry_sdk::{Resource, error::OTelSdkResult, runtime};

use serde_json::{Map, Value as Json};

use uuid::Uuid;

use super::OpenTelemetryTlsConfig;

use crate::api::event::{ATOF_VERSION, Event, LOG_SEVERITY_METADATA_KEY, LogSeverity};

use crate::api::runtime::EventSubscriberFn;

use crate::api::runtime::scope_stack::w3c_span_context;

use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};

use crate::observability_test_support::{relay_span_id, relay_trace_id};

use crate::plugin::OTEL_RUNTIME_DELIVERY_FAILURE_MARKER;

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
    build_grpc_metadata, build_in_owned_runtime, classify_metric_mark, drain_startup_queue,
    promoted_signal_resource_attributes, record_resource_pipeline_limit,
    reject_signal_header_environment, resolve_header_env, resolve_http_signal_endpoint,
    retry_batch_processor_channel_full, should_relog_runtime_diagnostic, signal_resource,
    telemetry_resource, validate_signal_headers, validate_telemetry_sdk_resource_attributes,
};

use super::{ExporterRuntimeDiagnostics, validate_metadata_promotion_prefixes};

#[cfg(test)]
#[path = "../../integration/observability/otel_logs_tests.rs"]
mod tests;
