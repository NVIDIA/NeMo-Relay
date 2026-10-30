// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unused_imports)]

//! Core integration test views for the extracted observability implementation.

use nemo_relay_observability::otel as implementation;

pub(crate) use implementation::*;

use std::borrow::Cow;

use std::cell::RefCell;

use std::collections::{BTreeMap, HashMap, HashSet};

use std::path::{Component, Path, PathBuf};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use std::sync::mpsc;

use std::sync::{Arc, Mutex};

use std::thread;

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::OpenTelemetryTlsConfig;

use super::header_file::{
    HeaderFileHttpClient, HeaderFileInterceptor, HeaderFileResolver, HeaderFiles, PlaintextPolicy,
    validate_header_files, validate_otlp_transport_typed,
};

use super::otel_file::{OtlpFileFormat, OtlpFileSpanExporter};

use super::otel_signal::{
    MetricMarkClassification, SignalRuntimeDiagnostics, TELEMETRY_SDK_RESOURCE_ATTRIBUTE_KEYS,
    automatic_otlp_http_client, automatic_protocol_is_grpc, automatic_protocol_is_unset,
    automatic_signal_endpoint, classify_metric_mark, drain_startup_queue, resolve_header_env,
    retry_batch_processor_channel_full, should_relog_runtime_diagnostic, telemetry_resource,
    validate_telemetry_sdk_resource_attributes,
};

use super::{
    ExporterRuntimeDiagnostics, MarkProjection, OpenTelemetryType, OtlpAttributeMapping,
    apply_attribute_mappings, attribute_mapping_aliases, attribute_mapping_inputs,
    default_mark_exclude_names, effective_mark_projection, estimate_cost_for_response_or_model,
    estimate_cost_for_response_or_requested_model, manual, model_name_for_llm_event,
    promote_event_metadata_attributes, push_serialized_top_level_attributes,
    push_session_identity_attributes, push_tool_result_annotation_attribute,
    push_top_level_json_attributes, relay_span_id, relay_trace_id, validate_attribute_mappings,
    validate_metadata_promotion_prefixes,
};

use crate::api::event::{Event, EventNormalizationExt, ScopeCategory};

use crate::api::runtime::EventSubscriberFn;

use crate::api::runtime::scope_stack::w3c_span_context;

use crate::api::scope::ScopeType;

use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};

use nemo_relay_types::codec::response::CostEstimate;

use crate::error::FlowError;

use chrono::{DateTime, Utc};

use opentelemetry::trace::{
    Span as _, SpanContext, SpanId, SpanKind, TraceContextExt, TraceFlags, TraceId, TraceState,
    Tracer, TracerProvider as _,
};

use opentelemetry::{Context, KeyValue};

use opentelemetry_otlp::{
    Protocol, SpanExporter as OtlpSpanExporter, WithExportConfig, WithHttpConfig,
};

use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};

use opentelemetry_sdk::trace::span_processor_with_async_runtime::BatchSpanProcessor as AsyncBatchSpanProcessor;

use opentelemetry_sdk::trace::{
    BatchConfigBuilder, IdGenerator, RandomIdGenerator, SdkTracer, SdkTracerProvider, Span,
    SpanData, SpanExporter, SpanProcessor,
};

use opentelemetry_sdk::{Resource, runtime};

use uuid::Uuid;

use crate::plugin::OTEL_RUNTIME_DELIVERY_FAILURE_MARKER;

use opentelemetry_otlp::WithTonicConfig;

use tonic::metadata::{MetadataKey, MetadataMap, MetadataValue};

#[allow(dead_code)]
fn span_name(event: &Event) -> String {
    implementation::span_name(&super::event_view(event))
}

#[allow(dead_code)]
fn start_attributes(event: &Event) -> Vec<KeyValue> {
    implementation::start_attributes(&super::event_view(event))
}

#[allow(dead_code)]
fn end_attributes(event: &Event) -> Vec<KeyValue> {
    implementation::end_attributes(&super::event_view(event))
}

#[cfg(test)]
#[path = "../../integration/observability/otel_tests.rs"]
mod tests;
