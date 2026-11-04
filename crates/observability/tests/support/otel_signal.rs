// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unused_imports)]
pub(crate) use crate::atif::{AtifAgentInfo, AtifExporter, AtifStepExtra};
pub(crate) use crate::otel::RelayIdGenerator;
use crate::otel_signal as implementation;
pub(crate) use crate::otel_signal::SignalRuntimeDiagnostics;
pub(crate) use crate::{
    MarkProjection, OpenTelemetryType, default_mark_exclude_names, relay_span_id, relay_trace_id,
};
use chrono::{DateTime, Utc};
pub(crate) use implementation::*;
use nemo_relay_types::Json;
use nemo_relay_types::api::event::*;
use nemo_relay_types::api::scope::*;
use nemo_relay_types::api::tool::*;
use nemo_relay_types::codec::request::*;
use nemo_relay_types::codec::response::*;
use opentelemetry::logs::AnyValue;
use opentelemetry::logs::{LogRecord, Logger, LoggerProvider, Severity};
use opentelemetry::metrics::MeterProvider;
use opentelemetry::{
    Key,
    trace::{SpanContext, SpanId, TraceFlags},
};
use opentelemetry::{KeyValue, trace::SpanKind};
use opentelemetry_sdk::{metrics::SdkMeterProvider, trace::SdkTracerProvider};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::UNIX_EPOCH;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

#[allow(dead_code)]
pub(super) fn classify_metric_mark(event: &Event) -> MetricMarkClassification {
    implementation::classify_metric_mark(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
pub(super) fn promoted_signal_resource_attributes(
    event: &Event,
    prefixes: &[String],
    service_name: Option<&str>,
    service_namespace: Option<&str>,
    service_version: Option<&str>,
    resource_attributes: &HashMap<String, String>,
    runtime_diagnostics: &SignalRuntimeDiagnostics,
) -> Option<(String, Vec<KeyValue>)> {
    implementation::promoted_signal_resource_attributes(
        &crate::test_support::event_view(event),
        prefixes,
        service_name,
        service_namespace,
        service_version,
        resource_attributes,
        runtime_diagnostics,
    )
}
