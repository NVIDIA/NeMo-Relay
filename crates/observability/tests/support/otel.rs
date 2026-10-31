// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unused_imports)]
pub(crate) use crate::atif::{AtifAgentInfo, AtifExporter, AtifStepExtra};
use crate::otel as implementation;
pub(crate) use crate::otel::RelayIdGenerator;
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
fn metadata_string<'a>(event: &'a Event, key: &str) -> Option<&'a str> {
    implementation::metadata_string(event, key)
}

#[allow(dead_code)]
fn span_kind(event: &Event) -> SpanKind {
    implementation::span_kind(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn span_name(event: &Event) -> String {
    implementation::span_name(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn semantic_scope_type(event: &Event) -> Option<ScopeType> {
    implementation::semantic_scope_type(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn start_attributes(event: &Event) -> Vec<KeyValue> {
    implementation::start_attributes(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn end_attributes(event: &Event) -> Vec<KeyValue> {
    implementation::end_attributes(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn cost_from_llm_event(event: &Event) -> Option<(f64, String)> {
    implementation::cost_from_llm_event(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn mark_attributes(event: &Event) -> Vec<KeyValue> {
    implementation::mark_attributes(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn common_attributes(event: &Event) -> Vec<KeyValue> {
    implementation::common_attributes(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
pub(super) fn accept_otlp_timestamp(event: &Event, diagnostics: &SignalRuntimeDiagnostics) -> bool {
    implementation::accept_otlp_timestamp(&crate::test_support::event_view(event), diagnostics)
}

#[path = "../unit/otel_projection_tests.rs"]
mod projection_tests;
