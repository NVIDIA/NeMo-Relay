// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unused_imports)]
pub(crate) use crate::atif::{AtifAgentInfo, AtifExporter, AtifStepExtra};
pub(crate) use crate::otel::RelayIdGenerator;
use crate::otel_genai as implementation;
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
fn has_gen_ai_semantics(event: &Event) -> bool {
    implementation::has_gen_ai_semantics(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn is_agent_turn(event: &Event) -> bool {
    implementation::is_agent_turn(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
pub(super) fn span_name(event: &Event) -> String {
    implementation::span_name(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
pub(super) fn span_kind(event: &Event) -> SpanKind {
    implementation::span_kind(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
pub(super) fn start_attributes(event: &Event) -> Vec<KeyValue> {
    implementation::start_attributes(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
pub(super) fn end_attributes(event: &Event) -> Vec<KeyValue> {
    implementation::end_attributes(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
pub(super) fn client_metric_attributes(event: &Event) -> Option<Json> {
    implementation::client_metric_attributes(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn push_embedding_response_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    implementation::push_embedding_response_attributes(
        attributes,
        &crate::test_support::event_view(event),
    )
}

#[allow(dead_code)]
fn operation_name(event: &Event) -> &'static str {
    implementation::operation_name(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn llm_operation_name(event: &Event) -> &'static str {
    implementation::llm_operation_name(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn push_provider_and_server_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    implementation::push_provider_and_server_attributes(
        attributes,
        &crate::test_support::event_view(event),
    )
}

#[allow(dead_code)]
fn push_conversation_attribute(attributes: &mut Vec<KeyValue>, event: &Event) {
    implementation::push_conversation_attribute(attributes, &crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn push_agent_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    implementation::push_agent_attributes(attributes, &crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn push_explicit_internal_agent_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    implementation::push_explicit_internal_agent_attributes(
        attributes,
        &crate::test_support::event_view(event),
    )
}

#[allow(dead_code)]
fn push_model_attribute(attributes: &mut Vec<KeyValue>, event: &Event) {
    implementation::push_model_attribute(attributes, &crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn push_embedding_request_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    implementation::push_embedding_request_attributes(
        attributes,
        &crate::test_support::event_view(event),
    )
}

#[allow(dead_code)]
fn push_llm_request_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    implementation::push_llm_request_attributes(attributes, &crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn push_llm_response_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    implementation::push_llm_response_attributes(
        attributes,
        &crate::test_support::event_view(event),
    )
}

#[allow(dead_code)]
fn request_reasoning_level(
    event: &Event,
    request: &nemo_relay_types::codec::request::AnnotatedLlmRequest,
) -> Option<String> {
    implementation::request_reasoning_level(&crate::test_support::event_view(event), request)
}

#[allow(dead_code)]
fn requested_output_type(
    event: &Event,
    api_specific: Option<&ApiSpecificRequest>,
) -> Option<String> {
    implementation::requested_output_type(&crate::test_support::event_view(event), api_specific)
}

#[allow(dead_code)]
fn reasoning_output_tokens(event: &Event, response: &AnnotatedLlmResponse) -> Option<i64> {
    implementation::reasoning_output_tokens(&crate::test_support::event_view(event), response)
}

#[allow(dead_code)]
fn gen_ai_input_tokens(event: &Event, response: &AnnotatedLlmResponse) -> Option<i64> {
    implementation::gen_ai_input_tokens(&crate::test_support::event_view(event), response)
}

#[allow(dead_code)]
fn push_tool_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    implementation::push_tool_attributes(attributes, &crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn push_tool_metadata(attributes: &mut Vec<KeyValue>, event: &Event) {
    implementation::push_tool_metadata(attributes, &crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn tool_metadata_string(event: &Event, keys: &[&str]) -> Option<String> {
    implementation::tool_metadata_string(&crate::test_support::event_view(event), keys)
}

#[allow(dead_code)]
fn tool_metadata_with_origin(event: &Event, keys: &[&str]) -> Option<(String, bool)> {
    implementation::tool_metadata_with_origin(&crate::test_support::event_view(event), keys)
}

#[allow(dead_code)]
pub(super) fn inferred_tool_attribute_keys(event: &Event) -> std::collections::HashSet<String> {
    implementation::inferred_tool_attribute_keys(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn push_tool_definitions(attributes: &mut Vec<KeyValue>, event: &Event) {
    implementation::push_tool_definitions(attributes, &crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn push_retrieval_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    implementation::push_retrieval_attributes(attributes, &crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn request_model(event: &Event) -> Option<String> {
    implementation::request_model(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn provider_name(event: &Event) -> Option<String> {
    implementation::provider_name(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn provider_from_event_name(event: &Event) -> Option<String> {
    implementation::provider_from_event_name(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn provider_from_normalized_request(event: &Event) -> Option<&'static str> {
    implementation::provider_from_normalized_request(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn agent_name(event: &Event) -> String {
    implementation::agent_name(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn tool_name(event: &Event) -> String {
    implementation::tool_name(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn data_source_id(event: &Event) -> Option<String> {
    implementation::data_source_id(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn push_error_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    implementation::push_error_attributes(attributes, &crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn scalar_string(event: &Event, keys: &[&str]) -> Option<String> {
    implementation::scalar_string(&crate::test_support::event_view(event), keys)
}

#[allow(dead_code)]
fn scalar_i64(event: &Event, keys: &[&str]) -> Option<i64> {
    implementation::scalar_i64(&crate::test_support::event_view(event), keys)
}

#[allow(dead_code)]
fn semantic_string(event: &Event, key: &str) -> Option<String> {
    implementation::semantic_string(&crate::test_support::event_view(event), key)
}

#[allow(dead_code)]
fn string_values(event: &Event, keys: &[&str]) -> Option<Vec<String>> {
    implementation::string_values(&crate::test_support::event_view(event), keys)
}

#[allow(dead_code)]
fn find_scalar<T>(event: &Event, keys: &[&str], convert: impl Fn(&Json) -> Option<T>) -> Option<T> {
    implementation::find_scalar(&crate::test_support::event_view(event), keys, convert)
}

#[allow(dead_code)]
fn event_objects(event: &Event) -> Vec<&Map<String, Json>> {
    implementation::event_objects(&crate::test_support::event_view(event))
}
