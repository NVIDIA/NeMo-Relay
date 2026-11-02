// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unused_imports)]

//! Core integration test views for the extracted observability implementation.

use nemo_relay_observability::otel_genai as implementation;

pub(crate) use implementation::*;

use nemo_relay_runtime::api::event::{Event, EventNormalizationExt};

use nemo_relay_runtime::api::scope::ScopeType;

use nemo_relay_types::codec::request::{
    ApiSpecificRequest, ContentPart, Message, MessageContent, tool_definition_identities,
};

use nemo_relay_types::codec::response::{AnnotatedLlmResponse, ApiSpecificResponse, FinishReason};

use nemo_relay_types::Json;

use opentelemetry::KeyValue;

use opentelemetry::trace::SpanKind;

use opentelemetry_semantic_conventions::attribute as semconv;

use serde_json::{Map, Value};

#[allow(dead_code)]
pub(super) fn span_name(event: &Event) -> String {
    implementation::span_name(&super::event_view(event))
}

#[allow(dead_code)]
pub(super) fn start_attributes(event: &Event) -> Vec<KeyValue> {
    implementation::start_attributes(&super::event_view(event))
}

#[allow(dead_code)]
pub(super) fn end_attributes(event: &Event) -> Vec<KeyValue> {
    implementation::end_attributes(&super::event_view(event))
}

#[allow(dead_code)]
fn tool_name(event: &Event) -> String {
    implementation::tool_name(&super::event_view(event))
}
