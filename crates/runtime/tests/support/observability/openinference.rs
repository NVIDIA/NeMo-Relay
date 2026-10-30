// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unused_imports)]

//! Core integration test views for the extracted observability implementation.

use nemo_relay_observability::openinference as implementation;

pub(crate) use implementation::*;

use super::{
    estimate_cost_for_response_or_model, estimate_cost_for_response_or_requested_model,
    input_tokens_including_cache, manual, merge_usage, model_name_for_llm_event,
    push_serialized_top_level_attributes, push_tool_result_annotation_attribute,
    push_top_level_json_attributes, total_tokens_including_cache,
};

use crate::api::event::{Event, EventNormalizationExt};

use crate::api::scope::ScopeType;

use nemo_relay_types::codec::request::{
    AnnotatedLlmRequest, ContentPart, Message, MessageContent, ToolDefinition,
};

use nemo_relay_types::codec::response::{
    AnnotatedLlmResponse, FinishReason, ResponseToolCall, Usage,
};

use crate::json::Json;
#[cfg(test)]
use chrono::{DateTime, Utc};

use opentelemetry::KeyValue;
#[cfg(test)]
use opentelemetry::trace::SpanContext;

use opentelemetry::trace::SpanKind;

use serde::Serialize;
#[cfg(test)]
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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

#[cfg(test)]
#[path = "../../integration/observability/openinference_tests.rs"]
mod tests;
