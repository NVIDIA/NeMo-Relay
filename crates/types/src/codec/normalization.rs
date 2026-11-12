// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider normalization supplied by a runtime or embedding application.

use crate::Json;
use crate::api::llm::LlmRequest;
use crate::codec::request::AnnotatedLlmRequest;
use crate::codec::response::AnnotatedLlmResponse;
use std::borrow::Cow;

/// A portable tool invocation and its sanitized provider metadata for telemetry.
#[derive(Debug, Clone)]
pub struct TelemetryToolCall {
    /// Invocation identifier used to correlate tool results.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Tool input; free-form input remains a JSON string.
    pub arguments: Json,
    /// Provider metadata from the same delivered invocation.
    pub extra: Option<Json>,
}

/// Best-effort request and response normalization for observability projections.
///
/// Implementations retain provider detection policy; an unsupported or invalid
/// payload returns `None` so event delivery can continue without an annotation.
pub trait LlmNormalizer: Send + Sync {
    /// Normalize a request, using a provider hint to resolve ambiguous payloads.
    fn normalize_request(
        &self,
        request: &LlmRequest,
        provider_hint: Option<&str>,
    ) -> Option<AnnotatedLlmRequest>;
    /// Normalize a provider's buffered response.
    fn normalize_response(&self, response: &Json) -> Option<AnnotatedLlmResponse>;

    /// Project recognized native request items for export without changing the
    /// annotation retained for provider encoding.
    fn telemetry_request<'a>(
        &self,
        request: Cow<'a, AnnotatedLlmRequest>,
    ) -> Cow<'a, AnnotatedLlmRequest> {
        request
    }

    /// Project invocation evidence from a delivered response. `None` leaves
    /// legacy exporter fallback available; an empty list explicitly has no calls.
    fn telemetry_tool_calls(
        &self,
        _annotation: Option<&AnnotatedLlmResponse>,
        _output: Option<&Json>,
    ) -> Option<Vec<TelemetryToolCall>> {
        None
    }
}

/// Select a field from a direct or replay-wrapped response message.
#[doc(hidden)]
pub fn raw_response_message_field<'a>(output: &'a Json, field: &str) -> Option<&'a Json> {
    let object = output.as_object()?;
    object
        .get("raw_response")
        .or(Some(output))
        .and_then(|raw_response| raw_response.as_object())
        .and_then(|raw_response| raw_response.get("choices"))
        .and_then(Json::as_array)
        .and_then(|choices| choices.first())
        .and_then(Json::as_object)
        .and_then(|choice| choice.get("message"))
        .and_then(Json::as_object)
        .and_then(|message| message.get(field))
}

/// Select the raw invocation array from supported manual response envelopes.
#[doc(hidden)]
pub fn tool_call_array(output: &Json) -> Option<&Vec<Json>> {
    output
        .as_object()
        .and_then(|object| object.get("tool_calls"))
        .and_then(Json::as_array)
        .or_else(|| {
            output
                .as_object()
                .and_then(|object| object.get("assistant_message"))
                .and_then(Json::as_object)
                .and_then(|assistant| assistant.get("tool_calls"))
                .and_then(Json::as_array)
        })
        .or_else(|| raw_response_message_field(output, "tool_calls").and_then(Json::as_array))
}
