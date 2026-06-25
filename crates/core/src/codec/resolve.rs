// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-surface detection and best-effort normalization: the preferred path
//! for turning raw provider JSON into normalized types when no codec annotation
//! is present.

use crate::api::llm::LlmRequest;
use crate::json::Json;

use super::anthropic::AnthropicMessagesCodec;
use super::openai_chat::OpenAIChatCodec;
use super::openai_responses::OpenAIResponsesCodec;
use super::request::AnnotatedLlmRequest;
use super::response::AnnotatedLlmResponse;
use super::traits::{LlmCodec, LlmResponseCodec};

/// A built-in provider request/response surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderSurface {
    /// OpenAI Chat Completions.
    OpenAIChat,
    /// OpenAI Responses.
    OpenAIResponses,
    /// Anthropic Messages.
    AnthropicMessages,
}

/// Detect the request surface from a raw request body by top-level key.
///
/// Priority: OpenAI Responses (`input`/`instructions`) > Anthropic Messages
/// (`system`) > OpenAI Chat (`messages`). `None` when no key matches or `body`
/// is not an object. This is a best-effort heuristic: an Anthropic request that
/// omits the optional top-level `system` is indistinguishable from OpenAI Chat
/// and classifies as `OpenAIChat`.
#[must_use]
pub fn detect_request_surface(body: &Json) -> Option<ProviderSurface> {
    let obj = body.as_object()?;
    if obj.contains_key("input") || obj.contains_key("instructions") {
        Some(ProviderSurface::OpenAIResponses)
    } else if obj.contains_key("system") {
        Some(ProviderSurface::AnthropicMessages)
    } else if obj.contains_key("messages") {
        Some(ProviderSurface::OpenAIChat)
    } else {
        None
    }
}

/// Detect the response surface from a raw provider response, classifying only
/// when exactly one built-in shape matches (the built-in codecs accept minimal
/// objects, so decode success alone is not a reliable classifier).
#[must_use]
pub fn detect_response_surface(raw: &Json) -> Option<ProviderSurface> {
    let obj = raw.as_object()?;
    let is_chat = obj.get("choices").is_some_and(Json::is_array);
    let is_responses = obj.get("output").is_some_and(Json::is_array)
        || obj.get("output_text").is_some_and(Json::is_string);
    let is_anthropic = obj.get("type").and_then(Json::as_str) == Some("message")
        && obj.get("content").is_some_and(Json::is_array);

    match (is_chat, is_responses, is_anthropic) {
        (true, false, false) => Some(ProviderSurface::OpenAIChat),
        (false, true, false) => Some(ProviderSurface::OpenAIResponses),
        (false, false, true) => Some(ProviderSurface::AnthropicMessages),
        _ => None,
    }
}

/// Best-effort decode of a raw request into [`AnnotatedLlmRequest`] (fail-open).
#[must_use]
pub fn normalize_request(request: &LlmRequest) -> Option<AnnotatedLlmRequest> {
    match detect_request_surface(&request.content)? {
        ProviderSurface::OpenAIChat => OpenAIChatCodec.decode(request),
        ProviderSurface::OpenAIResponses => OpenAIResponsesCodec.decode(request),
        ProviderSurface::AnthropicMessages => AnthropicMessagesCodec.decode(request),
    }
    .ok()
}

/// Best-effort decode of a raw response into [`AnnotatedLlmResponse`] (fail-open).
#[must_use]
pub fn normalize_response(raw: &Json) -> Option<AnnotatedLlmResponse> {
    match detect_response_surface(raw)? {
        ProviderSurface::OpenAIChat => OpenAIChatCodec.decode_response(raw),
        ProviderSurface::OpenAIResponses => OpenAIResponsesCodec.decode_response(raw),
        ProviderSurface::AnthropicMessages => AnthropicMessagesCodec.decode_response(raw),
    }
    .ok()
}

#[cfg(test)]
#[path = "../../tests/unit/codec/resolve_tests.rs"]
mod tests;
