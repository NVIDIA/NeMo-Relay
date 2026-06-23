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
    OpenAiChat,
    /// OpenAI Responses.
    OpenAiResponses,
    /// Anthropic Messages.
    AnthropicMessages,
}

impl ProviderSurface {
    /// Map a canonical codec name (the `ApiSpecificResponse` serde tags) to a surface.
    #[must_use]
    pub fn from_codec_name(name: &str) -> Option<Self> {
        match name {
            "openai_chat" => Some(Self::OpenAiChat),
            "openai_responses" => Some(Self::OpenAiResponses),
            "anthropic_messages" => Some(Self::AnthropicMessages),
            _ => None,
        }
    }

    /// The canonical codec name for this surface.
    #[must_use]
    pub fn codec_name(self) -> &'static str {
        match self {
            Self::OpenAiChat => "openai_chat",
            Self::OpenAiResponses => "openai_responses",
            Self::AnthropicMessages => "anthropic_messages",
        }
    }
}

/// Detect the request surface from a raw request body, using the request codecs'
/// priority order; `None` when no key matches or `body` is not an object.
#[must_use]
pub fn detect_request_surface(body: &Json) -> Option<ProviderSurface> {
    let obj = body.as_object()?;
    if obj.contains_key("input") || obj.contains_key("instructions") {
        Some(ProviderSurface::OpenAiResponses)
    } else if obj.contains_key("system") {
        Some(ProviderSurface::AnthropicMessages)
    } else if obj.contains_key("messages") {
        Some(ProviderSurface::OpenAiChat)
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
        (true, false, false) => Some(ProviderSurface::OpenAiChat),
        (false, true, false) => Some(ProviderSurface::OpenAiResponses),
        (false, false, true) => Some(ProviderSurface::AnthropicMessages),
        _ => None,
    }
}

/// Best-effort decode of a raw request into [`AnnotatedLlmRequest`] (fail-open).
#[must_use]
pub fn normalize_request(
    request: &LlmRequest,
    hint: Option<ProviderSurface>,
) -> Option<AnnotatedLlmRequest> {
    let surface = hint.or_else(|| detect_request_surface(&request.content))?;
    match surface {
        ProviderSurface::OpenAiChat => OpenAIChatCodec.decode(request),
        ProviderSurface::OpenAiResponses => OpenAIResponsesCodec.decode(request),
        ProviderSurface::AnthropicMessages => AnthropicMessagesCodec.decode(request),
    }
    .ok()
}

/// Best-effort decode of a raw response into [`AnnotatedLlmResponse`] (fail-open).
#[must_use]
pub fn normalize_response(
    raw: &Json,
    hint: Option<ProviderSurface>,
) -> Option<AnnotatedLlmResponse> {
    let surface = hint.or_else(|| detect_response_surface(raw))?;
    match surface {
        ProviderSurface::OpenAiChat => OpenAIChatCodec.decode_response(raw),
        ProviderSurface::OpenAiResponses => OpenAIResponsesCodec.decode_response(raw),
        ProviderSurface::AnthropicMessages => AnthropicMessagesCodec.decode_response(raw),
    }
    .ok()
}

#[cfg(test)]
#[path = "../../tests/unit/codec/resolve_tests.rs"]
mod tests;
