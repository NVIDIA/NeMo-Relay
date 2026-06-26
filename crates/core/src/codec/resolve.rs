// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-surface detection and best-effort normalization: the preferred path
//! for turning raw provider JSON into normalized types when no codec annotation
//! is present.

use crate::api::llm::LlmRequest;
use crate::error::Result;
use crate::json::Json;

use super::request::AnnotatedLlmRequest;
use super::response::AnnotatedLlmResponse;
use super::{anthropic, openai_chat, openai_responses};

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

/// Request shape detector; the optional `&str` is a provider hint a codec may use
/// to claim an otherwise-ambiguous shape.
type RequestDetector = fn(&serde_json::Map<String, Json>, Option<&str>) -> bool;
type ResponseDetector = fn(&serde_json::Map<String, Json>) -> bool;

pub(crate) struct SurfaceDescriptor {
    pub(crate) surface: ProviderSurface,
    pub(crate) detect_request: RequestDetector,
    pub(crate) detect_response: ResponseDetector,
    pub(crate) decode_request: fn(&LlmRequest) -> Result<AnnotatedLlmRequest>,
    pub(crate) decode_response: fn(&Json) -> Result<AnnotatedLlmResponse>,
}

/// Built-in surfaces in request-detection priority order (first match wins):
/// Responses > Anthropic > Chat. The order is authoritative — a hint-aware
/// detector must stay after any stronger-signal surface it could shadow.
static REGISTRY: &[SurfaceDescriptor] = &[
    openai_responses::SURFACE_DESCRIPTOR,
    anthropic::SURFACE_DESCRIPTOR,
    openai_chat::SURFACE_DESCRIPTOR,
];

/// Detect the request surface from a raw request body by top-level key.
///
/// Priority: OpenAI Responses (`input`/`instructions`) > Anthropic Messages
/// (`system`) > OpenAI Chat (`messages`). `None` when no key matches or `body`
/// is not an object. This is a best-effort heuristic: an Anthropic request that
/// omits the optional top-level `system` is indistinguishable from OpenAI Chat
/// and classifies as `OpenAIChat`.
#[must_use]
pub fn detect_request_surface(body: &Json) -> Option<ProviderSurface> {
    detect_request_surface_with_hint(body, None)
}

/// Like [`detect_request_surface`], but a recognized `provider_hint` resolves the
/// one ambiguous shape (an Anthropic request without a top-level `system`,
/// otherwise read as OpenAI Chat). Today, `"anthropic"` is the only hint that
/// changes detection; `None` or any other value is ignored and detection stays
/// shape-only.
#[must_use]
pub fn detect_request_surface_with_hint(
    body: &Json,
    provider_hint: Option<&str>,
) -> Option<ProviderSurface> {
    let obj = body.as_object()?;
    REGISTRY
        .iter()
        .find(|d| (d.detect_request)(obj, provider_hint))
        .map(|d| d.surface)
}

/// Classify a response object to exactly one built-in surface descriptor: the
/// single source of truth shared by [`detect_response_surface`] and
/// [`normalize_response`]. Zero or multiple matches yield `None` (the built-in
/// codecs accept minimal objects, so decode success alone is not a reliable
/// classifier).
fn detect_response_descriptor(
    obj: &serde_json::Map<String, Json>,
) -> Option<&'static SurfaceDescriptor> {
    let mut matches = REGISTRY.iter().filter(|d| (d.detect_response)(obj));
    match (matches.next(), matches.next()) {
        (Some(descriptor), None) => Some(descriptor),
        _ => None,
    }
}

/// Detect the response surface from a raw provider response, classifying only
/// when exactly one built-in shape matches (the built-in codecs accept minimal
/// objects, so decode success alone is not a reliable classifier).
#[must_use]
pub fn detect_response_surface(raw: &Json) -> Option<ProviderSurface> {
    detect_response_descriptor(raw.as_object()?).map(|d| d.surface)
}

/// Best-effort decode of a raw request into [`AnnotatedLlmRequest`] (fail-open).
#[must_use]
pub fn normalize_request(request: &LlmRequest) -> Option<AnnotatedLlmRequest> {
    let obj = request.content.as_object()?;
    let descriptor = REGISTRY.iter().find(|d| (d.detect_request)(obj, None))?;
    (descriptor.decode_request)(request).ok()
}

/// Best-effort decode of a raw response into [`AnnotatedLlmResponse`] (fail-open).
#[must_use]
pub fn normalize_response(raw: &Json) -> Option<AnnotatedLlmResponse> {
    let descriptor = detect_response_descriptor(raw.as_object()?)?;
    (descriptor.decode_response)(raw).ok()
}

#[cfg(test)]
#[path = "../../tests/unit/codec/resolve_tests.rs"]
mod tests;
