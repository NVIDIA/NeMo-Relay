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
type RequestSurfaceDetector = fn(&serde_json::Map<String, Json>, Option<&str>) -> bool;

/// Response shape detector; response routing is payload-only because provider
/// responses carry stronger built-in discriminators than request bodies.
type ResponseSurfaceDetector = fn(&serde_json::Map<String, Json>) -> bool;

/// Built-in provider extraction strategy for one request/response surface.
///
/// The descriptor keeps surface detection next to the codec that owns the
/// schema-specific decode logic while preserving the existing public
/// [`LlmCodec`](super::traits::LlmCodec) and
/// [`LlmResponseCodec`](super::traits::LlmResponseCodec) traits.
/// `decode_response` is the provider response-extraction interface: built-in
/// codecs populate [`AnnotatedLlmResponse`] with model names, finish reasons,
/// tool calls, usage, cost, provider-specific fields, and replayable response
/// data when the source payload supplies them.
pub(crate) struct ProviderSurfaceDescriptor {
    pub(crate) surface: ProviderSurface,
    pub(crate) detect_request: RequestSurfaceDetector,
    pub(crate) detect_response: ResponseSurfaceDetector,
    pub(crate) decode_request: fn(&LlmRequest) -> Result<AnnotatedLlmRequest>,
    pub(crate) decode_response: fn(&Json) -> Result<AnnotatedLlmResponse>,
}

/// Built-in provider surfaces in request-detection priority order.
///
/// First match wins for requests because some shapes overlap. The order is
/// authoritative: a hint-aware detector must stay after any stronger-signal
/// surface it could shadow. Response detection requires exactly one match
/// before decoding.
pub(crate) static BUILTIN_PROVIDER_SURFACES: &[ProviderSurfaceDescriptor] = &[
    openai_responses::PROVIDER_SURFACE,
    anthropic::PROVIDER_SURFACE,
    openai_chat::PROVIDER_SURFACE,
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
/// otherwise read as OpenAI Chat). Today, `"anthropic"` and provider paths such
/// as `"anthropic.messages"` are the only hints that change detection; `None`
/// or any other value is ignored and detection stays shape-only.
#[must_use]
pub fn detect_request_surface_with_hint(
    body: &Json,
    provider_hint: Option<&str>,
) -> Option<ProviderSurface> {
    request_descriptor(body, provider_hint).map(|descriptor| descriptor.surface)
}

/// Detect the response surface from a raw provider response, classifying only
/// when exactly one built-in shape matches (the built-in codecs accept minimal
/// objects, so decode success alone is not a reliable classifier).
#[must_use]
pub fn detect_response_surface(raw: &Json) -> Option<ProviderSurface> {
    response_descriptor(raw).map(|descriptor| descriptor.surface)
}

fn request_descriptor(
    body: &Json,
    provider_hint: Option<&str>,
) -> Option<&'static ProviderSurfaceDescriptor> {
    let obj = body.as_object()?;
    BUILTIN_PROVIDER_SURFACES
        .iter()
        .find(|descriptor| (descriptor.detect_request)(obj, provider_hint))
}

fn response_descriptor(raw: &Json) -> Option<&'static ProviderSurfaceDescriptor> {
    let obj = raw.as_object()?;
    let mut matches = BUILTIN_PROVIDER_SURFACES
        .iter()
        .filter(|descriptor| (descriptor.detect_response)(obj));
    match (matches.next(), matches.next()) {
        (Some(descriptor), None) => Some(descriptor),
        _ => None,
    }
}

/// Best-effort decode of a raw request into [`AnnotatedLlmRequest`] (fail-open).
#[must_use]
pub fn normalize_request(request: &LlmRequest) -> Option<AnnotatedLlmRequest> {
    normalize_request_with_hint(request, None)
}

/// Like [`normalize_request`], but a recognized `provider_hint` can
/// disambiguate provider request shapes that are otherwise identical.
#[must_use]
pub fn normalize_request_with_hint(
    request: &LlmRequest,
    provider_hint: Option<&str>,
) -> Option<AnnotatedLlmRequest> {
    let descriptor = request_descriptor(&request.content, provider_hint)?;
    (descriptor.decode_request)(request).ok()
}

/// Best-effort decode of a raw response into [`AnnotatedLlmResponse`] (fail-open).
#[must_use]
pub fn normalize_response(raw: &Json) -> Option<AnnotatedLlmResponse> {
    let descriptor = response_descriptor(raw)?;
    (descriptor.decode_response)(raw).ok()
}

#[cfg(test)]
#[path = "../../tests/unit/codec/resolve_tests.rs"]
mod tests;
