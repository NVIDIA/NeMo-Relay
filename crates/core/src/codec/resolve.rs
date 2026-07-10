// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-surface detection, best-effort normalization, and construction of
//! the matching built-in codecs from a raw payload, surface, or codec name.

use std::sync::Arc;

use crate::api::llm::LlmRequest;
use crate::error::Result;
use crate::json::Json;

use super::request::AnnotatedLlmRequest;
use super::response::AnnotatedLlmResponse;
use super::streaming::StreamingCodec;
use super::traits::{LlmCodec, LlmResponseCodec};
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
    pub(crate) codec_name: &'static str,
    pub(crate) request_codec: fn() -> Arc<dyn LlmCodec>,
    pub(crate) response_codec: fn() -> Arc<dyn LlmResponseCodec>,
    pub(crate) streaming_codec: fn() -> Box<dyn StreamingCodec>,
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
/// otherwise read as OpenAI Chat). Today, only the exact hints `"anthropic"`
/// and `"anthropic.messages"` change detection; `None` or any other value is
/// ignored and detection stays shape-only.
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

fn descriptor_for(surface: ProviderSurface) -> &'static ProviderSurfaceDescriptor {
    match surface {
        ProviderSurface::OpenAIChat => &openai_chat::PROVIDER_SURFACE,
        ProviderSurface::OpenAIResponses => &openai_responses::PROVIDER_SURFACE,
        ProviderSurface::AnthropicMessages => &anthropic::PROVIDER_SURFACE,
    }
}

impl ProviderSurface {
    /// The canonical codec name for this surface (e.g. `"openai_chat"`), the
    /// inverse of [`Self::from_codec_name`].
    #[must_use]
    pub fn codec_name(self) -> &'static str {
        descriptor_for(self).codec_name
    }

    /// Resolves a canonical codec name to its surface, or `None` when `name` is
    /// not a built-in provider codec.
    #[must_use]
    pub fn from_codec_name(name: &str) -> Option<Self> {
        BUILTIN_PROVIDER_SURFACES
            .iter()
            .find(|descriptor| descriptor.codec_name == name)
            .map(|descriptor| descriptor.surface)
    }
}

/// The canonical codec names of every built-in provider surface, for config
/// validation and "supported codec" messages.
#[must_use]
pub fn supported_codec_names() -> Vec<&'static str> {
    BUILTIN_PROVIDER_SURFACES
        .iter()
        .map(|descriptor| descriptor.codec_name)
        .collect()
}

/// Constructs the built-in bidirectional request codec ([`LlmCodec`]) for a surface.
#[must_use]
pub fn request_codec(surface: ProviderSurface) -> Arc<dyn LlmCodec> {
    (descriptor_for(surface).request_codec)()
}

/// Constructs the built-in decode-only response codec ([`LlmResponseCodec`]) for a surface.
#[must_use]
pub fn response_codec(surface: ProviderSurface) -> Arc<dyn LlmResponseCodec> {
    (descriptor_for(surface).response_codec)()
}

/// Constructs a fresh, single-use streaming codec ([`StreamingCodec`]) for a surface.
///
/// A [`StreamingCodec`] finalizer consumes its accumulator, so callers must
/// construct one instance per managed streaming call.
#[must_use]
pub fn streaming_codec(surface: ProviderSurface) -> Box<dyn StreamingCodec> {
    (descriptor_for(surface).streaming_codec)()
}

#[cfg(test)]
#[path = "../../tests/unit/codec/resolve_tests.rs"]
mod tests;
