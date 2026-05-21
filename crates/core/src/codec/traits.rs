// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LLM codec traits for bidirectional request translation.

use crate::api::llm::LlmRequest;
use crate::error::Result;
use crate::json::Json;

use super::request::AnnotatedLlmRequest;
use super::response::AnnotatedLlmResponse;

// ---------------------------------------------------------------------------
// LlmCodec trait
// ---------------------------------------------------------------------------

/// A bidirectional translator between opaque [`LlmRequest`] content and
/// structured [`AnnotatedLlmRequest`].
///
/// Codecs are implemented by integration patches (LangChain, LangChain-NVIDIA,
/// LangGraph, etc.) since each SDK has its own request format. They are
/// registered by name in the global codec registry.
///
/// # Design
///
/// - **Synchronous**: `decode`/`encode` are pure data transforms (JSON
///   restructuring), not I/O operations. This matches existing guardrails
///   and request intercepts.
/// - **`Send + Sync`**: Required because [`NemoRelayContextState`](crate::api::runtime::NemoRelayContextState)
///   is behind `Arc<RwLock<>>` and accessed from async contexts.
/// - **Trait object**: Codecs are registered at runtime (e.g., by Python
///   patches), so the Rust core cannot know concrete types at compile time.
///   Store as `Arc<dyn LlmCodec>`.
pub trait LlmCodec: Send + Sync {
    /// Parse opaque request content into structured form.
    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest>;

    /// Merge structured changes back into the opaque request.
    ///
    /// The `original` parameter is the pre-intercept [`LlmRequest`], used to
    /// preserve fields that the Codec does not structurally model. Implementations
    /// MUST use merge-not-replace semantics: overlay structured changes onto
    /// the original content, do not construct a fresh content object.
    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest>;
}

// ---------------------------------------------------------------------------
// LlmResponseCodec trait
// ---------------------------------------------------------------------------

/// Decode-only codec for LLM API responses.
///
/// Unlike [`LlmCodec`] (which is bidirectional for requests), response codecs
/// are introspection-only: they parse a raw response into structured form but
/// never need to encode back. This matches the pipeline design where responses
/// are observed, not modified.
///
/// # Design
///
/// - **Synchronous**: `decode_response` is a pure data transform (JSON parsing),
///   not an I/O operation.
/// - **`Send + Sync`**: Required for storage in `Arc` behind `RwLock`.
/// - **Trait object**: Codecs are registered at runtime, stored as
///   `Arc<dyn LlmResponseCodec>`.
/// - **Fallible**: Returns `Result`; managed call sites may omit annotations on
///   decode failure, while manual lifecycle bindings may surface the error.
///
/// # Two-Phase Decode
///
/// Implementations should use a two-phase decode pattern:
/// 1. Deserialize raw JSON into API-specific intermediate structs
/// 2. Map intermediate structs into the normalized `AnnotatedLlmResponse`
pub trait LlmResponseCodec: Send + Sync {
    /// Parse a raw JSON response into normalized structured form.
    ///
    /// Implementations should return `Err` only for genuinely unparseable input.
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse>;
}
