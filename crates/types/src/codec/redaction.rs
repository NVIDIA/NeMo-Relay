// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in provider editing and clean projection, independent of PII policy.

use super::{request::AnnotatedLlmRequest, response::AnnotatedLlmResponse};
use crate::{Json, api::llm::LlmRequest, error::Result};

/// A sanitized serialized annotation, with positions anchored to its baseline.
///
/// Both values use the codec's normalized schema. `baseline` must come from
/// decoding the supplied original payload with that same codec. Sanitizers must
/// preserve array lengths and order: a removed array entry becomes `null`, while
/// a removed object field is absent. Do not compact arrays before this handoff.
/// Codecs must reject mismatched baselines and edits they cannot safely map.
/// These coordinates apply only to this operation; they are not stable handles.
#[derive(Clone, Copy, Debug)]
pub struct NormalizedEdits<'a> {
    /// Original normalized annotation, before sanitization.
    pub baseline: &'a Json,
    /// Sanitized annotation, retaining original array slots.
    pub sanitized: &'a Json,
    /// Whether the policy covers output content or provider-specific output.
    ///
    /// Set this even when no matching field exists in the first normalized
    /// result. A response codec must reject unmodeled alternatives it cannot
    /// cover. Request editing ignores this flag.
    pub require_complete_output: bool,
}

/// Optional request redaction operations exposed by the active request codec.
///
/// Detection and policy belong to the caller. Implementations own provider
/// layout, edit provenance, and validation. Each operation may independently
/// return `Ok(None)` when unsupported; callers must omit the payload on failure.
pub trait LlmRequestRedaction: Send + Sync {
    /// Apply source-anchored edits, preserving unaffected provider fields.
    ///
    /// Return a payload only when all edits can be represented without moving
    /// values onto unrelated source parts. Reject ambiguous or uncovered edits.
    fn edit_request(
        &self,
        _original: &LlmRequest,
        _edits: NormalizedEdits<'_>,
    ) -> Result<Option<LlmRequest>> {
        Ok(None)
    }

    /// Construct clean provider output solely from approved sanitized data.
    ///
    /// Unlike ordinary `encode`, this operation has no original wire payload.
    /// Never restore original fields or consult retained per-call source data.
    /// The caller must remove or sanitize opaque extras, native payloads, and
    /// tool payloads before approval. Return no original request headers.
    fn project_request(&self, _sanitized: &AnnotatedLlmRequest) -> Result<Option<LlmRequest>> {
        Ok(None)
    }
}

/// Optional response editing and clean projection exposed by a response codec.
///
/// Implementations must account for content outside the normalized result (for
/// example, additional choices), or reject edits that could leave it unsanitized.
/// Unsupported operations default to omission, independently of codec identity.
pub trait LlmResponseRedaction: Send + Sync {
    /// Apply anchored edits to the original response, validating the result.
    fn edit_response(&self, _original: &Json, _edits: NormalizedEdits<'_>) -> Result<Option<Json>> {
        Ok(None)
    }

    /// Construct clean provider output solely from approved sanitized data.
    ///
    /// No original provider payload or retained per-call source data may be used.
    /// Exclude opaque extras, native payloads, and tool payloads; callers approve
    /// portable fields before invoking this operation.
    fn project_response(&self, _sanitized: &AnnotatedLlmResponse) -> Result<Option<Json>> {
        Ok(None)
    }
}
