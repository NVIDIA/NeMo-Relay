// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event types for Agent Trajectory Observability Format (ATOF) runtime events.

use std::borrow::Cow;

pub use nemo_relay_types::api::event::*;
use nemo_relay_types::api::llm::LlmRequest;
use nemo_relay_types::codec::request::AnnotatedLlmRequest;
use nemo_relay_types::codec::response::AnnotatedLlmResponse;

use crate::codec::resolve;

/// Core-only normalized LLM accessors for ATOF events.
///
/// These helpers use built-in codec resolution, so they live in the runtime
/// crate rather than the shared DTO crate.
pub trait EventNormalizationExt {
    /// Normalized LLM request: the codec annotation when present, otherwise a
    /// best-effort decode of the start-event input payload.
    ///
    /// The fallback decode requires the start-event input to be the serialized
    /// [`LlmRequest`] wire shape (`{headers, content}`) emitted by the managed
    /// LLM pipeline; events whose input is a bare payload or a non-LLM shape
    /// yield `None`.
    #[must_use]
    fn normalized_llm_request(&self) -> Option<Cow<'_, AnnotatedLlmRequest>>;

    /// Normalized LLM response: the codec annotation when present, otherwise a
    /// best-effort decode of the end-event output payload.
    #[must_use]
    fn normalized_llm_response(&self) -> Option<Cow<'_, AnnotatedLlmResponse>>;
}

impl EventNormalizationExt for Event {
    fn normalized_llm_request(&self) -> Option<Cow<'_, AnnotatedLlmRequest>> {
        if let Some(annotated) = self.annotated_request() {
            return Some(Cow::Borrowed(annotated.as_ref()));
        }
        let request: LlmRequest = serde_json::from_value(self.input()?.clone()).ok()?;
        // Managed LLM events use the provider route as the event name (for
        // example, "anthropic.messages"), which doubles as the codec hint for
        // shape-identical request bodies.
        resolve::normalize_request_with_hint(&request, Some(self.name())).map(Cow::Owned)
    }

    fn normalized_llm_response(&self) -> Option<Cow<'_, AnnotatedLlmResponse>> {
        if let Some(annotated) = self.annotated_response() {
            return Some(Cow::Borrowed(annotated.as_ref()));
        }
        resolve::normalize_response(self.output()?).map(Cow::Owned)
    }
}
