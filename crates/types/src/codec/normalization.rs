// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider normalization supplied by a runtime or embedding application.

use crate::Json;
use crate::api::llm::LlmRequest;
use crate::codec::request::AnnotatedLlmRequest;
use crate::codec::response::AnnotatedLlmResponse;

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
}
