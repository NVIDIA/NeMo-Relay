// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test accessors for runtime-supplied views of delivered, sanitized LLM events.

use super::{Event, EventNormalizationExt};
use crate::codec::RuntimeLlmNormalizer;
use nemo_relay_types::codec::normalization::{LlmNormalizer, TelemetryToolCall};
use nemo_relay_types::codec::request::AnnotatedLlmRequest;
use std::borrow::Cow;

/// Export-only semantic views of delivered LLM events.
pub(crate) trait EventTelemetryExt {
    /// Project native request items without modifying stored annotations.
    fn telemetry_llm_request(&self) -> Option<Cow<'_, AnnotatedLlmRequest>>;
    /// Project response invocations and their sanitized metadata.
    fn telemetry_llm_tool_calls(&self) -> Option<Vec<TelemetryToolCall>>;
}
impl EventTelemetryExt for Event {
    fn telemetry_llm_request(&self) -> Option<Cow<'_, AnnotatedLlmRequest>> {
        Some(RuntimeLlmNormalizer.telemetry_request(self.normalized_llm_request()?))
    }
    fn telemetry_llm_tool_calls(&self) -> Option<Vec<TelemetryToolCall>> {
        RuntimeLlmNormalizer.telemetry_tool_calls(
            self.annotated_response().map(|response| response.as_ref()),
            self.output(),
        )
    }
}
