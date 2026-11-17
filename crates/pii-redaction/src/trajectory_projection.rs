// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event projection adapters. Provider rendering belongs to the codec.

use nemo_relay_codecs::resolve::{ProviderSurface, request_codec, response_codec};
use nemo_relay_types::{
    Json,
    api::llm::LlmRequest,
    codec::{request::AnnotatedLlmRequest, response::AnnotatedLlmResponse},
};

pub(super) fn render_request(
    surface: ProviderSurface,
    request: &AnnotatedLlmRequest,
) -> Option<LlmRequest> {
    request_codec(surface)
        .redaction()?
        .project_request(request)
        .ok()?
}

pub(super) fn render_response(
    surface: ProviderSurface,
    response: &AnnotatedLlmResponse,
) -> Option<Json> {
    response_codec(surface)
        .redaction()?
        .project_response(response)
        .ok()?
}

#[cfg(test)]
#[path = "../tests/unit/trajectory_projection_tests.rs"]
mod tests;
