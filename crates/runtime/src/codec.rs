// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime normalization composes provider codecs with the configured pricing service.

use nemo_relay_codecs::resolve::{self, ProviderSurface};
use nemo_relay_types::{
    Json,
    api::llm::LlmRequest,
    codec::{
        identity::LlmCodecIdentity, normalization::LlmNormalizer, request::AnnotatedLlmRequest,
        response::AnnotatedLlmResponse, traits::LlmResponseCodec,
    },
    error::Result,
};
use std::sync::Arc;

/// Built-in response codec enriched by this runtime's configured pricing service.
pub fn response_codec(surface: ProviderSurface) -> Arc<dyn LlmResponseCodec> {
    Arc::new(PricedResponseCodec {
        inner: resolve::response_codec(surface),
        provider: match surface {
            ProviderSurface::OpenAIChat | ProviderSurface::OpenAIResponses => "openai",
            ProviderSurface::AnthropicMessages => "anthropic",
            ProviderSurface::OCIGenAI => "oci",
            ProviderSurface::GeminiGenerateContent => "google",
        },
    })
}

struct PricedResponseCodec {
    inner: Arc<dyn LlmResponseCodec>,
    provider: &'static str,
}
impl LlmResponseCodec for PricedResponseCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        self.inner.codec_identity()
    }
    fn allows_estimated_cost(&self, raw: &Json) -> bool {
        self.inner.allows_estimated_cost(raw)
    }
    fn decode_response(&self, raw: &Json) -> Result<AnnotatedLlmResponse> {
        let mut response = self.inner.decode_response(raw)?;
        if self.inner.allows_estimated_cost(raw) {
            let provider =
                crate::pricing::infer_model_provider(self.provider, response.model.as_deref());
            crate::pricing::attach_estimated_cost_for_provider(&mut response, provider.as_deref());
        }
        Ok(response)
    }
}

/// Provider normalization using the runtime's configured enrichment services.
#[derive(Debug, Clone, Copy, Default)]
pub struct RuntimeLlmNormalizer;
impl LlmNormalizer for RuntimeLlmNormalizer {
    fn normalize_request(
        &self,
        request: &LlmRequest,
        provider_hint: Option<&str>,
    ) -> Option<AnnotatedLlmRequest> {
        resolve::normalize_request_with_hint(request, provider_hint)
    }
    fn normalize_response(&self, raw: &Json) -> Option<AnnotatedLlmResponse> {
        response_codec(resolve::detect_response_surface(raw)?)
            .decode_response(raw)
            .ok()
    }
}
