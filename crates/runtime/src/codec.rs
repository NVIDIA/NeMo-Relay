// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime normalization composes provider codecs with the configured pricing service.

use nemo_relay_codecs::resolve::{self, ProviderSurface};
use nemo_relay_types::{
    Json,
    api::llm::LlmRequest,
    codec::{
        identity::LlmCodecIdentity,
        normalization::LlmNormalizer,
        redaction::{LlmResponseRedaction, NormalizedEdits},
        request::AnnotatedLlmRequest,
        response::AnnotatedLlmResponse,
        traits::LlmResponseCodec,
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
    fn redaction(&self) -> Option<&dyn nemo_relay_types::codec::redaction::LlmResponseRedaction> {
        self.inner
            .redaction()
            .map(|_| self as &dyn LlmResponseRedaction)
    }

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

impl LlmResponseRedaction for PricedResponseCodec {
    fn edit_response(&self, original: &Json, edits: NormalizedEdits<'_>) -> Result<Option<Json>> {
        let Some(capability) = self.inner.redaction() else {
            return Ok(None);
        };
        let serialize = |value| {
            serde_json::to_value(value)
                .map_err(|error| nemo_relay_types::error::FlowError::Internal(error.to_string()))
        };
        if &serialize(self.decode_response(original)?)? != edits.baseline {
            return Ok(None);
        }
        let baseline = serialize(self.inner.decode_response(original)?)?;
        let mut sanitized = edits.sanitized.clone();
        if baseline.pointer("/usage/cost") != edits.baseline.pointer("/usage/cost") {
            // Estimated cost belongs to the runtime annotation, not provider
            // JSON. Preserve it on later decode; never send it through the
            // provider editor as though it originated in the wire payload.
            if sanitized.pointer("/usage/cost") != edits.baseline.pointer("/usage/cost") {
                return Ok(None);
            }
            let Some(usage) = sanitized.get_mut("usage").and_then(Json::as_object_mut) else {
                return Ok(None);
            };
            match baseline.pointer("/usage/cost") {
                Some(cost) => {
                    usage.insert("cost".into(), cost.clone());
                }
                None => {
                    usage.remove("cost");
                }
            }
        }
        capability.edit_response(
            original,
            NormalizedEdits {
                baseline: &baseline,
                sanitized: &sanitized,
                require_complete_output: edits.require_complete_output,
            },
        )
    }

    fn project_response(&self, sanitized: &AnnotatedLlmResponse) -> Result<Option<Json>> {
        match self.inner.redaction() {
            Some(capability) => capability.project_response(sanitized),
            None => Ok(None),
        }
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

    fn telemetry_request<'a>(
        &self,
        request: std::borrow::Cow<'a, nemo_relay_types::codec::request::AnnotatedLlmRequest>,
    ) -> std::borrow::Cow<'a, nemo_relay_types::codec::request::AnnotatedLlmRequest> {
        nemo_relay_codecs::resolve::BuiltinLlmNormalizer.telemetry_request(request)
    }
    fn telemetry_tool_calls(
        &self,
        annotation: Option<&nemo_relay_types::codec::response::AnnotatedLlmResponse>,
        output: Option<&nemo_relay_types::Json>,
    ) -> Option<Vec<nemo_relay_types::codec::normalization::TelemetryToolCall>> {
        nemo_relay_codecs::resolve::BuiltinLlmNormalizer.telemetry_tool_calls(annotation, output)
    }
}
