// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_types::codec::request::MessageContent;
use nemo_relay_types::codec::response::{AnnotatedLlmResponse, FinishReason, Usage};
use nemo_relay_types::codec::traits::LlmResponseCodec;
use nemo_relay_types::error::{FlowError, Result};
use nemo_relay_types::Json;
use serde::Deserialize;
use serde_json::Map;

#[derive(Deserialize)]
struct FrameworkResponse {
    id: Option<String>,
    model_name: Option<String>,
    text: Option<String>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

struct FrameworkResponseCodec;

impl LlmResponseCodec for FrameworkResponseCodec {
    /// Decode a framework response into Relay's standard response fields.
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        let raw: FrameworkResponse = serde_json::from_value(response.clone())
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let total_tokens = match (raw.input_tokens, raw.output_tokens) {
            (Some(input), Some(output)) => input.checked_add(output),
            _ => None,
        };

        Ok(AnnotatedLlmResponse {
            id: raw.id,
            model: raw.model_name,
            message: raw.text.map(MessageContent::Text),
            tool_calls: None,
            finish_reason: Some(FinishReason::Complete),
            usage: Some(Usage {
                prompt_tokens: raw.input_tokens,
                completion_tokens: raw.output_tokens,
                total_tokens,
                cache_read_tokens: None,
                cache_write_tokens: None,
                uncached_input_tokens: None,
                cost: None,
            }),
            optimization_summary: None,
            api_specific: None,
            extra: Map::new(),
        })
    }
}
