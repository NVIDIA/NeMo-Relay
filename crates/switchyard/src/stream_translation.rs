// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Relay stream adapter for Switchyard's provider translation engine.

use nemo_relay::error::{FlowError, Result};
use serde_json::Value as Json;
use switchyard_translation::{StreamTranslationState, TranslationEngine};

use crate::component::WireProtocol;
use crate::translation::wire_format;

pub(crate) struct StreamTranscoder {
    engine: TranslationEngine,
    source: WireProtocol,
    target: WireProtocol,
    state: StreamTranslationState,
}

impl StreamTranscoder {
    pub(crate) fn new(
        source: WireProtocol,
        target: WireProtocol,
        effective_model: impl Into<String>,
    ) -> Self {
        let effective_model = effective_model.into();
        let mut state = StreamTranslationState::new(wire_format(source), wire_format(target));
        state.model = Some(effective_model.clone());
        state.target_model = Some(effective_model);
        Self {
            engine: TranslationEngine::default(),
            source,
            target,
            state,
        }
    }

    pub(crate) fn transcode(&mut self, chunk: &Json) -> Result<Vec<Json>> {
        if unsupported_stream_chunk(self.source, chunk) {
            return Err(FlowError::InvalidArgument(
                "provider-specific streaming extension cannot be translated safely".into(),
            ));
        }
        self.engine
            .translate_event(
                &mut self.state,
                wire_format(self.source),
                wire_format(self.target),
                chunk,
            )
            .map_err(|error| {
                FlowError::InvalidArgument(format!("Switchyard stream translation failed: {error}"))
            })
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<Json>> {
        self.engine
            .finish_stream(&mut self.state, wire_format(self.target))
            .map_err(|error| {
                FlowError::InvalidArgument(format!(
                    "Switchyard stream finalization failed: {error}"
                ))
            })
    }
}

fn unsupported_stream_chunk(source: WireProtocol, chunk: &Json) -> bool {
    match source {
        WireProtocol::OpenaiChat => {
            chunk["choices"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|choice| {
                    choice["delta"].get("audio").is_some()
                        || choice["delta"].get("reasoning_content").is_some()
                })
        }
        WireProtocol::OpenaiResponses => {
            chunk.get("type").and_then(Json::as_str) == Some("response.output_item.added")
                && matches!(
                    chunk["item"].get("type").and_then(Json::as_str),
                    Some("reasoning" | "computer_call" | "web_search_call")
                )
        }
        WireProtocol::AnthropicMessages => {
            chunk.get("type").and_then(Json::as_str) == Some("content_block_start")
                && !matches!(
                    chunk["content_block"].get("type").and_then(Json::as_str),
                    Some("text" | "tool_use")
                )
        }
    }
}
