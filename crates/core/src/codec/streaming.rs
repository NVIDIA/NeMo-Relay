// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming response codecs for the managed LLM execution pipeline.
//!
//! [`crate::codec::traits::LlmResponseCodec`] decodes a complete provider response into a
//! normalized [`AnnotatedLlmResponse`]. For streaming providers, the analogous job is to:
//!
//! 1. consume per-chunk events as they arrive on a streaming HTTP response, and
//! 2. assemble a single non-streaming-shape JSON payload at end of stream.
//!
//! Once assembled, the payload can be fed back through the matching
//! [`crate::codec::traits::LlmResponseCodec`] to produce an [`AnnotatedLlmResponse`] — meaning
//! streaming and non-streaming requests converge on the same observability output without
//! per-route shape duplication.
//!
//! [`StreamingCodec`] is the trait that bundles the two functions
//! ([`LlmCollectorFn`],
//! [`LlmFinalizerFn`]) used by
//! [`crate::api::llm::llm_stream_call_execute`]. Each provider supplies one impl whose internal
//! state holds whatever incremental information is needed to materialize the final payload.
//!
//! [`AnnotatedLlmResponse`]: crate::codec::response::AnnotatedLlmResponse

use crate::api::runtime::{LlmCollectorFn, LlmFinalizerFn};
use crate::error::{FlowError, Result};
use crate::json::Json;
use serde::{Deserialize, Serialize};

/// Provider-neutral incremental stream item used by cross-protocol transcoders.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NormalizedStreamEvent {
    /// Assistant text delta.
    TextDelta {
        /// Newly generated text.
        text: String,
    },
    /// Start of a function tool call.
    ToolCallStart {
        /// Stable tool-call identifier.
        id: String,
        /// Function name.
        name: String,
    },
    /// Incremental JSON arguments for a function tool call.
    ToolCallArgumentsDelta {
        /// Stable tool-call identifier.
        id: String,
        /// Newly generated argument bytes.
        delta: String,
    },
    /// Provider-reported terminal reason.
    Finish {
        /// Normalized or provider-native finish label.
        reason: Option<String>,
    },
    /// Incremental or terminal usage object.
    Usage {
        /// Provider-neutral usage JSON.
        usage: Json,
    },
    /// Provider stream error payload.
    Error {
        /// Structured provider error.
        error: Json,
    },
}

/// Per-provider streaming codec used with [`crate::api::llm::llm_stream_call_execute`].
///
/// `collector()` and `finalizer()` produce owned closures that share the codec's internal
/// accumulation state. Implementations typically wrap that state in `Arc<Mutex<...>>` so each
/// `&self`-produced closure captures a clone of the handle.
///
/// [`LlmFinalizerFn`] is `FnOnce`, so a [`StreamingCodec`] instance is single-use: callers
/// construct a fresh instance per managed-lifecycle call and discard it after the stream
/// completes.
pub trait StreamingCodec: Send + Sync {
    /// Returns a closure that consumes one decoded provider event per call.
    fn collector(&self) -> LlmCollectorFn;

    /// Returns a closure that, when called once at end of stream, produces the assembled response
    /// payload in the shape the matching [`crate::codec::traits::LlmResponseCodec`] can decode.
    fn finalizer(&self) -> LlmFinalizerFn;
}

/// Incremental decoder for `text/event-stream` byte streams that yields one JSON object per
/// complete `data:` payload.
///
/// SSE frames are separated by blank lines (`\n\n`); each frame may contain `event:` and `data:`
/// lines. Anthropic Messages, OpenAI Responses, and OpenAI Chat Completions all emit one JSON
/// object per `data:` line, so the decoder buffers received bytes, splits on frame boundaries,
/// parses the JSON payload, and tags it with the frame's event name when present.
///
/// The decoder is byte-stream-friendly: it accumulates partial frames across chunks and emits
/// completed frames only when their terminating blank line arrives. Bytes after the last
/// terminator are retained for the next call.
#[derive(Default)]
pub struct SseEventDecoder {
    buffer: String,
}

/// One decoded SSE frame, paired with the parsed `data:` payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// Value of the `event:` line if present.
    pub event: Option<String>,
    /// Parsed JSON payload from the `data:` line(s).
    pub data: Json,
}

impl SseEventDecoder {
    /// Creates a new decoder with an empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `bytes` to the internal buffer and returns every now-complete SSE event.
    ///
    /// Bytes are interpreted as UTF-8 with replacement characters for invalid sequences; provider
    /// SSE streams are well-formed UTF-8 in practice, but lossy decoding keeps the decoder honest
    /// rather than failing on a single corrupt chunk.
    ///
    /// Returns `Ok(events)` containing zero or more events whose `data:` payloads parsed
    /// successfully. Frames whose `data:` line is non-empty but does not parse as JSON are
    /// surfaced as [`FlowError::Internal`] so the caller can decide whether to abort the stream
    /// or skip the frame; frames with no `data:` line at all (e.g. SSE heartbeats) are silently
    /// dropped.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> Result<Vec<SseEvent>> {
        // Normalize CRLF to LF on append so the framing search only needs to find `\n\n`. Some
        // providers emit mixed line endings on the wire; normalizing once here keeps the inner
        // loop cheap.
        let chunk = String::from_utf8_lossy(bytes).replace("\r\n", "\n");
        self.buffer.push_str(&chunk);
        let mut events = Vec::new();
        while let Some(cut) = self.buffer.find("\n\n") {
            let frame: String = self.buffer.drain(..cut).collect();
            // Drop the `\n\n` terminator itself.
            self.buffer.drain(..2);
            if let Some(event) = parse_sse_frame(&frame)? {
                events.push(event);
            }
        }
        Ok(events)
    }

    /// Drains any remaining buffered frame at end of stream.
    ///
    /// Most well-formed SSE streams end with a terminating blank line, in which case this returns
    /// `Ok(None)`. Stops with no terminator are surfaced as a final partial frame so observability
    /// captures the last bytes the upstream sent before disconnect.
    pub fn finish(mut self) -> Result<Option<SseEvent>> {
        let trailing = std::mem::take(&mut self.buffer);
        if trailing.trim().is_empty() {
            Ok(None)
        } else {
            parse_sse_frame(&trailing)
        }
    }
}

// Parses a single SSE frame. Returns `None` for frames without a `data:` line, `Some(event)` for
// frames whose `data:` JSON parsed successfully.
fn parse_sse_frame(frame: &str) -> Result<Option<SseEvent>> {
    let mut event_name: Option<String> = None;
    let mut data_parts: Vec<&str> = Vec::new();
    for line in frame.split('\n') {
        if let Some(rest) = line.strip_prefix("event:") {
            event_name = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            // SSE allows a single space after the colon by convention; strip it lazily.
            data_parts.push(rest.strip_prefix(' ').unwrap_or(rest));
        }
        // Other lines (`id:`, `retry:`, comments starting with `:`) are ignored.
    }
    if data_parts.is_empty() {
        return Ok(None);
    }
    let payload = data_parts.join("\n");
    let trimmed = payload.trim();
    // OpenAI Chat Completions emits a `data: [DONE]` terminator as a wire-level end-of-stream
    // sentinel. It's not a JSON payload — drop it like a heartbeat. Other providers (Anthropic,
    // OpenAI Responses) have proper terminal events instead, so this only fires for OpenAI Chat.
    if trimmed == "[DONE]" {
        return Ok(None);
    }
    let data: Json = serde_json::from_str(trimmed).map_err(|error| {
        FlowError::Internal(format!(
            "streaming codec failed to parse SSE data payload: {error}: {payload}"
        ))
    })?;
    Ok(Some(SseEvent {
        event: event_name,
        data,
    }))
}

#[cfg(test)]
#[path = "../../tests/unit/codec/streaming_tests.rs"]
mod tests;
