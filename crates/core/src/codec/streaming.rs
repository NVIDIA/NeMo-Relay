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

use crate::api::runtime::{LlmBorrowedCollectorFn, LlmCollectorFn, LlmFinalizerFn, LlmJsonStream};
use crate::error::{FlowError, Result};
use crate::json::Json;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

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
    /// Wraps the source stream to check provider completion rules.
    ///
    /// Feed the returned stream to this instance's collector and finalizer. The default
    /// leaves the stream unchanged. An override must forward close and cleanup operations.
    fn validate_stream(&self, stream: LlmJsonStream) -> LlmJsonStream {
        stream
    }

    /// Returns a closure that consumes one decoded provider event per call.
    fn collector(&self) -> LlmCollectorFn;

    /// Returns an optional collector that reads decoded events by reference.
    ///
    /// Pass this to `LlmStreamCallExecuteParams::borrowed_collector` to avoid copying
    /// each whole JSON chunk. The default keeps the owned callback for custom codecs.
    fn borrowed_collector(&self) -> Option<LlmBorrowedCollectorFn> {
        None
    }

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
    buffer: Vec<u8>,
    // Start the next delimiter search near the only position where a newly
    // appended chunk can complete an unfinished `\n\n` terminator. Completed
    // frames are compacted once after each push instead of shifting the buffer
    // for every frame.
    scan_from: usize,
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
    /// SSE streams are well-formed UTF-8 in practice. Decoding happens only after a complete
    /// frame is buffered, preserving valid characters split across incoming chunks.
    ///
    /// Returns `Ok(events)` containing zero or more events whose `data:` payloads parsed
    /// successfully. Frames whose `data:` line is non-empty but does not parse as JSON are
    /// surfaced as [`FlowError::Internal`] so the caller can decide whether to abort the stream
    /// or skip the frame; frames with no `data:` line at all (e.g. SSE heartbeats) are silently
    /// dropped.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> Result<Vec<SseEvent>> {
        self.push_bytes_results(bytes).into_iter().collect()
    }

    /// Appends `bytes` and returns each completed frame's result in wire order.
    ///
    /// Unlike [`Self::push_bytes`], this preserves successful events that precede a malformed
    /// frame in the same byte batch. Decoding stops after the first error so callers can emit the
    /// preceding events, surface the error, and terminate the stream.
    pub fn push_bytes_results(&mut self, bytes: &[u8]) -> Vec<Result<SseEvent>> {
        // Normalize CRLF to LF on append so the framing search only needs to find `\n\n`. Some
        // providers emit mixed line endings on the wire; normalizing once here keeps the inner
        // loop cheap. If CRLF is split across chunks, retain the trailing CR until the next append
        // and remove it only when the next byte completes the sequence.
        let append_from = self.buffer.len();
        if bytes.contains(&b'\r') || self.buffer.last() == Some(&b'\r') {
            for &byte in bytes {
                if byte == b'\n' && self.buffer.last() == Some(&b'\r') {
                    self.buffer.pop();
                }
                self.buffer.push(byte);
            }
            // Normalization can complete a terminator immediately before the append.
            self.scan_from = self.scan_from.min(append_from.saturating_sub(2));
        } else {
            self.buffer.extend_from_slice(bytes);
        }
        let mut results = Vec::new();
        let mut consumed = 0;
        let mut search_from = self.scan_from.min(self.buffer.len());
        while let Some(relative_cut) = self.buffer[search_from..]
            .windows(2)
            .position(|pair| pair == b"\n\n")
        {
            let cut = search_from + relative_cut;
            // `consumed` marks the start of the unparsed frame. Borrowing the
            // frame avoids allocating a string for every SSE event.
            let frame = String::from_utf8_lossy(&self.buffer[consumed..cut]);
            consumed = cut + 2;
            match parse_sse_frame(&frame) {
                Ok(Some(event)) => results.push(Ok(event)),
                Ok(None) => {}
                Err(error) => {
                    results.push(Err(error));
                    break;
                }
            }
            search_from = consumed;
        }
        if consumed > 0 {
            self.buffer.drain(..consumed);
            self.scan_from = 0;
        }
        if results.last().is_none_or(Result::is_ok) {
            // The next terminator can only start at the prior final byte; do
            // not rescan an incomplete frame from its beginning on every chunk.
            self.scan_from = self.buffer.len() - usize::from(self.buffer.ends_with(b"\n"));
        }
        results
    }

    /// Drains any remaining buffered frame at end of stream.
    ///
    /// Most well-formed SSE streams end with a terminating blank line, in which case this returns
    /// `Ok(None)`. Stops with no terminator are surfaced as a final partial frame so observability
    /// captures the last bytes the upstream sent before disconnect.
    pub fn finish(mut self) -> Result<Option<SseEvent>> {
        let bytes = std::mem::take(&mut self.buffer);
        let trailing = String::from_utf8_lossy(&bytes);
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
    // Provider frames usually have one data line; borrow it until a second line needs joining.
    let mut payload: Option<Cow<'_, str>> = None;
    for line in frame.split('\n') {
        if let Some(rest) = line.strip_prefix("event:") {
            event_name = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            // SSE allows a single space after the colon by convention; strip it lazily.
            let data = rest.strip_prefix(' ').unwrap_or(rest);
            match &mut payload {
                None => payload = Some(Cow::Borrowed(data)),
                Some(Cow::Borrowed(first)) => {
                    // The complete frame bounds the joined payload, including later data lines.
                    let mut joined = String::with_capacity(frame.len());
                    joined.push_str(first);
                    joined.push('\n');
                    joined.push_str(data);
                    payload = Some(Cow::Owned(joined));
                }
                Some(Cow::Owned(joined)) => {
                    joined.push('\n');
                    joined.push_str(data);
                }
            }
        }
        // Other lines (`id:`, `retry:`, comments starting with `:`) are ignored.
    }
    let Some(payload) = payload else {
        return Ok(None);
    };
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
