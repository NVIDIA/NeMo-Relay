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
//! [`StreamingCodec`](crate::codec::streaming::StreamingCodec) is the trait that bundles the two functions
//! ([`LlmCollectorFn`](crate::api::runtime::LlmCollectorFn),
//! [`LlmFinalizerFn`](crate::api::runtime::LlmFinalizerFn)) used by
//! the runtime streaming execution API. Each provider supplies one impl whose internal
//! state holds whatever incremental information is needed to materialize the final payload.
//!
//! [`AnnotatedLlmResponse`]: crate::codec::response::AnnotatedLlmResponse

use crate::Json;
use crate::api::runtime::{LlmBorrowedCollectorFn, LlmCollectorFn, LlmFinalizerFn, LlmJsonStream};
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

/// Per-provider streaming codec used with the runtime streaming execution API.
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
