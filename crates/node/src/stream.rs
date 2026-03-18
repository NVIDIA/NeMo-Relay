// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming LLM response support for the Node.js NAPI bindings.
//!
//! Provides the `LlmStream` type, an async iterator that yields response chunks
//! from a streaming LLM call. Chunks are received over a Tokio MPSC channel and
//! exposed to JavaScript via the `next()` method.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value as Json;

/// An async iterator over chunks from a streaming LLM response.
///
/// Obtained from `llmStreamCallExecute()`. Call `next()` repeatedly to consume
/// response chunks. Returns `null` when the stream is fully consumed.
#[napi]
pub struct LlmStream {
    pub(crate) receiver:
        tokio::sync::Mutex<tokio::sync::mpsc::Receiver<nvidia_nat_nexus_core::Result<Json>>>,
}

#[napi]
impl LlmStream {
    /// Retrieve the next chunk from the stream.
    ///
    /// Returns the next JSON chunk, or `null` when the stream is exhausted.
    /// Throws if the underlying stream encountered an error.
    #[napi]
    pub async fn next(&self) -> Result<Option<Json>> {
        let mut guard = self.receiver.lock().await;
        match guard.recv().await {
            None => Ok(None),
            Some(Ok(value)) => Ok(Some(value)),
            Some(Err(e)) => Err(napi::Error::from_reason(e.to_string())),
        }
    }
}
