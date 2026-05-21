// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming LLM response wrapper for async iteration from JavaScript.
//!
//! Provides `LlmStream`, an async-iterator-compatible type that lets
//! JavaScript consumers pull text chunks from a streaming LLM response one
//! at a time via the `next()` method.

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
use wasm_bindgen::prelude::*;

use nemo_relay::error::Result as FlowResult;
use nemo_relay::json::Json;

/// Wraps a streaming LLM response for consumption from JavaScript/TypeScript.
///
/// Call `next()` repeatedly to receive JSON chunks until it returns `null`.
#[wasm_bindgen(js_name = LlmStream)]
pub struct LlmStream {
    /// Async MPSC receiver that yields Json chunks or errors from the
    /// underlying LLM stream. Wrapped in a `Mutex` to allow shared-ref
    /// `&self` calls from JavaScript.
    pub(crate) receiver: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<FlowResult<Json>>>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_class = LlmStream)]
impl LlmStream {
    /// Returns the next chunk from the stream.
    ///
    /// Returns the next JSON chunk, or `null` when the stream is exhausted.
    /// Throws on stream errors.
    #[wasm_bindgen(unchecked_return_type = "Json | null")]
    pub async fn next(&self) -> Result<JsValue, JsValue> {
        let mut guard = self.receiver.lock().await;
        match guard.recv().await {
            None => Ok(JsValue::NULL),
            Some(Ok(json_val)) => {
                let js_val = json_val
                    .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
                    .map_err(|e| JsValue::from_str(&format!("serialization error: {e}")))?;
                Ok(js_val)
            }
            Some(Err(e)) => Err(JsValue::from_str(&e.to_string())),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_class = LlmStream)]
impl LlmStream {
    #[allow(tail_expr_drop_order)]
    /// Return `null` for non-`wasm32` builds.
    ///
    /// This binding surface is only fully functional when compiled for
    /// WebAssembly. Native test builds expose the same method signature but do
    /// not stream values.
    #[wasm_bindgen(unchecked_return_type = "Json | null")]
    pub async fn next(&self) -> Result<JsValue, JsValue> {
        Ok(JsValue::NULL)
    }
}

#[cfg(test)]
#[path = "../tests/unit/stream_tests.rs"]
mod tests;
