// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming LLM response wrapper for async iteration from JavaScript.
//!
//! Provides `WasmLlmStream`, an async-iterator-compatible type that lets
//! JavaScript consumers pull text chunks from a streaming LLM response one
//! at a time via the `next()` method.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use nvidia_nat_nexus_core::Json;

/// Wraps a streaming LLM response for consumption from JavaScript/TypeScript.
///
/// Implements an async-iterator-like protocol: call `next()` repeatedly to
/// receive `{value: object, done: boolean}` chunks until `done` is `true`.
#[wasm_bindgen]
pub struct WasmLlmStream {
    /// Async MPSC receiver that yields Json chunks or errors from the
    /// underlying LLM stream. Wrapped in a `Mutex` to allow shared-ref
    /// `&self` calls from JavaScript.
    pub(crate) receiver:
        tokio::sync::Mutex<tokio::sync::mpsc::Receiver<nvidia_nat_nexus_core::Result<Json>>>,
}

#[wasm_bindgen]
impl WasmLlmStream {
    /// Returns the next chunk from the stream.
    ///
    /// The returned object has the shape `{value: object, done: boolean}`,
    /// compatible with the JavaScript AsyncIterator protocol. When the stream
    /// is exhausted, `done` is `true` and `value` is `undefined`. Throws on
    /// stream errors.
    pub async fn next(&self) -> Result<JsValue, JsValue> {
        let mut guard = self.receiver.lock().await;
        match guard.recv().await {
            None => {
                let obj = js_sys::Object::new();
                js_sys::Reflect::set(&obj, &"value".into(), &JsValue::UNDEFINED)?;
                js_sys::Reflect::set(&obj, &"done".into(), &JsValue::TRUE)?;
                Ok(obj.into())
            }
            Some(Ok(json_val)) => {
                let js_val = json_val
                    .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
                    .unwrap_or(JsValue::NULL);
                let obj = js_sys::Object::new();
                js_sys::Reflect::set(&obj, &"value".into(), &js_val)?;
                js_sys::Reflect::set(&obj, &"done".into(), &JsValue::FALSE)?;
                Ok(obj.into())
            }
            Some(Err(e)) => Err(JsValue::from_str(&e.to_string())),
        }
    }
}
