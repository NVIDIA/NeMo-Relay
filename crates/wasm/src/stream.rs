// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming LLM response wrapper for async iteration from JavaScript.
//!
//! Provides `WasmLlmStream`, an async-iterator-compatible type that lets
//! JavaScript consumers pull text chunks from a streaming LLM response one
//! at a time via the `next()` method.

use wasm_bindgen::prelude::*;

/// Wraps a streaming LLM response for consumption from JavaScript/TypeScript.
///
/// Implements an async-iterator-like protocol: call `next()` repeatedly to
/// receive `{value: string, done: boolean}` chunks until `done` is `true`.
#[wasm_bindgen]
pub struct WasmLlmStream {
    /// Async MPSC receiver that yields text chunks or errors from the
    /// underlying LLM stream. Wrapped in a `Mutex` to allow shared-ref
    /// `&self` calls from JavaScript.
    pub(crate) receiver:
        tokio::sync::Mutex<tokio::sync::mpsc::Receiver<nvagentrt_core::Result<String>>>,
}

#[wasm_bindgen]
impl WasmLlmStream {
    /// Returns the next chunk from the stream.
    ///
    /// The returned object has the shape `{value: string, done: boolean}`,
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
            Some(Ok(text)) => {
                let obj = js_sys::Object::new();
                js_sys::Reflect::set(&obj, &"value".into(), &JsValue::from_str(&text))?;
                js_sys::Reflect::set(&obj, &"done".into(), &JsValue::FALSE)?;
                Ok(obj.into())
            }
            Some(Err(e)) => Err(JsValue::from_str(&e.to_string())),
        }
    }
}
