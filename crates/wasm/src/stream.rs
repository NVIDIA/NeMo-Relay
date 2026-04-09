// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming LLM response wrapper for async iteration from JavaScript.
//!
//! Provides `WasmLlmStream`, an async-iterator-compatible type that lets
//! JavaScript consumers pull text chunks from a streaming LLM response one
//! at a time via the `next()` method.

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
use wasm_bindgen::prelude::*;

use nemo_flow_core::Json;

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
        tokio::sync::Mutex<tokio::sync::mpsc::Receiver<nemo_flow_core::Result<Json>>>,
}

#[cfg(target_arch = "wasm32")]
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
                    .map_err(|e| JsValue::from_str(&format!("serialization error: {e}")))?;
                let obj = js_sys::Object::new();
                js_sys::Reflect::set(&obj, &"value".into(), &js_val)?;
                js_sys::Reflect::set(&obj, &"done".into(), &JsValue::FALSE)?;
                Ok(obj.into())
            }
            Some(Err(e)) => Err(JsValue::from_str(&e.to_string())),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen]
impl WasmLlmStream {
    #[allow(tail_expr_drop_order)]
    pub async fn next(&self) -> Result<JsValue, JsValue> {
        Ok(JsValue::NULL)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "wasm32")]
    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(future)
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn next_returns_value_done_false_and_then_done_true() {
        use super::*;
        use serde_json::json;

        block_on(async {
            let (tx, rx) = tokio::sync::mpsc::channel(2);
            tx.send(Ok(json!({"chunk": 1}))).await.unwrap();
            drop(tx);

            let stream = WasmLlmStream {
                receiver: tokio::sync::Mutex::new(rx),
            };
            let first = stream.next().await.unwrap();
            assert_eq!(
                js_sys::Reflect::get(&first, &JsValue::from_str("done"))
                    .unwrap()
                    .as_bool(),
                Some(false)
            );
            let value = js_sys::Reflect::get(&first, &JsValue::from_str("value")).unwrap();
            assert_eq!(
                crate::convert::js_to_json(&value).unwrap(),
                json!({"chunk": 1})
            );

            let second = stream.next().await.unwrap();
            assert_eq!(
                js_sys::Reflect::get(&second, &JsValue::from_str("done"))
                    .unwrap()
                    .as_bool(),
                Some(true)
            );
        });
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn next_returns_js_error_for_stream_errors() {
        use super::*;

        block_on(async {
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            tx.send(Err(nemo_flow_core::FlowError::Internal(
                "stream failed".to_string(),
            )))
            .await
            .unwrap();
            drop(tx);

            let stream = WasmLlmStream {
                receiver: tokio::sync::Mutex::new(rx),
            };
            let err = stream.next().await.unwrap_err();
            assert!(err.as_string().unwrap().contains("stream failed"));
        });
    }
}
