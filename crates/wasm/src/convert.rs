// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversion utilities between JavaScript (`JsValue`) and Rust (`serde_json::Value`).
//!
//! These helpers are used throughout the WASM bindings to marshal data across
//! the JS/Rust boundary via `serde_wasm_bindgen`.

use serde::Serialize;
use serde_json::Value as Json;
use wasm_bindgen::prelude::*;

use nvmagic_core::MagicError;

/// Converts an `MagicError` into a `JsValue` string for use as a JS exception.
pub fn to_js_err(e: MagicError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Deserializes a `JsValue` into a `serde_json::Value`.
///
/// Returns a `JsValue` error string on deserialization failure.
pub fn js_to_json(val: &JsValue) -> Result<Json, JsValue> {
    serde_wasm_bindgen::from_value(val.clone()).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Serializes a `serde_json::Value` into a `JsValue`.
///
/// Returns `JsValue::NULL` if serialization fails.
pub fn json_to_js(val: &Json) -> JsValue {
    val.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .unwrap_or(JsValue::NULL)
}

/// Deserializes an optional `JsValue` into `Option<serde_json::Value>`.
///
/// Returns `Ok(None)` if the value is `null` or `undefined`.
pub fn opt_js_to_json(val: &JsValue) -> Result<Option<Json>, JsValue> {
    if val.is_null() || val.is_undefined() {
        Ok(None)
    } else {
        js_to_json(val).map(Some)
    }
}
