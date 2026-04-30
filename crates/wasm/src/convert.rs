// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversion utilities between JavaScript (`JsValue`) and Rust (`serde_json::Value`).
//!
//! These helpers are used throughout the WebAssembly bindings to marshal data across
//! the JS/Rust boundary via `serde_wasm_bindgen`.

use std::sync::{LazyLock, Mutex};

use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value as Json;
use wasm_bindgen::prelude::*;

static LAST_CALLBACK_ERROR: LazyLock<Mutex<Option<String>>> = LazyLock::new(|| Mutex::new(None));

/// Converts a displayable Rust error into a `JsValue` string for use as a JS exception.
pub fn to_js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Record the most recent callback error observed inside the WebAssembly binding.
pub fn record_callback_error(message: impl Into<String>) {
    let message = message.into();
    if let Ok(mut guard) = LAST_CALLBACK_ERROR.lock() {
        *guard = Some(message);
    }
}

/// Read the most recent callback error observed inside the WebAssembly binding.
pub fn get_last_callback_error() -> Option<String> {
    LAST_CALLBACK_ERROR
        .lock()
        .ok()
        .and_then(|guard| guard.clone())
}

/// Clear the most recent callback error observed inside the WebAssembly binding.
pub fn clear_last_callback_error() {
    if let Ok(mut guard) = LAST_CALLBACK_ERROR.lock() {
        *guard = None;
    }
}

/// Deserializes a `JsValue` into a `serde_json::Value`.
///
/// Returns a `JsValue` error string on deserialization failure.
pub fn js_to_json(val: &JsValue) -> Result<Json, JsValue> {
    serde_wasm_bindgen::from_value(val.clone()).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Deserializes a callback return value into JSON, treating `undefined` as `null`.
pub fn js_callback_to_json(val: &JsValue) -> Result<Json, JsValue> {
    if val.is_null() || val.is_undefined() {
        Ok(Json::Null)
    } else {
        js_to_json(val)
    }
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

/// Parses optional Unix microseconds since epoch into UTC.
pub fn opt_js_to_timestamp_micros(value: Option<f64>) -> Result<Option<DateTime<Utc>>, JsValue> {
    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    value
        .map(|timestamp| {
            if !timestamp.is_finite()
                || timestamp.fract() != 0.0
                || timestamp.abs() > MAX_SAFE_INTEGER
            {
                return Err(JsValue::from_str(
                    "timestamp must be a safe integer number of Unix microseconds",
                ));
            }
            DateTime::<Utc>::from_timestamp_micros(timestamp as i64).ok_or_else(|| {
                JsValue::from_str(
                    "invalid timestamp: unix microseconds are outside supported range",
                )
            })
        })
        .transpose()
}

#[cfg(test)]
#[path = "../tests/unit/convert_tests.rs"]
mod tests;
