// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversion utilities between JavaScript (`JsValue`) and Rust (`serde_json::Value`).
//!
//! These helpers are used throughout the WASM bindings to marshal data across
//! the JS/Rust boundary via `serde_wasm_bindgen`.

use std::sync::{LazyLock, Mutex};

use serde::Serialize;
use serde_json::Value as Json;
use wasm_bindgen::prelude::*;

use nvidia_nat_nexus_core::NexusError;

static LAST_CALLBACK_ERROR: LazyLock<Mutex<Option<String>>> = LazyLock::new(|| Mutex::new(None));

/// Converts an `NexusError` into a `JsValue` string for use as a JS exception.
pub fn to_js_err(e: NexusError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Record the most recent callback error observed inside the WASM binding.
pub fn record_callback_error(message: impl Into<String>) {
    let message = message.into();
    if let Ok(mut guard) = LAST_CALLBACK_ERROR.lock() {
        *guard = Some(message);
    }
}

/// Read the most recent callback error observed inside the WASM binding.
pub fn get_last_callback_error() -> Option<String> {
    LAST_CALLBACK_ERROR
        .lock()
        .ok()
        .and_then(|guard| guard.clone())
}

/// Clear the most recent callback error observed inside the WASM binding.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callback_error_store_round_trip() {
        clear_last_callback_error();
        assert_eq!(get_last_callback_error(), None);

        record_callback_error("wasm callback failed");
        assert_eq!(
            get_last_callback_error(),
            Some("wasm callback failed".to_string())
        );

        clear_last_callback_error();
        assert_eq!(get_last_callback_error(), None);
    }
}
