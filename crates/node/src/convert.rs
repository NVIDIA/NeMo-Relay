// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversion utilities for bridging between NeMo Relay core types and NAPI types.
//!
//! Provides helpers to convert errors and optional JSON values between the core
//! runtime representation and the NAPI binding layer.

use std::sync::{LazyLock, Mutex};

use chrono::{DateTime, Utc};
use serde_json::Value as Json;

use nemo_relay::error::FlowError;

static LAST_CALLBACK_ERROR: LazyLock<Mutex<Option<String>>> = LazyLock::new(|| Mutex::new(None));

/// Convert an `FlowError` into a `napi::Error` by formatting the error as a reason string.
pub fn to_napi_err(e: FlowError) -> napi::Error {
    napi::Error::from_reason(e.to_string())
}

/// Record the most recent callback error observed inside the Node binding.
pub fn record_callback_error(message: impl Into<String>) {
    let message = message.into();
    eprintln!("{message}");
    if let Ok(mut guard) = LAST_CALLBACK_ERROR.lock() {
        *guard = Some(message);
    }
}

/// Read the most recent callback error observed inside the Node binding.
pub fn get_last_callback_error() -> Option<String> {
    LAST_CALLBACK_ERROR
        .lock()
        .ok()
        .and_then(|guard| guard.clone())
}

/// Clear the most recent callback error observed inside the Node binding.
pub fn clear_last_callback_error() {
    if let Ok(mut guard) = LAST_CALLBACK_ERROR.lock() {
        *guard = None;
    }
}

/// Filter an optional JSON value, converting explicit `null` values to `None`.
///
/// NAPI's serde-json feature handles most conversion automatically, but JavaScript
/// may pass `null` where Rust expects `None`. This normalizes that case.
pub fn opt_json(val: Option<Json>) -> Option<Json> {
    val.filter(|v| !v.is_null())
}

/// Normalize a callback return into JSON, treating omitted/`undefined` as `null`.
pub fn callback_json(val: Option<Json>) -> Json {
    val.unwrap_or(Json::Null)
}

/// Parse optional Unix microseconds since epoch into a UTC timestamp.
pub fn parse_timestamp_micros(value: Option<f64>) -> napi::Result<Option<DateTime<Utc>>> {
    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    value
        .map(|timestamp| {
            if !timestamp.is_finite()
                || timestamp.fract() != 0.0
                || timestamp.abs() > MAX_SAFE_INTEGER
            {
                return Err(napi::Error::from_reason(
                    "timestamp must be a safe integer number of Unix microseconds",
                ));
            }
            DateTime::<Utc>::from_timestamp_micros(timestamp as i64).ok_or_else(|| {
                napi::Error::from_reason(
                    "invalid timestamp: unix microseconds are outside supported range",
                )
            })
        })
        .transpose()
}
