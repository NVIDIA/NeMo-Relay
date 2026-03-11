// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversion utilities for bridging between NVMagic core types and NAPI types.
//!
//! Provides helpers to convert errors and optional JSON values between the core
//! runtime representation and the NAPI binding layer.

use serde_json::Value as Json;

use nvmagic_core::MagicError;

/// Convert an `MagicError` into a `napi::Error` by formatting the error as a reason string.
pub fn to_napi_err(e: MagicError) -> napi::Error {
    napi::Error::from_reason(e.to_string())
}

/// Filter an optional JSON value, converting explicit `null` values to `None`.
///
/// NAPI's serde-json feature handles most conversion automatically, but JavaScript
/// may pass `null` where Rust expects `None`. This normalizes that case.
pub fn opt_json(val: Option<Json>) -> Option<Json> {
    val.filter(|v| !v.is_null())
}
