// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON utilities for the NeMo Relay runtime.
//!
//! This module provides a [`Json`] type alias for [`serde_json::Value`] used
//! throughout the crate, and a [`merge_json`] helper for shallow-merging
//! optional JSON values.

pub use nemo_relay_types::Json;

/// Shallow-merge two optional JSON values.
///
/// This is used throughout the runtime to combine optional `data` and
/// `metadata` payloads without recursively descending into nested objects.
///
/// # Parameters
/// - `a`: Base JSON value.
/// - `b`: Override JSON value.
///
/// # Returns
/// An [`Option`] containing the merged JSON value. When both inputs are JSON
/// objects, keys from `b` override keys from `a`. When only one input is
/// present, that input is returned. When both inputs are present but at least
/// one is not an object, `b` wins.
///
/// # Notes
/// The merge is shallow. Nested objects are replaced rather than merged
/// recursively.
pub fn merge_json(a: Option<Json>, b: Option<Json>) -> Option<Json> {
    match (a, b) {
        (Some(Json::Object(mut ma)), Some(Json::Object(mb))) => {
            for (k, v) in mb {
                ma.insert(k, v);
            }
            Some(Json::Object(ma))
        }
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (Some(_), Some(b)) => Some(b),
        (None, None) => None,
    }
}

#[cfg(test)]
#[path = "../tests/unit/json_tests.rs"]
mod tests;
