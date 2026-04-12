// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON utilities for the NeMo Flow runtime.
//!
//! This module provides a [`Json`] type alias for [`serde_json::Value`] used
//! throughout the crate, and a [`merge_json`] helper for shallow-merging
//! optional JSON values.

/// Type alias for [`serde_json::Value`], used as the universal JSON
/// representation throughout the NeMo Flow runtime.
pub type Json = serde_json::Value;

/// Shallow-merge two optional JSON values.
///
/// - If both values are JSON objects, keys from `b` override keys in `a`
///   (shallow merge — nested objects are replaced, not recursively merged).
/// - If only one value is `Some`, that value is returned.
/// - If both are `Some` but not both objects, `b` wins.
/// - If both are `None`, returns `None`.
///
/// This is used internally to merge `data` and `metadata` fields when
/// ending tool/LLM handles.
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
