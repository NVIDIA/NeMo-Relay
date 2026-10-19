// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small JSON path helpers shared by CLI adapter and alignment code.

use serde_json::Value;

/// Read a nested value using exact object-key traversal.
///
/// Missing intermediate keys stop the lookup without error so callers can
/// express schema precedence by chaining candidate paths.
pub(crate) fn value_at(payload: &Value, path: &[&str]) -> Option<Value> {
    borrowed_value_at(payload, path).cloned()
}

fn borrowed_value_at<'a>(payload: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = payload;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

/// Read the first JSON value from any candidate path.
///
/// The clone is intentional because extracted correlation data must live
/// independently of the payload it was read from.
pub(crate) fn value_at_any(payload: &Value, paths: &[&[&str]]) -> Option<Value> {
    paths.iter().find_map(|path| value_at(payload, path))
}

/// Read a nested value as a string-like scalar.
///
/// Numbers and booleans are accepted because both agent and provider schemas may
/// encode identifiers or flags without string types. Empty strings are treated
/// as absent.
pub(crate) fn string_at(payload: &Value, path: &[&str]) -> Option<String> {
    borrowed_value_at(payload, path)
        .and_then(|value| match value {
            Value::String(value) => Some(value.clone()),
            Value::Number(value) => Some(value.to_string()),
            Value::Bool(value) => Some(value.to_string()),
            _ => None,
        })
        .filter(|value| !value.is_empty())
}

/// Read the first string-like value from any candidate JSON path.
///
/// Paths that exist but contain non-scalar or empty values are skipped so a
/// later compatible field can supply the correlation value.
pub(crate) fn string_at_any(payload: &Value, paths: &[&[&str]]) -> Option<String> {
    paths.iter().find_map(|path| string_at(payload, path))
}
