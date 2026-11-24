// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LLM execution context with a separate map for application data.
//!
//! Relay checks the context structure. Middleware checks application data and
//! decides whether to trust it. Application keys stay inside the data map so
//! they cannot conflict with future Relay fields.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::Json;

const MAXIMUM_CONTEXT_SERIALIZED_BYTES: usize = 16_384;

/// Canonical Draft 2020-12 schema shared by managed execution and worker decoding.
pub const APPLICATION_EXECUTION_CONTEXT_SCHEMA: &str =
    include_str!("../../../schemas/llm-application-execution-context.schema.json");

/// A context validation error that does not include input values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ApplicationExecutionContextError {
    /// The value does not match the context schema.
    #[error("LLM execution context does not match ApplicationExecutionContext JSON Schema")]
    InvalidSchema,
    /// The context exceeds the nesting limit.
    #[error("LLM execution context exceeds 64 container levels (root object is level 1)")]
    TooDeep,
    /// The complete context exceeds the size limit.
    #[error("LLM execution context exceeds 16384 serialized bytes")]
    TooLarge,
}

/// Application context for one LLM call.
///
/// Put application keys inside [`Self::data`]. Relay checks the JSON structure
/// and resource limits. Middleware defines and checks the contents of `data`.
/// No context, `{}`, and `{"data":{}}` are three different values.
/// Use [`Self::try_from`] to read JSON input. Use [`Self::default`] to create
/// an empty context.
#[non_exhaustive]
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Json")]
pub struct ApplicationExecutionContext {
    /// Application JSON object. Each middleware call gets its own copy.
    ///
    /// Relay does not treat these keys as built-in fields or as permission to
    /// perform an action. Omitting data differs from passing an empty map.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<BTreeMap<String, Json>>,
}

// Application data must not leak through routine Debug output.
impl std::fmt::Debug for ApplicationExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApplicationExecutionContext")
            .finish_non_exhaustive()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawApplicationExecutionContext {
    data: Option<BTreeMap<String, Json>>,
}

impl ApplicationExecutionContext {
    /// Check wire bytes before parsing a JSON tree. Wire contexts use compact JSON.
    pub fn validate_json_input(input: &[u8]) -> Result<(), ApplicationExecutionContextError> {
        if input.len() > MAXIMUM_CONTEXT_SERIALIZED_BYTES {
            return Err(ApplicationExecutionContextError::TooLarge);
        }
        let mut depth = 0_usize;
        let mut quoted = false;
        let mut escaped = false;
        for &byte in input {
            if quoted {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    quoted = false;
                }
            } else {
                match byte {
                    b'"' => quoted = true,
                    b'{' | b'[' => {
                        depth += 1;
                        if depth > 64 {
                            return Err(ApplicationExecutionContextError::TooDeep);
                        }
                    }
                    b'}' | b']' => depth = depth.saturating_sub(1),
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Decode a wire context after enforcing its byte and nesting limits.
    pub fn from_json_slice(input: &[u8]) -> Result<Self, ApplicationExecutionContextError> {
        Self::validate_json_input(input)?;
        serde_json::from_slice(input).map_err(|_| ApplicationExecutionContextError::InvalidSchema)
    }

    /// Check the context structure, JSON data, and resource limits.
    ///
    /// Check size and depth before encoding nested data as JSON.
    /// Error messages do not include input values.
    pub fn validate_json(value: &Json) -> Result<(), ApplicationExecutionContextError> {
        if !value.is_object() {
            return Err(ApplicationExecutionContextError::InvalidSchema);
        }
        let mut remaining_bytes = MAXIMUM_CONTEXT_SERIALIZED_BYTES;
        validate_resource_budget(value, 1, &mut remaining_bytes)?;
        validate_serialized_size(value)?;
        if value.as_object().is_none_or(|fields| {
            fields.keys().any(|key| key != "data")
                || fields.get("data").is_some_and(|data| !data.is_object())
        }) {
            return Err(ApplicationExecutionContextError::InvalidSchema);
        }
        Ok(())
    }

    /// Return the context as JSON, including application data.
    #[must_use]
    pub fn to_json(&self) -> Json {
        serde_json::to_value(self).expect("application context contains only JSON values")
    }

    /// Check data set by application code before running a managed call.
    pub fn validate(&self) -> Result<(), ApplicationExecutionContextError> {
        let mut remaining_bytes = MAXIMUM_CONTEXT_SERIALIZED_BYTES;
        if let Some(data) = &self.data {
            charge_context_budget(&mut remaining_bytes, data.len().saturating_sub(1))?;
            for (key, value) in data {
                charge_context_budget(&mut remaining_bytes, key.len().saturating_add(3))?;
                validate_resource_budget(value, 3, &mut remaining_bytes)?;
            }
        }
        validate_serialized_size(self)?;
        Self::validate_json(&self.to_json())
    }
}

impl TryFrom<Json> for ApplicationExecutionContext {
    type Error = ApplicationExecutionContextError;

    fn try_from(value: Json) -> Result<Self, Self::Error> {
        if let Err(error) = Self::validate_json(&value) {
            drop_json_iteratively(value);
            return Err(error);
        }
        let raw: RawApplicationExecutionContext = serde_json::from_value(value)
            .map_err(|_| ApplicationExecutionContextError::InvalidSchema)?;
        Ok(Self { data: raw.data })
    }
}

// Rejected programmatically constructed trees may be much deeper than the
// accepted limit. Never let serde_json's recursive destructor unwind them.
impl Drop for ApplicationExecutionContext {
    fn drop(&mut self) {
        if let Some(data) = self.data.take() {
            for value in data.into_values() {
                drop_json_iteratively(value);
            }
        }
    }
}

fn drop_json_iteratively(value: Json) {
    let mut pending = vec![value];
    while let Some(value) = pending.pop() {
        match value {
            Json::Array(items) => pending.extend(items),
            Json::Object(fields) => pending.extend(fields.into_values()),
            _ => {}
        }
    }
}

fn charge_context_budget(
    remaining_bytes: &mut usize,
    minimum_bytes: usize,
) -> Result<(), ApplicationExecutionContextError> {
    *remaining_bytes = remaining_bytes
        .checked_sub(minimum_bytes)
        .ok_or(ApplicationExecutionContextError::TooLarge)?;
    Ok(())
}

fn validate_resource_budget(
    value: &Json,
    depth: usize,
    remaining_bytes: &mut usize,
) -> Result<(), ApplicationExecutionContextError> {
    if (value.is_object() || value.is_array()) && depth > 64 {
        return Err(ApplicationExecutionContextError::TooDeep);
    }
    // A lower bound rejects wide containers and long strings before scanning
    // or expanding them; the writer below enforces exact escaped UTF-8 size.
    let minimum_bytes = match value {
        Json::String(text) => text.len().saturating_add(2),
        Json::Object(fields) => fields.len().saturating_sub(1).saturating_add(2),
        Json::Array(items) => items.len().saturating_sub(1).saturating_add(2),
        _ => 1,
    };
    charge_context_budget(remaining_bytes, minimum_bytes)?;
    match value {
        Json::Object(fields) => {
            for (key, value) in fields {
                charge_context_budget(remaining_bytes, key.len().saturating_add(3))?;
                validate_resource_budget(value, depth + 1, remaining_bytes)?;
            }
        }
        Json::Array(items) => {
            for value in items {
                validate_resource_budget(value, depth + 1, remaining_bytes)?;
            }
        }
        _ => {}
    }
    Ok(())
}

struct ContextSizeLimitWriter {
    remaining_bytes: usize,
}

impl std::io::Write for ContextSizeLimitWriter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.remaining_bytes = self
            .remaining_bytes
            .checked_sub(buffer.len())
            .ok_or_else(|| {
                std::io::Error::other("application execution context size limit exceeded")
            })?;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn validate_serialized_size(
    value: &impl Serialize,
) -> Result<(), ApplicationExecutionContextError> {
    let mut writer = ContextSizeLimitWriter {
        remaining_bytes: MAXIMUM_CONTEXT_SERIALIZED_BYTES,
    };
    serde_json::to_writer(&mut writer, value).map_err(|error| {
        if error.is_io() {
            ApplicationExecutionContextError::TooLarge
        } else {
            ApplicationExecutionContextError::InvalidSchema
        }
    })
}
