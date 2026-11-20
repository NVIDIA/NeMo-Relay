// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete Relay execution context with a separate application-owned data map.
//!
//! JSON Schema validation establishes structure, never approval authenticity.
//! Middleware owns application-data schemas and trust decisions. Application
//! keys cannot collide with future fields of the outer Relay-owned context.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::Json;

/// Canonical Draft 2020-12 schema shared by managed execution and worker decoding.
pub const APPLICATION_EXECUTION_CONTEXT_SCHEMA: &str =
    include_str!("../../../schemas/application-execution-context.schema.json");

static VALIDATOR: LazyLock<jsonschema::Validator> = LazyLock::new(|| {
    let schema: Json = serde_json::from_str(APPLICATION_EXECUTION_CONTEXT_SCHEMA)
        .expect("embedded application execution context schema must be JSON");
    jsonschema::options()
        .with_draft(jsonschema::Draft::Draft202012)
        .build(&schema)
        .expect("embedded application execution context schema must compile")
});

/// A non-sensitive structural refusal; input values are never included.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ApplicationExecutionContextError {
    /// The value does not conform to the concrete context envelope schema.
    #[error("tool execution context does not match ApplicationExecutionContext JSON Schema")]
    InvalidSchema,
    /// Nesting is bounded before serialization, cloning or schema traversal.
    #[error("tool execution context exceeds 64 container levels (root object is level 1)")]
    TooDeep,
    /// The entire object, including application data, exceeds the shared byte budget.
    #[error("tool execution context exceeds 16384 serialized bytes")]
    TooLarge,
}

/// Concrete invocation-local context owned by Relay.
///
/// Application-defined keys belong under [`Self::data`], not at the top level.
/// Relay validates JSON structure and resource limits; middleware defines the
/// meaning, schema and trust policy of application data. `{}` and `{"data":{}}`
/// are distinct valid contexts, both different from omitting context entirely.
/// Use [`Self::try_from`] for JSON input or [`Self::default`] for typed construction.
#[non_exhaustive]
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Json")]
pub struct ApplicationExecutionContext {
    /// Application-owned JSON object, isolated for each middleware invocation.
    ///
    /// Its contents grant no authority by themselves and are never interpreted
    /// as built-in Relay fields. Omission differs from an explicitly empty map.
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
    /// Validate the concrete envelope, JSON data and shared resource limits.
    ///
    /// A bounded preflight runs before schema traversal or recursive serialization.
    /// Error messages deliberately exclude potentially sensitive input values.
    pub fn validate_json(value: &Json) -> Result<(), ApplicationExecutionContextError> {
        if !value.is_object() {
            return Err(ApplicationExecutionContextError::InvalidSchema);
        }
        validate_depth(vec![(value, 1)])?;
        if serde_json::to_vec(value)
            .map_err(|_| ApplicationExecutionContextError::InvalidSchema)?
            .len()
            > 16_384
        {
            return Err(ApplicationExecutionContextError::TooLarge);
        }
        if !VALIDATOR.is_valid(value) {
            return Err(ApplicationExecutionContextError::InvalidSchema);
        }
        Ok(())
    }

    /// Return the concrete JSON envelope, including application-owned data.
    #[must_use]
    pub fn to_json(&self) -> Json {
        serde_json::to_value(self).expect("application context contains only JSON values")
    }

    /// Revalidate programmatically supplied data before managed execution.
    pub fn validate(&self) -> Result<(), ApplicationExecutionContextError> {
        if let Some(data) = &self.data {
            // Envelope is level 1 and data is level 2; payload containers start
            // at level 3. Bound them before recursive JSON serialization.
            validate_depth(data.values().map(|value| (value, 3)).collect())?;
        }
        Self::validate_json(&self.to_json())
    }
}

impl TryFrom<Json> for ApplicationExecutionContext {
    type Error = ApplicationExecutionContextError;

    fn try_from(value: Json) -> Result<Self, Self::Error> {
        Self::validate_json(&value)?;
        let raw: RawApplicationExecutionContext = serde_json::from_value(value)
            .map_err(|_| ApplicationExecutionContextError::InvalidSchema)?;
        Ok(Self { data: raw.data })
    }
}

fn validate_depth(
    mut pending: Vec<(&Json, usize)>,
) -> Result<(), ApplicationExecutionContextError> {
    while let Some((value, depth)) = pending.pop() {
        if !value.is_object() && !value.is_array() {
            continue;
        }
        if depth > 64 {
            return Err(ApplicationExecutionContextError::TooDeep);
        }
        match value {
            Json::Object(fields) => pending.extend(fields.values().map(|value| (value, depth + 1))),
            Json::Array(items) => pending.extend(items.iter().map(|value| (value, depth + 1))),
            _ => unreachable!(),
        }
    }
    Ok(())
}
