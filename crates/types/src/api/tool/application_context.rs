// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Extensible application assertions carried separately from tool arguments.
//!
//! JSON Schema validation establishes structure, never approval authenticity.
//! Consumers must trust the calling host and bind its assertion to the current
//! operation. Additional properties are preserved but carry no implicit authority.

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
    /// The value does not conform to the standard extensible object schema.
    #[error("tool execution context does not match ApplicationExecutionContext JSON Schema")]
    InvalidSchema,
    /// Nesting is bounded before serialization, cloning or schema traversal.
    #[error("tool execution context exceeds 64 container levels (root object is level 1)")]
    TooDeep,
    /// The entire object, including extensions, exceeds the shared byte budget.
    #[error("tool execution context exceeds 16384 serialized bytes")]
    TooLarge,
}

/// Invocation-local application context, with recognized fields and open extensions.
///
/// Use [`Self::try_from`] to validate JSON against the published schema. `{}` is
/// valid and differs from omitting context. Unknown properties survive transport;
/// namespace custom keys to reduce collisions with future standard fields.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Json")]
pub struct ApplicationExecutionContext {
    /// Host-verified affirmative human approval, when supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_approval: Option<Box<HostApproval>>,
    /// Additional JSON properties, excluding the reserved `host_approval` key.
    #[serde(flatten)]
    pub additional_properties: BTreeMap<String, Json>,
}

/// Host assertion about one exact, already human-approved operation.
///
/// This is not a signed receipt. The host verifies identity, approval validity,
/// expiry and policy before supplying it. Middleware compares the named tool
/// and approved arguments with its current input before trusting the assertion.
/// Presence of this assertion means completed affirmative approval. The host
/// must stop denied/pending operations. The obsolete `decision` property is
/// reserved and rejected, including in extensions. Relay does not enforce an
/// approval workflow.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Json")]
pub struct HostApproval {
    /// Approval-claim format; the standard schema requires integer `1`.
    pub version: u8,
    /// Nonempty host identity checked against the consumer's configured trust.
    pub source: String,
    /// Nonempty opaque approval identity; its replay policy belongs to the consumer.
    pub approval_id: String,
    /// Exact nonempty tool name covered by this assertion.
    pub tool_name: String,
    /// Approved input snapshot; any JSON value is allowed, like tool arguments.
    pub approved_arguments: Json,
    /// Additional claim properties, with no implicit authorization meaning.
    #[serde(flatten)]
    pub additional_properties: BTreeMap<String, Json>,
}

// Neither assertion nor arbitrary extensions should leak through routine Debug.
impl std::fmt::Debug for ApplicationExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApplicationExecutionContext")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for HostApproval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostApproval").finish_non_exhaustive()
    }
}

#[derive(Deserialize)]
struct RawApplicationExecutionContext {
    host_approval: Option<Box<RawHostApproval>>,
    #[serde(flatten)]
    additional_properties: BTreeMap<String, Json>,
}

// Decode only after the enclosing schema check. Keeping this raw representation
// separate avoids recursively invoking HostApproval's validating deserializer.
#[derive(Deserialize)]
struct RawHostApproval {
    #[serde(deserialize_with = "deserialize_approval_version")]
    version: u8,
    source: String,
    approval_id: String,
    tool_name: String,
    approved_arguments: Json,
    #[serde(flatten)]
    additional_properties: BTreeMap<String, Json>,
}

impl From<RawHostApproval> for HostApproval {
    fn from(raw: RawHostApproval) -> Self {
        Self {
            version: raw.version,
            source: raw.source,
            approval_id: raw.approval_id,
            tool_name: raw.tool_name,
            approved_arguments: raw.approved_arguments,
            additional_properties: raw.additional_properties,
        }
    }
}

impl TryFrom<Json> for HostApproval {
    type Error = ApplicationExecutionContextError;

    /// Validate a standalone claim under the same schema and limits as an invocation.
    fn try_from(value: Json) -> Result<Self, Self::Error> {
        // Move into the envelope without serializing or traversing the input
        // before the shared bounded preflight. The raw decoder above prevents
        // this from recursing through HostApproval::deserialize again.
        let envelope = Json::Object(serde_json::Map::from_iter([(
            "host_approval".into(),
            value,
        )]));
        ApplicationExecutionContext::try_from(envelope)?
            .host_approval
            .map(|approval| *approval)
            .ok_or(ApplicationExecutionContextError::InvalidSchema)
    }
}

impl ApplicationExecutionContext {
    /// Validate object structure, known fields and shared resource limits.
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

    /// Return the standard JSON representation, including additional properties.
    #[must_use]
    pub fn to_json(&self) -> Json {
        serde_json::to_value(self).expect("application context contains only JSON values")
    }

    /// Revalidate a programmatically built value before managed execution.
    pub fn validate(&self) -> Result<(), ApplicationExecutionContextError> {
        // The extension maps may not redefine known fields. Reject before serde
        // could emit duplicate JSON object keys with ambiguous interpretations. The
        // removed decision key remains reserved: an old denial must not become
        // an ignored extension on an affirmative assertion.
        if self.additional_properties.contains_key("host_approval")
            || self.host_approval.as_ref().is_some_and(|approval| {
                approval.additional_properties.keys().any(|key| {
                    matches!(
                        key.as_str(),
                        "version"
                            | "source"
                            | "decision"
                            | "approval_id"
                            | "tool_name"
                            | "approved_arguments"
                    )
                })
            })
        {
            return Err(ApplicationExecutionContextError::InvalidSchema);
        }
        let mut roots = self
            .additional_properties
            .values()
            .map(|value| (value, 2))
            .collect::<Vec<_>>();
        if let Some(approval) = &self.host_approval {
            roots.push((&approval.approved_arguments, 3));
            roots.extend(
                approval
                    .additional_properties
                    .values()
                    .map(|value| (value, 3)),
            );
        }
        validate_depth(roots)?;
        Self::validate_json(&self.to_json())
    }
}

impl TryFrom<Json> for ApplicationExecutionContext {
    type Error = ApplicationExecutionContextError;

    fn try_from(value: Json) -> Result<Self, Self::Error> {
        Self::validate_json(&value)?;
        let raw: RawApplicationExecutionContext = serde_json::from_value(value)
            .map_err(|_| ApplicationExecutionContextError::InvalidSchema)?;
        Ok(Self {
            host_approval: raw
                .host_approval
                .map(|approval| Box::new((*approval).into())),
            additional_properties: raw.additional_properties,
        })
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

fn deserialize_approval_version<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<u8, D::Error> {
    let value = Json::deserialize(deserializer)?;
    // JSON Schema integer semantics include 1.0 and 1e0, but never true.
    if value.as_f64() == Some(1.0) {
        Ok(1)
    } else {
        Err(serde::de::Error::custom(
            "unsupported host approval version",
        ))
    }
}
