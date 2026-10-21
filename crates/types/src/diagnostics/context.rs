// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distinct execution and phase names with lossless future-value decoding.

use super::{PluginExecution, PluginPhase};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// A plugin execution mechanism name, preserving values unknown to this reader.
/// Typed conversion accepts only the corresponding catalogue enum.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(transparent)]
pub struct PluginExecutionName(Cow<'static, str>);
impl PluginExecutionName {
    /// Explicitly preserve a wire name; this does not authorize recording it.
    pub fn from_opaque_name(name: Cow<'static, str>) -> Self {
        Self(name)
    }
    /// Borrow the name without interpreting an unfamiliar value.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
    /// Recognize a supported value without discarding an unknown name.
    #[must_use]
    pub fn known(&self) -> Option<PluginExecution> {
        PluginExecution::from_canonical_name(&self.0)
    }
    /// Transfer the borrowed name or owned string without cloning.
    pub fn into_name(self) -> Cow<'static, str> {
        self.0
    }
}
impl From<PluginExecution> for PluginExecutionName {
    fn from(value: PluginExecution) -> Self {
        Self(Cow::Borrowed(value.as_str()))
    }
}

/// A plugin lifecycle phase name, preserving values unknown to this reader.
/// Typed conversion accepts only the corresponding catalogue enum.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(transparent)]
pub struct PluginPhaseName(Cow<'static, str>);
impl PluginPhaseName {
    /// Explicitly preserve a wire name; this does not authorize recording it.
    pub fn from_opaque_name(name: Cow<'static, str>) -> Self {
        Self(name)
    }
    /// Borrow the name without interpreting an unfamiliar value.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
    /// Recognize a supported value without discarding an unknown name.
    #[must_use]
    pub fn known(&self) -> Option<PluginPhase> {
        PluginPhase::from_canonical_name(&self.0)
    }
    /// Transfer the borrowed name or owned string without cloning.
    pub fn into_name(self) -> Cow<'static, str> {
        self.0
    }
}
impl From<PluginPhase> for PluginPhaseName {
    fn from(value: PluginPhase) -> Self {
        Self(Cow::Borrowed(value.as_str()))
    }
}

/// Shared plugin-operation facts for producers, projections, and decoded evidence.
///
/// Known producer values borrow canonical static names; decoded values own their
/// strings, including future names. Core validates and encodes recording input
/// into private fixed-size storage. Missing values remain distinct from names
/// unrecognized by a reader, and context never establishes counter dimensions.
/// Execution and phase have distinct wrappers, so mixing their typed values is
/// rejected before recording:
///
/// ```compile_fail
/// use nemo_relay_types::diagnostics::{PluginObservationContext, PluginPhase};
/// let context = PluginObservationContext {
///     execution: Some(PluginPhase::Activation.into()),
///     phase: None,
/// };
/// ```
///
/// Phase-one activation recording supplies only `phase = "activation"` and leaves
/// execution unset. Other producer integrations remain deferred.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PluginObservationContext {
    /// Observed execution name, when known; never inferred from a code.
    pub execution: Option<PluginExecutionName>,
    /// Observed lifecycle phase or policy-gate context, when known.
    pub phase: Option<PluginPhaseName>,
}
