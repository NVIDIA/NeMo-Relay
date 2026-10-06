// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared plugin diagnostic data types.

use serde::{Deserialize, Serialize};

/// Severity of a plugin configuration or activation diagnostic.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticLevel {
    /// Non-fatal compatibility, validation, or activation issue.
    Warning,
    /// Fatal validation or activation issue that blocks initialization.
    Error,
}

/// Structured diagnostic reported while validating or activating plugin configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ConfigDiagnostic {
    /// Severity level for the diagnostic.
    pub level: DiagnosticLevel,
    /// Stable diagnostic code suitable for machine checks.
    pub code: String,
    /// Optional component identifier associated with the diagnostic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component: Option<String>,
    /// Optional field path associated with the diagnostic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    /// Human-readable diagnostic message.
    pub message: String,
}
