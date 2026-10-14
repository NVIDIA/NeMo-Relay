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

/// Formats an upstream HTTP error for transport through string-based plugin callbacks.
///
/// Relay recognizes this HTTP prefix at runtime plugin error boundaries. Status codes
/// must be in `400..=599`. Older hosts continue to treat the result as an ordinary error string.
pub fn http_error_message(status: u16, message: &str) -> Result<String, String> {
    if !(400..=599).contains(&status) {
        return Err("plugin HTTP error status must be between 400 and 599".into());
    }
    Ok(format!("HTTP {status}: {message}"))
}

/// Reads the exact HTTP error prefix, leaving the diagnostic message unchanged.
///
/// Prefixes embedded in a message, malformed codes, and non-error statuses are not recognized.
pub fn parse_http_error_message(message: &str) -> Option<(u16, &str)> {
    let suffix = message.strip_prefix("HTTP ")?;
    let (code, body) = suffix.split_once(": ")?;
    if code.len() != 3 || !code.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let status = code.parse().ok()?;
    (400..=599).contains(&status).then_some((status, body))
}

#[cfg(test)]
#[path = "../tests/unit/plugin_http_error_tests.rs"]
mod http_error_tests;
