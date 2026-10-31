// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared plugin diagnostic data types.

use serde::{Deserialize, Serialize};

/// Severity of a plugin configuration or activation diagnostic.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticLevel {
    /// Non-fatal compatibility, validation, or activation issue.
    Warning,
    /// Fatal validation or activation issue that blocks initialization.
    Error,
}

/// Structured diagnostic reported while validating or activating plugin configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
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

/// Encode one plugin registration namespace field.
pub fn encode_plugin_component_field(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";

    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX[(byte >> 4) as usize]));
            encoded.push(char::from(HEX[(byte & 0x0f) as usize]));
        }
    }
    encoded
}

/// Decode a canonical plugin registration name into its owner and local name.
pub fn decode_plugin_component_effective_name(
    effective_name: &str,
) -> Option<(String, u32, String)> {
    const PREFIX: &str = "nemo-relay-plugin.v1.";

    let rest = effective_name.strip_prefix(PREFIX)?;
    let (encoded_kind, rest) = rest.split_once(':')?;
    let (ordinal_text, encoded_local_name) = rest.split_once(':')?;
    let ordinal = ordinal_text
        .parse::<u32>()
        .ok()
        .filter(|ordinal| *ordinal > 0)?;
    if ordinal.to_string() != ordinal_text {
        return None;
    }
    let plugin_kind = decode_plugin_component_field(encoded_kind)?;
    let local_name = decode_plugin_component_field(encoded_local_name)?;
    (!plugin_kind.is_empty() && !local_name.is_empty()).then_some((
        plugin_kind,
        ordinal,
        local_name,
    ))
}

fn decode_plugin_component_field(encoded: &str) -> Option<String> {
    let mut bytes = Vec::with_capacity(encoded.len());
    let encoded = encoded.as_bytes();
    let mut index = 0;

    while index < encoded.len() {
        let byte = encoded[index];
        if byte == b'%' {
            let high = *encoded.get(index + 1)?;
            let low = *encoded.get(index + 2)?;
            bytes.push((hex_value(high)? << 4) | hex_value(low)?);
            index += 3;
        } else if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            bytes.push(byte);
            index += 1;
        } else {
            return None;
        }
    }

    let decoded = String::from_utf8(bytes).ok()?;
    (encode_plugin_component_field(&decoded) == std::str::from_utf8(encoded).ok()?)
        .then_some(decoded)
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}
