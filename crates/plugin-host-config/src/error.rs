// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use nemo_relay::plugin::PluginError;
use thiserror::Error;

/// Failure while resolving or preparing a file-backed plugin host.
#[derive(Debug, Error)]
pub enum PluginHostConfigError {
    /// A configuration document, lifecycle record, or host policy is invalid.
    #[error("{0}")]
    InvalidConfig(String),
    /// A declared configuration resource was not found.
    #[error("dynamic plugin resource {} was not found: {message}", path.display())]
    NotFound {
        /// Missing resource path.
        path: PathBuf,
        /// Underlying failure detail.
        message: String,
    },
    /// An I/O operation required to prepare durable lifecycle state failed.
    #[error("failed to {operation} {}: {source}", path.display())]
    Io {
        /// Description of the attempted operation.
        operation: String,
        /// Affected path.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
    /// Relay core rejected the resolved configuration or activation plan.
    #[error(transparent)]
    Relay(#[from] PluginError),
    /// JSON serialization of a typed Relay configuration failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl PluginHostConfigError {
    pub(crate) fn io(
        operation: impl Into<String>,
        path: impl Into<PathBuf>,
        source: std::io::Error,
    ) -> Self {
        Self::Io {
            operation: operation.into(),
            path: path.into(),
            source,
        }
    }

    pub(crate) fn toml_parse(description: &str, path: &Path, error: &toml::de::Error) -> Self {
        let location = error
            .span()
            .map(|span| format!(" at bytes {}..{}", span.start, span.end))
            .unwrap_or_default();
        Self::InvalidConfig(format!(
            "invalid {description} in {}{location}: {}",
            path.display(),
            sanitize_parser_reason(error.message())
        ))
    }

    pub(crate) fn json_parse(description: &str, path: &Path, error: &serde_json::Error) -> Self {
        Self::InvalidConfig(format!(
            "invalid {description} in {} at line {}, column {}: {}",
            path.display(),
            error.line(),
            error.column(),
            sanitize_parser_reason(&error.to_string())
        ))
    }

    /// Converts the host-resolution failure to Relay's public plugin error taxonomy.
    pub fn into_plugin_error(self) -> PluginError {
        match self {
            Self::InvalidConfig(message) => PluginError::InvalidConfig(message),
            Self::NotFound { path, message } => {
                PluginError::NotFound(format!("{}: {message}", path.display()))
            }
            Self::Io {
                operation,
                path,
                source,
            } => PluginError::InvalidConfig(format!(
                "failed to {operation} {}: {source}",
                path.display()
            )),
            Self::Relay(error) => error,
            Self::Json(error) => PluginError::Serialization(error),
        }
    }
}

pub(crate) fn sanitize_parser_reason(message: &str) -> String {
    let first_line = message.lines().next().unwrap_or("parse failed");
    let mut sanitized = String::with_capacity(first_line.len());
    let mut characters = first_line.chars().peekable();
    while let Some(character) = characters.next() {
        if !matches!(character, '\'' | '"' | '`') {
            sanitized.push(character);
            continue;
        }
        if character == '`' {
            sanitized.push(character);
            for candidate in characters.by_ref() {
                sanitized.push(candidate);
                if candidate == '`' {
                    break;
                }
            }
            continue;
        }
        sanitized.push(character);
        sanitized.push_str("<redacted>");
        let delimiter = character;
        let mut escaped = false;
        for candidate in characters.by_ref() {
            if escaped {
                escaped = false;
                continue;
            }
            if candidate == '\\' {
                escaped = true;
                continue;
            }
            if candidate == delimiter {
                sanitized.push(delimiter);
                break;
            }
        }
    }
    sanitized
}

#[cfg(test)]
#[path = "../tests/unit/error.rs"]
mod tests;

/// Result returned by file-backed plugin host configuration operations.
pub type Result<T> = std::result::Result<T, PluginHostConfigError>;
