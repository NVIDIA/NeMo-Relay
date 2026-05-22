// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for the nemo-relay-adaptive crate.

use nemo_relay::plugin::PluginError;
use thiserror::Error;

/// The error type for all nemo-relay-adaptive operations.
#[derive(Debug, Error)]
pub enum AdaptiveError {
    /// Configuration validation failed.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// The requested resource was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// A storage operation failed.
    #[error("storage error: {0}")]
    Storage(String),

    /// A serialization or deserialization error.
    #[error("serialization error: {0}")]
    Serialization(serde_json::Error),

    /// An internal error (e.g., lock poisoning).
    #[error("internal error: {0}")]
    Internal(String),

    /// A registration with the NeMo Relay runtime failed.
    #[error("registration failed: {0}")]
    RegistrationFailed(String),

    /// The internal telemetry channel was closed unexpectedly.
    #[error("channel closed: {0}")]
    ChannelClosed(String),

    /// A Redis operation failed.
    #[cfg(feature = "redis-backend")]
    #[error("redis error: {0}")]
    Redis(#[from] redis::RedisError),
}

impl From<serde_json::Error> for AdaptiveError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value)
    }
}

impl From<PluginError> for AdaptiveError {
    fn from(value: PluginError) -> Self {
        match value {
            PluginError::InvalidConfig(message) => Self::InvalidConfig(message),
            PluginError::NotFound(message) => Self::NotFound(message),
            PluginError::Serialization(err) => Self::Serialization(err),
            PluginError::Internal(message) => Self::Internal(message),
            PluginError::RegistrationFailed(message) => Self::RegistrationFailed(message),
        }
    }
}

/// A specialized [`Result`](std::result::Result) type for nemo-relay-adaptive operations.
pub type Result<T> = std::result::Result<T, AdaptiveError>;

#[cfg(test)]
#[path = "../tests/coverage/error_tests.rs"]
mod tests;
