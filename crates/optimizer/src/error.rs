// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for the nemo-flow-optimizer crate.

use thiserror::Error;

/// The error type for all nemo-flow-optimizer operations.
#[derive(Debug, Error)]
pub enum OptimizerError {
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
    Serialization(#[from] serde_json::Error),

    /// An internal error (e.g., lock poisoning).
    #[error("internal error: {0}")]
    Internal(String),

    /// A registration with the NeMo Flow runtime failed.
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

/// A specialized [`Result`](std::result::Result) type for nemo-flow-optimizer operations.
pub type Result<T> = std::result::Result<T, OptimizerError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_found_display() {
        let e = OptimizerError::NotFound("x".into());
        assert_eq!(format!("{e}"), "not found: x");
    }

    #[test]
    fn test_invalid_config_display() {
        let e = OptimizerError::InvalidConfig("bad".into());
        assert_eq!(format!("{e}"), "invalid config: bad");
    }

    #[test]
    fn test_storage_display() {
        let e = OptimizerError::Storage("y".into());
        assert_eq!(format!("{e}"), "storage error: y");
    }

    #[test]
    fn test_internal_display() {
        let e = OptimizerError::Internal("z".into());
        assert_eq!(format!("{e}"), "internal error: z");
    }

    #[test]
    fn test_serialization_from_serde_json() {
        let serde_err = serde_json::from_str::<String>("bad").unwrap_err();
        let e = OptimizerError::from(serde_err);
        let msg = format!("{e}");
        assert!(msg.starts_with("serialization error:"), "got: {msg}");
    }

    #[test]
    fn test_registration_failed_display() {
        let e = OptimizerError::RegistrationFailed("subscriber".into());
        assert_eq!(format!("{e}"), "registration failed: subscriber");
    }

    #[test]
    fn test_channel_closed_display() {
        let e = OptimizerError::ChannelClosed("receiver dropped".into());
        assert_eq!(format!("{e}"), "channel closed: receiver dropped");
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(OptimizerError::Internal("test".into()));
        assert!(e.to_string().contains("internal error"));
    }

    #[cfg(feature = "redis-backend")]
    #[test]
    fn test_redis_error_variant_exists() {
        // Verify that the Redis variant exists and displays correctly.
        // We construct a redis error via an invalid URL to get a RedisError.
        let redis_err = redis::Client::open("invalid://url").unwrap_err();
        let e = OptimizerError::Redis(redis_err);
        let msg = format!("{e}");
        assert!(msg.starts_with("redis error:"), "got: {msg}");
    }
}
