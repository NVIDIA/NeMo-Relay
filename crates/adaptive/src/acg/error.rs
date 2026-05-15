// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for the Adaptive Cache Governor (ACG) crate.
//!
//! All fallible operations in the Adaptive Cache Governor (ACG) system return
//! [`Result<T>`], which uses [`AcgError`] as the error type.

use thiserror::Error;

/// The error type for all Adaptive Cache Governor (ACG) operations.
#[derive(Debug, Error)]
pub enum AcgError {
    /// An intent validation failed.
    #[error("invalid intent: {0}")]
    InvalidIntent(String),

    /// A serialization or deserialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// An internal error.
    #[error("internal error: {0}")]
    Internal(String),

    /// A plugin with this ID is already registered.
    #[error("plugin already registered: {0}")]
    PluginAlreadyRegistered(String),

    /// No plugin found with the given ID.
    #[error("plugin not found: {0}")]
    PluginNotFound(String),

    /// Plugin translation failed.
    #[error("plugin translation error: {0}")]
    TranslationFailed(String),

    /// IR construction failed due to invalid input.
    #[error("IR construction error: {0}")]
    IrConstructionError(String),
}

/// A specialized [`Result`](std::result::Result) type for ACG operations.
pub type Result<T> = std::result::Result<T, AcgError>;

#[cfg(test)]
#[path = "../../tests/unit/acg/error_tests.rs"]
mod tests;
