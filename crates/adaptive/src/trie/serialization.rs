// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NAT wire-format serialization for prediction tries.

use serde::{Deserialize, Serialize};

use super::data_models::PredictionTrieNode;

/// Version string for the trie wire format.
pub const CURRENT_VERSION: &str = "1.0";

/// Versioned envelope wrapping a prediction trie for JSON persistence.
///
/// Wire format matches NAT's `serialization.py`:
/// ```json
/// {
///   "version": "1.0",
///   "generated_at": "2026-03-31T12:00:00+00:00",
///   "workflow_name": "my_agent",
///   "root": { ... PredictionTrieNode ... }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrieEnvelope {
    pub version: String,
    pub generated_at: String,
    pub workflow_name: String,
    pub root: PredictionTrieNode,
}

impl TrieEnvelope {
    /// Create a new envelope with the current timestamp and version.
    pub fn new(root: PredictionTrieNode, workflow_name: impl Into<String>) -> Self {
        Self {
            version: CURRENT_VERSION.to_string(),
            generated_at: chrono::Utc::now().to_rfc3339(),
            workflow_name: workflow_name.into(),
            root,
        }
    }

    /// Serialize to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/trie/serialization_tests.rs"]
mod tests;
