// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core prediction trie data types with NAT wire-format compatible serialization.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Aggregated statistics for a single metric from profiler data.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PredictionMetrics {
    /// Number of samples.
    pub sample_count: u32,
    /// Mean value.
    pub mean: f64,
    /// 50th percentile (median).
    pub p50: f64,
    /// 90th percentile.
    pub p90: f64,
    /// 95th percentile.
    pub p95: f64,
}

/// Predictions for an LLM call at a given position in the call hierarchy.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct LlmCallPrediction {
    /// How many more LLM calls are expected after this one.
    pub remaining_calls: PredictionMetrics,
    /// Expected time in milliseconds until the next LLM call.
    pub interarrival_ms: PredictionMetrics,
    /// Expected output token count for this call.
    pub output_tokens: PredictionMetrics,
    /// Auto-computed latency sensitivity score from profiler analysis.
    /// `None` means no profiling data available -- fall back to default.
    pub latency_sensitivity: Option<u32>,
}

/// A node in the prediction trie representing a function in the call hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionTrieNode {
    /// Function name at this level in the hierarchy.
    pub name: String,
    /// Child nodes keyed by function name.
    pub children: HashMap<String, PredictionTrieNode>,
    /// Predictions keyed by call index (1-indexed).
    pub predictions_by_call_index: HashMap<u32, LlmCallPrediction>,
    /// Fallback predictions aggregated across all call indices.
    pub predictions_any_index: Option<LlmCallPrediction>,
}

impl PredictionTrieNode {
    /// Creates a new leaf node with the given name and no children or predictions.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            children: HashMap::new(),
            predictions_by_call_index: HashMap::new(),
            predictions_any_index: None,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/trie/data_models_tests.rs"]
mod tests;
