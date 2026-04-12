// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming statistics accumulator using Welford's algorithm and TDigest.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tdigest::TDigest;

use super::data_models::PredictionMetrics;

/// Custom serde module for `TDigest` that handles NaN values in `min`/`max`.
///
/// The `tdigest` crate initializes empty digests with `min=NaN, max=NaN`.
/// `serde_json` serializes `NaN` (via `OrderedFloat`) as JSON `null`, but
/// deserialization then fails because `null` is not a valid `f64`.
/// This module works around the issue by serializing the `TDigest` to
/// `serde_json::Value`, sanitizing any `null` floats to `0.0`, and
/// doing the reverse on deserialization.
mod tdigest_serde {
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};
    use serde_json::Value;
    use tdigest::TDigest;

    /// Replace JSON `null` values with `0.0` in the serialized TDigest.
    fn sanitize_nulls(value: &mut Value) {
        match value {
            Value::Null => *value = Value::Number(serde_json::Number::from_f64(0.0).unwrap()),
            Value::Array(arr) => {
                for item in arr.iter_mut() {
                    sanitize_nulls(item);
                }
            }
            Value::Object(map) => {
                for v in map.values_mut() {
                    sanitize_nulls(v);
                }
            }
            _ => {}
        }
    }

    pub fn serialize<S>(digest: &TDigest, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut value = serde_json::to_value(digest).map_err(serde::ser::Error::custom)?;
        sanitize_nulls(&mut value);
        value.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<TDigest, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut value = Value::deserialize(deserializer)?;
        sanitize_nulls(&mut value);
        serde_json::from_value(value).map_err(serde::de::Error::custom)
    }
}

/// Streaming statistics tracker combining Welford's online algorithm for mean/variance
/// with a TDigest for streaming percentile estimation.
///
/// This replaces NAT's batch `MetricsAccumulator` which stores all raw samples.
/// `RunningStats` provides O(1) memory usage with `merge()` support for incremental
/// trie updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningStats {
    /// Number of samples added.
    pub count: u64,
    /// Running mean (Welford).
    pub mean: f64,
    /// Sum of squared differences from the mean (Welford M2).
    pub m2: f64,
    /// TDigest for streaming percentile estimation.
    ///
    /// Uses custom serde to handle NaN `min`/`max` in empty digests.
    #[serde(with = "tdigest_serde")]
    pub digest: TDigest,
}

/// Per-node accumulators for all metric types, keyed by call index.
///
/// Mirrors NAT's `_NodeAccumulators` structure but uses streaming `RunningStats`
/// instead of batch `MetricsAccumulator`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeAccumulators {
    /// Remaining-calls stats per call index.
    pub remaining_calls: HashMap<u32, RunningStats>,
    /// Interarrival-time stats per call index.
    pub interarrival_ms: HashMap<u32, RunningStats>,
    /// Output-tokens stats per call index.
    pub output_tokens: HashMap<u32, RunningStats>,
    /// Sensitivity stats per call index.
    pub sensitivity: HashMap<u32, RunningStats>,
    /// Aggregated remaining-calls stats across all call indices.
    pub all_remaining_calls: RunningStats,
    /// Aggregated interarrival-time stats across all call indices.
    pub all_interarrival_ms: RunningStats,
    /// Aggregated output-tokens stats across all call indices.
    pub all_output_tokens: RunningStats,
    /// Aggregated sensitivity stats across all call indices.
    pub all_sensitivity: RunningStats,
}

/// Maps trie path strings to their node accumulators.
///
/// Keys are `/`-joined path strings (e.g., `"workflow/agent"`) because
/// `Vec<String>` is not `Hash`. This matches the research recommendation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccumulatorState {
    /// Node accumulators keyed by `/`-joined path string.
    pub nodes: HashMap<String, NodeAccumulators>,
}

impl RunningStats {
    /// Creates a new empty `RunningStats`.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            digest: TDigest::new_with_size(100),
        }
    }

    /// Returns `true` if any samples have been added.
    pub fn has_samples(&self) -> bool {
        self.count > 0
    }

    /// Adds a single sample, updating both Welford accumulators and TDigest.
    ///
    /// Welford's online algorithm maintains running mean and M2 (sum of squared
    /// differences from the mean). TDigest is updated via `merge_unsorted` which
    /// returns a new digest (it consumes `self` by value).
    pub fn add_sample(&mut self, value: f64) {
        // Welford update
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        // TDigest update -- merge_unsorted takes self by value and returns new
        self.digest = self.digest.merge_unsorted(vec![value]);
    }

    /// Merges another `RunningStats` into this one using parallel Welford merge
    /// and TDigest `merge_digests`.
    ///
    /// If `other` is empty, this is a no-op.
    pub fn merge(&mut self, other: &RunningStats) {
        if other.count == 0 {
            return;
        }

        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;

        // Combined Welford merge
        self.mean = if combined_count > 0 {
            (self.mean * self.count as f64 + other.mean * other.count as f64)
                / combined_count as f64
        } else {
            0.0
        };
        self.m2 +=
            other.m2 + delta * delta * (self.count * other.count) as f64 / combined_count as f64;
        self.count = combined_count;

        // TDigest merge
        self.digest = TDigest::merge_digests(vec![self.digest.clone(), other.digest.clone()]);
    }

    /// Computes `PredictionMetrics` from the current accumulator state.
    ///
    /// Returns `PredictionMetrics::default()` if no samples have been added.
    /// Percentiles (p50, p90, p95) are estimated from the TDigest.
    pub fn compute_metrics(&self) -> PredictionMetrics {
        if self.count == 0 {
            return PredictionMetrics::default();
        }

        PredictionMetrics {
            sample_count: self.count as u32,
            mean: self.mean,
            p50: self.digest.estimate_quantile(0.50),
            p90: self.digest.estimate_quantile(0.90),
            p95: self.digest.estimate_quantile(0.95),
        }
    }
}

impl Default for RunningStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes an exact percentile using NAT's linear interpolation algorithm.
/// Test-only helper for comparing TDigest estimates against NAT-exact values.
#[cfg(test)]
pub(crate) fn nat_exact_percentile(sorted_samples: &[f64], pct: f64) -> f64 {
    if sorted_samples.is_empty() {
        return 0.0;
    }
    if sorted_samples.len() == 1 {
        return sorted_samples[0];
    }
    let k = (sorted_samples.len() - 1) as f64 * (pct / 100.0);
    let f = k.floor() as usize;
    let c = k.ceil() as usize;
    if f == c {
        return sorted_samples[f];
    }
    sorted_samples[f] + (sorted_samples[c] - sorted_samples[f]) * (k - f as f64)
}

#[cfg(test)]
#[path = "../../tests/unit/trie/accumulator_tests.rs"]
mod tests;
