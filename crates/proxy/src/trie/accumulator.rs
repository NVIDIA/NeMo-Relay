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
mod tests {
    use super::*;

    /// Helper for approximate equality with both relative and absolute tolerance.
    fn assert_approx(actual: f64, expected: f64, tolerance: f64) {
        let diff = (actual - expected).abs();
        let rel = if expected.abs() > 1e-9 {
            diff / expected.abs()
        } else {
            diff
        };
        assert!(
            rel < tolerance || diff < 1.0,
            "expected ~{expected}, got {actual} (rel={rel:.4}, abs_diff={diff:.4})"
        );
    }

    // -----------------------------------------------------------------------
    // RunningStats basic tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_running_stats_new_defaults() {
        let rs = RunningStats::new();
        assert_eq!(rs.count, 0);
        assert_eq!(rs.mean, 0.0);
        assert_eq!(rs.m2, 0.0);
    }

    #[test]
    fn test_has_samples_false_for_new() {
        let rs = RunningStats::new();
        assert!(!rs.has_samples());
    }

    #[test]
    fn test_has_samples_true_after_add() {
        let mut rs = RunningStats::new();
        rs.add_sample(10.0);
        assert!(rs.has_samples());
    }

    #[test]
    fn test_add_sample_single() {
        let mut rs = RunningStats::new();
        rs.add_sample(10.0);
        assert_eq!(rs.count, 1);
        assert_eq!(rs.mean, 10.0);
    }

    #[test]
    fn test_add_sample_sequence() {
        let mut rs = RunningStats::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            rs.add_sample(v);
        }
        assert_eq!(rs.count, 5);
        assert!((rs.mean - 3.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // compute_metrics tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_metrics_empty() {
        let rs = RunningStats::new();
        let m = rs.compute_metrics();
        assert_eq!(m, PredictionMetrics::default());
    }

    #[test]
    fn test_compute_metrics_with_samples() {
        let mut rs = RunningStats::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            rs.add_sample(v);
        }
        let m = rs.compute_metrics();
        assert_eq!(m.sample_count, 5);
        assert!((m.mean - 3.0).abs() < 1e-10);
        // TDigest tolerance: within 10% relative or 1.0 absolute
        assert_approx(m.p50, 3.0, 0.10);
        assert_approx(m.p90, 4.6, 0.10);
        assert_approx(m.p95, 4.8, 0.10);
    }

    // -----------------------------------------------------------------------
    // merge tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_two_stats() {
        let mut a = RunningStats::new();
        for v in [1.0, 2.0, 3.0] {
            a.add_sample(v);
        }
        let mut b = RunningStats::new();
        for v in [4.0, 5.0, 6.0] {
            b.add_sample(v);
        }
        a.merge(&b);
        assert_eq!(a.count, 6);
        assert!((a.mean - 3.5).abs() < 1e-10);
        // Percentiles should be reasonable for merged data
        let m = a.compute_metrics();
        assert_approx(m.p50, 3.5, 0.20);
    }

    #[test]
    fn test_merge_with_empty_other_is_noop() {
        let mut a = RunningStats::new();
        for v in [1.0, 2.0, 3.0] {
            a.add_sample(v);
        }
        let count_before = a.count;
        let mean_before = a.mean;
        let empty = RunningStats::new();
        a.merge(&empty);
        assert_eq!(a.count, count_before);
        assert_eq!(a.mean, mean_before);
    }

    #[test]
    fn test_merge_into_empty_from_nonempty() {
        let mut empty = RunningStats::new();
        let mut other = RunningStats::new();
        for v in [10.0, 20.0, 30.0] {
            other.add_sample(v);
        }
        empty.merge(&other);
        assert_eq!(empty.count, 3);
        assert!((empty.mean - 20.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // serde round-trip test
    // -----------------------------------------------------------------------

    #[test]
    fn test_running_stats_serde_roundtrip() {
        let mut rs = RunningStats::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            rs.add_sample(v);
        }
        let json = serde_json::to_string(&rs).unwrap();
        let restored: RunningStats = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.count, rs.count);
        assert!((restored.mean - rs.mean).abs() < 1e-10);
        assert!((restored.m2 - rs.m2).abs() < 1e-10);
        // Verify percentiles survive serialization
        let orig_m = rs.compute_metrics();
        let rest_m = restored.compute_metrics();
        assert_approx(rest_m.p50, orig_m.p50, 0.01);
    }

    #[test]
    fn test_running_stats_serde_roundtrip_empty() {
        // Empty RunningStats has TDigest with min=NaN, max=NaN.
        // This test verifies that the custom tdigest_serde module handles
        // the NaN -> null -> 0.0 conversion correctly.
        let rs = RunningStats::new();
        let json = serde_json::to_string(&rs).unwrap();
        let restored: RunningStats = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.count, 0);
        assert_eq!(restored.mean, 0.0);
        assert_eq!(restored.m2, 0.0);
    }

    #[test]
    fn test_node_accumulators_serde_roundtrip() {
        // NodeAccumulators has multiple empty RunningStats fields that
        // all contain TDigests with NaN min/max. Verify full roundtrip.
        let na = NodeAccumulators::default();
        let json = serde_json::to_string(&na).unwrap();
        let restored: NodeAccumulators = serde_json::from_str(&json).unwrap();
        assert!(restored.remaining_calls.is_empty());
        assert!(!restored.all_remaining_calls.has_samples());
    }

    #[test]
    fn test_accumulator_state_serde_roundtrip() {
        // Full AccumulatorState with a node entry. Exercises the complete
        // serialization path that the Redis backend uses.
        let mut state = AccumulatorState::default();
        let mut na = NodeAccumulators::default();
        na.all_remaining_calls.add_sample(42.0);
        state.nodes.insert("workflow/agent".to_string(), na);
        let json = serde_json::to_string(&state).unwrap();
        let restored: AccumulatorState = serde_json::from_str(&json).unwrap();
        assert!(restored.nodes.contains_key("workflow/agent"));
        assert_eq!(
            restored.nodes["workflow/agent"].all_remaining_calls.count,
            1
        );
    }

    // -----------------------------------------------------------------------
    // NAT exact percentile comparison
    // -----------------------------------------------------------------------

    #[test]
    fn test_nat_exact_percentile_helper() {
        let samples: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let p50 = nat_exact_percentile(&samples, 50.0);
        let p90 = nat_exact_percentile(&samples, 90.0);
        let p95 = nat_exact_percentile(&samples, 95.0);
        assert!(
            (p50 - 55.0).abs() < 1e-10,
            "NAT p50 should be 55.0, got {p50}"
        );
        assert!(
            (p90 - 91.0).abs() < 1e-10,
            "NAT p90 should be 91.0, got {p90}"
        );
        assert!(
            (p95 - 95.5).abs() < 1e-10,
            "NAT p95 should be 95.5, got {p95}"
        );
    }

    #[test]
    fn test_tdigest_vs_nat_exact() {
        let samples: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let mut rs = RunningStats::new();
        for &v in &samples {
            rs.add_sample(v);
        }
        let m = rs.compute_metrics();

        // NAT-exact values
        let nat_p50 = 55.0;
        let nat_p90 = 91.0;
        let nat_p95 = 95.5;

        // TDigest should be within 10% relative tolerance
        assert_approx(m.p50, nat_p50, 0.10);
        assert_approx(m.p90, nat_p90, 0.10);
        assert_approx(m.p95, nat_p95, 0.10);
    }

    // -----------------------------------------------------------------------
    // NodeAccumulators tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_node_accumulators_default() {
        let na = NodeAccumulators::default();
        assert!(na.remaining_calls.is_empty());
        assert!(na.interarrival_ms.is_empty());
        assert!(na.output_tokens.is_empty());
        assert!(na.sensitivity.is_empty());
        assert!(!na.all_remaining_calls.has_samples());
        assert!(!na.all_interarrival_ms.has_samples());
        assert!(!na.all_output_tokens.has_samples());
        assert!(!na.all_sensitivity.has_samples());
    }

    // -----------------------------------------------------------------------
    // AccumulatorState tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_accumulator_state_default() {
        let state = AccumulatorState::default();
        assert!(state.nodes.is_empty());
    }

    #[test]
    fn test_accumulator_state_path_key() {
        let mut state = AccumulatorState::default();
        state
            .nodes
            .insert("workflow/agent".to_string(), NodeAccumulators::default());
        assert!(state.nodes.contains_key("workflow/agent"));
        let na = &state.nodes["workflow/agent"];
        assert!(na.remaining_calls.is_empty());
    }
}
