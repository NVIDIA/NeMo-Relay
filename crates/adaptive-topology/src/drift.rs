// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Centroid trajectory tracking for semantic drift detection.

use libm::sqrt;

/// Tracks the velocity of a centroid trajectory.
///
/// The detector stores the previous centroid and estimates an expected next
/// position from the last observed velocity. Drift is measured as the
/// distance between the actual next centroid and the predicted one.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftDetector<const D: usize> {
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays"))]
    previous: [f64; D],
    has_previous: bool,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays"))]
    velocity: [f64; D],
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays"))]
    expected: [f64; D],
}

impl<const D: usize> DriftDetector<D> {
    /// Create a new detector with no previous observation and zero velocity.
    pub fn new() -> Self {
        Self {
            previous: [0.0; D],
            has_previous: false,
            velocity: [0.0; D],
            expected: [0.0; D],
        }
    }

    /// Record a new centroid and return the drift from the predicted position.
    ///
    /// On the first observation the drift is zero because no prediction is
    /// available yet.
    pub fn update(&mut self, centroid: &[f64; D]) -> f64 {
        let drift = if self.has_previous {
            l2_distance(&self.expected, centroid)
        } else {
            0.0
        };

        if self.has_previous {
            for (d, vel) in self.velocity.iter_mut().enumerate().take(D) {
                *vel = centroid[d] - self.previous[d];
            }
        }

        for (d, exp) in self.expected.iter_mut().enumerate().take(D) {
            *exp = centroid[d] + self.velocity[d];
        }

        self.previous = *centroid;
        self.has_previous = true;

        drift
    }

    /// Return true if the current velocity magnitude exceeds `threshold`.
    pub fn is_drifting(&self, threshold: f64) -> bool {
        self.velocity_magnitude() > threshold
    }

    /// Return the magnitude of the current velocity vector.
    pub fn velocity_magnitude(&self) -> f64 {
        vector_norm(&self.velocity)
    }

    /// Reset the detector to its initial state.
    pub fn reset(&mut self) {
        self.previous = [0.0; D];
        self.has_previous = false;
        self.velocity = [0.0; D];
        self.expected = [0.0; D];
    }
}

impl<const D: usize> Default for DriftDetector<D> {
    fn default() -> Self {
        Self::new()
    }
}

fn l2_distance<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut sum = 0.0;
    for d in 0..D {
        let diff = a[d] - b[d];
        sum += diff * diff;
    }
    sqrt(sum)
}

fn vector_norm<const D: usize>(v: &[f64; D]) -> f64 {
    let mut sum = 0.0;
    for &coord in v.iter() {
        sum += coord * coord;
    }
    sqrt(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_update_has_zero_drift() {
        let mut detector = DriftDetector::<3>::new();
        assert_eq!(detector.update(&[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn steady_velocity_is_tracked() {
        let mut detector = DriftDetector::<3>::new();
        detector.update(&[0.0, 0.0, 0.0]);
        detector.update(&[1.0, 0.0, 0.0]);
        detector.update(&[2.0, 0.0, 0.0]);

        assert!((detector.velocity[0] - 1.0).abs() < 1e-10);
        assert!((detector.velocity_magnitude() - 1.0).abs() < 1e-10);
        assert!(!detector.is_drifting(2.0));
        assert!(detector.is_drifting(0.5));
    }

    #[test]
    fn sudden_drift_is_detected() {
        let mut detector = DriftDetector::<3>::new();
        detector.update(&[0.0, 0.0, 0.0]);
        detector.update(&[1.0, 0.0, 0.0]);
        detector.update(&[2.0, 0.0, 0.0]);

        let drift = detector.update(&[5.0, 0.0, 0.0]);
        assert!(drift > 1.0);
        assert!(detector.is_drifting(1.5));
    }

    #[test]
    fn velocity_tracks_last_step() {
        let mut detector = DriftDetector::<1>::new();
        for i in 0..10 {
            detector.update(&[i as f64]);
        }
        assert!((detector.velocity[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn reset_clears_detector() {
        let mut detector = DriftDetector::<3>::new();
        detector.update(&[1.0, 0.0, 0.0]);
        detector.update(&[2.0, 0.0, 0.0]);
        detector.reset();

        assert_eq!(detector.velocity_magnitude(), 0.0);
        assert_eq!(detector.update(&[5.0, 0.0, 0.0]), 0.0);
    }
}
