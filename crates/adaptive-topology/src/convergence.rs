// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Topological convergence detection via Betti-number stability and drift decay.

use libm::exp;

/// Maximum number of epochs retained in each history ring buffer.
const MAX_HISTORY: usize = 32;

/// Default minimum stability window length.
const MIN_STABILITY_WINDOW: usize = 3;

/// Drift values below this threshold are considered converged.
const FINAL_DRIFT_THRESHOLD: f64 = 0.01;

/// Weight given to Betti stability in [`ConvergenceDetector::convergence_score`].
const BETTI_STABILITY_WEIGHT: f64 = 0.4;

/// Weight given to drift in [`ConvergenceDetector::convergence_score`].
const DRIFT_WEIGHT: f64 = 0.3;

/// Weight given to error in [`ConvergenceDetector::convergence_score`].
const ERROR_WEIGHT: f64 = 0.3;

const DEFAULT_EPSILON: f64 = 0.001;

/// Topological Betti numbers describing the shape of a point cloud.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BettiNumbers {
    /// Number of connected components.
    pub beta_0: u32,
    /// Number of 1-dimensional holes (loops).
    pub beta_1: u32,
}

impl BettiNumbers {
    /// Create a new pair of Betti numbers.
    pub const fn new(beta_0: u32, beta_1: u32) -> Self {
        Self { beta_0, beta_1 }
    }

    /// Return true when the shape is a single connected component with no loops.
    pub fn is_singular(&self) -> bool {
        self.beta_0 == 1 && self.beta_1 == 0
    }

    /// L1 distance between two Betti signatures.
    pub fn distance(&self, other: &Self) -> u32 {
        self.beta_0
            .abs_diff(other.beta_0)
            .saturating_add(self.beta_1.abs_diff(other.beta_1))
    }
}

impl Default for BettiNumbers {
    fn default() -> Self {
        Self {
            beta_0: 1,
            beta_1: 0,
        }
    }
}

/// Fixed-size ring buffer for tracking a scalar history without allocation.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::Deserialize<'de>"
    ))
)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct RingBuffer<T: Copy + Default, const N: usize> {
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays"))]
    data: [T; N],
    len: usize,
    pos: usize,
}

impl<T: Copy + Default, const N: usize> RingBuffer<T, N> {
    fn new() -> Self {
        Self {
            data: [T::default(); N],
            len: 0,
            pos: 0,
        }
    }

    fn push(&mut self, value: T) {
        self.data[self.pos] = value;
        self.pos = (self.pos + 1) % N;
        if self.len < N {
            self.len += 1;
        }
    }

    fn last(&self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let idx = (self.pos + N - 1) % N;
        Some(self.data[idx])
    }

    fn len(&self) -> usize {
        self.len
    }

    fn copy_window(&self, window_size: usize, out: &mut [T]) -> usize {
        let window_size = window_size.min(self.len).min(out.len());
        if window_size == 0 {
            return 0;
        }

        let start = (self.pos + N - window_size) % N;
        for (i, slot) in out.iter_mut().enumerate().take(window_size) {
            *slot = self.data[(start + i) % N];
        }
        window_size
    }
}

/// Detects convergence using topological stability.
///
/// Convergence is declared when either:
/// * the last recorded error is below `epsilon`, or
/// * Betti numbers are stable over `stability_window` epochs, drift is
///   monotonically decreasing, and the final drift is below
///   `FINAL_DRIFT_THRESHOLD`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConvergenceDetector {
    betti_history: RingBuffer<BettiNumbers, MAX_HISTORY>,
    drift_history: RingBuffer<f64, MAX_HISTORY>,
    error_history: RingBuffer<f64, MAX_HISTORY>,
    stability_window: usize,
    epsilon: f64,
    epoch: u32,
}

impl ConvergenceDetector {
    /// Create a new detector.
    ///
    /// `epsilon` is the error threshold. `stability_window` is the minimum
    /// number of epochs required to judge stability and is clamped to at
    /// least `MIN_STABILITY_WINDOW`.
    pub fn new(epsilon: f64, stability_window: usize) -> Self {
        Self {
            betti_history: RingBuffer::new(),
            drift_history: RingBuffer::new(),
            error_history: RingBuffer::new(),
            stability_window: stability_window.max(MIN_STABILITY_WINDOW),
            epsilon: sanitize_positive(epsilon, DEFAULT_EPSILON),
            epoch: 0,
        }
    }

    /// Record an epoch's metrics.
    pub fn record_epoch(&mut self, betti: BettiNumbers, drift: f64, error: f64) {
        self.betti_history.push(betti);
        self.drift_history.push(sanitize_non_negative(drift));
        self.error_history.push(sanitize_non_negative(error));
        self.epoch += 1;
    }

    /// Return true if convergence criteria are satisfied.
    pub fn is_converged(&self) -> bool {
        if self.is_error_converged() {
            return true;
        }

        self.is_betti_stable() && self.is_drift_decreasing()
    }

    /// Return true if the most recent error is below epsilon.
    fn is_error_converged(&self) -> bool {
        self.error_history
            .last()
            .map(|e| e < self.epsilon)
            .unwrap_or(false)
    }

    /// Return true if Betti numbers are identical across the stability window.
    fn is_betti_stable(&self) -> bool {
        if self.betti_history.len() < self.stability_window {
            return false;
        }

        let mut window = [BettiNumbers::default(); MAX_HISTORY];
        let n = self
            .betti_history
            .copy_window(self.stability_window, &mut window);
        let first = window[0];
        window[..n].iter().all(|b| *b == first)
    }

    /// Return true if drift is monotonically non-increasing and the final
    /// value is below `FINAL_DRIFT_THRESHOLD`.
    fn is_drift_decreasing(&self) -> bool {
        if self.drift_history.len() < self.stability_window {
            return false;
        }

        let mut window = [0.0; MAX_HISTORY];
        let n = self
            .drift_history
            .copy_window(self.stability_window, &mut window);

        for pair in window[..n].windows(2) {
            if pair[1] > pair[0] {
                return false;
            }
        }

        window[n - 1] < FINAL_DRIFT_THRESHOLD
    }

    /// Return a score in `[0, 1]` indicating how close to converged the
    /// detector is, where `1.0` means fully converged.
    pub fn convergence_score(&self) -> f64 {
        let n = self.betti_history.len();
        if n == 0 {
            return 0.0;
        }

        let mut score = 0.0;

        // Betti stability contributes 40%.
        if n >= self.stability_window {
            let mut window = [BettiNumbers::default(); MAX_HISTORY];
            let count = self
                .betti_history
                .copy_window(self.stability_window, &mut window);
            let variations = window[..count].windows(2).filter(|w| w[0] != w[1]).count();
            let betti_score = 1.0 - (variations as f64 / self.stability_window as f64);
            score += BETTI_STABILITY_WEIGHT * betti_score;
        }

        // Drift contribution: exponential decay with drift.
        if let Some(last_drift) = self.drift_history.last() {
            let drift_score = exp(-last_drift * 10.0).min(1.0);
            score += DRIFT_WEIGHT * drift_score;
        }

        // Error contribution.
        if let Some(last_error) = self.error_history.last() {
            let error_score = if last_error < self.epsilon {
                1.0
            } else {
                (self.epsilon / last_error).min(1.0)
            };
            score += ERROR_WEIGHT * error_score;
        }

        score
    }

    /// Return the number of epochs recorded.
    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    /// Return the most recent Betti numbers, if any.
    pub fn last_betti(&self) -> Option<BettiNumbers> {
        self.betti_history.last()
    }

    /// Return the most recent error, if any.
    pub fn last_error(&self) -> Option<f64> {
        self.error_history.last()
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.betti_history = RingBuffer::new();
        self.drift_history = RingBuffer::new();
        self.error_history = RingBuffer::new();
        self.epoch = 0;
    }
}

impl Default for ConvergenceDetector {
    fn default() -> Self {
        Self::new(DEFAULT_EPSILON, MIN_STABILITY_WINDOW)
    }
}

fn sanitize_positive(value: f64, fallback: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

fn sanitize_non_negative(value: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_detector_is_not_converged() {
        let detector = ConvergenceDetector::new(0.001, 3);
        assert!(!detector.is_converged());
        assert_eq!(detector.convergence_score(), 0.0);
    }

    #[test]
    fn betti_distance_handles_extreme_public_inputs() {
        let a = BettiNumbers::new(u32::MAX, 0);
        let b = BettiNumbers::new(0, u32::MAX);
        assert_eq!(a.distance(&b), u32::MAX);
    }

    #[test]
    fn non_finite_metrics_do_not_poison_convergence_score() {
        let mut detector = ConvergenceDetector::new(f64::NAN, 0);
        detector.record_epoch(BettiNumbers::new(1, 0), f64::NAN, f64::NAN);
        assert!(!detector.is_converged());
        assert!(detector.convergence_score().is_finite());
        assert_eq!(detector.last_error(), Some(f64::INFINITY));
    }

    #[test]
    fn error_below_epsilon_converges() {
        let mut detector = ConvergenceDetector::new(0.01, 3);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.5, 0.005);
        assert!(detector.is_converged());
    }

    #[test]
    fn stable_betti_and_decreasing_drift_converges() {
        let mut detector = ConvergenceDetector::new(0.001, 3);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.1, 0.05);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.05, 0.04);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.001, 0.03);
        assert!(detector.is_converged());
    }

    #[test]
    fn unstable_betti_does_not_converge() {
        let mut detector = ConvergenceDetector::new(0.001, 3);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.05, 0.04);
        detector.record_epoch(BettiNumbers::new(2, 0), 0.03, 0.03);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.001, 0.02);
        assert!(!detector.is_converged());
    }

    #[test]
    fn increasing_drift_does_not_converge() {
        let mut detector = ConvergenceDetector::new(0.001, 3);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.01, 0.04);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.05, 0.03);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.001, 0.02);
        assert!(!detector.is_converged());
    }

    #[test]
    fn final_drift_above_threshold_does_not_converge() {
        let mut detector = ConvergenceDetector::new(0.001, 3);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.1, 0.04);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.05, 0.03);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.02, 0.02);
        assert!(!detector.is_converged());
    }

    #[test]
    fn convergence_score_increases_with_stability() {
        let mut detector = ConvergenceDetector::new(0.001, 3);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.5, 0.1);
        let score1 = detector.convergence_score();
        detector.record_epoch(BettiNumbers::new(1, 0), 0.1, 0.05);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.001, 0.001);
        let score2 = detector.convergence_score();
        assert!(score2 > score1);
        assert!(score2 <= 1.0);
    }

    #[test]
    fn ring_buffer_wraps_correctly() {
        let mut buf = RingBuffer::<u32, 4>::new();
        for i in 1..=6 {
            buf.push(i);
        }
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.last(), Some(6));

        let mut window = [0; 4];
        let n = buf.copy_window(4, &mut window);
        assert_eq!(n, 4);
        assert_eq!(window, [3, 4, 5, 6]);
    }

    #[test]
    fn reset_clears_detector() {
        let mut detector = ConvergenceDetector::new(0.001, 3);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.001, 0.0001);
        detector.reset();
        assert_eq!(detector.epoch(), 0);
        assert!(detector.last_betti().is_none());
        assert!(!detector.is_converged());
    }
}
