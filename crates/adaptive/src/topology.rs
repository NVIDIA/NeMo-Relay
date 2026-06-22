// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal topology-aware control primitives for adaptive learners.

/// Maximum number of epochs retained in each history ring buffer.
const MAX_HISTORY: usize = 32;

/// Minimum stability window length.
const MIN_STABILITY_WINDOW: usize = 3;

/// Drift values below this threshold are considered converged.
const FINAL_DRIFT_THRESHOLD: f64 = 0.01;

/// Default convergence error threshold.
const DEFAULT_EPSILON: f64 = 0.001;

/// Target effective tick rate the governor tries to maintain, measured in Hz.
const TARGET_TICK_RATE: f64 = 1000.0;

/// Proportional gain applied to the instantaneous control error.
const GOVERNOR_ALPHA: f64 = 0.01;

/// Derivative gain applied to the rate of change of the control error.
const GOVERNOR_BETA: f64 = 0.05;

/// Minimum allowed governor threshold.
const GOVERNOR_EPSILON_MIN: f64 = 0.001;

/// Maximum allowed governor threshold.
const GOVERNOR_EPSILON_MAX: f64 = 10.0;

/// Default governor threshold.
const GOVERNOR_EPSILON_INITIAL: f64 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BettiNumbers {
    pub(crate) beta_0: u32,
    pub(crate) beta_1: u32,
}

impl BettiNumbers {
    pub(crate) const fn new(beta_0: u32, beta_1: u32) -> Self {
        Self { beta_0, beta_1 }
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

#[derive(Debug, Clone, Copy, PartialEq)]
struct RingBuffer<T: Copy + Default, const N: usize> {
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

    fn len(&self) -> usize {
        self.len
    }

    fn copy_window(&self, window_size: usize, out: &mut [T]) -> usize {
        let window_size = window_size.min(self.len).min(out.len());
        if window_size == 0 {
            return 0;
        }

        let start = (self.pos + N - window_size) % N;
        for (index, slot) in out.iter_mut().enumerate().take(window_size) {
            *slot = self.data[(start + index) % N];
        }
        window_size
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ConvergenceDetector {
    betti_history: RingBuffer<BettiNumbers, MAX_HISTORY>,
    drift_history: RingBuffer<f64, MAX_HISTORY>,
    error_history: RingBuffer<f64, MAX_HISTORY>,
    stability_window: usize,
    epsilon: f64,
    epoch: u64,
}

impl ConvergenceDetector {
    pub(crate) fn new(epsilon: f64, stability_window: usize) -> Self {
        Self {
            betti_history: RingBuffer::new(),
            drift_history: RingBuffer::new(),
            error_history: RingBuffer::new(),
            stability_window: stability_window.clamp(MIN_STABILITY_WINDOW, MAX_HISTORY),
            epsilon: sanitize_positive(epsilon, DEFAULT_EPSILON),
            epoch: 0,
        }
    }

    pub(crate) fn record_epoch(&mut self, betti: BettiNumbers, drift: f64, error: f64) {
        self.betti_history.push(betti);
        self.drift_history.push(sanitize_non_negative(drift));
        self.error_history.push(sanitize_non_negative(error));
        self.epoch = self.epoch.saturating_add(1);
    }

    pub(crate) fn is_converged(&self) -> bool {
        self.is_betti_stable() && self.is_drift_decreasing() && self.is_error_window_converged()
    }

    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }

    fn is_error_window_converged(&self) -> bool {
        if self.error_history.len() < self.stability_window {
            return false;
        }

        let mut window = [0.0; MAX_HISTORY];
        let count = self
            .error_history
            .copy_window(self.stability_window, &mut window);
        window[..count]
            .iter()
            .all(|error| error.is_finite() && *error < self.epsilon)
    }

    fn is_betti_stable(&self) -> bool {
        if self.betti_history.len() < self.stability_window {
            return false;
        }

        let mut window = [BettiNumbers::default(); MAX_HISTORY];
        let count = self
            .betti_history
            .copy_window(self.stability_window, &mut window);
        let first = window[0];
        window[..count].iter().all(|betti| *betti == first)
    }

    fn is_drift_decreasing(&self) -> bool {
        if self.drift_history.len() < self.stability_window {
            return false;
        }

        let mut window = [0.0; MAX_HISTORY];
        let count = self
            .drift_history
            .copy_window(self.stability_window, &mut window);
        if window[..count].iter().any(|drift| !drift.is_finite()) {
            return false;
        }

        for pair in window[..count].windows(2) {
            if pair[1] > pair[0] {
                return false;
            }
        }

        window[count - 1] < FINAL_DRIFT_THRESHOLD
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DriftDetector<const D: usize> {
    previous: [f64; D],
    has_previous: bool,
    velocity: [f64; D],
    expected: [f64; D],
}

impl<const D: usize> DriftDetector<D> {
    pub(crate) fn new() -> Self {
        Self {
            previous: [0.0; D],
            has_previous: false,
            velocity: [0.0; D],
            expected: [0.0; D],
        }
    }

    pub(crate) fn update(&mut self, centroid: &[f64; D]) -> f64 {
        if centroid.iter().any(|coord| !coord.is_finite()) {
            self.reset();
            return f64::INFINITY;
        }

        let drift = if self.has_previous {
            l2_distance(&self.expected, centroid)
        } else {
            0.0
        };

        if self.has_previous {
            for (dimension, velocity) in self.velocity.iter_mut().enumerate() {
                *velocity = centroid[dimension] - self.previous[dimension];
            }
        }

        for (dimension, expected) in self.expected.iter_mut().enumerate() {
            *expected = centroid[dimension] + self.velocity[dimension];
        }

        self.previous = *centroid;
        self.has_previous = true;

        drift
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GeometricGovernor {
    epsilon: f64,
    last_error: f64,
    adjustment_count: u64,
}

impl GeometricGovernor {
    fn new() -> Self {
        Self {
            epsilon: GOVERNOR_EPSILON_INITIAL,
            last_error: 0.0,
            adjustment_count: 0,
        }
    }

    pub(crate) fn with_epsilon(epsilon: f64) -> Self {
        let mut governor = Self::new();
        if epsilon.is_finite() {
            governor.epsilon = epsilon.clamp(GOVERNOR_EPSILON_MIN, GOVERNOR_EPSILON_MAX);
        }
        governor
    }

    pub(crate) fn adapt(&mut self, deviation_delta: f64, dt: f64) -> f64 {
        if dt <= 0.0 || !dt.is_finite() || !deviation_delta.is_finite() || self.epsilon <= 0.0 {
            return self.epsilon;
        }

        let current_rate = deviation_delta / self.epsilon;
        let error = TARGET_TICK_RATE - current_rate;
        let d_error = (error - self.last_error) / dt;
        let adjustment = GOVERNOR_ALPHA * error + GOVERNOR_BETA * d_error;

        self.epsilon =
            (self.epsilon - adjustment).clamp(GOVERNOR_EPSILON_MIN, GOVERNOR_EPSILON_MAX);
        self.last_error = error;
        self.adjustment_count = self.adjustment_count.saturating_add(1);

        self.epsilon
    }

    pub(crate) fn should_trigger(&self, deviation: f64) -> bool {
        deviation.is_finite() && deviation >= self.epsilon
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

fn l2_distance<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut sum = 0.0;
    for dimension in 0..D {
        let diff = a[dimension] - b[dimension];
        sum += diff * diff;
    }
    sum.sqrt()
}

#[cfg(test)]
#[path = "../tests/unit/topology_tests.rs"]
mod tests;
