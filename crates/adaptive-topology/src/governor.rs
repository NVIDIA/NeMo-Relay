// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adaptive threshold controller using a PD control law on an effective tick rate.

/// Target effective tick rate the governor tries to maintain, measured in Hz.
///
/// This value balances responsiveness against overhead: higher rates react
/// faster but consume more CPU, while lower rates are cheaper but may miss
/// transients.
const TARGET_TICK_RATE: f64 = 1000.0;

/// Proportional gain applied to the instantaneous control error.
const ALPHA: f64 = 0.01;

/// Derivative gain applied to the rate of change of the control error.
///
/// The derivative term dampens oscillations caused by the proportional
/// response alone.
const BETA: f64 = 0.05;

/// Minimum allowed epsilon. Prevents the threshold from collapsing to zero,
/// which would force continuous triggering.
const EPSILON_MIN: f64 = 0.001;

/// Maximum allowed epsilon. Prevents the threshold from growing so large
/// that the system never wakes up.
const EPSILON_MAX: f64 = 10.0;

/// Default initial epsilon when no explicit starting threshold is supplied.
const EPSILON_INITIAL: f64 = 0.1;

/// Adaptive threshold controller.
///
/// The controller maintains a sensitivity threshold `epsilon`. It observes
/// the effective tick rate `deviation_delta / epsilon` and adjusts epsilon
/// so the rate stays near `TARGET_TICK_RATE`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricGovernor {
    epsilon: f64,
    last_error: f64,
    alpha: f64,
    beta: f64,
    adjustment_count: u64,
}

impl GeometricGovernor {
    /// Create a new governor with the default threshold and gains.
    pub fn new() -> Self {
        Self {
            epsilon: EPSILON_INITIAL,
            last_error: 0.0,
            alpha: ALPHA,
            beta: BETA,
            adjustment_count: 0,
        }
    }

    /// Create a governor with a custom initial threshold.
    ///
    /// The supplied value is clamped to `EPSILON_MIN` and `EPSILON_MAX`.
    pub fn with_epsilon(epsilon: f64) -> Self {
        let mut gov = Self::new();
        if epsilon.is_finite() {
            gov.epsilon = epsilon.clamp(EPSILON_MIN, EPSILON_MAX);
        }
        gov
    }

    /// Create a governor with custom proportional and derivative gains.
    pub fn with_gains(alpha: f64, beta: f64) -> Self {
        Self {
            epsilon: EPSILON_INITIAL,
            last_error: 0.0,
            alpha: if alpha.is_finite() { alpha } else { ALPHA },
            beta: if beta.is_finite() { beta } else { BETA },
            adjustment_count: 0,
        }
    }

    /// Return the current epsilon value.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Return the number of adaptations performed so far.
    pub fn adjustment_count(&self) -> u64 {
        self.adjustment_count
    }

    /// Return the error from the most recent adaptation.
    pub fn last_error(&self) -> f64 {
        self.last_error
    }

    /// Adapt epsilon based on the observed deviation over `dt` seconds.
    ///
    /// The effective tick rate is `deviation_delta / epsilon`. The control
    /// error is `TARGET_TICK_RATE - rate`. A positive error means the system
    /// is too slow and epsilon should be lowered; a negative error means the
    /// system is too fast and epsilon should be raised.
    pub fn adapt(&mut self, deviation_delta: f64, dt: f64) -> f64 {
        if dt <= 0.0 || !dt.is_finite() || !deviation_delta.is_finite() || self.epsilon <= 0.0 {
            return self.epsilon;
        }

        let current_rate = deviation_delta / self.epsilon;
        let error = TARGET_TICK_RATE - current_rate;
        let d_error = (error - self.last_error) / dt;
        let adjustment = self.alpha * error + self.beta * d_error;

        // Subtract the adjustment so that a negative error (too fast) raises
        // epsilon and a positive error (too slow) lowers epsilon. This fixes
        // the sign bug in the original Aether-Lang implementation.
        self.epsilon = (self.epsilon - adjustment).clamp(EPSILON_MIN, EPSILON_MAX);
        self.last_error = error;
        self.adjustment_count += 1;

        self.epsilon
    }

    /// Return true if `deviation` meets or exceeds the current threshold.
    pub fn should_trigger(&self, deviation: f64) -> bool {
        deviation.is_finite() && deviation >= self.epsilon
    }

    /// Reset the governor to its initial state.
    pub fn reset(&mut self) {
        self.epsilon = EPSILON_INITIAL;
        self.last_error = 0.0;
        self.adjustment_count = 0;
    }
}

impl Default for GeometricGovernor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_epsilon_is_default() {
        let gov = GeometricGovernor::new();
        assert!((gov.epsilon() - EPSILON_INITIAL).abs() < 1e-10);
    }

    #[test]
    fn custom_epsilon_is_clamped() {
        let low = GeometricGovernor::with_epsilon(0.0001);
        assert!((low.epsilon() - EPSILON_MIN).abs() < 1e-10);

        let high = GeometricGovernor::with_epsilon(100.0);
        assert!((high.epsilon() - EPSILON_MAX).abs() < 1e-10);
    }

    #[test]
    fn non_finite_inputs_keep_governor_state_valid() {
        let mut gov = GeometricGovernor::with_epsilon(f64::NAN);
        assert!((gov.epsilon() - EPSILON_INITIAL).abs() < 1e-10);
        assert_eq!(gov.adapt(f64::NAN, 1.0), EPSILON_INITIAL);
        assert_eq!(gov.adapt(1.0, f64::INFINITY), EPSILON_INITIAL);
        assert!(!gov.should_trigger(f64::NAN));
    }

    #[test]
    fn high_load_raises_epsilon() {
        let mut gov = GeometricGovernor::new();
        let initial = gov.epsilon();

        // High deviation means the system is waking too often, so epsilon
        // should increase to reduce sensitivity.
        gov.adapt(10_000.0, 0.001);

        assert!(gov.epsilon() > initial, "high deviation must raise epsilon");
    }

    #[test]
    fn low_load_lowers_epsilon() {
        let mut gov = GeometricGovernor::new();
        let initial = gov.epsilon();

        // Very low deviation means the system is too sluggish, so epsilon
        // should decrease to increase sensitivity.
        for _ in 0..10 {
            gov.adapt(0.0001, 0.001);
        }

        assert!(gov.epsilon() < initial, "low deviation must lower epsilon");
    }

    #[test]
    fn epsilon_stays_within_bounds() {
        let mut gov = GeometricGovernor::new();
        for _ in 0..10_000 {
            gov.adapt(1_000_000.0, 0.001);
        }
        assert!(gov.epsilon() >= EPSILON_MIN);
        assert!(gov.epsilon() <= EPSILON_MAX);
    }

    #[test]
    fn zero_dt_is_ignored() {
        let mut gov = GeometricGovernor::new();
        let before = gov.epsilon();
        assert_eq!(gov.adapt(100.0, 0.0), before);
        assert_eq!(gov.adjustment_count(), 0);
    }

    #[test]
    fn trigger_threshold() {
        let gov = GeometricGovernor::with_epsilon(0.5);
        assert!(!gov.should_trigger(0.4));
        assert!(gov.should_trigger(0.5));
        assert!(gov.should_trigger(0.6));
    }

    #[test]
    fn reset_clears_state() {
        let mut gov = GeometricGovernor::new();
        gov.adapt(10_000.0, 0.001);
        gov.reset();
        assert!((gov.epsilon() - EPSILON_INITIAL).abs() < 1e-10);
        assert_eq!(gov.adjustment_count(), 0);
        assert_eq!(gov.last_error(), 0.0);
    }
}
