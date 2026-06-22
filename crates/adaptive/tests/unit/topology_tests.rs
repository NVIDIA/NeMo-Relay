// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for internal topology-aware adaptive control primitives.

use super::*;

#[test]
fn empty_detector_is_not_converged() {
    let detector = ConvergenceDetector::new(0.001, 3);

    assert!(!detector.is_converged());
    assert_eq!(detector.epoch(), 0);
}

#[test]
fn non_finite_metrics_do_not_converge() {
    let mut detector = ConvergenceDetector::new(f64::NAN, 0);

    detector.record_epoch(BettiNumbers::new(1, 0), f64::NAN, f64::NAN);

    assert!(!detector.is_converged());
    assert_eq!(detector.epoch(), 1);
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
fn drift_detector_tracks_sudden_centroid_change() {
    let mut detector = DriftDetector::<3>::new();

    assert_eq!(detector.update(&[0.0, 0.0, 0.0]), 0.0);
    assert_eq!(detector.update(&[1.0, 0.0, 0.0]), 1.0);
    assert_eq!(detector.update(&[2.0, 0.0, 0.0]), 0.0);

    let drift = detector.update(&[5.0, 0.0, 0.0]);

    assert!(drift > 1.0);
    assert!(detector.velocity[0] > 1.0);
}

#[test]
fn governor_sheds_below_threshold_and_allows_at_threshold() {
    let governor = GeometricGovernor::with_epsilon(0.5);

    assert!(!governor.should_trigger(0.4));
    assert!(governor.should_trigger(0.5));
    assert!(governor.should_trigger(0.6));
}

#[test]
fn governor_clamps_non_finite_and_extreme_inputs() {
    let mut governor = GeometricGovernor::with_epsilon(f64::NAN);

    assert!((governor.epsilon - GOVERNOR_EPSILON_INITIAL).abs() < f64::EPSILON);
    assert_eq!(governor.adapt(f64::NAN, 1.0), GOVERNOR_EPSILON_INITIAL);
    assert_eq!(governor.adapt(1.0, f64::INFINITY), GOVERNOR_EPSILON_INITIAL);
    assert!(!governor.should_trigger(f64::NAN));

    governor.adapt(1_000_000.0, 0.001);
    assert!(governor.epsilon >= GOVERNOR_EPSILON_MIN);
    assert!(governor.epsilon <= GOVERNOR_EPSILON_MAX);
}
