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
fn stability_window_is_clamped_to_history_capacity() {
    let detector = ConvergenceDetector::new(0.001, MAX_HISTORY + 1);

    assert_eq!(detector.stability_window, MAX_HISTORY);
}

#[test]
fn low_error_requires_full_window_before_converging() {
    let mut detector = ConvergenceDetector::new(0.01, 3);

    detector.record_epoch(BettiNumbers::new(1, 0), 0.5, 0.005);
    assert!(!detector.is_converged());

    detector.record_epoch(BettiNumbers::new(1, 0), 0.1, 0.005);
    assert!(!detector.is_converged());

    detector.record_epoch(BettiNumbers::new(1, 0), 0.001, 0.005);
    assert!(detector.is_converged());
}

#[test]
fn stable_betti_and_decreasing_drift_converges() {
    let mut detector = ConvergenceDetector::new(0.001, 3);

    detector.record_epoch(BettiNumbers::new(1, 0), 0.1, 0.0005);
    detector.record_epoch(BettiNumbers::new(1, 0), 0.05, 0.0004);
    detector.record_epoch(BettiNumbers::new(1, 0), 0.001, 0.0003);

    assert!(detector.is_converged());
}

#[test]
fn convergence_decision_reports_latest_epoch_and_gate_status() {
    let mut detector = ConvergenceDetector::new(0.001, 3);

    detector.record_epoch(BettiNumbers::new(2, 1), 0.1, 0.0005);
    detector.record_epoch(BettiNumbers::new(2, 1), 0.05, 0.0004);
    let decision = detector.record_epoch(BettiNumbers::new(2, 1), 0.001, 0.0003);

    assert_eq!(decision.epoch, 3);
    assert_eq!(decision.stability_window, 3);
    assert_eq!(decision.latest_betti, BettiNumbers::new(2, 1));
    assert_eq!(decision.latest_drift, 0.001);
    assert_eq!(decision.latest_error, 0.0003);
    assert!(decision.betti_stable);
    assert!(decision.drift_decreasing);
    assert!(decision.error_converged);
    assert!(decision.converged);
}

#[test]
fn unstable_betti_does_not_converge() {
    let mut detector = ConvergenceDetector::new(0.001, 3);

    detector.record_epoch(BettiNumbers::new(1, 0), 0.05, 0.0005);
    detector.record_epoch(BettiNumbers::new(2, 0), 0.03, 0.0004);
    detector.record_epoch(BettiNumbers::new(1, 0), 0.001, 0.0003);

    assert!(!detector.is_converged());
}

#[test]
fn increasing_drift_does_not_converge() {
    let mut detector = ConvergenceDetector::new(0.001, 3);

    detector.record_epoch(BettiNumbers::new(1, 0), 0.01, 0.0005);
    detector.record_epoch(BettiNumbers::new(1, 0), 0.05, 0.0004);
    detector.record_epoch(BettiNumbers::new(1, 0), 0.001, 0.0003);

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
fn drift_detector_resets_on_non_finite_centroid() {
    let mut detector = DriftDetector::<3>::new();

    detector.update(&[0.0, 0.0, 0.0]);
    detector.update(&[1.0, 0.0, 0.0]);

    assert_eq!(detector.update(&[f64::NAN, 0.0, 0.0]), f64::INFINITY);
    assert!(!detector.has_previous);
    assert_eq!(detector.update(&[2.0, 0.0, 0.0]), 0.0);
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
