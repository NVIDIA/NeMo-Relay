# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for adaptive topology primitives."""

from nemo_relay.adaptive_topology import ConvergenceDetector, DriftDetector, GeometricGovernor


class TestConvergenceDetector:
    def test_converges_after_stable_epochs(self):
        detector = ConvergenceDetector(epsilon=0.001, stability_window=3)
        detector.record_epoch(1, 0, 0.1, 0.05)
        detector.record_epoch(1, 0, 0.05, 0.04)
        detector.record_epoch(1, 0, 0.001, 0.03)

        assert detector.is_converged
        assert detector.convergence_score > 0.0
        assert detector.epoch == 3

    def test_unstable_betti_numbers_do_not_converge(self):
        detector = ConvergenceDetector(epsilon=0.001, stability_window=3)
        detector.record_epoch(1, 0, 0.05, 0.04)
        detector.record_epoch(2, 0, 0.03, 0.03)
        detector.record_epoch(1, 0, 0.001, 0.02)

        assert not detector.is_converged


class TestGeometricGovernor:
    def test_high_deviation_raises_epsilon(self):
        governor = GeometricGovernor()
        initial = governor.epsilon
        governor.adapt(10_000.0, 0.001)

        assert governor.epsilon > initial
        assert governor.adjustment_count == 1

    def test_trigger_respects_epsilon(self):
        governor = GeometricGovernor()
        assert not governor.should_trigger(governor.epsilon - 0.001)
        assert governor.should_trigger(governor.epsilon)


class TestDriftDetector:
    def test_detects_sudden_drift(self):
        detector = DriftDetector()
        assert detector.update([0.0, 0.0, 0.0]) == 0.0
        detector.update([1.0, 0.0, 0.0])
        detector.update([2.0, 0.0, 0.0])

        drift = detector.update([5.0, 0.0, 0.0])
        assert drift > 1.0
        assert detector.is_drifting(1.5)
        assert not detector.is_drifting(10.0)

    def test_velocity_magnitude_tracks_steady_motion(self):
        detector = DriftDetector()
        detector.update([0.0, 0.0, 0.0])
        detector.update([1.0, 0.0, 0.0])
        detector.update([2.0, 0.0, 0.0])

        assert detector.velocity_magnitude() > 0.9
