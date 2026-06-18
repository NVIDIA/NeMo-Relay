# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive topology primitives.

This module exposes topology-aware helpers for threshold control, convergence
detection, and centroid drift tracking.
"""

from __future__ import annotations

from nemo_relay._native import ConvergenceDetector as ConvergenceDetector
from nemo_relay._native import DriftDetector as DriftDetector
from nemo_relay._native import GeometricGovernor as GeometricGovernor

__all__ = [
    "ConvergenceDetector",
    "DriftDetector",
    "GeometricGovernor",
]
