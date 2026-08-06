# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = EXAMPLE_ROOT / "scripts" / "validate_run.py"
    spec = importlib.util.spec_from_file_location("phase1_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harbor_job_summary_is_not_a_trial_result() -> None:
    module = load_validator()
    assert module.is_trial_result({"n_total_trials": 1, "stats": {}}) is False
    assert module.is_trial_result({"task_name": "task", "verifier_result": {}}) is True


def test_harbor_018_numeric_reward_is_normalized() -> None:
    module = load_validator()
    assert module.read_benchmark_passed({"verifier_result": {"rewards": {"reward": 0.0}}}) is False
    assert module.read_benchmark_passed({"verifier_result": {"rewards": {"reward": 1.0}}}) is True
