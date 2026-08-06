# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
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


def test_atof_reader_extracts_switchyard_selected_targets(tmp_path: Path) -> None:
    module = load_validator()
    path = tmp_path / "trajectory.atof.jsonl"
    events = [
        {"name": "switchyard.routing.requested", "data": {"algorithm": "llm_task_classifier"}},
        {
            "name": "switchyard.routing.decision",
            "data": {"selected_target": "weak", "routing_tier": "weak"},
        },
        {
            "name": "switchyard.routing.decision",
            "data": {"selected_target": "strong", "routing_tier": "strong"},
        },
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    count, marks, models, targets = module.read_atof(path)
    assert count == 3
    assert marks == ["switchyard.routing.decision", "switchyard.routing.requested"]
    assert models == []
    assert targets == ["strong", "weak"]


def test_secret_scan_finds_raw_key_within_artifact(tmp_path: Path) -> None:
    module = load_validator()
    artifact = tmp_path / "artifact.log"
    artifact.write_text("raw-provider-key")
    assert module.scan_secrets([artifact], [b"Bearer raw-provider-key", b"raw-provider-key"]) == [
        "artifact.log:secret[1]"
    ]
