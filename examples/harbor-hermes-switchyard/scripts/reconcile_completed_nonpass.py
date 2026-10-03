# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile verifier-backed benchmark completion after validator repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "harbor-hermes-switchyard.nonpass-reconciliation.v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_reconcilable(validation: dict[str, Any], upload: dict[str, Any]) -> None:
    benchmark = validation.get("benchmark")
    integration = validation.get("integration")
    if validation.get("status") != "passed" or not isinstance(benchmark, dict) or benchmark.get("status") != "passed":
        raise ValueError("candidate validation does not mark benchmark execution complete")
    benchmark_passed = validation.get("benchmark_task_passed")
    if not isinstance(benchmark_passed, bool):
        raise ValueError("reconciliation requires a normalized verifier result")
    timeout_completion = validation.get("terminal_agent_timeout_completion") is True
    turn_budget_completion = validation.get("terminal_turn_budget_completion") is True
    quiet_output_completion = validation.get("terminal_quiet_output_completion") is True
    if not timeout_completion and not turn_budget_completion and not quiet_output_completion:
        raise ValueError("candidate is not an explicitly recognized terminal completion")
    if not isinstance(integration, dict) or integration.get("status") != "passed":
        raise ValueError("candidate integration validation did not pass")
    if upload.get("status") != "passed":
        raise ValueError("Phoenix upload did not pass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--validation-candidate", type=Path, required=True)
    parser.add_argument("--phoenix-upload", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    attempt = args.attempt_root.resolve()
    candidate_path = args.validation_candidate.resolve()
    upload_path = args.phoenix_upload.resolve()
    validator_path = args.validator.resolve()
    if not attempt.is_dir() or not attempt.name.isdigit():
        raise ValueError(f"invalid numbered attempt root: {attempt}")
    artifact_root = candidate_path.parent
    if attempt not in candidate_path.parents or attempt not in upload_path.parents:
        raise ValueError("candidate validation and upload must belong to the selected attempt")

    validation = read_json(candidate_path)
    upload = read_json(upload_path)
    require_reconcilable(validation, upload)
    validation["integration"]["phoenix_upload"] = upload

    canonical_validation = artifact_root / "validation.json"
    preserved_validation = artifact_root / "validation.pre-reconciliation.json"
    if not canonical_validation.is_file():
        raise ValueError(f"original validation is missing: {canonical_validation}")
    if preserved_validation.exists():
        raise ValueError(f"preserved validation already exists: {preserved_validation}")
    old_validation_sha256 = sha256_file(canonical_validation)
    shutil.copy2(canonical_validation, preserved_validation)
    write_json(canonical_validation, validation)

    job_root = artifact_root.parents[4]
    trial_results = sorted(job_root.glob("*/result.json"))
    if len(trial_results) != 1:
        raise ValueError(f"expected exactly one Harbor trial result under {job_root}")

    summary = {
        "schema_version": "harbor-hermes-switchyard.task-summary.v1",
        "job_name": job_root.name,
        "task_name": attempt.parents[1].name.split("-", 1)[-1],
        "artifacts": str(artifact_root),
        "validation": validation,
        "phoenix_upload": upload,
        "benchmark_completion": validation["benchmark"],
        "integration_validation": validation["integration"],
        "status": "passed",
    }
    write_json(attempt / "summary.json", summary)

    state_path = attempt.parents[1] / "task-state.json"
    old_state = read_json(state_path)
    task = old_state.get("task")
    if not isinstance(task, dict):
        raise ValueError(f"task state has no task metadata: {state_path}")
    write_json(
        state_path,
        {
            "schema_version": old_state.get("schema_version", "harbor-hermes-switchyard.phase2-task-state.v1"),
            "status": "passed",
            "task": task,
            "successful_attempt": attempt.name,
        },
    )

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "reconciled_at": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "validator repair recognized a verifier-backed terminal agent timeout completion"
            if validation.get("terminal_agent_timeout_completion") is True
            else (
                "validator repair recognized verifier-backed completion at Hermes's logical-call budget"
                if validation.get("terminal_turn_budget_completion") is True
                else "validator repair recognized a clean Hermes quiet-output completion hidden by reversed framing"
            )
        ),
        "attempt_root": str(attempt),
        "validator": str(validator_path),
        "validator_sha256": sha256_file(validator_path),
        "original_validation": str(preserved_validation),
        "original_validation_sha256": old_validation_sha256,
        "reconciled_validation": str(canonical_validation),
        "reconciled_validation_sha256": sha256_file(canonical_validation),
        "harbor_trial_result": str(trial_results[0]),
        "harbor_trial_result_sha256": sha256_file(trial_results[0]),
        "phoenix_upload": str(upload_path),
        "phoenix_upload_sha256": sha256_file(upload_path),
        "immutable_runtime_modified": False,
    }
    write_json(attempt / "nonpass-reconciliation.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
