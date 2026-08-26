# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a preserved whole-task retry for one exact provider failure."""

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

SCHEMA_VERSION = "harbor-hermes-switchyard.provider-retry.v1"
FAILURE_TEXT = "internal error: trusted fallback: provider returned HTTP 400"
EXPECTED_VALIDATION_ERRORS = {
    "invalid direct result status: 'failed'",
    "direct result has no normalized final response",
}


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


def is_trial_result(value: dict[str, Any]) -> bool:
    return "task_name" in value and "verifier_result" in value


def retry_evidence(
    state: dict[str, Any],
    direct_result: dict[str, Any],
    receipt: dict[str, Any],
    validation: dict[str, Any],
    harbor_result: dict[str, Any],
    diagnostic_text: str,
) -> bool:
    error = direct_result.get("error")
    cleanup = receipt.get("cleanup")
    integration = validation.get("integration")
    exception = harbor_result.get("exception_info")
    return (
        state.get("status") == "failed"
        and state.get("failure_class") == "harness_or_integration"
        and direct_result.get("status") == "failed"
        and direct_result.get("final_response") is None
        and isinstance(error, dict)
        and error.get("phase") == "agent"
        and error.get("type") == "NonZeroAgentExitCodeError"
        and isinstance(cleanup, dict)
        and cleanup.get("late_failure") is True
        and cleanup.get("plugin_host_closed") is True
        and cleanup.get("exporters_flushed") is True
        and cleanup.get("completion_marker_written") is True
        and validation.get("status") == "failed"
        and set(validation.get("errors") or []) == EXPECTED_VALIDATION_ERRORS
        and isinstance(validation.get("benchmark_task_passed"), bool)
        and isinstance(integration, dict)
        and integration.get("status") == "passed"
        and isinstance(exception, dict)
        and exception.get("exception_type") == "NonZeroAgentExitCodeError"
        and FAILURE_TEXT in str(exception.get("exception_message", ""))
        and FAILURE_TEXT in diagnostic_text
        and "API call failed after 3 retries:" in diagnostic_text
    )


def prepare_task(task_root: Path, max_retries: int) -> bool:
    state_path = task_root / "task-state.json"
    if not state_path.is_file():
        return False
    state = read_json(state_path)
    if state.get("status") != "failed" or state.get("failure_class") != "harness_or_integration":
        return False
    latest_attempt = state.get("latest_attempt")
    if not isinstance(latest_attempt, str) or not latest_attempt.isdigit():
        return False
    attempt = task_root / "attempts" / latest_attempt
    artifacts = list(attempt.glob("jobs/*/*/artifacts/logs/agent/direct-hermes"))
    jobs = list(attempt.glob("jobs/*"))
    if len(artifacts) != 1 or len(jobs) != 1:
        return False
    artifact = artifacts[0]
    trial_paths: list[Path] = []
    harbor_result: dict[str, Any] = {}
    for result_path in jobs[0].glob("**/result.json"):
        try:
            candidate = read_json(result_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if is_trial_result(candidate):
            trial_paths.append(result_path)
            harbor_result = candidate
    if len(trial_paths) != 1:
        return False
    try:
        direct_result = read_json(artifact / "direct-hermes-result.json")
        receipt = read_json(artifact / "direct-hermes-receipt.json")
        validation = read_json(artifact / "validation.json")
        diagnostic = (artifact / "diagnostics" / "hermes-tail.txt").read_text(
            encoding="utf-8", errors="replace"
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not retry_evidence(state, direct_result, receipt, validation, harbor_result, diagnostic):
        return False

    existing_receipts = sorted(task_root.glob("provider-retry-*.json"))
    if len(existing_receipts) >= max_retries:
        raise ValueError(f"provider retry limit reached for {task_root.name}: {len(existing_receipts)}/{max_retries}")
    retry_number = len(existing_receipts) + 1
    preserved_state = task_root / f"task-state.pre-provider-retry-{retry_number:03d}.json"
    retry_receipt = task_root / f"provider-retry-{retry_number:03d}.json"
    if preserved_state.exists() or retry_receipt.exists():
        raise ValueError(f"provider retry evidence already exists for {task_root.name} retry {retry_number}")
    shutil.copy2(state_path, preserved_state)
    write_json(
        retry_receipt,
        {
            "schema_version": SCHEMA_VERSION,
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "reason": "whole-task retry after an agent-terminal trusted-fallback HTTP 400",
            "task": state.get("task"),
            "failed_attempt": latest_attempt,
            "failure_signature": FAILURE_TEXT,
            "preserved_state": str(preserved_state),
            "preserved_state_sha256": sha256_file(preserved_state),
            "direct_result_sha256": sha256_file(artifact / "direct-hermes-result.json"),
            "validation_sha256": sha256_file(artifact / "validation.json"),
            "harbor_trial_result_sha256": sha256_file(trial_paths[0]),
            "immutable_runtime_modified": False,
        },
    )
    write_json(
        state_path,
        {
            "schema_version": state.get(
                "schema_version", "harbor-hermes-switchyard.phase2-task-state.v1"
            ),
            "status": "retry_requested",
            "task": state.get("task"),
            "latest_attempt": latest_attempt,
            "failure_class": "provider_request",
            "provider_retry_count": retry_number,
            "provider_retry_receipt": str(retry_receipt),
        },
    )
    print(f"prepared preserved provider retry: {task_root.name} after attempt {latest_attempt}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if args.max_retries <= 0 or not (root / "plan.json").is_file():
        raise ValueError("provider retry preparation requires a planned run root and a positive retry limit")
    prepared = sum(prepare_task(task_root, args.max_retries) for task_root in sorted((root / "tasks").glob("*")))
    print(json.dumps({"status": "passed", "prepared_retries": prepared}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
