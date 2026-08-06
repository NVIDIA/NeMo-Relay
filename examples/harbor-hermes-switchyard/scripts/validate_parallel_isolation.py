# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate that concurrent Phase 1 tasks used isolated runtime state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "harbor-hermes-switchyard.parallel-isolation.v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def validate(run_roots: list[Path]) -> dict[str, Any]:
    if len(run_roots) < 2:
        raise ValueError("parallel isolation requires at least two task roots")
    records: list[dict[str, Any]] = []
    for root in run_roots:
        resolved = root.resolve(strict=True)
        summary = read_json(resolved / "summary.json")
        if summary.get("status") != "passed":
            raise ValueError(f"task summary did not pass: {resolved}")
        artifacts = Path(summary.get("artifacts", "")).resolve(strict=True)
        receipt = read_json(artifacts / "direct-hermes-receipt.json")
        provenance = read_json(resolved / "runtime" / "provenance.json")
        cleanup = receipt.get("cleanup") or {}
        if not all(cleanup.get(key) is True for key in ("plugin_host_closed", "exporters_flushed")):
            raise ValueError(f"task did not close plugin/exporter lifecycle: {resolved}")
        records.append(
            {
                "task": summary.get("task_name"),
                "run_root": str(resolved),
                "artifact_root": str(artifacts),
                "job_name": summary.get("job_name"),
                "session_handle": receipt.get("session_handle"),
                "relay_config_sha256": provenance.get("relay_config_sha256"),
                "phoenix_project": provenance.get("phoenix_project"),
                "evaluation_cohort": provenance.get("eval_cohort"),
                "relay_wheel_sha256": provenance.get("nemo_relay", {}).get("wheel_sha256"),
                "switchyard_library_sha256": provenance.get("switchyard", {}).get("library_sha256"),
            }
        )
    distinct_fields = (
        "run_root",
        "artifact_root",
        "job_name",
        "session_handle",
        "relay_config_sha256",
        "phoenix_project",
        "evaluation_cohort",
    )
    for field in distinct_fields:
        values = [record.get(field) for record in records]
        if any(not value for value in values) or len(set(values)) != len(values):
            raise ValueError(f"parallel tasks did not have distinct {field} values")
    shared_fields = ("relay_wheel_sha256", "switchyard_library_sha256")
    for field in shared_fields:
        values = [record.get(field) for record in records]
        if any(not value for value in values) or len(set(values)) != 1:
            raise ValueError(f"parallel tasks did not use one immutable {field}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "task_count": len(records),
        "distinct_fields": list(distinct_fields),
        "shared_input_fields": list(shared_fields),
        "tasks": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.run_root)
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
