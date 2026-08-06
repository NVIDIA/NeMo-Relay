# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate one Harbor/Hermes/Switchyard Phase 1 task evidence set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "harbor-hermes-switchyard.validation.v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def is_trial_result(value: dict[str, Any]) -> bool:
    return "task_name" in value and "verifier_result" in value


def read_benchmark_passed(value: dict[str, Any]) -> bool | None:
    reward = value.get("reward")
    if isinstance(reward, dict):
        candidate = reward.get("task_passed")
        if isinstance(candidate, bool):
            return candidate
    verifier = value.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    if isinstance(rewards, dict):
        candidate = rewards.get("task_passed")
        if isinstance(candidate, bool):
            return candidate
        candidate = rewards.get("reward")
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return candidate > 0
    return None


def contained_files(root: Path) -> list[Path]:
    resolved_root = root.resolve(strict=True)
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"artifact symlink is forbidden: {path}")
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        if resolved_root not in resolved.parents:
            raise ValueError(f"artifact escaped its root: {path}")
        files.append(path)
    return files


def scan_files(root: Path) -> list[Path]:
    """Return regular files below a scan root without following symlinks."""
    if not root.is_dir():
        return []
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file():
            files.append(path)
    return files


def scan_secrets(files: Iterable[Path], values: list[bytes]) -> list[str]:
    findings: list[str] = []
    for path in files:
        with path.open("rb") as stream:
            overlap = b""
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                haystack = overlap + block
                for index, secret in enumerate(values):
                    if secret and secret in haystack:
                        findings.append(f"{path.name}:secret[{index}]")
                max_width = max((len(secret) for secret in values), default=1)
                overlap = haystack[-max_width:]
    return sorted(set(findings))


def read_atof(path: Path) -> tuple[int, list[str], list[str], list[str]]:
    count = 0
    marks: list[str] = []
    models: list[str] = []
    targets: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"ATOF line {line_number} is not an object")
            count += 1
            name = payload.get("name")
            if isinstance(name, str) and name.startswith("switchyard.routing."):
                marks.append(name)
                for container in (payload.get("data"), payload.get("metadata")):
                    if isinstance(container, dict):
                        for key in ("model", "selected_model", "target_model"):
                            value = container.get(key)
                            if isinstance(value, str) and value:
                                models.append(value)
                        value = container.get("selected_target")
                        if isinstance(value, str) and value:
                            targets.append(value)
    return count, sorted(set(marks)), sorted(set(models)), sorted(set(targets))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--openinference", type=Path, required=True)
    parser.add_argument("--harbor-job-dir", type=Path)
    parser.add_argument("--scan-root", type=Path, action="append", default=[])
    parser.add_argument("--expect-late-failure", action="store_true")
    parser.add_argument("--secret-env", action="append", default=[])
    parser.add_argument("--secret-file", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    root = args.artifacts.resolve()
    try:
        files = contained_files(root)
    except Exception as error:
        files = []
        errors.append(str(error))

    required = {
        "result": root / "direct-hermes-result.json",
        "receipt": root / "direct-hermes-receipt.json",
        "completion": root / "completion.json",
        "atof": root / "relay" / "trajectory.atof.jsonl",
    }
    for name, path in required.items():
        if not path.is_file():
            errors.append(f"missing {name}: {path}")
    atif_files = sorted((root / "relay" / "atif").glob("trajectory-*.atif.json"))
    if not atif_files:
        errors.append("missing ATIF trajectory")
    if not args.openinference.is_file() or args.openinference.stat().st_size == 0:
        errors.append("missing OpenInference OTLP artifact")

    result: dict[str, Any] = {}
    receipt: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    if required["result"].is_file():
        result = read_json(required["result"])
        if result.get("status") not in {"completed", "preserved_completed_response"}:
            errors.append(f"invalid direct result status: {result.get('status')!r}")
        if not isinstance(result.get("final_response"), str) or not result["final_response"]:
            errors.append("direct result has no normalized final response")
        if args.expect_late_failure:
            if result.get("status") != "preserved_completed_response":
                errors.append("expected a preserved completed response after late failure")
            if result.get("error", {}).get("type") != "InjectedPostResponseFailure":
                errors.append("deterministic post-response failure was not recorded")
    if required["receipt"].is_file():
        receipt = read_json(required["receipt"])
    if args.provenance.is_file():
        provenance = read_json(args.provenance)
    else:
        errors.append("missing runtime provenance")

    if receipt:
        dependencies = receipt.get("dependencies", {})
        relay = dependencies.get("nemo_relay", {})
        hermes = dependencies.get("hermes", {})
        switchyard = dependencies.get("switchyard", {})
        if relay.get("version") != "0.7.0":
            errors.append("receipt did not record nemo-relay==0.7.0")
        if relay.get("wheel_sha256") != provenance.get("nemo_relay", {}).get("wheel_sha256"):
            errors.append("Relay wheel digest does not match runtime provenance")
        if hermes.get("commit") != provenance.get("hermes", {}).get("commit"):
            errors.append("Hermes commit does not match runtime provenance")
        if switchyard.get("commit") != provenance.get("switchyard", {}).get("commit"):
            errors.append("Switchyard commit does not match runtime provenance")
        if receipt.get("dynamic_plugin_ids") != ["nvidia.switchyard"]:
            errors.append("receipt did not record only nvidia.switchyard")
        if receipt.get("activation_mode") != "relay_standard_dynamic":
            errors.append("receipt did not record standard dynamic activation")
        cleanup = receipt.get("cleanup", {})
        if not cleanup.get("plugin_host_closed") or not cleanup.get("exporters_flushed"):
            errors.append("receipt did not prove plugin close and exporter flush")

    event_count = 0
    routing_marks: list[str] = []
    routed_models: list[str] = []
    routed_targets: list[str] = []
    if required["atof"].is_file():
        event_count, routing_marks, routed_models, routed_targets = read_atof(required["atof"])
        if event_count == 0:
            errors.append("ATOF artifact is empty")
        if not routing_marks:
            errors.append("ATOF artifact has no Switchyard routing evidence")
        if not routed_targets:
            errors.append("ATOF artifact has no selected Switchyard target")
        unexpected_targets = sorted(set(routed_targets) - {"strong", "weak"})
        if unexpected_targets:
            errors.append(f"ATOF artifact selected unexpected targets: {unexpected_targets}")

    caller_model = provenance.get("routing", {}).get("hermes_caller_model")
    if caller_model and caller_model in routed_models:
        errors.append("Hermes caller stub appeared as a routed provider model")

    secret_values: list[bytes] = []
    for name in args.secret_env:
        value = os.environ.get(name)
        if value:
            secret_values.append(value.encode())
            if value.startswith("Bearer ") and value[7:]:
                secret_values.append(value[7:].encode())
    for path in args.secret_file:
        for line in path.read_bytes().splitlines():
            value = line.split(b"=", 1)[-1].strip()
            if value:
                secret_values.append(value)
    files_to_scan = list(files)
    if args.openinference.is_file():
        files_to_scan.append(args.openinference)
    for scan_root in args.scan_root:
        files_to_scan.extend(scan_files(scan_root.resolve()))
    findings = scan_secrets(sorted(set(files_to_scan)), secret_values)
    if findings:
        errors.append(f"secret scan found {len(findings)} persisted value(s)")

    harbor_results: list[Path] = []
    benchmark_passed: bool | None = None
    if args.harbor_job_dir:
        harbor_results = [
            path for path in sorted(args.harbor_job_dir.glob("**/result.json")) if is_trial_result(read_json(path))
        ]
        if len(harbor_results) != 1:
            errors.append(f"expected one Harbor trial result, found {len(harbor_results)}")
        elif harbor_results:
            harbor_result = read_json(harbor_results[0])
            benchmark_passed = read_benchmark_passed(harbor_result)

    validation = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "direct_result_status": result.get("status"),
        "harbor_trial_count": len(harbor_results) if args.harbor_job_dir else None,
        "benchmark_task_passed": benchmark_passed,
        "atof_event_count": event_count,
        "atif_trajectory_count": len(atif_files),
        "switchyard_routing_marks": routing_marks,
        "routed_models": routed_models,
        "routed_targets": routed_targets,
        "secret_values_scanned": len(secret_values),
        "secret_findings": findings,
    }
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
