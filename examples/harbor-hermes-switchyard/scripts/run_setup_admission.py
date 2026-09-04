#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provision task environments through Harbor's provider-free install lifecycle.

The Phase 2 coordinator and the standalone diagnostic share this implementation.
It stops before agent execution and the verifier, uses Harbor's real Docker and
agent setup paths, reuses successful task evidence by content hash, and never
loads provider authorization.
"""

from __future__ import annotations

import argparse
import email.utils
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harbor import __version__ as harbor_version
from harbor.models.task.task import Task

PLAN_SCHEMA = "harbor-hermes-switchyard.setup-admission-plan.v2"
RESULT_SCHEMA = "harbor-hermes-switchyard.setup-admission-task.v2"
SUMMARY_SCHEMA = "harbor-hermes-switchyard.setup-admission-summary.v2"
PAYLOAD_SCHEMA = "harbor-hermes-switchyard.hermetic-runtime.v1"
CLOCK_PREFLIGHT_SCHEMA = "harbor-hermes-switchyard.clock-preflight.v1"
CLOCK_REFERENCE_URL = "https://deb.debian.org/debian-security/dists/bookworm-security/InRelease"
CLOCK_MAX_OFFSET_SECONDS = 300
CLOCK_PROBE_IMAGE = "python:3.11-bullseye"
PLUGIN_COMPATIBILITY_SCHEMA = "harbor-hermes-switchyard.plugin-compatibility.v1"
SETUP_AGENT_PATH = Path(__file__).resolve().parents[1] / "agents" / "harbor_hermes_agent.py"
INFRASTRUCTURE_PATTERNS = (
    "apt-get update",
    "connection refused",
    "connection reset",
    "connection timed out",
    "context deadline exceeded",
    "docker build failed",
    "failed to resolve source metadata",
    "failed to authorize",
    "i/o timeout",
    "network is unreachable",
    "no space left on device",
    "registry-1.docker.io",
    "temporary failure in name resolution",
    "remote end closed connection",
    "tls handshake timeout",
    "too many requests",
    "unexpected eof",
)


def classify_setup_failure(trial_root: Path, diagnostic: str) -> str:
    lowered = diagnostic.lower()
    if any(pattern in lowered for pattern in INFRASTRUCTURE_PATTERNS):
        return "infrastructure"
    for path in sorted(trial_root.rglob("*.log")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        if any(pattern in text for pattern in INFRASTRUCTURE_PATTERNS):
            return "infrastructure"
    return "harness_or_integration"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def hermetic_content_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name == "payload.json":
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.stat().st_mode.to_bytes(4, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate_clock_preflight(*, host_epoch: float, docker_epoch: float, reference_epoch: float) -> dict[str, Any]:
    host_reference_offset = round(reference_epoch - host_epoch, 3)
    docker_host_offset = round(docker_epoch - host_epoch, 3)
    passed = (
        abs(host_reference_offset) <= CLOCK_MAX_OFFSET_SECONDS and abs(docker_host_offset) <= CLOCK_MAX_OFFSET_SECONDS
    )
    return {
        "schema_version": CLOCK_PREFLIGHT_SCHEMA,
        "status": "passed" if passed else "failed",
        "checked_at": datetime.now(UTC).isoformat(),
        "reference_url": CLOCK_REFERENCE_URL,
        "maximum_offset_seconds": CLOCK_MAX_OFFSET_SECONDS,
        "host_reference_offset_seconds": host_reference_offset,
        "docker_host_offset_seconds": docker_host_offset,
    }


def run_clock_preflight() -> dict[str, Any]:
    date_header: str | None = None
    for attempt in range(1, 5):
        request = urllib.request.Request(
            CLOCK_REFERENCE_URL,
            method="HEAD",
            headers={"User-Agent": "harbor-hermes-switchyard-setup-admission/2"},
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                date_header = response.headers.get("Date")
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 ** (attempt - 1))
    if not date_header:
        raise RuntimeError("clock reference response omitted its Date header")
    reference_datetime = email.utils.parsedate_to_datetime(date_header)
    docker_output = ""
    for attempt in range(1, 5):
        try:
            docker_output = subprocess.check_output(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--pull=missing",
                    CLOCK_PROBE_IMAGE,
                    "date",
                    "+%s",
                ],
                text=True,
            ).strip()
            break
        except subprocess.CalledProcessError:
            if attempt == 4:
                raise
            time.sleep(2 ** (attempt - 1))
    return evaluate_clock_preflight(
        host_epoch=time.time(),
        docker_epoch=float(docker_output),
        reference_epoch=reference_datetime.timestamp(),
    )


def plugin_compatibility_command(plan: dict[str, Any]) -> list[str]:
    inputs = plan["inputs"]
    platform = {
        "aarch64": "linux/arm64",
        "x86_64": "linux/amd64",
    }[inputs["relay_architecture"]]
    return [
        "docker",
        "run",
        "--rm",
        "--pull=missing",
        "--platform",
        platform,
        "--volume",
        f"{inputs['switchyard_bundle']}:/bundle:ro",
        CLOCK_PROBE_IMAGE,
        "python",
        "-c",
        (
            "import ctypes; "
            "library=ctypes.CDLL('/bundle/libswitchyard_nemo_relay_plugin.so'); "
            "getattr(library, 'nemo_relay_register_plugin')"
        ),
    ]


def run_plugin_compatibility_preflight(plan: dict[str, Any]) -> dict[str, Any]:
    for attempt in range(1, 5):
        process = subprocess.run(
            plugin_compatibility_command(plan),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        if process.returncode == 0 or attempt == 4:
            break
        time.sleep(2 ** (attempt - 1))
    inputs = plan["inputs"]
    return {
        "schema_version": PLUGIN_COMPATIBILITY_SCHEMA,
        "status": "passed" if process.returncode == 0 else "failed",
        "checked_at": datetime.now(UTC).isoformat(),
        "probe_image": CLOCK_PROBE_IMAGE,
        "relay_architecture": inputs["relay_architecture"],
        "switchyard_library_sha256": inputs["switchyard_library_sha256"],
        "registration_symbol": "nemo_relay_register_plugin",
        "exit_code": process.returncode,
    }


def load_payload(path: Path) -> dict[str, Any]:
    marker = path / "payload.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PAYLOAD_SCHEMA or payload.get("status") != "passed":
        raise ValueError(f"invalid hermetic runtime marker: {marker}")
    observed = hermetic_content_sha256(path)
    if payload.get("content_sha256") != observed:
        raise ValueError(
            "hermetic runtime content does not match its marker: "
            f"expected {payload.get('content_sha256')}, observed {observed}"
        )
    return payload


def discover_tasks(dataset: Path) -> list[Task]:
    tasks = [Task(path) for path in sorted(dataset.iterdir()) if Task.is_valid_dir(path)]
    names = [task.name for task in tasks]
    if len(tasks) != 89 or len(set(names)) != 89:
        raise ValueError(f"expected 89 unique tasks, found {len(tasks)} tasks and {len(set(names))} names")
    return sorted(tasks, key=lambda task: task.name)


def task_record(task: Task) -> dict[str, Any]:
    task_dir = task.task_dir
    environment = task.paths.environment_dir
    tests = task.paths.tests_dir
    record = {
        "name": task.name,
        "task_dir": str(task_dir),
        "task_sha256": sha256_tree(task_dir),
        "environment_sha256": sha256_tree(environment),
        "instruction_sha256": sha256_file(task.paths.instruction_path),
        "verifier_sha256": sha256_tree(tests),
        "docker_image": task.config.environment.docker_image,
        "cpus": task.config.environment.cpus,
        "memory_mb": task.config.environment.memory_mb,
        "build_timeout_sec": task.config.environment.build_timeout_sec,
    }
    return record


def build_plan(args: argparse.Namespace, tasks: list[Task], payload: dict[str, Any]) -> dict[str, Any]:
    runtime = args.runtime_root
    provenance_path = runtime / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    relay_wheels = sorted((runtime / "wheels").glob("nemo_relay-*.whl"))
    if len(relay_wheels) != 1:
        raise ValueError(f"expected one Relay wheel below {runtime / 'wheels'}")
    libraries = sorted((runtime / "switchyard-plugin").glob("*.so"))
    if len(libraries) != 1:
        raise ValueError("expected one Linux Switchyard library")
    records = [task_record(task) for task in tasks]
    inputs = {
        "dataset_root": str(args.dataset),
        "dataset_sha256": canonical_sha256(
            [{key: value for key, value in record.items() if key != "task_dir"} for record in records]
        ),
        "hermetic_runtime_root": str(args.hermetic_runtime),
        "hermetic_runtime_sha256": payload["content_sha256"],
        "hermes_commit": payload["hermes_commit"],
        "relay_architecture": payload["relay_architecture"],
        "relay_wheel": str(relay_wheels[0]),
        "relay_wheel_sha256": sha256_file(relay_wheels[0]),
        "relay_config": str(runtime / "plugins.toml"),
        "relay_config_sha256": sha256_file(runtime / "plugins.toml"),
        "switchyard_bundle": str(runtime / "switchyard-plugin"),
        "switchyard_library_sha256": sha256_file(libraries[0]),
        "runtime_provenance_sha256": sha256_file(provenance_path),
        "runtime_provenance_schema": provenance.get("schema_version"),
        "setup_agent_sha256": sha256_file(SETUP_AGENT_PATH),
        "harbor_version": harbor_version,
        "concurrency": args.concurrency,
        "batch_size": args.batch_size,
        "maximum_infrastructure_attempts": args.max_infra_attempts,
        "force_build": args.force_build,
        "preserve_containers": args.preserve_containers,
    }
    if inputs["relay_wheel_sha256"] != payload["relay_wheel_sha256"]:
        raise ValueError("prepared runtime Relay wheel does not match hermetic runtime")
    if inputs["relay_architecture"] != provenance["nemo_relay"]["architecture"]:
        raise ValueError("prepared runtime architecture does not match hermetic runtime")
    return {
        "schema_version": PLAN_SCHEMA,
        "status": "planned",
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": inputs,
        "tasks": records,
    }


def result_path(root: Path, task_name: str) -> Path:
    return root / "task-results" / f"{task_name}.json"


def completed_names(root: Path, plan: dict[str, Any]) -> set[str]:
    bindings = {
        task["name"]: canonical_sha256(
            {
                "task": task,
                "inputs": plan["inputs"],
            }
        )
        for task in plan["tasks"]
    }
    completed: set[str] = set()
    for name, binding in bindings.items():
        path = result_path(root, name)
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result.get("schema_version") == RESULT_SCHEMA
            and result.get("status") == "passed"
            and result.get("binding_sha256") == binding
        ):
            completed.add(name)
    return completed


def task_bindings(plan: dict[str, Any]) -> dict[str, str]:
    return {task["name"]: canonical_sha256({"task": task, "inputs": plan["inputs"]}) for task in plan["tasks"]}


def parse_job_results(root: Path, plan: dict[str, Any]) -> None:
    bindings = task_bindings(plan)
    known = set(bindings)
    for path in sorted((root / "jobs").glob("*/*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        task_name = result.get("task_name")
        if task_name not in known:
            continue
        exception = result.get("exception_info")
        environment = result.get("environment_setup")
        setup = result.get("agent_setup")
        execution = result.get("agent_execution")
        verifier = result.get("verifier")
        passed = (
            exception is None
            and isinstance(environment, dict)
            and environment.get("finished_at")
            and isinstance(setup, dict)
            and setup.get("finished_at")
            and execution is None
            and verifier is None
        )
        output = {
            "schema_version": RESULT_SCHEMA,
            "status": "passed" if passed else "failed",
            "task_name": task_name,
            "binding_sha256": bindings[task_name],
            "trial_result": str(path.relative_to(root)),
            "environment_setup": environment,
            "agent_setup": setup,
            "agent_execution_skipped": execution is None,
            "verifier_skipped": verifier is None,
            "exception_type": exception.get("exception_type") if isinstance(exception, dict) else None,
            "exception_message": exception.get("exception_message") if isinstance(exception, dict) else None,
        }
        diagnostic = " ".join(str(value or "") for value in (output["exception_type"], output["exception_message"]))
        output["failure_class"] = None if passed else classify_setup_failure(path.parent, diagnostic)
        destination = result_path(root, task_name)
        # Paths are scanned in job-name order, so always replacing the cached
        # result preserves the newest attempt. A later pass overwrites an older
        # failure, and a repeated failure retains its current diagnosis.
        write_json(destination, output)


def write_summary(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    passed = completed_names(root, plan)
    failed: list[str] = []
    for task in plan["tasks"]:
        path = result_path(root, task["name"])
        if path.is_file() and task["name"] not in passed:
            failed.append(task["name"])
    failed_classes: dict[str, str] = {}
    for name in failed:
        result = json.loads(result_path(root, name).read_text(encoding="utf-8"))
        failed_classes[name] = str(result.get("failure_class") or "harness_or_integration")
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "passed" if len(passed) == len(plan["tasks"]) else "partial",
        "plan_sha256": canonical_sha256(plan),
        "planned": len(plan["tasks"]),
        "passed": len(passed),
        "failed": len(failed),
        "pending": len(plan["tasks"]) - len(passed) - len(failed),
        "failed_tasks": sorted(failed),
        "infrastructure_failures": sorted(
            name for name, failure_class in failed_classes.items() if failure_class == "infrastructure"
        ),
        "integration_failures": sorted(
            name for name, failure_class in failed_classes.items() if failure_class != "infrastructure"
        ),
    }
    write_json(root / "summary.json", summary)
    return summary


def run_harbor(args: argparse.Namespace, plan: dict[str, Any], pending: list[str]) -> int:
    inputs = plan["inputs"]
    job_name = f"setup-admission-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    mounts = json.dumps(
        [
            {
                "type": "bind",
                "source": str(args.hermetic_runtime),
                "target": "/opt/hermes-runtime",
                "read_only": True,
                "bind": {"create_host_path": False},
            }
        ],
        separators=(",", ":"),
    )
    command = [
        str(args.harbor),
        "run",
        "--path",
        str(args.dataset),
        "--n-tasks",
        str(len(pending)),
        "--agent",
        "harbor_hermes_agent:HarborHermesAgent",
        "--model",
        "openai/setup-admission-stub",
        "--ak",
        f"commit={inputs['hermes_commit']}",
        "--ak",
        f"relay_config_path={inputs['relay_config']}",
        "--ak",
        f"switchyard_bundle_dir={inputs['switchyard_bundle']}",
        "--ak",
        f"relay_wheel_path={inputs['relay_wheel']}",
        "--ak",
        f"relay_wheel_sha256={inputs['relay_wheel_sha256']}",
        "--ak",
        f"relay_architecture={inputs['relay_architecture']}",
        "--ak",
        f"hermetic_runtime_dir={inputs['hermetic_runtime_root']}",
        "--ak",
        f"hermetic_runtime_sha256={inputs['hermetic_runtime_sha256']}",
        "--mounts",
        mounts,
        "--install-only",
        "--disable-verification",
        "--n-concurrent",
        str(args.concurrency),
        "--n-attempts",
        "1",
        "--agent-setup-timeout-multiplier",
        "6",
        "--environment-build-timeout-multiplier",
        "6",
        "--job-name",
        job_name,
        "--jobs-dir",
        str(args.output / "jobs"),
        "--yes",
    ]
    if args.force_build:
        command.append("--force-build")
    if args.preserve_containers:
        command.append("--no-delete")
    for task_name in pending:
        command.extend(["--include-task-name", task_name])
    env = os.environ.copy()
    agent_path = Path(__file__).resolve().parents[1] / "agents"
    env["PYTHONPATH"] = f"{agent_path}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    with (args.output / "admission.log").open("a", encoding="utf-8") as log:
        log.write(f"[{datetime.now(UTC).isoformat()}] starting {len(pending)} task(s)\n")
        log.flush()
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[{datetime.now(UTC).isoformat()}] Harbor exit={process.returncode}\n")
    return process.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--hermetic-runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--harbor", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=89)
    parser.add_argument("--max-infra-attempts", type=int, default=4)
    parser.add_argument("--backoff-seconds", type=float, default=15)
    parser.add_argument(
        "--force-build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exercise Docker's task-image build path even when an image already exists",
    )
    parser.add_argument(
        "--preserve-containers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="retain admission containers for debugging (off by default)",
    )
    parser.add_argument(
        "--task-name",
        action="append",
        default=[],
        help="Run only this task in the immutable all-89 plan; repeat as needed.",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1 or args.batch_size < 1 or args.max_infra_attempts < 1:
        parser.error("concurrency, batch size, and infrastructure attempts must be positive")
    if args.backoff_seconds < 0:
        parser.error("--backoff-seconds cannot be negative")
    for name in ("dataset", "runtime_root", "hermetic_runtime", "harbor"):
        value = getattr(args, name).expanduser().resolve(strict=True)
        setattr(args, name, value)
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    (args.output / "jobs").mkdir(mode=0o700, exist_ok=True)
    (args.output / "task-results").mkdir(mode=0o700, exist_ok=True)

    if harbor_version != "0.18.0":
        raise RuntimeError(f"setup admission requires Harbor 0.18.0, found {harbor_version}")
    payload = load_payload(args.hermetic_runtime)
    tasks = discover_tasks(args.dataset)
    known_names = {task.name for task in tasks}
    selected_names = set(args.task_name)
    unknown_names = selected_names - known_names
    if unknown_names:
        parser.error(f"unknown --task-name values: {', '.join(sorted(unknown_names))}")
    candidate = build_plan(args, tasks, payload)
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        comparable = dict(candidate)
        comparable["created_at"] = plan.get("created_at")
        if plan != comparable:
            raise ValueError("existing setup-admission plan does not match current inputs")
    else:
        plan = candidate
        write_json(plan_path, plan)
    if args.plan_only:
        print(json.dumps(write_summary(args.output, plan), indent=2, sort_keys=True))
        return 0

    # Import completed/failed results before an infrastructure gate can stop a
    # resumed invocation. This preserves every finished task even when the
    # machine is not currently healthy enough to launch more work.
    parse_job_results(args.output, plan)
    clock_preflight = run_clock_preflight()
    write_json(args.output / "clock-preflight.json", clock_preflight)
    if clock_preflight["status"] != "passed":
        write_summary(args.output, plan)
        raise RuntimeError(
            "wall-clock preflight failed; synchronize the host clock before building "
            f"task environments (evidence: {args.output / 'clock-preflight.json'})"
        )
    plugin_preflight = run_plugin_compatibility_preflight(plan)
    write_json(args.output / "plugin-compatibility.json", plugin_preflight)
    if plugin_preflight["status"] != "passed":
        write_summary(args.output, plan)
        raise RuntimeError(
            "Switchyard plugin compatibility preflight failed in the oldest "
            "supported task base (evidence: "
            f"{args.output / 'plugin-compatibility.json'})"
        )

    passed = completed_names(args.output, plan)
    pending = [
        task["name"]
        for task in plan["tasks"]
        if task["name"] not in passed and (not selected_names or task["name"] in selected_names)
    ]
    integration_blockers: set[str] = set()
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        remaining = list(batch)
        for attempt in range(1, args.max_infra_attempts + 1):
            if not remaining:
                break
            run_harbor(args, plan, remaining)
            # Harbor writes result files atomically enough for a completed process;
            # a resumed invocation re-scans every prior job before selecting work.
            parse_job_results(args.output, plan)
            passed = completed_names(args.output, plan)
            remaining = [name for name in remaining if name not in passed]
            blockers: set[str] = set()
            for name in remaining:
                path = result_path(args.output, name)
                if not path.is_file():
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("failure_class") == "harness_or_integration":
                    blockers.add(name)
            if blockers:
                integration_blockers.update(blockers)
                break
            if remaining and attempt < args.max_infra_attempts:
                delay = args.backoff_seconds * (2 ** (attempt - 1))
                with (args.output / "admission.log").open("a", encoding="utf-8") as log:
                    log.write(
                        f"[{datetime.now(UTC).isoformat()}] retrying {len(remaining)} "
                        f"infrastructure setup failure(s) after {delay:g}s\n"
                    )
                time.sleep(delay)
        if integration_blockers:
            break
    summary = write_summary(args.output, plan)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if selected_names:
        passed = completed_names(args.output, plan)
        return 0 if selected_names <= passed else 1
    if integration_blockers:
        return 20
    return 0 if summary["status"] == "passed" else 75


if __name__ == "__main__":
    raise SystemExit(main())
