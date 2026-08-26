#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one resumable, isolated Terminal-Bench cohort for Phase 2."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPT_ROOT = Path(__file__).resolve().parent
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))
import run_setup_admission as setup_admission  # noqa: E402
from relay_version import wheel_version  # noqa: E402

SCHEMA_VERSION = "harbor-hermes-switchyard.phase2-cohort.v1"
PLAN_SCHEMA_VERSION = "harbor-hermes-switchyard.phase2-plan.v1"
TASK_STATE_SCHEMA_VERSION = "harbor-hermes-switchyard.phase2-task-state.v1"
EXPECTED_HERMES_COMMIT = "a3d472f0e6bdc376df87b1436a461c4796db6747"
HERMETIC_RUNTIME_SCHEMA = "harbor-hermes-switchyard.hermetic-runtime.v1"
SETUP_REUSE_SCHEMA = "harbor-hermes-switchyard.setup-evidence-reuse.v1"
INFRASTRUCTURE_PATTERNS = (
    "apt-get update && apt-get install",
    "cannot connect to the docker daemon",
    "connection refused",
    "connection reset",
    "connection timed out",
    "command failed (exit 137)",
    "connecterror",
    "context deadline exceeded",
    "docker build failed",
    "docker is not running",
    "failed to resolve source metadata",
    "error getting dataset",
    "i/o timeout",
    "network is unreachable",
    "no space left on device",
    "phoenix upload",
    "provider has been unresponsive",
    "provider returned http 408",
    "response remained truncated after",
    "registry-1.docker.io",
    "temporary failure in name resolution",
    "tls handshake timeout",
    "too many requests",
    "verifiertimeouterror",
)

TASK_RUNTIME_OVERRIDES: dict[str, dict[str, int]] = {
    # The published 2 GiB limit is insufficient when the task fans out
    # ImageMagick/Tesseract workers. Two preserved runs produced kernel-confirmed
    # cgroup OOM kills, including the Hermes process itself.
    "extract-moves-from-video": {"memory_mb": 8192},
    # The official verifier's CPU-only PyTorch multiprocessing test can hang in
    # autograd when OpenMP/MKL create worker pools inside the one-CPU container.
    "torch-pipeline-parallelism": {"verifier_cpu_thread_limit": 1},
}


@dataclass(frozen=True)
class Task:
    index: int
    name: str
    memory_gb: int

    @property
    def directory_name(self) -> str:
        return f"{self.index:03d}-{self.name}"

    @property
    def runtime_override(self) -> dict[str, int]:
        return dict(TASK_RUNTIME_OVERRIDES.get(self.name, {}))

    @property
    def effective_memory_gb(self) -> int:
        memory_mb = self.runtime_override.get("memory_mb")
        return max(self.memory_gb, memory_mb // 1024 if memory_mb else self.memory_gb)

    def as_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "memory_gb": self.memory_gb,
            "effective_memory_gb": self.effective_memory_gb,
            "runtime_override": self.runtime_override,
        }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file_set(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def parse_memory_gb(environment: dict[str, object], task_name: str) -> int:
    memory_mb = environment.get("memory_mb")
    if memory_mb is not None:
        if isinstance(memory_mb, bool) or not isinstance(memory_mb, int) or memory_mb <= 0 or memory_mb % 1024:
            raise ValueError(f"unsupported memory value for {task_name}: {memory_mb} MB")
        return memory_mb // 1024
    text = str(environment.get("memory") or "2G").strip()
    match = re.fullmatch(r"([1-9][0-9]*)G", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"unsupported memory value for {task_name}: {text}")
    return int(match.group(1))


def discover_tasks(dataset_root: Path, sample_count: int, excluded: set[str], canary_task: str | None) -> list[Task]:
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {dataset_root}")
    discovered: list[tuple[str, int]] = []
    for task_root in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        if task_root.name in excluded:
            continue
        task_toml = task_root / "task.toml"
        if not task_toml.is_file():
            continue
        with task_toml.open("rb") as stream:
            config = tomllib.load(stream)
        memory = parse_memory_gb(config.get("environment") or {}, task_root.name)
        discovered.append((task_root.name, memory))
    selected = discovered[:sample_count]
    if len(selected) != sample_count:
        raise ValueError(f"dataset contains {len(selected)} selectable tasks; requested {sample_count}")
    if canary_task:
        canaries = [item for item in selected if item[0] == canary_task]
        if len(canaries) != 1:
            raise ValueError(f"canary task is not uniquely selectable: {canary_task}")
        ordered = canaries + [item for item in selected if item[0] != canary_task]
    else:
        ordered = selected
    return [Task(index, name, memory) for index, (name, memory) in enumerate(ordered, 1)]


def task_summary_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        summary = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    validation = summary.get("validation")
    benchmark = validation.get("benchmark", {}) if isinstance(validation, dict) else {}
    benchmark_status = benchmark.get("status", validation.get("status") if isinstance(validation, dict) else None)
    return summary.get("status") == "passed" and benchmark_status == "passed"


def successful_attempt(task_root: Path) -> Path | None:
    attempts = task_root / "attempts"
    if not attempts.is_dir():
        return None
    for attempt in sorted((path for path in attempts.iterdir() if path.is_dir()), reverse=True):
        if task_summary_passed(attempt / "summary.json"):
            return attempt
    return None


def classify_failure(log_text: str) -> str:
    lowered = log_text.lower()
    if any(pattern in lowered for pattern in INFRASTRUCTURE_PATTERNS):
        return "infrastructure"
    if re.search(r"http (429|5[0-9][0-9])\b", lowered):
        return "infrastructure"
    return "harness_or_integration"


def classify_attempt_failure(log_text: str, attempt: Path) -> str:
    """Classify a failed attempt using the wrapper and Harbor's nested logs."""

    if classify_failure(log_text) == "infrastructure":
        return "infrastructure"
    diagnostic_paths = [
        *attempt.rglob("*.log"),
        *attempt.rglob("result.json"),
        *attempt.rglob("hermes-tail.txt"),
    ]
    for path in sorted(diagnostic_paths):
        try:
            with path.open(encoding="utf-8", errors="replace") as stream:
                while chunk := stream.read(1024 * 1024):
                    if classify_failure(chunk) == "infrastructure":
                        return "infrastructure"
        except OSError:
            continue
    return "harness_or_integration"


def plugin_contract(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    components = {component.get("kind"): component for component in config.get("components", [])}
    caller = components.get("observability", {}).get("config", {}).get("atif", {}).get("model_name")
    plugin_table = config.get("plugins")
    plugins = plugin_table.get("dynamic") if isinstance(plugin_table, dict) else None
    if plugins is None:
        entries = (
            components.get("pricing", {})
            .get("config", {})
            .get("sources", [{}])[0]
            .get("catalog", {})
            .get("entries", [])
        )
        if not isinstance(entries, list) or len(entries) != 1:
            raise ValueError("direct baseline configuration must price exactly one model")
        model = entries[0].get("model_id")
        if not isinstance(model, str) or not model or caller != model:
            raise ValueError("direct baseline pricing and Hermes must use the same model")
        return {
            "mode": "direct",
            "required_models": [model],
            "catalog_models": [model],
            "strong_model": model,
            "weak_model": None,
            "judge_model": None,
            "hermes_caller_model": caller,
            "provider_base_urls": ["https://inference-api.nvidia.com/v1"],
            "sha256": sha256_file(path),
        }
    if len(plugins) != 1:
        raise ValueError("plugin configuration must define exactly one dynamic plugin")
    plugin_config = plugins[0].get("config", {})
    targets = plugin_config.get("targets", {})
    algorithm = plugin_config.get("algorithm", {})
    if not isinstance(algorithm, dict):
        raise ValueError("plugin algorithm configuration must be a table")
    algorithm_kind = algorithm.get("kind")
    expected_targets = {"strong", "weak"}
    if algorithm_kind != "stage_router" or algorithm.get("classifier") is not None:
        expected_targets.add("judge")
    if set(targets) != expected_targets:
        raise ValueError(f"plugin configuration targets must be {sorted(expected_targets)}")
    for target in targets.values():
        if target.get("header_env") != {"authorization": "SWITCHYARD_PROVIDER_AUTHORIZATION"}:
            raise ValueError("plugin authorization must reference SWITCHYARD_PROVIDER_AUTHORIZATION")
    models = [targets[name].get("model") for name in ("weak", "strong")]
    catalog_models = [*models, *([targets["judge"].get("model")] if "judge" in targets else [])]
    base_urls = sorted({targets[name].get("base_url") for name in targets})
    if any(not isinstance(value, str) or not value for value in catalog_models + base_urls):
        raise ValueError("plugin target models and base URLs must be non-empty")
    for base_url in base_urls:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("plugin target base URLs must be credential-free HTTP(S) URLs")
    if not isinstance(caller, str) or caller in models:
        raise ValueError("Hermes caller model must be distinct from plugin target models")
    contract: dict[str, Any] = {
        "algorithm": algorithm_kind,
        "classifier_target": None,
        "classifier_mode": None,
        "picker": None,
        "confidence_threshold": None,
        "recent_turn_window": algorithm.get("recent_turn_window"),
        "random_weights": None,
        "escalation": None,
    }
    if algorithm_kind == "random":
        if algorithm != {"kind": "random"}:
            raise ValueError("random routing must omit a fixed seed")
        weights = {name: targets[name].get("weight", 1) for name in ("strong", "weak", "judge")}
        if weights != {"strong": 1, "weak": 1, "judge": 0}:
            raise ValueError("random routing must split equally across strong and weak with judge disabled")
        contract["random_weights"] = {"strong": 1.0, "weak": 1.0}
    elif algorithm_kind == "llm_classifier":
        if (
            algorithm.get("classifier_target") != "judge"
            or algorithm.get("weak_target") != "weak"
            or algorithm.get("strong_target") != "strong"
        ):
            raise ValueError("LLM-classifier targets do not match the configured semantic tiers")
        contract["classifier_target"] = "judge"
        classifier_mode = algorithm.get("mode", "capability")
        if classifier_mode not in {"capability", "escalation"}:
            raise ValueError("LLM-classifier mode must be capability or escalation")
        contract["classifier_mode"] = classifier_mode
        if classifier_mode == "escalation":
            escalation = algorithm.get("escalation")
            if escalation != {
                "confirmations": 1,
                "recent_turn_window": 28,
                "window_message_chars": 500,
            }:
                raise ValueError("escalation settings do not match the trial contract")
            contract["escalation"] = dict(escalation)
    elif algorithm_kind == "stage_router":
        classifier = algorithm.get("classifier")
        if (
            algorithm.get("capable_target") != "strong"
            or algorithm.get("efficient_target") != "weak"
            or algorithm.get("picker") not in {"capable_first", "efficient_first"}
            or (classifier is not None and (not isinstance(classifier, dict) or classifier.get("target") != "judge"))
        ):
            raise ValueError("stage-router targets do not match the configured semantic tiers")
        confidence_threshold = algorithm.get("confidence_threshold")
        if not isinstance(confidence_threshold, (int, float)) or not 0 <= confidence_threshold <= 1:
            raise ValueError("stage-router confidence threshold must be in [0, 1]")
        contract.update(
            {
                "classifier_target": "judge" if classifier is not None else None,
                "picker": algorithm["picker"],
                "confidence_threshold": float(confidence_threshold),
            }
        )
    else:
        raise ValueError(f"unsupported plugin algorithm: {algorithm_kind!r}")
    return {
        "mode": "switchyard",
        "required_models": sorted(set(models)),
        "catalog_models": sorted(set(catalog_models)),
        "strong_model": targets["strong"]["model"],
        "weak_model": targets["weak"]["model"],
        "judge_model": targets["judge"]["model"] if "judge" in targets else None,
        "hermes_caller_model": caller,
        "provider_base_urls": base_urls,
        "sha256": sha256_file(path),
        **contract,
    }


def switchyard_library(bundle: Path) -> Path:
    libraries = sorted(bundle.glob("libswitchyard_nemo_relay_plugin.*"))
    if len(libraries) != 1:
        raise ValueError("Switchyard bundle must contain exactly one native library")
    return libraries[0]


def load_hermetic_runtime(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = setup_admission.load_payload(path)
    expected = {
        "schema_version": HERMETIC_RUNTIME_SCHEMA,
        "status": "passed",
        "hermes_commit": EXPECTED_HERMES_COMMIT,
        "relay_version": wheel_version(args.relay_wheel),
        "relay_wheel_sha256": sha256_file(args.relay_wheel),
        "relay_architecture": args.relay_architecture,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"hermetic runtime does not match cohort inputs: {mismatches}")
    digest = payload.get("content_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("hermetic runtime content digest is missing or invalid")
    for relative in ("bin/hermes", "bin/python", "bin/uv", "hermes-agent-src/venv"):
        if not (path / relative).exists():
            raise FileNotFoundError(path / relative)
    return payload


def ensure_hermetic_runtime(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    wheel_digest = sha256_file(args.relay_wheel)
    name = f"hermes-{EXPECTED_HERMES_COMMIT[:8]}-relay-070-{args.relay_architecture}-{wheel_digest[:12]}"
    output = args.bootstrap_root / name
    args.bootstrap_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = args.bootstrap_root / f".{name}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.is_dir():
            return output, load_hermetic_runtime(output, args)
        if output.exists():
            raise ValueError(f"hermetic runtime cache path is not a directory: {output}")
        subprocess.run(
            [
                str(args.python_bin),
                str(args.hermetic_runtime_builder),
                "--output",
                str(output),
                "--relay-wheel",
                str(args.relay_wheel),
                "--relay-architecture",
                args.relay_architecture,
                "--hermes-commit",
                EXPECTED_HERMES_COMMIT,
            ],
            check=True,
        )
        return output, load_hermetic_runtime(output, args)


def prepare_setup_runtime(args: argparse.Namespace) -> Path:
    destination = args.run_root / "setup-runtime"
    runtime = destination / "runtime"
    provenance = runtime / "provenance.json"
    if provenance.is_file():
        observed = read_json(provenance)
        if (
            observed.get("nemo_relay", {}).get("wheel_sha256") != sha256_file(args.relay_wheel)
            or observed.get("switchyard", {}).get("library_sha256")
            != sha256_file(switchyard_library(args.switchyard_bundle))
            or observed.get("relay_config_sha256") != sha256_file(runtime / "plugins.toml")
        ):
            raise ValueError("existing setup runtime does not match cohort inputs")
        return runtime
    if destination.exists():
        raise ValueError(f"incomplete setup runtime already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.preparing-{os.getpid()}")
    if temporary.exists():
        raise ValueError(f"stale setup runtime preparation exists: {temporary}")
    try:
        subprocess.run(
            [
                str(args.python_bin),
                str(args.runtime_preparer),
                "--run-root",
                str(temporary),
                "--switchyard-bundle",
                str(args.switchyard_bundle),
                "--relay-wheel",
                str(args.relay_wheel),
                "--relay-architecture",
                args.relay_architecture,
                "--plugin-config-template",
                str(args.plugin_config_template),
                "--openinference-endpoint",
                "http://127.0.0.1:9/v1/traces",
                "--phoenix-project",
                args.phoenix_project,
                "--eval-cohort",
                args.eval_cohort,
            ],
            check=True,
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return runtime


def bootstrap_preflight(args: argparse.Namespace) -> dict[str, Any]:
    clock = setup_admission.run_clock_preflight()
    if clock["status"] != "passed":
        raise RuntimeError("wall-clock preflight failed; synchronize the host and Docker clocks")
    compatibility_plan = {
        "inputs": {
            "relay_architecture": args.relay_architecture,
            "switchyard_bundle": str(args.switchyard_bundle),
            "switchyard_library_sha256": sha256_file(switchyard_library(args.switchyard_bundle)),
        }
    }
    plugin = setup_admission.run_plugin_compatibility_preflight(compatibility_plan)
    if plugin["status"] != "passed":
        raise RuntimeError("Switchyard plugin does not load in the oldest supported task base")
    evidence = {"status": "passed", "clock": clock, "plugin_compatibility": plugin}
    write_json(args.run_root / "bootstrap-preflight.json", evidence)
    return evidence


def validate_smoke_evidence(
    path: Path,
    expected_count: int,
    dataset_root: Path,
    concurrency: int,
    relay_architecture: str,
    relay_wheel: Path,
    switchyard_bundle: Path,
    plugin_config_template: Path,
) -> None:
    evidence = read_json(path)
    dataset_root = dataset_root.resolve()
    task_tomls = sorted(dataset_root.glob("*/task.toml"))
    expected_names = [task_toml.parent.name for task_toml in task_tomls]
    observed_tasks = evidence.get("tasks")
    observed_names = (
        [task.get("name") for task in observed_tasks]
        if isinstance(observed_tasks, list) and all(isinstance(task, dict) for task in observed_tasks)
        else []
    )
    records_valid = len(observed_names) == expected_count
    if records_valid:
        for record in observed_tasks:
            name = record.get("name")
            try:
                instruction = (dataset_root / record["instruction_path"]).resolve(strict=True)
                verifier = (dataset_root / record["verifier_path"]).resolve(strict=True)
                instruction.relative_to(dataset_root)
                verifier.relative_to(dataset_root)
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                records_valid = False
                break
            task_toml = dataset_root / str(name) / "task.toml"
            if (
                instruction.parent != task_toml.parent
                or verifier.parent.parent != task_toml.parent
                or record.get("task_toml_sha256") != sha256_file(task_toml)
                or record.get("instruction_sha256") != sha256_file(instruction)
                or record.get("test_sha256") != sha256_file(verifier)
            ):
                records_valid = False
                break
    if (
        evidence.get("schema_version") != "harbor-hermes-switchyard.phase2-smoke.v1"
        or evidence.get("status") != "passed"
        or evidence.get("task_count") != expected_count
        or len(task_tomls) != expected_count
        or observed_names != expected_names
        or not records_valid
        or evidence.get("dataset_task_definitions_sha256") != sha256_file_set(dataset_root, task_tomls)
        or evidence.get("registry_network_attempts") != 0
        or evidence.get("concurrency") != concurrency
        or evidence.get("relay_architecture") != relay_architecture
        or not isinstance(evidence.get("relay_runtime"), dict)
        or evidence["relay_runtime"].get("status") != "passed"
        or evidence["relay_runtime"].get("relay_wheel_sha256") != sha256_file(relay_wheel)
        or evidence["relay_runtime"].get("switchyard_library_sha256")
        != sha256_file(switchyard_library(switchyard_bundle))
        or evidence["relay_runtime"].get("plugin_config_template_sha256") != sha256_file(plugin_config_template)
    ):
        raise ValueError(f"Phase 2 all-task smoke evidence is not passed: {path}")


def validate_offline_evidence(
    path: Path,
    relay_architecture: str,
    relay_wheel: Path,
    switchyard_bundle: Path,
    plugin_config_template: Path,
) -> None:
    evidence = read_json(path)
    if (
        evidence.get("schema_version") != "harbor-hermes-switchyard.phase2-offline-admission.v1"
        or evidence.get("status") != "passed"
        or evidence.get("hermes_commit") != EXPECTED_HERMES_COMMIT
        or evidence.get("relay_architecture") != relay_architecture
        or evidence.get("relay_wheel_sha256") != sha256_file(relay_wheel)
        or evidence.get("switchyard_library_sha256") != sha256_file(switchyard_library(switchyard_bundle))
        or evidence.get("plugin_config_template_sha256") != sha256_file(plugin_config_template)
        or evidence.get("provider_requests", 0) <= 0
        or evidence.get("surviving_shutdown_threads") != []
    ):
        raise ValueError(f"Phase 2 offline admission evidence is not passed: {path}")


def validate_reused_setup_evidence(
    args: argparse.Namespace,
    tasks: list[Task],
) -> dict[str, Any] | None:
    """Validate a complete prior setup admission for build-free reuse."""

    root = args.reuse_setup_evidence
    if root is None:
        return None
    root = root.resolve()
    plan_path = root / "plan.json"
    summary_path = root / "summary.json"
    plan = read_json(plan_path)
    summary = read_json(summary_path)
    if plan.get("schema_version") != setup_admission.PLAN_SCHEMA:
        raise ValueError(f"reused setup evidence has an unsupported plan schema: {plan_path}")
    if summary.get("schema_version") != setup_admission.SUMMARY_SCHEMA:
        raise ValueError(f"reused setup evidence has an unsupported summary schema: {summary_path}")

    harbor_tasks = setup_admission.discover_tasks(args.dataset_root)
    current_records = [setup_admission.task_record(task) for task in harbor_tasks]
    expected_names = [task.name for task in tasks]
    observed_names = [str(record.get("name", "")).rsplit("/", 1)[-1] for record in current_records]
    if len(current_records) != len(tasks) or observed_names != expected_names or plan.get("tasks") != current_records:
        raise ValueError("reused setup evidence does not match the selected dataset tasks")

    inputs = plan.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("reused setup evidence plan has no inputs object")
    expected_inputs = {
        "dataset_root": str(args.dataset_root),
        "dataset_sha256": setup_admission.canonical_sha256(
            [{key: value for key, value in record.items() if key != "task_dir"} for record in current_records]
        ),
        "hermetic_runtime_sha256": args.hermetic_runtime_payload["content_sha256"],
        "hermes_commit": args.hermetic_runtime_payload["hermes_commit"],
        "relay_architecture": args.relay_architecture,
        "relay_wheel_sha256": sha256_file(args.relay_wheel),
        "switchyard_library_sha256": sha256_file(switchyard_library(args.switchyard_bundle)),
        "harbor_version": setup_admission.harbor_version,
        "concurrency": args.setup_concurrency,
        "batch_size": args.setup_batch_size,
        "maximum_infrastructure_attempts": args.setup_max_infra_attempts,
        "force_build": True,
        "preserve_containers": False,
    }
    mismatches = {
        key: {"expected": value, "actual": inputs.get(key)}
        for key, value in expected_inputs.items()
        if inputs.get(key) != value
    }
    setup_agent_sha256 = inputs.get("setup_agent_sha256")
    if not isinstance(setup_agent_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", setup_agent_sha256):
        mismatches["setup_agent_sha256"] = {"expected": "sha256", "actual": setup_agent_sha256}
    if mismatches:
        raise ValueError(f"reused setup evidence input mismatch: {mismatches}")

    source_plan_sha256 = setup_admission.canonical_sha256(plan)
    if (
        summary.get("status") != "passed"
        or summary.get("plan_sha256") != source_plan_sha256
        or summary.get("planned") != len(tasks)
        or summary.get("passed") != len(tasks)
        or summary.get("failed") != 0
        or summary.get("pending") != 0
    ):
        raise ValueError("reused setup evidence summary is not a complete bound pass")

    bindings = setup_admission.task_bindings(plan)
    results: dict[str, Path] = {}
    for path in sorted((root / "task-results").rglob("*.json")):
        result = read_json(path)
        name = result.get("task_name")
        if name not in bindings:
            raise ValueError(f"reused setup evidence contains an unknown task result: {path}")
        if name in results:
            raise ValueError(f"reused setup evidence contains duplicate results for {name}")
        if (
            result.get("schema_version") != setup_admission.RESULT_SCHEMA
            or result.get("status") != "passed"
            or result.get("binding_sha256") != bindings[name]
            or result.get("agent_execution_skipped") is not True
            or result.get("verifier_skipped") is not True
        ):
            raise ValueError(f"reused setup evidence contains an invalid task result: {path}")
        results[str(name)] = path
    if set(results) != set(bindings):
        missing = sorted(set(bindings).difference(results))
        raise ValueError(f"reused setup evidence is missing task results: {missing}")

    evidence_files = [plan_path, summary_path, *results.values()]
    return {
        "schema_version": SETUP_REUSE_SCHEMA,
        "mode": "reused",
        "status": "passed",
        "source_root": str(root),
        "source_plan_sha256": source_plan_sha256,
        "source_setup_agent_sha256": setup_agent_sha256,
        "evidence_sha256": sha256_file_set(root, evidence_files),
        "task_count": len(results),
    }


def probe_url(url: str, label: str, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status >= 500:
                    raise RuntimeError(f"{label} returned HTTP {response.status}")
            return
        except urllib.error.HTTPError as error:
            if error.code < 500:
                return
            failure: Exception = error
        except (OSError, urllib.error.URLError) as error:
            failure = error
        if attempt < attempts:
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"{label} is unreachable after {attempts} attempts: {failure}") from failure


def verify_provider_catalog(
    base_url: str, authorization: str, required_models: list[str], attempts: int = 3
) -> list[str]:
    catalog_url = f"{base_url.rstrip('/')}/models"
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(catalog_url, headers={"Authorization": authorization}, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError(f"provider model catalog returned HTTP {response.status}")
                payload = json.load(response)
            break
        except urllib.error.HTTPError as error:
            if error.code < 500 and error.code != 429:
                raise RuntimeError(f"provider model catalog returned HTTP {error.code}") from error
            failure: Exception = error
        except (OSError, ValueError, urllib.error.URLError) as error:
            failure = error
        if attempt < attempts:
            time.sleep(2 ** (attempt - 1))
    else:
        raise RuntimeError(
            f"provider model catalog is unavailable or invalid after {attempts} attempts: {failure}"
        ) from failure
    records = payload.get("data") if isinstance(payload, dict) else None
    available = {
        record.get("id") for record in records or [] if isinstance(record, dict) and isinstance(record.get("id"), str)
    }
    missing = sorted(set(required_models) - available)
    if missing:
        raise RuntimeError(f"provider model catalog is missing configured model(s): {', '.join(missing)}")
    return sorted(required_models)


def normalize_architecture(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}
    if normalized not in aliases:
        raise RuntimeError(f"unsupported Docker architecture: {value}")
    return aliases[normalized]


def capacity_requirement_gb(args: argparse.Namespace, tasks: list[Task]) -> int:
    parallel = args.concurrency * args.parallel_max_memory_gb
    largest = max(task.effective_memory_gb for task in tasks)
    return max(parallel, largest) + args.docker_memory_reserve_gb


def shared_preflight(args: argparse.Namespace, tasks: list[Task]) -> dict[str, Any]:
    free_bytes = shutil.disk_usage(args.run_root.parent).free
    if free_bytes < args.minimum_free_gb * 1024**3:
        raise RuntimeError(f"fewer than {args.minimum_free_gb} GiB remain on the run volume")
    docker = json.loads(
        subprocess.run(
            ["docker", "info", "--format", "{{json .}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    docker_cpus = int(docker["NCPU"])
    docker_memory_bytes = int(docker["MemTotal"])
    docker_memory_gb = docker_memory_bytes // 1024**3
    docker_architecture = normalize_architecture(str(docker["Architecture"]))
    required_memory_gb = capacity_requirement_gb(args, tasks)
    if docker_architecture != args.relay_architecture:
        raise RuntimeError(
            f"Docker architecture {docker_architecture} does not match Relay architecture {args.relay_architecture}"
        )
    if args.concurrency > docker_cpus:
        raise RuntimeError(f"concurrency {args.concurrency} exceeds Docker CPU count {docker_cpus}")
    if required_memory_gb > docker_memory_gb:
        raise RuntimeError(f"Phase 2 requires {required_memory_gb} GiB but Docker exposes {docker_memory_gb} GiB")
    version = subprocess.run(
        [str(args.harbor_bin), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if version != "0.20.0":
        raise RuntimeError(f"Harbor 0.20.0 is required; {args.harbor_bin} reports {version}")
    python_version = subprocess.run(
        [
            str(args.python_bin),
            "-c",
            'import importlib.metadata; print(importlib.metadata.version("harbor"))',
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if python_version != "0.20.0":
        raise RuntimeError(f"Harbor Python 0.20.0 is required; {args.python_bin} provides {python_version}")
    probe_url(args.phoenix_url, "Phoenix")
    provider_authorization = os.environ.get("SWITCHYARD_PROVIDER_AUTHORIZATION")
    if not provider_authorization:
        raise RuntimeError("SWITCHYARD_PROVIDER_AUTHORIZATION is unset")
    verified_models: set[str] = set()
    for provider_url in args.plugin_contract["provider_base_urls"]:
        verified_models.update(
            verify_provider_catalog(
                provider_url,
                provider_authorization,
                args.plugin_contract["catalog_models"],
            )
        )
    if not args.switchyard_bundle.is_dir():
        raise RuntimeError(f"Switchyard bundle is missing: {args.switchyard_bundle}")
    if not args.relay_wheel.is_file():
        raise RuntimeError(f"Relay wheel is missing: {args.relay_wheel}")
    return {
        "harbor_version": version,
        "minimum_free_gb": args.minimum_free_gb,
        "disk_free_gb": free_bytes // 1024**3,
        "concurrency": args.concurrency,
        "parallel_task_memory_gb": args.parallel_max_memory_gb,
        "largest_task_memory_gb": max(task.effective_memory_gb for task in tasks),
        "docker_memory_reserve_gb": args.docker_memory_reserve_gb,
        "required_docker_memory_gb": required_memory_gb,
        "docker_memory_gb": docker_memory_gb,
        "docker_cpus": docker_cpus,
        "docker_architecture": docker_architecture,
        "relay_architecture": args.relay_architecture,
        "provider_endpoints": args.plugin_contract["provider_base_urls"],
        "provider_catalog_models_verified": sorted(verified_models),
        "phoenix_endpoint": args.phoenix_url,
        "phoenix_reachable": True,
        "provider_reachable": True,
        "docker_healthy": True,
    }


def make_plan(args: argparse.Namespace, tasks: list[Task]) -> dict[str, Any]:
    manifest = args.switchyard_bundle / "relay-plugin.toml"
    library_candidates = sorted(args.switchyard_bundle.glob("libswitchyard_nemo_relay_plugin.*"))
    if not manifest.is_file() or len(library_candidates) != 1:
        raise ValueError("Switchyard bundle must contain one manifest and one native library")
    example_root = args.task_runner.parent
    runtime_sources = [
        args.task_runner,
        example_root / "run_phase2_cohort.sh",
        example_root / "supervise_phase2_cohort.sh",
    ]
    for relative in ("agents", "config", "scripts"):
        runtime_sources.extend(
            path
            for path in (example_root / relative).rglob("*")
            if path.is_file() and path.suffix in {".py", ".sh", ".toml", ".yaml", ".in"}
        )
    task_definitions = [args.dataset_root / task.name / "task.toml" for task in tasks]
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "dataset": args.dataset,
        "dataset_root": str(args.dataset_root.resolve()),
        "sample_count": args.sample_count,
        "canary_task": args.canary_task,
        "concurrency": args.concurrency,
        "setup_concurrency": args.setup_concurrency,
        "setup_batch_size": args.setup_batch_size,
        "setup_max_infra_attempts": args.setup_max_infra_attempts,
        "parallel_max_memory_gb": args.parallel_max_memory_gb,
        "docker_memory_reserve_gb": args.docker_memory_reserve_gb,
        "minimum_free_gb": args.minimum_free_gb,
        "setup_admission": getattr(args, "reused_setup_evidence_payload", None)
        or {
            "schema_version": SETUP_REUSE_SCHEMA,
            "mode": "fresh",
        },
        "phoenix_project": args.phoenix_project,
        "evaluation_cohort": args.eval_cohort,
        "timeout_multipliers": {"agent": 3, "agent_setup": 6, "environment_build": 6},
        "required_models": args.plugin_contract["required_models"],
        "require_cache_hit": args.require_cache_hit,
        "routing": {
            "mode": args.plugin_contract["mode"],
            "algorithm": args.plugin_contract.get("algorithm"),
            "strong_model": args.plugin_contract["strong_model"],
            "weak_model": args.plugin_contract["weak_model"],
            "judge_model": args.plugin_contract["judge_model"],
            "classifier_target": args.plugin_contract.get("classifier_target"),
            "classifier_mode": args.plugin_contract.get("classifier_mode"),
            "picker": args.plugin_contract.get("picker"),
            "confidence_threshold": args.plugin_contract.get("confidence_threshold"),
            "recent_turn_window": args.plugin_contract.get("recent_turn_window"),
            "random_weights": args.plugin_contract.get("random_weights"),
            "escalation": args.plugin_contract.get("escalation"),
            "hermes_caller_model": args.plugin_contract["hermes_caller_model"],
            "provider_base_urls": args.plugin_contract["provider_base_urls"],
        },
        "inputs": {
            "runner_sha256": sha256_file(args.task_runner),
            "runtime_sources_sha256": sha256_file_set(example_root, runtime_sources),
            "dataset_task_definitions_sha256": sha256_file_set(args.dataset_root, task_definitions),
            "phase2_smoke_evidence_sha256": sha256_file(args.smoke_evidence),
            "phase2_offline_evidence_sha256": sha256_file(args.offline_evidence),
            "plugin_config_template_sha256": sha256_file(args.plugin_config_template),
            "admission_plugin_config_template_sha256": sha256_file(
                args.admission_plugin_config_template
            ),
            "relay_wheel_sha256": sha256_file(args.relay_wheel),
            "switchyard_manifest_sha256": sha256_file(manifest),
            "switchyard_library_sha256": sha256_file(library_candidates[0]),
            "hermetic_runtime_sha256": args.hermetic_runtime_payload["content_sha256"],
            "setup_runtime_provenance_sha256": sha256_file(args.setup_runtime / "provenance.json"),
        },
        "tasks": [task.as_json() for task in tasks],
    }


def load_or_create_plan(path: Path, plan: dict[str, Any]) -> None:
    if path.is_file():
        existing = read_json(path)
        if existing != plan:
            raise ValueError(f"existing Phase 2 plan does not match requested configuration: {path}")
        return
    write_json(path, plan)


def task_record(task: Task, attempt: Path | None) -> dict[str, Any]:
    record: dict[str, Any] = task.as_json()
    record["status"] = "pending"
    record["attempt_count"] = 0
    task_root = attempt.parents[1] if attempt else None
    if task_root:
        record["attempt_count"] = len([path for path in (task_root / "attempts").iterdir() if path.is_dir()])
    if not attempt:
        return record
    summary = read_json(attempt / "summary.json")
    validation = summary["validation"]
    upload = summary["phoenix_upload"]
    integration = validation.get("integration", {"status": validation.get("status"), "errors": []})
    record.update(
        {
            "status": "completed",
            "successful_attempt": attempt.name,
            "attempt_root": str(attempt),
            "benchmark_task_passed": validation.get("benchmark_task_passed"),
            "benchmark_completion": validation.get("benchmark", {"status": validation.get("status")}),
            "integration_validation": integration,
            "direct_result_status": validation.get("direct_result_status"),
            "switchyard_decision_count": validation.get("switchyard_decision_count", 0),
            "routed_models": validation.get("routed_models", []),
            "routed_targets": validation.get("routed_targets", []),
            "cache_read_tokens": validation.get("cache_read_tokens", 0),
            "cache_write_tokens": validation.get("cache_write_tokens", 0),
            "secret_findings": validation.get("secret_findings", []),
            "uploaded_spans": upload.get("uploaded_spans", 0),
            "phoenix_upload": upload.get("status"),
        }
    )
    return record


def aggregate_summary(args: argparse.Namespace, tasks: list[Task]) -> dict[str, Any]:
    records = []
    for task in tasks:
        task_root = args.run_root / "tasks" / task.directory_name
        records.append(task_record(task, successful_attempt(task_root)))
    complete = all(record["status"] == "completed" for record in records)
    cache_read_tokens = sum(int(record.get("cache_read_tokens") or 0) for record in records)
    observed_models = sorted(
        {model for record in records for model in record.get("routed_models", []) if isinstance(model, str)}
    )
    missing_models = sorted(set(args.required_model).difference(observed_models))
    completed_records = [record for record in records if record["status"] == "completed"]
    secrets_clean = all(not record.get("secret_findings") for record in completed_records)
    integration_failures = [
        {
            "task": record["name"],
            "errors": record.get("integration_validation", {}).get("errors", []),
        }
        for record in completed_records
        if record.get("integration_validation", {}).get("status") != "passed"
    ]
    gates = {
        "task_outputs": {"passed": complete, "completed": sum(r["status"] == "completed" for r in records)},
        "integration_validation": {
            "passed": not integration_failures,
            "failed_task_count": len(integration_failures),
            "failures": integration_failures,
        },
        "cache_hit": {
            "required": args.require_cache_hit,
            "passed": not args.require_cache_hit or cache_read_tokens > 0,
            "cache_read_tokens": cache_read_tokens,
        },
        "route_diversity": {
            "required_models": sorted(args.required_model),
            "observed_models": observed_models,
            "missing_models": missing_models,
            "passed": not missing_models,
        },
        "secret_scan": {"passed": secrets_clean},
    }
    passed = complete and all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "partial",
        "dataset": args.dataset,
        "phoenix_project": args.phoenix_project,
        "evaluation_cohort": args.eval_cohort,
        "planned_tasks": len(records),
        "completed_tasks": sum(record["status"] == "completed" for record in records),
        "benchmark_pass_count": sum(record.get("benchmark_task_passed") is True for record in records),
        "benchmark_nonpass_count": sum(record.get("benchmark_task_passed") is False for record in records),
        "uploaded_spans": sum(int(record.get("uploaded_spans") or 0) for record in records),
        "cohort_gates": gates,
        "tasks": records,
    }


def write_report(root: Path, summary: dict[str, Any]) -> None:
    gates = summary["cohort_gates"]
    lines = [
        "# Harbor + Hermes + Switchyard Phase 2 cohort",
        "",
        f"- Status: `{summary['status']}`",
        f"- Completed: {summary['completed_tasks']}/{summary['planned_tasks']}",
        f"- Benchmark pass/non-pass: {summary['benchmark_pass_count']}/{summary['benchmark_nonpass_count']}",
        f"- Uploaded spans: {summary['uploaded_spans']}",
        f"- Cache-read tokens: {gates['cache_hit']['cache_read_tokens']}",
        f"- Observed provider models: {', '.join(gates['route_diversity']['observed_models']) or 'none'}",
        "",
        "| # | Task | Memory (task/effective) | Evidence | Benchmark | Attempts | Spans |",
        "|---:|---|---:|---|---|---:|---:|",
    ]
    for task in summary["tasks"]:
        benchmark = task.get("benchmark_task_passed")
        benchmark_text = "pass" if benchmark is True else "non-pass" if benchmark is False else "pending"
        memory_text = f"{task['memory_gb']}G"
        if task["effective_memory_gb"] != task["memory_gb"]:
            memory_text += f"/{task['effective_memory_gb']}G"
        lines.append(
            f"| {task['index']:03d} | `{task['name']}` | {memory_text} | {task['status']} | "
            f"{benchmark_text} | {task['attempt_count']} | {task.get('uploaded_spans', 0)} |"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class CohortRunner:
    def __init__(self, args: argparse.Namespace, tasks: list[Task]):
        self.args = args
        self.tasks = tasks
        self.stop_scheduling = asyncio.Event()
        self.summary_lock = asyncio.Lock()

    async def refresh_summary(self) -> None:
        async with self.summary_lock:
            summary = aggregate_summary(self.args, self.tasks)
            write_json(self.args.run_root / "summary.json", summary)
            write_report(self.args.run_root, summary)

    def provision_environments(self) -> bool:
        reused = getattr(self.args, "reused_setup_evidence_payload", None)
        if reused is not None:
            write_json(
                self.args.run_root / "setup-state.json",
                {
                    "schema_version": "harbor-hermes-switchyard.phase2-setup-state.v1",
                    "status": "passed",
                    "failure_class": None,
                    "exit_code": 0,
                    "summary": {
                        "schema_version": setup_admission.SUMMARY_SCHEMA,
                        "status": "passed",
                        "planned": len(self.tasks),
                        "passed": len(self.tasks),
                        "failed": 0,
                        "pending": 0,
                    },
                    "hermetic_runtime_sha256": self.args.hermetic_runtime_payload["content_sha256"],
                    "force_build": True,
                    "setup_concurrency": self.args.setup_concurrency,
                    "reused_setup_evidence": reused,
                },
            )
            return True
        output = self.args.run_root / "setup-admission"
        command = [
            str(self.args.python_bin),
            str(self.args.setup_admission_runner),
            "--dataset",
            str(self.args.dataset_root),
            "--runtime-root",
            str(self.args.setup_runtime),
            "--hermetic-runtime",
            str(self.args.hermetic_runtime),
            "--output",
            str(output),
            "--harbor",
            str(self.args.harbor_bin),
            "--concurrency",
            str(self.args.setup_concurrency),
            "--batch-size",
            str(self.args.setup_batch_size),
            "--max-infra-attempts",
            str(self.args.setup_max_infra_attempts),
            "--backoff-seconds",
            str(self.args.backoff_seconds),
            "--force-build",
            "--no-preserve-containers",
        ]
        log_path = self.args.run_root / "setup-admission-coordinator.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        summary_path = output / "summary.json"
        summary = read_json(summary_path) if summary_path.is_file() else {}
        passed = (
            process.returncode == 0
            and summary.get("status") == "passed"
            and summary.get("planned") == len(self.tasks)
            and summary.get("passed") == len(self.tasks)
        )
        write_json(
            self.args.run_root / "setup-state.json",
            {
                "schema_version": "harbor-hermes-switchyard.phase2-setup-state.v1",
                "status": "passed" if passed else "failed",
                "failure_class": (
                    None
                    if passed
                    else "harness_or_integration"
                    if process.returncode == 20 or summary.get("integration_failures")
                    else "infrastructure"
                ),
                "exit_code": process.returncode,
                "summary": summary,
                "hermetic_runtime_sha256": self.args.hermetic_runtime_payload["content_sha256"],
                "force_build": True,
                "setup_concurrency": self.args.setup_concurrency,
            },
        )
        return passed

    async def run_attempt(self, task: Task, attempt_number: int) -> tuple[int, Path, str]:
        task_root = self.args.run_root / "tasks" / task.directory_name
        attempts_root = task_root / "attempts"
        attempts_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        attempt = attempts_root / f"{attempt_number:03d}"
        log_path = task_root / f"attempt-{attempt_number:03d}.log"
        if attempt.exists():
            raise RuntimeError(f"attempt root already exists: {attempt}")
        env = os.environ.copy()
        env.update(
            {
                "TASK_NAME": task.name,
                "EVAL_PHASE": "phase2",
                "TBENCH_DATASET_PATH": str(self.args.dataset_root),
                "PHOENIX_PROJECT": self.args.phoenix_project,
                "EVAL_COHORT": self.args.eval_cohort,
                "PHOENIX_BASE_URL": self.args.phoenix_url,
                "SWITCHYARD_BUNDLE": str(self.args.switchyard_bundle),
                "RELAY_WHEEL": str(self.args.relay_wheel),
                "RELAY_ARCHITECTURE": self.args.relay_architecture,
                "PLUGIN_CONFIG_TEMPLATE": str(self.args.plugin_config_template),
                "HARBOR_BIN": str(self.args.harbor_bin),
                "EVAL_PYTHON": str(self.args.python_bin),
                "AGENT_TIMEOUT_MULTIPLIER": "3",
                "AGENT_SETUP_TIMEOUT_MULTIPLIER": "6",
                "ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER": "6",
                "HERMETIC_RUNTIME_DIR": str(self.args.hermetic_runtime),
                "HERMETIC_RUNTIME_SHA256": self.args.hermetic_runtime_payload["content_sha256"],
                # Harbor assigns trial-specific local image tags. Rebuild from the
                # pinned task Dockerfile so an incompatible published prebuilt image
                # cannot replace the architecture-validated setup-admission image.
                # The setup lane has already populated Docker's layer cache.
                "HARBOR_FORCE_BUILD": "true",
                # Clear host values so overrides are always derived from the
                # immutable per-task policy below.
                "TASK_MEMORY_OVERRIDE_MB": "",
                "VERIFIER_CPU_THREAD_LIMIT": "",
            }
        )
        override = task.runtime_override
        if memory_mb := override.get("memory_mb"):
            env["TASK_MEMORY_OVERRIDE_MB"] = str(memory_mb)
        if thread_limit := override.get("verifier_cpu_thread_limit"):
            env["VERIFIER_CPU_THREAD_LIMIT"] = str(thread_limit)
        with log_path.open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                str(self.args.task_runner), str(attempt), env=env, stdout=log, stderr=asyncio.subprocess.STDOUT
            )
            status = await process.wait()
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        return status, attempt, log_text

    async def run_task(self, task: Task) -> bool:
        task_root = self.args.run_root / "tasks" / task.directory_name
        passed = successful_attempt(task_root)
        if passed:
            print(f"[phase2] already complete: {task.directory_name}", flush=True)
            return True
        state_path = task_root / "task-state.json"
        if state_path.is_file():
            state = read_json(state_path)
            if state.get("status") == "failed" and state.get("failure_class") == "harness_or_integration":
                print(f"[phase2] preserved integration blocker: {task.directory_name}", flush=True)
                self.stop_scheduling.set()
                return False
        attempts_root = task_root / "attempts"
        existing_attempts = len(list(attempts_root.glob("[0-9][0-9][0-9]")))
        attempt_number = existing_attempts + 1
        infrastructure_failures = 0
        retry_preflight_failures = 0
        needs_retry_preflight = False
        while infrastructure_failures < self.args.max_infra_attempts:
            if self.stop_scheduling.is_set():
                return False
            if needs_retry_preflight:
                try:
                    preflight = shared_preflight(self.args, self.tasks)
                except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
                    retry_preflight_failures += 1
                    write_json(
                        state_path,
                        {
                            "schema_version": TASK_STATE_SCHEMA_VERSION,
                            "status": "waiting_for_infrastructure",
                            "task": task.as_json(),
                            "failure_class": "infrastructure",
                            "preflight_error": str(error),
                            "infrastructure_failure_count": infrastructure_failures,
                            "retry_preflight_failure_count": retry_preflight_failures,
                        },
                    )
                    print(
                        f"[phase2] retry preflight blocked {task.directory_name} "
                        f"({retry_preflight_failures}/{self.args.max_infra_attempts}): {error}",
                        flush=True,
                    )
                    await self.refresh_summary()
                    if retry_preflight_failures >= self.args.max_infra_attempts:
                        self.stop_scheduling.set()
                        return False
                    await asyncio.sleep(self.args.backoff_seconds * (2 ** (retry_preflight_failures - 1)))
                    continue
                write_json(self.args.run_root / "preflight.json", {"status": "passed", **preflight})
                needs_retry_preflight = False
                retry_preflight_failures = 0
            print(f"[phase2] starting {task.directory_name} attempt {attempt_number:03d}", flush=True)
            status, attempt, log_text = await self.run_attempt(task, attempt_number)
            if status == 0 and task_summary_passed(attempt / "summary.json"):
                write_json(
                    state_path,
                    {
                        "schema_version": TASK_STATE_SCHEMA_VERSION,
                        "status": "passed",
                        "task": task.as_json(),
                        "successful_attempt": attempt.name,
                    },
                )
                await self.refresh_summary()
                print(f"[phase2] completed {task.directory_name} attempt {attempt_number:03d}", flush=True)
                return True
            failure_class = classify_attempt_failure(log_text, attempt)
            write_json(
                state_path,
                {
                    "schema_version": TASK_STATE_SCHEMA_VERSION,
                    "status": "failed",
                    "task": task.as_json(),
                    "latest_attempt": attempt.name,
                    "failure_class": failure_class,
                    "exit_code": status,
                },
            )
            await self.refresh_summary()
            print(
                f"[phase2] failed {task.directory_name} attempt {attempt_number:03d}: {failure_class}",
                flush=True,
            )
            if failure_class != "infrastructure":
                self.stop_scheduling.set()
                return False
            infrastructure_failures += 1
            attempt_number += 1
            needs_retry_preflight = True
            if infrastructure_failures < self.args.max_infra_attempts:
                await asyncio.sleep(self.args.backoff_seconds * (2 ** (infrastructure_failures - 1)))
        self.stop_scheduling.set()
        return False

    async def run_parallel_lane(self, tasks: list[Task]) -> bool:
        semaphore = asyncio.Semaphore(self.args.concurrency)

        async def guarded(task: Task) -> bool:
            async with semaphore:
                if self.stop_scheduling.is_set():
                    return False
                return await self.run_task(task)

        results = await asyncio.gather(*(guarded(task) for task in tasks))
        return all(results)

    async def run(self) -> bool:
        preflight = shared_preflight(self.args, self.tasks)
        write_json(self.args.run_root / "preflight.json", {"status": "passed", **preflight})
        await self.refresh_summary()
        if not self.provision_environments():
            return False
        remaining = self.tasks
        if self.args.canary_task:
            first = self.tasks[0]
            if not await self.run_task(first):
                return False
            remaining = self.tasks[1:]
        parallel = [task for task in remaining if task.effective_memory_gb <= self.args.parallel_max_memory_gb]
        serial = [task for task in remaining if task.effective_memory_gb > self.args.parallel_max_memory_gb]
        if not await self.run_parallel_lane(parallel):
            return False
        for task in serial:
            if not await self.run_task(task):
                return False
        await self.refresh_summary()
        return aggregate_summary(self.args, self.tasks)["status"] == "passed"


def parse_args() -> argparse.Namespace:
    example_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", default="terminal-bench@2.0")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=89)
    parser.add_argument("--exclude-task", action="append", default=[])
    parser.add_argument(
        "--canary-task",
        default="",
        help="task to run alone before the cohort; pass an empty value to disable the canary",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--setup-concurrency", type=int, default=2)
    parser.add_argument("--setup-batch-size", type=int, default=89)
    parser.add_argument("--setup-max-infra-attempts", type=int, default=4)
    parser.add_argument("--parallel-max-memory-gb", type=int, default=2)
    parser.add_argument("--docker-memory-reserve-gb", type=int, default=4)
    parser.add_argument("--max-infra-attempts", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=30)
    parser.add_argument("--minimum-free-gb", type=int, default=100)
    parser.add_argument("--smoke-evidence", type=Path, required=True)
    parser.add_argument("--offline-evidence", type=Path, required=True)
    parser.add_argument(
        "--reuse-setup-evidence",
        type=Path,
        help="complete prior setup-admission directory to verify and reuse without rebuilding",
    )
    parser.add_argument("--plugin-config-template", type=Path, required=True)
    parser.add_argument(
        "--admission-plugin-config-template",
        type=Path,
        help="routed template that produced smoke/offline evidence for a direct control",
    )
    parser.add_argument("--task-runner", type=Path, default=example_root / "run_terminal_bench.sh")
    parser.add_argument(
        "--setup-admission-runner",
        type=Path,
        default=example_root / "scripts" / "run_setup_admission.py",
    )
    parser.add_argument(
        "--hermetic-runtime-builder",
        type=Path,
        default=example_root / "scripts" / "build_hermetic_runtime.py",
    )
    parser.add_argument(
        "--runtime-preparer",
        type=Path,
        default=example_root / "scripts" / "prepare_runtime.py",
    )
    parser.add_argument(
        "--bootstrap-root",
        type=Path,
        help="shared content-addressed bootstrap cache; generated automatically when absent",
    )
    parser.add_argument("--harbor-bin", type=Path, default=example_root / ".venv" / "bin" / "harbor")
    parser.add_argument("--python-bin", type=Path, default=example_root / ".venv" / "bin" / "python")
    parser.add_argument("--phoenix-url", required=True)
    parser.add_argument("--phoenix-project", required=True)
    parser.add_argument("--eval-cohort", required=True)
    parser.add_argument("--switchyard-bundle", type=Path, required=True)
    parser.add_argument("--relay-wheel", type=Path, required=True)
    parser.add_argument("--relay-architecture", choices=("x86_64", "aarch64"), default="x86_64")
    parser.add_argument("--require-cache-hit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    for name in (
        "sample_count",
        "concurrency",
        "setup_concurrency",
        "setup_batch_size",
        "setup_max_infra_attempts",
        "parallel_max_memory_gb",
        "docker_memory_reserve_gb",
        "max_infra_attempts",
        "minimum_free_gb",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not args.run_root.is_absolute():
        parser.error("--run-root must be absolute")
    args.bootstrap_root = (
        (args.bootstrap_root or args.run_root.parent / "harbor-hermes-switchyard-bootstrap").expanduser().resolve()
    )
    if args.reuse_setup_evidence is not None:
        args.reuse_setup_evidence = args.reuse_setup_evidence.expanduser().resolve()
    if args.plan_only and args.preflight_only:
        parser.error("--plan-only and --preflight-only are mutually exclusive")
    return args


def main() -> int:
    args = parse_args()
    args.plugin_contract = plugin_contract(args.plugin_config_template)
    args.required_model = args.plugin_contract["required_models"]
    admission_plugin_template = args.admission_plugin_config_template or args.plugin_config_template
    if args.plugin_contract["mode"] == "direct":
        # Dataset/setup admission was already proven with the routed template.
        # The direct baseline config is validated independently and does not
        # load that admitted plugin during provider execution.
        admission_plugin_template = args.admission_plugin_config_template or (
            args.task_runner.parent / "config" / "plugins.toml.in"
        )
    elif args.admission_plugin_config_template is not None:
        raise ValueError("a routed cohort must use its own plugin template for admission evidence")
    args.admission_plugin_config_template = admission_plugin_template.resolve(strict=True)
    validate_smoke_evidence(
        args.smoke_evidence,
        args.sample_count,
        args.dataset_root,
        args.concurrency,
        args.relay_architecture,
        args.relay_wheel,
        args.switchyard_bundle,
        admission_plugin_template,
    )
    validate_offline_evidence(
        args.offline_evidence,
        args.relay_architecture,
        args.relay_wheel,
        args.switchyard_bundle,
        admission_plugin_template,
    )
    args.canary_task = args.canary_task or None
    tasks = discover_tasks(args.dataset_root, args.sample_count, set(args.exclude_task), args.canary_task)
    args.run_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = args.run_root / ".phase2.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another Phase 2 supervisor owns {args.run_root}") from None
        bootstrap_preflight(args)
        args.hermetic_runtime, args.hermetic_runtime_payload = ensure_hermetic_runtime(args)
        args.setup_runtime = prepare_setup_runtime(args)
        args.reused_setup_evidence_payload = validate_reused_setup_evidence(args, tasks)
        plan = make_plan(args, tasks)
        load_or_create_plan(args.run_root / "plan.json", plan)
        if args.plan_only:
            summary = aggregate_summary(args, tasks)
            write_json(args.run_root / "summary.json", summary)
            write_report(args.run_root, summary)
            print(json.dumps(plan, indent=2))
            return 0
        if args.preflight_only:
            preflight = shared_preflight(args, tasks)
            write_json(args.run_root / "preflight.json", {"status": "passed", **preflight})
            summary = aggregate_summary(args, tasks)
            write_json(args.run_root / "summary.json", summary)
            write_report(args.run_root, summary)
            print(json.dumps({"status": "passed", "preflight": preflight}, indent=2))
            return 0
        passed = asyncio.run(CohortRunner(args, tasks).run())
        summary = aggregate_summary(args, tasks)
        print(json.dumps(summary, indent=2))
        if passed:
            return 0
        if (
            summary["completed_tasks"] == summary["planned_tasks"]
            and not summary["cohort_gates"]["integration_validation"]["passed"]
        ):
            return 20
        states = [read_json(path) for path in (args.run_root / "tasks").glob("*/task-state.json") if path.is_file()]
        setup_state = args.run_root / "setup-state.json"
        if setup_state.is_file():
            states.append(read_json(setup_state))
        if any(state.get("failure_class") == "harness_or_integration" for state in states):
            return 20
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
