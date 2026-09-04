#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise all Phase 2 dataset and runtime wiring without provider work."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import socket
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import patch

_SCRIPT_ROOT = Path(__file__).resolve().parent
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))
from relay_version import wheel_version

from harbor.job import Job
from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.task.task import Task as HarborTask
from harbor.models.trial.config import AgentConfig, EnvironmentConfig

SCHEMA_VERSION = "harbor-hermes-switchyard.phase2-smoke.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def parse_memory_gb(task: HarborTask) -> int:
    memory_mb = task.config.environment.memory_mb or 2048
    if memory_mb % 1024:
        raise ValueError(f"unsupported memory for {task.name}: {memory_mb} MB")
    return memory_mb // 1024


async def validate_local_dataset(
    dataset_root: Path,
    expected_count: int,
    jobs_dir: Path,
    concurrency: int,
    relay_architecture: str,
    authorization_file: Path,
) -> tuple[list[dict[str, Any]], int]:
    network_attempts = 0

    def deny_network(*_args: Any, **_kwargs: Any) -> None:
        nonlocal network_attempts
        network_attempts += 1
        raise AssertionError("local dataset smoke attempted network access")

    dataset = DatasetConfig(path=dataset_root)
    with (
        patch("socket.create_connection", side_effect=deny_network),
        patch.object(socket.socket, "connect", side_effect=deny_network),
        patch(
            "harbor.registry.client.factory.RegistryClientFactory.create",
            side_effect=deny_network,
        ),
    ):
        configs = await dataset.get_task_configs()
        if len(configs) != expected_count:
            raise ValueError(f"Harbor resolved {len(configs)} tasks; expected {expected_count}")
        task_records: list[dict[str, Any]] = []
        expected_names = sorted(path.parent.name for path in dataset_root.glob("*/task.toml"))
        resolved_names = sorted(config.get_local_path().name for config in configs)
        if resolved_names != expected_names:
            raise ValueError("Harbor local dataset resolution did not preserve the exported task set")
        for name in resolved_names:
            selected = await DatasetConfig(
                path=dataset_root,
                task_names=[name],
                n_tasks=1,
            ).get_task_configs()
            if len(selected) != 1 or selected[0].get_local_path().name != name:
                raise ValueError(f"Harbor did not uniquely select local task {name}")
            task = HarborTask(selected[0].get_local_path())
            if not task.instruction.strip():
                raise ValueError(f"Harbor loaded an empty instruction for {name}")
            test_path = task.paths.discovered_test_path
            if not test_path.is_file():
                raise ValueError(f"Harbor did not discover a verifier test for {name}")
            task_records.append(
                {
                    "name": name,
                    "memory_gb": parse_memory_gb(task),
                    "instruction_path": task.paths.instruction_path.relative_to(dataset_root).as_posix(),
                    "verifier_path": test_path.relative_to(dataset_root).as_posix(),
                    "task_toml_sha256": sha256_file(task.paths.config_path),
                    "instruction_sha256": sha256_file(task.paths.instruction_path),
                    "test_sha256": sha256_file(test_path),
                    "has_steps": task.has_steps,
                }
            )

        job = await Job.create(
            JobConfig(
                job_name="phase2-all-task-smoke",
                jobs_dir=jobs_dir,
                n_attempts=1,
                n_concurrent_trials=concurrency,
                agent_timeout_multiplier=3,
                agent_setup_timeout_multiplier=6,
                environment_build_timeout_multiplier=6,
                agents=[
                    AgentConfig(
                        import_path="harbor_hermes_agent:HarborHermesAgent",
                        model_name="openai/ollama-route-stub",
                        kwargs={
                            "repository_url": "https://github.com/bbednarski9/hermes-agent.git",
                            "repository_ref": "feat/relay-native-plugin-init",
                            "commit": "a3d472f0e6bdc376df87b1436a461c4796db6747",
                            "relay_config_path": "/smoke/runtime/plugins.toml",
                            "switchyard_bundle_dir": "/smoke/runtime/switchyard-plugin",
                            "relay_wheel_path": "/smoke/runtime/nemo-relay.whl",
                            "relay_architecture": relay_architecture,
                        },
                        env={
                            "OPENAI_API_KEY": "${OPENAI_API_KEY}",
                            "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
                            "OPENROUTER_API_KEY": "relay-intercepted",
                            "OPENROUTER_BASE_URL": "http://127.0.0.1:9/v1",
                        },
                    )
                ],
                environment=EnvironmentConfig(
                    mounts=[
                        {
                            "type": "bind",
                            "source": str(authorization_file),
                            "target": "/run/secrets/switchyard-provider-authorization",
                            "read_only": True,
                            "bind": {"create_host_path": False},
                        }
                    ]
                ),
                datasets=[dataset],
                artifacts=["/logs/agent/direct-hermes"],
            )
        )
        if len(job) != expected_count:
            raise ValueError(f"Harbor constructed {len(job)} trials; expected {expected_count}")
    return task_records, network_attempts


def validate_cli_local_projection(
    harbor_bin: Path, dataset_root: Path, task_name: str, authorization_file: Path
) -> None:
    mounts = json.dumps(
        [
            {
                "type": "bind",
                "source": str(authorization_file),
                "target": "/run/secrets/switchyard-provider-authorization",
                "read_only": True,
                "bind": {"create_host_path": False},
            }
        ],
        separators=(",", ":"),
    )
    result = subprocess.run(
        [
            str(harbor_bin),
            "run",
            "--path",
            str(dataset_root),
            "--include-task-name",
            task_name,
            "--n-tasks",
            "1",
            "--agent",
            "harbor_hermes_agent:HarborHermesAgent",
            "--model",
            "openai/ollama-route-stub",
            "--mounts",
            mounts,
            "--print-config",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    config = json.loads(result.stdout)
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError("Harbor CLI did not render one local dataset")
    dataset = datasets[0]
    if (
        Path(dataset.get("path", "")).resolve() != dataset_root
        or dataset.get("task_names") != [task_name]
        or dataset.get("n_tasks") != 1
        or dataset.get("name") is not None
    ):
        raise ValueError("Harbor CLI local dataset projection is incorrect")
    if config.get("environment", {}).get("mounts") != json.loads(mounts):
        raise ValueError("Harbor CLI did not preserve the protected authorization mount")


def validate_relay_runtime(
    example_root: Path,
    temporary_root: Path,
    switchyard_bundle: Path,
    relay_wheel: Path,
    relay_architecture: str,
    plugin_config_template: Path,
) -> dict[str, Any]:
    run_root = temporary_root / "runtime-smoke"
    subprocess.run(
        [
            sys.executable,
            str(example_root / "scripts" / "prepare_runtime.py"),
            "--run-root",
            str(run_root),
            "--switchyard-bundle",
            str(switchyard_bundle),
            "--relay-wheel",
            str(relay_wheel),
            "--relay-architecture",
            relay_architecture,
            "--plugin-config-template",
            str(plugin_config_template),
            "--openinference-endpoint",
            "http://127.0.0.1:4318/v1/traces",
            "--phoenix-project",
            "phase2-smoke",
            "--eval-cohort",
            "phase2-smoke",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    compatibility_path = run_root / "artifacts" / "harbor-hermes-compatibility.json"
    subprocess.run(
        [
            sys.executable,
            str(example_root / "scripts" / "verify_harbor_hermes_compat.py"),
            "--bridge",
            str(example_root / "agents" / "harbor_hermes_agent.py"),
            "--relay-config",
            str(run_root / "runtime" / "plugins.toml"),
            "--output",
            str(compatibility_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    provenance = json.loads((run_root / "runtime" / "provenance.json").read_text())
    compatibility = json.loads(compatibility_path.read_text())
    with (run_root / "runtime" / "plugins.toml").open("rb") as stream:
        relay_config = tomllib.load(stream)
    with (switchyard_bundle / "relay-plugin.toml").open("rb") as stream:
        switchyard_manifest = tomllib.load(stream)
    dynamic_plugins = relay_config.get("plugins", {}).get("dynamic", [])
    components = {component["kind"]: component for component in relay_config.get("components", [])}
    plugin_id = switchyard_manifest.get("plugin", {}).get("id")
    if (
        provenance.get("nemo_relay", {}).get("version") != wheel_version(relay_wheel)
        or provenance.get("nemo_relay", {}).get("wheel_sha256") != sha256_file(relay_wheel)
        or provenance.get("switchyard", {}).get("library_sha256") is None
        or compatibility.get("status") != "passed"
        or len(dynamic_plugins) != 1
        or dynamic_plugins[0].get("manifest") != "/opt/relay-plugins/nvidia.switchyard/relay-plugin.toml"
        or plugin_id != "nvidia.switchyard"
        or components.get("observability", {}).get("config", {}).get("version") != 3
    ):
        raise ValueError("Relay/Hermes/Switchyard smoke wiring did not pass")
    return {
        "status": "passed",
        "relay_version": provenance["nemo_relay"]["version"],
        "relay_wheel_sha256": provenance["nemo_relay"]["wheel_sha256"],
        "switchyard_library_sha256": provenance["switchyard"]["library_sha256"],
        "plugin_config_template_sha256": provenance["plugin_config_template_sha256"],
        "relay_config_sha256": provenance["relay_config_sha256"],
        "relay_architecture": provenance["nemo_relay"]["architecture"],
        "routing": provenance["routing"],
        "dynamic_plugin_id": plugin_id,
        "observability_version": 3,
        "compatibility_status": compatibility["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=89)
    parser.add_argument("--harbor-bin", type=Path, required=True)
    parser.add_argument("--switchyard-bundle", type=Path, required=True)
    parser.add_argument("--relay-wheel", type=Path, required=True)
    parser.add_argument("--relay-architecture", choices=("x86_64", "aarch64"), required=True)
    parser.add_argument("--plugin-config-template", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    example_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if importlib.metadata.version("harbor") != "0.18.0":
        raise RuntimeError("Phase 2 smoke requires Harbor 0.18.0")
    if (
        subprocess.run([str(args.harbor_bin), "--version"], check=True, capture_output=True, text=True).stdout.strip()
        != "0.18.0"
    ):
        raise RuntimeError("Phase 2 smoke Harbor CLI is not 0.18.0")

    with tempfile.TemporaryDirectory(prefix="harbor-phase2-smoke-") as directory:
        temporary_root = Path(directory)
        authorization_file = temporary_root / "switchyard-provider-authorization"
        authorization_file.write_text("Bearer offline-placeholder", encoding="utf-8")
        authorization_file.chmod(0o600)
        task_records, network_attempts = asyncio.run(
            validate_local_dataset(
                dataset_root,
                args.expected_count,
                temporary_root / "jobs",
                args.concurrency,
                args.relay_architecture,
                authorization_file,
            )
        )
        validate_cli_local_projection(args.harbor_bin, dataset_root, task_records[0]["name"], authorization_file)
        relay_runtime = validate_relay_runtime(
            example_root,
            temporary_root,
            args.switchyard_bundle.expanduser().resolve(strict=True),
            args.relay_wheel.expanduser().resolve(strict=True),
            args.relay_architecture,
            args.plugin_config_template.expanduser().resolve(strict=True),
        )

    task_tomls = [dataset_root / record["name"] / "task.toml" for record in task_records]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "harbor_version": "0.18.0",
        "dataset_name": dataset_root.name,
        "task_count": len(task_records),
        "dataset_task_definitions_sha256": combined_digest(dataset_root, task_tomls),
        "registry_network_attempts": network_attempts,
        "local_cli_projection": "passed",
        "job_trial_count": len(task_records),
        "concurrency": args.concurrency,
        "relay_architecture": args.relay_architecture,
        "relay_runtime": relay_runtime,
        "memory_lanes": {
            "2G": sum(record["memory_gb"] == 2 for record in task_records),
            "4G": sum(record["memory_gb"] == 4 for record in task_records),
            "8G": sum(record["memory_gb"] == 8 for record in task_records),
        },
        "tasks": task_records,
    }
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "tasks"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
