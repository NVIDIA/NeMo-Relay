# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent_module = load_module(
    "setup_admission_agent",
    EXAMPLE_ROOT / "agents" / "harbor_hermes_agent.py",
)
admission_module = load_module(
    "setup_admission_runner",
    EXAMPLE_ROOT / "scripts" / "run_setup_admission.py",
)
builder_module = load_module(
    "hermetic_runtime_builder",
    EXAMPLE_ROOT / "scripts" / "build_hermetic_runtime.py",
)
runtime_preparer_module = load_module(
    "phase2_runtime_preparer",
    EXAMPLE_ROOT / "scripts" / "prepare_runtime.py",
)


def make_payload(root: Path, *, digest: str = "a" * 64) -> dict[str, object]:
    for relative in (
        "bin/hermes",
        "bin/python",
        "bin/uv",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub", encoding="utf-8")
    (root / "hermes-agent-src" / "venv").mkdir(parents=True)
    ca_bundle = root / agent_module._HERMETIC_CA_BUNDLE_RELATIVE
    ca_bundle.parent.mkdir(parents=True, exist_ok=True)
    ca_bundle.write_text("test CA bundle", encoding="utf-8")
    marker = {
        "schema_version": agent_module._HERMETIC_RUNTIME_SCHEMA,
        "status": "passed",
        "content_sha256": digest,
        "hermes_commit": "b" * 40,
        "relay_wheel_sha256": "c" * 64,
        "relay_architecture": "aarch64",
    }
    (root / "payload.json").write_text(json.dumps(marker), encoding="utf-8")
    return marker


def test_hermetic_runtime_contract_accepts_bound_payload(tmp_path: Path) -> None:
    marker = make_payload(tmp_path)
    actual = agent_module._load_hermetic_runtime(
        tmp_path,
        expected_digest=str(marker["content_sha256"]),
        hermes_commit=str(marker["hermes_commit"]),
        relay_wheel_sha256=str(marker["relay_wheel_sha256"]),
        relay_architecture=str(marker["relay_architecture"]),
    )
    assert actual == marker


def test_hermetic_runtime_contract_rejects_changed_architecture(tmp_path: Path) -> None:
    marker = make_payload(tmp_path)
    with pytest.raises(ValueError, match="metadata mismatch"):
        agent_module._load_hermetic_runtime(
            tmp_path,
            expected_digest=str(marker["content_sha256"]),
            hermes_commit=str(marker["hermes_commit"]),
            relay_wheel_sha256=str(marker["relay_wheel_sha256"]),
            relay_architecture="x86_64",
        )


def test_hermetic_runtime_readiness_retries_nested_entrypoints(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    bin_dir = runtime / "bin"
    bin_dir.mkdir(parents=True)
    counter = tmp_path / "attempts"
    (bin_dir / "python").write_text(
        "#!/bin/sh\n"
        f'counter="{counter}"\n'
        'attempts="$(cat "$counter" 2>/dev/null || printf 0)"\n'
        'attempts="$((attempts + 1))"\n'
        'printf "%s" "$attempts" > "$counter"\n'
        '[ "$attempts" -ge 3 ]\n',
        encoding="utf-8",
    )
    (bin_dir / "hermes").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    os.chmod(bin_dir / "python", 0o755)
    os.chmod(bin_dir / "hermes", 0o755)
    command = agent_module._hermetic_runtime_readiness_command(str(runtime), attempts=4, delay_seconds=0)
    completed = subprocess.run(["bash", "-c", f"set -euo pipefail; {command}"], check=False)
    assert completed.returncode == 0
    assert counter.read_text(encoding="utf-8") == "3"


def test_setup_admission_binds_agent_source() -> None:
    expected = EXAMPLE_ROOT / "agents" / "harbor_hermes_agent.py"
    assert admission_module.SETUP_AGENT_PATH == expected
    assert admission_module.sha256_file(expected) == agent_module._sha256(expected)


def test_bridge_accepts_current_classifier_routing_contract() -> None:
    agent_module._validate_relay_config(EXAMPLE_ROOT / "config" / "plugins.toml.in")


def test_runtime_provenance_derives_classifier_target_from_plugin_config() -> None:
    with (EXAMPLE_ROOT / "config" / "plugins.toml.in").open("rb") as stream:
        config = tomllib.load(stream)
    assert runtime_preparer_module.plugin_settings(config)["classifier_target"] == "judge"


def test_offline_overrides_keep_classifier_pricing_aliases_distinct(tmp_path: Path) -> None:
    output = tmp_path / "plugins.toml"
    settings = runtime_preparer_module.render_config(
        EXAMPLE_ROOT / "config" / "plugins.toml.in",
        output,
        {
            "HERMES_COMMIT": "a" * 40,
            "OPENINFERENCE_ENDPOINT": "http://127.0.0.1:4318/v1/traces",
            "PHOENIX_PROJECT": "offline",
            "EVAL_COHORT": "offline",
        },
        {
            "provider_base_url": "http://127.0.0.1:8000/v1",
            "strong_model": "phase2/fake-strong",
            "weak_model": "phase2/fake-weak",
            "judge_model": "phase2/fake-judge",
        },
    )
    with output.open("rb") as stream:
        config = tomllib.load(stream)
    entries = config["components"][0]["config"]["sources"][0]["catalog"]["entries"]
    assert settings["judge_model"] == "phase2/fake-judge"
    assert {entry["model_id"] for entry in entries} == {
        "phase2/fake-strong",
        "phase2/fake-weak",
        "phase2/fake-judge",
    }


def test_hermetic_runtime_requires_portable_ca_bundle(tmp_path: Path) -> None:
    marker = make_payload(tmp_path)
    (tmp_path / agent_module._HERMETIC_CA_BUNDLE_RELATIVE).unlink()
    with pytest.raises(FileNotFoundError, match="hermetic runtime is incomplete"):
        agent_module._load_hermetic_runtime(
            tmp_path,
            expected_digest=str(marker["content_sha256"]),
            hermes_commit=str(marker["hermes_commit"]),
            relay_wheel_sha256=str(marker["relay_wheel_sha256"]),
            relay_architecture=str(marker["relay_architecture"]),
        )


def test_payload_tree_digest_ignores_its_marker(tmp_path: Path) -> None:
    content = tmp_path / "bin" / "python"
    content.parent.mkdir(parents=True)
    content.write_text("payload", encoding="utf-8")
    first = builder_module.sha256_tree(tmp_path)
    (tmp_path / "payload.json").write_text("first", encoding="utf-8")
    assert builder_module.sha256_tree(tmp_path) == first
    (tmp_path / "payload.json").write_text("second", encoding="utf-8")
    assert builder_module.sha256_tree(tmp_path) == first
    content.write_text("changed", encoding="utf-8")
    assert builder_module.sha256_tree(tmp_path) != first


def test_admission_rejects_tampered_hermetic_runtime(tmp_path: Path) -> None:
    content = tmp_path / "bin" / "python"
    content.parent.mkdir(parents=True)
    content.write_text("payload", encoding="utf-8")
    marker = {
        "schema_version": admission_module.PAYLOAD_SCHEMA,
        "status": "passed",
        "content_sha256": admission_module.hermetic_content_sha256(tmp_path),
    }
    (tmp_path / "payload.json").write_text(json.dumps(marker), encoding="utf-8")
    assert admission_module.load_payload(tmp_path) == marker
    content.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        admission_module.load_payload(tmp_path)


def test_payload_builder_forwards_non_secret_version_pins() -> None:
    source = (EXAMPLE_ROOT / "scripts" / "build_hermetic_runtime.py").read_text(encoding="utf-8")
    assert 'f"UV_VERSION={UV_VERSION}"' in source
    assert 'f"PYTHON_VERSION={PYTHON_VERSION}"' in source
    assert 'f"RELAY_WHEEL_NAME={relay_wheel.name}"' in source


def test_completed_result_is_invalidated_by_plan_input_change(tmp_path: Path) -> None:
    plan = {
        "inputs": {"concurrency": 4, "hermetic_runtime_sha256": "a" * 64},
        "tasks": [{"name": "task-one", "task_sha256": "b" * 64}],
    }
    binding = admission_module.task_bindings(plan)["task-one"]
    results = tmp_path / "task-results"
    results.mkdir()
    (results / "task-one.json").write_text(
        json.dumps(
            {
                "schema_version": admission_module.RESULT_SCHEMA,
                "status": "passed",
                "binding_sha256": binding,
            }
        ),
        encoding="utf-8",
    )
    assert admission_module.completed_names(tmp_path, plan) == {"task-one"}
    plan["inputs"]["concurrency"] = 5
    assert admission_module.completed_names(tmp_path, plan) == set()


def test_job_result_import_keeps_newest_attempt(tmp_path: Path) -> None:
    plan = {
        "inputs": {"concurrency": 4},
        "tasks": [{"name": "task-one", "task_sha256": "b" * 64}],
    }
    for job_name, message in (("job-001", "old failure"), ("job-002", "new failure")):
        trial = tmp_path / "jobs" / job_name / "trial-one"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "task-one",
                    "exception_info": {
                        "exception_type": "RuntimeError",
                        "exception_message": message,
                    },
                    "environment_setup": None,
                    "agent_setup": None,
                    "agent_execution": None,
                    "verifier": None,
                }
            ),
            encoding="utf-8",
        )
    (tmp_path / "task-results").mkdir()
    admission_module.parse_job_results(tmp_path, plan)
    result = json.loads((tmp_path / "task-results" / "task-one.json").read_text())
    assert result["exception_message"] == "new failure"


def test_clock_preflight_rejects_remote_time_drift() -> None:
    evidence = admission_module.evaluate_clock_preflight(
        host_epoch=1_000.0,
        docker_epoch=1_001.0,
        reference_epoch=4_611.0,
    )
    assert evidence["status"] == "failed"
    assert evidence["host_reference_offset_seconds"] == 3_611.0
    assert evidence["docker_host_offset_seconds"] == 1.0


def test_clock_preflight_accepts_small_offsets() -> None:
    evidence = admission_module.evaluate_clock_preflight(
        host_epoch=1_000.0,
        docker_epoch=1_001.0,
        reference_epoch=1_002.0,
    )
    assert evidence["status"] == "passed"


def test_plugin_compatibility_uses_oldest_supported_base_without_secrets(
    tmp_path: Path,
) -> None:
    plan = {
        "inputs": {
            "relay_architecture": "aarch64",
            "switchyard_bundle": str(tmp_path / "switchyard"),
        }
    }
    command = admission_module.plugin_compatibility_command(plan)
    assert "python:3.11-bullseye" in command
    assert "linux/arm64" in command
    assert "nemo_relay_register_plugin" in command[-1]
    rendered = " ".join(command)
    assert "SWITCHYARD_PROVIDER_AUTHORIZATION" not in rendered
    assert "Bearer " not in rendered


def test_harbor_command_uses_install_only_without_provider_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    payload = tmp_path / "payload"
    payload.mkdir()
    harbor = tmp_path / "harbor"
    harbor.write_text("stub", encoding="utf-8")
    args = argparse.Namespace(
        harbor=harbor,
        dataset=tmp_path / "dataset",
        hermetic_runtime=payload,
        output=output,
        concurrency=6,
        force_build=True,
        preserve_containers=False,
    )
    plan = {
        "inputs": {
            "hermes_commit": "b" * 40,
            "relay_config": str(tmp_path / "plugins.toml"),
            "switchyard_bundle": str(tmp_path / "switchyard"),
            "relay_wheel": str(tmp_path / "relay.whl"),
            "relay_wheel_sha256": "c" * 64,
            "relay_architecture": "aarch64",
            "hermetic_runtime_root": str(payload),
            "hermetic_runtime_sha256": "d" * 64,
        }
    }
    captured: list[str] = []

    class Completed:
        returncode = 0

    def fake_run(command: list[str], **_: object) -> Completed:
        captured.extend(command)
        return Completed()

    monkeypatch.setattr(admission_module.subprocess, "run", fake_run)
    assert admission_module.run_harbor(args, plan, ["one", "two"]) == 0
    assert "--install-only" in captured
    assert "--disable-verification" in captured
    assert "--force-build" in captured
    assert "--no-delete" not in captured
    assert captured.count("--include-task-name") == 2
    rendered = " ".join(captured)
    assert "SWITCHYARD_PROVIDER_AUTHORIZATION" not in rendered
    assert "provider-authorization" not in rendered


def test_setup_failure_classifies_transient_downloads_for_retry(tmp_path: Path) -> None:
    root = tmp_path / "admission"
    (root / "task-results").mkdir(parents=True)
    result = root / "jobs" / "job" / "trial" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "task_name": "one",
                "exception_info": {
                    "exception_type": "DockerBuildError",
                    "exception_message": "TLS handshake timeout contacting registry-1.docker.io",
                },
                "environment_setup": None,
                "agent_setup": None,
                "agent_execution": None,
                "verifier": None,
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "inputs": {"concurrency": 2},
        "tasks": [{"name": "one", "task_sha256": "a" * 64}],
    }
    admission_module.parse_job_results(root, plan)
    parsed = json.loads(admission_module.result_path(root, "one").read_text())
    assert parsed["status"] == "failed"
    assert parsed["failure_class"] == "infrastructure"
