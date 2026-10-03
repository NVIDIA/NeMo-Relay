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
    python_version = "3.14.7"
    ca_bundle = root / agent_module._hermetic_ca_bundle_relative(python_version)
    ca_bundle.parent.mkdir(parents=True, exist_ok=True)
    ca_bundle.write_text("test CA bundle", encoding="utf-8")
    marker = {
        "schema_version": agent_module._HERMETIC_RUNTIME_SCHEMA,
        "status": "passed",
        "content_sha256": digest,
        "hermes_commit": "b" * 40,
        "relay_wheel_sha256": "c" * 64,
        "relay_architecture": "aarch64",
        "python_version": python_version,
    }
    (root / "payload.json").write_text(json.dumps(marker), encoding="utf-8")
    return marker


def make_switchyard_bundle_dir(tmp_path: Path, experiment: str) -> Path:
    """Mimic prepare_runtime.py's staged bundle layout for one experiment,
    for tests that call _validate_relay_config directly against a real
    on-disk switchyard-routes.toml/pricing.json."""
    switchyard_source, pricing_source = runtime_preparer_module.experiment_paths(
        EXAMPLE_ROOT / "config", experiment
    )
    bundle = tmp_path / f"switchyard-bundle-{experiment}"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "switchyard-routes.toml").write_bytes(switchyard_source.read_bytes())
    (bundle / "pricing.json").write_bytes(pricing_source.read_bytes())
    return bundle


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


def test_bridge_accepts_current_classifier_routing_contract(tmp_path: Path) -> None:
    bundle = make_switchyard_bundle_dir(tmp_path, "default")
    agent_module._validate_relay_config(EXAMPLE_ROOT / "config" / "plugins.toml.in", bundle)


@pytest.mark.parametrize("experiment", ["random", "escalation"])
def test_bridge_accepts_additional_router_group_contracts(experiment: str, tmp_path: Path) -> None:
    bundle = make_switchyard_bundle_dir(tmp_path, experiment)
    assert agent_module._validate_relay_config(EXAMPLE_ROOT / "config" / "plugins.toml.in", bundle) == "switchyard"


def test_bridge_accepts_opus_only_direct_baseline(tmp_path: Path) -> None:
    path = EXAMPLE_ROOT / "config" / "plugins.opus48-baseline.toml.in"
    assert agent_module._validate_relay_config(path, tmp_path) == "direct"


def test_bridge_accepts_sol_direct_and_both_stage_picker_contracts(tmp_path: Path) -> None:
    assert (
        agent_module._validate_relay_config(EXAMPLE_ROOT / "config" / "plugins.sol56-baseline.toml.in", tmp_path)
        == "direct"
    )
    for experiment in ("sol56-deepseek-v4-cf03", "sol56-deepseek-v4-ef03"):
        bundle = make_switchyard_bundle_dir(tmp_path, experiment)
        assert (
            agent_module._validate_relay_config(EXAMPLE_ROOT / "config" / "plugins.toml.in", bundle)
            == "switchyard"
        )


def test_sol_stage_contracts_use_two_models_and_deepseek_judge_controls() -> None:
    expected_picker = {
        "sol56-deepseek-v4-cf03": "capable_first",
        "sol56-deepseek-v4-ef03": "efficient_first",
    }
    with (EXAMPLE_ROOT / "config" / "plugins.toml.in").open("rb") as stream:
        config = tomllib.load(stream)
    for experiment, picker in expected_picker.items():
        switchyard_source, _pricing_source = runtime_preparer_module.experiment_paths(
            EXAMPLE_ROOT / "config", experiment
        )
        with switchyard_source.open("rb") as stream:
            switchyard_routes = tomllib.load(stream)
        settings = runtime_preparer_module.plugin_settings(config, switchyard_routes)
        targets = switchyard_routes["targets"]
        assert settings["picker"] == picker
        assert settings["confidence_threshold"] == 0.3
        assert settings["strong_model"] == "openai/openai/gpt-5.6-sol"
        assert settings["weak_model"] == "nvidia/deepseek-ai/deepseek-v4-flash"
        assert settings["judge_model"] == settings["weak_model"]
        assert targets["weak"]["extra_body"] == {"reasoning": {"enabled": False}}
        assert targets["judge"]["extra_body"] == {"reasoning": {"enabled": False}}


def test_glm_stage_signal_and_classifier_arms_are_distinct_and_valid(tmp_path: Path) -> None:
    expected = {
        "sol56-glm52-stage-ef05-signal": None,
        "sol56-glm52-stage-ef05-classifier": "judge",
    }
    path = EXAMPLE_ROOT / "config" / "plugins.toml.in"
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    for experiment, classifier_target in expected.items():
        bundle = make_switchyard_bundle_dir(tmp_path, experiment)
        assert agent_module._validate_relay_config(path, bundle) == "switchyard"
        switchyard_source, _pricing_source = runtime_preparer_module.experiment_paths(
            EXAMPLE_ROOT / "config", experiment
        )
        with switchyard_source.open("rb") as stream:
            switchyard_routes = tomllib.load(stream)
        settings = runtime_preparer_module.plugin_settings(config, switchyard_routes)
        targets = switchyard_routes["targets"]
        assert settings["picker"] == "efficient_first"
        assert settings["confidence_threshold"] == 0.5
        assert settings["classifier_target"] == classifier_target
        assert settings["strong_model"] == "openai/openai/gpt-5.6-sol"
        assert settings["weak_model"] == "nvidia/zai-org/glm-5.2"
        # Switchyard's pinned native plugin accepts routed OpenAI Chat
        # targets; the Responses API is not yet a valid router target.
        assert switchyard_routes["llm_clients"]["nvidia"]["format"] == "openai_chat"
        # The signal arm raises max_tokens above the provider default after
        # real-task failures where Hermes reported the model exhausting its
        # output-token budget on reasoning before producing a final
        # response; the classifier arm has not needed that adjustment.
        if experiment == "sol56-glm52-stage-ef05-signal":
            assert targets["strong"]["extra_body"] == {"max_tokens": 32768, "reasoning": {"effort": "medium"}}
            assert targets["weak"]["extra_body"] == {"max_tokens": 32768, "reasoning": {"enabled": False}}
        else:
            assert targets["strong"]["extra_body"] == {"reasoning": {"effort": "medium"}}
            assert targets["weak"]["extra_body"] == {"reasoning": {"enabled": False}}
        if classifier_target is None:
            assert set(targets) == {"strong", "weak"}
            assert "judge_model" not in settings
        else:
            assert targets["judge"]["id"] == "nvidia/zai-org/glm-5.2"


def test_runtime_provenance_derives_stage_router_contract() -> None:
    with (EXAMPLE_ROOT / "config" / "plugins.toml.in").open("rb") as stream:
        config = tomllib.load(stream)
    switchyard_source, _pricing_source = runtime_preparer_module.experiment_paths(EXAMPLE_ROOT / "config", "default")
    with switchyard_source.open("rb") as stream:
        switchyard_routes = tomllib.load(stream)
    settings = runtime_preparer_module.plugin_settings(config, switchyard_routes)
    assert settings["algorithm"] == "stage_router"
    assert settings["classifier_target"] == "judge"
    assert settings["picker"] == "efficient_first"
    assert settings["confidence_threshold"] == 0.5
    assert settings["recent_turn_window"] == 3


def test_runtime_provenance_derives_random_router_contract() -> None:
    with (EXAMPLE_ROOT / "config" / "plugins.toml.in").open("rb") as stream:
        config = tomllib.load(stream)
    switchyard_source, _pricing_source = runtime_preparer_module.experiment_paths(EXAMPLE_ROOT / "config", "random")
    with switchyard_source.open("rb") as stream:
        switchyard_routes = tomllib.load(stream)
    settings = runtime_preparer_module.plugin_settings(config, switchyard_routes)
    assert settings["algorithm"] == "random"
    assert settings["random_weights"] == {"strong": 1.0, "weak": 1.0}
    assert "classifier_target" not in settings


def test_runtime_provenance_derives_escalation_router_contract() -> None:
    with (EXAMPLE_ROOT / "config" / "plugins.toml.in").open("rb") as stream:
        config = tomllib.load(stream)
    switchyard_source, _pricing_source = runtime_preparer_module.experiment_paths(
        EXAMPLE_ROOT / "config", "escalation"
    )
    with switchyard_source.open("rb") as stream:
        switchyard_routes = tomllib.load(stream)
    settings = runtime_preparer_module.plugin_settings(config, switchyard_routes)
    assert settings["algorithm"] == "llm_classifier"
    assert settings["classifier_mode"] == "escalation"
    assert settings["classifier_target"] == "judge"
    assert settings["escalation"] == {
        "confirmations": 1,
        "recent_turn_window": 28,
        "window_message_chars": 500,
    }


def test_runtime_provenance_derives_direct_opus_baseline() -> None:
    with (EXAMPLE_ROOT / "config" / "plugins.opus48-baseline.toml.in").open("rb") as stream:
        config = tomllib.load(stream)
    settings = runtime_preparer_module.plugin_settings(config, None)
    assert settings == {
        "algorithm": "direct",
        "direct_model": "aws/anthropic/bedrock-claude-opus-4-8",
        "direct_base_url": "https://inference-api.nvidia.com/v1",
        "hermes_caller_model": "aws/anthropic/bedrock-claude-opus-4-8",
    }


def test_runtime_preparer_admits_only_collector_bootstrap_state(tmp_path: Path) -> None:
    run_root = tmp_path / "attempt"
    telemetry = run_root / "telemetry"
    telemetry.mkdir(parents=True)
    (run_root / "collector.container-id").write_text("collector-id\n", encoding="utf-8")

    runtime_preparer_module.initialize_run_root(run_root, allow_existing_collector_state=True)

    (run_root / "unexpected").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="active collector bootstrap"):
        runtime_preparer_module.initialize_run_root(run_root, allow_existing_collector_state=True)


def test_offline_overrides_keep_classifier_pricing_aliases_distinct(tmp_path: Path) -> None:
    output = tmp_path / "plugins.toml"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    switchyard_source, pricing_source = runtime_preparer_module.experiment_paths(EXAMPLE_ROOT / "config", "default")
    settings = runtime_preparer_module.render_config(
        EXAMPLE_ROOT / "config" / "plugins.toml.in",
        output,
        bundle,
        {
            "HERMES_COMMIT": "a" * 40,
            "OPENINFERENCE_ENDPOINT": "http://127.0.0.1:4318/v1/traces",
            "PHOENIX_PROJECT": "offline",
            "EVAL_COHORT": "offline",
        },
        switchyard_source,
        pricing_source,
        {
            "provider_base_url": "http://127.0.0.1:8000/v1",
            "strong_model": "phase2/fake-strong",
            "weak_model": "phase2/fake-weak",
            "judge_model": "phase2/fake-judge",
        },
    )
    entries = json.loads((bundle / "pricing.json").read_text(encoding="utf-8"))["entries"]
    assert settings["judge_model"] == "phase2/fake-judge"
    assert {entry["model_id"] for entry in entries} == {
        "phase2/fake-strong",
        "phase2/fake-weak",
        "phase2/fake-judge",
    }


def test_hermetic_runtime_requires_portable_ca_bundle(tmp_path: Path) -> None:
    marker = make_payload(tmp_path)
    (tmp_path / agent_module._hermetic_ca_bundle_relative(str(marker["python_version"]))).unlink()
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
    assert 'f"PYTHON_VERSION={python_version}"' in source
    assert 'f"PYTHON_GLOB_VERSION={python_glob_version}"' in source
    assert 'f"RELAY_WHEEL_NAME={relay_wheel.name}"' in source
    assert 'f"nofile={BUILDER_NOFILE_LIMIT}"' in source
    assert 'f"HOST_UID={os.getuid()}"' in source
    assert 'f"HOST_GID={os.getgid()}"' in source


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


def test_namespaced_task_result_uses_a_single_safe_filename(tmp_path: Path) -> None:
    task_name = "terminal-bench/task-one"
    plan = {
        "inputs": {"concurrency": 4},
        "tasks": [{"name": task_name, "task_sha256": "b" * 64}],
    }
    trial = tmp_path / "jobs" / "job-001" / "trial-one"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": task_name,
                "exception_info": None,
                "environment_setup": {"finished_at": "2026-08-08T00:00:00Z"},
                "agent_setup": {"finished_at": "2026-08-08T00:00:01Z"},
                "agent_execution": None,
                "verifier": None,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "task-results").mkdir()

    admission_module.parse_job_results(tmp_path, plan)

    path = admission_module.result_path(tmp_path, task_name)
    assert path.name == "terminal-bench%2Ftask-one.json"
    assert path.parent == tmp_path / "task-results"
    assert json.loads(path.read_text())["task_name"] == task_name
    assert admission_module.completed_names(tmp_path, plan) == {task_name}


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
    assert "python:3.11-trixie" in command
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
    assert admission_module.run_harbor(args, plan, ["terminal-bench/one", "two"]) == 0
    assert "--install-only" in captured
    assert "--disable-verification" in captured
    assert "--force-build" in captured
    assert "--no-delete" not in captured
    assert captured.count("--include-task-name") == 2
    included = [captured[index + 1] for index, value in enumerate(captured) if value == "--include-task-name"]
    assert included == ["one", "two"]
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
