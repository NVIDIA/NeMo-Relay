# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


def load_coordinator():
    path = EXAMPLE_ROOT / "scripts" / "run_phase2_cohort.py"
    spec = importlib.util.spec_from_file_location("phase2_coordinator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_task(dataset: Path, name: str, memory: str) -> None:
    task = dataset / name
    task.mkdir(parents=True)
    (task / "task.toml").write_text(f'[environment]\nmemory = "{memory}"\n', encoding="utf-8")
    (task / "instruction.md").write_text(f"instruction for {name}\n", encoding="utf-8")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")


def write_passed_attempt(
    root: Path,
    task,
    *,
    model: str,
    cache_read: int,
    benchmark_passed: bool,
) -> None:
    attempt = root / "tasks" / task.directory_name / "attempts" / "001"
    attempt.mkdir(parents=True)
    summary = {
        "status": "passed",
        "validation": {
            "status": "passed",
            "benchmark_task_passed": benchmark_passed,
            "direct_result_status": "completed",
            "switchyard_decision_count": 2,
            "routed_models": [model],
            "routed_targets": ["strong"],
            "cache_read_tokens": cache_read,
            "cache_write_tokens": 0,
            "secret_findings": [],
        },
        "phoenix_upload": {"status": "passed", "uploaded_spans": 10},
    }
    (attempt / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def cohort_args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_root=root,
        dataset="terminal-bench@2.0",
        phoenix_project="phase2-project",
        eval_cohort="phase2-cohort",
        required_model=["sonnet", "opus"],
        require_cache_hit=True,
    )


def test_task_discovery_places_explicit_canary_before_lexical_lane(tmp_path: Path) -> None:
    module = load_coordinator()
    write_task(tmp_path, "task-z", "4G")
    write_task(tmp_path, "task-a", "2G")
    tasks = module.discover_tasks(tmp_path, 2, set(), "task-z")
    assert [(task.index, task.name, task.memory_gb) for task in tasks] == [
        (1, "task-z", 4),
        (2, "task-a", 2),
    ]


def test_task_discovery_without_a_canary_preserves_dataset_order(tmp_path: Path) -> None:
    module = load_coordinator()
    write_task(tmp_path, "task-z", "4G")
    write_task(tmp_path, "task-a", "2G")
    tasks = module.discover_tasks(tmp_path, 2, set(), None)
    assert [(task.index, task.name, task.memory_gb) for task in tasks] == [
        (1, "task-a", 2),
        (2, "task-z", 4),
    ]


def test_failed_attempt_is_preserved_and_passed_attempt_wins(tmp_path: Path) -> None:
    module = load_coordinator()
    task = module.Task(1, "task", 2)
    attempts = tmp_path / task.directory_name / "attempts"
    failed = attempts / "001"
    failed.mkdir(parents=True)
    (failed / "summary.json").write_text('{"status":"failed"}', encoding="utf-8")
    passed = attempts / "002"
    passed.mkdir()
    (passed / "summary.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "validation": {"status": "passed"},
                "phoenix_upload": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )
    assert module.successful_attempt(tmp_path / task.directory_name) == passed
    assert failed.is_dir()


def test_cohort_summary_requires_completion_cache_routes_and_secret_scan(tmp_path: Path) -> None:
    module = load_coordinator()
    tasks = [module.Task(1, "one", 2), module.Task(2, "two", 2)]
    write_passed_attempt(tmp_path, tasks[0], model="sonnet", cache_read=12, benchmark_passed=True)
    write_passed_attempt(tmp_path, tasks[1], model="opus", cache_read=0, benchmark_passed=False)
    summary = module.aggregate_summary(cohort_args(tmp_path), tasks)
    assert summary["status"] == "passed"
    assert summary["completed_tasks"] == 2
    assert summary["benchmark_pass_count"] == 1
    assert summary["benchmark_nonpass_count"] == 1
    assert summary["cohort_gates"]["cache_hit"]["cache_read_tokens"] == 12
    assert summary["cohort_gates"]["route_diversity"]["observed_models"] == ["opus", "sonnet"]


def test_cohort_summary_blocks_missing_route_even_when_tasks_pass(tmp_path: Path) -> None:
    module = load_coordinator()
    tasks = [module.Task(1, "one", 2)]
    write_passed_attempt(tmp_path, tasks[0], model="opus", cache_read=12, benchmark_passed=True)
    summary = module.aggregate_summary(cohort_args(tmp_path), tasks)
    assert summary["status"] == "partial"
    assert summary["cohort_gates"]["route_diversity"]["missing_models"] == ["sonnet"]


def test_integration_failure_does_not_erase_completed_benchmark_output(tmp_path: Path) -> None:
    module = load_coordinator()
    task = module.Task(1, "one", 2)
    attempt = tmp_path / "tasks" / task.directory_name / "attempts" / "001"
    attempt.mkdir(parents=True)
    attempt_summary = {
        "status": "passed",
        "validation": {
            "status": "passed",
            "benchmark": {"status": "passed", "errors": []},
            "integration": {
                "status": "failed",
                "errors": ["missing route mark", "Phoenix upload did not pass"],
                "warnings": [],
                "phoenix_upload": {"status": "failed"},
            },
            "benchmark_task_passed": True,
            "routed_models": ["sonnet"],
            "routed_targets": ["weak"],
            "cache_read_tokens": 12,
            "secret_findings": [],
        },
        "phoenix_upload": {"status": "failed", "error": "Phoenix upload command failed"},
    }
    (attempt / "summary.json").write_text(json.dumps(attempt_summary), encoding="utf-8")

    assert module.task_summary_passed(attempt / "summary.json") is True
    args = cohort_args(tmp_path)
    args.required_model = ["sonnet"]
    summary = module.aggregate_summary(args, [task])
    assert summary["completed_tasks"] == 1
    assert summary["tasks"][0]["status"] == "completed"
    assert summary["cohort_gates"]["integration_validation"] == {
        "passed": False,
        "failed_task_count": 1,
        "failures": [
            {"task": "one", "errors": ["missing route mark", "Phoenix upload did not pass"]}
        ],
    }
    assert summary["status"] == "partial"


def test_failure_classifier_retries_only_known_infrastructure_failures() -> None:
    module = load_coordinator()
    assert module.classify_failure("TLS handshake timeout contacting registry-1.docker.io") == "infrastructure"
    assert module.classify_failure("ConnectError: Error getting dataset terminal-bench@2.0") == "infrastructure"
    assert module.classify_failure("Command failed (exit 100): apt-get update && apt-get install") == "infrastructure"
    assert (
        module.classify_failure(
            "Provider has been unresponsive (no response received) for 11 consecutive stale attempts"
        )
        == "infrastructure"
    )
    assert module.classify_failure("trusted fallback: provider returned HTTP 408") == "infrastructure"
    assert module.classify_failure("provider returned HTTP 400") == "harness_or_integration"
    assert module.classify_failure("Command failed (exit 137): agent process") == "infrastructure"
    assert module.classify_failure("receipt did not prove plugin close") == "harness_or_integration"


def test_failure_classifier_reads_nested_harbor_logs(tmp_path: Path) -> None:
    module = load_coordinator()
    attempt = tmp_path / "attempt"
    nested = attempt / "jobs" / "task" / "trial.log"
    nested.parent.mkdir(parents=True)
    nested.write_text(
        'failed to do request: Head "https://registry-1.docker.io/v2/library/debian/manifests/13.0-slim": '
        "context deadline exceeded\n",
        encoding="utf-8",
    )

    assert module.classify_attempt_failure("expected one direct Hermes result, found 0", attempt) == "infrastructure"


def test_failure_classifier_reads_nested_harbor_result_json(tmp_path: Path) -> None:
    module = load_coordinator()
    attempt = tmp_path / "attempt"
    result = attempt / "jobs" / "task" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "exception_info": {
                    "exception_message": (
                        "Provider has been unresponsive (no response received) "
                        "for 10 consecutive stale attempts"
                    )
                }
            }
        ),
        encoding="utf-8",
    )

    assert module.classify_attempt_failure("invalid direct result status", attempt) == "infrastructure"


def test_smoke_evidence_is_bound_to_exact_local_dataset(tmp_path: Path) -> None:
    module = load_coordinator()
    dataset = tmp_path / "dataset"
    write_task(dataset, "task-a", "2G")
    write_task(dataset, "task-b", "4G")
    task_tomls = sorted(dataset.glob("*/task.toml"))
    relay_wheel = tmp_path / "relay.whl"
    relay_wheel.write_bytes(b"relay")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    library = bundle / "libswitchyard_nemo_relay_plugin.so"
    library.write_bytes(b"switchyard")
    plugin_template = EXAMPLE_ROOT / "config" / "plugins.toml.in"
    evidence_path = tmp_path / "smoke.json"
    evidence = {
        "schema_version": "harbor-hermes-switchyard.phase2-smoke.v1",
        "status": "passed",
        "task_count": 2,
        "dataset_task_definitions_sha256": module.sha256_file_set(dataset, task_tomls),
        "registry_network_attempts": 0,
        "concurrency": 3,
        "relay_architecture": "aarch64",
        "relay_runtime": {
            "status": "passed",
            "relay_wheel_sha256": module.sha256_file(relay_wheel),
            "switchyard_library_sha256": module.sha256_file(library),
            "plugin_config_template_sha256": module.sha256_file(plugin_template),
        },
        "tasks": [
            {
                "name": name,
                "instruction_path": f"{name}/instruction.md",
                "verifier_path": f"{name}/tests/test.sh",
                "task_toml_sha256": module.sha256_file(dataset / name / "task.toml"),
                "instruction_sha256": module.sha256_file(dataset / name / "instruction.md"),
                "test_sha256": module.sha256_file(dataset / name / "tests" / "test.sh"),
            }
            for name in ("task-a", "task-b")
        ],
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    module.validate_smoke_evidence(evidence_path, 2, dataset, 3, "aarch64", relay_wheel, bundle, plugin_template)

    (dataset / "task-b" / "task.toml").write_text('[environment]\nmemory = "8G"\n', encoding="utf-8")
    try:
        module.validate_smoke_evidence(evidence_path, 2, dataset, 3, "aarch64", relay_wheel, bundle, plugin_template)
    except ValueError:
        pass
    else:
        raise AssertionError("stale smoke evidence accepted a changed dataset")

    (dataset / "task-b" / "task.toml").write_text('[environment]\nmemory = "4G"\n', encoding="utf-8")
    (dataset / "task-a" / "instruction.md").write_text("changed instruction\n", encoding="utf-8")
    try:
        module.validate_smoke_evidence(evidence_path, 2, dataset, 3, "aarch64", relay_wheel, bundle, plugin_template)
    except ValueError:
        pass
    else:
        raise AssertionError("stale smoke evidence accepted a changed instruction")
    (dataset / "task-a" / "instruction.md").write_text("instruction for task-a\n", encoding="utf-8")

    try:
        module.validate_smoke_evidence(evidence_path, 2, dataset, 4, "aarch64", relay_wheel, bundle, plugin_template)
    except ValueError:
        pass
    else:
        raise AssertionError("smoke evidence accepted changed concurrency")


def test_durable_supervisor_owns_the_coordinator_process_group() -> None:
    supervisor = (EXAMPLE_ROOT / "supervise_phase2_cohort.sh").read_text(encoding="utf-8")
    assert "scripts/exec_process_group.py" in supervisor
    assert 'kill -TERM -- "-$child_pid"' in supervisor
    assert 'kill -KILL -- "-$child_pid"' in supervisor


def test_tmux_launcher_projects_only_the_protected_file_path() -> None:
    launcher = (EXAMPLE_ROOT / "scripts" / "launch_phase2_tmux.sh").read_text(encoding="utf-8")
    child = (EXAMPLE_ROOT / "scripts" / "run_phase2_from_env.sh").read_text(encoding="utf-8")
    assert 'env_file="$example_root/.env"' in launcher
    assert 'usage: $0 [env-file] tmux-session-name' in launcher
    assert '-e "TERMINAL_BENCH_ENV_FILE=$env_file"' in launcher
    assert 'source "$env_file"' not in launcher
    assert "tmux has-session" in launcher
    assert 'source "$env_file"' in child
    assert "set +x" in child
    assert "stage_phase2_runtime.py" in child
    assert 'runtime_harness="$TERMINAL_BENCH_RUN_ROOT/runtime-harness"' in child
    assert 'exec "$runtime_harness/supervise_phase2_cohort.sh"' in child
    assert "supervisor.log" in child


def test_runtime_snapshot_remains_bound_after_checkout_changes(tmp_path: Path) -> None:
    import importlib.util

    script = EXAMPLE_ROOT / "scripts" / "stage_phase2_runtime.py"
    spec = importlib.util.spec_from_file_location("stage_phase2_runtime", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = tmp_path / "source"
    for relative in ("agents", "config", "scripts"):
        (source / relative).mkdir(parents=True)
    for relative in module.RUNTIME_TOP_LEVEL:
        (source / relative).write_text(f"{relative}\n", encoding="utf-8")
    (source / "scripts" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    digest = module.runtime_digest(source, module.runtime_files(source))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"inputs": {"runtime_sources_sha256": digest}}), encoding="utf-8")
    destination = tmp_path / "runtime-harness"

    assert module.stage_runtime(source, destination, plan)["status"] == "staged"
    (source / "scripts" / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert module.stage_runtime(source, destination, plan)["status"] == "verified"
    assert (destination / "scripts" / "helper.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_phase2_launcher_is_local_dataset_only() -> None:
    launcher = (EXAMPLE_ROOT / "run_phase2_cohort.sh").read_text(encoding="utf-8")
    assert 'dataset_root="${TBENCH_DATASET_PATH:-$dataset_export_root/$dataset_name}"' in launcher
    assert "datasets download" not in launcher
    assert "never downloads or resolves a dataset through the Harbor registry" in launcher
    assert '--smoke-evidence "$smoke_evidence"' in launcher
    assert '--offline-evidence "$offline_evidence"' in launcher
    assert "PHASE1_EVIDENCE_ROOT" not in launcher
    assert "INFERENCE_SECRETS_FILE" not in launcher


def test_coordinator_owns_bounded_force_build_setup_lane(
    tmp_path: Path,
) -> None:
    module = load_coordinator()
    args = argparse.Namespace(
        run_root=tmp_path / "run",
        python_bin=tmp_path / "python",
        setup_admission_runner=tmp_path / "run_setup_admission.py",
        dataset_root=tmp_path / "dataset",
        setup_runtime=tmp_path / "setup-runtime",
        hermetic_runtime=tmp_path / "hermetic-runtime",
        harbor_bin=tmp_path / "harbor",
        setup_concurrency=2,
        setup_batch_size=89,
        setup_max_infra_attempts=4,
        backoff_seconds=0,
        hermetic_runtime_payload={"content_sha256": "a" * 64},
    )
    args.run_root.mkdir()
    captured: list[str] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        captured.extend(command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        (output / "summary.json").write_text(
            json.dumps({"status": "passed", "planned": 2, "passed": 2}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    tasks = [module.Task(1, "one", 2), module.Task(2, "two", 2)]
    with patch.object(module.subprocess, "run", side_effect=fake_run):
        assert module.CohortRunner(args, tasks).provision_environments()
    assert captured[captured.index("--concurrency") + 1] == "2"
    assert captured[captured.index("--batch-size") + 1] == "89"
    assert "--force-build" in captured
    assert "--no-preserve-containers" in captured
    rendered = " ".join(captured)
    assert "SWITCHYARD_PROVIDER_AUTHORIZATION" not in rendered
    assert "Bearer " not in rendered


def test_provider_attempt_rebuilds_from_cached_layers_and_uses_hermetic_runtime() -> None:
    runner = (EXAMPLE_ROOT / "run_terminal_bench.sh").read_text(encoding="utf-8")
    coordinator = (EXAMPLE_ROOT / "scripts" / "run_phase2_cohort.py").read_text(encoding="utf-8")
    assert 'HARBOR_FORCE_BUILD": "true"' in coordinator
    assert 'HERMETIC_RUNTIME_DIR": str(self.args.hermetic_runtime)' in coordinator
    assert 'harbor_force_build="${HARBOR_FORCE_BUILD:-true}"' in runner
    assert "harbor_build_args+=(--force-build)" in runner
    assert '--ak "hermetic_runtime_sha256=$hermetic_runtime_sha256"' in runner
    assert '"target": "/opt/hermes-runtime"' in runner
    assert "--ae 'OPENROUTER_API_KEY=relay-intercepted'" in runner
    assert '--ae "OPENROUTER_BASE_URL=$fail_closed_openai_base_url"' in runner


def test_all_task_smoke_fail_closes_parent_and_delegated_provider_urls() -> None:
    smoke = (EXAMPLE_ROOT / "scripts" / "smoke_phase2_dataset.py").read_text(encoding="utf-8")
    assert '"OPENAI_BASE_URL": "http://127.0.0.1:9/v1"' in smoke
    assert '"OPENROUTER_API_KEY": "relay-intercepted"' in smoke
    assert '"OPENROUTER_BASE_URL": "http://127.0.0.1:9/v1"' in smoke


def test_runner_does_not_expand_an_empty_validation_expectations_array() -> None:
    runner = (EXAMPLE_ROOT / "run_terminal_bench.sh").read_text(encoding="utf-8")
    assert "validation_expectations=()" not in runner
    assert '"${validation_expectations[@]}"' not in runner
    assert "validation_args+=(--expect-late-failure)" in runner


def test_runner_keeps_benchmark_completion_separate_from_integration_validation() -> None:
    runner = (EXAMPLE_ROOT / "run_terminal_bench.sh").read_text(encoding="utf-8")
    assert 'summary["benchmark_completion"] = summary["validation"].get("benchmark", {})' in runner
    assert 'integration["phoenix_upload"] = summary["phoenix_upload"]' in runner
    assert 'summary["integration_validation"] = integration' in runner
    assert "benchmark completion did not pass" in runner
    assert "benchmark completion or Phoenix upload did not pass" not in runner


def test_plugin_contract_owns_routes_and_authorization_name() -> None:
    module = load_coordinator()
    contract = module.plugin_contract(EXAMPLE_ROOT / "config" / "plugins.toml.in")
    assert contract["strong_model"] == "aws/anthropic/bedrock-claude-opus-5"
    assert contract["weak_model"] == "aws/anthropic/bedrock-claude-sonnet-5"
    assert contract["judge_model"] == "aws/anthropic/bedrock-claude-sonnet-4-6"
    assert contract["hermes_caller_model"] == "ollama-route-stub"
    assert contract["provider_base_urls"] == ["https://inference-api.nvidia.com/v1"]


def test_provider_catalog_requires_every_configured_route_without_persisting_authorization() -> None:
    module = load_coordinator()

    class Response(BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    models = ["provider/sonnet", "provider/opus"]
    response = Response(json.dumps({"data": [{"id": model} for model in models]}).encode())
    with patch.object(module.urllib.request, "urlopen", return_value=response) as request:
        assert module.verify_provider_catalog("https://provider.example/v1", "Bearer secret", models) == sorted(models)
    projected = request.call_args.args[0]
    assert projected.full_url == "https://provider.example/v1/models"
    assert projected.headers["Authorization"] == "Bearer secret"

    missing = Response(json.dumps({"data": [{"id": models[0]}]}).encode())
    with patch.object(module.urllib.request, "urlopen", return_value=missing):
        try:
            module.verify_provider_catalog("https://provider.example/v1", "Bearer secret", models)
        except RuntimeError as error:
            assert models[1] in str(error)
            assert "Bearer secret" not in str(error)
        else:
            raise AssertionError("provider catalog accepted a missing configured model")


def test_capacity_requirement_covers_parallel_lane_and_largest_serial_task() -> None:
    module = load_coordinator()
    args = argparse.Namespace(concurrency=6, parallel_max_memory_gb=2, docker_memory_reserve_gb=4)
    tasks = [module.Task(1, "canary", 2), module.Task(2, "large", 8)]
    assert module.capacity_requirement_gb(args, tasks) == 16
    args.concurrency = 2
    assert module.capacity_requirement_gb(args, tasks) == 12
    assert module.normalize_architecture("amd64") == "x86_64"
    assert module.normalize_architecture("arm64") == "aarch64"


def test_preflight_hard_rejects_capacity_above_docker_memory(tmp_path: Path) -> None:
    module = load_coordinator()
    args = argparse.Namespace(
        run_root=tmp_path / "run",
        minimum_free_gb=100,
        concurrency=4,
        parallel_max_memory_gb=2,
        docker_memory_reserve_gb=4,
        relay_architecture="aarch64",
    )
    docker_info = json.dumps({"NCPU": 8, "MemTotal": 11 * 1024**3, "Architecture": "arm64"})
    completed = SimpleNamespace(stdout=docker_info)
    tasks = [module.Task(1, "canary", 2)]
    with (
        patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=200 * 1024**3)),
        patch.object(module.subprocess, "run", return_value=completed),
    ):
        try:
            module.shared_preflight(args, tasks)
        except RuntimeError as error:
            assert "requires 12 GiB" in str(error)
        else:
            raise AssertionError("unsafe Docker memory capacity was accepted")


def test_existing_plan_is_immutable(tmp_path: Path) -> None:
    module = load_coordinator()
    path = tmp_path / "plan.json"
    module.load_or_create_plan(path, {"concurrency": 4})
    module.load_or_create_plan(path, {"concurrency": 4})
    try:
        module.load_or_create_plan(path, {"concurrency": 6})
    except ValueError:
        pass
    else:
        raise AssertionError("existing plan accepted changed concurrency")


def test_offline_evidence_is_bound_to_runtime_inputs(tmp_path: Path) -> None:
    module = load_coordinator()
    relay_wheel = tmp_path / "relay.whl"
    relay_wheel.write_bytes(b"relay")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    library = bundle / "libswitchyard_nemo_relay_plugin.so"
    library.write_bytes(b"switchyard")
    plugin_template = EXAMPLE_ROOT / "config" / "plugins.toml.in"
    evidence_path = tmp_path / "offline.json"
    evidence = {
        "schema_version": "harbor-hermes-switchyard.phase2-offline-admission.v1",
        "status": "passed",
        "hermes_commit": module.EXPECTED_HERMES_COMMIT,
        "relay_architecture": "x86_64",
        "relay_wheel_sha256": module.sha256_file(relay_wheel),
        "switchyard_library_sha256": module.sha256_file(library),
        "plugin_config_template_sha256": module.sha256_file(plugin_template),
        "provider_requests": 4,
        "surviving_shutdown_threads": [],
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    module.validate_offline_evidence(evidence_path, "x86_64", relay_wheel, bundle, plugin_template)
    evidence["relay_architecture"] = "aarch64"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    try:
        module.validate_offline_evidence(evidence_path, "x86_64", relay_wheel, bundle, plugin_template)
    except ValueError:
        pass
    else:
        raise AssertionError("offline evidence accepted a changed architecture")
