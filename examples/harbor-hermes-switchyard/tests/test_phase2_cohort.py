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


def reusable_setup_fixture(module, tmp_path: Path):
    dataset = tmp_path / "dataset"
    for index in range(89):
        name = f"task-{index:03d}"
        write_task(dataset, name, "2G")
        task = dataset / name
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
        (task / "task.toml").write_text(
            "\n".join(
                (
                    'schema_version = "1.1"',
                    "artifacts = []",
                    "",
                    "[task]",
                    f'name = "terminal-bench/{name}"',
                    'description = "setup reuse test"',
                    "keywords = []",
                    "",
                    "[verifier]",
                    "timeout_sec = 900.0",
                    "",
                    "[agent]",
                    "timeout_sec = 900.0",
                    "",
                    "[environment]",
                    f'docker_image = "example/{name}:test"',
                    "cpus = 1",
                    "memory_mb = 2048",
                    "storage_mb = 1024",
                    "gpus = 0",
                    "allow_internet = false",
                    "mcp_servers = []",
                    "",
                    "[verifier.env]",
                    "",
                    "[environment.env]",
                    "",
                    "[solution.env]",
                    "",
                )
            ),
            encoding="utf-8",
        )
    harbor_tasks = module.setup_admission.discover_tasks(dataset)
    records = [module.setup_admission.task_record(task) for task in harbor_tasks]
    tasks = [module.Task(index, str(record["name"]).rsplit("/", 1)[-1], 2) for index, record in enumerate(records, 1)]
    relay_wheel = tmp_path / "relay.whl"
    relay_wheel.write_bytes(b"relay")
    switchyard_bundle = tmp_path / "switchyard"
    switchyard_bundle.mkdir()
    library = switchyard_bundle / "libswitchyard_nemo_relay_plugin.so"
    library.write_bytes(b"switchyard")
    evidence = tmp_path / "setup-admission"
    evidence.mkdir()
    inputs = {
        "dataset_root": str(dataset),
        "dataset_sha256": module.setup_admission.canonical_sha256(
            [{key: value for key, value in record.items() if key != "task_dir"} for record in records]
        ),
        "hermetic_runtime_sha256": "a" * 64,
        "hermes_commit": module.EXPECTED_HERMES_COMMIT,
        "relay_architecture": "x86_64",
        "relay_wheel_sha256": module.sha256_file(relay_wheel),
        "switchyard_library_sha256": module.sha256_file(library),
        "harbor_version": module.setup_admission.harbor_version,
        "concurrency": 24,
        "batch_size": 89,
        "maximum_infrastructure_attempts": 4,
        "force_build": True,
        "preserve_containers": False,
        "setup_agent_sha256": "b" * 64,
    }
    plan = {
        "schema_version": module.setup_admission.PLAN_SCHEMA,
        "status": "planned",
        "inputs": inputs,
        "tasks": records,
    }
    (evidence / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    summary = {
        "schema_version": module.setup_admission.SUMMARY_SCHEMA,
        "status": "passed",
        "plan_sha256": module.setup_admission.canonical_sha256(plan),
        "planned": 89,
        "passed": 89,
        "failed": 0,
        "pending": 0,
    }
    (evidence / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    bindings = module.setup_admission.task_bindings(plan)
    for name, binding in bindings.items():
        result = evidence / "task-results" / f"{name}.json"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(
            json.dumps(
                {
                    "schema_version": module.setup_admission.RESULT_SCHEMA,
                    "status": "passed",
                    "task_name": name,
                    "binding_sha256": binding,
                    "agent_execution_skipped": True,
                    "verifier_skipped": True,
                }
            ),
            encoding="utf-8",
        )
    args = argparse.Namespace(
        reuse_setup_evidence=evidence,
        dataset_root=dataset,
        hermetic_runtime_payload={
            "content_sha256": "a" * 64,
            "hermes_commit": module.EXPECTED_HERMES_COMMIT,
        },
        relay_architecture="x86_64",
        relay_wheel=relay_wheel,
        switchyard_bundle=switchyard_bundle,
        setup_concurrency=24,
        setup_batch_size=89,
        setup_max_infra_attempts=4,
    )
    return args, tasks, evidence


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


def test_task_discovery_accepts_harbor_memory_mb(tmp_path: Path) -> None:
    module = load_coordinator()
    task = tmp_path / "task-a"
    task.mkdir()
    (task / "task.toml").write_text("[environment]\nmemory_mb = 8192\n", encoding="utf-8")
    tasks = module.discover_tasks(tmp_path, 1, set(), None)
    assert [(item.name, item.memory_gb) for item in tasks] == [("task-a", 8)]


def test_task_runtime_overrides_are_explicit_and_affect_effective_memory() -> None:
    module = load_coordinator()
    video = module.Task(1, "extract-moves-from-video", 2)
    torch = module.Task(2, "torch-pipeline-parallelism", 8)
    ordinary = module.Task(3, "ordinary", 2)

    assert video.as_json() == {
        "index": 1,
        "name": "extract-moves-from-video",
        "memory_gb": 2,
        "effective_memory_gb": 8,
        "runtime_override": {"memory_mb": 8192},
    }
    assert torch.runtime_override == {"verifier_cpu_thread_limit": 1}
    assert torch.effective_memory_gb == 8
    assert ordinary.runtime_override == {}
    assert ordinary.effective_memory_gb == 2


def test_direct_opus_baseline_contract_has_no_router_targets() -> None:
    module = load_coordinator()
    contract = module.plugin_contract(EXAMPLE_ROOT / "config" / "plugins.opus48-baseline.toml.in", None)
    assert contract["mode"] == "direct"
    assert contract["required_models"] == ["aws/anthropic/bedrock-claude-opus-4-8"]
    assert contract["weak_model"] is None
    assert contract["judge_model"] is None


def test_new_sol_matrix_contracts_are_two_model_stage_routes() -> None:
    module = load_coordinator()
    direct = module.plugin_contract(EXAMPLE_ROOT / "config" / "plugins.sol56-baseline.toml.in", None)
    assert direct["required_models"] == ["openai/openai/gpt-5.6-sol"]
    for experiment, picker in (
        ("sol56-deepseek-v4-cf03", "capable_first"),
        ("sol56-deepseek-v4-ef03", "efficient_first"),
    ):
        contract = module.plugin_contract(EXAMPLE_ROOT / "config" / "plugins.toml.in", experiment)
        assert contract["required_models"] == [
            "nvidia/deepseek-ai/deepseek-v4-flash",
            "openai/openai/gpt-5.6-sol",
        ]
        assert contract["picker"] == picker
        assert contract["confidence_threshold"] == 0.3
        assert contract["judge_model"] == contract["weak_model"]


def test_capacity_uses_effective_task_memory_for_runtime_overrides() -> None:
    module = load_coordinator()
    args = argparse.Namespace(concurrency=1, parallel_max_memory_gb=2, docker_memory_reserve_gb=4)
    tasks = [module.Task(1, "extract-moves-from-video", 2)]

    assert module.capacity_requirement_gb(args, tasks) == 12


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
        "failures": [{"task": "one", "errors": ["missing route mark", "Phoenix upload did not pass"]}],
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
    assert module.classify_failure("harbor.errors.VerifierTimeoutError") == "infrastructure"
    assert module.classify_failure("verifier assertion failed") == "harness_or_integration"
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
                        "Provider has been unresponsive (no response received) for 10 consecutive stale attempts"
                    )
                }
            }
        ),
        encoding="utf-8",
    )

    assert module.classify_attempt_failure("invalid direct result status", attempt) == "infrastructure"


def test_failure_classifier_retries_exhausted_hermes_continuations(tmp_path: Path) -> None:
    module = load_coordinator()
    attempt = tmp_path / "attempt"
    diagnostic = attempt / "jobs" / "task" / "agent" / "direct-hermes" / "diagnostics" / "hermes-tail.txt"
    diagnostic.parent.mkdir(parents=True)
    diagnostic.write_text(
        "Error: Response remained truncated after 4 continuation attempts\n",
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
    switchyard_experiment = "default"
    switchyard_source, pricing_source = module.experiment_paths(plugin_template.parent, switchyard_experiment)
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
            "plugin_config_template_sha256": module.plugin_config_identity_sha256(
                plugin_template, switchyard_source, pricing_source
            ),
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
    module.validate_smoke_evidence(
        evidence_path, 2, dataset, 3, "aarch64", relay_wheel, bundle, plugin_template, switchyard_experiment
    )

    (dataset / "task-b" / "task.toml").write_text('[environment]\nmemory = "8G"\n', encoding="utf-8")
    try:
        module.validate_smoke_evidence(
            evidence_path, 2, dataset, 3, "aarch64", relay_wheel, bundle, plugin_template, switchyard_experiment
        )
    except ValueError:
        pass
    else:
        raise AssertionError("stale smoke evidence accepted a changed dataset")

    (dataset / "task-b" / "task.toml").write_text('[environment]\nmemory = "4G"\n', encoding="utf-8")
    (dataset / "task-a" / "instruction.md").write_text("changed instruction\n", encoding="utf-8")
    try:
        module.validate_smoke_evidence(
            evidence_path, 2, dataset, 3, "aarch64", relay_wheel, bundle, plugin_template, switchyard_experiment
        )
    except ValueError:
        pass
    else:
        raise AssertionError("stale smoke evidence accepted a changed instruction")
    (dataset / "task-a" / "instruction.md").write_text("instruction for task-a\n", encoding="utf-8")

    try:
        module.validate_smoke_evidence(
            evidence_path, 2, dataset, 4, "aarch64", relay_wheel, bundle, plugin_template, switchyard_experiment
        )
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
    docker_group_runner = (EXAMPLE_ROOT / "scripts" / "run_phase2_with_docker_group.sh").read_text(encoding="utf-8")
    child = (EXAMPLE_ROOT / "scripts" / "run_phase2_from_env.sh").read_text(encoding="utf-8")
    assert 'env_file="$example_root/.env"' in launcher
    assert "usage: $0 [env-file] tmux-session-name" in launcher
    assert '-e "TERMINAL_BENCH_ENV_FILE=$env_file"' in launcher
    assert 'source "$env_file"' not in launcher
    assert "tmux has-session" in launcher
    assert 'scripts/run_phase2_with_docker_group.sh"' in launcher
    assert "docker info" in docker_group_runner
    assert "exec sg docker" in docker_group_runner
    assert 'exec "$runner"' in docker_group_runner
    assert 'source "$env_file"' not in docker_group_runner
    assert 'source "$env_file"' in child
    assert "set +x" in child
    assert "stage_phase2_runtime.py" in child
    assert 'runtime_harness="$TERMINAL_BENCH_RUN_ROOT/runtime-harness"' in child
    assert 'exec "$runtime_harness/supervise_phase2_cohort.sh"' in child
    assert "supervisor.log" in child


def test_phase2_environment_allows_an_explicitly_blank_canary() -> None:
    validator = (EXAMPLE_ROOT / "scripts" / "validate_phase2_environment.sh").read_text(encoding="utf-8")
    required_values = validator.split("required_values=(", 1)[1].split(")", 1)[0]
    assert "TBENCH_CANARY_TASK" not in required_values
    assert "[[ ! ${TBENCH_CANARY_TASK+x} ]]" in validator
    assert "non-placeholder bare provider credential" in validator


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
    assert 'reuse_setup_evidence="${TBENCH_REUSE_SETUP_EVIDENCE:-}"' in launcher
    assert '--reuse-setup-evidence "$reuse_setup_evidence"' in launcher
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


def test_complete_bound_setup_evidence_can_be_reused_without_builds(tmp_path: Path) -> None:
    module = load_coordinator()
    args, tasks, _ = reusable_setup_fixture(module, tmp_path)
    reused = module.validate_reused_setup_evidence(args, tasks)
    assert reused is not None
    assert reused["mode"] == "reused"
    assert reused["status"] == "passed"
    assert reused["task_count"] == 89

    run_root = tmp_path / "run"
    run_root.mkdir()
    provision_args = argparse.Namespace(
        run_root=run_root,
        reused_setup_evidence_payload=reused,
        hermetic_runtime_payload={"content_sha256": "a" * 64},
        setup_concurrency=24,
    )
    with patch.object(module.subprocess, "run") as subprocess_run:
        assert module.CohortRunner(provision_args, tasks).provision_environments()
    subprocess_run.assert_not_called()
    state = json.loads((run_root / "setup-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "passed"
    assert state["reused_setup_evidence"]["evidence_sha256"] == reused["evidence_sha256"]


def test_reused_setup_evidence_rejects_a_changed_task_binding(tmp_path: Path) -> None:
    module = load_coordinator()
    args, tasks, evidence = reusable_setup_fixture(module, tmp_path)
    result = next((evidence / "task-results").rglob("*.json"))
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["binding_sha256"] = "0" * 64
    result.write_text(json.dumps(payload), encoding="utf-8")

    try:
        module.validate_reused_setup_evidence(args, tasks)
    except ValueError as error:
        assert "invalid task result" in str(error)
    else:
        raise AssertionError("reused setup evidence accepted a changed task binding")


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
    assert 'agent_openrouter_base_url="$fail_closed_openai_base_url"' in runner
    assert 'agent_openrouter_base_url="$agent_openai_base_url"' in runner
    assert '"$routing_mode" != "random" && "$routing_mode" != "llm_classifier"' in runner
    assert '--ae "OPENROUTER_BASE_URL=$agent_openrouter_base_url"' in runner
    assert 'harbor_resource_args+=(--override-memory-mb "$task_memory_override_mb")' in runner
    assert 'verifier_env_args+=(--ve "$variable=$verifier_cpu_thread_limit")' in runner
    assert '"TASK_MEMORY_OVERRIDE_MB": ""' in coordinator
    assert '"VERIFIER_CPU_THREAD_LIMIT": ""' in coordinator


def test_provider_attempt_requires_a_writable_ready_collector_and_host_gateway() -> None:
    runner = (EXAMPLE_ROOT / "run_terminal_bench.sh").read_text(encoding="utf-8")
    assert '--user "$(id -u):$(id -g)"' in runner
    assert 'docker_host_gateway="$(docker network inspect bridge' in runner
    assert '--publish "$docker_host_gateway::4318"' in runner
    assert 'free_port="${published_endpoint##*:}"' in runner
    assert 'mkdir -p -m 0700 "$run_root/telemetry"' in runner
    assert "docker run --detach --rm" not in runner
    assert "OpenTelemetry collector exited before becoming ready" in runner
    assert "OpenTelemetry collector did not become ready" in runner
    assert "capture_collector_logs" in runner
    assert 'docker rm --force "$collector_name"' in runner
    assert '"extra_hosts": ["host.docker.internal:host-gateway"]' in runner
    assert '--extra-docker-compose "$host_gateway_compose"' in runner


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
    contract = module.plugin_contract(EXAMPLE_ROOT / "config" / "plugins.toml.in", "default")
    assert contract["strong_model"] == "aws/anthropic/bedrock-claude-opus-4-8"
    assert contract["weak_model"] == "nvidia/nvidia/nemotron-3-ultra-nvfp4"
    assert contract["judge_model"] == "aws/anthropic/bedrock-claude-sonnet-4-6"
    assert contract["hermes_caller_model"] == "ollama-route-stub"
    assert contract["provider_base_urls"] == ["https://inference-api.nvidia.com/v1"]
    assert contract["algorithm"] == "stage_router"
    assert contract["classifier_target"] == "judge"
    assert contract["picker"] == "efficient_first"
    assert contract["confidence_threshold"] == 0.5
    assert contract["recent_turn_window"] == 3


def test_random_router_contract_is_equal_weight_and_entropy_backed() -> None:
    module = load_coordinator()
    contract = module.plugin_contract(EXAMPLE_ROOT / "config" / "plugins.toml.in", "random")
    assert contract["algorithm"] == "random"
    assert contract["random_weights"] == {"strong": 1.0, "weak": 1.0}
    assert contract["classifier_target"] is None
    assert contract["required_models"] == [
        "aws/anthropic/bedrock-claude-opus-4-8",
        "nvidia/nvidia/nemotron-3-ultra-nvfp4",
    ]


def test_escalation_router_contract_uses_tuned_defaults_explicitly() -> None:
    module = load_coordinator()
    contract = module.plugin_contract(EXAMPLE_ROOT / "config" / "plugins.toml.in", "escalation")
    assert contract["algorithm"] == "llm_classifier"
    assert contract["classifier_mode"] == "escalation"
    assert contract["classifier_target"] == "judge"
    assert contract["escalation"] == {
        "confirmations": 1,
        "recent_turn_window": 28,
        "window_message_chars": 500,
    }


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
    switchyard_experiment = "default"
    switchyard_source, pricing_source = module.experiment_paths(plugin_template.parent, switchyard_experiment)
    evidence_path = tmp_path / "offline.json"
    evidence = {
        "schema_version": "harbor-hermes-switchyard.phase2-offline-admission.v1",
        "status": "passed",
        "hermes_commit": module.EXPECTED_HERMES_COMMIT,
        "relay_architecture": "x86_64",
        "relay_wheel_sha256": module.sha256_file(relay_wheel),
        "switchyard_library_sha256": module.sha256_file(library),
        "plugin_config_template_sha256": module.plugin_config_identity_sha256(
            plugin_template, switchyard_source, pricing_source
        ),
        "provider_requests": 4,
        "surviving_shutdown_threads": [],
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    module.validate_offline_evidence(
        evidence_path, "x86_64", relay_wheel, bundle, plugin_template, switchyard_experiment
    )
    evidence["relay_architecture"] = "aarch64"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    try:
        module.validate_offline_evidence(
            evidence_path, "x86_64", relay_wheel, bundle, plugin_template, switchyard_experiment
        )
    except ValueError:
        pass
    else:
        raise AssertionError("offline evidence accepted a changed architecture")
