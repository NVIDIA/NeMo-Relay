# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = EXAMPLE_ROOT / "terminal-bench-report" / "scripts"


def load_library():
    if str(SCRIPT_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPT_ROOT))
    path = SCRIPT_ROOT / "report_lib.py"
    spec = importlib.util.spec_from_file_location("terminal_bench_report_lib", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_synthetic_run(root: Path, *, complete: bool = False) -> None:
    root.mkdir()
    plan = {
        "dataset": "terminal-bench/test",
        "sample_count": 2,
        "concurrency": 2,
        "parallel_max_memory_gb": 4,
        "timeout_multipliers": {"agent": 3},
        "inputs": {
            "dataset_task_definitions_sha256": "a" * 64,
            "runtime_sources_sha256": "b" * 64,
            "runner_sha256": "c" * 64,
            "relay_wheel_sha256": "d" * 64,
            "switchyard_manifest_sha256": "e" * 64,
            "switchyard_library_sha256": "f" * 64,
            "plugin_config_template_sha256": "1" * 64,
        },
        "tasks": [
            {"index": 1, "name": "task-a", "memory_gb": 2},
            {"index": 2, "name": "task-b", "memory_gb": 2},
        ],
    }
    (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    completed_task = {
        "index": 1,
        "name": "task-a",
        "status": "completed",
        "successful_attempt": "001",
        "attempt_count": 1,
        "benchmark_completion": {"status": "passed"},
        "benchmark_task_passed": True,
        "integration_validation": {"status": "passed", "errors": []},
        "phoenix_upload": "passed",
    }
    second_task = {
        "index": 2,
        "name": "task-b",
        "status": "completed" if complete else "pending",
        "successful_attempt": "001" if complete else None,
        "attempt_count": 1 if complete else 0,
        "benchmark_completion": {"status": "passed"} if complete else {},
        "benchmark_task_passed": False if complete else None,
        "integration_validation": {"status": "passed", "errors": []} if complete else {},
        "phoenix_upload": "passed" if complete else None,
    }
    summary = {
        "status": "passed" if complete else "partial",
        "planned_tasks": 2,
        "completed_tasks": 2 if complete else 1,
        "evaluation_cohort": "synthetic-cohort",
        "phoenix_project": "synthetic-project",
        "tasks": [completed_task, second_task],
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    config_root = root / "setup-runtime" / "runtime"
    config_root.mkdir(parents=True)
    config_root.joinpath("plugins.toml").write_text(
        """
[[components]]
kind = "pricing"
[[components.config.sources]]
type = "inline"
[components.config.sources.catalog]
version = 1
[[components.config.sources.catalog.entries]]
provider = "openai"
model_id = "strong-model"
currency = "USD"
unit = "per_token"
pricing_as_of = "2026-01-01"
pricing_source = "synthetic"
[components.config.sources.catalog.entries.rates]
input_per_million = 5.0
output_per_million = 25.0
cache_read_per_million = 0.5
cache_write_per_million = 6.25
[components.config.sources.catalog.entries.prompt_cache]
read_accounting = "included_in_prompt_tokens"
[[components.config.sources.catalog.entries]]
provider = "openai"
model_id = "weak-model"
currency = "USD"
unit = "per_token"
pricing_as_of = "2026-01-01"
pricing_source = "synthetic"
[components.config.sources.catalog.entries.rates]
input_per_million = 0.6
output_per_million = 2.4
cache_read_per_million = 0.119
cache_write_per_million = 0.119
[components.config.sources.catalog.entries.prompt_cache]
read_accounting = "included_in_prompt_tokens"
[[plugins.dynamic]]
manifest = "/plugin.toml"
[plugins.dynamic.config]
version = 2
max_retries = 1
[plugins.dynamic.config.algorithm]
kind = "classifier"
[plugins.dynamic.config.targets.strong]
model = "strong-model"
protocol = "openai_chat"
[plugins.dynamic.config.targets.weak]
model = "weak-model"
protocol = "openai_chat"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    schema_root = config_root / "switchyard-plugin"
    schema_root.mkdir()
    schema_root.joinpath("config.schema.json").write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {
                    "version": {"const": 2},
                    "priority": {"type": "integer", "default": 0},
                    "max_retries": {"type": "integer", "default": 3},
                    "algorithm": {
                        "oneOf": [
                            {
                                "properties": {
                                    "kind": {"const": "classifier"},
                                    "threshold_step": {"type": "number", "default": 0},
                                    "max_output_tokens": {"type": "integer", "default": 4096},
                                    "prompt": {"type": "string"},
                                }
                            }
                        ]
                    },
                    "targets": {
                        "additionalProperties": {
                            "properties": {
                                "model": {"type": "string"},
                                "protocol": {"type": "string"},
                                "base_url": {"type": "string"},
                                "weight": {"type": "number", "default": 1},
                                "drop_caller_extra_body": {"type": "boolean", "default": False},
                            }
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    for index, name in ((1, "task-a"), (2, "task-b")):
        if index == 2 and not complete:
            continue
        attempt = root / "tasks" / f"{index:03d}-{name}" / "attempts" / "001"
        telemetry = attempt / "telemetry"
        telemetry.mkdir(parents=True)
        span = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": f"trace-{index}",
                                    "spanId": f"span-{index}",
                                    "name": "openai.chat_completions",
                                    "attributes": [
                                        {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
                                        {"key": "nemo_relay.end.data.model", "value": {"stringValue": "weak-model"}},
                                        {"key": "llm.token_count.prompt", "value": {"intValue": "1000"}},
                                        {"key": "llm.token_count.completion", "value": {"intValue": "100"}},
                                        {
                                            "key": "llm.token_count.prompt_details.cache_read",
                                            "value": {"intValue": "100"},
                                        },
                                        {"key": "llm.cost.total", "value": {"doubleValue": 0.0007919}},
                                    ],
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        telemetry.joinpath("trajectory.openinference.json").write_text(json.dumps(span) + "\n", encoding="utf-8")
        atof = attempt / "jobs" / "job" / "trial" / "artifacts" / "logs" / "agent" / "direct-hermes" / "relay"
        atof.mkdir(parents=True)
        events = [
            {
                "uuid": f"llm-{index}",
                "parent_uuid": f"logical-{index}",
                "kind": "scope",
                "category": "llm",
                "scope_category": "start",
                "name": "openai.chat_completions",
            },
            {
                "uuid": f"decision-{index}",
                "parent_uuid": f"logical-{index}",
                "kind": "mark",
                "name": "switchyard.routing.decision",
                "data": {
                    "selected_target": "weak",
                    "routing_tier": "weak",
                    "decision_source": "dimensions",
                    "reasoning": "stage_router selected weak (confidence 0.625)",
                },
            },
            {
                "uuid": f"usage-{index}",
                "parent_uuid": f"llm-{index}",
                "kind": "mark",
                "name": "llm.chunk",
                "data": {
                    "chunk_index": 1,
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 100,
                        "cache_read_tokens": 100,
                    },
                },
            },
        ]
        atof.joinpath("trajectory.atof.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )


def test_interim_run_keeps_performance_and_cost_denominators_separate(tmp_path: Path) -> None:
    module = load_library()
    root = tmp_path / "run"
    write_synthetic_run(root)
    run, tasks, calls = module.analyze_run(root, "trial")

    assert run["status"] == "interim"
    assert run["pass_at_1"] is None
    assert run["observed_accuracy"] == 1.0
    assert run["planned_task_lower_bound"] == 0.5
    assert run["provider_calls"] == 1
    assert run["cost_coverage"] == 1.0
    assert abs(run["actual_cost"] - 0.0007919) < 1e-12
    assert abs(run["counterfactual_baseline_cost"] - 0.00705) < 1e-12
    assert tasks[1]["benchmark_complete"] is False
    assert calls[0]["actual_cost_source"] == "derived_from_recorded_usage"
    assert calls[0]["usage_present"] is True
    assert calls[0]["routing_decision_source"] == "dimensions"
    assert calls[0]["routing_confidence"] == 0.625
    assert calls[0]["telemetry_receipt_present"] is False
    router = run["runtime_profile"]
    assert router["algorithm"]["max_output_tokens"] == 4096
    assert router["algorithm_sources"]["max_output_tokens"] == "schema_default"
    assert router["algorithm_sources"]["prompt"] == "plugin_builtin_not_serialized"
    assert router["targets"]["weak"]["weight"] == 1
    assert run["router_overhead_usage_status"] == "not_applicable"


def test_report_surfaces_quiet_output_reconciliation_receipts(tmp_path: Path) -> None:
    module = load_library()
    root = tmp_path / "run"
    write_synthetic_run(root)
    attempt = root / "tasks" / "001-task-a" / "attempts" / "001"
    attempt.joinpath("summary.json").write_text(
        json.dumps({"validation": {"terminal_quiet_output_completion": True}}),
        encoding="utf-8",
    )
    attempt.joinpath("nonpass-reconciliation.json").write_text("{}\n", encoding="utf-8")

    run, tasks, _ = module.analyze_run(root, "trial")

    assert tasks[0]["terminal_completion_class"] == "quiet_output_reconciled"
    assert tasks[0]["reconciliation_receipt_present"] is True
    assert run["terminal_completion_counts"] == {"ordinary": 1, "quiet_output_reconciled": 1}
    assert run["reconciliation_receipt_count"] == 1


def test_report_surfaces_preserved_provider_retry_receipts(tmp_path: Path) -> None:
    module = load_library()
    root = tmp_path / "run"
    write_synthetic_run(root)
    task_root = root / "tasks" / "001-task-a"
    task_root.joinpath("provider-retry-001.json").write_text("{}\n", encoding="utf-8")

    run, tasks, _ = module.analyze_run(root, "trial")

    assert tasks[0]["provider_retry_receipt_count"] == 1
    assert run["provider_retry_receipt_count"] == 1


def test_end_to_end_total_requires_complete_execution_cost_coverage(tmp_path: Path) -> None:
    module = load_library()
    root = tmp_path / "run"
    write_synthetic_run(root, complete=True)

    attempt = root / "tasks" / "002-task-b" / "attempts" / "001"
    atof = next(attempt.rglob("trajectory.atof.jsonl"))
    events = [json.loads(line) for line in atof.read_text(encoding="utf-8").splitlines()]
    atof.write_text(
        "".join(json.dumps(event) + "\n" for event in events if event.get("name") != "llm.chunk"),
        encoding="utf-8",
    )
    telemetry = attempt / "telemetry" / "trajectory.openinference.json"
    document = json.loads(telemetry.read_text(encoding="utf-8"))
    attributes = document["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
    document["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"] = [
        attribute for attribute in attributes if not attribute["key"].startswith("llm.token_count")
    ]
    telemetry.write_text(json.dumps(document) + "\n", encoding="utf-8")

    run, _, _ = module.analyze_run(root, "trial")

    assert run["status"] == "final"
    assert run["cost_coverage"] == 0.5
    assert run["covered_model_cost_including_routing"] == run["actual_cost"]
    assert run["total_cost_including_routing"] is None


def test_classifier_judge_cost_is_not_fabricated_when_usage_is_absent(tmp_path: Path) -> None:
    module = load_library()
    root = tmp_path / "classifier-run"
    write_synthetic_run(root, complete=True)
    config = root / "setup-runtime" / "runtime" / "plugins.toml"
    rendered = config.read_text(encoding="utf-8").replace(
        'kind = "classifier"', 'kind = "llm_classifier"\nclassifier_target = "strong"'
    )
    config.write_text(rendered, encoding="utf-8")

    run, _, _ = module.analyze_run(root, "classifier")

    assert run["router_judge_target"] == "strong"
    assert run["router_judge_model"] == "strong-model"
    assert run["router_overhead_usage_status"] == "unavailable_from_run_artifacts"
    assert run["router_overhead_cost_included"] is False
    assert run["observable_execution_cost"] == run["actual_cost"]


def test_classifier_report_does_not_price_unparsed_overhead_span(tmp_path: Path) -> None:
    module = load_library()
    root = tmp_path / "classifier-overhead-run"
    write_synthetic_run(root, complete=True)
    config = root / "setup-runtime" / "runtime" / "plugins.toml"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            'kind = "classifier"', 'kind = "llm_classifier"\nclassifier_target = "strong"'
        ),
        encoding="utf-8",
    )
    telemetry = root / "tasks" / "001-task-a" / "attempts" / "001" / "telemetry" / "trajectory.openinference.json"
    document = json.loads(telemetry.read_text(encoding="utf-8"))
    document["resourceSpans"][0]["scopeSpans"][0]["spans"].append(
        {
            "traceId": "router-trace",
            "spanId": "router-span",
            "name": "libsy.client_call",
            "attributes": [{"key": "openinference.span.kind", "value": {"stringValue": "LLM"}}],
        }
    )
    telemetry.write_text(json.dumps(document) + "\n", encoding="utf-8")

    run, _, _ = module.analyze_run(root, "classifier")

    assert run["router_overhead_candidate_spans"] == 1
    assert run["router_overhead_usage_status"] == "unavailable_from_run_artifacts"
    assert run["total_cost_including_routing"] is None


def test_classifier_report_prices_routing_llm_call_marks(tmp_path: Path) -> None:
    module = load_library()
    root = tmp_path / "classifier-mark-run"
    write_synthetic_run(root, complete=True)
    config = root / "setup-runtime" / "runtime" / "plugins.toml"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            'kind = "classifier"', 'kind = "llm_classifier"\nclassifier_target = "weak"'
        ),
        encoding="utf-8",
    )
    for index, name in ((1, "task-a"), (2, "task-b")):
        atof = (
            root
            / "tasks"
            / f"{index:03d}-{name}"
            / "attempts"
            / "001"
            / "jobs"
            / "job"
            / "trial"
            / "artifacts"
            / "logs"
            / "agent"
            / "direct-hermes"
            / "relay"
            / "trajectory.atof.jsonl"
        )
        mark = {
            "uuid": f"router-overhead-{index}",
            "parent_uuid": f"logical-{index}",
            "kind": "mark",
            "name": "switchyard.routing.llm_call",
            "data": {
                "algorithm": "stage_router",
                "attempt": 1,
                "call_index": 1,
                "selected_target": "weak",
                "call_role": "judge",
                "outcome": "ok",
                "latency_ms": 5.0,
                "contributes_to_routing_overhead": True,
                "usage": {
                    "input_tokens": 900,
                    "cached_input_tokens": 100,
                    "output_tokens": 40,
                    "reasoning_tokens": 10,
                    "total_tokens": 1050,
                },
            },
        }
        with atof.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(mark) + "\n")

    run, tasks, _ = module.analyze_run(root, "classifier")

    expected_per_mark = module.estimate_cost(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "cache_read_tokens": 100,
            "cache_write_tokens": 0,
        },
        module.pricing_by_model(module.parse_runtime_config(root))["weak-model"],
    )
    assert expected_per_mark is not None
    assert run["router_overhead_usage_status"] == "complete"
    assert run["router_overhead_marks"] == 2
    assert run["router_overhead_prompt_tokens"] == 2000
    assert run["router_overhead_completion_tokens"] == 100
    assert run["router_overhead_reasoning_tokens"] == 20
    assert run["router_overhead_cache_read_tokens"] == 200
    assert run["router_overhead_cost"] == pytest.approx(2 * expected_per_mark)
    assert run["total_cost_including_routing"] == pytest.approx(
        run["actual_cost"] + 2 * expected_per_mark
    )
    assert all(task["router_overhead_marks"] == 1 for task in tasks)


def test_routing_mark_with_unpriced_cache_write_is_cost_uncovered() -> None:
    module = load_library()
    event = {
        "uuid": "cache-write-mark",
        "parent_uuid": "logical",
        "kind": "mark",
        "name": "switchyard.routing.llm_call",
        "data": {
            "selected_target": "judge",
            "call_role": "judge",
            "outcome": "ok",
            "contributes_to_routing_overhead": True,
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 20,
                "output_tokens": 10,
            },
        },
    }
    pricing = {
        "judge-model": {
            "rates": {
                "input_per_million": 0.14,
                "output_per_million": 0.28,
                "cache_read_per_million": 0.028,
                "cache_write_per_million": None,
            },
            "cache_read_accounting": "included_in_prompt_tokens",
        }
    }

    result = module._routing_overhead_from_mark(event, pricing, {"judge": "judge-model"})

    assert result["prompt_tokens"] == 120
    assert result["cache_write_tokens"] == 20
    assert result["cost"] is None
    assert result["cost_covered"] is False


def test_identical_complete_runs_aggregate_with_variance(tmp_path: Path) -> None:
    module = load_library()
    runs = []
    tasks = []
    for index in (1, 2):
        root = tmp_path / f"run-{index}"
        write_synthetic_run(root, complete=True)
        run, run_tasks, _ = module.analyze_run(root, f"run-{index}")
        runs.append(run)
        tasks.extend(run_tasks)
    aggregate = module.aggregate_analysis(runs, tasks, "auto")

    assert aggregate["mode"] == "aggregate"
    assert aggregate["all_runs_final"] is True
    assert aggregate["configuration_groups"][0]["final_pass_at_1_mean"] == 0.5
    assert aggregate["configuration_groups"][0]["final_pass_at_1_sample_sd"] == 0.0
    assert aggregate["repeatability"]["pass_count_distribution"] == {"0": 1, "1": 0, "2": 1}


def test_scientific_signature_excludes_raw_runtime_config_hash() -> None:
    module = load_library()
    plan = {"tasks": [], "inputs": {}}
    left = {"digest": "normalized", "pricing_entries": [], "switchyard": {}, "source_sha256": "left"}
    right = {"digest": "normalized", "pricing_entries": [], "switchyard": {}, "source_sha256": "right"}

    left_config = module._scientific_config(plan, left)
    right_config = module._scientific_config(plan, right)

    assert left_config == right_config
    assert "source_sha256" not in left_config["runtime_profile"]


def test_retry_and_fallback_are_one_costed_logical_call(tmp_path: Path) -> None:
    module = load_library()
    attempt = tmp_path / "attempt"
    telemetry = attempt / "telemetry"
    telemetry.mkdir(parents=True)
    span = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "trace",
                                "spanId": "span",
                                "attributes": [
                                    {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
                                    {"key": "nemo_relay.uuid", "value": {"stringValue": "llm"}},
                                    {"key": "nemo_relay.end.data.model", "value": {"stringValue": "strong-model"}},
                                    {"key": "llm.token_count.prompt", "value": {"intValue": "1000"}},
                                    {"key": "llm.token_count.completion", "value": {"intValue": "100"}},
                                    {"key": "llm.token_count.prompt_details.cache_read", "value": {"intValue": "100"}},
                                    {"key": "llm.cost.total", "value": {"doubleValue": 0.00705}},
                                ],
                            }
                        ]
                    }
                ]
            }
        ],
    }
    telemetry.joinpath("trajectory.openinference.json").write_text(json.dumps(span) + "\n", encoding="utf-8")
    atof = attempt / "jobs" / "job" / "trial" / "artifacts" / "logs" / "agent" / "direct-hermes" / "relay"
    atof.mkdir(parents=True)
    events = [
        {"uuid": "llm", "parent_uuid": "logical", "kind": "scope", "category": "llm", "scope_category": "start"},
        {
            "uuid": "d1",
            "parent_uuid": "logical",
            "kind": "mark",
            "name": "switchyard.routing.decision",
            "data": {"attempt": 1, "selected_target": "weak"},
        },
        {
            "uuid": "d2",
            "parent_uuid": "logical",
            "kind": "mark",
            "name": "switchyard.routing.decision",
            "data": {"attempt": 2, "selected_target": "weak"},
        },
        {
            "uuid": "fb",
            "parent_uuid": "logical",
            "kind": "mark",
            "name": "switchyard.routing.fallback",
            "data": {"selected_target": "strong"},
        },
    ]
    atof.joinpath("trajectory.atof.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    pricing = {
        "strong-model": {
            "currency": "USD",
            "cache_read_accounting": "included_in_prompt_tokens",
            "rates": {
                "input_per_million": 5.0,
                "output_per_million": 25.0,
                "cache_read_per_million": 0.5,
                "cache_write_per_million": 6.25,
            },
        },
        "weak-model": {
            "currency": "USD",
            "cache_read_accounting": "included_in_prompt_tokens",
            "rates": {
                "input_per_million": 0.6,
                "output_per_million": 2.4,
                "cache_read_per_million": 0.119,
                "cache_write_per_million": 0.119,
            },
        },
    }
    calls, evidence = module.parse_atof_calls(
        attempt, "run", "task", pricing, "strong-model", {"strong": "strong-model", "weak": "weak-model"}
    )

    assert len(calls) == 1
    assert evidence["route_decisions"] == 2
    assert evidence["route_retry_decisions"] == 1
    assert evidence["fallback_calls"] == 1
    assert calls[0]["route_target"] == "strong"
    assert calls[0]["usage_source"] == "openinference_fallback"
    assert calls[0]["cost_covered"] is True


def test_direct_run_costs_logical_calls_without_routing_decisions(tmp_path: Path) -> None:
    module = load_library()
    attempt = tmp_path / "attempt"
    atof = attempt / "jobs" / "job" / "trial" / "artifacts" / "logs" / "agent" / "direct-hermes" / "relay"
    atof.mkdir(parents=True)
    events = [
        {
            "uuid": "llm",
            "parent_uuid": "logical",
            "kind": "scope",
            "category": "llm",
            "scope_category": "start",
        },
        {
            "uuid": "usage",
            "parent_uuid": "llm",
            "kind": "mark",
            "name": "llm.chunk",
            "data": {
                "chunk_index": 1,
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "cache_read_tokens": 100},
            },
        },
    ]
    atof.joinpath("trajectory.atof.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    pricing = {
        "strong-model": {
            "currency": "USD",
            "cache_read_accounting": "included_in_prompt_tokens",
            "rates": {
                "input_per_million": 5.0,
                "output_per_million": 25.0,
                "cache_read_per_million": 0.5,
                "cache_write_per_million": 6.25,
            },
        }
    }

    calls, evidence = module.parse_atof_calls(attempt, "control", "task", pricing, "strong-model", {})

    assert len(calls) == 1
    assert calls[0]["route_target"] == "direct"
    assert calls[0]["routing_attempt_count"] == 0
    assert calls[0]["model"] == "strong-model"
    assert calls[0]["actual_cost"] == calls[0]["baseline_cost"]
    assert calls[0]["routing_savings"] == 0.0
    assert evidence["direct_logical_calls"] == 1
    assert evidence["route_decisions"] == 0


def test_paired_inference_requires_identical_task_manifests(tmp_path: Path) -> None:
    module = load_library()
    runs = []
    tasks = []
    for index in (1, 2):
        root = tmp_path / f"run-{index}"
        write_synthetic_run(root, complete=True)
        run, run_tasks, _ = module.analyze_run(root, f"run-{index}")
        runs.append(run)
        tasks.extend(run_tasks)
    runs[1]["task_manifest_signature"] = "different"

    aggregate = module.aggregate_analysis(runs, tasks, "compare")
    comparison = aggregate["pairwise_comparisons"][0]

    assert comparison["paired_inference_available"] is False
    assert comparison["paired_task_count"] == 0
    assert comparison["paired_bootstrap_95"] is None


def test_generator_emits_observed_control_bundle(tmp_path: Path) -> None:
    roots = [tmp_path / "control", tmp_path / "trial"]
    for root in roots:
        write_synthetic_run(root, complete=True)
    output = tmp_path / "report"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "analyze.py"),
            "--run-root",
            str(roots[0]),
            "--label",
            "control",
            "--run-root",
            str(roots[1]),
            "--label",
            "trial",
            "--output-dir",
            str(output),
            "--baseline-run",
            "control",
        ],
        check=True,
    )

    markdown = output.joinpath("README.md").read_text(encoding="utf-8")
    validation = json.loads(output.joinpath("evidence", "report-validation.json").read_text(encoding="utf-8"))
    assert "Observed direct baseline" in markdown
    assert "Cost delta" in markdown
    assert "| Run | Reported model cost | Basis | Observed direct baseline | Cost delta |" in markdown
    assert (
        "| Run | Reported model cost | Basis | Observed direct baseline | Cost delta | "
        "Execution-call coverage |" not in markdown
    )
    assert "Execution-call and routing-overhead coverage are audited" in markdown
    assert "Planned lower bound" not in markdown
    assert "pass-count/planned-task lower bound" not in markdown
    assert "disclosed capability harness-snapshot exception" not in markdown
    assert "Planned-task lower bound" not in output.joinpath("charts", "pass-at-1-by-run.svg").read_text(
        encoding="utf-8"
    )
    assert output.joinpath("task-aggregate-metrics.csv").is_file()
    assert validation["status"] == "passed"


def test_final_performance_cost_partial_disclosure_does_not_claim_incomplete_runs(tmp_path: Path) -> None:
    roots = [tmp_path / "control", tmp_path / "trial"]
    for root in roots:
        write_synthetic_run(root, complete=True)
        config = root / "setup-runtime" / "runtime" / "plugins.toml"
        config.write_text(
            config.read_text(encoding="utf-8").replace(
                'kind = "classifier"', 'kind = "llm_classifier"\nclassifier_target = "strong"'
            ),
            encoding="utf-8",
        )
    output = tmp_path / "cost-partial-report"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "analyze.py"),
            "--run-root",
            str(roots[0]),
            "--label",
            "control",
            "--run-root",
            str(roots[1]),
            "--label",
            "trial",
            "--output-dir",
            str(output),
            "--baseline-run",
            "control",
        ],
        check=True,
    )

    markdown = output.joinpath("README.md").read_text(encoding="utf-8")
    assert "Report status: PERFORMANCE FINAL / COST PARTIAL" in markdown
    assert "Performance is final: all 2 source runs have complete benchmark outcomes" in markdown
    assert "source run is incomplete" not in markdown
    assert "routing-only model token or price coverage is incomplete" in markdown


def test_generator_aggregates_n_run_groups_against_observed_group_baseline(tmp_path: Path) -> None:
    roots = [tmp_path / name for name in ("control-1", "control-2", "trial-1", "trial-2")]
    for root in roots:
        write_synthetic_run(root, complete=True)
    changed_plan = json.loads(roots[3].joinpath("plan.json").read_text(encoding="utf-8"))
    changed_plan["inputs"]["runner_sha256"] = "7" * 64
    changed_plan["inputs"]["runtime_sources_sha256"] = "8" * 64
    roots[3].joinpath("plan.json").write_text(json.dumps(changed_plan), encoding="utf-8")
    compatibility = tmp_path / "compatibility.json"
    compatibility.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "id": "trial",
                        "compatibility_exception": {
                            "allowed_scientific_difference_fields": [
                                "runner_sha256",
                                "runtime_sources_sha256",
                            ],
                            "rationale": "Synthetic test evidence proves these harness-only changes.",
                        },
                    }
                ],
                "excluded_groups": [
                    {
                        "id": "invalid-trial",
                        "reason": "Synthetic incomplete evidence is not admitted.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "grouped-report"
    command = [sys.executable, str(SCRIPT_ROOT / "analyze.py")]
    for root, label, group in zip(
        roots,
        ("control-1", "control-2", "trial-1", "trial-2"),
        ("control", "control", "trial", "trial"),
        strict=True,
    ):
        command.extend(["--run-root", str(root), "--label", label, "--group", group])
    command.extend(
        [
            "--output-dir",
            str(output),
            "--mode",
            "compare",
            "--baseline-group",
            "control",
            "--expected-group-size",
            "2",
            "--group-compatibility-manifest",
            str(compatibility),
        ]
    )

    subprocess.run(command, check=True)

    metrics = json.loads(output.joinpath("aggregate-metrics.json").read_text(encoding="utf-8"))
    markdown = output.joinpath("README.md").read_text(encoding="utf-8")
    validation = json.loads(output.joinpath("evidence", "report-validation.json").read_text(encoding="utf-8"))
    groups = metrics["configuration_group_metrics"]
    comparisons = metrics["aggregate"]["configuration_group_comparisons"]
    assert [group["group"] for group in groups] == ["control", "trial"]
    assert all(group["run_count"] == 2 for group in groups)
    assert groups[1]["configuration_signature_count"] == 2
    assert groups[1]["scientific_difference_fields"] == ["runner_sha256", "runtime_sources_sha256"]
    assert metrics["cost_baseline"]["kind"] == "observed_control_group"
    assert metrics["cost_baseline"]["run_count"] == 2
    assert groups[0]["counterfactual_baseline_cost_mean"] > groups[0]["actual_cost_mean"]
    assert groups[0]["routing_savings_mean"] > 0.0
    assert groups[1]["routing_savings_pct_mean"] > 0.0
    assert "Configuration-level N-run comparison" in markdown
    assert "Observed cost by run group" in markdown
    assert "Mean final pass@1 ± SD" in markdown
    assert "Passes / planned across runs" in markdown
    assert "Same-workload all-expensive-model counterfactual by run group" in markdown
    assert "Mean reported model cost ± SD" in markdown
    assert "Completion-target usage by configuration" in markdown
    assert "Raw call share alone is not a cost share" in markdown
    assert "Audit of observed versus counterfactual savings" in markdown
    assert (
        "The observed covered-cost delta and same-workload counterfactual should not be expected to match"
        in markdown
    )
    assert "Decision-oriented findings" in markdown
    assert "Empirical reported-cost/performance Pareto set" in markdown
    assert output.joinpath("configuration-group-metrics.csv").is_file()
    assert output.joinpath("configuration-group-comparisons.csv").is_file()
    configuration_differences = output.joinpath("configuration-differences.csv").read_text(encoding="utf-8")
    assert "runner_sha256" in configuration_differences
    assert "runtime_sources_sha256" in configuration_differences
    assert comparisons[0]["task_aligned_count"] == 2
    assert comparisons[0]["mean_task_pass_frequency_delta"] == 0.0
    assert comparisons[0]["task_bootstrap_95"] == [0.0, 0.0]
    assert "Task-aligned configuration comparisons" in markdown
    assert "Most configuration-sensitive tasks" in markdown
    assert "Consistently difficult tasks" in markdown
    assert "Declared exclusions" in markdown
    assert "invalid-trial" in markdown
    admission = json.loads(output.joinpath("evidence", "admission.json").read_text(encoding="utf-8"))
    assert admission["declared_exclusions"] == [
        {"id": "invalid-trial", "reason": "Synthetic incomplete evidence is not admitted."}
    ]
    assert output.joinpath("charts", "pass-at-1-by-configuration.svg").is_file()
    assert output.joinpath("charts", "cost-by-configuration.svg").is_file()
    assert output.joinpath("charts", "cost-counterfactual-by-configuration.svg").is_file()
    assert "Planned lower bound" not in markdown
    assert validation["status"] == "passed"
