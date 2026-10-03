#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Generate a structured quantitative report from Terminal-Bench run roots."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_ROOT.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from report_lib import (  # noqa: E402
    AdmissionError,
    aggregate_analysis,
    analyze_run,
    canonical_digest,
    ratio,
    sha256_file,
)
from svg_charts import grouped_bars, mean_bars_with_points, outcome_matrix, stacked_model_bars  # noqa: E402

REPORT_SCHEMA = "terminal-bench-report.bundle.v1"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def _money(value: float | None, currency: str = "USD") -> str:
    return "n/a" if value is None else f"{currency} {value:,.4f}"


def _compact_hash(value: str | None) -> str:
    return str(value)[:12] if value else "missing"


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def task_aggregate_rows(tasks: list[dict[str, Any]], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signatures = {run["label"]: run["configuration_signature"] for run in runs}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task in tasks:
        key = (signatures[task["run_label"]], task["task_name"])
        groups.setdefault(key, []).append(task)
    rows = []
    for (signature, task_name), observations in sorted(groups.items()):
        complete = [task for task in observations if task["benchmark_complete"]]
        costs = [float(task["actual_cost"]) for task in observations if task["provider_calls"] > 0]
        covered = sum(task["cost_covered_calls"] for task in observations)
        calls = sum(task["provider_calls"] for task in observations)
        passes = sum(task["benchmark_passed"] is True for task in complete)
        rows.append(
            {
                "configuration_signature": signature,
                "task_name": task_name,
                "run_count": len(observations),
                "benchmark_complete_runs": len(complete),
                "benchmark_missing_runs": len(observations) - len(complete),
                "benchmark_passes": passes,
                "benchmark_nonpasses": len(complete) - passes,
                "observed_pass_rate": passes / len(complete) if complete else None,
                "planned_run_lower_bound": passes / len(observations),
                "cost_observed_runs": len(costs),
                "estimated_cost_mean": statistics.mean(costs) if costs else None,
                "estimated_cost_sample_sd": statistics.stdev(costs) if len(costs) >= 2 else None,
                "provider_calls": calls,
                "cost_covered_calls": covered,
                "cost_coverage": covered / calls if calls else None,
            }
        )
    return rows


def configuration_group_rows(
    runs: list[dict[str, Any]],
    baseline_group: str | None,
    expected_group_size: int | None,
    compatibility_exceptions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate independent repeated runs within user-declared groups."""

    members_by_group: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        group = str(run.get("analysis_group") or run["configuration_signature"][:12])
        members_by_group.setdefault(group, []).append(run)
    if expected_group_size is not None:
        wrong_sizes = {
            group: len(members) for group, members in members_by_group.items() if len(members) != expected_group_size
        }
        if wrong_sizes:
            raise AdmissionError(
                f"configuration groups do not match --expected-group-size {expected_group_size}: {wrong_sizes}"
            )
    group_identity: dict[str, dict[str, Any]] = {}
    for group, members in members_by_group.items():
        signatures = {run["configuration_signature"] for run in members}
        flattened = {run["label"]: _flatten_configuration(run["scientific_configuration"]) for run in members}
        fields = sorted({field for values in flattened.values() for field in values})
        differing = [
            field
            for field in fields
            if len({json.dumps(values.get(field), sort_keys=True) for values in flattened.values()}) > 1
        ]
        exception = compatibility_exceptions.get(group)
        if len(signatures) != 1:
            allowed = sorted(exception.get("allowed_scientific_difference_fields", [])) if exception else []
            if differing != allowed:
                raise AdmissionError(
                    f"analysis group {group} contains multiple scientific configuration signatures; "
                    f"observed differing fields {differing}, declared compatibility fields {allowed}"
                )
            rationale = exception.get("rationale") if exception else None
            if not isinstance(rationale, str) or not rationale.strip():
                raise AdmissionError(f"analysis group {group} compatibility exception requires a rationale")
        elif exception:
            raise AdmissionError(f"analysis group {group} declares a stale compatibility exception")
        group_identity[group] = {
            "configuration_signature_count": len(signatures),
            "scientific_difference_fields": differing,
            "compatibility_exception": exception,
        }
    if baseline_group is not None and baseline_group not in members_by_group:
        raise AdmissionError("--baseline-group must identify one supplied analysis group")
    currencies = {run["currency"] for run in runs}
    if baseline_group is not None and len(currencies) != 1:
        raise AdmissionError("observed baseline-group comparisons require one shared currency")
    baseline_cost = None
    baseline_total_cost = None
    baseline_covered_cost = None
    if baseline_group is not None:
        baseline_cost = statistics.mean(float(run["actual_cost"]) for run in members_by_group[baseline_group])
        baseline_covered_cost = statistics.mean(
            float(run["covered_model_cost_including_routing"]) for run in members_by_group[baseline_group]
        )
        baseline_totals = [run.get("total_cost_including_routing") for run in members_by_group[baseline_group]]
        if all(value is not None for value in baseline_totals):
            baseline_total_cost = statistics.mean(float(value) for value in baseline_totals)
    rows: list[dict[str, Any]] = []
    for group, members in members_by_group.items():
        pass_rates = [float(run["pass_at_1"]) for run in members if run["pass_at_1"] is not None]
        costs = [float(run["actual_cost"]) for run in members]
        total_costs = [
            float(run["total_cost_including_routing"])
            for run in members
            if run.get("total_cost_including_routing") is not None
        ]
        covered_model_costs = [float(run["covered_model_cost_including_routing"]) for run in members]
        overhead_costs = [float(run["router_overhead_cost"]) for run in members]
        counterfactual_costs = [float(run["counterfactual_baseline_cost"]) for run in members]
        savings = [float(run["routing_savings"]) for run in members]
        savings_pcts = [float(run["routing_savings_pct"]) for run in members]
        total_savings = [
            float(run["total_savings_including_routing"])
            for run in members
            if run.get("total_savings_including_routing") is not None
        ]
        total_savings_pcts = [
            float(run["total_savings_including_routing_pct"])
            for run in members
            if run.get("total_savings_including_routing_pct") is not None
        ]
        cache_ratios = [float(run["cache_read_ratio"]) for run in members if run["cache_read_ratio"] is not None]
        cost_mean = statistics.mean(costs)
        model_counts: dict[str, int] = {}
        model_usage: dict[str, dict[str, float | int]] = {}
        route_target_counts: dict[str, int] = {}
        route_reason_counts: dict[str, int] = {}
        route_decision_source_counts: dict[str, int] = {}
        for run in members:
            for destination, source in (
                (model_counts, run["model_counts"]),
                (route_target_counts, run["route_target_counts"]),
                (route_reason_counts, run["route_reason_counts"]),
                (route_decision_source_counts, run.get("route_decision_source_counts", {})),
            ):
                for key, value in source.items():
                    destination[str(key)] = destination.get(str(key), 0) + int(value)
            for model, source in run["model_usage"].items():
                destination = model_usage.setdefault(
                    str(model),
                    {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 0,
                        "covered_execution_cost": 0.0,
                    },
                )
                for field in (
                    "calls",
                    "prompt_tokens",
                    "completion_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                ):
                    destination[field] += int(source[field])
                destination["covered_execution_cost"] += float(source["covered_execution_cost"])
        total_cost_mean = statistics.mean(total_costs) if len(total_costs) == len(members) else None
        covered_model_cost_mean = statistics.mean(covered_model_costs)
        reported_cost_mean = total_cost_mean if total_cost_mean is not None else covered_model_cost_mean
        reported_costs = total_costs if total_cost_mean is not None else covered_model_costs
        reported_cost_basis = "end_to_end" if total_cost_mean is not None else "covered_lower_bound"
        delta = cost_mean - baseline_cost if baseline_cost is not None else None
        total_delta = (
            total_cost_mean - baseline_total_cost
            if total_cost_mean is not None and baseline_total_cost is not None
            else None
        )
        covered_delta = (
            covered_model_cost_mean - baseline_covered_cost if baseline_covered_cost is not None else None
        )
        reported_delta = total_delta if reported_cost_basis == "end_to_end" else covered_delta
        reported_baseline = (
            baseline_total_cost if reported_cost_basis == "end_to_end" else baseline_covered_cost
        )
        rows.append(
            {
                "group": group,
                "configuration_signature": (
                    members[0]["configuration_signature"]
                    if group_identity[group]["configuration_signature_count"] == 1
                    else None
                ),
                **group_identity[group],
                "run_labels": [run["label"] for run in members],
                "run_count": len(members),
                "all_runs_final": all(run["status"] == "final" for run in members),
                "pass_at_1_mean": statistics.mean(pass_rates) if pass_rates else None,
                "pass_at_1_sample_sd": statistics.stdev(pass_rates) if len(pass_rates) >= 2 else None,
                "pass_at_1_min": min(pass_rates) if pass_rates else None,
                "pass_at_1_max": max(pass_rates) if pass_rates else None,
                "pooled_pass_at_1": ratio(
                    sum(run["benchmark_passes"] for run in members),
                    sum(run["planned_tasks"] for run in members),
                ),
                "actual_cost_mean": cost_mean,
                "actual_cost_sample_sd": statistics.stdev(costs) if len(costs) >= 2 else None,
                "actual_cost_min": min(costs),
                "actual_cost_max": max(costs),
                "observable_execution_cost_mean": cost_mean,
                "observable_execution_cost_sample_sd": statistics.stdev(costs) if len(costs) >= 2 else None,
                "observable_execution_cost_min": min(costs),
                "observable_execution_cost_max": max(costs),
                "total_cost_including_routing_mean": total_cost_mean,
                "total_cost_including_routing_sample_sd": (
                    statistics.stdev(total_costs) if len(total_costs) >= 2 and len(total_costs) == len(members) else None
                ),
                "total_cost_including_routing_min": min(total_costs) if len(total_costs) == len(members) else None,
                "total_cost_including_routing_max": max(total_costs) if len(total_costs) == len(members) else None,
                "router_overhead_cost_mean": statistics.mean(overhead_costs),
                "router_overhead_cost_sample_sd": statistics.stdev(overhead_costs) if len(overhead_costs) >= 2 else None,
                "covered_model_cost_including_routing_mean": covered_model_cost_mean,
                "covered_model_cost_including_routing_sample_sd": (
                    statistics.stdev(covered_model_costs) if len(covered_model_costs) >= 2 else None
                ),
                "reported_model_cost_mean": reported_cost_mean,
                "reported_model_cost_sample_sd": statistics.stdev(reported_costs) if len(reported_costs) >= 2 else None,
                "reported_model_cost_basis": reported_cost_basis,
                "counterfactual_baseline_cost_mean": statistics.mean(counterfactual_costs),
                "counterfactual_baseline_cost_sample_sd": (
                    statistics.stdev(counterfactual_costs) if len(counterfactual_costs) >= 2 else None
                ),
                "routing_savings_mean": statistics.mean(savings),
                "routing_savings_sample_sd": statistics.stdev(savings) if len(savings) >= 2 else None,
                "routing_savings_pct_mean": statistics.mean(savings_pcts),
                "routing_savings_pct_sample_sd": statistics.stdev(savings_pcts) if len(savings_pcts) >= 2 else None,
                "total_savings_including_routing_mean": (
                    statistics.mean(total_savings) if len(total_savings) == len(members) else None
                ),
                "total_savings_including_routing_sample_sd": (
                    statistics.stdev(total_savings)
                    if len(total_savings) >= 2 and len(total_savings) == len(members)
                    else None
                ),
                "total_savings_including_routing_pct_mean": (
                    statistics.mean(total_savings_pcts) if len(total_savings_pcts) == len(members) else None
                ),
                "total_savings_including_routing_pct_sample_sd": (
                    statistics.stdev(total_savings_pcts)
                    if len(total_savings_pcts) >= 2 and len(total_savings_pcts) == len(members)
                    else None
                ),
                "observed_baseline_cost_mean": baseline_cost,
                "observed_baseline_total_cost_mean": baseline_total_cost,
                "actual_cost_delta_vs_baseline": delta,
                "actual_cost_delta_vs_baseline_pct": ratio(delta, baseline_cost),
                "total_cost_delta_vs_baseline": total_delta,
                "total_cost_delta_vs_baseline_pct": ratio(total_delta, baseline_total_cost),
                "covered_model_cost_delta_vs_baseline": covered_delta,
                "covered_model_cost_delta_vs_baseline_pct": ratio(covered_delta, baseline_covered_cost),
                "reported_model_cost_delta_vs_baseline": reported_delta,
                "reported_model_cost_delta_vs_baseline_pct": ratio(reported_delta, reported_baseline),
                "observable_execution_cost_delta_vs_baseline": delta,
                "observable_execution_cost_delta_vs_baseline_pct": ratio(delta, baseline_cost),
                "currency": members[0]["currency"],
                "cost_coverage_min": min(float(run["cost_coverage"] or 0) for run in members),
                "router_judge_models": sorted(
                    {run["router_judge_model"] for run in members if run.get("router_judge_model")}
                ),
                "router_overhead_usage_statuses": sorted({str(run["router_overhead_usage_status"]) for run in members}),
                "router_overhead_cost_complete": all(
                    run["router_overhead_usage_status"] in {"complete", "not_applicable"} for run in members
                ),
                "router_overhead_marks": sum(int(run["router_overhead_marks"]) for run in members),
                "router_overhead_prompt_tokens": sum(
                    int(run["router_overhead_prompt_tokens"]) for run in members
                ),
                "router_overhead_completion_tokens": sum(
                    int(run["router_overhead_completion_tokens"]) for run in members
                ),
                "router_overhead_reasoning_tokens": sum(
                    int(run["router_overhead_reasoning_tokens"]) for run in members
                ),
                "router_overhead_cache_read_tokens": sum(
                    int(run["router_overhead_cache_read_tokens"]) for run in members
                ),
                "router_overhead_cache_write_tokens": sum(
                    int(run["router_overhead_cache_write_tokens"]) for run in members
                ),
                "provider_calls": sum(int(run["provider_calls"]) for run in members),
                "cache_read_ratio_mean": statistics.mean(cache_ratios) if cache_ratios else None,
                "cache_read_ratio_sample_sd": statistics.stdev(cache_ratios) if len(cache_ratios) >= 2 else None,
                "prompt_tokens": sum(int(run["prompt_tokens"]) for run in members),
                "completion_tokens": sum(int(run["completion_tokens"]) for run in members),
                "cache_read_tokens": sum(int(run["cache_read_tokens"]) for run in members),
                "cache_write_tokens": sum(int(run["cache_write_tokens"]) for run in members),
                "model_counts": dict(sorted(model_counts.items())),
                "model_usage": dict(sorted(model_usage.items())),
                "route_target_counts": dict(sorted(route_target_counts.items())),
                "route_reason_counts": dict(sorted(route_reason_counts.items())),
                "route_decision_source_counts": dict(sorted(route_decision_source_counts.items())),
            }
        )
    return rows


def _bootstrap_mean(values: list[float], seed: str, samples: int = 10_000) -> list[float] | None:
    if not values:
        return None
    generator = random.Random(int(seed[:16], 16))
    estimates = []
    for _ in range(samples):
        selected = [values[generator.randrange(len(values))] for _ in values]
        estimates.append(statistics.mean(selected))
    estimates.sort()
    return [estimates[int(samples * 0.025)], estimates[min(samples - 1, int(samples * 0.975))]]


def configuration_group_comparison_rows(
    runs: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    baseline_group: str | None,
) -> list[dict[str, Any]]:
    """Compare repeated configurations using task-aligned group pass frequencies."""

    if baseline_group is None:
        return []
    group_by_run = {run["label"]: str(run.get("analysis_group") or run["configuration_signature"][:12]) for run in runs}
    members_by_group: dict[str, list[str]] = {}
    for run_label, group in group_by_run.items():
        members_by_group.setdefault(group, []).append(run_label)
    by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for task in tasks:
        by_run.setdefault(task["run_label"], {})[task["task_name"]] = task
    baseline_runs = members_by_group[baseline_group]
    rows: list[dict[str, Any]] = []
    for group, trial_runs in members_by_group.items():
        if group == baseline_group:
            continue
        compared_runs = [*baseline_runs, *trial_runs]
        task_names = set().union(*(set(by_run[label]) for label in compared_runs))
        deltas: list[float] = []
        baseline_rates: list[float] = []
        trial_rates: list[float] = []
        incomplete_or_unmatched = 0
        better = equal = worse = 0
        for task_name in sorted(task_names):
            observations = [by_run[label].get(task_name) for label in compared_runs]
            if any(item is None or item["benchmark_passed"] is None for item in observations):
                incomplete_or_unmatched += 1
                continue
            baseline_rate = statistics.mean(
                int(bool(by_run[label][task_name]["benchmark_passed"])) for label in baseline_runs
            )
            trial_rate = statistics.mean(
                int(bool(by_run[label][task_name]["benchmark_passed"])) for label in trial_runs
            )
            delta = trial_rate - baseline_rate
            baseline_rates.append(baseline_rate)
            trial_rates.append(trial_rate)
            deltas.append(delta)
            if delta > 0:
                better += 1
            elif delta < 0:
                worse += 1
            else:
                equal += 1
        seed = canonical_digest([baseline_group, group, sorted(compared_runs), sorted(task_names)])
        rows.append(
            {
                "baseline_group": baseline_group,
                "comparison_group": group,
                "baseline_run_count": len(baseline_runs),
                "comparison_run_count": len(trial_runs),
                "task_aligned_count": len(deltas),
                "incomplete_or_unmatched_task_count": incomplete_or_unmatched,
                "baseline_mean_task_pass_frequency": statistics.mean(baseline_rates) if baseline_rates else None,
                "comparison_mean_task_pass_frequency": statistics.mean(trial_rates) if trial_rates else None,
                "mean_task_pass_frequency_delta": statistics.mean(deltas) if deltas else None,
                "task_bootstrap_95": _bootstrap_mean(deltas, seed),
                "tasks_better_than_baseline": better,
                "tasks_equal_to_baseline": equal,
                "tasks_worse_than_baseline": worse,
                "inference_method": "task bootstrap of group-mean pass-frequency delta",
            }
        )
    return rows


def _validate_labels(labels: list[str]) -> None:
    if len(labels) != len(set(labels)):
        raise AdmissionError("run labels must be unique")
    for label in labels:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", label):
            raise AdmissionError(f"invalid run label: {label}")


def _load_group_compatibility_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    groups = payload.get("groups") if isinstance(payload, dict) else None
    if not isinstance(groups, list):
        raise AdmissionError("group compatibility manifest must contain a groups list")
    result = {}
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("id"), str):
            raise AdmissionError("every compatibility-manifest group must have a string id")
        exception = group.get("compatibility_exception")
        if exception is not None:
            if not isinstance(exception, dict):
                raise AdmissionError(f"group {group['id']} compatibility_exception must be an object")
            fields = exception.get("allowed_scientific_difference_fields")
            if not isinstance(fields, list) or not fields or not all(isinstance(field, str) for field in fields):
                raise AdmissionError(
                    f"group {group['id']} compatibility exception requires nonempty string difference fields"
                )
            if len(fields) != len(set(fields)):
                raise AdmissionError(f"group {group['id']} compatibility difference fields must be unique")
            result[group["id"]] = exception
    return result


def _load_declared_exclusions(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    exclusions = payload.get("excluded_groups", []) if isinstance(payload, dict) else []
    if not isinstance(exclusions, list):
        raise AdmissionError("group compatibility manifest excluded_groups must be a list")
    result = []
    for exclusion in exclusions:
        if not isinstance(exclusion, dict):
            raise AdmissionError("every declared exclusion must be an object")
        identifier = exclusion.get("id")
        reason = exclusion.get("reason")
        if not isinstance(identifier, str) or not identifier or not isinstance(reason, str) or not reason:
            raise AdmissionError("every declared exclusion requires nonempty string id and reason fields")
        result.append({"id": identifier, "reason": reason})
    if len({item["id"] for item in result}) != len(result):
        raise AdmissionError("declared exclusion ids must be unique")
    return result


def _configuration_rows(runs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Run | Dataset | Configuration | Tasks | Status |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for run in runs:
        lines.append(
            f"| {run['label']} | `{run['dataset']}` | `{_compact_hash(run['configuration_signature'])}` | "
            f"{run['planned_tasks']} | {run['status']} |"
        )
    lines.extend(["", "Configured model targets:", ""])
    members_by_group: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        group = str(run.get("analysis_group") or run["configuration_signature"][:12])
        members_by_group.setdefault(group, []).append(run)
    for group, members in members_by_group.items():
        run = members[0]
        targets = run["runtime_profile"].get("targets", {})
        if not targets:
            lines.append(f"- {group} (N={len(members)}): direct `{run['baseline_model']}`; no Switchyard router.")
            continue
        target_models = "; ".join(
            f"{name} `{target.get('model', 'unresolved')}`" for name, target in sorted(targets.items())
        )
        lines.append(f"- {group} (N={len(members)}): {target_models}.")
    return lines


def _declared_exclusion_rows(exclusions: list[dict[str, str]]) -> list[str]:
    if not exclusions:
        return []
    lines = [
        "### Declared exclusions",
        "",
        "The analysis manifest excludes the following cohorts from every statistic and plot:",
        "",
    ]
    lines.extend(f"- **{item['id']}:** {item['reason']}" for item in exclusions)
    return lines


def _flatten_configuration(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in sorted(value.items()):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_configuration(item, path))
        return result
    if isinstance(value, list):
        return {prefix: f"list[{len(value)}] sha256:{canonical_digest(value)[:12]}"}
    return {prefix: value}


def _configuration_difference_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(runs) < 2:
        return []
    members_by_group: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        group = str(run.get("analysis_group") or run["configuration_signature"][:12])
        members_by_group.setdefault(group, []).append(run)
    flattened = {run["label"]: _flatten_configuration(run["scientific_configuration"]) for run in runs}
    fields = sorted({field for values in flattened.values() for field in values})
    differing = [
        field
        for field in fields
        if len({json.dumps(values.get(field), sort_keys=True) for values in flattened.values()}) > 1
    ]
    rows = []
    for field in differing:
        for group, members in members_by_group.items():
            values_by_json = {
                json.dumps(flattened[run["label"]].get(field), sort_keys=True, ensure_ascii=False): flattened[
                    run["label"]
                ].get(field)
                for run in members
            }
            rows.append(
                {
                    "field": field,
                    "configuration_group": group,
                    "run_labels": ",".join(run["label"] for run in members),
                    "distinct_values_within_group": len(values_by_json),
                    "value_json": " || ".join(values_by_json),
                }
            )
    return rows


def _configuration_differences(runs: list[dict[str, Any]]) -> list[str]:
    rows = _configuration_difference_rows(runs)
    if not rows:
        return ["", "No scientific configuration differences were detected across the supplied runs."]
    members_by_group: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        group = str(run.get("analysis_group") or run["configuration_signature"][:12])
        members_by_group.setdefault(group, []).append(run)
    lines = [
        "",
        "### Scientific configuration differences",
        "",
        "The compact summaries below identify the scientific configuration used by each declared group. The complete field-by-field comparison, including exact JSON values and within-group variation, is in `configuration-differences.csv`.",
        "",
    ]
    for group, members in members_by_group.items():
        run = members[0]
        router = run["runtime_profile"]
        algorithm = router.get("algorithm", {})
        targets = router.get("targets", {})
        target_models = ", ".join(
            f"{name}={target.get('model', 'unresolved')}" for name, target in sorted(targets.items())
        )
        mode = router.get("routing_mode", "unresolved")
        algorithm_kind = algorithm.get("kind", "not applicable" if mode == "direct" else "unresolved")
        within_group_signatures = len({member["configuration_signature"] for member in members})
        lines.extend(
            [
                f"- **{group} (N={len(members)}):** mode `{mode}`; algorithm `{algorithm_kind}`; "
                f"targets `{target_models or run['baseline_model']}`; concurrency `{run['scientific_configuration'].get('concurrency')}`; "
                f"{within_group_signatures} scientific provenance signature(s).",
            ]
        )
    return lines


def _setting_value(value: Any) -> str:
    if value is None:
        return "unresolved"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _setting_source(value: str | None) -> str:
    return {
        "configured": "configured",
        "schema_default": "schema default",
        "configured_digest": "configured; prompt stored by digest",
        "plugin_builtin_not_serialized": "library-bound; not serialized",
        "not_exposed_by_schema": "not exposed by schema",
        "configured_names_only": "configured header names",
        "not_configured": "not configured",
        "unresolved": "unresolved",
    }.get(str(value), str(value or "unresolved").replace("_", " "))


def _router_configuration_sections(runs: list[dict[str, Any]]) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        router_digest = canonical_digest(run["runtime_profile"])
        groups.setdefault(router_digest, []).append(run)
    lines = [
        "## Router configuration",
        "",
        "Every value below is resolved from the rendered run configuration and its bundled plugin schema. Schema defaults are distinguished from explicitly configured values. Settings unavailable from those artifacts remain explicit rather than inferred from conversation history.",
    ]
    for signature, members in groups.items():
        run = members[0]
        router = run["runtime_profile"]
        algorithm = router.get("algorithm", {})
        algorithm_sources = router.get("algorithm_sources", {})
        top_sources = router.get("top_level_sources", {})
        labels = ", ".join(member["label"] for member in members)
        if router.get("routing_mode") == "direct":
            lines.extend(
                [
                    "",
                    f"### Configuration `{signature[:12]}`",
                    "",
                    f"Applies to run(s): {labels}.",
                    "",
                    "- Routing mode: `direct`; the rendered configuration contains no dynamic Switchyard plugin.",
                    f"- Provider model: `{run['baseline_model']}`; this is the sole run-bound pricing entry.",
                    "- Router, classifier, and judge: not present.",
                    "- Routing retries, thresholds, and session policy: not applicable.",
                    "",
                    "Bound direct-run evidence:",
                    "",
                    f"- Rendered Relay configuration: `{_compact_hash(run['source'].get('runtime_config_sha256'))}`.",
                    "- Provider calls are reconstructed from `hermes.logical_llm_call` → LLM scope relationships and final ATOF/OpenInference usage; no synthetic Switchyard decision is counted.",
                    "",
                    "Pricing used by cost analysis:",
                    "",
                ]
            )
            for entry in run["pricing_catalog"]:
                rates = entry["rates"]
                lines.append(
                    f"- `{entry['model_id']}`: input `{rates['input_per_million']}`, output "
                    f"`{rates['output_per_million']}`, cache read `{rates['cache_read_per_million']}`, "
                    f"cache write `{rates['cache_write_per_million']}` {entry['currency']} per million tokens; "
                    f"as of `{entry['pricing_as_of']}` ({entry['pricing_source']})."
                )
            continue
        lines.extend(
            [
                "",
                f"### Configuration `{signature[:12]}`",
                "",
                f"Applies to run(s): {labels}.",
                "",
                "| Router setting | Resolved value and provenance |",
                "| --- | --- |",
                f"| Plugin priority | `{_setting_value(router.get('priority'))}` ({_setting_source(top_sources.get('priority'))}) |",
            ]
        )
        excluded_algorithm_fields = {
            "prompt_configured",
            "prompt_length",
            "prompt_sha256",
            "unexposed_generation_controls",
        }
        flattened_algorithm = _flatten_configuration(algorithm)
        for key, value in flattened_algorithm.items():
            if key.split(".", 1)[0] in excluded_algorithm_fields:
                continue
            source_key = key.split(".", 1)[0]
            lines.append(
                f"| Algorithm `{key}` | `{_setting_value(value)}` "
                f"({_setting_source(algorithm_sources.get(source_key))}) |"
            )
        prompt_source = algorithm_sources.get("prompt", "unresolved")
        if algorithm.get("prompt_configured"):
            prompt_description = (
                f"custom prompt, length {algorithm.get('prompt_length')}, "
                f"SHA-256 `{_compact_hash(algorithm.get('prompt_sha256'))}`"
            )
        elif prompt_source == "plugin_builtin_not_serialized":
            prompt_description = "plugin built-in; no override"
        else:
            prompt_description = "unresolved"
        unexposed = algorithm.get("unexposed_generation_controls", [])
        lines.extend(
            [
                f"| Judge prompt | {prompt_description} ({_setting_source(prompt_source)}) |",
                f"| Other judge generation controls | {', '.join(unexposed)}: unavailable (not exposed by schema) |",
                "",
                "The judge prompt text is not reconstructed when the run uses the plugin built-in. Its behavior is instead bound by the Switchyard library digest shown below. `temperature`, `top_p`, and `seed` are reported as unavailable when the bundled classifier schema does not expose them; this does not imply a value of zero.",
                "",
                "| Role | Target | Model |",
                "| --- | --- | --- |",
            ]
        )
        targets = router.get("targets", {})
        judge_target = algorithm.get("classifier_target")
        if judge_target is None and isinstance(algorithm.get("classifier"), dict):
            judge_target = algorithm["classifier"].get("target")
        strong_target = algorithm.get("strong_target") or algorithm.get("capable_target")
        weak_target = algorithm.get("weak_target") or algorithm.get("efficient_target")
        roles = {
            "judge": judge_target or ("judge" if float(targets.get("judge", {}).get("weight", 0) or 0) > 0 else None),
            "strong": strong_target or ("strong" if "strong" in targets else None),
            "weak": weak_target or ("weak" if "weak" in targets else None),
        }
        for role, target_name in roles.items():
            if target_name is None:
                lines.append(f"| {role} | not used by this algorithm | n/a |")
                continue
            target = targets.get(str(target_name), {})
            lines.append(f"| {role} | `{target_name}` | `{target.get('model', 'unresolved')}` |")
        lines.extend(["", "Target transport and request handling:", ""])
        for name, target in sorted(targets.items()):
            lines.append(
                f"- `{name}`: protocol `{target.get('protocol', 'unresolved')}`; "
                f"base URL `{target.get('base_url', 'unresolved')}`; "
                f"extra body `{_setting_value(target.get('extra_body'))}`."
            )
        scientific = run["scientific_configuration"]
        lines.extend(
            [
                "",
                "Bound router evidence:",
                "",
                f"- Rendered plugin configuration: `{_compact_hash(run['source'].get('runtime_config_sha256'))}`.",
                f"- Plugin configuration schema: `{_compact_hash(router.get('config_schema_sha256'))}`.",
                f"- Switchyard library: `{_compact_hash(scientific.get('switchyard_library_sha256'))}`.",
                f"- Switchyard manifest: `{_compact_hash(scientific.get('switchyard_manifest_sha256'))}`.",
                "",
                "Pricing bound to observable execution-call cost analysis:",
                "",
            ]
        )
        for entry in run["pricing_catalog"]:
            rates = entry["rates"]
            lines.append(
                f"- `{entry['model_id']}`: input `{rates['input_per_million']}`, output "
                f"`{rates['output_per_million']}`, cache read `{rates['cache_read_per_million']}`, "
                f"cache write `{rates['cache_write_per_million']}` {entry['currency']} per million tokens; "
                f"as of `{entry['pricing_as_of']}` ({entry['pricing_source']})."
            )
    return lines


def _executive_rows(
    runs: list[dict[str, Any]], groups: list[dict[str, Any]], cost_baseline: dict[str, Any]
) -> list[str]:
    all_final = all(run["status"] == "final" for run in runs)
    lines = ["### Performance summary", ""]
    repeated_groups = any(group["run_count"] > 1 for group in groups)
    if all_final and repeated_groups:
        lines.extend(
            [
                "| Group | N | Mean final pass@1 ± SD | Passes / planned across runs |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for group in groups:
            group_runs = [run for run in runs if run.get("analysis_group") == group["group"]]
            lines.append(
                f"| {group['group']} | {group['run_count']} | "
                f"{_percent(group['pass_at_1_mean'])} ± {_percent(group['pass_at_1_sample_sd'])} | "
                f"{sum(run['benchmark_passes'] for run in group_runs)}/"
                f"{sum(run['planned_tasks'] for run in group_runs)} |"
            )
    elif all_final:
        lines.extend(["| Run | Final pass@1 | Passes / planned |", "| --- | ---: | ---: |"])
        for run in runs:
            lines.append(
                f"| {run['label']} | {_percent(run['pass_at_1'])} | {run['benchmark_passes']}/{run['planned_tasks']} |"
            )
    else:
        lines.extend(
            [
                "| Run | Final pass@1 | Observed accuracy | Planned lower bound |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for run in runs:
            performance = _percent(run["pass_at_1"]) if run["pass_at_1"] is not None else "interim"
            lines.append(
                f"| {run['label']} | {performance} | {run['benchmark_passes']}/{run['benchmark_complete_tasks']} "
                f"({_percent(run['observed_accuracy'])}) | {run['benchmark_passes']}/{run['planned_tasks']} "
                f"({_percent(run['planned_task_lower_bound'])}) |"
            )
    lines.extend(
        [
            "",
            "### Cost summary",
            "",
            "Two cost comparisons are reported because they answer different questions. The observed-control comparison uses end-to-end model cost (served completions plus recorded routing-only calls) when coverage is complete; otherwise it uses an explicitly labeled covered-cost lower bound. The same-workload counterfactual reprices each routed run's covered served calls entirely as its configured expensive model and compares that value with the same reported-cost basis.",
            "",
        ]
    )
    if cost_baseline["kind"] == "observed_control_group":
        lines.extend(
            [
                "#### Observed cost by run group",
                "",
                "| Group | N | Mean reported model cost ± SD | Basis | Delta vs direct mean |",
                "| --- | ---: | ---: | --- | ---: |",
            ]
        )
        baseline_group = cost_baseline.get("group")
        for group in groups:
            delta = (
                "observed baseline"
                if group["group"] == baseline_group
                else f"{_money(group['reported_model_cost_delta_vs_baseline'], group['currency'])} "
                f"({_percent(group['reported_model_cost_delta_vs_baseline_pct'])})"
            )
            lines.append(
                f"| {group['group']} | {group['run_count']} | "
                f"{_money(group['reported_model_cost_mean'], group['currency'])} ± "
                f"{_money(group['reported_model_cost_sample_sd'], group['currency'])} | "
                f"{group['reported_model_cost_basis'].replace('_', ' ')} | {delta} |"
            )
    elif cost_baseline["kind"] == "observed_control_run":
        lines.extend(
            [
                "#### Observed cost by run",
                "",
                "| Run | Reported model cost | Basis | Observed direct baseline | Cost delta |",
                "| --- | ---: | --- | ---: | ---: |",
            ]
        )
        for run in runs:
            end_to_end = run["total_cost_including_routing"]
            reported = end_to_end if end_to_end is not None else run["covered_model_cost_including_routing"]
            basis = "end to end" if end_to_end is not None else "covered lower bound"
            delta = reported - cost_baseline["actual_cost"] if cost_baseline["actual_cost"] is not None else None
            lines.append(
                f"| {run['label']} | {_money(reported, run['currency'])} | {basis} | "
                f"{_money(cost_baseline['actual_cost'], run['currency'])} | "
                f"{_money(delta, run['currency'])} |"
            )
    lines.extend(
        [
            "",
            "#### Same-workload all-expensive-model counterfactual by run group",
            "",
            "| Group | Counterfactual cost mean ± SD | End-to-end savings mean ± SD | Mean savings % ± SD |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for group in groups:
        lines.append(
            f"| {group['group']} | "
            f"{_money(group['counterfactual_baseline_cost_mean'], group['currency'])} ± "
            f"{_money(group['counterfactual_baseline_cost_sample_sd'], group['currency'])} | "
            f"{_money(group['total_savings_including_routing_mean'], group['currency'])} ± "
            f"{_money(group['total_savings_including_routing_sample_sd'], group['currency'])} | "
            f"{_percent(group['total_savings_including_routing_pct_mean'])} ± "
            f"{_percent(group['total_savings_including_routing_pct_sample_sd'])} |"
        )
    lines.extend(
        [
            "",
            "The percentage column is the mean and sample SD of per-run end-to-end savings percentages. Execution-call and routing-overhead coverage are audited in the Cost and routing section.",
        ]
    )
    return lines


def _model_usage_rows(groups: list[dict[str, Any]]) -> list[str]:
    lines = [
        "### Completion-target usage by configuration",
        "",
        "Raw call share alone is not a cost share. Prompt tokens, completion tokens, and catalog-derived covered execution cost are therefore shown beside it.",
        "",
        "| Group | Effective model | Calls | Prompt-token share |",
        "| --- | --- | ---: | ---: |",
    ]
    for group in groups:
        usage = group["model_usage"]
        totals = {
            field: sum(float(model[field]) for model in usage.values())
            for field in ("calls", "prompt_tokens", "completion_tokens", "covered_execution_cost")
        }
        for model, model_usage in usage.items():
            lines.append(
                f"| {group['group']} | `{model}` | "
                f"{int(model_usage['calls'])}/{int(totals['calls'])} "
                f"({_percent(ratio(model_usage['calls'], totals['calls']))}) | "
                f"{_percent(ratio(model_usage['prompt_tokens'], totals['prompt_tokens']))} |"
            )
    lines.extend(
        [
            "",
            "| Group | Effective model | Completion-token share | Covered-cost share |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for group in groups:
        usage = group["model_usage"]
        totals = {
            field: sum(float(model[field]) for model in usage.values())
            for field in ("completion_tokens", "covered_execution_cost")
        }
        for model, model_usage in usage.items():
            lines.append(
                f"| {group['group']} | `{model}` | "
                f"{_percent(ratio(model_usage['completion_tokens'], totals['completion_tokens']))} | "
                f"{_percent(ratio(model_usage['covered_execution_cost'], totals['covered_execution_cost']))} |"
            )
    return lines


def _routing_decision_source_rows(groups: list[dict[str, Any]]) -> list[str]:
    lines = [
        "### Routing decision sources by configuration",
        "",
        "Decision-source counts are request-level evidence. `dimensions` and `override` are signal decisions; `llm-classifier` is the optional judge path; `fall_open` is the configured default tier after an ambiguous signal or unavailable judge.",
        "",
        "| Group | Decision source | Decisions | Share |",
        "| --- | --- | ---: | ---: |",
    ]
    for group in groups:
        counts = group.get("route_decision_source_counts", {})
        total = sum(int(value) for value in counts.values())
        if not counts:
            lines.append(f"| {group['group']} | unavailable | 0 | n/a |")
            continue
        for source, count in counts.items():
            lines.append(f"| {group['group']} | `{source}` | {count} | {_percent(ratio(count, total))} |")
    return lines


def _cost_estimand_audit_rows(
    runs: list[dict[str, Any]], groups: list[dict[str, Any]], cost_baseline: dict[str, Any]
) -> list[str]:
    if cost_baseline.get("kind") != "observed_control_group":
        return []
    baseline_group = str(cost_baseline["group"])
    control = next(group for group in groups if group["group"] == baseline_group)
    baseline_model = runs[0]["baseline_model"]
    price_entries = {entry["model_id"]: entry for run in runs for entry in run["pricing_catalog"]}
    baseline_price = price_entries[baseline_model]
    rates = baseline_price["rates"]
    has_partial_end_to_end = any(
        group["group"] != baseline_group and group["total_savings_including_routing_mean"] is None
        for group in groups
    )
    observed_rows: list[str] = []
    counterfactual_rows: list[str] = []
    for group in groups:
        observed = (
            "observed baseline"
            if group["group"] == baseline_group
            else f"{_money(group['reported_model_cost_delta_vs_baseline'], group['currency'])} "
            f"({_percent(group['reported_model_cost_delta_vs_baseline_pct'])}); "
            f"{group['reported_model_cost_basis'].replace('_', ' ')}"
        )
        if group["group"] == baseline_group:
            savings = "n/a"
        elif group["total_savings_including_routing_mean"] is not None:
            savings = (
                f"{_money(group['total_savings_including_routing_mean'], group['currency'])} "
                f"({_percent(group['total_savings_including_routing_pct_mean'])})"
            )
        else:
            maximum_savings = float(group["counterfactual_baseline_cost_mean"]) - float(
                group["reported_model_cost_mean"]
            )
            savings = f"≤ {_money(maximum_savings, group['currency'])}; partial-cost upper bound"
        observed_rows.append(f"| {group['group']} | {observed} |")
        counterfactual_rows.append(
            f"| {group['group']} | {savings} | {_percent(group['cache_read_ratio_mean'])} |"
        )
    lines = [
        "### Audit of observed versus counterfactual savings",
        "",
        "The observed covered-cost delta and same-workload counterfactual should not be expected to match. Independent configurations generate different trajectories, call counts, token volumes, cache behavior, and routing overhead. The counterfactual reprices each routed run's served-call usage as the expensive model; exact end-to-end savings subtract routing overhead only when its usage and price coverage are complete. With partial coverage, the displayed maximum savings is an upper bound, not a provider-invoice estimate.",
        "",
        "| Group | Observed reported-cost delta vs direct |",
        "| --- | ---: |",
        *observed_rows,
        "",
        "| Group | Same-workload end-to-end savings / upper bound | Cache-read prompt share |",
        "| --- | ---: | ---: |",
        *counterfactual_rows,
    ]
    lines.extend(
        [
            "",
            "Workload-size diagnostics:",
            "",
            "| Group | Mean calls / run | Mean prompt tokens / run |",
            "| --- | ---: | ---: |",
        ]
    )
    for group in groups:
        lines.append(
            f"| {group['group']} | {group['provider_calls'] / group['run_count']:,.1f} | "
            f"{group['prompt_tokens'] / group['run_count']:,.0f} |"
        )
    lines.extend(
        [
            "",
            f"The direct group has a {_percent(control['cache_read_ratio_mean'])} mean cache-read prompt share. Under the bound `{baseline_model}` catalog, cache-read input is priced at {rates['cache_read_per_million']} versus {rates['input_per_million']} {baseline_price['currency']} per million uncached input tokens. That high direct-run cache share materially lowers the observed control cost. This explains why the realized cross-run differences are much smaller than the same-workload routing {'savings upper bounds' if has_partial_end_to_end else 'savings'}; it is evidence of different workload/cache conditions, not a reversal of the lower-priced model's contribution.",
        ]
    )
    return lines


def _configuration_group_section(groups: list[dict[str, Any]], cost_baseline: dict[str, Any]) -> list[str]:
    if not groups:
        return []
    baseline_group = cost_baseline.get("group")
    lines = [
        "## Configuration-level N-run comparison",
        "",
        "Each bar summarizes independent complete runs in one declared configuration group. Circles retain each run-level observation; whiskers are the sample standard deviation. Groups normally require one identical scientific signature; any exact, predeclared provenance exception is shown below and makes the variance estimate include that possible harness effect. With N=3, these variance estimates are descriptive rather than precise population estimates.",
        "",
        "### Performance by configuration",
        "",
        "| Group | N | Provenance | Mean pass@1 ± SD | Range |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for group in groups:
        pass_summary = f"{_percent(group['pass_at_1_mean'])} ± {_percent(group['pass_at_1_sample_sd'])}"
        pass_range = f"{_percent(group['pass_at_1_min'])}–{_percent(group['pass_at_1_max'])}"
        identity = (
            "identical signature"
            if group["configuration_signature_count"] == 1
            else "declared compatible; see limitation"
        )
        lines.append(f"| {group['group']} | {group['run_count']} | {identity} | {pass_summary} | {pass_range} |")
    lines.extend(
        [
            "",
            "### Cost by configuration",
            "",
            "| Group | Mean reported model cost ± SD | Basis | Delta vs baseline |",
            "| --- | ---: | --- | ---: |",
        ]
    )
    for group in groups:
        cost_summary = (
            f"{_money(group['reported_model_cost_mean'], group['currency'])} ± "
            f"{_money(group['reported_model_cost_sample_sd'], group['currency'])}"
        )
        if group["group"] == baseline_group:
            delta = "observed baseline"
        elif group["reported_model_cost_delta_vs_baseline"] is None:
            delta = "n/a"
        else:
            delta = (
                f"{_money(group['reported_model_cost_delta_vs_baseline'], group['currency'])} "
                f"({_percent(group['reported_model_cost_delta_vs_baseline_pct'])})"
            )
        lines.append(
            f"| {group['group']} | {cost_summary} | {group['reported_model_cost_basis'].replace('_', ' ')} | "
            f"{delta} |"
        )
    exceptions = [group for group in groups if group["configuration_signature_count"] > 1]
    for group in exceptions:
        exception = group["compatibility_exception"]
        fields = ", ".join(f"`{field}`" for field in group["scientific_difference_fields"])
        lines.extend(
            [
                "",
                f"**{group['group']} compatibility limitation:** the N={group['run_count']} group spans "
                f"{group['configuration_signature_count']} provenance signatures differing in {fields}. "
                f"{exception['rationale']} Its sample variance can therefore include a harness-snapshot effect.",
            ]
        )
    lines.extend(
        [
            "",
            "![Configuration mean pass@1 with run observations](charts/pass-at-1-by-configuration.svg)",
            "",
            "![Configuration mean cost with run observations](charts/cost-by-configuration.svg)",
        ]
    )
    return lines


def _configuration_findings(
    groups: list[dict[str, Any]], comparisons: list[dict[str, Any]], baseline_group: str | None
) -> list[str]:
    complete = [group for group in groups if group["pass_at_1_mean"] is not None]
    if not complete or baseline_group is None:
        return []
    baseline = next((group for group in complete if group["group"] == baseline_group), None)
    if baseline is None:
        return []
    best_pass = max(float(group["pass_at_1_mean"]) for group in complete)
    cost_complete = complete
    lowest_cost = min(float(group["reported_model_cost_mean"]) for group in cost_complete)
    performance_leaders = [group["group"] for group in complete if float(group["pass_at_1_mean"]) == best_pass]
    cost_leaders = [
        group["group"]
        for group in cost_complete
        if float(group["reported_model_cost_mean"]) == lowest_cost
    ]
    pareto = []
    for candidate in cost_complete:
        candidate_pass = float(candidate["pass_at_1_mean"])
        candidate_cost = float(candidate["reported_model_cost_mean"])
        dominated = any(
            other is not candidate
            and float(other["pass_at_1_mean"]) >= candidate_pass
            and float(other["reported_model_cost_mean"]) <= candidate_cost
            and (
                float(other["pass_at_1_mean"]) > candidate_pass
                or float(other["reported_model_cost_mean"]) < candidate_cost
            )
            for other in cost_complete
        )
        if not dominated:
            pareto.append(candidate["group"])
    comparison_by_group = {item["comparison_group"]: item for item in comparisons}
    partial_costs = any(group["reported_model_cost_basis"] != "end_to_end" for group in cost_complete)
    cost_qualifier = " (covered lower bound for cost-partial groups)" if partial_costs else ""
    lines = [
        "## Decision-oriented findings",
        "",
        f"- Highest mean final pass@1: {', '.join(f'`{label}`' for label in performance_leaders)} at {_percent(best_pass)}.",
        f"- Lowest mean reported model cost{cost_qualifier}: {', '.join(f'`{label}`' for label in cost_leaders)} at {_money(lowest_cost, baseline['currency'])}.",
        f"- Empirical reported-cost/performance Pareto set{cost_qualifier}: {', '.join(f'`{label}`' for label in pareto)}. A configuration is excluded only when another observed group has both at least as high mean pass@1 and no higher mean reported model cost, with one strict improvement.",
        "",
    ]
    if len(complete) != len(groups):
        missing = [group["group"] for group in groups if group["pass_at_1_mean"] is None]
        lines.extend(
            [
                f"This finding set is interim because {', '.join(f'`{label}`' for label in missing)} does not yet have final pass@1 for all N runs.",
                "",
            ]
        )
    lines.extend(
        [
            "### Performance effects",
            "",
            "| Group | Mean pass@1 | Delta vs baseline | Task-bootstrap 95% interval |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for group in complete:
        if group["group"] == baseline_group:
            pass_delta = 0.0
            interval_text = "observed baseline"
        else:
            comparison = comparison_by_group.get(group["group"])
            pass_delta = float(group["pass_at_1_mean"]) - float(baseline["pass_at_1_mean"])
            interval = comparison.get("task_bootstrap_95") if comparison else None
            interval_text = "n/a" if interval is None else f"[{_percent(interval[0])}, {_percent(interval[1])}]"
        lines.append(
            f"| {group['group']} | {_percent(group['pass_at_1_mean'])} | {_percent(pass_delta)} | {interval_text} |"
        )
    lines.extend(
        [
            "",
            "### Cost effects",
            "",
            "| Group | Mean reported model cost | Basis | Delta vs baseline |",
            "| --- | ---: | --- | ---: |",
        ]
    )
    for group in complete:
        cost_delta = float(group["reported_model_cost_mean"]) - float(baseline["reported_model_cost_mean"])
        lines.append(
            f"| {group['group']} | {_money(group['reported_model_cost_mean'], group['currency'])} | "
            f"{group['reported_model_cost_basis'].replace('_', ' ')} | {_money(cost_delta, group['currency'])} |"
        )
    minimum_coverage = min(float(group["cost_coverage_min"]) for group in groups)
    if minimum_coverage < 1.0:
        lines.extend(
            [
                "",
                f"Cost rankings apply to logical execution calls with recorded usable token evidence. Minimum run-level execution-call coverage across the compared groups is {_percent(minimum_coverage)}; uncovered calls are not imputed, so small cost differences should not be interpreted as provider-invoice precision.",
            ]
        )
    exception_qualifier = (
        ", or the disclosed provenance compatibility exception"
        if any(group.get("compatibility_exception") for group in groups)
        else ""
    )
    lines.extend(
        [
            "",
            "These are empirical benchmark conclusions from N-run means. The bootstrap intervals describe task-level outcome variation; they do not remove confounding from model endpoints or routing policy"
            f"{exception_qualifier}.",
        ]
    )
    return lines


def _variance_section(aggregate: dict[str, Any]) -> list[str]:
    lines = ["## Cross-run variation", ""]
    run_count = sum(group["run_count"] for group in aggregate["configuration_groups"])
    if run_count == 1:
        lines.extend(
            [
                "Only one run is included. Cross-run variance and task-repeatability statistics require at least two independent runs and are therefore unavailable.",
                "",
            ]
        )
    elif len(aggregate["configuration_groups"]) == 1:
        group = aggregate["configuration_groups"][0]
        lines.extend(
            [
                "Runs share one scientific configuration and are aggregated as repeated observations.",
                "",
                f"- Mean observed accuracy: {_percent(group['observed_accuracy_mean'])}",
                f"- Sample standard deviation across runs: {_percent(group['observed_accuracy_sample_sd'])}",
                f"- Minimum / maximum observed accuracy: {_percent(group['observed_accuracy_min'])} / {_percent(group['observed_accuracy_max'])}",
                f"- Pooled trial-weighted observed accuracy: {_percent(group['pooled_observed_accuracy'])}",
            ]
        )
    else:
        lines.extend(
            [
                "Runs do not all share one scientific configuration. Results are presented as a configuration comparison; pooled headline accuracy is intentionally omitted.",
                "",
            ]
        )
    group_comparisons = aggregate.get("configuration_group_comparisons", [])
    if group_comparisons:
        lines.extend(
            [
                "### Task-aligned configuration comparisons",
                "",
                "Each comparison first averages the N independent binary outcomes for a task within each configuration, then subtracts the observed baseline task frequency. The interval is a deterministic bootstrap over tasks. Because group means can be fractional and replicate indices are not natural pairs, McNemar's binary paired test is not applied to these configuration-level effects.",
                "",
                "| Comparison vs baseline | Aligned / missing tasks | Mean pass-frequency delta | Bootstrap 95% interval |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for item in group_comparisons:
            interval = item["task_bootstrap_95"]
            interval_text = "n/a" if interval is None else f"[{_percent(interval[0])}, {_percent(interval[1])}]"
            lines.append(
                f"| {item['comparison_group']} vs {item['baseline_group']} | {item['task_aligned_count']} / "
                f"{item['incomplete_or_unmatched_task_count']} | {_percent(item['mean_task_pass_frequency_delta'])} | "
                f"{interval_text} |"
            )
        lines.extend(
            [
                "",
                "The complete run-to-run binary comparisons remain in `aggregate-metrics.json` for auditability; they are not promoted to the narrative because enumerating arbitrary replicate pairings would overstate their scientific role.",
            ]
        )
        return lines
    repeatability = aggregate.get("repeatability")
    if repeatability:
        distribution = repeatability["pass_count_distribution"]
        lines.extend(
            [
                "",
                f"Task repeatability is available for {repeatability['complete_task_count']} task(s); "
                f"{repeatability['incomplete_task_count']} task(s) have at least one missing observation.",
                "",
                "| Passes across runs | Task count |",
                "| ---: | ---: |",
                *[
                    f"| {count}/{repeatability['run_count']} | {distribution[str(count)]} |"
                    for count in range(repeatability["run_count"] + 1)
                ],
            ]
        )
    if aggregate["pairwise_comparisons"]:
        lines.extend(
            [
                "",
                "### Paired task comparisons",
                "",
                "| Left | Right | Pairing | Paired tasks | Incomplete / unmatched | Accuracy delta (right-left) | Bootstrap 95% interval | McNemar exact p |",
                "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
            ]
        )
        for item in aggregate["pairwise_comparisons"]:
            interval = item["paired_bootstrap_95"]
            interval_text = "n/a" if interval is None else f"[{_percent(interval[0])}, {_percent(interval[1])}]"
            p_value = item["mcnemar_exact_p"]
            pairing = "available" if item["paired_inference_available"] else item["paired_inference_unavailable_reason"]
            unavailable = (
                item["incomplete_common_task_count"] + item["left_only_task_count"] + item["right_only_task_count"]
            )
            lines.append(
                f"| {item['left']} | {item['right']} | {pairing} | {item['paired_task_count']} | {unavailable} | "
                f"{_percent(item['pass_rate_delta_right_minus_left'])} | {interval_text} | "
                f"{'n/a' if p_value is None else f'{p_value:.4f}'} |"
            )
    return lines


def _performance_rows(runs: list[dict[str, Any]]) -> list[str]:
    if all(run["status"] == "final" for run in runs):
        lines = [
            "| Run | Passes / planned | Wilson 95% interval | Final pass@1 |",
            "| --- | ---: | --- | ---: |",
        ]
        for run in runs:
            interval = run["observed_accuracy_wilson_95"]
            interval_text = "n/a" if interval is None else f"[{_percent(interval[0])}, {_percent(interval[1])}]"
            lines.append(
                f"| {run['label']} | {run['benchmark_passes']}/{run['planned_tasks']} | "
                f"{interval_text} | {_percent(run['pass_at_1'])} |"
            )
        return lines
    lines = [
        "| Run | Passes / complete | Wilson 95% interval | Passes / planned | Final pass@1 |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for run in runs:
        interval = run["observed_accuracy_wilson_95"]
        interval_text = "n/a" if interval is None else f"[{_percent(interval[0])}, {_percent(interval[1])}]"
        final = _percent(run["pass_at_1"]) if run["pass_at_1"] is not None else "not emitted (incomplete)"
        lines.append(
            f"| {run['label']} | {run['benchmark_passes']}/{run['benchmark_complete_tasks']} "
            f"({_percent(run['observed_accuracy'])}) | {interval_text} | "
            f"{run['benchmark_passes']}/{run['planned_tasks']} "
            f"({_percent(run['planned_task_lower_bound'])}) | {final} |"
        )
    return lines


def _task_breakdown(tasks: list[dict[str, Any]], runs: list[dict[str, Any]]) -> list[str]:
    group_order = list(
        dict.fromkeys(str(run.get("analysis_group") or run["configuration_signature"][:12]) for run in runs)
    )
    run_group = {run["label"]: str(run.get("analysis_group") or run["configuration_signature"][:12]) for run in runs}
    group_sizes = {group: sum(value == group for value in run_group.values()) for group in group_order}
    observations: dict[str, dict[str, list[bool | None]]] = {}
    for task in tasks:
        group = run_group[task["run_label"]]
        result = task["benchmark_passed"] if task["benchmark_complete"] else None
        observations.setdefault(task["task_name"], {}).setdefault(group, []).append(result)

    def task_summary(task_name: str) -> dict[str, Any]:
        cells = {}
        frequencies = []
        total_passes = 0
        total_complete = 0
        for group in group_order:
            values = observations.get(task_name, {}).get(group, [])
            complete = [value for value in values if value is not None]
            passes = sum(value is True for value in complete)
            missing = group_sizes[group] - len(complete)
            total_passes += passes
            total_complete += len(complete)
            frequencies.append(passes / len(complete) if complete else None)
            cells[group] = f"{passes}/{len(complete)}" + (f"; {missing} missing" if missing else "")
        available = [value for value in frequencies if value is not None]
        spread = max(available) - min(available) if len(available) >= 2 else 0.0
        return {
            "task_name": task_name,
            "cells": cells,
            "spread": spread,
            "total_passes": total_passes,
            "total_complete": total_complete,
        }

    summaries = [task_summary(task_name) for task_name in sorted(observations)]
    sensitive = sorted(summaries, key=lambda item: (-item["spread"], item["total_passes"], item["task_name"]))[:15]
    hard = sorted(
        summaries, key=lambda item: (item["total_passes"] / max(1, item["total_complete"]), item["task_name"])
    )[:10]

    def table(items: list[dict[str, Any]]) -> list[str]:
        result = [
            "| Task | " + " | ".join(group_order) + " | Spread |",
            "| --- | " + " | ".join("---:" for _ in group_order) + " | ---: |",
        ]
        for item in items:
            result.append(
                f"| `{item['task_name']}` | "
                + " | ".join(item["cells"][group] for group in group_order)
                + f" | {_percent(item['spread'])} |"
            )
        return result

    lines = [
        "## Task-level findings",
        "",
        "The tables report passes / benchmark-complete observations for each configuration. Missing observations are shown explicitly. Full run/task evidence remains in `task-metrics.csv`; within-configuration repeatability and cost variation are in `task-aggregate-metrics.csv`.",
        "",
        "### Most configuration-sensitive tasks",
        "",
        "These tasks have the largest spread in pass frequency across configurations (up to 15 shown).",
        "",
        *table(sensitive),
        "",
        "### Consistently difficult tasks",
        "",
        "These tasks have the lowest pooled pass frequency across all supplied runs (up to 10 shown).",
        "",
        *table(hard),
    ]
    return lines


def _cost_reconciliation_rows(runs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Run | Logical calls | Route attempts | Retry attempts | Fallbacks |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        lines.append(
            f"| {run['label']} | {run['provider_calls']} | {run['route_decisions']} | "
            f"{run['route_retry_decisions']} | {run['fallback_calls']} |"
        )
    lines.extend(
        [
            "",
            "| Run | Usage coverage | OpenInference receipts | Recorded-cost mismatches |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for run in runs:
        lines.append(
            f"| {run['label']} | "
            f"{run['usage_covered_calls']}/{run['provider_calls']} ({_percent(run['usage_coverage'])}) | "
            f"{run['telemetry_receipted_calls']}/{run['provider_calls']} "
            f"({_percent(run['telemetry_receipt_coverage'])}) | "
            f"{run['telemetry_cost_mismatch_calls']}/{run['telemetry_cost_comparable_calls']} |"
        )
    return lines


def _cost_baseline_prose(cost_baseline: dict[str, Any]) -> list[str]:
    if cost_baseline["kind"] == "observed_control_group":
        if cost_baseline.get("cost_basis") == "covered_lower_bound":
            return [
                f"The primary cost baseline is the mean covered model-cost lower bound across the {cost_baseline['run_count']} independent direct runs in control group `{cost_baseline['group']}`. Group deltas compare like-for-like covered-cost means. Missing execution or routing usage is not imputed, so these monetary values are lower bounds rather than exact end-to-end totals.",
                "",
                "The per-run all-expensive-model counterfactual remains available in `aggregate-metrics.json` for secondary routing analysis but is not the executive cost baseline.",
            ]
        return [
            f"The primary cost baseline is the mean observed end-to-end model cost across the {cost_baseline['run_count']} independent direct runs in control group `{cost_baseline['group']}`. Group deltas compare configuration-level means; run-level deltas use the same control mean. Routed totals include serving calls and every routing-only call represented by a `switchyard.routing.llm_call` usage mark.",
            "",
            "The per-run all-expensive-model counterfactual remains available in `aggregate-metrics.json` for secondary routing analysis but is not the executive cost baseline.",
        ]
    if cost_baseline["kind"] == "observed_control_run":
        return [
            f"The primary cost baseline is the observed end-to-end model cost for control run `{cost_baseline['run_label']}`. Each other run's delta is a run-level comparison. Routed totals include serving calls and recorded routing-only model calls.",
            "",
            "The per-run all-expensive-model counterfactual remains available in `aggregate-metrics.json` for secondary routing analysis but is not the executive cost baseline.",
        ]
    return [
        "The expensive-model baseline reprices the same covered execution calls with the configured strong-model rate. The resulting difference estimates completion-target selection only. Cache use is priced according to the bound catalog but is not added as a separate savings category.",
    ]


def _router_overhead_cost_disclosure(runs: list[dict[str, Any]]) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        group = str(run.get("analysis_group") or run["configuration_signature"][:12])
        groups.setdefault(group, []).append(run)
    lines = [
        "### Routing-only token and cost accounting",
        "",
        "Switchyard emits one `switchyard.routing.llm_call` ATOF mark for every completed routing-only model call and deliberately excludes the successful serving call already represented by Relay's outer LLM event. The totals below can therefore be added without double-counting. Completion tokens include separately reported reasoning tokens; reasoning is also shown independently for auditability.",
        "",
        "| Group | N | Judge |",
        "| --- | ---: | --- |",
    ]
    summary_rows: list[str] = []
    for group, members in groups.items():
        judge_models = {run.get("router_judge_model") for run in members if run.get("router_judge_model")}
        statuses = {run.get("router_overhead_usage_status") for run in members}
        costs = [float(run.get("router_overhead_cost", 0.0)) for run in members]
        model = ", ".join(f"`{value}`" for value in sorted(judge_models)) if judge_models else "none"
        coverage = "complete" if statuses.issubset({"complete", "not_applicable"}) else ", ".join(sorted(statuses))
        lines.append(f"| {group} | {len(members)} | {model} |")
        summary_rows.append(
            f"| {group} | "
            f"{sum(int(run.get('router_overhead_marks', 0)) for run in members):,} | "
            f"{_money(statistics.mean(costs), members[0]['currency'])} ± "
            f"{_money(statistics.stdev(costs) if len(costs) >= 2 else None, members[0]['currency'])} | "
            f"{coverage} |"
        )
    lines.extend(
        [
            "",
            "| Group | Routing calls | Mean router cost ± SD | Coverage |",
            "| --- | ---: | ---: | --- |",
            *summary_rows,
            "",
            "Routing-only tokens aggregated across the N runs:",
            "",
            "| Group | Prompt | Completion | Reasoning | Cache read | Cache write |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, members in groups.items():
        lines.append(
            f"| {group} | {sum(int(run.get('router_overhead_prompt_tokens', 0)) for run in members):,} | "
            f"{sum(int(run.get('router_overhead_completion_tokens', 0)) for run in members):,} | "
            f"{sum(int(run.get('router_overhead_reasoning_tokens', 0)) for run in members):,} | "
            f"{sum(int(run.get('router_overhead_cache_read_tokens', 0)) for run in members):,} | "
            f"{sum(int(run.get('router_overhead_cache_write_tokens', 0)) for run in members):,} |"
        )
    lines.extend(
        [
            "",
            "A routed total is admitted only when every routing-only mark has both usage and a complete run-bound price. If a provider reports cache-write tokens without an advertised cache-write price, coverage becomes partial rather than silently pricing those tokens at zero.",
        ]
    )
    return lines


def render_markdown(
    title: str,
    analysis_request: str,
    runs: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    aggregate: dict[str, Any],
    report_status: str,
    cost_baseline: dict[str, Any],
    configuration_groups: list[dict[str, Any]],
    declared_exclusions: list[dict[str, str]],
) -> str:
    status_label = "PERFORMANCE FINAL / COST PARTIAL" if report_status == "partial" else report_status.upper()
    interim_methodology = (
        ["- Interim runs show observed accuracy over completed tasks and the conservative pass-count/planned-task lower bound separately."]
        if not aggregate["all_runs_final"]
        else []
    )
    lines = [
        f"# {title}",
        "",
        f"**Report status: {status_label}**",
        "",
    ]
    if report_status != "final":
        incomplete_runs = [run for run in runs if run["status"] != "final"]
        minimum_cost_coverage = min(float(run["cost_coverage"]) for run in runs)
        overhead_unavailable = any(
            run["router_overhead_usage_status"] not in {"complete", "not_applicable"} for run in runs
        )
        if incomplete_runs:
            disclosure = (
                f"> This is an interim benchmark report: {len(incomplete_runs)} of {len(runs)} source runs "
                "are incomplete. Performance and cost denominators keep missing observations explicit."
            )
        else:
            cost_reasons = []
            if minimum_cost_coverage < 1.0:
                cost_reasons.append(
                    f"execution-call cost coverage is below 100% (minimum {_percent(minimum_cost_coverage)})"
                )
            if overhead_unavailable:
                cost_reasons.append("routing-only model token or price coverage is incomplete")
            reason = " and ".join(cost_reasons) or "some required cost evidence is unavailable"
            disclosure = (
                f"> Performance is final: all {len(runs)} source runs have complete benchmark outcomes. "
                f"Monetary results remain cost-partial because {reason}. Missing cost evidence is explicit "
                "and is not imputed."
            )
        lines.extend(
            [
                disclosure,
                "",
            ]
        )
    lines.extend(
        [
            "## Executive summary",
            "",
            f"Requested analysis: {analysis_request or 'Resolve the scientifically appropriate analysis from the supplied run configurations.'}",
            "",
            f"Analysis mode: **{aggregate['mode']}**. Cost baseline: **{cost_baseline['description']}**.",
            "",
            *_executive_rows(runs, configuration_groups, cost_baseline),
            "",
            "![Performance by run](charts/pass-at-1-by-run.svg)",
            "",
            "![Reported model cost versus baseline](charts/cost-baseline-vs-actual-by-run.svg)",
            "",
            "![Group mean reported model cost versus same-workload counterfactual](charts/cost-counterfactual-by-configuration.svg)",
            "",
            *_configuration_group_section(configuration_groups, cost_baseline),
            "",
            *_configuration_findings(
                configuration_groups,
                aggregate.get("configuration_group_comparisons", []),
                cost_baseline.get("group"),
            ),
            "",
            "## Compared configurations",
            "",
            *_configuration_rows(runs),
            "",
            *_declared_exclusion_rows(declared_exclusions),
            "",
            "Configuration signatures include dataset/task identity, runtime and plugin source digests, routing algorithm and targets, pricing catalog, timeout policy, concurrency, and task resource policy. Run roots, cohort IDs, Phoenix projects, and telemetry endpoints are excluded from the scientific signature.",
            *_configuration_differences(runs),
            "",
            *_router_configuration_sections(runs),
            "",
            "## Performance detail",
            "",
            *_performance_rows(runs),
            "",
            *_variance_section(aggregate),
            "",
            "![Task outcomes](charts/task-outcomes.svg)",
            "",
            "## Cost and routing",
            "",
            "Serving-call cost is calculated once per agent-facing logical LLM call from its effective ATOF route, recorded usage, and run-bound pricing catalog. The final ATOF usage chunk is preferred; a route-scoped OpenInference receipt is used only when that chunk is absent. Routing-only calls are costed independently from `switchyard.routing.llm_call` marks and then added to form the end-to-end model total. All monetary figures are catalog-derived estimates, not provider invoices or randomized controls.",
            "",
            *_cost_baseline_prose(cost_baseline),
            "",
            *_cost_estimand_audit_rows(runs, configuration_groups, cost_baseline),
            "",
            *_router_overhead_cost_disclosure(runs),
            "",
            *_cost_reconciliation_rows(runs),
            "",
            "A recorded-cost mismatch means the OpenInference `llm.cost.total` receipt differs from repricing that receipt's usage with the run-bound catalog. Headline cost remains catalog-derived so every covered call uses one consistent accounting method; mismatches remain explicit evidence for integration follow-up.",
            "",
            *_model_usage_rows(configuration_groups),
            "",
            *_routing_decision_source_rows(configuration_groups),
            "",
            "![Completion-target selection distribution](charts/model-choice-distribution-by-run.svg)",
            "",
            "## Cache efficiency",
            "",
            "Cache-read efficiency is cache-read prompt tokens divided by total prompt tokens. It is reported diagnostically and never added to routing savings.",
            "",
            "![Cache-read efficiency](charts/cache-efficiency-by-run.svg)",
            "",
            *_task_breakdown(tasks, runs),
            "",
            "## Methodology and limitations",
            "",
            "- Final `pass@1` is emitted only when every planned task has a benchmark-complete outcome.",
            *interim_methodology,
            "- Performance admission is independent from telemetry admission: a verifier result can remain valid even when tracing or upload evidence is incomplete.",
            "- Verifier-backed terminal completion repairs are counted per run in `evidence/reconciliation.json`; each repaired task retains its original validation plus a content-addressed reconciliation receipt.",
            "- Preserved whole-task provider retries are counted separately in `evidence/reconciliation.json`; retry receipts retain the failed-attempt hashes and are never conflated with routing-decision retries.",
            "- Eligible calls are unique ATOF logical LLM calls with routing evidence; repeated decisions are retry attempts, not additional completed calls. Cost coverage requires an unambiguous LLM child, recorded usage, an effective routed model, and complete run-bound prices; uncovered calls are never silently imputed.",
            "- Internal router classifier/judge calls are not agent-facing logical calls. Their dedicated ATOF usage marks exclude the final serving call by construction; the report admits an end-to-end total only when every mark has usable token evidence and a complete price.",
            "- OpenInference receipt coverage and comparable-field mismatches are retained in reconciliation evidence separately from ATOF usage coverage.",
            "- Paired comparisons use common benchmark-complete task observations, deterministic paired bootstrap intervals, and an exact McNemar test.",
            "- Cross-run variability uses sample standard deviation; with very few runs it is descriptive, not a population estimate.",
            "",
            "## Reproducibility artifacts",
            "",
            "- `aggregate-metrics.json`: complete run/configuration statistics",
            "- `run-metrics.csv`: one row per run",
            "- `configuration-group-metrics.csv`: N-run performance, cost, cache, and routing aggregates by declared configuration group",
            "- `configuration-group-comparisons.csv`: task-aligned N-run group effects against the declared observed baseline",
            "- `configuration-differences.csv`: exact field-by-field scientific configuration differences",
            "- `task-metrics.csv`: one row per run/task observation",
            "- `task-aggregate-metrics.csv`: per-task pass frequency, cost variation, and coverage within each scientific configuration",
            "- `call-metrics.csv`: one row per eligible provider call",
            "- `evidence/`: admission, configuration, reconciliation, validation, and bundle manifests",
            "",
            "No credentials, transcripts, managed runtime trees, or absolute source paths are included in this bundle.",
        ]
    )
    return "\n".join(lines) + "\n"


def _sanitized_run(run: dict[str, Any]) -> dict[str, Any]:
    return run


def _scan_bundle(root: Path, forbidden: list[str]) -> list[dict[str, str]]:
    findings = []
    patterns = ["Bearer ", "SWITCHYARD_PROVIDER_AUTHORIZATION=", "/localhome/", "/home/"]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".json", ".csv", ".svg", ".yaml", ".css", ".tex"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in [*patterns, *forbidden]:
            if pattern and pattern in text:
                findings.append({"file": str(path.relative_to(root)), "pattern": pattern})
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", action="append", type=Path, required=True, help="Terminal-Bench cohort root; repeatable"
    )
    parser.add_argument("--label", action="append", default=[], help="sanitized run label; repeat in run-root order")
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="configuration-group label; repeat in run-root order (defaults to the scientific signature)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Terminal-Bench quantitative analysis")
    parser.add_argument("--analysis-request", default="")
    parser.add_argument("--mode", choices=("auto", "aggregate", "compare"), default="auto")
    parser.add_argument("--baseline-model", help="override the expensive counterfactual model for every run")
    parser.add_argument("--baseline-run", help="label of an observed control run for total-cost comparison")
    parser.add_argument(
        "--baseline-group",
        help="configuration group whose mean observed cost is the total-cost comparison baseline",
    )
    parser.add_argument(
        "--expected-group-size",
        type=int,
        help="require every analysis group to contain exactly this many independent runs",
    )
    parser.add_argument(
        "--group-compatibility-manifest",
        type=Path,
        help="manifest declaring exact, justified scientific-signature differences allowed within a group",
    )
    parser.add_argument("--allow-partial", action="store_true", help="emit an interim report from incomplete runs")
    parser.add_argument("--replace", action="store_true", help="replace an existing output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = [path.expanduser().resolve() for path in args.run_root]
    if len(roots) != len(set(roots)):
        raise AdmissionError("run roots must be distinct")
    labels = args.label or [f"run-{index}" for index in range(1, len(roots) + 1)]
    if len(labels) != len(roots):
        raise AdmissionError("--label count must match --run-root count")
    _validate_labels(labels)
    groups = args.group or ["" for _ in roots]
    if len(groups) != len(roots):
        raise AdmissionError("--group count must match --run-root count")
    if any(groups):
        if not all(groups):
            raise AdmissionError("--group must be supplied for every run or omitted for every run")
        for group in groups:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", group):
                raise AdmissionError(f"invalid group label: {group}")
    if args.baseline_run and args.baseline_group:
        raise AdmissionError("--baseline-run and --baseline-group are mutually exclusive")
    if args.expected_group_size is not None and args.expected_group_size < 2:
        raise AdmissionError("--expected-group-size must be at least 2")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        if not args.replace:
            raise AdmissionError(f"output directory already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(mode=0o700, parents=True)
    try:
        runs: list[dict[str, Any]] = []
        tasks: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        for root, label, group in zip(roots, labels, groups, strict=True):
            run, run_tasks, run_calls = analyze_run(root, label, args.baseline_model)
            if group:
                run["analysis_group"] = group
            runs.append(_sanitized_run(run))
            tasks.extend(run_tasks)
            calls.extend(run_calls)
        if not args.allow_partial and any(run["status"] != "final" for run in runs):
            raise AdmissionError("at least one run is incomplete; pass --allow-partial to emit an interim report")
        aggregate = aggregate_analysis(runs, tasks, args.mode)
        task_aggregates = task_aggregate_rows(tasks, runs)
        compatibility_exceptions = _load_group_compatibility_manifest(args.group_compatibility_manifest)
        declared_exclusions = _load_declared_exclusions(args.group_compatibility_manifest)
        configuration_groups = configuration_group_rows(
            runs, args.baseline_group, args.expected_group_size, compatibility_exceptions
        )
        configuration_group_comparisons = configuration_group_comparison_rows(runs, tasks, args.baseline_group)
        aggregate["configuration_group_comparisons"] = configuration_group_comparisons
        if args.baseline_group:
            controls = [item for item in configuration_groups if item["group"] == args.baseline_group]
            if len(controls) != 1:
                raise AdmissionError("--baseline-group must identify exactly one supplied analysis group")
            control = controls[0]
            control_cost_basis = control["reported_model_cost_basis"]
            cost_baseline = {
                "kind": "observed_control_group",
                "group": control["group"],
                "run_labels": control["run_labels"],
                "run_count": control["run_count"],
                "cost_basis": control_cost_basis,
                "description": (
                    f"mean observed end-to-end model cost of control group {control['group']} "
                    f"(N={control['run_count']})"
                    if control_cost_basis == "end_to_end"
                    else f"mean covered model-cost lower bound of control group {control['group']} "
                    f"(N={control['run_count']})"
                ),
                "actual_cost": control["reported_model_cost_mean"],
                "actual_cost_sample_sd": control["reported_model_cost_sample_sd"],
            }
            for run in runs:
                run_reported_cost = (
                    run["total_cost_including_routing"]
                    if run["total_cost_including_routing"] is not None
                    else run["covered_model_cost_including_routing"]
                )
                run["observed_cost_delta_vs_control"] = (
                    run_reported_cost - control["reported_model_cost_mean"]
                )
        elif args.baseline_run:
            controls = [run for run in runs if run["label"] == args.baseline_run]
            if len(controls) != 1:
                raise AdmissionError("--baseline-run must identify exactly one supplied run label")
            control = controls[0]
            if any(run["currency"] != control["currency"] for run in runs):
                raise AdmissionError("observed control comparisons require one shared currency")
            cost_baseline = {
                "kind": "observed_control_run",
                "run_label": control["label"],
                "description": f"observed end-to-end model cost of control run {control['label']}",
                "actual_cost": control["total_cost_including_routing"],
            }
            for run in runs:
                run["observed_cost_delta_vs_control"] = (
                    run["total_cost_including_routing"] - control["total_cost_including_routing"]
                    if run["total_cost_including_routing"] is not None
                    and control["total_cost_including_routing"] is not None
                    else None
                )
        else:
            cost_baseline = {
                "kind": "expensive_model_counterfactual",
                "description": "same covered execution calls repriced entirely with each run's configured expensive model",
                "override_model": args.baseline_model,
            }
        cost_complete = all(run["cost_coverage"] == 1.0 for run in runs)
        router_overhead_complete = all(
            run["router_overhead_usage_status"] in {"complete", "not_applicable"} for run in runs
        )
        report_status = (
            "interim"
            if not aggregate["all_runs_final"]
            else "final"
            if cost_complete and router_overhead_complete
            else "partial"
        )
        charts = output / "charts"
        evidence = output / "evidence"
        charts.mkdir()
        evidence.mkdir()
        performance_series = [("Final pass@1", [run["pass_at_1"] for run in runs])]
        if not aggregate["all_runs_final"]:
            performance_series = [
                ("Observed accuracy", [run["observed_accuracy"] for run in runs]),
                ("Planned-task lower bound", [run["planned_task_lower_bound"] for run in runs]),
            ]
        grouped_bars(
            charts / "pass-at-1-by-run.svg",
            "Performance by run",
            labels,
            performance_series,
            percent=True,
        )
        group_labels = [str(group["group"]) for group in configuration_groups]
        members_by_group = {
            group: [
                run for run in runs if str(run.get("analysis_group") or run["configuration_signature"][:12]) == group
            ]
            for group in group_labels
        }
        mean_bars_with_points(
            charts / "pass-at-1-by-configuration.svg",
            "Mean pass@1 by configuration",
            group_labels,
            [group["pass_at_1_mean"] for group in configuration_groups],
            [group["pass_at_1_sample_sd"] for group in configuration_groups],
            [
                [float(run["pass_at_1"]) for run in members_by_group[group] if run["pass_at_1"] is not None]
                for group in group_labels
            ],
            percent=True,
        )
        mean_bars_with_points(
            charts / "cost-by-configuration.svg",
            "Mean reported model cost by configuration",
            group_labels,
            [group["reported_model_cost_mean"] for group in configuration_groups],
            [group["reported_model_cost_sample_sd"] for group in configuration_groups],
            [
                [
                    float(
                        run["total_cost_including_routing"]
                        if run["total_cost_including_routing"] is not None
                        else run["covered_model_cost_including_routing"]
                    )
                    for run in members_by_group[group]
                ]
                for group in group_labels
            ],
            currency=True,
        )
        grouped_bars(
            charts / "cost-counterfactual-by-configuration.svg",
            "Group mean reported cost versus all-expensive-model counterfactual",
            group_labels,
            [
                (
                    "Reported model cost",
                    [group["reported_model_cost_mean"] for group in configuration_groups],
                ),
                (
                    "Same-workload counterfactual",
                    [group["counterfactual_baseline_cost_mean"] for group in configuration_groups],
                ),
            ],
            currency=True,
        )
        observed_control = cost_baseline["kind"] in {"observed_control_run", "observed_control_group"}
        baseline_values = (
            [cost_baseline["actual_cost"]] * len(runs)
            if observed_control
            else [run["counterfactual_baseline_cost"] for run in runs]
        )
        grouped_bars(
            charts / "cost-baseline-vs-actual-by-run.svg",
            "Reported model cost versus observed direct control"
            if observed_control
            else "End-to-end model cost versus expensive-model counterfactual",
            labels,
            [
                (
                    "Reported model cost",
                    [
                        run["total_cost_including_routing"]
                        if run["total_cost_including_routing"] is not None
                        else run["covered_model_cost_including_routing"]
                        for run in runs
                    ],
                ),
                ("Observed control" if observed_control else "Expensive-model baseline", baseline_values),
            ],
            currency=True,
        )
        stacked_model_bars(charts / "model-choice-distribution-by-run.svg", "Completion-target calls by run", runs)
        grouped_bars(
            charts / "cache-efficiency-by-run.svg",
            "Cache-read prompt-token share",
            labels,
            [("Cache-read ratio", [run["cache_read_ratio"] for run in runs])],
            percent=True,
        )
        outcome_matrix(charts / "task-outcomes.svg", "Task outcomes by run", tasks, labels)
        payload = {
            "schema_version": REPORT_SCHEMA,
            "generated_at": datetime.now(UTC).isoformat(),
            "report_status": report_status,
            "title": args.title,
            "analysis_request": args.analysis_request,
            "cost_baseline": cost_baseline,
            "runs": runs,
            "aggregate": aggregate,
            "configuration_group_metrics": configuration_groups,
            "declared_exclusions": declared_exclusions,
        }
        (output / "aggregate-metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        run_fields = [
            "label",
            "analysis_group",
            "status",
            "dataset",
            "planned_tasks",
            "benchmark_complete_tasks",
            "benchmark_missing_tasks",
            "benchmark_passes",
            "benchmark_nonpasses",
            "pass_at_1",
            "observed_accuracy",
            "planned_task_lower_bound",
            "provider_calls",
            "cost_covered_calls",
            "cost_coverage",
            "usage_covered_calls",
            "usage_coverage",
            "telemetry_receipted_calls",
            "telemetry_receipt_coverage",
            "observable_execution_cost",
            "actual_cost",
            "router_judge_target",
            "router_judge_model",
            "router_overhead_candidate_spans",
            "router_overhead_scanned_attempts",
            "router_overhead_usage_status",
            "router_overhead_marks",
            "router_overhead_usage_covered_marks",
            "router_overhead_usage_coverage",
            "router_overhead_cost_covered_marks",
            "router_overhead_cost_coverage",
            "router_overhead_prompt_tokens",
            "router_overhead_completion_tokens",
            "router_overhead_reasoning_tokens",
            "router_overhead_cache_read_tokens",
            "router_overhead_cache_write_tokens",
            "router_overhead_cost",
            "router_overhead_cost_included",
            "router_overhead_uncovered_marks",
            "covered_model_cost_including_routing",
            "total_cost_including_routing",
            "counterfactual_baseline_cost",
            "routing_savings",
            "routing_savings_pct",
            "total_savings_including_routing",
            "total_savings_including_routing_pct",
            "baseline_model",
            "currency",
            "cache_read_ratio",
            "prompt_tokens",
            "completion_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "model_usage",
            "terminal_completion_counts",
            "reconciliation_receipt_count",
            "provider_retry_receipt_count",
            "configuration_signature",
            "task_manifest_signature",
        ]
        write_csv(output / "run-metrics.csv", runs, run_fields)
        write_csv(output / "configuration-group-metrics.csv", configuration_groups)
        write_csv(output / "configuration-group-comparisons.csv", configuration_group_comparisons)
        write_csv(output / "configuration-differences.csv", _configuration_difference_rows(runs))
        task_fields = [
            "run_label",
            "task_index",
            "task_name",
            "benchmark_complete",
            "benchmark_passed",
            "integration_passed",
            "attempt_count",
            "terminal_completion_class",
            "reconciliation_receipt_present",
            "provider_retry_receipt_count",
            "provider_calls",
            "cost_covered_calls",
            "actual_cost",
            "baseline_cost",
            "routing_savings",
            "router_overhead_marks",
            "router_overhead_usage_covered_marks",
            "router_overhead_cost_covered_marks",
            "router_overhead_cost",
            "covered_model_cost_including_routing",
            "router_overhead_prompt_tokens",
            "router_overhead_completion_tokens",
            "router_overhead_reasoning_tokens",
            "router_overhead_cache_read_tokens",
            "router_overhead_cache_write_tokens",
            "currency",
            "usage_covered_calls",
            "telemetry_receipted_calls",
            "prompt_tokens",
            "completion_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "cache_read_ratio",
            "model_counts",
            "route_target_counts",
            "route_reason_counts",
            "route_decisions",
            "routing_call_delta",
            "route_retry_decisions",
            "fallback_calls",
            "call_evidence",
            "integration_errors",
        ]
        write_csv(output / "task-metrics.csv", tasks, task_fields)
        write_csv(output / "task-aggregate-metrics.csv", task_aggregates)
        write_csv(output / "call-metrics.csv", calls)
        admission = {
            "schema_version": "terminal-bench-report.admission.v1",
            "status": report_status,
            "run_count": len(runs),
            "run_labels": labels,
            "distinct_roots": len(roots) == len(set(roots)),
            "all_runs_final": aggregate["all_runs_final"],
            "same_task_manifest": aggregate["same_task_manifest"],
            "configuration_group_count": len(aggregate["configuration_groups"]),
            "declared_analysis_group_count": len(configuration_groups),
            "expected_analysis_group_size": args.expected_group_size,
            "declared_exclusions": declared_exclusions,
            "source_hashes": {run["label"]: run["source"] for run in runs},
        }
        (evidence / "admission.json").write_text(
            json.dumps(admission, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        configurations = {
            "schema_version": "terminal-bench-report.configurations.v1",
            "runs": {
                run["label"]: {
                    "configuration_signature": run["configuration_signature"],
                    "task_manifest_signature": run["task_manifest_signature"],
                    "scientific_configuration": run["scientific_configuration"],
                }
                for run in runs
            },
        }
        (evidence / "configurations.json").write_text(
            json.dumps(configurations, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        reconciliation = {
            "schema_version": "terminal-bench-report.reconciliation.v1",
            "runs": {
                run["label"]: {
                    "provider_calls": run["provider_calls"],
                    "route_decisions": run["route_decisions"],
                    "route_retry_decisions": run["route_retry_decisions"],
                    "fallback_calls": run["fallback_calls"],
                    "cost_covered_calls": run["cost_covered_calls"],
                    "cost_coverage": run["cost_coverage"],
                    "usage_covered_calls": run["usage_covered_calls"],
                    "usage_coverage": run["usage_coverage"],
                    "telemetry_receipted_calls": run["telemetry_receipted_calls"],
                    "telemetry_receipt_coverage": run["telemetry_receipt_coverage"],
                    "telemetry_usage_comparable_calls": run["telemetry_usage_comparable_calls"],
                    "telemetry_usage_mismatch_calls": run["telemetry_usage_mismatch_calls"],
                    "telemetry_model_comparable_calls": run["telemetry_model_comparable_calls"],
                    "telemetry_model_mismatch_calls": run["telemetry_model_mismatch_calls"],
                    "telemetry_cost_comparable_calls": run["telemetry_cost_comparable_calls"],
                    "telemetry_cost_mismatch_calls": run["telemetry_cost_mismatch_calls"],
                    "telemetry_recorded_cost_total": run["telemetry_recorded_cost_total"],
                    "derived_cost_for_telemetry_comparable_calls": run["derived_cost_for_telemetry_comparable_calls"],
                    "telemetry_recorded_minus_derived_cost": (
                        run["telemetry_recorded_cost_total"] - run["derived_cost_for_telemetry_comparable_calls"]
                    ),
                    "terminal_completion_counts": run["terminal_completion_counts"],
                    "reconciliation_receipt_count": run["reconciliation_receipt_count"],
                    "provider_retry_receipt_count": run["provider_retry_receipt_count"],
                    "observable_execution_cost": run["observable_execution_cost"],
                    "router_judge_target": run["router_judge_target"],
                    "router_judge_model": run["router_judge_model"],
                    "router_overhead_candidate_spans": run["router_overhead_candidate_spans"],
                    "router_overhead_scanned_attempts": run["router_overhead_scanned_attempts"],
                    "router_overhead_usage_status": run["router_overhead_usage_status"],
                    "router_overhead_marks": run["router_overhead_marks"],
                    "router_overhead_usage_covered_marks": run["router_overhead_usage_covered_marks"],
                    "router_overhead_usage_coverage": run["router_overhead_usage_coverage"],
                    "router_overhead_cost_covered_marks": run["router_overhead_cost_covered_marks"],
                    "router_overhead_cost_coverage": run["router_overhead_cost_coverage"],
                    "router_overhead_prompt_tokens": run["router_overhead_prompt_tokens"],
                    "router_overhead_completion_tokens": run["router_overhead_completion_tokens"],
                    "router_overhead_reasoning_tokens": run["router_overhead_reasoning_tokens"],
                    "router_overhead_cache_read_tokens": run["router_overhead_cache_read_tokens"],
                    "router_overhead_cache_write_tokens": run["router_overhead_cache_write_tokens"],
                    "router_overhead_cost": run["router_overhead_cost"],
                    "router_overhead_uncovered_marks": run["router_overhead_uncovered_marks"],
                    "router_overhead_cost_included": run["router_overhead_cost_included"],
                    "actual_cost": run["actual_cost"],
                    "covered_model_cost_including_routing": run["covered_model_cost_including_routing"],
                    "total_cost_including_routing": run["total_cost_including_routing"],
                    "baseline_cost": run["counterfactual_baseline_cost"],
                    "routing_savings": run["routing_savings"],
                    "baseline_minus_actual": run["counterfactual_baseline_cost"] - run["actual_cost"],
                    "reconciles": abs(run["counterfactual_baseline_cost"] - run["actual_cost"] - run["routing_savings"])
                    <= 1e-6,
                    "end_to_end_reconciles": (
                        run["total_cost_including_routing"] is not None
                        and abs(
                            run["actual_cost"]
                            + run["router_overhead_cost"]
                            - run["total_cost_including_routing"]
                        )
                        <= 1e-6
                    ),
                    "routing_calls_reconcile": (
                        run["provider_calls"] + run["route_retry_decisions"] == run["route_decisions"]
                        if run["runtime_profile"].get("targets")
                        else run["route_decisions"] == 0
                    ),
                }
                for run in runs
            },
        }
        (evidence / "reconciliation.json").write_text(
            json.dumps(reconciliation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        markdown = render_markdown(
            args.title,
            args.analysis_request,
            runs,
            tasks,
            aggregate,
            report_status,
            cost_baseline,
            configuration_groups,
            declared_exclusions,
        )
        (output / "README.md").write_text(markdown, encoding="utf-8")
        shutil.copyfile(SKILL_ROOT / "assets" / "report.css", output / "report.css")
        shutil.copyfile(SKILL_ROOT / "assets" / "pandoc-defaults.yaml", output / "pandoc-defaults.yaml")
        shutil.copyfile(SKILL_ROOT / "assets" / "report-header.tex", output / "report-header.tex")
        forbidden = [str(root) for root in roots]
        findings = _scan_bundle(output, forbidden)
        validation_checks = {
            "run_roots_distinct": len(roots) == len(set(roots)),
            "labels_distinct": len(labels) == len(set(labels)),
            "partial_runs_explicit": aggregate["all_runs_final"] or report_status == "interim",
            "performance_denominators_present": all(run["planned_tasks"] > 0 for run in runs),
            "cost_reconciliation": all(item["reconciles"] for item in reconciliation["runs"].values()),
            "end_to_end_cost_reconciliation": all(
                item["end_to_end_reconciles"] for item in reconciliation["runs"].values()
            )
            or report_status != "final",
            "group_cost_reconciliation": all(
                abs(
                    group["counterfactual_baseline_cost_mean"]
                    - group["actual_cost_mean"]
                    - group["routing_savings_mean"]
                )
                <= 1e-6
                for group in configuration_groups
            ),
            "group_end_to_end_cost_reconciliation": all(
                group["total_cost_including_routing_mean"] is not None
                and group["total_savings_including_routing_mean"] is not None
                and abs(
                    group["counterfactual_baseline_cost_mean"]
                    - group["total_cost_including_routing_mean"]
                    - group["total_savings_including_routing_mean"]
                )
                <= 1e-6
                for group in configuration_groups
            )
            or report_status != "final",
            "routing_call_reconciliation": all(
                item["routing_calls_reconcile"] for item in reconciliation["runs"].values()
            ),
            "absolute_paths_and_secrets_absent": not findings,
        }
        validation = {
            "schema_version": "terminal-bench-report.validation.v1",
            "status": "passed" if all(validation_checks.values()) else "failed",
            "checks": validation_checks,
            "findings": findings,
        }
        (evidence / "report-validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_files = [
            path for path in sorted(output.rglob("*")) if path.is_file() and path.name != "bundle-manifest.json"
        ]
        manifest = {
            "schema_version": "terminal-bench-report.manifest.v1",
            "analysis_digest": canonical_digest(payload),
            "files": {str(path.relative_to(output)): sha256_file(path) for path in manifest_files},
        }
        (evidence / "bundle-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": report_status, "output": str(output), "runs": labels}, indent=2))
        return 0 if validation["status"] == "passed" else 2
    except Exception:
        if output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AdmissionError as error:
        print(f"report admission failed: {error}", file=sys.stderr)
        raise SystemExit(2) from None
