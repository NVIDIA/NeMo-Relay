#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Terminal-Bench artifact parsing and quantitative analysis."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "terminal-bench-report.analysis.v1"
CALL_SCHEMA_VERSION = "terminal-bench-report.call.v1"
COST_TOLERANCE = 1e-6
ROUTING_CONFIDENCE = re.compile(r"\bconfidence\s+([0-9]+(?:\.[0-9]+)?)\b")


class AdmissionError(ValueError):
    """Raised when source artifacts cannot support the requested analysis."""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdmissionError(f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _first_file(paths: Iterable[Path], label: str, *, required: bool = True) -> Path | None:
    values = sorted({path.resolve() for path in paths if path.is_file()})
    if len(values) == 1:
        return values[0]
    if not values and not required:
        return None
    raise AdmissionError(f"expected exactly one {label}, found {len(values)}")


def _runtime_config_path(root: Path) -> Path:
    candidates = [
        root / "setup-runtime" / "runtime" / "plugins.toml",
        root / "runtime" / "plugins.toml",
    ]
    direct = [path for path in candidates if path.is_file()]
    if direct:
        return direct[0]
    return _first_file(root.glob("tasks/*/attempts/*/runtime/plugins.toml"), "rendered plugin config")  # type: ignore[return-value]


def _resolved_mapping(
    configured: dict[str, Any], schema_properties: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for key in sorted(set(configured) | set(schema_properties)):
        if key in configured:
            values[key] = configured[key]
            sources[key] = "configured"
        elif isinstance(schema_properties.get(key), dict) and "default" in schema_properties[key]:
            values[key] = schema_properties[key]["default"]
            sources[key] = "schema_default"
    return values, sources


def parse_runtime_config(root: Path) -> dict[str, Any]:
    path = _runtime_config_path(root)
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    components = {component.get("kind"): component for component in config.get("components", [])}
    pricing_component = components.get("pricing", {})
    pricing_sources = pricing_component.get("config", {}).get("sources", [])
    pricing_catalog_version: int | None = None
    entries: list[dict[str, Any]] = []
    for source in pricing_sources:
        if source.get("type") == "file":
            pricing_path = path.parent / "switchyard-plugin" / "pricing.json"
            catalog = read_json(pricing_path) if pricing_path.is_file() else {}
        else:
            catalog = source.get("catalog", {})
        if pricing_catalog_version is None:
            pricing_catalog_version = catalog.get("version")
        for entry in catalog.get("entries", []):
            rates = entry.get("rates", {})
            entries.append(
                {
                    "provider": entry.get("provider"),
                    "model_id": entry.get("model_id"),
                    "currency": entry.get("currency"),
                    "unit": entry.get("unit"),
                    "pricing_as_of": entry.get("pricing_as_of"),
                    "pricing_source": entry.get("pricing_source"),
                    "rates": {
                        "input_per_million": rates.get("input_per_million"),
                        "output_per_million": rates.get("output_per_million"),
                        "cache_read_per_million": rates.get("cache_read_per_million"),
                        "cache_write_per_million": rates.get("cache_write_per_million"),
                    },
                    "cache_read_accounting": entry.get("prompt_cache", {}).get(
                        "read_accounting", "included_in_prompt_tokens"
                    ),
                }
            )
    dynamic = config.get("plugins", {}).get("dynamic", [])
    if not isinstance(dynamic, list) or len(dynamic) > 1:
        raise AdmissionError("rendered configuration must contain zero or one dynamic Switchyard plugin")
    switchyard = dynamic[0].get("config", {}) if len(dynamic) == 1 else {}
    schema_path = path.parent / "switchyard-plugin" / "config.schema.json"
    schema = read_json(schema_path) if schema_path.is_file() else {}
    schema_properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    top_values, top_sources = _resolved_mapping(switchyard, schema_properties)

    # The Switchyard-native plugin's own config.schema.json only describes the
    # priority/switchyard_config_path wrapper; the routing algorithm and
    # target/client details live in a separate switchyard-routes.toml
    # deployment file with no bundled schema, so "configured vs. schema
    # default" provenance is only available for the wrapper fields above.
    switchyard_routes: dict[str, Any] = {}
    if len(dynamic) == 1:
        routes_path = path.parent / "switchyard-plugin" / "switchyard-routes.toml"
        if routes_path.is_file():
            with routes_path.open("rb") as stream:
                switchyard_routes = tomllib.load(stream)
    llm_clients = switchyard_routes.get("llm_clients", {})
    route = switchyard_routes.get("routes", {}).get("default", {})

    algorithm = {key: value for key, value in route.items() if key not in {"id", "type"}}
    algorithm["kind"] = route.get("type")
    classifier = algorithm.get("classifier")
    prompt = classifier.get("prompt") if isinstance(classifier, dict) else algorithm.get("prompt")
    algorithm.pop("prompt", None)
    algorithm["prompt_configured"] = isinstance(prompt, str)
    algorithm["prompt_length"] = len(prompt) if isinstance(prompt, str) else None
    algorithm["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest() if isinstance(prompt, str) else None
    algorithm["unexposed_generation_controls"] = ["temperature", "top_p", "seed"]
    algorithm_sources = {
        key: "configured"
        for key, value in algorithm.items()
        if key not in {"prompt_configured", "prompt_length", "prompt_sha256", "unexposed_generation_controls"}
        and value is not None
    }
    algorithm_sources["prompt"] = "configured_digest" if isinstance(prompt, str) else "plugin_builtin_not_serialized"

    targets = switchyard_routes.get("targets", {})
    resolved_targets: dict[str, dict[str, Any]] = {}
    target_sources: dict[str, dict[str, str]] = {}
    for name, target in sorted(targets.items()):
        client_name = target.get("llm_client")
        client = llm_clients.get(client_name, {}) if isinstance(client_name, str) else {}
        resolved_targets[name] = {
            "model": target.get("id"),
            "protocol": client.get("format"),
            "base_url": client.get("base_url"),
            "extra_body": target.get("extra_body"),
        }
        target_sources[name] = {
            "header_names": "configured_names_only" if client.get("api_key_env") else "not_configured",
        }
    profile = {
        "pricing_catalog_version": pricing_catalog_version,
        "pricing_entries": sorted(entries, key=lambda item: str(item.get("model_id"))),
        "switchyard": {
            "routing_mode": "switchyard" if len(dynamic) == 1 else "direct",
            "enabled": len(dynamic) == 1,
            "priority": top_values.get("priority"),
            "algorithm": algorithm,
            "algorithm_sources": algorithm_sources,
            "targets": resolved_targets,
            "target_sources": target_sources,
            "top_level_sources": {"priority": top_sources.get("priority", "unresolved")},
            "config_schema_sha256": sha256_file(schema_path) if schema_path.is_file() else None,
        },
    }
    profile["digest"] = canonical_digest(profile)
    profile["source_sha256"] = sha256_file(path)
    return profile


def pricing_by_model(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(entry["model_id"]): entry for entry in profile["pricing_entries"]}


def estimate_cost(usage: dict[str, int], pricing: dict[str, Any]) -> float | None:
    rates = pricing.get("rates", {})
    input_rate = rates.get("input_per_million")
    output_rate = rates.get("output_per_million")
    if not isinstance(input_rate, (int, float)) or not isinstance(output_rate, (int, float)):
        return None
    prompt = max(0, int(usage.get("prompt_tokens", 0)))
    completion = max(0, int(usage.get("completion_tokens", 0)))
    cache_read = max(0, int(usage.get("cache_read_tokens", 0)))
    cache_write = max(0, int(usage.get("cache_write_tokens", 0)))
    billable_prompt = prompt
    if pricing.get("cache_read_accounting") == "included_in_prompt_tokens":
        billable_prompt = max(0, prompt - cache_read)
    total = billable_prompt * float(input_rate) / 1_000_000
    total += completion * float(output_rate) / 1_000_000
    cache_read_rate = rates.get("cache_read_per_million")
    cache_write_rate = rates.get("cache_write_per_million")
    if isinstance(cache_read_rate, (int, float)):
        total += cache_read * float(cache_read_rate) / 1_000_000
    elif cache_read:
        return None
    if isinstance(cache_write_rate, (int, float)):
        total += cache_write * float(cache_write_rate) / 1_000_000
    elif cache_write:
        return None
    return round(total, 12)


def _any_value(value: dict[str, Any]) -> Any:
    for key in ("stringValue", "intValue", "doubleValue", "boolValue", "bytesValue"):
        if key in value:
            return value[key]
    if "arrayValue" in value:
        return [_any_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {item.get("key"): _any_value(item.get("value", {})) for item in value["kvlistValue"].get("values", [])}
    return None


def _attributes(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {str(item["key"]): _any_value(item.get("value", {})) for item in items if item.get("key")}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return max(0, int(number)) if number is not None else None


def _usage_from_attributes(attrs: dict[str, Any]) -> dict[str, int]:
    usage = {
        "prompt_tokens": _integer(attrs.get("llm.token_count.prompt")),
        "completion_tokens": _integer(attrs.get("llm.token_count.completion")),
        "cache_read_tokens": _integer(attrs.get("llm.token_count.prompt_details.cache_read")),
        "cache_write_tokens": _integer(attrs.get("llm.token_count.prompt_details.cache_write")),
    }
    if usage["prompt_tokens"] is None or usage["completion_tokens"] is None:
        raw = attrs.get("nemo_relay.end.data.usage") or attrs.get("nemo_relay.end.output.usage")
        try:
            decoded = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            usage["prompt_tokens"] = usage["prompt_tokens"] or _integer(decoded.get("prompt_tokens"))
            usage["completion_tokens"] = usage["completion_tokens"] or _integer(decoded.get("completion_tokens"))
            details = decoded.get("prompt_tokens_details") or {}
            usage["cache_read_tokens"] = usage["cache_read_tokens"] or _integer(
                decoded.get("cache_read_tokens", details.get("cached_tokens"))
            )
            usage["cache_write_tokens"] = usage["cache_write_tokens"] or _integer(decoded.get("cache_write_tokens"))
    return {key: int(value or 0) for key, value in usage.items()}


def _routing_overhead_from_mark(
    event: dict[str, Any],
    pricing: dict[str, dict[str, Any]],
    target_models: dict[str, str],
) -> dict[str, Any]:
    """Normalize one Switchyard routing-only LLM mark without double-counting serving calls."""

    data = event.get("data") or {}
    target = data.get("selected_target")
    model = target_models.get(str(target)) if target is not None else None
    raw_usage = data.get("usage")
    usage_present = isinstance(raw_usage, dict)
    noncached_input = int(_integer(raw_usage.get("input_tokens")) or 0) if usage_present else 0
    cache_read = int(_integer(raw_usage.get("cached_input_tokens")) or 0) if usage_present else 0
    cache_write = int(_integer(raw_usage.get("cache_creation_input_tokens")) or 0) if usage_present else 0
    output = int(_integer(raw_usage.get("output_tokens")) or 0) if usage_present else 0
    reasoning = int(_integer(raw_usage.get("reasoning_tokens")) or 0) if usage_present else 0
    # Switchyard normalizes input_tokens to exclude cache details and output_tokens
    # to exclude separately reported reasoning. Relay pricing expects aggregate
    # prompt/completion totals plus cache details for the discounted portions.
    usage = {
        "prompt_tokens": noncached_input + cache_read + cache_write,
        "completion_tokens": output + reasoning,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
    }
    cost = estimate_cost(usage, pricing[model]) if usage_present and model in pricing else None
    return {
        "mark_uuid": event.get("uuid"),
        "parent_uuid": event.get("parent_uuid"),
        "algorithm": data.get("algorithm"),
        "attempt": _integer(data.get("attempt")),
        "call_index": _integer(data.get("call_index")),
        "call_role": data.get("call_role"),
        "outcome": data.get("outcome"),
        "selected_target": target,
        "routing_tier": data.get("routing_tier"),
        "model": model,
        "latency_ms": _number(data.get("latency_ms")),
        "usage_present": usage_present,
        **usage,
        "noncached_input_tokens": noncached_input,
        "output_tokens_excluding_reasoning": output,
        "reasoning_tokens": reasoning,
        "cost": cost,
        "cost_covered": cost is not None,
        "contributes_to_routing_overhead": data.get("contributes_to_routing_overhead") is True,
    }


def _iter_otlp_spans(path: Path) -> Iterable[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as error:
                raise AdmissionError(f"invalid OTLP JSON at {path.name}:{line_number}: {error}") from error
            for resource in document.get("resourceSpans", []):
                for scope in resource.get("scopeSpans", []):
                    for span in scope.get("spans", []):
                        identity = (str(span.get("traceId", "")), str(span.get("spanId", "")))
                        if identity in seen:
                            continue
                        seen.add(identity)
                        yield span


def _telemetry_receipts(attempt: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    telemetry = attempt / "telemetry" / "trajectory.openinference.json"
    if not telemetry.is_file():
        return {}, {
            "telemetry_present": False,
            "candidate_llm_spans": 0,
            "router_overhead_candidate_spans": 0,
        }
    receipts: dict[str, dict[str, Any]] = {}
    candidate_llm_spans = 0
    router_overhead_candidate_spans = 0
    for span in _iter_otlp_spans(telemetry):
        attrs = _attributes(span.get("attributes", []))
        if attrs.get("openinference.span.kind") != "LLM":
            continue
        candidate_llm_spans += 1
        call_role = str(
            attrs.get("openinference.metadata.call_role")
            or attrs.get("metadata.call_role")
            or attrs.get("call_role")
            or ""
        ).lower()
        span_name = str(span.get("name") or "").lower()
        if span_name in {"libsy.client_call", "switchyard.classifier", "switchyard.judge"} or call_role in {
            "router_overhead",
            "classifier",
            "judge",
        }:
            router_overhead_candidate_spans += 1
        relay_uuid = attrs.get("nemo_relay.uuid")
        if not isinstance(relay_uuid, str) or not relay_uuid:
            continue
        model = (
            attrs.get("nemo_relay.llm.optimization.effective_model")
            or attrs.get("nemo_relay.end.data.model")
            or attrs.get("nemo_relay.end.output.model")
            or attrs.get("llm.model_name")
        )
        usage = _usage_from_attributes(attrs)
        receipts[relay_uuid] = {
            "trace_id": str(span.get("traceId", "")),
            "span_id": str(span.get("spanId", "")),
            "model": str(model) if model is not None else None,
            "usage": usage,
            "recorded_actual_cost": _number(
                attrs.get("nemo_relay.llm.optimization.actual_cost", attrs.get("llm.cost.total"))
            ),
            "recorded_baseline_cost": _number(attrs.get("nemo_relay.llm.optimization.baseline_cost")),
            "recorded_routing_savings": _number(attrs.get("nemo_relay.llm.optimization.estimated_cost_saved")),
        }
    return receipts, {
        "telemetry_present": True,
        "telemetry_sha256": sha256_file(telemetry),
        "candidate_llm_spans": candidate_llm_spans,
        "router_overhead_candidate_spans": router_overhead_candidate_spans,
        "relay_uuid_receipts": len(receipts),
    }


def parse_atof_calls(
    attempt: Path,
    run_label: str,
    task_name: str,
    pricing: dict[str, dict[str, Any]],
    baseline_model: str,
    target_models: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = attempt.glob("jobs/*/*/artifacts/logs/agent/direct-hermes/relay/trajectory.atof.jsonl")
    path = _first_file(paths, "ATOF routing trajectory", required=False)
    if path is None:
        return [], {
            "atof_present": False,
            "route_decisions": 0,
            "usage_chunks": 0,
            "routing_overhead_marks": 0,
            "routing_overhead_usage_covered_marks": 0,
            "routing_overhead_cost_covered_marks": 0,
            "routing_overhead_cost": 0.0,
        }
    telemetry, telemetry_evidence = _telemetry_receipts(attempt)
    decisions_by_logical: dict[str, list[dict[str, Any]]] = {}
    direct_model = baseline_model if not target_models else None
    fallback_targets: dict[str, list[str]] = {}
    llm_children: dict[str, list[str]] = {}
    usage_by_llm: dict[str, tuple[int, dict[str, int]]] = {}
    seen_decisions: set[str] = set()
    seen_overhead_marks: set[str] = set()
    routing_overhead: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                event.get("kind") == "scope"
                and event.get("category") == "llm"
                and event.get("scope_category") == "start"
            ):
                parent_uuid = event.get("parent_uuid")
                llm_uuid = event.get("uuid")
                if isinstance(parent_uuid, str) and isinstance(llm_uuid, str):
                    llm_children.setdefault(parent_uuid, []).append(llm_uuid)
            elif event.get("kind") == "mark" and event.get("name") == "llm.chunk":
                llm_uuid = event.get("parent_uuid")
                data = event.get("data") or {}
                raw_usage = data.get("usage")
                if isinstance(llm_uuid, str) and isinstance(raw_usage, dict):
                    usage = {
                        key: int(_integer(raw_usage.get(key)) or 0)
                        for key in ("prompt_tokens", "completion_tokens", "cache_read_tokens", "cache_write_tokens")
                    }
                    chunk_index = int(_integer(data.get("chunk_index")) or 0)
                    prior = usage_by_llm.get(llm_uuid)
                    if prior is None or chunk_index >= prior[0]:
                        usage_by_llm[llm_uuid] = (chunk_index, usage)
            elif event.get("kind") == "mark" and event.get("name") == "switchyard.routing.decision":
                identity = str(event.get("uuid", ""))
                if identity in seen_decisions:
                    continue
                seen_decisions.add(identity)
                data = event.get("data") or {}
                logical_uuid = str(event.get("parent_uuid", ""))
                reasoning = data.get("reasoning")
                confidence_match = ROUTING_CONFIDENCE.search(reasoning) if isinstance(reasoning, str) else None
                decisions_by_logical.setdefault(logical_uuid, []).append(
                    {
                        "decision_uuid": identity,
                        "selected_target": data.get("selected_target"),
                        "routing_tier": data.get("routing_tier"),
                        "reasoning": reasoning,
                        "decision_source": data.get("decision_source"),
                        "confidence": float(confidence_match.group(1)) if confidence_match else None,
                        "algorithm": data.get("algorithm"),
                        "attempt": _integer(data.get("attempt")),
                    }
                )
            elif event.get("kind") == "mark" and event.get("name") == "switchyard.routing.llm_call":
                identity = str(event.get("uuid", ""))
                if identity in seen_overhead_marks:
                    continue
                seen_overhead_marks.add(identity)
                overhead = _routing_overhead_from_mark(event, pricing, target_models)
                if overhead["contributes_to_routing_overhead"]:
                    routing_overhead.append(overhead)
            elif event.get("kind") == "mark" and event.get("name") == "switchyard.routing.fallback":
                logical_uuid = event.get("parent_uuid")
                target = (event.get("data") or {}).get("selected_target")
                if isinstance(logical_uuid, str) and isinstance(target, str) and target:
                    fallback_targets.setdefault(logical_uuid, []).append(target)

    if direct_model is not None:
        for logical_uuid in llm_children:
            decisions_by_logical[logical_uuid] = [
                {
                    "decision_uuid": None,
                    "selected_target": "direct",
                    "routing_tier": "direct",
                    "reasoning": "no_switchyard_router",
                    "decision_source": "direct",
                    "confidence": None,
                    "algorithm": "direct",
                    "attempt": None,
                    "synthetic_direct": True,
                }
            ]

    calls: list[dict[str, Any]] = []
    ambiguous_children = 0
    for call_index, (logical_uuid, decisions) in enumerate(decisions_by_logical.items(), 1):
        decision = decisions[-1]
        children = list(dict.fromkeys(llm_children.get(logical_uuid, [])))
        if len(children) != 1:
            ambiguous_children += 1
        llm_uuid = children[0] if len(children) == 1 else None
        atof_usage_entry = usage_by_llm.get(llm_uuid) if llm_uuid else None
        receipt = telemetry.get(llm_uuid or "")
        receipt_usage = receipt.get("usage") if receipt else None
        receipt_has_usage = isinstance(receipt_usage, dict) and any(receipt_usage.values())
        usage_source = (
            "atof_final_chunk" if atof_usage_entry else "openinference_fallback" if receipt_has_usage else None
        )
        usage = (
            atof_usage_entry[1]
            if atof_usage_entry
            else receipt_usage
            if receipt_has_usage
            else {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            }
        )
        target = fallback_targets.get(logical_uuid, [decision.get("selected_target")])[-1]
        model = direct_model or target_models.get(str(target), "")
        recorded_actual = receipt.get("recorded_actual_cost") if receipt else None
        recorded_baseline = receipt.get("recorded_baseline_cost") if receipt else None
        recorded_savings = receipt.get("recorded_routing_savings") if receipt else None
        derived_actual = estimate_cost(usage, pricing[model]) if usage_source and model in pricing else None
        baseline = estimate_cost(usage, pricing[baseline_model]) if usage_source and baseline_model in pricing else None
        usage_matches_receipt = receipt_usage == usage if receipt_has_usage else None
        model_matches_receipt = receipt.get("model") == model if receipt and receipt.get("model") in pricing else None
        actual_matches_receipt = (
            abs(recorded_actual - derived_actual) <= COST_TOLERANCE
            if recorded_actual is not None and derived_actual is not None
            else None
        )
        actual = derived_actual
        savings = baseline - actual if actual is not None and baseline is not None else None
        baseline_matches_receipt = (
            abs(recorded_baseline - baseline) <= COST_TOLERANCE
            if recorded_baseline is not None and baseline is not None
            else None
        )
        savings_matches_receipt = (
            abs(recorded_savings - savings) <= COST_TOLERANCE
            if recorded_savings is not None and savings is not None
            else None
        )
        cost_covered = actual is not None and baseline is not None and savings is not None
        calls.append(
            {
                "schema_version": CALL_SCHEMA_VERSION,
                "run_label": run_label,
                "task_name": task_name,
                "call_index": call_index,
                "call_id": canonical_digest([task_name, logical_uuid])[:16],
                "trace_id": receipt.get("trace_id", "") if receipt else "",
                "span_id": receipt.get("span_id", "") if receipt else "",
                "route_target": target,
                "routing_tier": decision.get("routing_tier"),
                "routing_reason": decision.get("reasoning"),
                "routing_decision_source": decision.get("decision_source"),
                "routing_confidence": decision.get("confidence"),
                "routing_algorithm": decision.get("algorithm"),
                "routing_attempt_count": 0 if decision.get("synthetic_direct") else len(decisions),
                "routing_attempt_targets": (
                    [] if decision.get("synthetic_direct") else [item.get("selected_target") for item in decisions]
                ),
                "fallback_target": fallback_targets.get(logical_uuid, [None])[-1],
                "model": model or None,
                "baseline_model": baseline_model,
                **usage,
                "usage_present": usage_source is not None,
                "usage_source": usage_source,
                "atof_usage_present": atof_usage_entry is not None,
                "actual_cost": actual,
                "actual_cost_source": "derived_from_recorded_usage" if derived_actual is not None else None,
                "derived_actual_cost": derived_actual,
                "baseline_cost": baseline,
                "baseline_cost_source": "derived_counterfactual" if baseline is not None else None,
                "routing_savings": savings,
                "currency": pricing.get(model, {}).get("currency"),
                "cost_covered": cost_covered,
                "telemetry_receipt_present": receipt is not None,
                "telemetry_usage_present": receipt_has_usage,
                "usage_matches_telemetry_receipt": usage_matches_receipt,
                "model_matches_telemetry_receipt": model_matches_receipt,
                "recorded_actual_cost": recorded_actual,
                "actual_matches_telemetry_receipt": actual_matches_receipt,
                "recorded_baseline_cost": recorded_baseline,
                "baseline_matches_telemetry_receipt": baseline_matches_receipt,
                "recorded_routing_savings": recorded_savings,
                "savings_matches_telemetry_receipt": savings_matches_receipt,
            }
        )
    overhead_by_role = Counter(str(item["call_role"]) for item in routing_overhead if item.get("call_role"))
    overhead_by_model = Counter(str(item["model"]) for item in routing_overhead if item.get("model"))
    return calls, {
        "atof_present": True,
        "atof_sha256": sha256_file(path),
        "logical_routed_calls": 0 if direct_model is not None else len(decisions_by_logical),
        "direct_logical_calls": len(decisions_by_logical) if direct_model is not None else 0,
        "route_decisions": (
            0 if direct_model is not None else sum(len(items) for items in decisions_by_logical.values())
        ),
        "route_retry_decisions": (
            0 if direct_model is not None else sum(max(0, len(items) - 1) for items in decisions_by_logical.values())
        ),
        "fallback_calls": len(fallback_targets),
        "usage_chunks": len(usage_by_llm),
        "routing_overhead_marks": len(routing_overhead),
        "routing_overhead_successful_marks": sum(item["outcome"] == "ok" for item in routing_overhead),
        "routing_overhead_error_marks": sum(item["outcome"] == "error" for item in routing_overhead),
        "routing_overhead_usage_covered_marks": sum(item["usage_present"] for item in routing_overhead),
        "routing_overhead_cost_covered_marks": sum(item["cost_covered"] for item in routing_overhead),
        "routing_overhead_prompt_tokens": sum(item["prompt_tokens"] for item in routing_overhead),
        "routing_overhead_noncached_input_tokens": sum(
            item["noncached_input_tokens"] for item in routing_overhead
        ),
        "routing_overhead_completion_tokens": sum(item["completion_tokens"] for item in routing_overhead),
        "routing_overhead_output_tokens_excluding_reasoning": sum(
            item["output_tokens_excluding_reasoning"] for item in routing_overhead
        ),
        "routing_overhead_reasoning_tokens": sum(item["reasoning_tokens"] for item in routing_overhead),
        "routing_overhead_cache_read_tokens": sum(item["cache_read_tokens"] for item in routing_overhead),
        "routing_overhead_cache_write_tokens": sum(item["cache_write_tokens"] for item in routing_overhead),
        "routing_overhead_latency_ms": sum(float(item["latency_ms"] or 0.0) for item in routing_overhead),
        "routing_overhead_cost": sum(float(item["cost"] or 0.0) for item in routing_overhead),
        "routing_overhead_by_role": dict(sorted(overhead_by_role.items())),
        "routing_overhead_by_model": dict(sorted(overhead_by_model.items())),
        "ambiguous_llm_children": ambiguous_children,
        "telemetry_receipted_calls": sum(call["telemetry_receipt_present"] for call in calls),
        **telemetry_evidence,
    }


def _task_root(root: Path, task: dict[str, Any]) -> Path | None:
    prefix = f"{int(task['index']):03d}-"
    task_dirs = sorted(path for path in (root / "tasks").glob(f"{prefix}*") if path.is_dir())
    if not task_dirs and not task.get("successful_attempt"):
        return None
    if len(task_dirs) != 1:
        raise AdmissionError(f"could not uniquely resolve task {task.get('name')} below {root.name}")
    return task_dirs[0]


def _task_attempt(task_root: Path | None, task: dict[str, Any]) -> Path | None:
    successful = task.get("successful_attempt")
    if not successful or task_root is None:
        return None
    attempt = task_root / "attempts" / str(successful)
    return attempt if attempt.is_dir() else None


def _task_metrics(
    root: Path,
    run_label: str,
    task: dict[str, Any],
    pricing: dict[str, dict[str, Any]],
    baseline_model: str,
    target_models: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_root = _task_root(root, task)
    attempt = _task_attempt(task_root, task)
    benchmark = task.get("benchmark_completion") or {}
    benchmark_complete = task.get("status") == "completed" and benchmark.get("status") == "passed"
    benchmark_passed = task.get("benchmark_task_passed") if benchmark_complete else None
    integration = task.get("integration_validation") or {}
    integration_passed = integration.get("status") == "passed" and task.get("phoenix_upload") == "passed"
    calls: list[dict[str, Any]] = []
    call_evidence: dict[str, Any] = {"atof_present": False, "telemetry_present": False}
    terminal_completion_class = "ordinary"
    reconciliation_receipt_present = False
    provider_retry_receipt_count = (
        len(list(task_root.glob("provider-retry-*.json"))) if task_root is not None else 0
    )
    if attempt is not None:
        attempt_summary_path = attempt / "summary.json"
        if attempt_summary_path.is_file():
            attempt_summary = read_json(attempt_summary_path)
            validation = attempt_summary.get("validation")
            if isinstance(validation, dict):
                if validation.get("terminal_quiet_output_completion") is True:
                    terminal_completion_class = "quiet_output_reconciled"
                elif validation.get("terminal_agent_timeout_completion") is True:
                    terminal_completion_class = "agent_timeout_reconciled"
                elif validation.get("terminal_turn_budget_completion") is True:
                    terminal_completion_class = "turn_budget_reconciled"
        reconciliation_receipt_present = (attempt / "nonpass-reconciliation.json").is_file()
        calls, call_evidence = parse_atof_calls(
            attempt,
            run_label,
            str(task["name"]),
            pricing,
            baseline_model,
            target_models,
        )
    covered = [call for call in calls if call["cost_covered"]]
    model_counts = Counter(str(call["model"]) for call in calls if call.get("model"))
    target_counts = Counter(str(call["route_target"]) for call in calls if call.get("route_target"))
    reason_counts = Counter(str(call["routing_reason"]) for call in calls if call.get("routing_reason"))
    decision_source_counts = Counter(
        str(call["routing_decision_source"]) for call in calls if call.get("routing_decision_source")
    )
    routing_confidences = [float(call["routing_confidence"]) for call in calls if call.get("routing_confidence") is not None]
    prompt = sum(call["prompt_tokens"] for call in calls)
    cache_read = sum(call["cache_read_tokens"] for call in calls)
    result = {
        "run_label": run_label,
        "task_index": int(task["index"]),
        "task_name": str(task["name"]),
        "benchmark_complete": benchmark_complete,
        "benchmark_passed": benchmark_passed,
        "integration_passed": integration_passed,
        "attempt_count": int(task.get("attempt_count") or 0),
        "terminal_completion_class": terminal_completion_class,
        "reconciliation_receipt_present": reconciliation_receipt_present,
        "provider_retry_receipt_count": provider_retry_receipt_count,
        "provider_calls": len(calls),
        "cost_covered_calls": len(covered),
        "actual_cost": sum(float(call["actual_cost"]) for call in covered),
        "baseline_cost": sum(float(call["baseline_cost"]) for call in covered),
        "routing_savings": sum(float(call["routing_savings"]) for call in covered),
        "router_overhead_marks": int(call_evidence.get("routing_overhead_marks", 0)),
        "router_overhead_usage_covered_marks": int(
            call_evidence.get("routing_overhead_usage_covered_marks", 0)
        ),
        "router_overhead_cost_covered_marks": int(
            call_evidence.get("routing_overhead_cost_covered_marks", 0)
        ),
        "router_overhead_cost": float(call_evidence.get("routing_overhead_cost", 0.0)),
        "covered_model_cost_including_routing": (
            sum(float(call["actual_cost"]) for call in covered)
            + float(call_evidence.get("routing_overhead_cost", 0.0))
        ),
        "router_overhead_prompt_tokens": int(call_evidence.get("routing_overhead_prompt_tokens", 0)),
        "router_overhead_completion_tokens": int(
            call_evidence.get("routing_overhead_completion_tokens", 0)
        ),
        "router_overhead_reasoning_tokens": int(call_evidence.get("routing_overhead_reasoning_tokens", 0)),
        "router_overhead_cache_read_tokens": int(
            call_evidence.get("routing_overhead_cache_read_tokens", 0)
        ),
        "router_overhead_cache_write_tokens": int(
            call_evidence.get("routing_overhead_cache_write_tokens", 0)
        ),
        "currency": pricing[baseline_model].get("currency"),
        "prompt_tokens": prompt,
        "completion_tokens": sum(call["completion_tokens"] for call in calls),
        "cache_read_tokens": cache_read,
        "cache_write_tokens": sum(call["cache_write_tokens"] for call in calls),
        "cache_read_ratio": ratio(cache_read, prompt),
        "model_counts": dict(sorted(model_counts.items())),
        "route_target_counts": dict(sorted(target_counts.items())),
        "route_reason_counts": dict(sorted(reason_counts.items())),
        "route_decision_source_counts": dict(sorted(decision_source_counts.items())),
        "routing_confidence_count": len(routing_confidences),
        "routing_confidence_mean": statistics.mean(routing_confidences) if routing_confidences else None,
        "route_decisions": call_evidence.get("route_decisions", 0),
        "routing_call_delta": call_evidence.get("route_decisions", 0) - len(calls),
        "route_retry_decisions": call_evidence.get("route_retry_decisions", 0),
        "fallback_calls": call_evidence.get("fallback_calls", 0),
        "usage_covered_calls": sum(call["usage_present"] for call in calls),
        "telemetry_receipted_calls": sum(call["telemetry_receipt_present"] for call in calls),
        "call_evidence": call_evidence,
        "integration_errors": integration.get("errors", []),
    }
    return result, calls


def _scientific_config(plan: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    inputs = plan.get("inputs", {})
    normalized_profile = {key: value for key, value in profile.items() if key != "source_sha256"}
    return {
        "dataset": plan.get("dataset"),
        "sample_count": plan.get("sample_count"),
        "dataset_task_definitions_sha256": inputs.get("dataset_task_definitions_sha256"),
        "task_manifest": [
            {
                key: task.get(key)
                for key in ("index", "name", "memory_gb", "effective_memory_gb", "runtime_override")
                if key in task
            }
            for task in plan.get("tasks", [])
        ],
        "runtime_sources_sha256": inputs.get("runtime_sources_sha256"),
        "runner_sha256": inputs.get("runner_sha256"),
        "relay_wheel_sha256": inputs.get("relay_wheel_sha256"),
        "switchyard_manifest_sha256": inputs.get("switchyard_manifest_sha256"),
        "switchyard_library_sha256": inputs.get("switchyard_library_sha256"),
        "plugin_config_template_sha256": inputs.get("plugin_config_template_sha256"),
        "concurrency": plan.get("concurrency"),
        "parallel_max_memory_gb": plan.get("parallel_max_memory_gb"),
        "timeout_multipliers": plan.get("timeout_multipliers"),
        "runtime_profile": normalized_profile,
    }


def analyze_run(
    root: Path, label: str, baseline_model_override: str | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = root.expanduser().resolve()
    summary_path = root / "summary.json"
    plan_path = root / "plan.json"
    if not summary_path.is_file() or not plan_path.is_file():
        raise AdmissionError(f"run root must contain summary.json and plan.json: {root}")
    summary = read_json(summary_path)
    plan = read_json(plan_path)
    profile = parse_runtime_config(root)
    scientific_config = _scientific_config(plan, profile)
    config_signature = canonical_digest(scientific_config)
    task_manifest_signature = canonical_digest(scientific_config["task_manifest"])
    pricing = pricing_by_model(profile)
    targets = profile.get("switchyard", {}).get("targets", {})
    target_models = {
        str(name): str(target.get("model"))
        for name, target in targets.items()
        if isinstance(target, dict) and target.get("model")
    }
    router_profile = profile.get("switchyard", {})
    algorithm = router_profile.get("algorithm", {})
    judge_target = algorithm.get("classifier_target")
    if judge_target is None and isinstance(algorithm.get("classifier"), dict):
        judge_target = algorithm["classifier"].get("target")
    judge_model = target_models.get(str(judge_target)) if judge_target is not None else None
    strong = target_models.get("strong")
    sole_catalog_model = next(iter(pricing)) if len(pricing) == 1 else None
    baseline_model = baseline_model_override or strong or sole_catalog_model
    if not isinstance(baseline_model, str) or baseline_model not in pricing:
        raise AdmissionError(
            f"baseline model is missing from the run-bound pricing catalog for {label}: {baseline_model}"
        )
    tasks: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for task in summary.get("tasks", []):
        metrics, task_calls = _task_metrics(root, label, task, pricing, baseline_model, target_models)
        tasks.append(metrics)
        calls.extend(task_calls)
    planned = int(summary.get("planned_tasks") or len(plan.get("tasks", [])))
    complete = sum(task["benchmark_complete"] for task in tasks)
    passed = sum(task["benchmark_passed"] is True for task in tasks)
    nonpass = sum(task["benchmark_passed"] is False for task in tasks)
    covered = [call for call in calls if call["cost_covered"]]
    prompt = sum(call["prompt_tokens"] for call in calls)
    cache_read = sum(call["cache_read_tokens"] for call in calls)
    model_counts = Counter(str(call["model"]) for call in calls)
    model_usage: dict[str, dict[str, float | int]] = {}
    for call in calls:
        model = str(call["model"])
        usage = model_usage.setdefault(
            model,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "covered_execution_cost": 0.0,
            },
        )
        usage["calls"] += 1
        usage["prompt_tokens"] += int(call["prompt_tokens"])
        usage["completion_tokens"] += int(call["completion_tokens"])
        usage["cache_read_tokens"] += int(call["cache_read_tokens"])
        usage["cache_write_tokens"] += int(call["cache_write_tokens"])
        if call["cost_covered"]:
            usage["covered_execution_cost"] += float(call["actual_cost"])
    target_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    decision_source_counts: Counter[str] = Counter()
    terminal_completion_counts = Counter(str(task["terminal_completion_class"]) for task in tasks)
    for task in tasks:
        target_counts.update(task["route_target_counts"])
        reason_counts.update(task["route_reason_counts"])
        decision_source_counts.update(task["route_decision_source_counts"])
    router_overhead_candidate_spans = sum(
        int(task["call_evidence"].get("router_overhead_candidate_spans", 0)) for task in tasks
    )
    router_overhead_scanned_attempts = sum(bool(task["call_evidence"].get("telemetry_present")) for task in tasks)
    router_overhead_marks = sum(task["router_overhead_marks"] for task in tasks)
    router_overhead_usage_covered_marks = sum(task["router_overhead_usage_covered_marks"] for task in tasks)
    router_overhead_cost_covered_marks = sum(task["router_overhead_cost_covered_marks"] for task in tasks)
    router_overhead_cost = sum(task["router_overhead_cost"] for task in tasks)
    if judge_model is None:
        router_overhead_usage_status = "not_applicable"
    elif router_overhead_marks == 0:
        router_overhead_usage_status = "unavailable_from_run_artifacts"
    elif (
        router_overhead_usage_covered_marks == router_overhead_marks
        and router_overhead_cost_covered_marks == router_overhead_marks
    ):
        router_overhead_usage_status = "complete"
    else:
        router_overhead_usage_status = "partial"
    actual = sum(float(call["actual_cost"]) for call in covered)
    baseline = sum(float(call["baseline_cost"]) for call in covered)
    savings = sum(float(call["routing_savings"]) for call in covered)
    execution_cost_complete = len(covered) == len(calls)
    total_cost = (
        actual
        if judge_model is None and execution_cost_complete
        else actual + router_overhead_cost
        if execution_cost_complete and router_overhead_usage_status == "complete"
        else None
    )
    total_savings = baseline - total_cost if total_cost is not None else None
    run = {
        "label": label,
        "source": {
            "summary_sha256": sha256_file(summary_path),
            "plan_sha256": sha256_file(plan_path),
            "runtime_config_sha256": profile["source_sha256"],
        },
        "status": "final" if summary.get("status") == "passed" and complete == planned else "interim",
        "source_summary_status": summary.get("status"),
        "dataset": plan.get("dataset", summary.get("dataset")),
        "evaluation_cohort": summary.get("evaluation_cohort"),
        "phoenix_project": summary.get("phoenix_project"),
        "planned_tasks": planned,
        "benchmark_complete_tasks": complete,
        "benchmark_missing_tasks": planned - complete,
        "benchmark_passes": passed,
        "benchmark_nonpasses": nonpass,
        "pass_at_1": ratio(passed, planned) if complete == planned else None,
        "observed_accuracy": ratio(passed, complete),
        "observed_accuracy_wilson_95": wilson_interval(passed, complete),
        "planned_task_lower_bound": ratio(passed, planned),
        "integration_passed_tasks": sum(task["integration_passed"] for task in tasks),
        "terminal_completion_counts": dict(sorted(terminal_completion_counts.items())),
        "reconciliation_receipt_count": sum(task["reconciliation_receipt_present"] for task in tasks),
        "provider_retry_receipt_count": sum(task["provider_retry_receipt_count"] for task in tasks),
        "provider_calls": len(calls),
        "cost_covered_calls": len(covered),
        "cost_coverage": ratio(len(covered), len(calls)),
        "usage_covered_calls": sum(call["usage_present"] for call in calls),
        "usage_coverage": ratio(sum(call["usage_present"] for call in calls), len(calls)),
        "telemetry_receipted_calls": sum(call["telemetry_receipt_present"] for call in calls),
        "telemetry_receipt_coverage": ratio(sum(call["telemetry_receipt_present"] for call in calls), len(calls)),
        "telemetry_usage_comparable_calls": sum(call["usage_matches_telemetry_receipt"] is not None for call in calls),
        "telemetry_usage_mismatch_calls": sum(call["usage_matches_telemetry_receipt"] is False for call in calls),
        "telemetry_model_comparable_calls": sum(call["model_matches_telemetry_receipt"] is not None for call in calls),
        "telemetry_model_mismatch_calls": sum(call["model_matches_telemetry_receipt"] is False for call in calls),
        "telemetry_cost_comparable_calls": sum(call["actual_matches_telemetry_receipt"] is not None for call in calls),
        "telemetry_cost_mismatch_calls": sum(call["actual_matches_telemetry_receipt"] is False for call in calls),
        "telemetry_recorded_cost_total": sum(
            float(call["recorded_actual_cost"])
            for call in calls
            if call["recorded_actual_cost"] is not None and call["derived_actual_cost"] is not None
        ),
        "derived_cost_for_telemetry_comparable_calls": sum(
            float(call["derived_actual_cost"])
            for call in calls
            if call["recorded_actual_cost"] is not None and call["derived_actual_cost"] is not None
        ),
        "actual_cost": actual,
        "observable_execution_cost": actual,
        "router_judge_target": judge_target,
        "router_judge_model": judge_model,
        "router_overhead_candidate_spans": router_overhead_candidate_spans,
        "router_overhead_scanned_attempts": router_overhead_scanned_attempts,
        "router_overhead_usage_status": router_overhead_usage_status,
        "router_overhead_marks": router_overhead_marks,
        "router_overhead_usage_covered_marks": router_overhead_usage_covered_marks,
        "router_overhead_usage_coverage": ratio(router_overhead_usage_covered_marks, router_overhead_marks),
        "router_overhead_cost_covered_marks": router_overhead_cost_covered_marks,
        "router_overhead_cost_coverage": ratio(router_overhead_cost_covered_marks, router_overhead_marks),
        "router_overhead_uncovered_marks": router_overhead_marks - router_overhead_cost_covered_marks,
        "router_overhead_cost": router_overhead_cost if router_overhead_cost_covered_marks else 0.0,
        "router_overhead_cost_included": total_cost is not None,
        "covered_model_cost_including_routing": actual + router_overhead_cost,
        "router_overhead_prompt_tokens": sum(task["router_overhead_prompt_tokens"] for task in tasks),
        "router_overhead_completion_tokens": sum(task["router_overhead_completion_tokens"] for task in tasks),
        "router_overhead_reasoning_tokens": sum(task["router_overhead_reasoning_tokens"] for task in tasks),
        "router_overhead_cache_read_tokens": sum(task["router_overhead_cache_read_tokens"] for task in tasks),
        "router_overhead_cache_write_tokens": sum(task["router_overhead_cache_write_tokens"] for task in tasks),
        "total_cost_including_routing": total_cost,
        "counterfactual_baseline_cost": baseline,
        "routing_savings": savings,
        "routing_savings_pct": ratio(savings, baseline),
        "total_savings_including_routing": total_savings,
        "total_savings_including_routing_pct": ratio(total_savings, baseline) if total_savings is not None else None,
        "baseline_model": baseline_model,
        "currency": pricing[baseline_model].get("currency"),
        "model_counts": dict(sorted(model_counts.items())),
        "model_usage": dict(sorted(model_usage.items())),
        "route_target_counts": dict(sorted(target_counts.items())),
        "route_reason_counts": dict(sorted(reason_counts.items())),
        "route_decision_source_counts": dict(sorted(decision_source_counts.items())),
        "routing_confidence_count": sum(task["routing_confidence_count"] for task in tasks),
        "routing_confidence_mean": (
            statistics.mean(
                [float(call["routing_confidence"]) for call in calls if call.get("routing_confidence") is not None]
            )
            if any(call.get("routing_confidence") is not None for call in calls)
            else None
        ),
        "route_decisions": sum(task["route_decisions"] for task in tasks),
        "route_retry_decisions": sum(task["route_retry_decisions"] for task in tasks),
        "fallback_calls": sum(task["fallback_calls"] for task in tasks),
        "prompt_tokens": prompt,
        "completion_tokens": sum(call["completion_tokens"] for call in calls),
        "cache_read_tokens": cache_read,
        "cache_write_tokens": sum(call["cache_write_tokens"] for call in calls),
        "cache_read_ratio": ratio(cache_read, prompt),
        "pricing_catalog": profile["pricing_entries"],
        "runtime_profile": router_profile,
        "configuration_signature": config_signature,
        "task_manifest_signature": task_manifest_signature,
        "scientific_configuration": scientific_config,
    }
    return run, tasks, calls


def _sample_sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _bootstrap_paired_delta(pairs: list[tuple[int, int]], seed: str, samples: int = 10_000) -> list[float] | None:
    if not pairs:
        return None
    generator = random.Random(int(seed[:16], 16))
    deltas: list[float] = []
    for _ in range(samples):
        selected = [pairs[generator.randrange(len(pairs))] for _ in pairs]
        deltas.append(sum(right - left for left, right in selected) / len(selected))
    deltas.sort()
    return [deltas[int(samples * 0.025)], deltas[min(samples - 1, int(samples * 0.975))]]


def _exact_mcnemar_p(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if discordant == 0:
        return None
    tail = sum(math.comb(discordant, value) for value in range(0, min(left_only, right_only) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def aggregate_analysis(runs: list[dict[str, Any]], tasks: list[dict[str, Any]], requested_mode: str) -> dict[str, Any]:
    signatures = {run["configuration_signature"] for run in runs}
    mode = requested_mode
    if mode == "auto":
        mode = "aggregate" if len(signatures) == 1 else "compare"
    if mode == "aggregate" and len(signatures) != 1:
        raise AdmissionError("aggregate mode requires identical scientific configuration signatures")
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault(run["configuration_signature"], []).append(run)
    config_groups = []
    for signature, members in groups.items():
        final_rates = [float(run["pass_at_1"]) for run in members if run["pass_at_1"] is not None]
        observed_rates = [float(run["observed_accuracy"]) for run in members if run["observed_accuracy"] is not None]
        total_costs = [
            float(run["total_cost_including_routing"])
            for run in members
            if run.get("total_cost_including_routing") is not None
        ]
        overhead_costs = [float(run["router_overhead_cost"]) for run in members]
        config_groups.append(
            {
                "configuration_signature": signature,
                "run_labels": [run["label"] for run in members],
                "run_count": len(members),
                "final_pass_at_1_mean": statistics.mean(final_rates) if final_rates else None,
                "final_pass_at_1_sample_sd": _sample_sd(final_rates),
                "observed_accuracy_mean": statistics.mean(observed_rates) if observed_rates else None,
                "observed_accuracy_sample_sd": _sample_sd(observed_rates),
                "observed_accuracy_min": min(observed_rates) if observed_rates else None,
                "observed_accuracy_max": max(observed_rates) if observed_rates else None,
                "total_cost_including_routing_mean": statistics.mean(total_costs) if total_costs else None,
                "total_cost_including_routing_sample_sd": _sample_sd(total_costs),
                "router_overhead_cost_mean": statistics.mean(overhead_costs) if overhead_costs else None,
                "router_overhead_cost_sample_sd": _sample_sd(overhead_costs),
                "pooled_observed_accuracy": ratio(
                    sum(run["benchmark_passes"] for run in members),
                    sum(run["benchmark_complete_tasks"] for run in members),
                ),
            }
        )
    by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for task in tasks:
        by_run.setdefault(task["run_label"], {})[task["task_name"]] = task
    repeatability = None
    if len(runs) >= 2 and len({run["task_manifest_signature"] for run in runs}) == 1:
        counts = Counter()
        common_names = set.intersection(*(set(by_run[run["label"]]) for run in runs))
        incomplete = 0
        for name in common_names:
            observations = [by_run[run["label"]][name]["benchmark_passed"] for run in runs]
            if any(value is None for value in observations):
                incomplete += 1
                continue
            counts[sum(value is True for value in observations)] += 1
        repeatability = {
            "run_count": len(runs),
            "complete_task_count": sum(counts.values()),
            "incomplete_task_count": incomplete,
            "pass_count_distribution": {str(key): counts.get(key, 0) for key in range(len(runs) + 1)},
        }
    comparisons = []
    for left_index, left in enumerate(runs):
        for right in runs[left_index + 1 :]:
            left_tasks = by_run[left["label"]]
            right_tasks = by_run[right["label"]]
            common_names = sorted(set(left_tasks).intersection(right_tasks))
            same_manifest = left["task_manifest_signature"] == right["task_manifest_signature"]
            pairs = []
            incomplete_common = 0
            for name in common_names:
                left_value = left_tasks[name]["benchmark_passed"]
                right_value = right_tasks[name]["benchmark_passed"]
                if left_value is not None and right_value is not None:
                    pairs.append((int(bool(left_value)), int(bool(right_value))))
                else:
                    incomplete_common += 1
            if not same_manifest:
                pairs = []
            left_only = sum(pair == (1, 0) for pair in pairs)
            right_only = sum(pair == (0, 1) for pair in pairs)
            seed = canonical_digest([left["configuration_signature"], right["configuration_signature"]])
            comparisons.append(
                {
                    "left": left["label"],
                    "right": right["label"],
                    "paired_inference_available": same_manifest,
                    "paired_inference_unavailable_reason": None if same_manifest else "task manifest signatures differ",
                    "paired_task_count": len(pairs),
                    "incomplete_common_task_count": incomplete_common,
                    "left_only_task_count": len(set(left_tasks) - set(right_tasks)),
                    "right_only_task_count": len(set(right_tasks) - set(left_tasks)),
                    "pass_rate_delta_right_minus_left": ratio(sum(b - a for a, b in pairs), len(pairs)),
                    "paired_bootstrap_95": _bootstrap_paired_delta(pairs, seed),
                    "left_only_passes": left_only,
                    "right_only_passes": right_only,
                    "mcnemar_exact_p": _exact_mcnemar_p(left_only, right_only),
                    "actual_cost_delta_right_minus_left": right["actual_cost"] - left["actual_cost"],
                }
            )
    return {
        "mode": mode,
        "configuration_groups": config_groups,
        "repeatability": repeatability,
        "pairwise_comparisons": comparisons,
        "all_runs_final": all(run["status"] == "final" for run in runs),
        "same_task_manifest": len({run["task_manifest_signature"] for run in runs}) == 1,
    }
