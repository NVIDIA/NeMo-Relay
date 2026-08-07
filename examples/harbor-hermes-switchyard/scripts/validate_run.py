# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate one Harbor/Hermes/Switchyard Phase 1 task evidence set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parent
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))
from relay_version import require_supported_version
from typing import Any, Iterable

SCHEMA_VERSION = "harbor-hermes-switchyard.validation.v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def is_trial_result(value: dict[str, Any]) -> bool:
    return "task_name" in value and "verifier_result" in value


def read_benchmark_passed(value: dict[str, Any]) -> bool | None:
    reward = value.get("reward")
    if isinstance(reward, dict):
        candidate = reward.get("task_passed")
        if isinstance(candidate, bool):
            return candidate
    verifier = value.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    if isinstance(rewards, dict):
        candidate = rewards.get("task_passed")
        if isinstance(candidate, bool):
            return candidate
        candidate = rewards.get("reward")
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return candidate > 0
    return None


def is_verifier_backed_agent_timeout_nonpass(
    direct_result: dict[str, Any], harbor_result: dict[str, Any]
) -> bool:
    """Recognize a completed benchmark non-pass caused by Harbor's agent deadline.

    Harbor cancels the agent subprocess at its configured deadline, so the
    direct adapter records ``CancelledError`` and cannot emit a final response
    or terminal ATIF/AGENT span.  The outcome is complete only when Harbor
    independently records ``AgentTimeoutError`` and the verifier produces a
    non-passing reward.  Other cancellation and missing-artifact cases remain
    integration failures.
    """
    error = direct_result.get("error")
    exception = harbor_result.get("exception_info")
    return (
        direct_result.get("status") == "failed"
        and isinstance(error, dict)
        and error.get("phase") == "agent"
        and error.get("type") == "CancelledError"
        and isinstance(exception, dict)
        and exception.get("exception_type") == "AgentTimeoutError"
        and read_benchmark_passed(harbor_result) is False
    )


def validate_harbor_job_config(job_dir: Path) -> tuple[dict[str, float], list[str]]:
    """Validate the timeout multipliers serialized by Harbor for this job."""
    errors: list[str] = []
    path = job_dir / "config.json"
    expected = {
        "agent_timeout_multiplier": 3.0,
        "agent_setup_timeout_multiplier": 6.0,
        "environment_build_timeout_multiplier": 6.0,
    }
    if not path.is_file():
        return {}, [f"missing Harbor job config: {path}"]
    config = read_json(path)
    observed: dict[str, float] = {}
    for key, expected_value in expected.items():
        value = config.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"Harbor job config has no numeric {key}")
            continue
        observed[key] = float(value)
        if observed[key] != expected_value:
            errors.append(f"Harbor job config {key}={observed[key]} does not match required {expected_value}")
    return observed, errors


def contained_files(root: Path) -> list[Path]:
    resolved_root = root.resolve(strict=True)
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"artifact symlink is forbidden: {path}")
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        if resolved_root not in resolved.parents:
            raise ValueError(f"artifact escaped its root: {path}")
        files.append(path)
    return files


def scan_files(root: Path) -> list[Path]:
    """Return regular files below a scan root without following symlinks."""
    if not root.is_dir():
        return []
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file():
            files.append(path)
    return files


def scan_secrets(files: Iterable[Path], values: list[bytes]) -> list[str]:
    findings: list[str] = []
    for path in files:
        with path.open("rb") as stream:
            overlap = b""
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                haystack = overlap + block
                for index, secret in enumerate(values):
                    if secret and secret in haystack:
                        findings.append(f"{path.name}:secret[{index}]")
                max_width = max((len(secret) for secret in values), default=1)
                overlap = haystack[-max_width:]
    return sorted(set(findings))


def validate_receipt_provenance(receipt: dict[str, Any], provenance: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    dependencies = receipt.get("dependencies", {})
    relay = dependencies.get("nemo_relay", {})
    hermes = dependencies.get("hermes", {})
    switchyard = dependencies.get("switchyard", {})
    provenance_relay = provenance.get("nemo_relay", {})
    provenance_hermes = provenance.get("hermes", {})
    provenance_switchyard = provenance.get("switchyard", {})
    if relay.get("version") != provenance_relay.get("version"):
        errors.append("receipt Relay version does not match runtime provenance")
    else:
        try:
            require_supported_version(relay.get("version", ""))
        except ValueError:
            errors.append("receipt did not record nemo-relay>=0.7.0")
    if relay.get("wheel_sha256") != provenance_relay.get("wheel_sha256"):
        errors.append("Relay wheel digest does not match runtime provenance")
    if receipt.get("relay_config_sha256") != provenance.get("relay_config_sha256"):
        errors.append("Relay config digest does not match runtime provenance")
    if hermes.get("commit") != provenance_hermes.get("commit"):
        errors.append("Hermes commit does not match runtime provenance")
    if switchyard.get("commit") != provenance_switchyard.get("commit"):
        errors.append("Switchyard commit does not match runtime provenance")
    if switchyard.get("manifest_sha256") != provenance_switchyard.get("manifest_sha256"):
        errors.append("Switchyard manifest digest does not match runtime provenance")
    if switchyard.get("library_sha256") != provenance_switchyard.get("library_sha256"):
        errors.append("Switchyard library digest does not match runtime provenance")
    if receipt.get("dynamic_plugin_ids") != ["nvidia.switchyard"]:
        errors.append("receipt did not record only nvidia.switchyard")
    if receipt.get("activation_mode") != "relay_standard_dynamic":
        errors.append("receipt did not record standard dynamic activation")
    if receipt.get("routing_contract") != {
        "relay_outer_lifecycle": True,
        "execution_intercept_owner": "nvidia.switchyard",
        "provider_http_client_owner": "switchyard-llm-client",
        "separate_switchyard_service": False,
    }:
        errors.append("receipt did not record the expected Relay/Switchyard ownership contract")
    cleanup = receipt.get("cleanup", {})
    if not cleanup.get("plugin_host_closed") or not cleanup.get("exporters_flushed"):
        errors.append("receipt did not prove plugin close and exporter flush")
    return errors


def _otlp_attribute_value(attribute: dict[str, Any]) -> Any:
    value = attribute.get("value")
    if not isinstance(value, dict):
        return None
    for key in ("stringValue", "boolValue", "intValue", "doubleValue"):
        if key in value:
            return value[key]
    return None


def inspect_openinference(path: Path) -> dict[str, Any]:
    documents = 0
    spans = 0
    span_kinds: set[str] = set()
    scope_names: set[str] = set()
    lineage_spans = 0
    resource_attributes: dict[str, set[Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"OpenInference line {line_number} is not an object")
            documents += 1
            for resource_span in payload.get("resourceSpans", []):
                for attribute in resource_span.get("resource", {}).get("attributes", []):
                    key = attribute.get("key")
                    value = _otlp_attribute_value(attribute)
                    if isinstance(key, str) and value is not None:
                        resource_attributes.setdefault(key, set()).add(value)
                for scope_span in resource_span.get("scopeSpans", []):
                    scope_name = scope_span.get("scope", {}).get("name")
                    if isinstance(scope_name, str) and scope_name:
                        scope_names.add(scope_name)
                    for span in scope_span.get("spans", []):
                        spans += 1
                        attributes = {
                            attribute.get("key"): _otlp_attribute_value(attribute)
                            for attribute in span.get("attributes", [])
                            if isinstance(attribute, dict)
                        }
                        kind = attributes.get("openinference.span.kind")
                        if isinstance(kind, str) and kind:
                            span_kinds.add(kind)
                        scope_lineage = attributes.get("nemo_relay.uuid") and attributes.get(
                            "nemo_relay.scope_type"
                        )
                        mark_lineage = (
                            attributes.get("nemo_relay.mark.uuid")
                            and attributes.get("nemo_relay.mark.parent_uuid")
                            and span.get("parentSpanId")
                        )
                        if scope_lineage or mark_lineage:
                            lineage_spans += 1
    return {
        "documents": documents,
        "spans": spans,
        "span_kinds": sorted(span_kinds),
        "scope_names": sorted(scope_names),
        "lineage_spans": lineage_spans,
        "resource_attributes": {key: sorted(values, key=str) for key, values in sorted(resource_attributes.items())},
    }


def inspect_atof(path: Path) -> dict[str, Any]:
    count = 0
    marks: list[str] = []
    models: list[str] = []
    targets: list[str] = []
    cache_read_tokens = 0
    cache_write_tokens = 0
    decision_count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"ATOF line {line_number} is not an object")
            count += 1
            name = payload.get("name")
            if isinstance(name, str) and name.startswith("switchyard.routing."):
                marks.append(name)
                if name == "switchyard.routing.decision":
                    decision_count += 1
                for container in (payload.get("data"), payload.get("metadata")):
                    if isinstance(container, dict):
                        for key in ("model", "selected_model", "target_model"):
                            value = container.get(key)
                            if isinstance(value, str) and value:
                                models.append(value)
                        value = container.get("selected_target")
                        if isinstance(value, str) and value:
                            targets.append(value)
            profile = payload.get("category_profile")
            response = profile.get("annotated_response") if isinstance(profile, dict) else None
            data = payload.get("data")
            if isinstance(response, dict):
                model = response.get("model")
                if isinstance(model, str) and model:
                    models.append(model)
            if name == "openai.chat_completions" and payload.get("scope_category") == "end":
                model = data.get("model") if isinstance(data, dict) else None
                if isinstance(model, str) and model:
                    models.append(model)
            if name == "llm.chunk" and isinstance(data, dict):
                usage = data.get("usage")
                if isinstance(usage, dict):
                    cache_read = usage.get("cache_read_tokens")
                    cache_write = usage.get("cache_write_tokens")
                    if isinstance(cache_read, int) and not isinstance(cache_read, bool):
                        cache_read_tokens += cache_read
                    if isinstance(cache_write, int) and not isinstance(cache_write, bool):
                        cache_write_tokens += cache_write
    return {
        "count": count,
        "marks": sorted(set(marks)),
        "models": sorted(set(models)),
        "targets": sorted(set(targets)),
        "decision_count": decision_count,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
    }


def read_atof(path: Path) -> tuple[int, list[str], list[str], list[str]]:
    """Retain the Phase 1 tuple API while Phase 2 consumes richer evidence."""
    evidence = inspect_atof(path)
    return evidence["count"], evidence["marks"], evidence["models"], evidence["targets"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--openinference", type=Path, required=True)
    parser.add_argument("--harbor-job-dir", type=Path)
    parser.add_argument("--scan-root", type=Path, action="append", default=[])
    parser.add_argument("--expect-late-failure", action="store_true")
    parser.add_argument("--secret-env", action="append", default=[])
    parser.add_argument("--secret-file", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Keep benchmark completion distinct from the optional-but-audited
    # Relay/Switchyard evidence contract. A verifier-backed task result is
    # still a completed Terminal-Bench evaluation if observability evidence is
    # incomplete; the cohort summary decides whether that evidence is strong
    # enough for this integration example to pass as a whole.
    integration_errors: list[str] = []
    benchmark_errors: list[str] = []
    integration_warnings: list[str] = []
    root = args.artifacts.resolve()
    try:
        files = contained_files(root)
    except Exception as error:
        files = []
        integration_errors.append(str(error))

    required = {
        "result": root / "direct-hermes-result.json",
        "receipt": root / "direct-hermes-receipt.json",
        "completion": root / "completion.json",
        "atof": root / "relay" / "trajectory.atof.jsonl",
    }
    for name, path in required.items():
        if not path.is_file():
            target = benchmark_errors if name == "result" else integration_errors
            target.append(f"missing {name}: {path}")
    atif_files = sorted((root / "relay" / "atif").glob("trajectory-*.atif.json"))
    if not atif_files:
        integration_errors.append("missing ATIF trajectory")
    if not args.openinference.is_file() or args.openinference.stat().st_size == 0:
        integration_errors.append("missing OpenInference OTLP artifact")

    result: dict[str, Any] = {}
    receipt: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    if required["result"].is_file():
        result = read_json(required["result"])
        if result.get("status") not in {"completed", "preserved_completed_response"}:
            benchmark_errors.append(f"invalid direct result status: {result.get('status')!r}")
        if not isinstance(result.get("final_response"), str) or not result["final_response"]:
            benchmark_errors.append("direct result has no normalized final response")
        if args.expect_late_failure:
            if result.get("status") != "preserved_completed_response":
                benchmark_errors.append("expected a preserved completed response after late failure")
            if result.get("error", {}).get("type") != "InjectedPostResponseFailure":
                benchmark_errors.append("deterministic post-response failure was not recorded")
    if required["receipt"].is_file():
        receipt = read_json(required["receipt"])
    if args.provenance.is_file():
        provenance = read_json(args.provenance)
    else:
        integration_errors.append("missing runtime provenance")

    if receipt:
        integration_errors.extend(validate_receipt_provenance(receipt, provenance))

    openinference_evidence = {
        "documents": 0,
        "spans": 0,
        "span_kinds": [],
        "scope_names": [],
        "lineage_spans": 0,
        "resource_attributes": {},
    }
    if args.openinference.is_file() and args.openinference.stat().st_size > 0:
        try:
            openinference_evidence = inspect_openinference(args.openinference)
        except Exception as error:
            integration_errors.append(f"invalid OpenInference OTLP artifact: {error}")
        if openinference_evidence["documents"] == 0 or openinference_evidence["spans"] == 0:
            integration_errors.append("OpenInference artifact contains no spans")
        if not {"AGENT", "LLM"}.issubset(openinference_evidence["span_kinds"]):
            integration_errors.append("OpenInference artifact does not contain both AGENT and LLM span kinds")
        if openinference_evidence["scope_names"] != ["harbor-hermes-switchyard"]:
            integration_errors.append("OpenInference instrumentation scope does not match the example")
        if openinference_evidence["lineage_spans"] != openinference_evidence["spans"]:
            integration_errors.append("OpenInference spans are missing Relay UUID or scope-type lineage")
        expected_resources = {
            "openinference.project.name": provenance.get("phoenix_project"),
            "evaluation.cohort": provenance.get("eval_cohort"),
            "service.name": "harbor-hermes-switchyard",
            "service.namespace": "nemo-relay-examples",
        }
        for key, expected in expected_resources.items():
            if openinference_evidence["resource_attributes"].get(key) != [expected]:
                integration_errors.append(f"OpenInference resource attribute {key!r} does not match runtime provenance")

    event_count = 0
    routing_marks: list[str] = []
    routed_models: list[str] = []
    routed_targets: list[str] = []
    switchyard_decision_count = 0
    cache_read_tokens = 0
    cache_write_tokens = 0
    if required["atof"].is_file():
        atof_evidence = inspect_atof(required["atof"])
        event_count = atof_evidence["count"]
        routing_marks = atof_evidence["marks"]
        routed_models = atof_evidence["models"]
        routed_targets = atof_evidence["targets"]
        switchyard_decision_count = atof_evidence["decision_count"]
        cache_read_tokens = atof_evidence["cache_read_tokens"]
        cache_write_tokens = atof_evidence["cache_write_tokens"]
        if event_count == 0:
            integration_errors.append("ATOF artifact is empty")
        if not routing_marks:
            integration_errors.append("ATOF artifact has no Switchyard routing evidence")
        if not routed_targets:
            integration_errors.append("ATOF artifact has no selected Switchyard target")
        unexpected_targets = sorted(set(routed_targets) - {"strong", "weak"})
        if unexpected_targets:
            integration_errors.append(f"ATOF artifact selected unexpected targets: {unexpected_targets}")

    caller_model = provenance.get("routing", {}).get("hermes_caller_model")
    target_models = {
        "strong": provenance.get("routing", {}).get("strong_model"),
        "weak": provenance.get("routing", {}).get("weak_model"),
    }
    routed_models = sorted(
        {
            *routed_models,
            *(target_models[target] for target in routed_targets if target_models.get(target)),
        }
    )
    if caller_model and caller_model in routed_models:
        integration_errors.append("Hermes caller stub appeared as a routed provider model")

    secret_values: list[bytes] = []
    for name in args.secret_env:
        value = os.environ.get(name)
        if value:
            secret_values.append(value.encode())
            if value.startswith("Bearer ") and value[7:]:
                secret_values.append(value[7:].encode())
    for path in args.secret_file:
        for line in path.read_bytes().splitlines():
            value = line.split(b"=", 1)[-1].strip()
            if value:
                secret_values.append(value)
    files_to_scan = list(files)
    if args.openinference.is_file():
        files_to_scan.append(args.openinference)
    for scan_root in args.scan_root:
        files_to_scan.extend(scan_files(scan_root.resolve()))
    findings = scan_secrets(sorted(set(files_to_scan)), secret_values)
    if findings:
        integration_errors.append(f"secret scan found {len(findings)} persisted value(s)")

    harbor_results: list[Path] = []
    harbor_timeout_multipliers: dict[str, float] = {}
    benchmark_passed: bool | None = None
    harbor_result: dict[str, Any] = {}
    if args.harbor_job_dir:
        harbor_timeout_multipliers, timeout_errors = validate_harbor_job_config(args.harbor_job_dir)
        integration_errors.extend(timeout_errors)
        harbor_results = [
            path for path in sorted(args.harbor_job_dir.glob("**/result.json")) if is_trial_result(read_json(path))
        ]
        if len(harbor_results) != 1:
            benchmark_errors.append(f"expected one Harbor trial result, found {len(harbor_results)}")
        elif harbor_results:
            harbor_result = read_json(harbor_results[0])
            benchmark_passed = read_benchmark_passed(harbor_result)
            if benchmark_passed is None:
                benchmark_errors.append("Harbor trial did not contain a normalized benchmark reward")
    else:
        benchmark_errors.append("missing Harbor job directory for benchmark completion")

    terminal_timeout_nonpass = is_verifier_backed_agent_timeout_nonpass(result, harbor_result)
    if terminal_timeout_nonpass:
        tolerated_benchmark_errors = {
            "invalid direct result status: 'failed'",
            "direct result has no normalized final response",
        }
        benchmark_errors = [error for error in benchmark_errors if error not in tolerated_benchmark_errors]
        tolerated_integration_errors = {
            "missing ATIF trajectory",
        }
        if "LLM" in openinference_evidence["span_kinds"]:
            tolerated_integration_errors.add("OpenInference artifact does not contain both AGENT and LLM span kinds")
        for error in integration_errors:
            if error in tolerated_integration_errors:
                integration_warnings.append(error)
        integration_errors = [error for error in integration_errors if error not in tolerated_integration_errors]

    validation = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not benchmark_errors else "failed",
        "errors": benchmark_errors,
        "benchmark": {"status": "passed" if not benchmark_errors else "failed", "errors": benchmark_errors},
        "integration": {
            "status": "passed" if not integration_errors else "failed",
            "errors": integration_errors,
            "warnings": integration_warnings,
        },
        "direct_result_status": result.get("status"),
        "harbor_trial_count": len(harbor_results) if args.harbor_job_dir else None,
        "harbor_timeout_multipliers": harbor_timeout_multipliers,
        "benchmark_task_passed": benchmark_passed,
        "terminal_agent_timeout_nonpass": terminal_timeout_nonpass,
        "atof_event_count": event_count,
        "atif_trajectory_count": len(atif_files),
        "openinference": openinference_evidence,
        "switchyard_routing_marks": routing_marks,
        "switchyard_decision_count": switchyard_decision_count,
        "routed_models": routed_models,
        "routed_targets": routed_targets,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "secret_values_scanned": len(secret_values),
        "secret_findings": findings,
    }
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return 0 if not benchmark_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
