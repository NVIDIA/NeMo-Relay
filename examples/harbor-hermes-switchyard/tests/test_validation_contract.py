# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = EXAMPLE_ROOT / "scripts" / "validate_run.py"
    spec = importlib.util.spec_from_file_location("phase1_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harbor_job_summary_is_not_a_trial_result() -> None:
    module = load_validator()
    assert module.is_trial_result({"n_total_trials": 1, "stats": {}}) is False
    assert module.is_trial_result({"task_name": "task", "verifier_result": {}}) is True


def test_harbor_018_numeric_reward_is_normalized() -> None:
    module = load_validator()
    assert module.read_benchmark_passed({"verifier_result": {"rewards": {"reward": 0.0}}}) is False
    assert module.read_benchmark_passed({"verifier_result": {"rewards": {"reward": 1.0}}}) is True


def test_verifier_backed_agent_timeout_is_a_completed_nonpass() -> None:
    module = load_validator()
    direct_result = {
        "status": "failed",
        "error": {"phase": "agent", "type": "CancelledError"},
    }
    harbor_result = {
        "exception_info": {"exception_type": "AgentTimeoutError"},
        "verifier_result": {"rewards": {"reward": 0.0}},
    }
    assert module.is_verifier_backed_agent_timeout_nonpass(direct_result, harbor_result) is True

    harbor_result["exception_info"]["exception_type"] = "RuntimeError"
    assert module.is_verifier_backed_agent_timeout_nonpass(direct_result, harbor_result) is False

    harbor_result["exception_info"]["exception_type"] = "AgentTimeoutError"
    harbor_result["verifier_result"]["rewards"]["reward"] = 1.0
    assert module.is_verifier_backed_agent_timeout_nonpass(direct_result, harbor_result) is False


def test_harbor_timeout_multipliers_are_validated(tmp_path: Path) -> None:
    module = load_validator()
    config = {
        "agent_timeout_multiplier": 3.0,
        "agent_setup_timeout_multiplier": 6.0,
        "environment_build_timeout_multiplier": 6.0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    observed, errors = module.validate_harbor_job_config(tmp_path)
    assert observed == config
    assert errors == []

    config["agent_timeout_multiplier"] = 1.0
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    _, errors = module.validate_harbor_job_config(tmp_path)
    assert errors == ["Harbor job config agent_timeout_multiplier=1.0 does not match required 3.0"]


def test_atof_reader_extracts_switchyard_selected_targets(tmp_path: Path) -> None:
    module = load_validator()
    path = tmp_path / "trajectory.atof.jsonl"
    events = [
        {"name": "switchyard.routing.requested", "data": {"algorithm": "llm_task_classifier"}},
        {
            "name": "switchyard.routing.decision",
            "data": {"selected_target": "weak", "routing_tier": "weak"},
        },
        {
            "name": "switchyard.routing.decision",
            "data": {"selected_target": "strong", "routing_tier": "strong"},
        },
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    count, marks, models, targets = module.read_atof(path)
    assert count == 3
    assert marks == ["switchyard.routing.decision", "switchyard.routing.requested"]
    assert models == []
    assert targets == ["strong", "weak"]


def test_atof_inspection_extracts_provider_models_and_cache_usage(tmp_path: Path) -> None:
    module = load_validator()
    path = tmp_path / "trajectory.atof.jsonl"
    events = [
        {
            "name": "switchyard.routing.decision",
            "data": {"selected_target": "weak"},
        },
        {
            "name": "openai.chat_completions",
            "category_profile": {"annotated_response": {"model": "aws/anthropic/bedrock-claude-sonnet-4-6"}},
            "data": {"model": "aws/anthropic/bedrock-claude-sonnet-4-6"},
        },
        {
            "name": "llm.chunk",
            "data": {"usage": {"cache_read_tokens": 120, "cache_write_tokens": 30}},
        },
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    evidence = module.inspect_atof(path)
    assert evidence == {
        "count": 3,
        "marks": ["switchyard.routing.decision"],
        "models": ["aws/anthropic/bedrock-claude-sonnet-4-6"],
        "targets": ["weak"],
        "decision_count": 1,
        "cache_read_tokens": 120,
        "cache_write_tokens": 30,
    }


def test_secret_scan_finds_raw_key_within_artifact(tmp_path: Path) -> None:
    module = load_validator()
    artifact = tmp_path / "artifact.log"
    artifact.write_text("raw-provider-key")
    assert module.scan_secrets([artifact], [b"Bearer raw-provider-key", b"raw-provider-key"]) == [
        "artifact.log:secret[1]"
    ]


def test_receipt_provenance_validation_covers_every_staged_digest() -> None:
    module = load_validator()
    provenance = {
        "nemo_relay": {"version": "0.7.1", "wheel_sha256": "relay"},
        "hermes": {"commit": "hermes"},
        "switchyard": {
            "commit": "switchyard",
            "manifest_sha256": "manifest",
            "library_sha256": "library",
        },
        "relay_config_sha256": "config",
    }
    receipt = {
        "activation_mode": "relay_standard_dynamic",
        "dynamic_plugin_ids": ["nvidia.switchyard"],
        "relay_config_sha256": "config",
        "dependencies": {
            "nemo_relay": {"version": "0.7.1", "wheel_sha256": "relay"},
            "hermes": {"commit": "hermes"},
            "switchyard": {
                "commit": "switchyard",
                "manifest_sha256": "manifest",
                "library_sha256": "library",
            },
        },
        "routing_contract": {
            "relay_outer_lifecycle": True,
            "execution_intercept_owner": "nvidia.switchyard",
            "provider_http_client_owner": "switchyard-llm-client",
            "separate_switchyard_service": False,
        },
        "cleanup": {"plugin_host_closed": True, "exporters_flushed": True},
    }
    assert module.validate_receipt_provenance(receipt, provenance) == []
    receipt["dependencies"]["switchyard"]["library_sha256"] = "tampered"
    assert module.validate_receipt_provenance(receipt, provenance) == [
        "Switchyard library digest does not match runtime provenance"
    ]


def test_openinference_inspection_extracts_semantic_and_lineage_evidence(tmp_path: Path) -> None:
    module = load_validator()
    artifact = tmp_path / "openinference.jsonl"
    payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "openinference.project.name", "value": {"stringValue": "project"}},
                        {"key": "evaluation.cohort", "value": {"stringValue": "cohort"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "harbor-hermes-switchyard"},
                        "spans": [
                            {
                                "attributes": [
                                    {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
                                    {"key": "nemo_relay.uuid", "value": {"stringValue": "uuid"}},
                                    {"key": "nemo_relay.scope_type", "value": {"stringValue": "llm"}},
                                ]
                            },
                            {
                                "parentSpanId": "parent-span",
                                "attributes": [
                                    {"key": "openinference.span.kind", "value": {"stringValue": "CHAIN"}},
                                    {"key": "nemo_relay.mark.uuid", "value": {"stringValue": "mark-uuid"}},
                                    {
                                        "key": "nemo_relay.mark.parent_uuid",
                                        "value": {"stringValue": "parent-uuid"},
                                    },
                                ],
                            },
                            {
                                "attributes": [
                                    {"key": "openinference.span.kind", "value": {"stringValue": "CHAIN"}},
                                    {"key": "nemo_relay.mark.uuid", "value": {"stringValue": "orphan-mark"}},
                                ],
                            },
                        ],
                    }
                ],
            }
        ]
    }
    artifact.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence = module.inspect_openinference(artifact)
    assert evidence["documents"] == 1
    assert evidence["spans"] == 3
    assert evidence["span_kinds"] == ["CHAIN", "LLM"]
    assert evidence["scope_names"] == ["harbor-hermes-switchyard"]
    assert evidence["lineage_spans"] == 2
    assert evidence["resource_attributes"]["openinference.project.name"] == ["project"]
