# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import tomllib
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


def render_template(**values: str) -> dict:
    text = (EXAMPLE_ROOT / "config" / "relay.toml.in").read_text(encoding="utf-8")
    defaults = {
        "STRONG_MODEL": "phase1-test-strong",
        "WEAK_MODEL": "phase1-test-weak",
        "HERMES_CALLER_MODEL": "ollama-route-stub",
        "HERMES_COMMIT": "efb63e714abc436af88af9b0d6734751c199aa6d",
        "OPENINFERENCE_ENDPOINT": "http://127.0.0.1:4318/v1/traces",
        "PHOENIX_PROJECT": "phase1-test",
        "EVAL_COHORT": "phase1-test",
        "UPSTREAM_BASE_URL": "http://127.0.0.1:8000/v1",
        "UPSTREAM_AUTH_ENV": "SWITCHYARD_PROVIDER_AUTHORIZATION",
    }
    defaults.update(values)
    for key, value in defaults.items():
        text = text.replace(f"@{key}@", value)
    assert not re.search(r"@[A-Z0-9_]+@", text)
    return tomllib.loads(text)


def test_config_uses_static_schema_v3_and_one_standard_dynamic_plugin() -> None:
    config = render_template()
    assert config["version"] == 1
    components = {item["kind"]: item for item in config["components"]}
    assert components["pricing"]["enabled"] is True
    assert components["observability"]["config"]["version"] == 3
    assert "dynamic_plugins" not in config
    assert len(config["plugins"]["dynamic"]) == 1
    plugin = config["plugins"]["dynamic"][0]
    assert plugin["manifest"].endswith("/nvidia.switchyard/relay-plugin.toml")
    algorithm = plugin["config"]["algorithm"]
    assert algorithm == {
        "kind": "llm_classifier",
        "classifier_target": "weak",
        "weak_target": "weak",
        "strong_target": "strong",
        "base_threshold": 0.5,
        "min_confidence": 0.0,
        "recent_turn_window": 0,
        "session_affinity": True,
        "message_hash_fallback": True,
    }
    assert plugin["config"]["default_targets"] == {"openai_chat": "strong"}
    assert set(plugin["config"]["targets"]) == {"strong", "weak"}
    for target in plugin["config"]["targets"].values():
        assert target["header_env"] == {
            "authorization": "SWITCHYARD_PROVIDER_AUTHORIZATION"
        }


def test_config_contains_no_literal_provider_headers_or_credentials() -> None:
    config = render_template()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            assert "headers" not in value
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(config)


def test_pricing_does_not_duplicate_relay_generated_aliases() -> None:
    config = render_template(
        STRONG_MODEL="namespace/strong",
        WEAK_MODEL="namespace/weak",
    )
    entries = config["components"][0]["config"]["sources"][0]["catalog"]["entries"]
    assert [entry["model_id"] for entry in entries] == ["namespace/strong", "namespace/weak"]
    assert all("aliases" not in entry for entry in entries)


def test_switchyard_models_are_distinct_from_fail_closed_hermes_caller() -> None:
    config = render_template()
    plugin = config["plugins"]["dynamic"][0]
    provider_models = {target["model"] for target in plugin["config"]["targets"].values()}
    assert provider_models == {"phase1-test-strong", "phase1-test-weak"}
    observability = next(item for item in config["components"] if item["kind"] == "observability")
    assert observability["config"]["atif"]["model_name"] == "ollama-route-stub"
    assert "ollama-route-stub" not in provider_models


def test_task_runner_defaults_to_production_x86_64_architecture() -> None:
    runner = (EXAMPLE_ROOT / "run_terminal_bench.sh").read_text(encoding="utf-8")
    assert 'relay_architecture="${RELAY_ARCHITECTURE:-x86_64}"' in runner
    assert '--relay-architecture "$relay_architecture"' in runner
    assert 'SWITCHYARD_TARGET_ARCHITECTURE="$relay_architecture"' in runner
    assert '--ak "relay_architecture=$relay_architecture"' in runner


def test_task_runner_defaults_to_inference_hub_tiers_and_fail_closed_caller() -> None:
    runner = (EXAMPLE_ROOT / "run_terminal_bench.sh").read_text(encoding="utf-8")
    assert "aws/anthropic/bedrock-claude-opus-4-6" in runner
    assert "aws/anthropic/bedrock-claude-sonnet-4-6" in runner
    assert 'hermes_caller_model="${HERMES_CALLER_MODEL:-ollama-route-stub}"' in runner
    assert '--model "openai/$hermes_caller_model"' in runner
    assert 'fail_closed_openai_base_url="http://127.0.0.1:9/v1"' in runner
    assert '--ae "OPENAI_BASE_URL=$fail_closed_openai_base_url"' in runner
