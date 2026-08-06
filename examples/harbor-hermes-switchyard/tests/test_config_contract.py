# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import importlib.util
import re
import sys
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock, patch

from harbor.agents.installed.hermes import Hermes

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


def render_template(**values: str) -> dict:
    text = (EXAMPLE_ROOT / "config" / "plugins.toml.in").read_text(encoding="utf-8")
    defaults = {
        "HERMES_COMMIT": "efb63e714abc436af88af9b0d6734751c199aa6d",
        "OPENINFERENCE_ENDPOINT": "http://127.0.0.1:4318/v1/traces",
        "PHOENIX_PROJECT": "phase1-test",
        "EVAL_COHORT": "phase1-test",
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
        "recent_turn_window": 0,
        "session_affinity": True,
        "message_hash_fallback": True,
    }
    assert plugin["config"]["default_targets"] == {"openai_chat": "strong"}
    assert set(plugin["config"]["targets"]) == {"strong", "weak"}
    for target in plugin["config"]["targets"].values():
        assert target["header_env"] == {"authorization": "SWITCHYARD_PROVIDER_AUTHORIZATION"}
        assert target["drop_caller_extra_body"] is True


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
    config = render_template()
    entries = config["components"][0]["config"]["sources"][0]["catalog"]["entries"]
    assert [entry["model_id"] for entry in entries] == [
        "aws/anthropic/bedrock-claude-opus-4-6",
        "aws/anthropic/bedrock-claude-sonnet-4-6",
    ]
    assert all("aliases" not in entry for entry in entries)


def test_pricing_uses_nonzero_claude_46_list_rates() -> None:
    config = render_template()
    entries = config["components"][0]["config"]["sources"][0]["catalog"]["entries"]
    rates = {entry["model_id"]: entry["rates"] for entry in entries}
    assert rates["aws/anthropic/bedrock-claude-opus-4-6"] == {
        "input_per_million": 5.0,
        "output_per_million": 25.0,
        "cache_read_per_million": 0.5,
        "cache_write_per_million": 6.25,
    }
    assert rates["aws/anthropic/bedrock-claude-sonnet-4-6"] == {
        "input_per_million": 3.0,
        "output_per_million": 15.0,
        "cache_read_per_million": 0.3,
        "cache_write_per_million": 3.75,
    }


def test_switchyard_models_are_distinct_from_fail_closed_hermes_caller() -> None:
    config = render_template()
    plugin = config["plugins"]["dynamic"][0]
    provider_models = {target["model"] for target in plugin["config"]["targets"].values()}
    assert provider_models == {
        "aws/anthropic/bedrock-claude-opus-4-6",
        "aws/anthropic/bedrock-claude-sonnet-4-6",
    }
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
    assert "STRONG_MODEL" not in runner
    assert "WEAK_MODEL" not in runner
    assert 'plugin_config_template="${PLUGIN_CONFIG_TEMPLATE:-$example_root/config/plugins.toml.in}"' in runner
    assert '--model "openai/$hermes_caller_model"' in runner
    assert 'fail_closed_openai_base_url="http://127.0.0.1:9/v1"' in runner
    assert '--ae "OPENAI_BASE_URL=$fail_closed_openai_base_url"' in runner


def test_task_runner_projects_provider_authorization_by_read_only_mount() -> None:
    runner = (EXAMPLE_ROOT / "run_terminal_bench.sh").read_text(encoding="utf-8")
    assert '--ae "$upstream_auth_env=' not in runner
    assert 'host_temporary_root="$(cd "${TMPDIR:-/tmp}" && pwd -P)"' in runner
    assert '"$(dirname "$run_root")/.phase2-secret.' not in runner
    assert '"$run_root/"*)' in runner
    assert 'provider_authorization_target="/run/secrets/switchyard-provider-authorization"' in runner
    assert '"read_only": True' in runner
    assert '"bind": {"create_host_path": False}' in runner
    assert '--mounts "$mounts_json"' in runner
    assert "'OPENAI_API_KEY=${OPENAI_API_KEY}'" in runner


def test_agent_reads_provider_authorization_inside_container_only() -> None:
    path = EXAMPLE_ROOT / "agents" / "harbor_hermes_agent.py"
    spec = importlib.util.spec_from_file_location("phase2_secret_agent", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    agent = object.__new__(module.HarborHermesAgent)
    agent._load_provider_authorization = True
    with patch.object(Hermes, "exec_as_agent", new_callable=AsyncMock) as parent:
        asyncio.run(agent.exec_as_agent(object(), "hermes --yolo chat", env={"SAFE": "value"}))
    command = parent.await_args.args[-1]
    assert "cat -- /run/secrets/switchyard-provider-authorization" in command
    assert 'export SWITCHYARD_PROVIDER_AUTHORIZATION="$(cat --' in command
    assert parent.await_args.kwargs["env"] == {"SAFE": "value"}


def test_agent_install_retries_transient_apt_failures() -> None:
    agent = (EXAMPLE_ROOT / "agents" / "harbor_hermes_agent.py").read_text(encoding="utf-8")
    assert "for attempt in 1 2 3; do " in agent
    assert "apt-get update && apt-get install -y --no-install-recommends " in agent
    assert "sleep $((attempt * 5))" in agent


def test_phase2_environment_template_consolidates_secret_without_legacy_file() -> None:
    template = (EXAMPLE_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "SWITCHYARD_PROVIDER_AUTHORIZATION='Bearer replace-with-provider-token'" in template
    assert "INFERENCE_SECRETS_FILE" not in template
    assert "NV_INFERENCEHUB_KEY" not in template
    assert "NV_INFERENCEHUB_ENDPOINT" not in template
    assert "STRONG_MODEL" not in template
    assert "WEAK_MODEL" not in template
    assert "UPSTREAM_BASE_URL" not in template
    assert ".env" in (EXAMPLE_ROOT / ".gitignore").read_text(encoding="utf-8")


def test_readme_uses_admissions_instead_of_regression_smokes() -> None:
    readme = (EXAMPLE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "run_regression_smokes.sh" not in readme
    assert "PHASE1_EVIDENCE_ROOT" not in readme
    assert "INFERENCE_SECRETS_FILE" not in readme
    assert "all-89 no-token admission" in readme.lower()
    assert "Docker offline runtime admission" in readme


def test_phase2_runner_requires_and_uses_the_local_dataset_export() -> None:
    runner = (EXAMPLE_ROOT / "run_terminal_bench.sh").read_text(encoding="utf-8")
    assert 'tbench_dataset_path="${TBENCH_DATASET_PATH:-}"' in runner
    assert 'dataset_args=(--path "$tbench_dataset_path")' in runner
    assert '"${dataset_args[@]}"' in runner
    assert "Phase 2 requires TBENCH_DATASET_PATH" in runner
