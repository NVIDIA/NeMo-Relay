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
        "TARGET_MODEL": "phase1-test-model",
        "HERMES_COMMIT": "a07830e086b3055e313b74cc0c8fd5326a4c2c00",
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
    assert plugin["config"]["targets"]["primary"]["header_env"] == {
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
    config = render_template(TARGET_MODEL="namespace/model")
    entry = config["components"][0]["config"]["sources"][0]["catalog"]["entries"][0]
    assert entry["model_id"] == "namespace/model"
    assert "aliases" not in entry
