# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provision disposable agent hosts and provider configuration for the documented commands."""

from __future__ import annotations

import json
import re
import shlex
import tomllib
from pathlib import Path
from typing import Any

from quickstarts.provider import Fixture

PYTHON_CASES = {"python", "langgraph", "langchain", "deep-agents"}
HERMES_COMMIT = "4d04303818b61bf3aa09278af48356315515965b"


def prepare(
    name: str,
    work: Path,
    folder: Path,
    environment: dict[str, str],
    resolved: dict[str, Any],
    fixture: Fixture,
    execute: Any,
    hermes_ref: str = HERMES_COMMIT,
) -> dict[str, Any]:
    def run(label: str, command: str, timeout: int = 600) -> None:
        execute(command, work, environment, folder / (label + ".stdout.log"), folder / (label + ".stderr.log"), timeout)

    url = fixture.url
    if name in PYTHON_CASES:
        environment.update(NVIDIA_API_KEY="quickstart-fixture", NVIDIA_BASE_URL=url + "/v1")
        return {"provider": "local-chat-completions"}
    binary_dir = work / "bin"
    binary_dir.mkdir()
    environment["PATH"] = str(binary_dir) + ":" + environment["PATH"]
    if name != "hermes":
        (binary_dir / "nemo-relay").symlink_to(resolved["cli"])
    if name == "codex":
        managed = Path("/etc/codex/managed_config.toml")
        if managed.is_file() and tomllib.loads(managed.read_text()).get("model_provider"):
            raise ValueError(
                "System-managed Codex provider overrides fixture isolation; use a clean CI host. "
                "Policy file: /etc/codex/managed_config.toml"
            )
        config = work / ".codex"
        config.mkdir()
        environment["CODEX_HOME"] = str(config)
        environment["OPENAI_API_KEY"] = "quickstart-fixture"
        (config / "config.toml").write_text(
            'model = "gpt-5.3-codex"\nmodel_provider = "quickstart"\napproval_policy = "never"\n'
            '[model_providers.quickstart]\nname = "Local Quick Start fixture"\nwire_api = "responses"\n'
            f'base_url = "{url}/v1"\nenv_key = "OPENAI_API_KEY"\n'
            f'[projects.{json.dumps(str(fixture.project))}]\ntrust_level = "trusted"\n'
        )
        run("agent-version", "codex --version")
    elif name == "claude-code":
        config = work / ".claude"
        config.mkdir()
        environment.update(
            CLAUDE_CONFIG_DIR=str(config),
            CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
            DISABLE_AUTOUPDATER="1",
            ANTHROPIC_AUTH_TOKEN="quickstart-fixture",
        )
        (config / ".claude.json").write_text(json.dumps({"hasCompletedOnboarding": True, "theme": "dark"}))
        run("agent-version", "claude --version")
    elif name == "pi":
        config = work / ".pi"
        config.mkdir()
        environment["PI_CODING_AGENT_DIR"] = str(config)
        environment["NVIDIA_API_KEY"] = "quickstart-fixture"
        # Defaults let the guide's pi command run unchanged, through Relay's real extension.
        (config / "settings.json").write_text(
            json.dumps({"defaultProvider": "quickstart", "defaultModel": "relay-quickstart"})
        )
        (config / "models.json").write_text(
            json.dumps(
                {
                    "providers": {
                        "quickstart": {
                            "baseUrl": url + "/v1",
                            "api": "openai-completions",
                            "apiKey": "quickstart-fixture",
                            "models": [{"id": "relay-quickstart", "contextWindow": 128000, "maxTokens": 4096}],
                        }
                    }
                }
            )
        )
    elif name == "hermes":
        if not re.fullmatch(r"[0-9a-f]{40}", hermes_ref):
            raise ValueError("Hermes ref must be an immutable full commit SHA")
        source = work / "hermes-source"
        q = shlex.quote
        run(
            "hermes-clone",
            f"git init {q(str(source))}\ngit -C {q(str(source))} remote add origin https://github.com/NousResearch/hermes-agent.git\n"
            f"git -C {q(str(source))} fetch --depth 1 origin {hermes_ref}\n"
            f"git -C {q(str(source))} checkout --detach FETCH_HEAD",
        )
        # Hermes currently supports runtime dependency installation on Python 3.14.
        # Install Hermes first, then explicitly select the candidate Relay wheel for compatibility testing.
        venv = work / "hermes-venv"
        environment.pop("UV_PYTHON_DOWNLOADS", None)
        run(
            "hermes-install",
            f"uv venv --python 3.14 --managed-python {q(str(venv))}\n"
            f"uv pip install --python {q(str(venv / 'bin/python'))} --editable {q(str(source))}\n"
            f"uv pip install --python {q(str(venv / 'bin/python'))} --reinstall {q(str(resolved['python']))}",
            1200,
        )
        environment["PATH"] = str(venv / "bin") + ":" + environment["PATH"]
        config = work / ".hermes"
        config.mkdir()
        environment.update(HERMES_HOME=str(config), OPENAI_BASE_URL=url + "/v1", OPENAI_API_KEY="quickstart-fixture")
        (config / "config.yaml").write_text(
            "model:\n  provider: custom\n  default: relay-quickstart\n"
            f'  base_url: "{url}/v1"\n  api_key: quickstart-fixture\n'
            "terminal:\n  backend: local\n  cwd: .\ntelemetry:\n  enabled: false\n"
        )
        run("agent-version", "hermes --version")
        return {
            "provider": "local-chat-completions",
            "commit": hermes_ref,
            "relay_dependency_override": "Selected wheel installed after the Hermes dependency resolution",
        }
    else:
        raise ValueError(f"No agent provisioner for {name}")
    version = (folder / "agent-version.stdout.log").read_text().strip() if name != "pi" else "0.84.0 (guide pin)"
    return {"version": version, "provider": "local-protocol-fixture"}


def command(block: dict[str, Any], name: str, work: Path, fixture: Fixture) -> tuple[str | None, str]:
    """Handle declared manual choices explicitly; reject unknown additions in the runner."""
    if name == "hermes" and block["role"] == "manual":
        reasons = {
            "install": "Hermes provisioned from the recorded immutable source commit",
            "nvidia-endpoint": "Optional provider alternative; using the configured local provider",
            "nvidia-run": "Optional alternative; executing the provider-neutral run block",
            "gateway-stop": "Fresh HERMES_HOME has no gateway; no existing user service is stopped",
        }
        if block["id"] in reasons:
            return None, reasons[block["id"]]
        if block["id"] == "doctor":
            # Doctor also checks unrelated optional services. Preserve its diagnostics;
            # actual native Relay lifecycle checks remain mandatory below.
            return "hermes doctor", "diagnostic"
    if block["role"] == "input" and block["id"] == "credentials" and block.get("env") == "NVIDIA_API_KEY":
        return None, "Dummy key provisioned for the loopback provider"
    if name in {"codex", "claude-code", "pi"} and block["id"] == "install":
        if name == "pi":
            match = re.search(r"@earendil-works/pi-coding-agent@([\d.]+)", block["content"])
            if not match:
                raise ValueError("The pi install pin changed; review its unattended adapter")
            prefix = work / "agent-npm"
            return (
                f"npm install --prefix {shlex.quote(str(prefix))} --ignore-scripts "
                f"@earendil-works/pi-coding-agent@{match[1]}\n"
                f'export PATH={shlex.quote(str(prefix / "node_modules/.bin"))}:"$PATH"\n'
                "pi --version\nnemo-relay --version"
            ), "isolated-install"
        return "nemo-relay --version", "Selected CLI binary replaces the user-level installer"
    return block["content"], "verbatim"


def gateway_config(content: str, fixture: Fixture) -> str:
    return content + f'\n[upstream]\nopenai_base_url = "{fixture.url}/v1"\nanthropic_base_url = "{fixture.url}"\n'
