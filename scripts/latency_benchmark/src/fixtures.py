# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize static benchmark data in an isolated workspace."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import CONFIG_ROOT, DATA_ROOT, MiddlewareVariant

_BLOCKED_ENVIRONMENT_PREFIXES = (
    "ANTHROPIC_",
    "NEMO_RELAY",
    "OPENAI_",
    "OTEL_",
    "RUST_LOG",
)
_PROXY_ENVIRONMENT_NAMES = {
    "ALL_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
}


def _read_data(name: str) -> str:
    return (DATA_ROOT / name).read_text(encoding="utf-8")


def _read_config(name: str) -> str:
    return (CONFIG_ROOT / name).read_text(encoding="utf-8")


def _render_config(name: str, replacements: dict[str, str]) -> str:
    rendered = _read_config(name)
    for marker, value in replacements.items():
        if marker not in rendered:
            raise RuntimeError(f"static fixture {name} is missing marker {marker}")
        rendered = rendered.replace(marker, value)
    return rendered


def toml_string(value: str | Path) -> str:
    """Encode a string using TOML-compatible JSON quoting."""
    return json.dumps(str(value))


def write_relay_config(root: Path) -> Path:
    path = root / "config.toml"
    path.write_text(_read_config("relay-config.toml"), encoding="utf-8")
    return path


def write_plugin_configs(
    root: Path,
    otlp_url: str,
    middleware: tuple[MiddlewareVariant, ...] = (),
) -> dict[str, Path]:
    """Write default plugin configs and add opt-in middleware variants."""
    paths = {
        "relay-minimal": root / "plugins-minimal.toml",
        "relay-file": root / "plugins-file.toml",
        "relay-otlp": root / "plugins-otlp.toml",
    }
    paths["relay-minimal"].write_text(_read_config("plugins-minimal.toml"), encoding="utf-8")

    atof_dir = root / "atof"
    atof_dir.mkdir()
    paths["relay-file"].write_text(
        _render_config("plugins-file.toml", {'"__ATOF_OUTPUT_DIRECTORY__"': toml_string(atof_dir)}),
        encoding="utf-8",
    )
    paths["relay-otlp"].write_text(
        _render_config("plugins-otlp.toml", {'"__OTLP_ENDPOINT__"': toml_string(f"{otlp_url}/v1/traces")}),
        encoding="utf-8",
    )
    paths.update((variant.relay_name, variant.plugin_config) for variant in middleware)
    return paths


def write_mock_codex(root: Path) -> Path:
    """Copy the platform-specific static mock Codex client into the workspace."""
    source_name = "mock-codex.cmd" if os.name == "nt" else "mock-codex.sh"
    target_name = "mock-codex.cmd" if os.name == "nt" else "mock-codex"
    path = root / target_name
    newline = "\r\n" if os.name == "nt" else "\n"
    path.write_text(_read_data(source_name), encoding="utf-8", newline=newline)
    if os.name != "nt":
        path.chmod(0o755)
    return path


def write_agent_config(root: Path, name: str, mock_codex: Path) -> Path:
    path = root / f"{name}-config.toml"
    path.write_text(
        _render_config("agent-config.toml", {'"__CODEX_COMMAND__"': toml_string(mock_codex)}),
        encoding="utf-8",
    )
    return path


def isolated_environment(root: Path) -> dict[str, str]:
    """Return an environment that cannot discover the developer's Relay state."""
    environment = os.environ.copy()
    for name in tuple(environment):
        normalized = name.upper()
        if normalized.startswith(_BLOCKED_ENVIRONMENT_PREFIXES) or normalized in _PROXY_ENVIRONMENT_NAMES:
            environment.pop(name)
    environment.update(
        {
            "HOME": str(root / "home"),
            "NO_PROXY": "127.0.0.1,localhost",
            "XDG_CONFIG_HOME": str(root / "xdg-config"),
            "XDG_DATA_HOME": str(root / "xdg-data"),
            "NO_COLOR": "1",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    for directory in ("home", "xdg-config", "xdg-data"):
        (root / directory).mkdir(exist_ok=True)
    return environment
