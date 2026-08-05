# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize static benchmark data in an isolated workspace."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import CONFIG_ROOT, DATA_ROOT


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


def write_plugin_configs(root: Path, otlp_url: str) -> dict[str, Path]:
    """Write the three Relay plugin configurations used for paired runs."""
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
    return paths


def write_fake_codex(root: Path) -> Path:
    """Copy the platform-specific static fake Codex client into the workspace."""
    source_name = "fake-codex.cmd" if os.name == "nt" else "fake-codex.sh"
    target_name = "benchmark-codex.cmd" if os.name == "nt" else "benchmark-codex"
    path = root / target_name
    path.write_text(_read_data(source_name), encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o755)
    return path


def write_agent_config(root: Path, name: str, fake_codex: Path) -> Path:
    path = root / f"{name}-config.toml"
    path.write_text(
        _render_config("agent-config.toml", {'"__CODEX_COMMAND__"': toml_string(fake_codex)}),
        encoding="utf-8",
    )
    return path


def isolated_environment(root: Path) -> dict[str, str]:
    """Return an environment that cannot discover the developer's Relay state."""
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(root / "home"),
            "XDG_CONFIG_HOME": str(root / "xdg-config"),
            "XDG_DATA_HOME": str(root / "xdg-data"),
            "NO_COLOR": "1",
        }
    )
    for directory in ("home", "xdg-config", "xdg-data"):
        (root / directory).mkdir(exist_ok=True)
    return environment
