# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hatch metadata hooks for the LangChain integration package."""

from __future__ import annotations

import tomllib
from email.parser import Parser
from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface


class CargoVersionMetadataHook(MetadataHookInterface):
    """Populate package metadata from the repository Cargo workspace."""

    def update(self, metadata: dict) -> None:
        metadata["version"] = read_version(Path(self.root))


def read_version(root: Path) -> str:
    """Read the package version from the repo Cargo.toml or sdist metadata."""
    cargo_toml = (root / "../.." / "Cargo.toml").resolve()
    if cargo_toml.is_file():
        with cargo_toml.open("rb") as cargo_file:
            cargo_metadata = tomllib.load(cargo_file)
        version = cargo_metadata.get("workspace", {}).get("package", {}).get("version")
        if isinstance(version, str):
            return version

    pkg_info = root / "PKG-INFO"
    if pkg_info.is_file():
        message = Parser().parsestr(pkg_info.read_text(encoding="utf-8"))
        version = message.get("Version")
        if version:
            return version

    raise RuntimeError("Failed to read langchain-nemo-flow version from Cargo.toml or PKG-INFO")
