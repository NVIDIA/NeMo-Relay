# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive the per-experiment Switchyard/pricing file paths for a plugin template."""

from __future__ import annotations

import hashlib
from pathlib import Path

SWITCHYARD_ROUTES_PATH = "/opt/relay-plugins/nvidia.switchyard/switchyard-routes.toml"
SWITCHYARD_PRICING_PATH = "/opt/relay-plugins/nvidia.switchyard/pricing.json"


def derive_paired_paths(template_path: Path) -> tuple[Path, Path]:
    """plugins.toml.in -> switchyard/default.toml, pricing/default.json
    plugins.<name>.toml.in -> switchyard/<name>.toml, pricing/<name>.json"""
    stem = template_path.name
    if not stem.startswith("plugins.") or not stem.endswith(".toml.in"):
        raise ValueError(f"unexpected plugin config template name: {stem}")
    name = stem[len("plugins.") : -len(".toml.in")] or "default"
    config_dir = template_path.parent
    return config_dir / "switchyard" / f"{name}.toml", config_dir / "pricing" / f"{name}.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plugin_config_identity_sha256(template_path: Path) -> str:
    """Combined hash covering the shared template plus its per-experiment
    switchyard-routes/pricing files. Multiple experiments now share the same
    physical template file, so hashing it alone can no longer distinguish
    them; direct-baseline templates (no sibling files) fall back to just the
    template hash."""
    digest = hashlib.sha256()
    digest.update(_sha256_file(template_path).encode())
    switchyard_source, pricing_source = derive_paired_paths(template_path)
    for sibling in (switchyard_source, pricing_source):
        if sibling.is_file():
            digest.update(_sha256_file(sibling).encode())
    return digest.hexdigest()
