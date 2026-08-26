# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the per-experiment Switchyard/pricing file paths for a plugin template."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

SWITCHYARD_ROUTES_PATH = "/opt/relay-plugins/nvidia.switchyard/switchyard-routes.toml"
SWITCHYARD_PRICING_PATH = "/opt/relay-plugins/nvidia.switchyard/pricing.json"

EXPERIMENT_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")


def experiment_paths(config_dir: Path, experiment: str) -> tuple[Path, Path]:
    """Given the config/ directory and an experiment name, return the source
    (switchyard-routes.toml, pricing.json) file paths for it."""
    if not EXPERIMENT_NAME.fullmatch(experiment):
        raise ValueError(f"invalid switchyard experiment name: {experiment!r}")
    return config_dir / "switchyard" / f"{experiment}.toml", config_dir / "pricing" / f"{experiment}.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plugin_config_identity_sha256(
    template_path: Path, switchyard_path: Path | None, pricing_path: Path | None
) -> str:
    """Combined hash covering the shared template plus its per-experiment
    switchyard-routes/pricing files, so identical shared templates used by
    different experiments still produce distinct identities. Direct-baseline
    configs pass None for both sibling paths."""
    digest = hashlib.sha256()
    digest.update(_sha256_file(template_path).encode())
    for sibling in (switchyard_path, pricing_path):
        if sibling is not None:
            digest.update(_sha256_file(sibling).encode())
    return digest.hexdigest()
