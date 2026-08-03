#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize a PEP 440 Python package version for release packaging."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

VERSION_PATTERN = re.compile(
    r"^(?P<release>\d+\.\d+\.\d+)"
    r"(?:-(?P<pre_label>alpha|beta|rc)(?:\.(?P<pre_num>\d+))?)?"
    r"(?:\+(?P<local>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


def semver_to_pep440(version: str) -> str:
    """Translate the release version used by Cargo into a PEP 440 version."""
    match = VERSION_PATTERN.fullmatch(version)
    if not match:
        raise ValueError(
            "Unsupported Python package version format. Expected SemVer with optional "
            "alpha/beta/rc prerelease and optional build metadata."
        )

    pep440 = match.group("release")
    pre_label = match.group("pre_label")
    if pre_label:
        pre_map = {"alpha": "a", "beta": "b", "rc": "rc"}
        pep440 += f"{pre_map[pre_label]}{match.group('pre_num') or '0'}"

    local = match.group("local")
    if local:
        normalized_local = ".".join(part.lower() for part in re.split(r"[._-]+", local) if part)
        if not normalized_local:
            raise ValueError("Python package local version metadata cannot be empty")
        pep440 += f"+{normalized_local}"

    return pep440


def materialize_python_version(source: Path, version: str) -> None:
    """Set the explicit Python package version in a packaging source tree."""
    pyproject = source / "pyproject.toml"
    text = pyproject.read_text()
    if 'dynamic = ["version"]' not in text:
        raise ValueError("Failed to find dynamic version field in pyproject.toml")
    pyproject.write_text(text.replace('dynamic = ["version"]', f'version = "{version}"', 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Raw SemVer release version")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.cwd(),
        help="Packaging source tree containing pyproject.toml",
    )
    args = parser.parse_args()
    materialize_python_version(args.source.resolve(), semver_to_pep440(args.version))


if __name__ == "__main__":
    main()
