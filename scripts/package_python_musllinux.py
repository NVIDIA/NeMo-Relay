#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a musllinux wheel without modifying the checked-out source tree."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from python_package_version import materialize_python_version, semver_to_pep440


def copy_source(source: Path, destination: Path) -> None:
    """Copy only source inputs needed by Maturin into the disposable workspace."""
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", ".venv", "target", "tmp", "__pycache__"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Raw SemVer release version")
    parser.add_argument("--out", required=True, type=Path, help="Directory for the built wheel")
    parser.add_argument("--interpreter", required=True, help="CPython executable used to build the ABI3 wheel")
    args = parser.parse_args()

    source = Path.cwd().resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    version = semver_to_pep440(args.version)

    with tempfile.TemporaryDirectory(prefix="nemo-relay-musllinux-") as temporary_directory:
        build_source = Path(temporary_directory) / "source"
        copy_source(source, build_source)
        materialize_python_version(build_source, version)
        subprocess.run(
            [
                "maturin",
                "build",
                "--release",
                "--compatibility",
                "musllinux_1_2",
                "--interpreter",
                args.interpreter,
                "--out",
                str(output),
            ],
            check=True,
            cwd=build_source,
        )

    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one musllinux wheel in {output}, found {len(wheels)}")


if __name__ == "__main__":
    main()
