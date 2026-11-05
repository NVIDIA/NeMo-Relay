# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build host assets from one immutable Relay checkout, leaving the user's tree alone."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from quickstarts import shell
from quickstarts.assets import host_platform, sha256


def checkout(root: Path, ref: str, output: Path) -> tuple[Path, str, dict[str, Any]]:
    """Pin and inspect source before selecting cases or starting expensive builds."""
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"], text=True, timeout=60
    ).strip()
    source, packages = output / "source", output / "packages"
    packages.mkdir()
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(root), str(source)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    subprocess.run(
        ["git", "-C", str(source), "checkout", "--detach", commit], check=True, capture_output=True, timeout=120
    )
    version = tomllib.loads((source / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    record = {"commit": commit, "ref": ref, "profile": "dev", "status": "checked-out"}
    (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    return source, version, record


def prepare(root: Path, source: Path, output: Path, surfaces: set[str], execute: Any, record: dict[str, Any]) -> None:
    packages = output / "packages"
    version = tomllib.loads((source / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    environment = shell.clean_environment()
    environment.update(
        UV_PROJECT_ENVIRONMENT=str(source / ".venv"),
        UV_PYTHON=sys.executable,
        npm_config_cache=str(output / "npm-cache"),
    )
    environment.setdefault("CARGO_TARGET_DIR", str(root / "target" / "quickstart-build"))
    target = Path(environment["CARGO_TARGET_DIR"]).resolve()
    environment["CARGO_TARGET_DIR"] = str(target)
    record["status"] = "building"
    provenance = output / "build.json"
    provenance.write_text(json.dumps(record, indent=2) + "\n")
    # The harness's canonical recipes own build policy, including when testing
    # an older source ref. Only their source root and output settings change;
    # record the recipe hash alongside the tested source commit.
    justfile = root / "justfile"
    record["justfile_sha256"] = sha256(justfile)
    arguments = (
        "just",
        "--justfile",
        justfile,
        "--set",
        "REPO_ROOT",
        source,
        "--set",
        "package_profile",
        "dev",
        "--set",
        "ref_name",
        version,
        "--set",
        "output_dir",
        packages,
        "--set",
        "node_platform",
        host_platform(),
        "--set",
        "node_metapackage",
        "true",
    )
    commands = [
        (surface, shell.command(*arguments, recipe))
        for surface, recipe in (("cli", "build-cli"), ("python", "package-python"), ("nodejs", "package-node"))
        if surface in surfaces
    ]
    record["commands"] = commands
    provenance.write_text(json.dumps(record, indent=2) + "\n")
    for name, command in commands:
        print(f"Building {name} from {record['commit'][:12]}...", flush=True)
        execute(command, source, environment, output / (name + ".stdout.log"), output / (name + ".stderr.log"), 2400)
    if "cli" in surfaces:
        triples = {
            "macos-arm64": "aarch64-apple-darwin",
            "linux-amd64": "x86_64-unknown-linux-gnu",
            "linux-arm64": "aarch64-unknown-linux-gnu",
            "windows-amd64": "x86_64-pc-windows-msvc",
        }
        suffix = ".exe" if shell.WINDOWS else ""
        shutil.copy2(
            target / ("debug/nemo-relay" + suffix),
            packages / f"nemo-relay-cli-{triples[host_platform()]}-{version}{suffix}",
        )
    record.update(
        status="built",
        commands=commands,
        files={path.name: sha256(path) for path in packages.rglob("*") if path.is_file()},
    )
    provenance.write_text(json.dumps(record, indent=2) + "\n")
