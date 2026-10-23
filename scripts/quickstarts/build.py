# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build host assets from one immutable Relay checkout, leaving the user's tree alone."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from quickstarts.assets import host_platform, sha256


def checkout(root: Path, ref: str, output: Path) -> tuple[Path, str, dict[str, Any]]:
    """Pin and inspect source before selecting cases or starting expensive builds."""
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--verify", ref + "^{commit}"], text=True
    ).strip()
    source, packages = output / "source", output / "packages"
    packages.mkdir()
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(root), str(source)], check=True, capture_output=True
    )
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", commit], check=True, capture_output=True)
    version = tomllib.loads((source / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    record = {"commit": commit, "ref": ref, "profile": "dev", "status": "checked-out"}
    (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    return source, version, record


def prepare(root: Path, source: Path, output: Path, surfaces: set[str], execute: Any, record: dict[str, Any]) -> None:
    packages = output / "packages"
    version = tomllib.loads((source / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    environment = dict(os.environ)
    environment.update(UV_PROJECT_ENVIRONMENT=str(source / ".venv"), npm_config_cache=str(output / "npm-cache"))
    environment.setdefault("CARGO_TARGET_DIR", str(root / "target" / "quickstart-build"))
    target = Path(environment["CARGO_TARGET_DIR"]).resolve()
    environment["CARGO_TARGET_DIR"] = str(target)
    record["status"] = "building"
    provenance = output / "build.json"
    provenance.write_text(json.dumps(record, indent=2) + "\n")
    commands = []
    if "cli" in surfaces:
        commands.append(("cli", "cargo build --locked -p nemo-relay-cli"))
    if "python" in surfaces:
        commands.append(
            (
                "python",
                shlex.quote(sys.executable)
                + " -m maturin build --locked --profile dev --interpreter "
                + shlex.quote(sys.executable)
                + " --out "
                + shlex.quote(str(packages)),
            )
        )
    if "nodejs" in surfaces:
        commands.extend(
            [
                ("node-install", "npm ci --workspace=nemo-relay-node --ignore-scripts"),
                ("node-build", "npm run build-debug --workspace=nemo-relay-node"),
                (
                    "node-pack",
                    f"{shlex.quote(sys.executable)} scripts/package-node-bin.py --platform {host_platform()} "
                    f"--version {shlex.quote(version)} --output-dir {shlex.quote(str(packages))} --metapackage",
                ),
            ]
        )
    for name, command in commands:
        print(f"Building {name} from {record['commit'][:12]}...", flush=True)
        execute(command, source, environment, output / (name + ".stdout.log"), output / (name + ".stderr.log"), 2400)
    if "cli" in surfaces:
        triples = {
            "macos-arm64": "aarch64-apple-darwin",
            "linux-amd64": "x86_64-unknown-linux-gnu",
            "linux-arm64": "aarch64-unknown-linux-gnu",
        }
        shutil.copy2(target / "debug/nemo-relay", packages / f"nemo-relay-cli-{triples[host_platform()]}-{version}")
    record.update(
        status="built",
        commands=commands,
        files={path.name: sha256(path) for path in packages.iterdir() if path.is_file()},
    )
    provenance.write_text(json.dumps(record, indent=2) + "\n")
