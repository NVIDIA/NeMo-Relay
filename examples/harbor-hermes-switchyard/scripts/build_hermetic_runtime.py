#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one reusable, provider-free Hermes runtime for Phase 2 setup.

The coordinator invokes this content-addressed materialization automatically
when it is absent. Network access is confined to this one step; task containers
consume the resulting directory through a read-only bind mount and perform no
apt, Git, or Python package installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parent
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))
from relay_version import wheel_version

SCHEMA_VERSION = "harbor-hermes-switchyard.hermetic-runtime.v1"
DEFAULT_HERMES_REPOSITORY = "https://github.com/bbednarski9/hermes-agent.git"
DEFAULT_HERMES_REF = "feat/relay-native-plugin-init"
DEFAULT_HERMES_COMMIT = "a3d472f0e6bdc376df87b1436a461c4796db6747"
UV_VERSION = "0.11.16"
PYTHON_VERSION = "3.11.13"
BUILDER_IMAGE = "python:3.11-bullseye"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name == "payload.json":
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.stat().st_mode.to_bytes(4, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, attempts: int = 4, **kwargs: object) -> None:
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(command, check=True, **kwargs)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            time.sleep(5 * (2 ** (attempt - 1)))


def materialize_source(
    destination: Path,
    *,
    repository: str,
    repository_ref: str,
    commit: str,
) -> None:
    clone = destination.parent / "clone"
    for attempt in range(1, 5):
        shutil.rmtree(clone, ignore_errors=True)
        try:
            run(
                [
                    "git",
                    "clone",
                    "--no-tags",
                    "--filter=blob:none",
                    "--branch",
                    repository_ref,
                    repository,
                    str(clone),
                ],
                attempts=1,
            )
            break
        except subprocess.CalledProcessError:
            if attempt == 4:
                raise
            time.sleep(5 * (2 ** (attempt - 1)))
    run(["git", "-C", str(clone), "fetch", "--depth", "1", "origin", commit])
    run(["git", "-C", str(clone), "checkout", "--detach", commit])
    actual = subprocess.check_output(
        ["git", "-C", str(clone), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise RuntimeError(f"Hermes checkout mismatch: expected {commit}, got {actual}")
    shutil.copytree(clone, destination, ignore=shutil.ignore_patterns(".git"))


def build_payload(
    output: Path,
    *,
    source: Path,
    relay_wheel: Path,
    platform: str,
) -> None:
    script = r'''
set -euo pipefail
python -m pip install --no-cache-dir "uv==${UV_VERSION}"

mkdir -p /opt/hermes-runtime/bin /opt/hermes-runtime/lib
cp -a /source /opt/hermes-runtime/hermes-agent-src
cp /usr/local/bin/uv /opt/hermes-runtime/bin/uv

/usr/local/bin/uv python install "${PYTHON_VERSION}" \
  --install-dir /opt/hermes-runtime/python --no-bin --compile-bytecode
python_bin="$(find /opt/hermes-runtime/python -type f -path '*/bin/python3.11' -print -quit)"
test -n "$python_bin"

UV_PROJECT_ENVIRONMENT=/opt/hermes-runtime/hermes-agent-src/venv \
UV_PYTHON_DOWNLOADS=never \
  /usr/local/bin/uv sync --frozen --extra all \
    --project /opt/hermes-runtime/hermes-agent-src --python "$python_bin"
/usr/local/bin/uv pip install \
  --python /opt/hermes-runtime/hermes-agent-src/venv/bin/python \
  --force-reinstall --no-deps "/input/${RELAY_WHEEL_NAME}"

cat > /opt/hermes-runtime/bin/python <<'EOF'
#!/bin/sh
set -eu
export LD_LIBRARY_PATH="/opt/hermes-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /opt/hermes-runtime/hermes-agent-src/venv/bin/python "$@"
EOF
cat > /opt/hermes-runtime/bin/hermes <<'EOF'
#!/bin/sh
set -eu
export LD_LIBRARY_PATH="/opt/hermes-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /opt/hermes-runtime/hermes-agent-src/venv/bin/hermes "$@"
EOF
chmod 0755 /opt/hermes-runtime/bin/python /opt/hermes-runtime/bin/hermes \
  /opt/hermes-runtime/bin/uv

/opt/hermes-runtime/bin/hermes version
/opt/hermes-runtime/bin/python -c \
  'import importlib.metadata as m; assert tuple(map(int, m.version("nemo-relay").split("."))) >= (0, 7, 0)'
'''
    env = os.environ.copy()
    env.update({"UV_VERSION": UV_VERSION, "PYTHON_VERSION": PYTHON_VERSION})
    run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            platform,
            "--env",
            f"UV_VERSION={UV_VERSION}",
            "--env",
            f"PYTHON_VERSION={PYTHON_VERSION}",
            "--env",
            f"RELAY_WHEEL_NAME={relay_wheel.name}",
            "--volume",
            f"{source}:/source:ro",
            "--volume",
            f"{relay_wheel}:/input/{relay_wheel.name}:ro",
            "--volume",
            f"{output}:/opt/hermes-runtime",
            BUILDER_IMAGE,
            "bash",
            "-lc",
            script,
        ],
        attempts=1,
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relay-wheel", type=Path, required=True)
    parser.add_argument("--relay-architecture", choices=("x86_64", "aarch64"), required=True)
    parser.add_argument("--hermes-repository", default=DEFAULT_HERMES_REPOSITORY)
    parser.add_argument("--hermes-ref", default=DEFAULT_HERMES_REF)
    parser.add_argument("--hermes-commit", default=DEFAULT_HERMES_COMMIT)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    relay_wheel = args.relay_wheel.expanduser().resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    expected_arch = args.relay_architecture
    if "manylinux" not in relay_wheel.name or expected_arch not in relay_wheel.name:
        raise ValueError(f"Relay wheel does not target Linux {expected_arch}: {relay_wheel.name}")
    platform = {"x86_64": "linux/amd64", "aarch64": "linux/arm64"}[expected_arch]

    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.building")
    if temporary_output.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary_output}")
    temporary_output.mkdir(mode=0o700)
    try:
        with tempfile.TemporaryDirectory(
            prefix="hermes-source-", dir=output.parent
        ) as temporary:
            source = Path(temporary) / "source"
            materialize_source(
                source,
                repository=args.hermes_repository,
                repository_ref=args.hermes_ref,
                commit=args.hermes_commit,
            )
            for attempt in range(1, 5):
                try:
                    build_payload(
                        temporary_output,
                        source=source,
                        relay_wheel=relay_wheel,
                        platform=platform,
                    )
                    break
                except subprocess.CalledProcessError:
                    if attempt == 4:
                        raise
                    shutil.rmtree(temporary_output)
                    temporary_output.mkdir(mode=0o700)
                    time.sleep(5 * (2 ** (attempt - 1)))
        content_sha256 = sha256_tree(temporary_output)
        marker = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "content_sha256": content_sha256,
            "hermes_repository": args.hermes_repository,
            "hermes_ref": args.hermes_ref,
            "hermes_commit": args.hermes_commit,
            "relay_version": wheel_version(relay_wheel),
            "relay_wheel_sha256": sha256_file(relay_wheel),
            "relay_architecture": expected_arch,
            "builder_image": BUILDER_IMAGE,
            "python_version": PYTHON_VERSION,
            "uv_version": UV_VERSION,
        }
        (temporary_output / "payload.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.chmod(temporary_output / "payload.json", 0o444)
        temporary_output.rename(output)
        print(json.dumps({"output": str(output), **marker}, indent=2, sort_keys=True))
    except BaseException:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
