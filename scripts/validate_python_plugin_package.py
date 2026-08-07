# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate Python worker plugin package code generation and installation."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="nemo-relay-plugin-package-") as temporary_directory:
        _validate_package(repository_root, Path(temporary_directory))


def _validate_package(repository_root: Path, temporary_root: Path) -> None:
    plugin_source = repository_root / "python/plugin"
    workspace_root = temporary_root / "workspace"
    project_root = workspace_root / "python/plugin"
    shutil.copytree(plugin_source, project_root, ignore=_ignore_build_outputs)
    for generated in (project_root / "src/nemo_relay_plugin/_proto").glob("plugin_worker_pb2*.py"):
        generated.unlink()

    canonical_proto = workspace_root / "crates/worker-proto/proto/nemo/relay/worker/v1/plugin_worker.proto"
    canonical_proto.parent.mkdir(parents=True)
    shutil.copy2(
        repository_root / "crates/worker-proto/proto/nemo/relay/worker/v1/plugin_worker.proto",
        canonical_proto,
    )

    repository_wheel_dir = temporary_root / "repository-wheel"
    _run(["uv", "build", "--wheel", "--out-dir", str(repository_wheel_dir), str(project_root)])
    _assert_wheel_contains_worker_bindings(next(repository_wheel_dir.glob("*.whl")))
    if (project_root / "proto").exists():
        raise AssertionError("wheel build left generated proto sources in the project tree")

    distribution_dir = temporary_root / "dist"
    _run(["uv", "build", "--sdist", "--out-dir", str(distribution_dir), str(project_root)])
    sdist = next(distribution_dir.glob("*.tar.gz"))
    extraction_root = (temporary_root / "extracted").resolve()
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        if not any(name.endswith("/proto/plugin_worker.proto") for name in names):
            raise AssertionError("source distribution is missing plugin_worker.proto")
        if any(name.endswith(("plugin_worker_pb2.py", "plugin_worker_pb2_grpc.py")) for name in names):
            raise AssertionError("source distribution contains generated worker bindings")
        for member in archive.getmembers():
            destination = (extraction_root / member.name).resolve()
            if not destination.is_relative_to(extraction_root):
                raise AssertionError(f"unsafe source distribution path: {member.name}")
            archive.extract(member, extraction_root)

    extracted_project = next(extraction_root.iterdir())
    wheel_dir = temporary_root / "wheel"
    _run(["uv", "build", "--wheel", "--out-dir", str(wheel_dir), str(extracted_project)])
    rebuilt_wheel = next(wheel_dir.glob("*.whl"))
    _assert_wheel_contains_worker_bindings(rebuilt_wheel)

    venv = temporary_root / "venv"
    _run(["uv", "venv", "--python", sys.executable, str(venv)])
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    _run(["uv", "pip", "install", "--python", str(python), str(rebuilt_wheel)])
    _run([str(python), "-c", "import nemo_relay_plugin._proto.plugin_worker_pb2_grpc"])


def _ignore_build_outputs(directory: str, names: list[str]) -> set[str]:
    del directory
    return {
        name
        for name in names
        if name in {".ruff_cache", ".venv", "__pycache__", "build", "dist", "proto"} or name.endswith(".egg-info")
    }


def _assert_wheel_contains_worker_bindings(wheel: Path) -> None:
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
    required = {
        "nemo_relay_plugin/_proto/plugin_worker_pb2.py",
        "nemo_relay_plugin/_proto/plugin_worker_pb2_grpc.py",
    }
    missing = required - names
    if missing:
        raise AssertionError(f"wheel is missing generated worker bindings: {sorted(missing)}")


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise AssertionError(
            f"command failed ({completed.returncode}): {shlex.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


if __name__ == "__main__":
    main()
