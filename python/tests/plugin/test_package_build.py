# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build regression tests for the Python worker plugin package."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile


def test_sdist_rebuilds_worker_bindings_without_checked_in_codegen(tmp_path: Path):
    repository_root = Path(__file__).parents[3]
    plugin_source = repository_root / "python/plugin"
    workspace_root = tmp_path / "workspace"
    project_root = workspace_root / "python/plugin"
    shutil.copytree(plugin_source, project_root, ignore=_ignore_build_outputs)
    for generated in (project_root / "src/nemo_relay_plugin/_proto").glob("plugin_worker_pb2*.py"):
        generated.unlink()

    canonical_proto = workspace_root / "crates/worker-proto/proto/nemo/relay/worker/v1/plugin_worker.proto"
    canonical_proto.parent.mkdir(parents=True)
    source_proto = repository_root / "crates/worker-proto/proto/nemo/relay/worker/v1/plugin_worker.proto"
    shutil.copy2(source_proto, canonical_proto)

    repository_wheel_dir = tmp_path / "repository-wheel"
    _run(
        ["uv", "build", "--wheel", "--out-dir", str(repository_wheel_dir), str(project_root)],
    )
    _assert_wheel_contains_worker_bindings(next(repository_wheel_dir.glob("*.whl")))
    assert not (project_root / "proto").exists()

    distribution_dir = tmp_path / "dist"
    _run(
        ["uv", "build", "--sdist", "--out-dir", str(distribution_dir), str(project_root)],
    )
    sdist = next(distribution_dir.glob("*.tar.gz"))
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        assert any(name.endswith("/proto/plugin_worker.proto") for name in names)
        assert not any(name.endswith("plugin_worker_pb2.py") for name in names)
        assert not any(name.endswith("plugin_worker_pb2_grpc.py") for name in names)
        extraction_root = (tmp_path / "extracted").resolve()
        for member in archive.getmembers():
            destination = (extraction_root / member.name).resolve()
            assert destination.is_relative_to(extraction_root)
            archive.extract(member, extraction_root)

    extracted_project = next(extraction_root.iterdir())
    wheel_dir = tmp_path / "wheel"
    _run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir), str(extracted_project)],
    )
    _assert_wheel_contains_worker_bindings(next(wheel_dir.glob("*.whl")))

    venv = tmp_path / "venv"
    _run(
        ["uv", "venv", "--python", sys.executable, str(venv)],
    )
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    _run(
        ["uv", "pip", "install", "--python", str(python), "-e", str(extracted_project)],
    )
    _run(
        [str(python), "-c", "import nemo_relay_plugin._proto.plugin_worker_pb2_grpc"],
    )


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
    assert "nemo_relay_plugin/_proto/plugin_worker_pb2.py" in names
    assert "nemo_relay_plugin/_proto/plugin_worker_pb2_grpc.py" in names


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise AssertionError(
            f"command failed ({completed.returncode}): {shlex.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
