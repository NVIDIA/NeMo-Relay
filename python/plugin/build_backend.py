# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PEP 517 backend that generates private worker protobuf bindings."""

from __future__ import annotations

import argparse
import contextlib
import shutil
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent


def _project_file(relative_path: str) -> Path:
    candidate = (_PROJECT_ROOT / relative_path).resolve()
    if candidate.parent != _PROJECT_ROOT:
        raise ValueError(f"path must be a direct child of {_PROJECT_ROOT}")
    return candidate


_REPOSITORY_PROTO = _PROJECT_ROOT.parents[1] / "crates/worker-proto/proto/nemo/relay/worker/v1/plugin_worker.proto"
_SDIST_PROTO = _PROJECT_ROOT / "proto/plugin_worker.proto"
_GENERATED_DIR = _PROJECT_ROOT / "src/nemo_relay_plugin/_proto"
_MANIFEST = _project_file("MANIFEST.in")


def generate_worker_proto(output_dir: Path = _GENERATED_DIR) -> None:
    """Generate Python worker bindings from the canonical protobuf schema."""
    proto = _proto_source()
    output_dir.mkdir(parents=True, exist_ok=True)
    protoc: Any = import_module("grpc_tools.protoc")
    exit_code = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{proto.parent}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(proto),
        ]
    )
    if exit_code != 0:
        raise RuntimeError(f"grpc_tools.protoc failed with exit code {exit_code}")

    grpc_module = output_dir / "plugin_worker_pb2_grpc.py"
    generated = grpc_module.read_text(encoding="utf-8")
    absolute_import = "import plugin_worker_pb2 as plugin__worker__pb2"
    if absolute_import not in generated:
        raise RuntimeError("grpc_tools.protoc did not generate the expected worker protobuf import")
    grpc_module.write_text(
        generated.replace(
            absolute_import,
            "from . import plugin_worker_pb2 as plugin__worker__pb2",
            1,
        ),
        encoding="utf-8",
    )


def get_requires_for_build_wheel(config_settings: Any = None) -> list[str]:
    """Return setuptools wheel-build requirements."""
    return _setuptools_backend().get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings: Any = None) -> list[str]:
    """Return setuptools source-distribution build requirements."""
    return _setuptools_backend().get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings: Any = None) -> list[str]:
    """Return setuptools editable-build requirements."""
    return _setuptools_backend().get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory: str, config_settings: Any = None) -> str:
    """Delegate wheel metadata generation to setuptools."""
    with _staged_proto():
        return _setuptools_backend().prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(wheel_directory: str, config_settings: Any = None, metadata_directory: str | None = None) -> str:
    """Generate bindings, then build a wheel containing them."""
    with _staged_proto():
        generate_worker_proto()
        return _setuptools_backend().build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(
    wheel_directory: str,
    config_settings: Any = None,
    metadata_directory: str | None = None,
) -> str:
    """Generate bindings before building an editable wheel."""
    with _staged_proto():
        generate_worker_proto()
        return _setuptools_backend().build_editable(wheel_directory, config_settings, metadata_directory)


def prepare_metadata_for_build_editable(metadata_directory: str, config_settings: Any = None) -> str:
    """Delegate editable metadata generation to setuptools."""
    with _staged_proto():
        return _setuptools_backend().prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_sdist(sdist_directory: str, config_settings: Any = None) -> str:
    """Stage the canonical schema and generate bindings before building an sdist."""
    with _staged_proto(), _sdist_manifest():
        generate_worker_proto()
        return _setuptools_backend().build_sdist(sdist_directory, config_settings)


def _proto_source() -> Path:
    if _REPOSITORY_PROTO.is_file():
        return _REPOSITORY_PROTO
    if _SDIST_PROTO.is_file():
        return _SDIST_PROTO
    raise FileNotFoundError("could not find the worker protobuf schema")


@contextlib.contextmanager
def _staged_proto() -> Iterator[None]:
    proto_source = _proto_source()
    staged_proto = proto_source != _SDIST_PROTO
    if staged_proto:
        _SDIST_PROTO.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(proto_source, _SDIST_PROTO)
    try:
        yield
    finally:
        if staged_proto:
            _SDIST_PROTO.unlink(missing_ok=True)
            try:
                _SDIST_PROTO.parent.rmdir()
            except OSError:
                pass


@contextlib.contextmanager
def _sdist_manifest() -> Iterator[None]:
    previous = _MANIFEST.read_bytes() if _MANIFEST.exists() else None
    _MANIFEST.write_text(
        "\n".join(
            [
                "include build_backend.py",
                "include proto/plugin_worker.proto",
                "global-exclude plugin_worker_pb2.py",
                "global-exclude plugin_worker_pb2_grpc.py",
                "",
            ]
        ),
        encoding="utf-8",
    )
    try:
        yield
    finally:
        if previous is None:
            _MANIFEST.unlink(missing_ok=True)
        else:
            _MANIFEST.write_bytes(previous)


def _setuptools_backend() -> Any:
    return import_module("setuptools.build_meta")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", type=Path, metavar="DIRECTORY", required=True)
    arguments = parser.parse_args()
    generate_worker_proto(arguments.generate)
