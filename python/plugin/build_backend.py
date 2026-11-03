# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PEP 517 backend that generates private worker protobuf bindings."""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Generator
from importlib import import_module
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent


def _project_file(relative_path: str) -> Path:
    candidate = (_PROJECT_ROOT / relative_path).resolve()
    if candidate.parent != _PROJECT_ROOT:
        raise ValueError(f"path must be a direct child of {_PROJECT_ROOT}")
    return candidate


_REPOSITORY_ROOT = _PROJECT_ROOT.parents[1] / "crates/worker-proto"
_SDIST_ROOT = _PROJECT_ROOT / "proto"
_GENERATED_DIR = _PROJECT_ROOT / "src/nemo_relay_plugin/_proto"
_MANIFEST = _project_file("MANIFEST.in")
_PACKAGE_PATH = Path("nvidia/nemo/relay/worker/v1")


def generate_worker_proto(output_dir: Path = _GENERATED_DIR) -> None:
    """Generate the complete namespaced contract and its locked validation schema."""
    source, lock = _proto_sources()
    dependency = resolve_proto_dependencies(lock)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dependency / "LICENSE", output_dir / "PROTOVALIDATE-LICENSE")
    for obsolete in output_dir.glob("plugin_worker_pb2*.py"):
        obsolete.unlink()
    protoc: Any = import_module("grpc_tools.protoc")
    grpc_tools: Any = import_module("grpc_tools")
    sources = sorted((source / _PACKAGE_PATH).glob("*.proto"))
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{source}",
            f"-I{dependency}",
            f"-I{Path(grpc_tools.__file__).parent / '_proto'}",
            f"--python_out={output_dir}",
            f"--pyi_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(dependency / "buf/validate/validate.proto"),
            *map(str, sources),
        ]
    )
    if result:
        raise RuntimeError(f"grpc_tools.protoc failed with exit code {result}")
    # Keep generated packages private; installing this wheel must not populate
    # the application's global NVIDIA or Buf Python namespaces. Protobuf descriptor
    # package names remain unchanged: update the Buf Protovalidate lock and the
    # Python protovalidate dependency pin together to keep descriptors identical.
    for generated in sorted((*output_dir.rglob("*_pb2*.py"), *output_dir.rglob("*_pb2.pyi"))):
        text = generated.read_text(encoding="utf-8")
        text = re.sub(r"^from (nvidia\.|buf\.)", r"from nemo_relay_plugin._proto.\1", text, flags=re.MULTILINE)
        text = re.sub(
            r"(BuildTopDescriptorsAndMessages\(DESCRIPTOR, ')((?:nvidia|buf)\.)", r"\1nemo_relay_plugin._proto.\2", text
        )
        generated.write_text(text, encoding="utf-8")
        for directory in (generated.parent, *generated.parent.parents):
            if not directory.is_relative_to(output_dir):
                break
            (directory / "__init__.py").touch(exist_ok=True)


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


def _proto_sources() -> tuple[Path, Path]:
    source = _REPOSITORY_ROOT / "proto"
    lock = _REPOSITORY_ROOT / "buf.lock"
    if (source / _PACKAGE_PATH / "plugin_worker_service.proto").is_file():
        return source, lock
    if (_SDIST_ROOT / _PACKAGE_PATH / "plugin_worker_service.proto").is_file():
        return _SDIST_ROOT, _SDIST_ROOT / "buf.lock"
    raise FileNotFoundError("could not find the complete worker protobuf contract")


def resolve_proto_dependencies(lock: Path | None = None) -> Path:
    """Export the immutable Buf dependency into the project's ignored build cache."""
    if lock is None:
        _, lock = _proto_sources()
    match = re.search(r"name: buf\.build/bufbuild/protovalidate\s+commit: ([0-9a-f]{32})", lock.read_text())
    if match is None:
        raise ValueError("buf.lock must pin an immutable Protovalidate commit")
    cache = _PROJECT_ROOT / "build/proto-deps"
    output = cache / match[1]
    if not ((output / "buf/validate/validate.proto").is_file() and (output / "LICENSE").is_file()):
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            try:
                subprocess.run(
                    [
                        os.environ.get("BUF", "buf"),
                        "export",
                        f"buf.build/bufbuild/protovalidate:{match[1]}",
                        "--all",
                        "--path",
                        "buf/validate/validate.proto",
                        "--output",
                        temporary,
                    ],
                    check=True,
                    timeout=120,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Buf dependency export timed out after 120 seconds") from exc
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Buf is required to build nemo-relay-plugin from source; install Buf or set BUF"
                ) from exc
            if output.exists() and not (output / "LICENSE").is_file():
                shutil.rmtree(output)
            try:
                Path(temporary).rename(output)
            except OSError:
                if not (output / "buf/validate/validate.proto").is_file():
                    raise
    return output


@contextlib.contextmanager
def _staged_proto() -> Generator[None]:
    source, lock = _proto_sources()
    staged = source != _SDIST_ROOT
    if staged:
        if _SDIST_ROOT.exists():
            raise FileExistsError(f"refusing to replace existing staged sources: {_SDIST_ROOT}")
        shutil.copytree(source / _PACKAGE_PATH, _SDIST_ROOT / _PACKAGE_PATH)
        shutil.copy2(lock, _SDIST_ROOT / "buf.lock")
    try:
        yield
    finally:
        if staged:
            shutil.rmtree(_SDIST_ROOT)


@contextlib.contextmanager
def _sdist_manifest() -> Generator[None]:
    previous = _MANIFEST.read_bytes() if _MANIFEST.exists() else None
    _MANIFEST.write_text(
        "\n".join(
            [
                "include build_backend.py",
                "recursive-include proto *.proto buf.lock",
                "global-exclude *_pb2.py",
                "global-exclude *_pb2.pyi",
                "global-exclude *_pb2_grpc.py",
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
            # `_MANIFEST` is resolved and constrained to a direct child of this backend's directory.
            _MANIFEST.write_bytes(previous)  # NOSONAR


def _setuptools_backend() -> Any:
    return import_module("setuptools.build_meta")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", type=Path, metavar="DIRECTORY", required=True)
    arguments = parser.parse_args()
    generate_worker_proto(arguments.generate)
