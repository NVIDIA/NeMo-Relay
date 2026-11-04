# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure recovery for temporary source-distribution staging."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "plugin/build_backend.py"
    spec = importlib.util.spec_from_file_location("worker_build_backend_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    package = source / module._PACKAGE_PATH
    package.mkdir(parents=True)
    (package / "service.proto").write_text('syntax = "proto3";\n')
    lock = source / "buf.lock"
    lock.write_text("version: v2\n")
    monkeypatch.setattr(module, "_SDIST_ROOT", tmp_path / "staged")
    monkeypatch.setattr(module, "_proto_sources", lambda: (source, lock))
    return module


@pytest.mark.parametrize("failure", ["before_copytree", "partial_copytree", "copy2"])
def test_partial_staging_is_removed_and_build_can_retry(
    backend: ModuleType, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def fail_copy(source: Path, destination: Path) -> None:
        del source
        if failure == "partial_copytree":
            destination.mkdir(parents=True)
            (destination / "partial.proto").write_text("partial")
        raise OSError("injected copy failure")

    with monkeypatch.context() as patch:
        patch.setattr(backend.shutil, "copy2" if failure == "copy2" else "copytree", fail_copy)
        with pytest.raises(OSError, match="injected copy failure"), backend._staged_proto():
            pytest.fail("failed staging must not enter the build")
    assert not backend._SDIST_ROOT.exists()
    with backend._staged_proto():
        assert (backend._SDIST_ROOT / backend._PACKAGE_PATH / "service.proto").is_file()
        assert (backend._SDIST_ROOT / "buf.lock").read_text() == "version: v2\n"
    assert not backend._SDIST_ROOT.exists()


def test_existing_staging_is_preserved(backend: ModuleType) -> None:
    backend._SDIST_ROOT.mkdir()
    marker = backend._SDIST_ROOT / "owned-by-user"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError, match="refusing to replace"), backend._staged_proto():
        pytest.fail("existing staging must not enter the build")
    assert marker.read_text() == "preserve"
