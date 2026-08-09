# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Rampart native plugin archive builder."""

from __future__ import annotations

import importlib.util
import stat
import sys
import tarfile
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/package-pii-rampart-plugin.py"
SPEC = importlib.util.spec_from_file_location("package_pii_rampart_plugin", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)


def fixture_repository(root: Path) -> Path:
    """Create the package source files used by one unit test."""
    repository = root / "repository"
    plugin = repository / "plugins/pii-rampart"
    plugin.mkdir(parents=True)
    (repository / "LICENSE").write_text("license\n")
    (plugin / "README.md").write_text("readme\n")
    (plugin / "config.schema.json").write_text("{}\n")
    (plugin / "relay-plugin.toml").write_text(
        """manifest_version = 1
[plugin]
id = "pii_rampart"
kind = "rust_dynamic"
[source]
artifact = "target/release/<platform-library-file>"
[integrity]
sha256 = "sha256:<artifact-sha256>"
[load]
library = "target/release/<platform-library-file>"
symbol = "nemo_relay_register_plugin"
"""
    )
    return repository


class PackagePiiRampartPluginTests(unittest.TestCase):
    """Verify Rampart native plugin archive assembly and validation."""

    def test_builds_and_verifies_archives(self) -> None:
        targets = {
            "x86_64-unknown-linux-gnu": ".tar.gz",
            "x86_64-pc-windows-msvc": ".zip",
        }
        for target, extension in targets.items():
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                repository = fixture_repository(root)
                library = root / PACKAGE.library_name(target)
                library.write_bytes(b"native-plugin")
                attributions = root / "ATTRIBUTIONS-Rust.md"
                attributions.write_text("attributions\n")
                args = SimpleNamespace(
                    repository=repository,
                    library=library,
                    attributions=attributions,
                    target=target,
                    version="0.8.0",
                    output_dir=root / "output",
                )

                first = PACKAGE.build_archive(args)
                first_bytes = first.read_bytes()
                second = PACKAGE.build_archive(args)
                self.assertEqual(second.read_bytes(), first_bytes)
                self.assertTrue(first.name.endswith(extension))

                extracted = root / "extracted"
                PACKAGE.extract_archive(first, extracted)
                PACKAGE.verify_package(extracted / PACKAGE.ARCHIVE_ROOT, target, False)

    def test_rejects_unsafe_archive_member(self) -> None:
        for member in ("../escape", "..\\escape", "C:/escape"):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "unsafe.zip"
                with zipfile.ZipFile(archive, "w") as output:
                    output.writestr(member, b"bad")
                with self.assertRaisesRegex(ValueError, "unsafe archive member"):
                    PACKAGE.extract_archive(archive, root / "output")

    def test_rejects_archive_member_outside_package_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unexpected-root.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("unexpected-root/file", b"bad")
            with self.assertRaisesRegex(ValueError, "outside"):
                PACKAGE.extract_archive(archive, root / "output")

    def test_rejects_zip_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                link = zipfile.ZipInfo(f"{PACKAGE.ARCHIVE_ROOT}/link")
                link.create_system = 3
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                output.writestr(link, b"target")
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                PACKAGE.extract_archive(archive, root / "output")

    def test_rejects_non_regular_tar_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                link = tarfile.TarInfo(f"{PACKAGE.ARCHIVE_ROOT}/link")
                link.type = tarfile.SYMTYPE
                link.linkname = "/tmp/target"
                output.addfile(link)
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                PACKAGE.extract_archive(archive, root / "output")

    def test_rejects_duplicate_archive_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "duplicate.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(f"{PACKAGE.ARCHIVE_ROOT}/README.md", b"first")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    output.writestr(f"{PACKAGE.ARCHIVE_ROOT}/README.md", b"second")
            with self.assertRaisesRegex(ValueError, "duplicate archive member"):
                PACKAGE.extract_archive(archive, root / "output")

    def test_rejects_wrong_library_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = fixture_repository(root)
            library = root / "wrong.so"
            library.write_bytes(b"native-plugin")
            attributions = root / "ATTRIBUTIONS-Rust.md"
            attributions.write_text("attributions\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                PACKAGE.archive_entries(
                    repository,
                    library,
                    attributions,
                    "x86_64-unknown-linux-gnu",
                )

    def test_rejects_empty_attributions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = fixture_repository(root)
            target = "x86_64-unknown-linux-gnu"
            library = root / PACKAGE.library_name(target)
            library.write_bytes(b"native-plugin")
            attributions = root / "ATTRIBUTIONS-Rust.md"
            attributions.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                PACKAGE.archive_entries(repository, library, attributions, target)

    def test_rejects_unsupported_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported target"):
            PACKAGE.library_name("aarch64-pc-windows-msvc")

    def test_rejects_unsafe_archive_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = fixture_repository(root)
            target = "x86_64-unknown-linux-gnu"
            library = root / PACKAGE.library_name(target)
            library.write_bytes(b"native-plugin")
            attributions = root / "ATTRIBUTIONS-Rust.md"
            attributions.write_text("attributions\n")
            args = SimpleNamespace(
                repository=repository,
                library=library,
                attributions=attributions,
                target=target,
                version="../../escape",
                output_dir=root / "output",
            )
            with self.assertRaisesRegex(ValueError, "unsupported plugin archive version"):
                PACKAGE.build_archive(args)


if __name__ == "__main__":
    unittest.main()
