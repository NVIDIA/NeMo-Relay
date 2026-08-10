# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CLI wheel assembly."""

import importlib.util
import os
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("package_cli_bin", ROOT / "scripts" / "package-cli-bin.py")
assert SPEC is not None and SPEC.loader is not None
PACKAGE_CLI_BIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE_CLI_BIN
SPEC.loader.exec_module(PACKAGE_CLI_BIN)


class PackageCliBinTests(unittest.TestCase):
    def test_packages_linux_binaries_for_python(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            binary = output / "nemo-relay"
            binary.write_bytes(b"test-binary")
            gnu_platform = PACKAGE_CLI_BIN.PLATFORMS["x86_64-unknown-linux-gnu"]
            musl_platform = PACKAGE_CLI_BIN.PLATFORMS["x86_64-unknown-linux-musl"]

            previous_directory = Path.cwd()
            try:
                os.chdir(output)
                wheel = PACKAGE_CLI_BIN.build_wheel(binary, gnu_platform, "0.7.0-rc.1", output)
                musl_wheel = PACKAGE_CLI_BIN.build_wheel(binary, musl_platform, "0.7.0-rc.1", output)
            finally:
                os.chdir(previous_directory)

            self.assertIn("0.7.0rc1-py3-none-manylinux_2_17_x86_64", wheel.name)
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
                script = next(info for info in archive.infolist() if info.filename.endswith(".data/scripts/nemo-relay"))
                script_mode = script.external_attr >> 16
                self.assertTrue(stat.S_ISREG(script_mode))
                self.assertEqual(stat.S_IMODE(script_mode), 0o755)
                wheel_metadata = archive.read(next(name for name in names if name.endswith("/WHEEL")))
                self.assertIn(b"Tag: py3-none-manylinux_2_17_x86_64", wheel_metadata)

            self.assertIn("0.7.0rc1-py3-none-musllinux_1_2_x86_64", musl_wheel.name)

    def test_rejects_unsupported_version(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported package version"):
            PACKAGE_CLI_BIN.wheel_version("dev-deadbeef")

    def test_translates_release_versions_to_pep440(self) -> None:
        versions = {
            "0.7.0": "0.7.0",
            "0.7.0-alpha.1": "0.7.0a1",
            "0.7.0-rc.1": "0.7.0rc1",
            "0.7.0+deadbeef": "0.7.0+deadbeef",
        }
        for version, expected in versions.items():
            with self.subTest(version=version):
                self.assertEqual(PACKAGE_CLI_BIN.wheel_version(version), expected)


if __name__ == "__main__":
    unittest.main()
