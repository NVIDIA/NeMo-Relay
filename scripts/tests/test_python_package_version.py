# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Python release version materialization."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("python_package_version", ROOT / "scripts" / "python_package_version.py")
assert SPEC is not None and SPEC.loader is not None
PYTHON_PACKAGE_VERSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PYTHON_PACKAGE_VERSION)


class PythonPackageVersionTests(unittest.TestCase):
    def test_converts_release_versions_to_pep440(self) -> None:
        cases = {
            "0.8.0": "0.8.0",
            "0.8.0-alpha.1": "0.8.0a1",
            "0.8.0-beta.2": "0.8.0b2",
            "0.8.0-rc.3": "0.8.0rc3",
            "0.8.0+ABC-def.1": "0.8.0+abc.def.1",
        }

        for version, expected in cases.items():
            with self.subTest(version=version):
                self.assertEqual(PYTHON_PACKAGE_VERSION.semver_to_pep440(version), expected)

    def test_materializes_version_in_pyproject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            pyproject = source / "pyproject.toml"
            pyproject.write_text(
                '[project]\ndynamic = ["version"]\n'
                '[project.optional-dependencies]\ncli = ["nemo-relay-cli-bin==0.8.0"]\n'
            )

            PYTHON_PACKAGE_VERSION.materialize_python_version(source, "0.8.0rc1")

            self.assertEqual(
                pyproject.read_text(),
                '[project]\nversion = "0.8.0rc1"\n'
                '[project.optional-dependencies]\ncli = ["nemo-relay-cli-bin==0.8.0rc1"]\n',
            )

    def test_rejects_missing_cli_extra(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "pyproject.toml").write_text('[project]\ndynamic = ["version"]\n')

            with self.assertRaisesRegex(ValueError, "CLI extra version"):
                PYTHON_PACKAGE_VERSION.materialize_python_version(source, "0.8.0")

    def test_rejects_unsupported_version(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported Python package version"):
            PYTHON_PACKAGE_VERSION.semver_to_pep440("0.8")


if __name__ == "__main__":
    unittest.main()
