# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify independent interface versions and release gates using temporary manifests."""

import tempfile
import unittest
from pathlib import Path

from scripts.native_abi_release import check, set_version


class NativeAbiReleaseTests(unittest.TestCase):
    def test_interface_update_keeps_relay_and_frozen_fixtures_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interface = root / "crates/plugin-abi/Cargo.toml"
            interface.parent.mkdir(parents=True)
            interface.write_text('[package]\nversion = "1.0.0"\n')
            workspace = root / "Cargo.toml"
            workspace.write_text(
                '[workspace.package]\nversion = "5.3.0"\n'
                "[workspace.dependencies]\n"
                'nemo-relay-plugin-abi = { version = "=1.0.0", path = "crates/plugin-abi" }\n'
            )
            frozen = interface.parent / "tests/fixtures/interface-v1/Cargo.toml"
            frozen.parent.mkdir(parents=True)
            frozen.write_text('[package]\nversion = "1.0.0"\n')
            set_version(root, "1.1.0")
            self.assertEqual(check(root, "1.1.0"), "1.1.0")
            self.assertIn('version = "5.3.0"', workspace.read_text())
            self.assertEqual(frozen.read_text(), '[package]\nversion = "1.0.0"\n')
            with self.assertRaisesRegex(ValueError, "does not match"):
                check(root, "1.0.0")
            workspace.write_text(workspace.read_text().replace('"=1.1.0"', '"1.1.0"'))
            with self.assertRaisesRegex(ValueError, "requires workspace pin"):
                check(root)

    def test_invalid_release_version_does_not_change_manifests(self):
        root = Path(__file__).resolve().parents[2]
        for version in ["v1.0.0", "native-abi/1.0.0", "01.0.0", "1.0.0-01", "1.0"]:
            with self.subTest(version=version), self.assertRaises(ValueError):
                set_version(root, version)


if __name__ == "__main__":
    unittest.main()
