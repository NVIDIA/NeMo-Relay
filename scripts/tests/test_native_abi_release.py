# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify independent interface versions and release gates using temporary manifests."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.native_abi_release import check, set_version


class NativeAbiReleaseTests(unittest.TestCase):
    def test_relay_artifact_versions_preserve_independent_interface(self):
        source = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy2(source / "justfile", root / "justfile")
            workspace = (source / "Cargo.toml").read_text()
            for field in ("members", "default-members"):
                workspace = re.sub(
                    rf"(?ms)^{field} = \[.*?^\]",
                    f'{field} = ["crates/types", "crates/plugin-abi"]',
                    workspace,
                )
            (root / "Cargo.toml").write_text(workspace)
            for name, version in (("types", "version.workspace = true"), ("plugin-abi", 'version = "1.0.0"')):
                crate = root / "crates" / name
                (crate / "src").mkdir(parents=True)
                (crate / "src/lib.rs").write_text("")
                (crate / "Cargo.toml").write_text(f'[package]\nname = "nemo-relay-{name}"\n{version}\n')
            interface = root / "crates/plugin-abi/Cargo.toml"
            original_interface = interface.read_bytes()
            for version in ("1.0.0+deadbeef", "2.3.4-rc.1"):
                with self.subTest(version=version):
                    subprocess.run(
                        ["just", "set-cargo-version", version],
                        cwd=root,
                        env=os.environ | {"UV_PYTHON": sys.executable},
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(check(root), "1.0.0")
                    self.assertEqual(interface.read_bytes(), original_interface)
                    self.assertIn(f'version = "{version}"', (root / "Cargo.toml").read_text())

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
