# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify package version automation preserves the explicit native ABI contract."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.native_abi_contract import check, read_contract


class NativeAbiContractTests(unittest.TestCase):
    """Keep package release automation separate from native ABI compatibility."""

    def test_package_release_updates_preserve_native_contract(self):
        """Package major, prerelease, and build changes must not alter the native contract."""
        source = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy2(source / "justfile", root / "justfile")
            workspace = (source / "Cargo.toml").read_text()
            for field in ("members", "default-members"):
                workspace = re.sub(
                    rf"(?ms)^{field} = \[.*?^\]", f'{field} = ["crates/types", "crates/plugin-abi"]', workspace
                )
            (root / "Cargo.toml").write_text(workspace)
            for name in ("types", "plugin-abi"):
                crate = root / "crates" / name
                (crate / "src").mkdir(parents=True)
                (crate / "src/lib.rs").write_text("")
                (crate / "Cargo.toml").write_text(f'[package]\nname = "nemo-relay-{name}"\nversion.workspace = true\n')
            contract = root / "crates/plugin-abi/src/contract.rs"
            shutil.copy2(source / "crates/plugin-abi/src/contract.rs", contract)
            original = contract.read_bytes()
            expected = read_contract(root)
            for version in ("0.99.0", "2.3.4-rc.1", "9.7.1+deadbeef"):
                with self.subTest(version=version):
                    subprocess.run(
                        ["just", "set-cargo-version", version],
                        cwd=root,
                        env=os.environ | {"UV_PYTHON": sys.executable},
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(check(root), expected)
                    self.assertEqual(contract.read_bytes(), original)

    def test_canonical_contract_notation_and_package_alignment(self):
        """Malformed contracts and mismatched Cargo pins fail without changing source files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crate = root / "crates/plugin-abi"
            (crate / "src").mkdir(parents=True)
            (crate / "Cargo.toml").write_text("[package]\nversion.workspace = true\n")
            workspace = root / "Cargo.toml"
            workspace.write_text(
                '[workspace.package]\nversion = "9.0.0"\n'
                '[workspace.dependencies]\nnemo-relay-plugin-abi = { version = "=9.0.0" }\n'
            )
            contract = crate / "src/contract.rs"
            for version in ("v1.0", "v1.1", "v2.0"):
                contract.write_text(f'pub const NATIVE_API: &str = "{version}";\n')
                self.assertEqual(check(root), version)
            for version in ("1", "1.0", "v0.1", "v01.0", "v1.01", "v1.0.0", "v4294967296.0"):
                contract.write_text(f'pub const NATIVE_API: &str = "{version}";\n')
                with self.subTest(version=version), self.assertRaisesRegex(ValueError, "expected native contract"):
                    check(root)
            contract.write_text('pub const NATIVE_API: &str = "v1.0";\n')
            workspace.write_text(workspace.read_text().replace('"=9.0.0"', '"9.0.0"'))
            with self.assertRaisesRegex(ValueError, "SDK pin"):
                check(root)
