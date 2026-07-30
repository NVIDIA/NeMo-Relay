# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for split Node.js package assembly."""

import importlib.util
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import IO

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("package_node_bin", ROOT / "scripts" / "package-node-bin.py")
assert SPEC is not None and SPEC.loader is not None
PACKAGE_NODE_BIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE_NODE_BIN
SPEC.loader.exec_module(PACKAGE_NODE_BIN)


def required_member(archive: tarfile.TarFile, name: str) -> IO[bytes]:
    """Return a required regular file from a tar archive."""
    member = archive.extractfile(name)
    if member is None:
        raise AssertionError(f"archive is missing {name}")
    return member


class PackageNodeBinTests(unittest.TestCase):
    """Verify Node metapackage and native package assembly."""

    def test_builds_metapackage_and_linux_native_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            node_dir = output / "node"
            node_dir.mkdir()
            source_manifest = {
                "name": "nemo-relay-node",
                "version": "0.7.0",
                "description": "Node bindings.",
                "main": "index.js",
                "types": "index.d.ts",
                "exports": {
                    ".": {"types": "./index.d.ts", "default": "./index.js"},
                    "./typed": {"types": "./typed.d.ts", "default": "./typed.js"},
                },
                "engines": {"node": ">=24.0.0"},
                "license": "Apache-2.0",
            }
            (node_dir / "package.json").write_text(json.dumps(source_manifest))
            for filename in ("index.js", "index.d.ts", "typed.js", "typed.d.ts", "README.md"):
                (node_dir / filename).write_text(filename)
            binary_name = "nemo-relay.linux-x64-gnu.node"
            (node_dir / binary_name).write_bytes(b"native")

            platform = PACKAGE_NODE_BIN.PLATFORMS["linux-amd64"]
            native = PACKAGE_NODE_BIN.build_native_package(node_dir, platform, "0.7.0-rc.1", output)
            metapackage = PACKAGE_NODE_BIN.build_metapackage(node_dir, "0.7.0-rc.1", output)

            self.assertEqual(native.name, "nemo-relay-node-npm-linux-x64-0.7.0-rc.1.tgz")
            with tarfile.open(native) as archive:
                manifest = json.load(required_member(archive, "package/package.json"))
                self.assertEqual(manifest["name"], "nemo-relay-node-linux-x64-gnu")
                self.assertEqual(manifest["os"], ["linux"])
                self.assertEqual(manifest["cpu"], ["x64"])
                self.assertEqual(manifest["libc"], ["glibc"])
                self.assertEqual(manifest["main"], binary_name)
                self.assertEqual(required_member(archive, f"package/{binary_name}").read(), b"native")
                self.assertNotIn("package/index.js", archive.getnames())

            self.assertEqual(metapackage.name, "nemo-relay-node-npm-0.7.0-rc.1.tgz")
            with tarfile.open(metapackage) as archive:
                manifest = json.load(required_member(archive, "package/package.json"))
                self.assertEqual(
                    manifest["optionalDependencies"]["nemo-relay-node-linux-x64-gnu"],
                    "0.7.0-rc.1",
                )
                self.assertIn("package/index.js", archive.getnames())
                self.assertFalse(any(name.endswith(".node") for name in archive.getnames()))

            development = PACKAGE_NODE_BIN.build_metapackage(node_dir, "0.7.0+deadbeef", output)
            with tarfile.open(development) as archive:
                manifest = json.load(required_member(archive, "package/package.json"))
                self.assertEqual(manifest["version"], "0.7.0+deadbeef")
                self.assertTrue(
                    all(version == "0.7.0+deadbeef" for version in manifest["optionalDependencies"].values())
                )


if __name__ == "__main__":
    unittest.main()
