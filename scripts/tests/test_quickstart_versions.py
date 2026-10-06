# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version sync tests use disposable docs, not the repository checkout."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("docs_version_pins", ROOT / "scripts/docs/docs_version_pins.py")
assert SPEC is not None and SPEC.loader is not None
VERSIONS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERSIONS)


class QuickstartVersionTests(unittest.TestCase):
    def test_update_keeps_copyable_pins_aligned_and_preserves_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "docs/getting-started/quick-start"
            folder.mkdir(parents=True)
            (root / "Cargo.toml").write_text('[workspace.package]\nversion = "0.11.0"\n')
            (root / "docs/getting-started/installation.mdx").write_text("uv add nemo-relay==0.10.0\n")
            openclaw = folder / "openclaw.mdx"
            openclaw.write_text(
                "The path was verified with Relay `0.10.0`.\n"
                "openclaw plugins install npm:nemo-relay-openclaw@0.10.0 --pin\n"
            )
            switchyard = folder / "switchyard.mdx"
            switchyard.write_text("Switchyard host Relay 0.9.2\n")
            integrations = root / "docs/supported-integrations"
            integrations.mkdir(parents=True)
            (integrations / "openclaw-plugin.mdx").write_text("npm install nemo-relay-openclaw@0.10.0\n")
            native = root / "docs/build-plugins/native"
            native.mkdir(parents=True)
            (native / "build-and-package.mdx").write_text(
                'nemo-relay-plugin = { version = "0.10.0" }\nrelay = ">=0.10.0,<1.0"\n'
            )
            workers = root / "docs/build-plugins/workers"
            workers.mkdir(parents=True)
            (workers / "rust.mdx").write_text('nemo-relay-worker = { version = "0.10.0" }\n')
            VERSIONS.update(root, "0.10.0", "0.11.0")
            VERSIONS.check(root)
            self.assertIn("nemo-relay==0.11.0", (root / "docs/getting-started/installation.mdx").read_text())
            self.assertIn("verified with Relay `0.10.0`", openclaw.read_text())
            self.assertIn("nemo-relay-openclaw@0.11.0", openclaw.read_text())
            self.assertIn("Relay 0.9.2", switchyard.read_text())
            self.assertIn("nemo-relay-openclaw@0.11.0", (integrations / "openclaw-plugin.mdx").read_text())
            self.assertIn('version = "0.11.0"', (native / "build-and-package.mdx").read_text())
            self.assertIn('relay = ">=0.10.0,<1.0"', (native / "build-and-package.mdx").read_text())
            VERSIONS.update(root, "0.11.0", "0.11.0-rc.1")
            self.assertIn("nemo-relay==0.11.0", (root / "docs/getting-started/installation.mdx").read_text())

    def test_check_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs/getting-started/quick-start").mkdir(parents=True)
            (root / "Cargo.toml").write_text('[workspace.package]\nversion = "0.11.0"\n')
            (root / "docs/getting-started/installation.mdx").write_text("uv add nemo-relay==0.10.0\n")
            for path in VERSIONS.pages(root)[1:]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with self.assertRaisesRegex(ValueError, "expected Relay 0.11.0"):
                VERSIONS.check(root)


if __name__ == "__main__":
    unittest.main()
