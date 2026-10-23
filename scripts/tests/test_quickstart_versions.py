# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version sync tests use disposable docs, not the repository checkout."""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("docs_version_pins", ROOT / "scripts/docs/docs_version_pins.py")
assert SPEC is not None and SPEC.loader is not None
VERSIONS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERSIONS)


class QuickstartVersionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("just") and shutil.which("uv"), "requires just and uv")
    def test_set_version_rejects_docs_drift_before_changing_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / "justfile", root / "justfile")
            scripts = root / "scripts/docs"
            scripts.mkdir(parents=True)
            shutil.copyfile(ROOT / "scripts/docs/docs_version_pins.py", scripts / "docs_version_pins.py")
            docs = root / "docs/getting-started"
            docs.mkdir(parents=True)
            (docs / "installation.mdx").write_text("uv add nemo-relay==0.10.0\n", encoding="utf-8")
            manifest = '[workspace.package]\nversion = "0.11.0"\n'
            (root / "Cargo.toml").write_text(manifest, encoding="utf-8")
            result = subprocess.run(
                ["just", "--justfile", str(root / "justfile"), "set-version", "0.12.0"],
                cwd=root,
                env={**os.environ, "UV_PYTHON": sys.executable},
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected Relay 0.11.0, found 0.10.0", result.stderr)
            self.assertEqual((root / "Cargo.toml").read_text(encoding="utf-8"), manifest)

    def test_main_installer_source_stays_independent_of_release_selection(self):
        for script in ("install.sh", "install.ps1"):
            source = f"https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/main/{script}"
            line = f"curl -fsSL {source} | NEMO_RELAY_VERSION=0.10.0 sh"
            updated = VERSIONS.replace_version(line, "0.10.0", "0.11.0")
            self.assertIn(source, updated)
            self.assertIn("NEMO_RELAY_VERSION=0.11.0", updated)
            self.assertEqual(len(list(VERSIONS.matches(updated))), 1)

    def test_copyable_installer_forms_update_only_the_relay_pin(self):
        examples = (
            'uv add "nemo-relay[langchain,langchain-nvidia]==0.10.0" # SDK 0.10.0',
            "npm install --save-exact nemo-relay-node@0.10.0 # Node 24.0.0",
            'cargo add "nemo-relay@=0.10.0" # Rust 1.99.0',
            "openclaw plugins install npm:nemo-relay-openclaw@0.10.0 --pin",
            "https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/0.10.0/install.sh",
            "$env:NEMO_RELAY_VERSION = '0.10.0'",
        )
        for example in examples:
            with self.subTest(example=example):
                updated = VERSIONS.replace_version(example, "0.10.0", "0.11.0")
                self.assertEqual(len(list(VERSIONS.matches(updated))), 1)
                self.assertEqual(next(VERSIONS.matches(updated)).group("version"), "0.11.0")
                self.assertEqual(updated.replace("0.11.0", "0.10.0", 1), example)

    def test_update_keeps_copyable_pins_aligned_and_preserves_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "docs/getting-started/quick-start"
            folder.mkdir(parents=True)
            (root / "Cargo.toml").write_text('[workspace.package]\nversion = "0.11.0"\n')
            (root / "docs/getting-started/installation.mdx").write_text(
                "uv add nemo-relay==0.10.0 # unrelated tool version 0.10.0\n"
            )
            python = folder / "python.mdx"
            python.write_text("The path was verified with Relay `0.10.0`.\nuv add nemo-relay==0.10.0\n")
            plugins = root / "docs/configure-plugins"
            plugins.mkdir(parents=True)
            switchyard = plugins / "switchyard.mdx"
            switchyard.write_text("NEMO_RELAY_VERSION=0.9.2\n")
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
            self.assertIn("unrelated tool version 0.10.0", (root / "docs/getting-started/installation.mdx").read_text())
            self.assertIn("verified with Relay `0.10.0`", python.read_text())
            self.assertIn("nemo-relay==0.11.0", python.read_text())
            self.assertIn("NEMO_RELAY_VERSION=0.9.2", switchyard.read_text())
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

    def test_check_rejects_prerelease_install_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs/getting-started").mkdir(parents=True)
            (root / "Cargo.toml").write_text('[workspace.package]\nversion = "0.11.0-rc.1"\n')
            (root / "docs/getting-started/installation.mdx").write_text("uv add nemo-relay==0.11.0-rc.1\n")
            with self.assertRaisesRegex(ValueError, "expected Relay 0.11.0, found 0.11.0-rc.1"):
                VERSIONS.check(root)


if __name__ == "__main__":
    unittest.main()
