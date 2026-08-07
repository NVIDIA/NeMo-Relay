# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


def load_attributions_module() -> ModuleType:
    path = Path(__file__).parents[1] / "licensing" / "attributions_lockfile_md.py"
    spec = importlib.util.spec_from_file_location("attributions_lockfile_md", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StableRustLicenseTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_attributions_module()

    def test_prefers_bundled_text_when_only_layout_differs(self) -> None:
        bundled = "Heading\n\n\tIndented line\nFinal line\n"
        detected = "   Heading\n\n       Indented line\n   Final line\n"
        with tempfile.TemporaryDirectory() as directory:
            package_dir = Path(directory)
            manifest = package_dir / "Cargo.toml"
            manifest.write_text('[package]\nname = "fixture"\n')
            (package_dir / "LICENSE").write_text(bundled)

            normalized = self.module._stable_rust_license_text({"manifest_path": str(manifest)}, detected)

        self.assertEqual(normalized, bundled)

    def test_preserves_unrecognized_license_text(self) -> None:
        license_text = "   A custom license\n   with meaningful indentation.\n"

        normalized = self.module._stable_rust_license_text({}, license_text)

        self.assertEqual(normalized, license_text)


if __name__ == "__main__":
    unittest.main()
