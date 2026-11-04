# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep rustdoc's generic Deref section headings valid in generated MDX."""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "docs"), *sys.path]):
    REFERENCE = importlib.import_module("generate_rust_library_reference")


class RustReferenceTests(unittest.TestCase):
    def test_deref_heading_escapes_generic_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "struct.EventView.html"
            source.write_text(
                '<main id="main-content">'
                '<div class="main-heading"><h1>Struct EventView</h1></div>'
                '<h2 id="deref-methods">Methods from <a>Deref</a> '
                "&lt;Target = <a>Event</a>&gt;</h2>"
                "</main>",
                encoding="utf-8",
            )
            page = REFERENCE.Page(source, root / "event-view.mdx", "/event-view", "observability", "observability")
            rendered = REFERENCE._render_page(page, {source: page}, 1)

        self.assertIn("## Methods from Deref &lt;Target = Event &gt;", rendered)
        self.assertNotIn("<Target", rendered)


if __name__ == "__main__":
    unittest.main()
