# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check source-derived requirements and failures that would let docs drift."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "prerequisites", Path(__file__).resolve().parents[1] / "docs/prerequisites.py"
)
assert SPEC is not None and SPEC.loader is not None
PREREQUISITES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREREQUISITES)


class PrerequisiteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.write("pyproject.toml", '[project]\nrequires-python = ">=3.12"\n')
        self.write("crates/node/package.json", json.dumps({"engines": {"node": ">=24.20.0"}}))
        for agent in ("claude", "codex", "pi"):
            self.write(
                f"crates/cli/src/agents/{agent}/mod.rs",
                "minimum_version: (3, 2, 1),\nverified_through: Some((3, 2)),\n",
            )
        self.write(
            "integrations/openclaw/package.json",
            json.dumps(
                {"openclaw": {"build": {"openclawVersion": "2027.1.2"}}, "peerDependencies": {"openclaw": "2027.1.2"}}
            ),
        )
        self.write(
            "package-lock.json",
            json.dumps(
                {
                    "packages": {
                        "node_modules/openclaw": {"version": "2027.1.2", "engines": {"node": ">=26.2 <27 || >=28.1"}}
                    }
                }
            ),
        )

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_generates_from_declarations_and_detects_a_changed_floor(self):
        self.assertEqual(PREREQUISITES.update(self.root, check=False), [])
        output = self.root / PREREQUISITES.OUTPUT
        values = json.loads((output / "requirements.json").read_text())
        self.assertEqual(values["claude"], ">=3.2.1")
        self.assertEqual(values["python"], ">=3.12")
        self.assertIn("3.2.x", (output / "pi-verified.mdx").read_text())
        self.assertIn("`>=26.2 <27` or `>=28.1`", (output / "openclaw-node.mdx").read_text())
        self.write("pyproject.toml", '[project]\nrequires-python = ">=3.13"\n')
        errors = PREREQUISITES.update(self.root, check=True)
        self.assertTrue(any("python.mdx: stale" in error for error in errors))
        self.assertIn("3.12", (output / "python.mdx").read_text(), "check must not rewrite files")
        self.assertEqual(PREREQUISITES.update(self.root, check=False), [])
        self.assertEqual(PREREQUISITES.update(self.root, check=True), [])

    def test_rejects_changed_rust_declarations_and_complex_runtime_ranges(self):
        source = "crates/cli/src/agents/claude/mod.rs"
        for text in ("minimum_version: FLOOR,\n", "minimum_version: (1, 0, 0),\nminimum_version: (2, 0, 0),\n"):
            self.write(source, text)
            with self.assertRaisesRegex(ValueError, "expected one matching"):
                PREREQUISITES.generated(self.root)
        self.write(source, "minimum_version: (3, 2, 1),\n")
        self.write("crates/node/package.json", json.dumps({"engines": {"node": ">=24 <25 || >=26"}}))
        with self.assertRaisesRegex(ValueError, "expected one matching"):
            PREREQUISITES.generated(self.root)

    def test_rejects_mismatched_tested_openclaw_and_lockfile(self):
        self.write(
            "package-lock.json",
            json.dumps({"packages": {"node_modules/openclaw": {"version": "2027.2.0"}}}),
        )
        with self.assertRaisesRegex(ValueError, "tested host version"):
            PREREQUISITES.generated(self.root)

    def test_detects_obsolete_values_and_missing_nested_includes(self):
        PREREQUISITES.update(self.root, check=False)
        stale = self.write(str(PREREQUISITES.OUTPUT / "old-floor.mdx"), "0.1.0\n")
        self.assertTrue(any("obsolete" in error for error in PREREQUISITES.update(self.root, check=True)))
        PREREQUISITES.update(self.root, check=False)
        self.assertFalse(stale.exists())
        self.write(
            "docs/_snippets/prerequisites/project.mdx",
            'Requires <Markdown src="values/missing.mdx" />.\n',
        )
        self.assertTrue(any("missing or unsafe" in error for error in PREREQUISITES.update(self.root, check=True)))

    def test_renders_whole_tables_and_rejects_unknown_template_values(self):
        template = self.write(
            str(PREREQUISITES.TEMPLATES / "agents.mdx"),
            "| Agent | Minimum |\n| --- | --- |\n| Claude Code | {{claude}} |\n",
        )
        self.assertEqual(PREREQUISITES.update(self.root, check=False), [])
        table = (self.root / PREREQUISITES.OUTPUT / "agents.mdx").read_text()
        self.assertIn("| Claude Code | 3.2.1 |\n", table)
        self.assertNotIn("{{claude}}", table)
        template.write_text("{{unknown}}\n")
        with self.assertRaisesRegex(ValueError, "unknown prerequisite"):
            PREREQUISITES.generated(self.root)

    def test_rejects_missing_suffix_and_inline_table_includes(self):
        PREREQUISITES.update(self.root, check=False)
        page = self.write(
            "docs/guide.mdx",
            '<Markdown src="_snippets/prerequisites/values/claude.mdx" />\n',
        )
        self.assertTrue(any("nonempty suffix" in error for error in PREREQUISITES.update(self.root, check=True)))
        page.write_text('| Agent | <Markdown src="_snippets/prerequisites/values/claude.mdx" suffix=" " /> |\n')
        self.assertTrue(any("complete table" in error for error in PREREQUISITES.update(self.root, check=True)))

    def test_validates_single_quotes_and_reordered_static_attributes(self):
        PREREQUISITES.update(self.root, check=False)
        source = "_snippets/prerequisites/values/claude.mdx"
        page = self.write("docs/guide.mdx", "")
        for tag in (
            f"<Markdown src='{source}' />",
            f'<Markdown\n src="{source}"\n />',
        ):
            with self.subTest(tag=tag):
                page.write_text(tag)
                self.assertTrue(any("nonempty suffix" in error for error in PREREQUISITES.check_includes(self.root)))
        page.write_text(f"<Markdown suffix=\".\" src='{source}' />")
        self.assertEqual(PREREQUISITES.check_includes(self.root), [])
        page.write_text("<Markdown suffix=\".\" src='_snippets/prerequisites/missing.mdx' />")
        self.assertTrue(any("missing or unsafe" in error for error in PREREQUISITES.check_includes(self.root)))

    def test_fails_closed_on_unsupported_or_ambiguous_markdown_tags(self):
        PREREQUISITES.update(self.root, check=False)
        source = "_snippets/prerequisites/values/claude.mdx"
        page = self.write("docs/guide.mdx", "")
        for tag in (
            f'<Markdown src="{source}" suffix={{" "}} />',
            f'<Markdown src={{"{source}"}} suffix="." />',
            f'<Markdown src="{source}" suffix="."',
            f'<Markdown src="{source}" src="{source}" suffix="." />',
        ):
            with self.subTest(tag=tag):
                page.write_text(tag)
                self.assertTrue(PREREQUISITES.check_includes(self.root), tag)

    def test_rejects_leaked_placeholders_but_ignores_examples_comments_and_jsx_objects(self):
        page = self.write("docs/guide.mdx", "Leaked {{suffix}}\n")
        self.assertTrue(any("unexpanded prerequisite" in error for error in PREREQUISITES.check_includes(self.root)))
        page.write_text(
            '```mdx\n<Markdown suffix={" "} src="missing.mdx" />\n{{example}}\n```\n'
            '`<Markdown src="missing.mdx" />` and `{{example}}`\n'
            '{/* <Markdown src="missing.mdx" /> {{comment}} */}\n'
            '<div style={{ display: "flex" }}>Text</div>\n'
        )
        self.assertEqual(PREREQUISITES.check_includes(self.root), [])


if __name__ == "__main__":
    unittest.main()
