# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check grouped reference coverage, page boundaries, and cross-references."""

import importlib
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "docs"), *sys.path]):
    REFERENCE = importlib.import_module("generate_rust_library_reference")


class RustReferenceTests(unittest.TestCase):
    def _source(self, root: Path, name: str, body: str) -> Path:
        path = root / "rustdoc" / "sample" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        title = "Module nested" if name.endswith("index.html") else "Struct " + path.stem.split(".")[-1]
        path.write_text(
            f'<main id="main-content"><div class="main-heading"><h1>{title}</h1></div>{body}</main>',
            encoding="utf-8",
        )
        return path

    def _generate(self, root: Path, ceiling: int = 50_000) -> dict[Path, str]:
        with (
            patch.object(REFERENCE, "CRATES", (("sample", "sample", "Sample crate"),)),
            patch.object(REFERENCE, "GENERATED_BY", "Generated from fixture rustdoc."),
            patch("rust_reference_grouping.MAX_PAGE_CHARACTERS", ceiling),
        ):
            count = REFERENCE.generate(root, root / "rustdoc", root / "output", run_cargo_doc=False)
        output = {path.relative_to(root / "output"): path.read_text() for path in (root / "output").rglob("*.mdx")}
        self.assertEqual(count, len(output))
        self.assertTrue(all(len(body) <= ceiling for body in output.values()))
        return output

    def test_small_modules_keep_items_and_reexports_with_distinct_member_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root, "index.html", '<p><a href="struct.First.html#method.new">First::new</a></p>')
            self._source(root, "struct.First.html", '<h3 id="method.new">new</h3><p>First constructor.</p>')
            self._source(root, "struct.Second.html", '<h3 id="method.new">new</h3><p>Second constructor.</p>')
            self._source(root, "nested_module/index.html", '<p><a href="../struct.First.html">First</a></p>')
            self._source(root, "nested_module/struct.First.html", "<p>Re-export documentation.</p>")
            output = self._generate(root)

        self.assertEqual(len(output), 3)  # Reference landing page and two modules.
        module = output[Path("sample/index.mdx")]
        self.assertIn("/sample#struct-first--method.new)", module)
        self.assertIn('<a id="struct-first--method.new"></a>', module)
        self.assertIn('<a id="struct-second--method.new"></a>', module)
        nested = output[Path("sample/nested-module/index.mdx")]
        self.assertIn('slug: "/reference/api/rust-library-reference/sample/nested_module"', nested)
        self.assertIn("Re-export documentation.", nested)
        self.assertIn("/sample#struct-first)", nested)

    def test_large_items_split_at_members_and_rewrite_links_into_continuations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root, "index.html", '<p><a href="struct.Big.html#method.last">Last method</a></p>')
            members = "".join(
                f'<h3 id="method.{name}">{name}</h3><p>{name} ' + "Documentation. " * 65 + "</p>"
                for name in ("first", "second", "third", "last")
            )
            self._source(root, "struct.Big.html", members)
            self._source(root, "struct.Small.html", '<p><a href="struct.Big.html#method.last">Jump to last</a></p>')
            output = self._generate(root, ceiling=3_000)

        text = "\n".join(output.values())
        for name in ("first", "second", "third", "last"):
            self.assertEqual(text.count(f'<a id="struct-big--method.{name}"></a>'), 1)
            self.assertIn(f"{name} Documentation.", text)
        destination = next(
            path.with_suffix("").as_posix()
            for path, body in output.items()
            if '<a id="struct-big--method.last"></a>' in body
        )
        self.assertIn(f"/{destination}#struct-big--method.last)", text)
        self.assertIn("## Reference pages", output[Path("sample/index.mdx")])

    def test_fenced_examples_survive_splitting_and_output_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root, "index.html", "<p>Module documentation.</p>")
            example = 'let example = "## heading inside a string";\n' * 15
            self._source(
                root,
                "struct.Big.html",
                '<h2 id="examples">Examples</h2><pre>'
                + example
                + '</pre><h2 id="methods">Methods</h2>'
                + "<p>Member documentation.</p>" * 100,
            )
            output = self._generate(root, ceiling=3_000)
            self.assertEqual(output, self._generate(root, ceiling=3_000))

        self.assertTrue(any(example.strip() in body for body in output.values()))
        for body in output.values():
            self.assertEqual(len(re.findall(r"^```", body, flags=re.MULTILINE)) % 2, 0)

    def test_large_llm_module_uses_topics_and_signature_links_follow_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root, "index.html", "<p>Crate overview.</p>")
            self._source(root, "llm/index.html", "<p>LLM overview.</p>")
            for name in ("LlmRequest", "LlmStream", "LlmCall"):
                self._source(root, f"llm/struct.{name}.html", "<p>" + name + " documentation. " * 70 + "</p>")
            self._source(
                root,
                "llm/fn.execute.html",
                '<pre>fn execute(request: <a href="struct.LlmRequest.html">LlmRequest</a>)</pre>',
            )
            output = self._generate(root, ceiling=3_000)

        self.assertIn(Path("sample/llm/reference-streaming.mdx"), output)
        self.assertIn(Path("sample/llm/reference-request-types.mdx"), output)
        execution = output[Path("sample/llm/reference-llm-execution.mdx")]
        self.assertIn("reference-request-types#struct-llmrequest", execution)
        self.assertIn('href="reference-request-types#struct-llmrequest"', execution)
        self.assertIn("<code>fn execute(request: <a href=", execution)
        self.assertNotIn("dangerouslySetInnerHTML", execution)

    def test_large_categories_group_related_names_without_numbered_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root, "index.html", "<p>Module overview.</p>")
            for name in ("AlphaConfig", "BetaState", "AlphaState"):
                self._source(root, f"struct.{name}.html", "<p>" + name + " documentation " * 45 + "</p>")
            output = self._generate(root, ceiling=3_000)

        alpha = output[Path("sample/reference-alpha-types.mdx")]
        self.assertIn("Struct AlphaConfig", alpha)
        self.assertIn("Struct AlphaState", alpha)
        self.assertNotIn("Struct BetaState", alpha)
        self.assertIn("Struct BetaState", output[Path("sample/reference-beta-types.mdx")])
        for body in output.values():
            title = yaml.safe_load(body.split("---", 2)[1])["title"]
            self.assertNotRegex(title, r" \(\d+\)$")

    def test_split_item_pages_are_named_for_complete_member_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root, "index.html", "<p>Module overview.</p>")
            self._source(
                root,
                "struct.Big.html",
                "".join(
                    f'<h3 id="method.{name}">{name}</h3><p>' + "Documentation. " * 65 + "</p>"
                    for name in ("start", "finish", "cancel")
                ),
            )
            output = self._generate(root, ceiling=3_000)

        for name in ("start", "finish", "cancel"):
            page = output[Path(f"sample/reference-struct-big-{name}.mdx")]
            self.assertIn(f'title: "struct-big: {name}"', page)
            self.assertIn(f'id="struct-big--method.{name}"', page)

    def test_numeric_acronyms_remain_meaningful_family_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root, "index.html", "<p>Module overview.</p>")
            for name in ("S3Config", "BetaState", "S3Builder"):
                self._source(root, f"struct.{name}.html", "<p>" + name + " documentation " * 45 + "</p>")
            output = self._generate(root, ceiling=3_000)

        page = output[Path("sample/reference-s3-types.mdx")]
        self.assertIn('title: "S3 types"', page)
        self.assertIn("Struct S3Config", page)
        self.assertIn("Struct S3Builder", page)

    def test_oversized_module_overview_preserves_each_list_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [f"Entry {index}: " + "description " * 12 for index in range(40)]
            self._source(root, "index.html", "<ul>" + "".join(f"<li>{entry}</li>" for entry in entries) + "</ul>")
            output = self._generate(root, ceiling=3_000)

        text = "\n".join(output.values())
        self.assertIn("reference-module-overview", text)
        for entry in entries:
            self.assertIn(entry.strip(), text)

    def test_large_module_inventory_stays_complete_without_summary_pagination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("one", "two", "three", "four", "five", "six")
            description = "Complete item documentation. " * 20
            self._source(
                root,
                "index.html",
                "<ul>"
                + "".join(f'<li><a href="fn.{name}.html">{name}</a>: {description}</li>' for name in names)
                + "</ul>",
            )
            for name in names:
                self._source(root, f"fn.{name}.html", f"<p>{description}</p>")
            output = self._generate(root, ceiling=3_000)

        overview = output[Path("sample/index.mdx")]
        for name in names:
            self.assertIn(f"[{name}]", overview)
        self.assertFalse(any("reference-module-overview" in str(path) for path in output))
        self.assertIn(description.strip(), "\n".join(output.values()))

    def test_oversized_atomic_block_fails_without_replacing_previous_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root, "index.html", "<p>Overview.</p>")
            self._source(root, "struct.Big.html", "<pre>" + "let x = 1;\n" * 400 + "</pre>")
            output = root / "output"
            output.mkdir()
            previous = output / "previous.mdx"
            previous.write_text("Previous successful output.")
            with self.assertRaisesRegex(ValueError, "indivisible block"):
                self._generate(root, ceiling=3_000)
            self.assertEqual(previous.read_text(), "Previous successful output.")

    def test_large_byte_string_keeps_every_escape_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = r"\x00\xff\n\"hello" * 5_000
            source = self._source(
                root, "constant.DESCRIPTORS.html", '<pre>pub const DESCRIPTORS: &amp;[u8] = b"' + value + '";</pre>'
            )
            page = REFERENCE.Page(source, root / "descriptors.mdx", "/descriptors", "sample", "sample")
            rendered = REFERENCE._render_page(page, {source: page}, 1)

        segments = re.findall(r'```rust\nb"(.*?)"\n```', rendered, flags=re.DOTALL)
        self.assertGreater(len(segments), 1)
        self.assertEqual("".join(segments), value)
        self.assertTrue(all(len(segment) <= 12_000 for segment in segments))
        self.assertIn("Concatenate their contents in order", rendered)

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
