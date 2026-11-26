# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve re-exports and Runtime content in generated Node.js reference pages."""

import importlib
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "docs"), *sys.path]):
    REFERENCE = importlib.import_module("generate_node_library_reference")

ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


class NodeReferenceTests(unittest.TestCase):
    def test_namespace_exports_keep_member_names_qualified(self) -> None:
        pages = self.generate(
            {
                "dist/index.d.ts": """
                /** Typed content snapshots. */
                export * as codecs from './content.js';
            """,
                "dist/content.d.ts": """
                /** Read-only snapshot. */
                export declare class MessageView {
                    text(): string | null;
                }
                /** Select the last supplied item. */
                export declare function lastMessage(request: unknown): MessageView | null;
                export declare class _InternalStorage {}
            """,
                "dist/helper.d.ts": "export declare const helper: true;",
            }
        )
        runtime = pages["sample.mdx"]
        self.assertIn("### `codecs`", runtime)
        self.assertIn("### `codecs.MessageView`", runtime)
        self.assertIn("### `codecs.lastMessage`", runtime)
        self.assertIn("Read-only snapshot.", runtime)
        self.assertNotIn("### `MessageView`", runtime)
        self.assertNotIn("### `lastMessage`", runtime)
        self.assertNotIn("_InternalStorage", runtime)

    def test_namespace_exports_preserve_overloads_and_merged_declarations(self) -> None:
        pages = self.generate(
            {
                "dist/index.d.ts": "export * as codecs from './content.js';",
                "dist/content.d.ts": """
                /** Parse text. */
                export declare function parse(value: string): string;
                /** Parse a number. */
                export declare function parse(value: number): number;
                /** Result value. */
                export interface Result { value: string; }
                /** Additional result metadata. */
                export interface Result { complete: boolean; }
            """,
                "dist/helper.d.ts": "export declare const helper: true;",
            }
        )
        runtime = pages["sample.mdx"]
        for declaration in (
            "function parse(value: string): string;",
            "function parse(value: number): number;",
            "interface Result { value: string; }",
            "interface Result { complete: boolean; }",
        ):
            self.assertIn(declaration, runtime)
        for summary in ("Parse text.", "Parse a number.", "Result value.", "Additional result metadata."):
            self.assertIn(summary, runtime)
        self.assertEqual(runtime.count("### `codecs.parse`"), 2)
        self.assertEqual(runtime.count("### `codecs.Result`"), 2)

    def generate(self, files: dict[str, str]) -> dict[str, str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_root = root / "package"
            exports = {
                "." if name == "index" else f"./{name}": {
                    "types": f"./dist/{name}.d.ts",
                    "default": f"./dist/{name}.js",
                }
                for name in ("index", "helper")
            }
            write(package_root / "package.json", json.dumps({"name": "sample", "exports": exports}))
            for relative, content in files.items():
                write(package_root / relative, content)
            output = root / "generated"
            REFERENCE.generate(package_root, output)
            return {path.name: path.read_text(encoding="utf-8") for path in output.glob("*.mdx")}

    def test_star_reexport_inlines_semicolon_free_declarations(self) -> None:
        pages = self.generate(
            {
                "binding.d.ts": """
                /** Scope stack. */
                export declare class ScopeStack {
                  get depth(): number
                }
                export declare const __napiBindingTarget: 'native' | 'wasm32-wasi'
                export type Visibility =
                  | 'public'
                  | 'private'
                /** Emits one event. */
                export declare function event(name: string, handle?: ScopeStack | null): void
                export declare function configure(
                  name: string,
                  options: { retries: number },
                ): void
                export declare const limits: {
                  depth: number
                }
                export interface Inner {
                  value: number
                }
                """,
                "dist/index.d.ts": """
                export * from '../binding.js';
                export type { Inner } from '../binding.js';
                export declare class ScopeStack {
                    wrapped: true;
                }
                export declare class LlmRequest {
                    readonly content: string;
                }
                """,
                "dist/helper.d.ts": """
                export type { Inner, Visibility as Kind } from '../binding.js';
                export { event } from '../binding.js';
                """,
            }
        )

        runtime = pages["sample.mdx"]
        self.assertIn("### `LlmRequest`", runtime)
        self.assertIn("### `Visibility`", runtime)
        self.assertIn("### `event`", runtime)
        self.assertIn("### `Inner`", runtime)
        self.assertNotIn("## Re-exports", runtime)
        self.assertIn(
            "export declare function configure(\n  name: string,\n  options: { retries: number },\n): void", runtime
        )
        self.assertIn("### `limits`", runtime)
        self.assertIn("export declare const limits: {\n  depth: number\n}", runtime)
        self.assertIn("export type Visibility =\n  | 'public'\n  | 'private'", runtime)
        self.assertEqual(runtime.count("### `ScopeStack`"), 1)
        self.assertIn("wrapped: true;", runtime)
        self.assertNotIn("get depth(): number", runtime)
        self.assertNotIn("__napiBindingTarget", runtime)

        helper = pages["sample-helper.mdx"]
        self.assertIn("## Re-exports\n\n- `Inner`\n- `Kind`\n- `event`\n", helper)

    def test_missing_reexport_target_fails(self) -> None:
        with self.assertRaises(SystemExit):
            self.generate(
                {
                    "dist/index.d.ts": "export * from './missing.js';\n",
                    "dist/helper.d.ts": "",
                }
            )

    def test_runtime_page_lists_binding_declarations(self) -> None:
        binding = ROOT / "crates" / "node" / "binding.d.ts"
        self.assertTrue(binding.is_file(), f"build the Node package first: {binding} is missing")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            REFERENCE.generate(ROOT / "crates" / "node", output)
            runtime = "\n".join(page.read_text(encoding="utf-8") for page in (output / "nemo-relay-node").glob("*.mdx"))
        for name in (
            "ScopeStack",
            "ScopeHandle",
            "LlmRequest",
            "withScope",
            "toolCallExecute",
            "llmCallExecute",
            "AtofExporter",
            "registerLlmExecutionIntercept",
        ):
            self.assertIn(f"### `{name}`", runtime)


class RuntimeReferenceTests(unittest.TestCase):
    def _package(self, root: Path, declarations: str, *, typed: str = "") -> Path:
        package = root / "package"
        package.mkdir()
        exports = {".": {"types": "./index.d.ts"}}
        (package / "index.d.ts").write_text(declarations)
        if typed:
            exports["./typed"] = {"types": "./typed.d.ts"}
            (package / "typed.d.ts").write_text(typed)
        (package / "package.json").write_text(json.dumps({"name": "nemo-relay-node", "exports": exports}))
        return package

    def test_runtime_item_content_is_unchanged_and_present_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = self._package(
                root,
                "/** Initialize with the original text.\n"
                " * @param config Original **parameter** description.\n"
                " * @returns Original return description.\n"
                " * @remarks Original remarks.\n"
                " */\n"
                "export function initialize(config: Json): Promise<void>;\n"
                "/** Preserve this class description. */\n"
                "export class LlmRequest {\n"
                "  value: Json\n"
                "}\n"
                "export { OriginalAlias };\n",
            )
            output = root / "reference"
            REFERENCE.generate(package, output)
            overview = (output / "nemo-relay-node/index.mdx").read_text()
            pages = "\n".join(path.read_text() for path in output.rglob("*.mdx"))

        original_function = (
            "### `initialize`\n\nInitialize with the original text.\n\n"
            "**Parameters**\n\n- `config`: Original **parameter** description.\n\n"
            "**Returns**\n\nOriginal return description.\n\n"
            "**Remarks**\n\nOriginal remarks.\n\n"
            "```ts\nexport function initialize(config: Json): Promise<void>;\n```\n\n"
        )
        original_class = (
            "### `LlmRequest`\n\nPreserve this class description.\n\n"
            "```ts\nexport class LlmRequest {\n  value: Json\n}\n```\n\n"
        )
        self.assertEqual(pages.count(original_function), 1)
        self.assertEqual(pages.count(original_class), 1)
        self.assertIn("## Re-exports\n\n- `OriginalAlias`\n", overview)
        self.assertIn('<a id="initialize"></a>', overview)
        self.assertIn('href="nemo-relay-node/lifecycle#initialize"', overview)
        self.assertIn('href="nemo-relay-node/llm#llmrequest"', overview)
        self.assertNotIn("### `initialize`", overview)

    def test_related_exports_follow_topics_including_scoped_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = self._package(
                root,
                "export type ResourceMeasurementScope = string;\n"
                "export function scopeRegisterSubscriber(): void;\n"
                "export function scopeRegisterLlmExecutionIntercept(): void;\n"
                "export function registerPlugin(): void;\n"
                "export function llmCall(): void;\n"
                "export function toolCall(): void;\n"
                "export function capturePropagationContext(): void;\n"
                "export function validateAdaptiveConfig(): void;\n"
                "export function futureRuntimeExport(): void;\n",
            )
            output = root / "reference"
            REFERENCE.generate(package, output)
            pages = {path.stem: path.read_text() for path in (output / "nemo-relay-node").glob("*.mdx")}

        for name, topic in (
            ("ResourceMeasurementScope", "resource-metrics"),
            ("scopeRegisterSubscriber", "observability"),
            ("scopeRegisterLlmExecutionIntercept", "middleware"),
            ("registerPlugin", "lifecycle"),
            ("futureRuntimeExport", "lifecycle"),
            ("llmCall", "llm"),
            ("toolCall", "tools"),
            ("capturePropagationContext", "scopes"),
            ("validateAdaptiveConfig", "adaptive"),
        ):
            self.assertIn(f"### `{name}`", pages[topic])
            self.assertEqual(sum(f"### `{name}`" in page for page in pages.values()), 1)

    def test_duplicate_names_keep_the_original_kind_disambiguation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = self._package(
                root, "export interface Settings {\n value: string\n}\nexport function Settings(): string;\n"
            )
            output = root / "reference"
            REFERENCE.generate(package, output)
            runtime = output / "nemo-relay-node"
            page = (runtime / "lifecycle.mdx").read_text()
            overview = (runtime / "index.mdx").read_text()

        self.assertIn("### `Settings` interface\n", page)
        self.assertIn("### `Settings` function\n", page)
        self.assertIn('#settings-interface"', overview)
        self.assertIn('#settings-function"', overview)

    def test_other_entry_point_body_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = self._package(
                root,
                "export function initialize(): void;\n",
                typed=(
                    "/** Original helper documentation. */\nexport type TypedValue = string;\nexport { KeptAlias };\n"
                ),
            )
            output = root / "reference"
            REFERENCE.generate(package, output)
            actual = (output / "nemo-relay-node-typed.mdx").read_text().split("Import from ", 1)[1]

        self.assertEqual(
            actual,
            "`nemo-relay-node/typed`.\n\nTyped wrappers and codec-aware execution helpers.\n\n"
            "## Type Aliases\n\n### `TypedValue`\n\nOriginal helper documentation.\n\n"
            "```ts\nexport type TypedValue = string;\n```\n\n## Re-exports\n\n- `KeptAlias`\n\n",
        )

    def test_oversized_topic_fails_instead_of_truncating_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = self._package(
                root, "/** " + "Unmodified prose. " * 3_000 + "*/\nexport function initialize(): void;\n"
            )
            output = root / "reference"
            output.mkdir()
            previous = output / "previous.mdx"
            previous.write_text("Previous successful output.")
            with self.assertRaisesRegex(ValueError, "Runtime reference topic exceeds"):
                REFERENCE.generate(package, output)
            self.assertEqual(previous.read_text(), "Previous successful output.")


if __name__ == "__main__":
    unittest.main()
