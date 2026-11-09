# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep re-exported declarations in generated Node.js reference pages."""

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
                export interface Inner {
                  value: number
                }
                """,
                "dist/index.d.ts": """
                export * from '../binding.js';
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
            runtime = (output / "nemo-relay-node.mdx").read_text(encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
