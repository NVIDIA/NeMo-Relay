<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Help application developers pick a NeMo Flow binding and get to a first working scope, tool call, and LLM call
---

# Get Started With NeMo Flow

Use this skill for first-time users who want the shortest path to a working
example.

## Default Path

- Pick the user's host language first: Python, Go, Node.js, or WASM.
- Prefer the managed execution APIs over manual lifecycle APIs.
- Start with one scope, one tool call, and one LLM call.
- Add observability only after the basic flow works.

## Guidance

- **Python**: `uv sync`, then use `nemo_flow.scope.scope(...)`,
  `nemo_flow.tools.execute(...)`, and `nemo_flow.llm.execute(...)`
- **Go**: build the FFI shared library first, then use `PushScope`,
  `ToolCallExecute`, and `LlmCallExecute`
- **Node.js**: build the addon, then use `pushScope`, `toolCallExecute`, and
  `llmCallExecute`
- **WASM**: build with `wasm-pack`, call `init()`, then use the same managed
  execute path from the generated package

## Common Pitfalls

- calling execute APIs without an active scope
- skipping the build step for Python, Go, Node.js, or WASM
- mixing manual lifecycle APIs into a first example

## References

- `docs/getting-started-python.md`
- `docs/getting-started-go.md`
- `docs/getting-started-node.md`
- `docs/getting-started-wasm.md`
