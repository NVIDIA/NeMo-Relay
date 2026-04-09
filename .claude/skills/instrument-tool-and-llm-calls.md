<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Wrap application tool calls and LLM/provider calls with NeMo Flow scopes and managed execution APIs
---

# Instrument Tool And LLM Calls

Use this skill when an app already has tool functions or model/provider calls and
needs to run them through NeMo Flow correctly.

## Default Guidance

- Put a scope around the natural agent, request, workflow, or graph boundary.
- Use managed execution APIs first:
  - Python: `tools.execute(...)`, `llm.execute(...)`
  - Go: `ToolCallExecute(...)`, `LlmCallExecute(...)`
  - Node.js/WASM: `toolCallExecute(...)`, `llmCallExecute(...)`
- Use manual lifecycle APIs only when the host framework cannot be wrapped by the
  managed execute helpers.

## Checklist

- [ ] Scope boundary chosen before the first tool or LLM call
- [ ] Existing tool function wrapped without losing its original arguments/result
- [ ] Existing LLM/provider call wrapped at the right abstraction layer
- [ ] Optional metadata, attributes, or model name attached where useful
- [ ] Context propagation handled if the call hops threads or async tasks

## Use Another Skill When

- you need traces, ATIF, or export setup -> `set-up-observability`
- you are debugging missing events or load failures -> `debug-runtime-integration`
- you need per-request isolation or worker-pool advice -> `use-context-isolation`

## References

- `docs/language-bindings.md`
- `docs/concepts.md`
- `docs/context-isolation.md`
