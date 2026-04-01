<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nat_nexus

Python wrapper package for the Nexus core runtime.

## Overview

This package provides a thin Python layer over the native Rust extension (`_native`), offering a Pythonic API with module-level organization, type stubs, and convenience wrappers. The native extension is compiled from `crates/python/` via Maturin and exposed here as `nat_nexus._native`.

## What It Provides

- **Module-level organization** -- Each domain area has its own module for clean imports (e.g., `from nat_nexus.tools import register_tool`).
- **Type stubs** -- `__init__.pyi` provides full type annotations for IDE autocompletion and static analysis.
- **Scope context managers** -- Python context managers for automatic scope push/pop with cleanup.
- **Async support** -- All async operations are exposed as native Python coroutines compatible with `asyncio`.
- **Scope-local middleware** -- `scope_local.py` exposes scope-scoped guardrail, intercept, and subscriber registration.

## Key Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package root, re-exports from `_native` and submodules |
| `__init__.pyi` | Type stubs for the full public API |
| `scope.py` | Scope management (push, pop, context managers) |
| `tools.py` | Tool registration and invocation |
| `llm.py` | LLM call registration and invocation |
| `guardrails.py` | Guardrail registration (input/output) |
| `intercepts.py` | Request and execution intercept registration |
| `subscribers.py` | Event subscriber registration |
| `scope_local.py` | Scope-local middleware registration |
| `typed.py` | Typed helper utilities |

## Install

```bash
uv sync
```

This builds the native Rust extension and installs the package in a virtual environment. Tests can be run with:

```bash
uv run pytest
```

## Documentation

See [docs/api-reference.md](../../docs/api-reference.md) for the full Python API reference.
