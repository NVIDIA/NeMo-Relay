<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvidia-nat-nexus-python

PyO3 Python bindings for the Nexus core runtime, built as a native C extension using the abi3 stable ABI.

## Overview

This crate compiles to a `_native` shared library (`.so` / `.pyd`) that the `nat_nexus` Python package imports. It wraps the full Nexus core API for Python, bridging Rust async with Python's `asyncio` via `pyo3-async-runtimes` and converting between Rust/Python types using `pythonize`.

## What It Provides

- **Full API surface** -- Scope management, tool/LLM calls, guardrails, intercepts, subscribers, and ATIF export exposed as Python-callable functions and classes.
- **Async bridge** -- Rust `Future`s are bridged to Python coroutines through `pyo3-async-runtimes` with Tokio backend.
- **abi3 stable ABI** -- Targets Python 3.11+ with a single compiled artifact that works across Python minor versions.
- **Type conversion** -- Automatic conversion between Python dicts/lists and Rust `serde_json::Value` via `pythonize`.
- **Callable wrapping** -- Python callables (sync and async) are wrapped for use as Nexus tool handlers, guardrails, and intercepts.

## Key Files

| File | Purpose |
|------|---------|
| `src/lib.rs` | PyO3 module definition and registration |
| `src/py_api.rs` | Python-exposed API functions |
| `src/py_types.rs` | Python type wrappers for core types |
| `src/py_callable.rs` | Wrapping Python callables for Rust callbacks |
| `src/py_context.rs` | Scope stack context propagation for Python |
| `src/convert.rs` | Rust-Python type conversion utilities |

## Build

The native extension is built automatically by Maturin when you run:

```bash
uv sync
```

This creates a virtual environment, installs dependencies, and compiles the Rust extension in-place. The build is configured via the root `pyproject.toml`.

## Documentation

See [docs/language-bindings.md](../../docs/language-bindings.md) for Python binding details.
