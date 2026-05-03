<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Flow)](https://github.com/NVIDIA/NeMo-Flow/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Flow/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Flow?color=green)](https://github.com/NVIDIA/NeMo-Flow/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Flow/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Flow)
[![PyPI](https://img.shields.io/pypi/v/nemo-flow?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-flow/)
[![npm node](https://img.shields.io/npm/v/nemo-flow-node?label=nemo-flow-node&color=CC3534&logo=npm)](https://registry.npmjs.org/nemo-flow-node)
[![npm wasm](https://img.shields.io/npm/v/nemo-flow-wasm?label=nemo-flow-wasm&color=CC3534&logo=npm)](https://registry.npmjs.org/nemo-flow-wasm)
[![Crates.io](https://img.shields.io/crates/v/nemo-flow?label=nemo-flow&color=B7410E&logo=rust)](https://crates.io/crates/nemo-flow)
[![Crates.io](https://img.shields.io/crates/v/nemo-flow-adaptive?label=nemo-flow-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-flow-adaptive)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Flow)

# NeMo Flow Python Bindings

This crate builds the native extension behind the public Python package
`nemo-flow`. It connects Python applications to the Rust NeMo Flow runtime
through PyO3 and Maturin.

Most Python users should install the Python package rather than depend on this
crate directly.

## Why Use It?

- 🧩 **Bridge Python to the shared runtime**: Connect Python applications to the
  Rust NeMo Flow runtime without reimplementing runtime semantics in Python.
- 🛠️ **Build through standard Python packaging**: Use the repository
  `pyproject.toml`, Maturin, and PyO3 to produce the native extension behind
  `nemo-flow`.
- 🔁 **Keep binding behavior aligned**: Expose the same scopes, middleware,
  plugins, lifecycle events, and adaptive helpers used by the rest of NeMo Flow.

## What You Get

- ✅ **Native extension**: The compiled `nemo_flow._native` module used by the
  public Python package.
- ✅ **Runtime APIs for Python**: Access to scopes, tool calls, LLM calls,
  middleware, subscribers, plugins, typed helpers, codecs, and adaptive helpers.
- ✅ **Shared Rust semantics**: Python behavior backed by the same runtime
  contract as the Rust crate.
- ✅ **Local development path**: `uv sync` builds the editable package and native
  extension from the repository root.

## Installation

Install the published Python package:

```bash
uv add nemo-flow
```

If you are not using `uv`, install it with `pip`:

```bash
pip install nemo-flow
```

For local source development from the repository root:

```bash
uv sync
```

## Getting Started

Import the public Python package and create a scoped runtime boundary:

```python
import nemo_flow

with nemo_flow.scope.scope("demo-agent", nemo_flow.ScopeType.Agent) as handle:
    nemo_flow.scope.event("initialized", handle=handle, data={"binding": "python"})
```

## Documentation

NeMo Flow Documentation: https://nvidia.github.io/NeMo-Flow
