<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Relay)](https://github.com/NVIDIA/NeMo-Relay/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Relay/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Relay?color=green)](https://github.com/NVIDIA/NeMo-Relay/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Relay/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Relay)
[![PyPI](https://img.shields.io/pypi/v/nemo-relay?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-relay/)
[![npm node](https://img.shields.io/npm/v/nemo-relay-node?label=nemo-relay-node&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-relay-node)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay?label=nemo-relay&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-adaptive?label=nemo-relay-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-adaptive)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-cli?label=nemo-relay-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-cli)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Relay)

# NeMo Relay Python Bindings

This crate builds the native extension behind the public Python package
`nemo-relay`. It connects Python applications to the Rust NeMo Relay runtime
through PyO3 and Maturin.

Most Python users should install the Python package rather than depend on this
crate directly.

## Why Use It?

- **Bridge Python to the shared runtime**: Connect Python applications to the
  Rust NeMo Relay runtime without reimplementing runtime semantics in Python.
- **Build through standard Python packaging**: Use the repository
  `pyproject.toml`, Maturin, and PyO3 to produce the native extension behind
  `nemo-relay`.
- **Keep binding behavior aligned**: Expose the same scopes, middleware,
  plugins, lifecycle events, and adaptive helpers used by the rest of NeMo Relay.

## What You Get

- **Native extension**: The compiled `nemo_relay._native` module used by the
  public Python package.
- **Runtime APIs for Python**: Access to scopes, tool calls, LLM calls,
  middleware, subscribers, plugins, typed helpers, codecs, and adaptive helpers.
- **Shared Rust semantics**: Python behavior backed by the same runtime
  contract as the Rust crate.
- **Local development path**: `uv sync` builds the editable package and native
  extension from the repository root.

## Installation

Install the published Python package:

```bash
uv add nemo-relay
```

If you are not using `uv`, install it with `pip`:

```bash
pip install nemo-relay
```

For local source development from the repository root:

```bash
uv sync
```

## Getting Started

Import the public Python package and create a scoped runtime boundary:

```python
import nemo_relay

with nemo_relay.scope.scope("demo-agent", nemo_relay.ScopeType.Agent) as handle:
    nemo_relay.scope.event("initialized", handle=handle, data={"binding": "python"})
```

## Documentation

NeMo Relay Documentation: https://docs.nvidia.com/nemo/relay
