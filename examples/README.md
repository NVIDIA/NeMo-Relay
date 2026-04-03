<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nexus Examples

This directory contains runnable examples demonstrating NeMo Agent Toolkit
Nexus features. The catalog is split into minimal runtime examples and
heavier-weight integration examples.

All minimal examples use active scopes plus the managed execution APIs. They do
not demonstrate the low-level manual lifecycle start/end APIs.

## Minimal Examples

| Example | Runtime | Description | Prerequisites |
|---------|---------|-------------|---------------|
| [python/basic_scope_and_tool.py](python/basic_scope_and_tool.py) | Python | Minimal scope, subscriber, and tool execution example. | Python >= 3.11, `uv sync` |
| [python/atif_export.py](python/atif_export.py) | Python | Minimal ATIF export example around a single tool call. | Python >= 3.11, `uv sync` |
| [node/basic.cjs](node/basic.cjs) | Node.js | Minimal scope and tool execution example using the Node binding. | Node.js LTS, `cd crates/node && npm install && npm run build` |
| [go/basic/main.go](go/basic/main.go) | Go | Minimal scope and tool execution example using the Go binding. | Go >= 1.21, `cargo build --release -p nvidia-nat-nexus-ffi` |
| [wasm/basic.cjs](wasm/basic.cjs) | WASM | Minimal scope and tool execution example using the generated WASM package. | `wasm-pack`, Node.js, `wasm-pack build crates/wasm --scope nvidia --target nodejs` |

## Integration Examples

| Example | Runtime | Description | Prerequisites |
|---------|---------|-------------|---------------|
| [agent_with_logging.py](agent_with_logging.py) | Python + LangChain | A LangChain ReAct agent using ChatNVIDIA with Nexus event logging and ATIF trajectory export. | Python >= 3.11, `uv sync`, `NVIDIA_API_KEY` |

## Running Examples

### Python

```bash
uv sync

uv run python examples/python/basic_scope_and_tool.py
uv run python examples/python/atif_export.py
uv run python examples/agent_with_logging.py
```

### Node.js

```bash
cd crates/node
npm install
npm run build
node ../../examples/node/basic.cjs
```

### Go

```bash
cargo build --release -p nvidia-nat-nexus-ffi
cd go/nat_nexus
CGO_LDFLAGS="-L../../target/release" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}../../target/release" \
go run ../../examples/go/basic/main.go
```

### WASM

```bash
wasm-pack build crates/wasm --scope nvidia --target nodejs
node examples/wasm/basic.cjs
```

## Adding New Examples

When adding a new example:

1. Include the SPDX license header at the top of the file.
2. Add a module-level docstring describing what the example demonstrates, its prerequisites, and how to run it.
3. Place minimal runtime examples under `examples/<runtime>/`.
4. Update the relevant table in this README.
5. Keep examples focused on a single feature or integration pattern.
