<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Installation

Choose the installation path that matches how you plan to use NeMo Flow.

## Install from a Package Manager

Use a package manager when you are consuming a published release in an application workspace.

### Python

Install the Python package when your application uses NeMo Flow through the
Python wrapper.

```bash
uv add nemo-flow
```

Use `uv add` from an application project that has a `pyproject.toml`; it records
`nemo-flow` as a project dependency. If you are only installing into an active
virtual environment and do not have project metadata, use `uv pip install
nemo-flow` instead. You can also use `pip install nemo-flow` if you are not
managing the environment with `uv`.

### Node.js

Install the Node.js package when your application uses NeMo Flow through the
JavaScript API.

```bash
npm install nemo-flow-node
```

### Rust

Add the Rust crates when your application uses NeMo Flow directly from Rust.

```toml
[dependencies]
nemo-flow = "0.1.*"
nemo-flow-adaptive = "0.1.*"
```

- `nemo-flow` provides the core runtime APIs for scopes, middleware, subscribers, plugins, tool calls, and LLM calls.
- `nemo-flow-adaptive` provides adaptive runtime primitives and Redis-backed learning components when you want adaptive optimization behavior in Rust.
- `nemo-flow-cli` is a published binary crate for the NeMo Flow CLI. Install it
  with `cargo install nemo-flow-cli` when you need the `nemo-flow` executable
  for coding-agent hook and LLM gateway observability.

## Install from Source

Use a source checkout when you want an application to consume a local NeMo Flow repository:

```bash
git clone https://github.com/NVIDIA/NeMo-Flow.git ../NeMo-Flow
```

### Python

Use an editable Python install when your application should import a local
checkout.

```bash
pip install -e ../NeMo-Flow
```

If your application is a `uv` project, add the local checkout as an editable
dependency from the application directory:

```bash
uv add --editable ../NeMo-Flow
```

That command records the source path in your application's `pyproject.toml`.

Run `uv sync` from the cloned `../NeMo-Flow` checkout when you are developing
the NeMo Flow repository itself.

### Node.js

Build and install the local Node.js package when your application should consume
the checkout version.

```bash
cd ../NeMo-Flow
npm install --ignore-scripts
npm run build --workspace=nemo-flow-node
cd -
npm install ../NeMo-Flow/crates/node
```

### Rust

Use path dependencies from your application:

```toml
[dependencies]
nemo-flow = { path = "../NeMo-Flow/crates/core" }
nemo-flow-adaptive = { path = "../NeMo-Flow/crates/adaptive" }
```

Install the published NeMo Flow CLI when you need coding-agent hook and LLM
gateway observability:

```bash
cargo install nemo-flow-cli
```

## Install from the Repository

Use the repository workflow when you are developing against local source, validating unpublished changes, or working across multiple bindings.

### Development Tooling

Install `uv` by following the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/), then verify that it is available:

```bash
uv --version
```

Install the required development task runner before you use the repository build, test, or documentation commands:

```bash
cargo install just --locked
```

Verify that `just` is available:

```bash
just --version
```

### Python Development Setup

Use this setup when you need the Python development environment for tests or docs
tooling.

```bash
uv sync
```

This command installs the Python package in editable mode, builds the native extension through `maturin`, and installs the docs, test, and development dependencies used across the repo.

### Node.js Development Setup

Use this setup when you need the Node.js package dependencies for binding work.

```bash
npm install --ignore-scripts
npm run build --workspace=nemo-flow-node
```

### Rust Development Setup

Use this setup when you need Rust tooling for core runtime or native binding work.

```bash
cargo build --workspace
```

If you only need the core Rust crate:

```bash
cargo build -p nemo-flow
```

## Documentation Tooling

Use these commands when you work on the documentation site:

```bash
just docs
just docs-linkcheck
```
