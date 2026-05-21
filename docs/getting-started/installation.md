<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Installation

Use this page when you are consuming a published NeMo Flow release from a
package manager.

If you are working from a source checkout, validating unpublished changes, or
contributing to the repository, use
[Development Setup](../contribute/development-setup.md) instead.

## CLI

Install the NeMo Flow CLI when you want the `nemo-flow` executable for
coding-agent hook and LLM gateway observability.

```bash
cargo install nemo-flow-cli@0.3.0
```

## Python

Install the Python package when your application uses NeMo Flow through the
Python wrapper.

```bash
uv add nemo-flow@0.3.0
```

Use `uv add` from an application project that has a `pyproject.toml`; it records
`nemo-flow` as a project dependency. If you are only installing into an active
virtual environment and do not have project metadata, use `uv pip install
nemo-flow` instead. You can also use `pip install nemo-flow` if you are not
managing the environment with `uv`.

## Node.js

Install the Node.js package when your application uses NeMo Flow through the
JavaScript API.

```bash
npm install nemo-flow-node
```

## Rust

Add the Rust crates when your application uses NeMo Flow directly from Rust.

```bash
cargo add nemo-flow@0.3.0
cargo add nemo-flow-adaptive@0.3.0
```

- `nemo-flow` provides the core runtime APIs for scopes, middleware, subscribers, plugins, tool calls, and LLM calls.
- `nemo-flow-adaptive` provides adaptive runtime primitives and Redis-backed learning components when you want adaptive tuning behavior in Rust.

## Integrations

Install integration packages when your application already uses one of the
supported framework or agent harness surfaces.

### OpenClaw

Install the OpenClaw plugin through OpenClaw so OpenClaw can register and manage
the package:

```bash
openclaw plugins install npm:nemo-flow-openclaw@0.3.0
openclaw gateway restart
```

Use the package name `nemo-flow-openclaw` for installation. Use the plugin ID
`nemo-flow` in OpenClaw configuration, inspection, and gateway status commands.
See the [OpenClaw Plugin Guide](../integrations/openclaw-plugin.md) for
configuration and verification steps.

### Python Framework Integrations

Install the Python package with the supported framework extras when your
application uses LangChain, LangGraph, or Deep Agents.

```bash
uv add "nemo-flow[langchain,langgraph,deepagents]@0.3.0"
```

The extras install the NeMo Flow Python package plus the dependencies needed by
the maintained public integrations. See
[Supported Integrations](../integrations/about.md) for guide links and support
levels.
