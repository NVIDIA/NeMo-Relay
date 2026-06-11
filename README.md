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
[![npm wasm](https://img.shields.io/npm/v/nemo-relay-wasm?label=nemo-relay-wasm&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-relay-wasm)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay?label=nemo-relay&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-adaptive?label=nemo-relay-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-adaptive)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-cli?label=nemo-relay-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-cli)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Relay)

# NVIDIA NeMo Relay

NVIDIA NeMo Relay helps you see and control what happens inside agent runs
without rewriting the agent stack you already have. It gives coding agents,
applications, framework integrations, middleware, and observability backends a
shared runtime for scopes, policy, plugins, and lifecycle events.

The best first step is to get one real run on disk. Once Relay is writing raw
events and a trajectory file, you have something concrete to inspect, debug, and
build from.

## Start Here: Capture One Local Agent Run

This walkthrough gives you an end-to-end success signal. You install the
`nemo-relay` CLI, turn on local exporters, run either Codex or Claude Code
through Relay, and check that Relay wrote both raw events and normalized
trajectories.

> [!TIP]
> Start by trusting the raw Agent Trajectory Observability Format (ATOF) JSONL.
> It shows the lifecycle events Relay actually captured before anything is
> translated into Agent Trajectory Interchange Format (ATIF), OpenTelemetry, or
> OpenInference output.

### 1. Install the CLI

```bash
cargo install nemo-relay-cli
```

If you use `cargo-binstall`, the CLI can also be installed with:

```bash
cargo binstall nemo-relay-cli
```

### 2. Enable Local Observability Output

From the project directory you want to observe, open the project-scoped plugin
editor:

```bash
nemo-relay plugins edit --project
```

The editor creates or updates the nearest project plugin file at
`.nemo-relay/plugins.toml`. In the menu:

1. Enable the `Observability` component.
2. Open `ATOF`, toggle the section on, and set:
   - `output_directory` to `.nemo-relay/atof`
   - `filename` to `events.jsonl`
   - `mode` to `overwrite`
3. Open `ATIF`, toggle the section on, and set:
   - `output_directory` to `.nemo-relay/atif`
   - `filename_template` to `trajectory-{session_id}.json`
4. Press `p` to preview the generated TOML.
5. Press `s` to save.

> [!NOTE]
> Use `nemo-relay plugins edit` without `--project` only when you want these
> exporter settings in your user-level Relay config instead of this one project.

### 3. Run Codex or Claude Code Through Relay

Use the host CLI that is installed on your machine.

```bash
nemo-relay codex -- exec "Summarize this repository."
```

```bash
nemo-relay claude -- "Summarize this repository."
```

The transparent wrapper starts a local Relay gateway, injects host-specific hook
and provider settings for that launched process, then shuts the gateway down
when the agent exits.

> [!WARNING]
> Codex users may need to review and activate generated hooks before events
> appear. See the [Codex CLI guide](https://docs.nvidia.com/nemo/relay/nemo-relay-cli/codex) for the
> current hook activation caveat and troubleshooting steps.

### 4. Verify the Run

After the run exits, check that raw events and trajectory files were written:

```bash
test -s .nemo-relay/atof/events.jsonl
ls .nemo-relay/atif/*.json
for file in .nemo-relay/atif/*.json; do
  python3 -m json.tool "$file" >/dev/null
done
```

Then verify that at least one raw ATOF `0.1` event exists:

```bash
python3 - <<'PY'
from pathlib import Path
import json

events_path = Path(".nemo-relay/atof/events.jsonl")
events = [
    json.loads(line)
    for line in events_path.read_text().splitlines()
    if line.strip()
]

assert events, "no ATOF events were written"
assert any(event.get("atof_version") == "0.1" for event in events), "no ATOF 0.1 events found"
print(f"validated {len(events)} ATOF event(s)")
PY
```

A successful run gives you two things to inspect:

- `.nemo-relay/atof/events.jsonl`, the raw canonical event stream.
- One or more `.nemo-relay/atif/*.json` trajectory files for analysis and
  evaluation workflows.

> [!TIP]
> If raw ATOF events exist but LLM spans are missing, provider traffic probably
> isn't flowing through the Relay gateway. If ATIF is missing, make sure the
> agent session or turn ended and the output directory is writable. Use
> [NeMo Relay CLI](https://docs.nvidia.com/nemo/relay/nemo-relay-cli/about) when you are ready for
> persistent host plugin installation, gateway configuration, exporter options,
> and agent-specific diagnostics.

## Choose Your Next Path

Pick the row closest to what you are trying to do next.

| Goal | Start With |
|---|---|
| Observe Codex, Claude Code, Cursor, or Hermes locally | [NeMo Relay CLI](https://docs.nvidia.com/nemo/relay/nemo-relay-cli/about) |
| Instrument app-owned LLM or tool calls | [Quick Start](https://docs.nvidia.com/nemo/relay/getting-started/quick-start) |
| Use LangChain, LangGraph, Deep Agents, or OpenClaw | [Supported Integrations](https://docs.nvidia.com/nemo/relay/supported-integrations/about) |
| Build a framework or provider integration | [Integrate into Frameworks](https://docs.nvidia.com/nemo/relay/integrate-into-frameworks/about) |
| Export ATOF, ATIF, OpenTelemetry, or OpenInference | [Observability Plugin](https://docs.nvidia.com/nemo/relay/observability-plugin/about) |
| Package reusable middleware or exporters | [Build Plugins](https://docs.nvidia.com/nemo/relay/build-plugins/about) |
| Develop or test this repository from source | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Application Quick Starts

If you own the code that calls the model or tool, install the binding for your
language and route that boundary through Relay directly.

```bash
# Python
uv add nemo-relay

# Node.js
npm install nemo-relay-node

# Rust
cargo add nemo-relay
```

Then run the smallest workflow for that binding:

- [Python Quick Start](https://docs.nvidia.com/nemo/relay/getting-started/quick-start/python)
- [Node.js Quick Start](https://docs.nvidia.com/nemo/relay/getting-started/quick-start/nodejs)
- [Rust Quick Start](https://docs.nvidia.com/nemo/relay/getting-started/quick-start/rust)

The Node.js package requires Node.js 24 or newer.

## What Relay Adds

Relay is for the messy middle of agent systems. A production application may
combine NeMo Agent Toolkit, LangChain, LangGraph, provider SDKs, custom harness
code, NeMo Guardrails, tracing systems, and evaluation pipelines. Relay gives
those pieces one runtime contract instead of asking every layer to invent its
own wrappers and trace vocabulary.

Relay gives those systems:

- 🧭 **Scopes** so runs, turns, tools, LLM calls, and subagents have clear
  ownership, parent-child lineage, cleanup boundaries, and
  request isolation.
- ⚙️ **Managed LLM and tool calls** so the same lifecycle and middleware rules
  apply around each callback.
- 🛡️ **Middleware** for the places where Relay must block, sanitize, transform,
  route, retry, or replace execution.
- 🧩 **Plugins** so reusable observability, guardrail, adaptive, and exporter
  behavior can be turned on from configuration.
- 📡 **Events and subscribers** so raw ATOF, normalized ATIF, OpenTelemetry, and
  OpenInference output all come from the same runtime stream.

Relay does not replace your framework, model provider, application logic,
observability backend, or guardrail authoring system. It gives those systems a
common boundary to meet at.

```mermaid
flowchart LR
    App[Application, Framework, or CLI Harness]

    subgraph Runtime[NeMo Relay Runtime]
        direction TB
        Scopes[Scopes]
        Middleware[Middleware]
        Plugins[Plugins]
        Events[Lifecycle Events]
    end

    Output[Subscribers and Exporters]

    App --> Scopes
    App --> Middleware
    Plugins --> Middleware
    Scopes --> Events
    Middleware --> Events
    Events --> Output
```

## Support Status

> [!NOTE]
> The main supported paths today are Rust, Python, and Node.js. Go,
> WebAssembly, and raw C FFI are available for source-first users, but they are
> still experimental.

The following table shows which language bindings and CLI features are currently supported:

| Binding | Status | Notes |
|---|---|---|
| Python | ✅ Fully supported | Documented with Quick Start and Guides. |
| Node.js | ✅ Fully supported | Documented with Quick Start and Guides. |
| Rust | ✅ Fully supported | Documented with Quick Start and Guides. |
| NeMo Relay CLI | 🚧 Experimental | Install with `cargo install nemo-relay-cli`. |
| Go | 🚧 Experimental | Source-first under `go/nemo_relay`. |
| WebAssembly | 🚧 Experimental | Source-first under `crates/wasm`. |
| FFI | 🚧 Experimental | Source-first under `crates/ffi`. |

### Agent Harness Support

The CLI support is intentionally conservative. Observability works for the
listed harnesses; security and optimization depend on the hooks, provider
routes, and execution controls each harness exposes.

| Agent | Observability | Security | Optimization | Notes |
|:--|:--:|:--:|:--:|:--|
| Claude Code | ✅ Yes | ⚠️ Partial | ⚠️ Partial | Hook forwarding and gateway-routed LLM observability are supported. |
| Codex | ✅ Yes | ⚠️ Partial | ⚠️ Partial | Hook activation and missing session-end behavior limit full feature parity. |
| Hermes Agent | ✅ Yes | ⚠️ Partial | ⚠️ Partial | Hook forwarding and gateway-routed LLM observability are supported. |
| Cursor | 🚧 Partial | ⚠️ Partial | ❌ No | Missing hooks under `cursor-agent` limit full feature coverage. |

### Public API Integrations

Use these integrations when the framework exposes stable callbacks, middleware,
or plugin hooks that preserve enough lifecycle fidelity.

| Agent / Library | Observability | Security | Optimization | Notes |
|:--|:--:|:--:|:--:|:--|
| LangChain | ✅ Yes | ✅ Yes | ✅ Yes | Wrapped tool and LLM calling. |
| LangGraph | ✅ Yes | ✅ Yes | ✅ Yes | Wrapped tool and LLM calling. |
| Deep Agents | ✅ Yes | ✅ Yes | ✅ Yes | Wrapped tool and LLM calling. |
| OpenClaw | ✅ Yes | ⚠️ Partial | ❌ No | Hook-backed telemetry with pre-tool guardrails. Managed execution rewrites require the patch-based integration. |

The Python `nemo-relay` package ships extras for LangChain, LangGraph, and Deep
Agents:

```bash
uv add "nemo-relay[langchain,langgraph,deepagents]"
```

See [Supported Integrations](https://docs.nvidia.com/nemo/relay/supported-integrations/about) for setup
guides and current caveats.

### Patch-based Integrations

Patch-based integrations are experimental samples maintained against pinned
upstream checkouts. Use [third_party/README.md](third_party/README.md) for the
clone, checkout, and patch-application workflow.

| Integration | Observability | Security | Optimization | Notes |
|:---|:---:|:---:|:---:|:---|
| [LangChain](third_party/README-langchain.md), [LangGraph](third_party/README-langgraph.md), [LangChain NVIDIA](third_party/README-langchain-nvidia.md) | ✅ Yes | ✅ Yes | ✅ Yes | Directly patches behavior into code. |
| [opencode](third_party/README-opencode.md) | ✅ Yes | ✅ Yes | ✅ Yes | Directly patches behavior into code. |
| [OpenClaw](third_party/README-openclaw.md) | ✅ Yes | ✅ Yes | ✅ Yes | Adds middleware support to OpenClaw and a built-in plugin. |
| [Hermes Agent](third_party/README-hermes-agent.md) | ✅ Yes | ✅ Yes | ✅ Yes | Directly patches behavior into code. |

## Documentation

End-user documentation lives at
[docs.nvidia.com/nemo/relay](https://docs.nvidia.com/nemo/relay).

Important local entry points:

- [Overview](https://docs.nvidia.com/nemo/relay/about-nemo-relay/overview)
- [Installation](https://docs.nvidia.com/nemo/relay/getting-started/installation)
- [Agent Runtime Primer](https://docs.nvidia.com/nemo/relay/getting-started/agent-runtime-primer)
- [Testing and Docs](https://docs.nvidia.com/nemo/relay/contribute/testing-and-docs)

For source builds, tests, and contribution workflow, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

- [ ] NemoClaw support and integration for managed tool and LLM execution flows.
- [ ] Deeper NVIDIA NeMo ecosystem integration across agent, guardrail,
      evaluation, and observability workflows.
- [ ] Expanded adaptive optimization capabilities for performance-aware
      scheduling, hints, and cache behavior.
- [ ] First-party plugins and packages for common agent runtimes and frameworks
      where upstream extension points allow it.

## License

NVIDIA NeMo Relay is licensed under the Apache License 2.0.
