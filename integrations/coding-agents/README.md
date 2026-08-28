<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Coding Agent Integrations

NeMo Relay CLI supports Claude Code and Codex through generated marketplace
plugins, lifecycle hooks, provider routing, and a shared local gateway.

The source manifests in this directory are development inputs. End users should
install the generated integrations through the CLI:

```bash
nemo-relay install claude-code
nemo-relay install codex
```

Use `nemo-relay doctor --plugin <host>` to verify an installation and
`nemo-relay uninstall <host>` to remove Relay-owned host state. The transparent
wrappers remain available as `nemo-relay claude` and `nemo-relay codex`.

If normal removal is blocked by stale Relay state or MCP generation metadata,
use `nemo-relay uninstall <host> --force`. It attempts each Relay-owned cleanup
step independently, then reports any steps that could not complete. If host
setup cleanup fails, Relay keeps its state so a later forced uninstall can retry.

Host-specific source notes live in:

- [`claude-code/`](claude-code/README.md)
- [`codex/`](codex/README.md)

Hermes Agent does not use a NeMo Relay CLI integration. NeMo Relay is built
into Hermes Agent, with no separate observability plugin or Relay CLI setup
required. Hermes Agent understands NeMo Relay plugin configurations.

For repository validation, use the canonical Rust and documentation recipes:

```bash
just test-rust
just docs
just docs-linkcheck
```
