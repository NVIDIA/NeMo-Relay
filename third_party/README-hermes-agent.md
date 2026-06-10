<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hermes Agent Integration Notes

Use this page to choose the right Hermes integration path from the NeMo Relay
repository.

Most users should not patch Hermes from `third_party/hermes-agent`. Use the
table below to pick the runtime path that matches your goal.

| Path | Use When | Documentation |
|:--|:--|:--|
| NeMo Relay CLI wrapper | You want NeMo Relay to manage the local gateway lifetime for a Hermes process. Start with `nemo-relay hermes`. | [docs/nemo-relay-cli/hermes.mdx](../docs/nemo-relay-cli/hermes.mdx) |
| Standalone gateway and manual hook setup | You want to run the NeMo Relay gateway yourself and point Hermes hooks or provider traffic at it. | [docs/nemo-relay-cli/hermes.mdx](../docs/nemo-relay-cli/hermes.mdx) |
| Upstream Hermes plugin | You want Hermes itself to load the bundled `observability/nemo_relay` plugin and emit NeMo Relay observability. Observe-only plugin builds keep Hermes in control of LLM and tool execution. | [Hermes NeMo Relay plugin README](https://github.com/NousResearch/hermes-agent/blob/main/plugins/observability/nemo_relay/README.md) |
| Adaptive execution | You have a Hermes build with adaptive middleware support and a NeMo Relay runtime that exposes `llm.execute(...)` and `tools.execute(...)`. Verify the Hermes release tag before treating this as released behavior. | [Hermes NeMo Relay plugin README](https://github.com/NousResearch/hermes-agent/blob/main/plugins/observability/nemo_relay/README.md) |
| Legacy patch | You are maintaining or validating the pinned patch in this repository. | [patches/hermes-agent/notes.md](../patches/hermes-agent/notes.md) |

## Patch Maintenance Path

The files under `patches/hermes-agent/` are source-first maintenance artifacts
for replaying the tracked patch against the pinned third-party checkout. Use
them only when maintaining or validating the patch set itself.

Use [patches/hermes-agent/notes.md](../patches/hermes-agent/notes.md) as the
detailed patch-maintenance runbook. It covers the pinned checkout, editable
install with the `nemo-relay` extra, legacy environment variables, ATIF output,
OpenInference export, and smoke validation.

From the NeMo Relay repository root:

```bash
./scripts/bootstrap-third-party.sh
./scripts/apply-patches.sh --check
git -C third_party/hermes-agent apply ../../patches/hermes-agent/0001-add-nemo-relay-integration.patch
```

Then follow [patches/hermes-agent/notes.md](../patches/hermes-agent/notes.md)
for the Hermes-specific virtual environment and runtime configuration.

## Legacy Patch Runtime Example

If you are validating the patch path, enable the patch runtime in
`${HERMES_HOME:-$HOME/.hermes}/.env`:

```bash
HERMES_NEMO_RELAY_ENABLED=1
HERMES_NEMO_RELAY_ACG_ENABLED=1
HERMES_NEMO_RELAY_ATIF_DIR=${HERMES_HOME:-$HOME/.hermes}/atif
```

Then start Hermes from the patched checkout:

```bash
cd third_party/hermes-agent
. .venv/bin/activate
uv run hermes
```

The plugin registers Hermes lifecycle hooks and writes ATIF trajectory JSON on
session finalization. For patch smoke validation and OpenInference settings, use
the snippets in
[patches/hermes-agent/notes.md](../patches/hermes-agent/notes.md).
