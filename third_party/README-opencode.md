<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# opencode Patch Setup

This directory contains the NeMo Flow integration patch for
`third_party/opencode`.

The patch adds optional NeMo Flow tracing, LLM stream wrapping, tool execution
wrapping, raw ATOF JSONL export, and optional direct ATIF export support to the
opencode package. It depends on the local NeMo Flow Node binding through a
`file:` dependency that resolves from `third_party/opencode/packages/opencode`
back to `crates/node`.

## Setup

From the NeMo Flow repository root:

```bash
./scripts/bootstrap-third-party.sh
./scripts/apply-patches.sh --check
git -C third_party/opencode apply ../../patches/opencode/0001-add-nemo-flow-integration.patch
```

Install opencode dependencies with Bun:

```bash
cd third_party/opencode
bun install --frozen-lockfile
```

For runtime smoke tests that load `nemo-flow-node`, build the Node binding from
the NeMo Flow repository root first:

```bash
cd ../../crates/node
npm install
npm run build
```

Enable the integration at runtime with either `NEMO_FLOW_ENABLED=1` or the
opencode experimental `nemo_flow` config flag. If the native addon is missing,
the integration logs a warning and disables itself.

## Usage Example

Run opencode with the NeMo Flow integration enabled by environment variable:

```bash
cd third_party/opencode
NEMO_FLOW_ENABLED=1 bun --cwd packages/opencode run dev
```

Alternatively, enable the patched experimental config flag:

```json
{
  "experimental": {
    "nemo_flow": true
  }
}
```

When enabled, opencode creates NeMo Flow scopes for agents and batched tool
execution, wraps LLM streams and tool calls, and registers a raw ATOF JSONL
subscriber. Set `NEMO_FLOW_ATOF_DIR` to control where `events.jsonl` is written;
otherwise it defaults to the opencode data directory's `atof` subdirectory.

Direct ATIF export is optional comparison output. Set `NEMO_FLOW_ATIF_DIR` to
control where exported ATIF JSON files are written when a session becomes idle;
otherwise it defaults to the opencode data directory's `atif` subdirectory.

The tool wrapper keeps opencode's execution on original JavaScript values while
passing JSON-safe snapshots to the NeMo Flow native observer. This avoids
`structuredClone()` failures in opencode while still preserving NeMo Flow tool
events.

## Validation

Run the opencode package typecheck:

```bash
cd third_party/opencode/packages/opencode
bun run typecheck
```

Also rerun the patch applicability check from the NeMo Flow repository root:

```bash
./scripts/apply-patches.sh --check
```

For an end-to-end smoke, run an opencode task with `NEMO_FLOW_ENABLED=1` and
verify that the configured `NEMO_FLOW_ATOF_DIR` contains an `events.jsonl` file
with scope and tool/LLM events.
