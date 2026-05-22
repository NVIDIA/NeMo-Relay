<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hermes Agent Patch Setup

This directory contains the maintained NeMo Relay integration patch for
`third_party/hermes-agent`.

Use [patches/hermes-agent/notes.md](../patches/hermes-agent/notes.md) as the
detailed operator runbook. It covers the pinned checkout, editable install with
the `nemo-relay` extra, environment variables, ATIF output, OpenInference export,
and smoke validation.

## Quick Path

From the NeMo Relay repository root:

```bash
./scripts/bootstrap-third-party.sh
./scripts/apply-patches.sh --check
git -C third_party/hermes-agent apply ../../patches/hermes-agent/0001-add-nemo-relay-integration.patch
```

Then follow [patches/hermes-agent/notes.md](../patches/hermes-agent/notes.md)
for the Hermes-specific virtual environment and runtime configuration.

## Usage Example

Enable the integration in `${HERMES_HOME:-$HOME/.hermes}/.env`:

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
session finalization. For a no-model smoke path and OpenInference settings, use
the validation snippets in
[patches/hermes-agent/notes.md](../patches/hermes-agent/notes.md).
