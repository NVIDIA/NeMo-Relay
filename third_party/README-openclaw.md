<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenClaw Patch Setup

This directory contains the NeMo Flow integration patch for
`third_party/openclaw`.

The patch adds an OpenClaw NeMo Flow extension plus agent runtime middleware
registration points. It depends on the local NeMo Flow Node binding through a
`file:` dependency that resolves from `third_party/openclaw` back to
`crates/node`.

## Setup

From the NeMo Flow repository root:

```bash
./scripts/bootstrap-third-party.sh
./scripts/apply-patches.sh --check
git -C third_party/openclaw apply ../../patches/openclaw/0001-add-nemo-flow-integration.patch
```

Install OpenClaw dependencies using its pinned package manager. `pnpm` is not
assumed to be globally installed, so this command uses the version declared by
the OpenClaw workspace:

```bash
cd third_party/openclaw
npx -y pnpm@10.32.1 install --frozen-lockfile --ignore-scripts
```

For runtime smoke tests that load `nemo-flow-node`, build the Node binding from
the NeMo Flow repository root first:

```bash
cd ../../crates/node
npm install
npm run build
```

## Usage Example

Install or enable the local extension from the patched OpenClaw checkout, then
configure the NeMo Flow plugin host directly under the OpenClaw plugin config:

```json
{
  "plugins": {
    "entries": {
      "nemo-flow": {
        "enabled": true,
        "config": {
          "version": 1,
          "components": [
            {
              "kind": "observability",
              "enabled": true,
              "config": {
                "version": 1,
                "atif": {
                  "enabled": true,
                  "agent_name": "openclaw",
                  "output_directory": "./nemo-flow-atif"
                }
              }
            }
          ],
          "policy": {
            "unknown_component": "warn",
            "unknown_field": "warn",
            "unsupported_value": "error"
          }
        }
      }
    }
  }
}
```

With that config, the patched plugin initializes the NeMo Flow plugin host and
activates the `observability` component. Wrapping is implicit when the plugin is
enabled and initialized: the extension registers PI runtime streaming LLM and
tool-call middleware with OpenClaw.

The patched plugin config is the canonical NeMo Flow plugin-host document. Old
wrapper keys are rejected, including `enabled`, `backend`, `capture`,
`correlation`, `plugins`, `nemoFlow`, `atif`, and `telemetry`. Configure
observability through component-local keys such as `atof`, `atif`,
`opentelemetry`, and `openinference`.

## Validation

Run the focused OpenClaw NeMo Flow tests:

```bash
cd third_party/openclaw
npx -y pnpm@10.32.1 exec node scripts/run-vitest.mjs run \
  --config vitest.config.ts \
  extensions/nemo-flow/src/runtime.test.ts \
  src/plugins/agent-runtime-middleware.test.ts
```

Also rerun the patch applicability check from the NeMo Flow repository root:

```bash
./scripts/apply-patches.sh --check
```
