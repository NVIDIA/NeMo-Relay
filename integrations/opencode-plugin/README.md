<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow OpenCode Plugin

This package is a standalone OpenCode server plugin for NeMo Flow observability.
It uses OpenCode's public plugin API and does not require patching OpenCode.

For the illustrated setup guide and demo recording script, see
`docs/integrate-frameworks/opencode.md` in the NeMo Flow source checkout.

## Configuration

Use the plugin from an OpenCode config file. From a NeMo Flow source checkout,
use a file URL:

```json
{
  "plugin": [
    [
      "file:///absolute/path/to/NeMo-Flow/integrations/opencode-plugin",
      {
        "enabled": true,
        "atofPath": "./.nemoflow/opencode.atof.jsonl",
        "atifPath": "./.nemoflow/opencode.atif.json",
        "logPath": "./.nemoflow/opencode-plugin.log"
      }
    ]
  ]
}
```

When this package is published, replace the file URL with the package name:

```json
{
  "plugin": [
    [
      "@nvidia/nemoflow-opencode-plugin",
      {
        "enabled": true,
        "atofPath": "./.nemoflow/opencode.atof.jsonl",
        "atifPath": "./.nemoflow/opencode.atif.json",
        "logPath": "./.nemoflow/opencode-plugin.log"
      }
    ]
  ]
}
```

The package loads `nemo-flow-node` dynamically. If the native Node binding is
missing or cannot initialize, the plugin logs one pass-through warning and does
not change OpenCode behavior.

## Compatibility

The plugin declares support for OpenCode `>=1.14.40`. It uses the public
OpenCode server plugin hooks that are available in `@opencode-ai/plugin`
`1.14.40` and were verified against OpenCode `1.14.41`.

## Output

- `atofPath` receives raw NeMo Flow ATOF JSONL events for OpenCode session,
  message, LLM request metadata, error, and successful tool lifecycle records.
- `atifPath` receives a session trajectory when OpenCode reports a session as
  idle or deleted.
- `logPath` receives JSONL plugin diagnostics.

## Current Limitations

This plugin uses only existing OpenCode hooks. OpenCode does not yet expose an
around-style LLM stream hook or tool execution hook, so the plugin cannot record
exact LLM stream duration, tool error spans for every failure path, request
intercepts, execution intercepts, or conditional guardrail blocking.
