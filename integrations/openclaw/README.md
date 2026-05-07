<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow OpenClaw Observability

This package provides the `nemo-flow` OpenClaw plugin. It replays OpenClaw hook
events through NeMo Flow manual lifecycle APIs so OpenClaw sessions can emit ATIF
JSON, OpenTelemetry, and OpenInference telemetry without patching OpenClaw.

The package declares both OpenClaw entrypoint styles:

- `openclaw.extensions`: `./index.ts` for source-based plugin workflows.
- `openclaw.runtimeExtensions`: `./dist/index.js` for built runtime workflows.

## Build And Validate

```bash
npm --prefix integrations/openclaw install
npm --prefix integrations/openclaw run typecheck
npm --prefix integrations/openclaw test
npm --prefix integrations/openclaw run build
npm --prefix integrations/openclaw run pack:check
```

`npm run build` emits production files under `dist/`. Tests compile to
`.test-dist/` so test artifacts do not enter the installable package.

The optional live smoke test requires a working installed `nemo-flow-node` binding:

```bash
npm --prefix integrations/openclaw run test:live
```

## Enablement

Allow the plugin id and grant conversation hook access when OpenClaw runs with a
restrictive plugin configuration:

```json
{
  "plugins": {
    "allow": ["nemo-flow"],
    "entries": {
      "nemo-flow": {
        "enabled": true,
        "hooks": {
          "allowConversationAccess": true
        },
        "config": {}
      }
    }
  }
}
```

`plugins.allow` controls plugin trust and loading. `hooks.allowConversationAccess`
lets trusted non-bundled plugins receive conversation-sensitive hook payloads such
as LLM prompts, LLM responses, agent finalization messages, and tool payloads.

## Configuration

ATIF export is enabled by default. OTel and OpenInference subscribers are disabled
until explicitly configured.

ATIF-only local export:

```json
{
  "atif": {
    "enabled": true,
    "outputDir": "./nemo-flow-atif"
  },
  "telemetry": {
    "otel": {
      "enabled": false
    },
    "openInference": {
      "enabled": false
    }
  }
}
```

OpenTelemetry OTLP export:

```json
{
  "telemetry": {
    "otel": {
      "enabled": true,
      "transport": "http_binary",
      "endpoint": "http://localhost:4318/v1/traces",
      "serviceName": "openclaw-nemo-flow"
    }
  }
}
```

OpenInference/Phoenix OTLP export:

```json
{
  "telemetry": {
    "openInference": {
      "enabled": true,
      "transport": "http_binary",
      "endpoint": "http://localhost:6006/v1/traces",
      "serviceName": "openclaw-nemo-flow"
    }
  }
}
```

Privacy defaults:

```json
{
  "capture": {
    "includePrompts": true,
    "includeResponses": true,
    "stripToolArgs": true,
    "stripToolResults": true
  }
}
```

Prompts and responses are captured by default. Tool arguments and tool results are
stripped by default because they often contain user data, local paths, tokens, or
large payloads.

## Event Mapping

| OpenClaw hook | NeMo Flow behavior |
| --- | --- |
| `gateway_start` | Emits a gateway lifecycle mark. |
| `gateway_stop` | Stops the runtime, drains sessions, shuts down subscribers, and clears the NeMo Flow plugin host. |
| `session_start` | Opens or aliases a NeMo Flow session scope. |
| `session_end` | Closes the session, flushes pending replay state, and exports ATIF if enabled. |
| `llm_input` / `llm_output` | Replays an LLM span through `llmCall` and `llmCallEnd` with bounded FIFO correlation. |
| `model_call_started` / `model_call_ended` | Enriches matching LLM spans with provider timing when correlation is unambiguous. |
| `after_tool_call` | Replays successful tool calls through `toolCall` and `toolCallEnd`; blocked tools emit marks. |
| `agent_end` | Emits an agent lifecycle mark and closes any remaining session state for the run. |
| `before_agent_finalize` | Emits a lifecycle mark and does not mutate the finalization payload. |
| `subagent_spawned` / `subagent_ended` | Emits subagent lifecycle marks under the best available parent or child session. |

## Health

The plugin registers the admin-scoped gateway method `nemoFlow.status`.

The response reports:

- backend status: `not_initialized`, `disabled`, `ready`, `degraded`, `stopping`, or `stopped`
- output health for `atif`, `otel`, and `openInference`
- replay counters, including replayed LLM spans, replayed tool spans, emitted marks,
  ATIF files written, replay errors, and skipped events
- last degraded or unavailable reason when present

Use the output health independently:

- ATIF: confirm JSON files appear in the configured `atif.outputDir`.
- OTel: confirm spans arrive at the configured OTLP collector.
- OpenInference: confirm spans arrive at the configured OpenInference/Phoenix endpoint.

## Packaging

`npm run pack:check` builds a fresh production `dist/`, runs `npm pack --dry-run`,
and verifies that:

- declared OpenClaw source and runtime entrypoints are present
- production source files needed by `index.ts` are present
- compiled tests and `.test-dist/` files are absent
- packed `dist/**` matches the fresh production build

The package is currently marked `private` while the in-tree integration and
distribution path are finalized.

## Hook Type Surface

OpenClaw's public `api.on(...)` typing infers hook event and context types at hook
registration sites. The concrete hook payload/context types are not directly
exported through a public package subpath in `openclaw@2026.5.6`, so
`src/openclaw-hook-types.ts` keeps narrow structural aliases for backend method
boundaries. Those aliases should be replaced with package imports if OpenClaw
publishes hook contract types through a public subpath.
