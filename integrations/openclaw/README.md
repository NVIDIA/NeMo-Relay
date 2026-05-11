<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow OpenClaw Observability

This package provides the `nemo-flow` OpenClaw plugin. It converts supported
OpenClaw hook events into NeMo Flow sessions, LLM spans, tool spans, lifecycle
marks, ATIF JSON, OpenTelemetry spans, and OpenInference/Phoenix spans.

For user-facing setup, configuration, and troubleshooting, see the
[OpenClaw Plugin Guide](../../docs/integrate-frameworks/openclaw-plugin.md).

The package declares both OpenClaw entrypoint styles:

- `openclaw.extensions`: `./index.ts` for source-based plugin workflows.
- `openclaw.runtimeExtensions`: `./dist/index.js` for built runtime workflows.

## Build and Validate

Run these commands from the repository root:

```bash
npm ci --ignore-scripts
npm run build --workspace=nemo-flow-openclaw
npm run typecheck --workspace=nemo-flow-openclaw
npm test --workspace=nemo-flow-openclaw
```

The CI-equivalent repo recipe is `just --set ci true test-openclaw`.

Optional package payload check:

```bash
npm run pack:check --workspace=nemo-flow-openclaw
```

`npm run build --workspace=nemo-flow-openclaw` emits production files under
`integrations/openclaw/dist/`. Tests compile to `integrations/openclaw/.test-dist/`
so test artifacts do not enter the installable package.

The optional live smoke test requires a working installed `nemo-flow-node`
binding:

```bash
npm run test:live --workspace=nemo-flow-openclaw
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
until explicitly configured. The snippets below are values for
`plugins.entries["nemo-flow"].config`.

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

## Hook Mapping

| OpenClaw hook | NeMo Flow behavior |
| --- | --- |
| `gateway_start` | Touches the replay backend early; session roots still open lazily from session-scoped hooks. |
| `gateway_stop` | Drains open sessions, shuts down subscribers, and clears the NeMo Flow plugin host. |
| `session_start` | Opens or aliases a NeMo Flow session scope. |
| `session_end` | Closes the session, flushes pending replay state, and exports ATIF if enabled. |
| `model_call_started` / `model_call_ended` | Records provider timing for later LLM span correlation. |
| `llm_input` / `llm_output` | Replays direct LLM spans when request and response hooks can be paired safely. |
| `before_message_write` | Records assistant turns for ordered LLM replay when provider timing can be paired later. |
| `after_tool_call` | Replays successful tool calls as tool spans; blocked tools emit marks. |
| `agent_end` | Emits an agent lifecycle mark, flushes recorded assistant-turn LLM spans, and preserves the final assistant answer as the session output. |
| `before_agent_finalize` | Preserves the last assistant message as fallback session output and emits a lifecycle mark without mutating the finalization payload. |
| `subagent_spawned` / `subagent_ended` | Emits subagent lifecycle marks under the best available parent or child session. |

For LLM spans, OpenClaw currently exposes request, response, message-write, and
provider-timing details through separate hook events. The plugin correlates those
events within the same session, provider, model, and run. When model timing cannot
be safely paired with an assistant turn, the plugin emits diagnostic marks instead
of inventing latency.

The OpenClaw hook payload and context types used by this plugin are represented
by narrow structural aliases in `src/openclaw-hook-types.ts`. Replace those
aliases with package imports if OpenClaw publishes the relevant hook contract
types through a stable public subpath.

## Health

The plugin registers the `operator.admin` scoped gateway method
`nemoFlow.status`.

The response reports:

- backend status: `not_initialized`, `disabled`, `ready`, `degraded`, `stopping`, or `stopped`
- output health for `atif`, `otel`, and `openInference`
- replay counters, including replayed LLM spans, replayed tool spans, emitted
  marks, ATIF files written, replay errors, and skipped events
- last degraded or unavailable reason when present

Use the output health independently:

- ATIF: confirm JSON files appear in the configured `atif.outputDir`.
- OTel: confirm spans arrive at the configured OTLP collector.
- OpenInference: confirm spans arrive at the configured OpenInference/Phoenix endpoint.

## Packaging

`npm run pack:check --workspace=nemo-flow-openclaw` builds a fresh production
`dist/`, runs `npm pack --dry-run`, and verifies that:

- declared OpenClaw source and runtime entrypoints are present
- production source files needed by `index.ts` are present
- compiled tests and `.test-dist/` files are absent
- packed `dist/**` matches the fresh production build
