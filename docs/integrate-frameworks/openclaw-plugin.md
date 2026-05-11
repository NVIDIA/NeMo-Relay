<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenClaw Plugin Guide

Use the OpenClaw plugin when OpenClaw owns the agent, tool, and LLM lifecycle
that needs NeMo Flow observability. The plugin observes supported OpenClaw
plugin hooks and converts them into NeMo Flow sessions, LLM spans, tool spans,
marks, ATIF JSON, OpenTelemetry spans, and OpenInference/Phoenix spans.

The plugin lives under `integrations/openclaw/` and declares both OpenClaw
entrypoint styles:

- `openclaw.extensions`: `./index.ts` for source-based plugin workflows
- `openclaw.runtimeExtensions`: `./dist/index.js` for built runtime workflows

Use this guide when you want to enable the plugin, configure telemetry outputs,
understand the hook mapping, or troubleshoot LLM replay fidelity.

## Build And Validate

Run these commands from the repository root:

```bash
npm ci --ignore-scripts
npm run build --workspace=nemo-flow-openclaw
npm run typecheck --workspace=nemo-flow-openclaw
npm test --workspace=nemo-flow-openclaw
```

The CI-equivalent repository recipe is:

```bash
just --set ci true test-openclaw
```

Use the optional package payload check before changing package metadata or
entrypoints:

```bash
npm run pack:check --workspace=nemo-flow-openclaw
```

The build emits production files under `integrations/openclaw/dist/`. Test
build artifacts are written under `integrations/openclaw/.test-dist/` and are
excluded from the installable package.

## Enable The Plugin

Allow the `nemo-flow` plugin id and grant conversation hook access when OpenClaw
runs with restrictive plugin settings:

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

`plugins.allow` controls plugin trust and loading.
`hooks.allowConversationAccess` lets trusted non-bundled plugins receive
conversation-sensitive hook payloads such as LLM prompts, LLM responses, agent
finalization messages, and tool payloads.

## Configure Outputs

Plugin configuration is passed through
`plugins.entries["nemo-flow"].config`.

ATIF export is enabled by default. OpenTelemetry and OpenInference subscribers
are disabled until explicitly configured.

For ATIF-only local export:

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

For OpenTelemetry OTLP export:

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

For OpenInference/Phoenix OTLP export:

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

Privacy defaults capture prompts and responses, and strip tool arguments and
tool results:

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

Tool arguments and tool results are stripped by default because they often
contain user data, local paths, tokens, or large payloads.

Correlation uses bounded in-memory records. By default, the plugin waits 250 ms
for a matching `llm_input` after an `llm_output`, keeps correlation records for
600 seconds, and keeps at most 32 records per correlation key.

## Runtime Mapping

The plugin maps supported OpenClaw hook events into NeMo Flow telemetry without
changing OpenClaw execution behavior.

| OpenClaw Hook | NeMo Flow Behavior |
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

## LLM Replay Fidelity

OpenClaw currently exposes request, response, message-write, and provider-timing
details through separate hook events. The plugin correlates those events within
the same session, provider, model, and run.

When model timing cannot be safely paired with an assistant turn, the plugin
emits diagnostic marks instead of inventing latency. This keeps traces honest
and makes current fidelity boundaries explicit.

When OpenClaw provides usage data, the plugin maps input, output, total, cache
read, cache write, and cost fields into OpenInference-friendly usage fields.

## Health

The plugin registers the `operator.admin` scoped gateway method
`nemoFlow.status`.

The response reports:

- backend status: `not_initialized`, `disabled`, `ready`, `degraded`,
  `stopping`, or `stopped`
- output health for `atif`, `otel`, and `openInference`
- replay counters for replayed LLM spans, replayed tool spans, emitted marks,
  ATIF files written, replay errors, and skipped events
- last degraded or unavailable reason when present

Use output health to verify each configured sink:

- ATIF: confirm JSON files appear in the configured `atif.outputDir`
- OTel: confirm spans arrive at the configured OTLP collector
- OpenInference: confirm spans arrive at the configured OpenInference/Phoenix
  endpoint

## Verify The Integration

Use this verification flow after enabling the plugin:

1. Build the plugin.
2. Run `just --set ci true test-openclaw`.
3. Start an OpenClaw session with the `nemo-flow` plugin enabled.
4. Confirm ATIF files or OpenInference/Phoenix spans are produced, depending on
   the configured outputs.
5. Check `nemoFlow.status` for backend state, output health, replay counters,
   and degraded-output reasons.

The optional live smoke test requires a working installed `nemo-flow-node`
binding:

```bash
npm run test:live --workspace=nemo-flow-openclaw
```

## Troubleshooting

If the plugin does not load:

- verify `plugins.allow` includes `nemo-flow`
- verify the source or built OpenClaw entrypoint exists
- verify `plugins.entries["nemo-flow"].enabled` is not disabled

If conversation payloads are missing:

- verify `hooks.allowConversationAccess` is enabled for the plugin
- verify the OpenClaw session emits the relevant LLM, message-write, and tool
  hooks

If tool spans exist but LLM spans are incomplete:

- verify `llm_input` and `llm_output` hooks are emitted
- verify `before_message_write` hooks are emitted when relying on assistant-turn
  replay
- verify `model_call_started` and `model_call_ended` hooks are emitted when
  timing attribution is expected
- check diagnostic marks for ambiguous or unpaired timing records

If no export output appears:

- verify `atif.outputDir`, `telemetry.otel.endpoint`, or
  `telemetry.openInference.endpoint`
- verify the configured collector or output directory is reachable
- verify session end or gateway stop hooks fired so pending replay state can
  drain

If ambiguous timing marks appear, treat them as expected conservative behavior.
The plugin avoids attaching unsafe latency when multiple timing candidates could
match the same assistant turn.

## Known Limitations

Current OpenClaw public hooks are separate event streams, so some LLM timing
attribution is best-effort. If a matching request hook is missing, the plugin
may replay an LLM output with a placeholder request after the configured grace
window. If timing is ambiguous, the plugin emits diagnostic marks instead of
unsafe latency.
