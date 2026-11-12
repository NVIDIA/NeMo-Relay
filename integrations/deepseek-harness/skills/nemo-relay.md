<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay

NeMo Relay runs in this app as the `nemo-relay` plugin. Every model call and tool call, including subagents', passes through it. By default it:

- writes each session's trace (ATOF, shown in the NeMo Relay view) and per-turn ATIF trajectories under its data directory, with prompts, responses, and tool calls, and removes the ones untouched for 30 days (`retention.days`);
- scrubs credentials (API keys, bearer tokens, JWTs, AWS access key IDs, Google Cloud API keys) from everything it records or exports;
- blocks tool calls that match its guardrail rules: the call fails with `<reason> (NeMo Relay rule "<id>")`;
- sends nothing off the machine unless the user or an administrator configures an exporter; `status` lists every exporter.

Use the `nemo_relay` tool for all of the following. Do not edit Relay's files (status files, `rules.json`, traces) by hand.

## Approval

These run immediately: `status`, `rules.list`, `rules.test`, `rules.add`, `rules.enable`, `guardrails.failMode` with `closed`, `phoenix.status`, `phoenix.link`.

These first ask the user in the approval prompt and change nothing if the user declines: `rules.update`, `rules.disable`, `rules.remove`, `rules.reset`, `guardrails.failMode` with `open`, `phoenix.start`, `phoenix.connect`, `phoenix.disconnect`, `phoenix.stop`.

Weaken a protection, or stop tracing, only when the user asked for that change. Never disable, delete, or loosen a rule to get your own call through, and never rephrase a command to slip past a pattern: when a rule blocks you, tell the user which rule blocked what, and offer an alternative.

## Rules

A rule blocks a call when its tool selector matches the tool and its pattern matches the arguments; the first enabled match wins. A change applies from the next tool call.

| Field | Meaning |
| --- | --- |
| `id` | Letters, digits, `.`, `_`, `-`. Names the rule in block messages and later actions. |
| `tool` | A tool name such as `bash`; a comma list such as `bash, pwsh`; a `/regex/flags` literal; or `*` for any tool. Names match case-insensitively. The shell tools are `bash` and `pwsh`. |
| `match` | JavaScript regular expression tested against the arguments as JSON and against each top-level string argument; for `bash` that includes `command` and `description`. Empty matches every call of the selected tools. Nested repetition such as `(a+)+` is refused, and so is a pattern that is slow on ordinary text. |
| `flags` | From `imsu`; default `i`. |
| `reason` | What the agent is told when the rule blocks; say what to do instead. |

`rules.add` appends an enabled rule and refuses an id that exists. `rules.update` changes only the fields you pass. `rules.enable`, `rules.disable`, and `rules.remove` take `id`. `rules.reset` restores the shipped rules: `no-recursive-delete-of-root-or-home`, which blocks recursive deletion of `/`, `~`, or `$HOME` through `bash` or `pwsh`.

`rules.test` runs a sample call against the current rules without running it; use it to check a pattern before and after adding it. In JSON, escape each regex backslash (`\\b`).

Block force pushes but allow `--force-with-lease`:

```json
{"action": "rules.add", "id": "no-force-push", "tool": "bash", "match": "\\bgit\\s+push\\b.*(?:--force(?!-with-lease)|\\s-f\\b)", "reason": "Force-pushing is blocked by NeMo Relay policy. Use --force-with-lease, or ask the user to push."}
```

```json
{"action": "rules.test", "tool": "bash", "arguments": {"command": "git push --force origin main"}}
```

Block publishing packages:

```json
{"action": "rules.add", "id": "no-npm-publish", "tool": "bash", "match": "\\bnpm\\s+publish\\b", "reason": "Publishing packages is blocked by NeMo Relay policy; ask the user to publish."}
```

`guardrails.failMode` decides what happens when a rule cannot be checked, because it ran out of time or the arguments are longer than 2,000,000 characters: `open`, the default, allows the call; `closed` blocks it.

## Phoenix

Arize Phoenix shows this app's traces, one Phoenix session per conversation.

- `phoenix.status`: whether Phoenix runs and receives traces.
- `phoenix.start`: uses a Phoenix that answers at the configured URL (default `http://127.0.0.1:6006`) or launches one on this machine with `uv`, then sends traces to it. The first launch downloads Phoenix and can take minutes.
- `phoenix.connect`: sends traces to a Phoenix that already answers, without launching one.
- `phoenix.link`: this conversation's page in Phoenix; give it to the user.
- `phoenix.disconnect` stops sending traces; `phoenix.stop` also stops a Phoenix this app launched. A Phoenix started elsewhere keeps running.

When a Phoenix action fails, report its error. A launch needs `uv` installed.

## Exporters and other settings

`status` lists the active exporters, the redaction detectors, and the `plugins.toml` layers Relay merged, including system layers the user did not set up here. Exporters, capture, and redaction are fields of the plugin's configuration, which `nemo_relay` does not change.

To send traces to an OpenTelemetry collector, add an `observability` component under the configuration's `plugins` field. Do this only when the user asks and names the destination: the collector receives prompts, responses, and tool calls. Remote endpoints must use `https`. Put credentials in environment variables named in `header_env`, never in the configuration.

```json
{
  "plugins": {
    "version": 1,
    "components": [
      {
        "kind": "observability",
        "enabled": true,
        "config": {
          "version": 4,
          "opentelemetry": {
            "enabled": true,
            "endpoints": [
              {
                "type": "gen_ai",
                "endpoint": "https://collector.example.com/v1/traces",
                "service_name": "deepseek-harness",
                "header_env": { "authorization": "OTEL_AUTHORIZATION" }
              }
            ]
          },
          "atif": {
            "enabled": true,
            "agent_name": "deepseek-harness",
            "output_directory": "atif",
            "filename_template": "{metadata.session_id:-unassigned}/trajectory-{session_id}.json",
            "include_observed_events": false,
            "include_llm_requests": false
          }
        }
      }
    ]
  }
}
```

- `type` is `gen_ai` (OpenTelemetry GenAI conventions), `openinference` (Arize), or `full` (every Relay attribute).
- `capture` decides what traces contain: `includePrompts` and `includeResponses` (default `true`), `stripToolArgs` and `stripToolResults` (default `false`). Suggest them when the user wants less recorded.
- An `observability` component under `plugins` replaces the default per-turn ATIF output; keep the `atif` section to keep it. Relative directories resolve under the plugin's data directory.
- Settings does not edit this configuration. Show the user the block for the `config` of the `nemo-relay` entry, or, if they ask you to apply it, follow the `cordis-plugin-development` skill. A patch for the entry replaces its whole `config`, so keep its other fields, including saved `guardrails.rules`.
- The plugin restarts with the new configuration; `status` then lists the exporter.
