<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-relay-deepseek-harness

`nemo-relay-deepseek-harness` is the NeMo Relay plugin bundle for DeepSeek
Harness. DeepSeek Harness gives plugins its real model stream and its real tool
execution, so this plugin runs both through NeMo Relay's managed execution
helpers: guardrails, sanitizers, intercepts, provider codecs, and lifecycle
events behave as they do for any managed Relay call. The generic NeMo Relay
plugin host activates the same `plugins.toml` components as every other Relay
host.

## What You Get

One bundle, no configuration required:

- **Managed execution.** Every model stream (`llm/stream`) and tool call
  (`tools/execute`), including subagents, runs through Relay middleware. Model
  calls are normalized with the OpenAI Chat Completions codec.
- **Guardrail rules you edit in the app.** Rules are a Relay tool
  conditional-execution guardrail. Edit, test, and save them in the
  **NeMo Relay** tab; they apply to the next tool call. The shipped rule blocks
  recursive deletion of `/`, `~`, or `$HOME` in the `bash` and `pwsh` tools.
- **Secret redaction.** Relay's built-in `pii_redaction` component scrubs API
  keys, bearer tokens, JWTs, AWS access key IDs, and Google Cloud API keys
  before any subscriber or exporter sees an event.
- **Traces and trajectories.** One `Custom` turn scope per DeepSeek Harness
  turn (`nemo_relay_scope_role: "turn"`), per-turn ATIF trajectories, and a
  per-session ATOF trace.
- **Phoenix in one click.** Start or reuse a local
  [Arize Phoenix](https://github.com/Arize-ai/phoenix) and send OpenInference
  spans through Relay's own exporter, one Phoenix session per conversation.
- **Agent-driven setup.** A `nemo_relay` tool and a `nemo-relay` skill let you
  ask the DeepSeek Harness agent to add rules, explain a block, or open the
  session in Phoenix. The agent tightens protections on its own; anything that
  loosens them, or starts or stops tracing to Phoenix, waits for your approval.
- **A NeMo Relay conversation view** and a `/relay` command for status, rules,
  and Phoenix.

Nothing leaves the machine unless you or an operator configure an exporter.

## Installation

Open the **Plugins** page in the DeepSeek Harness sidebar, install a bundle, and
enter `nemo-relay-deepseek-harness`, or run:

```bash
dsh plugin --profile <profile> add nemo-relay-deepseek-harness
```

To install from a source checkout, build the Node.js binding and this package
from the repository root, then install the bundle spec
`link:/absolute/path/to/NeMo-Relay/integrations/deepseek-harness`:

```bash
npm install --ignore-scripts
npm run build --workspace=nemo-relay-node
npm run build --workspace=nemo-relay-deepseek-harness
```

The bundle adds the Loader row in [`cordis.patch.yml`](cordis.patch.yml).

## Configure the Plugin

The Loader row `config` object accepts `enabled`, `dataDir`, `view`, `capture`,
`redaction`, `guardrails`, `phoenix`, and `plugins`. The plugin exports a
`Config` schema, so DeepSeek Harness validates the row. `plugins` is a generic
NeMo Relay plugin document whose keys use `snake_case`; Relay merges the
discovered `plugins.toml` layers first and then applies it. Refer to the
[DeepSeek Harness Plugin Guide](https://docs.nvidia.com/nemo/relay/supported-integrations/deepseek-harness)
for every field and default.

## Commands

| Command | Result |
| --- | --- |
| `/relay`, `/relay status` | Runtime state, merged `plugins.toml` layers, active exporters, redaction, rules, Phoenix, and counters. |
| `/relay json` | The same snapshot as JSON; each DeepSeek Harness process also writes it under `<dataDir>/status`. |
| `/relay rules` | The current guardrail rules. |
| `/relay rules test <json>` | Evaluates a sample tool call against the rules without running it. |
| `/relay phoenix [start\|stop\|connect\|disconnect]` | Shows or changes the Phoenix connection. |

## Middleware Behavior

| Relay middleware | Behavior in DeepSeek Harness |
| --- | --- |
| Conditional-execution guardrails | A rejected model call ends the step with an error that names the policy reason. A rejected tool call returns a failed tool result to the model. An approval prompt for a rejected call is denied before the user is asked. |
| Sanitize guardrails and PII redaction | Change only what telemetry records. |
| Stream intercepts | Rewrite or drop streamed deltas; DeepSeek Harness records the modified stream. |
| Execution intercepts | Can answer a model or tool call without running it. |
| Tool response intercepts | Change the content or value the model receives, through `tools/post-execute`. |
| Request intercepts | Edits to model-visible fields block the call. Other added fields are not sent and are recorded in a `deepseek-harness.request_edit_dropped` mark. |
| Tool request intercepts | Argument rewrites block the call. |

DeepSeek Harness sends model requests and runs tools exactly as its session log
records them, so the plugin refuses rewrites rather than silently diverging
from the log. Request headers that Relay adds are not forwarded to providers.

## Development

```bash
npm run typecheck --workspace=nemo-relay-deepseek-harness
npm test --workspace=nemo-relay-deepseek-harness
npm run test:live --workspace=nemo-relay-deepseek-harness
npm run pack:check --workspace=nemo-relay-deepseek-harness
```

Unit tests run against recording fakes of the binding. The live smoke test uses
the real binding and plugin host and is skipped unless
`NEMO_RELAY_DEEPSEEK_HARNESS_LIVE_SMOKE=1`, which `test:live` sets.
