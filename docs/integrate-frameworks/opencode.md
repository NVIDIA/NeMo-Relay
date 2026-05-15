<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenCode Plugin

NeMo Flow integrates with OpenCode through a standalone server plugin. The
plugin uses OpenCode's public plugin hooks and does not require a patched
OpenCode checkout.

Use this plugin when you want NeMo Flow observability for OpenCode sessions,
messages, LLM request metadata, successful tool calls, and session errors.
The OpenCode plugin maps those hook payloads into NeMo Flow scopes and events.
The generic NeMo Flow `observability` plugin config controls ATOF, ATIF,
OpenTelemetry, and OpenInference export.

## What You Build

You will configure stock OpenCode to load the NeMo Flow plugin in the
background. After that, you can use OpenCode normally through the interactive
interface or `opencode run`.

The diagram shows the split between OpenCode hook capture and generic NeMo Flow
export configuration.

```{mermaid}
flowchart LR
    User[Developer]
    OpenCode[Stock OpenCode]
    Plugin[NeMo Flow<br/>OpenCode plugin]
    Runtime[NeMo Flow<br/>Node.js binding]
    Host[Generic NeMo Flow<br/>plugin host]
    ATOF[(ATOF JSONL)]
    ATIF[(ATIF JSON)]
    OTLP[(OTLP traces)]

    User -->|uses normally| OpenCode
    OpenCode -->|public plugin hooks| Plugin
    Plugin -->|scopes, marks,<br/>tool lifecycle| Runtime
    Plugin -->|plugins config| Host
    Host -->|observability component| Runtime
    Runtime -->|raw events| ATOF
    Runtime -->|agent trajectories| ATIF
    Runtime -->|optional traces| OTLP

    class User blue-lightest;
    class OpenCode green-lightest;
    class Plugin purple-lightest;
    class Host purple-light;
    class Runtime green-light;
    class ATOF yellow-lightest;
    class ATIF yellow-lightest;
    class OTLP yellow-lightest;
```

The plugin is passive. It records observability output but does not rewrite
prompts, tool arguments, model requests, or OpenCode execution behavior.

## Install

Build the NeMo Flow Node.js binding before loading the plugin from a source
checkout. `crates/node` is under the NeMo Flow repository root:

```bash
export NEMO_FLOW_REPO=/absolute/path/to/NeMo-Flow
cd "$NEMO_FLOW_REPO/crates/node"
npm install
npm run build
```

For local development, install or use stock OpenCode and point `opencode.json`
at the plugin directory:

```bash
npm install -g opencode-ai@latest
opencode --version
```

When the plugin package is published, use `nemo-flow-opencode` in the OpenCode
config instead of the local file URL.

## Configure OpenCode

Create or update `opencode.json` in the OpenCode project directory. The
top-level OpenCode plugin fields use JavaScript-style names such as `logPath`.
The nested `plugins` object is the generic NeMo Flow plugin config, so its
field names are `snake_case` in every language.

```json
{
  "plugin": [
    [
      "file:///absolute/path/to/NeMo-Flow/integrations/opencode",
      {
        "enabled": true,
        "logPath": "./.nemoflow/opencode-plugin.log",
        "plugins": {
          "version": 1,
          "components": [
            {
              "kind": "observability",
              "enabled": true,
              "config": {
                "version": 1,
                "atof": {
                  "enabled": true,
                  "output_directory": "./.nemoflow",
                  "filename": "opencode.atof.jsonl",
                  "mode": "overwrite"
                },
                "atif": {
                  "enabled": true,
                  "agent_name": "opencode",
                  "output_directory": "./.nemoflow",
                  "filename_template": "opencode-{session_id}.atif.json"
                }
              }
            }
          ]
        }
      }
    ]
  ]
}
```

The OpenCode plugin resolves `logPath` and observability `output_directory`
values relative to the OpenCode project directory. Other generic plugin fields
keep NeMo Flow's standard behavior. If `nemo-flow-node` is missing, plugin
validation fails, or plugin initialization fails, the plugin logs one warning
and returns no hooks, so OpenCode continues in pass-through mode.

The ATIF filename placeholder `{session_id}` is the NeMo Flow top-level agent
scope UUID. The OpenCode session ID is still recorded in event metadata.

For the complete `observability` component schema, see
[Configure the Observability Plugin](../export-observability-data/observability-plugin.md).

## Run A Local Smoke

Use this smoke when you want to check the integration end to end. It uses a
source checkout plugin path because the package is not published yet.

```bash
export NEMO_FLOW_REPO=/absolute/path/to/NeMo-Flow
export NEMO_FLOW_DEMO_DIR="$NEMO_FLOW_REPO/tmp/opencode-nemoflow-demo"

rm -rf "$NEMO_FLOW_DEMO_DIR"
mkdir -p "$NEMO_FLOW_DEMO_DIR/.nemoflow"
cd "$NEMO_FLOW_DEMO_DIR"

cat > opencode.json <<JSON
{
  "plugin": [
    [
      "file://$NEMO_FLOW_REPO/integrations/opencode",
      {
        "enabled": true,
        "logPath": "./.nemoflow/opencode-plugin.log",
        "plugins": {
          "version": 1,
          "components": [
            {
              "kind": "observability",
              "enabled": true,
              "config": {
                "version": 1,
                "atof": {
                  "enabled": true,
                  "output_directory": "./.nemoflow",
                  "filename": "opencode.atof.jsonl",
                  "mode": "overwrite"
                },
                "atif": {
                  "enabled": true,
                  "agent_name": "opencode",
                  "output_directory": "./.nemoflow",
                  "filename_template": "opencode-{session_id}.atif.json"
                }
              }
            }
          ]
        }
      }
    ]
  ]
}
JSON
```

Check that stock OpenCode sees the plugin:

```bash
opencode debug info | tee ./.nemoflow/debug-info.txt
grep "integrations/opencode" ./.nemoflow/debug-info.txt
```

Run OpenCode normally. For a repeatable smoke, use `opencode run`:

```bash
opencode run \
  --title "nemo-flow opencode smoke" \
  --dangerously-skip-permissions \
  "Create phase1-demo.txt with one line: hello from NeMo Flow OpenCode. Then read it back."
```

For an interactive run, start OpenCode and use it as you normally would:

```bash
opencode
```

Ask OpenCode to create or read a small file so the plugin can observe a
successful tool call. When the session becomes idle or is deleted, the generic
observability component writes the ATIF trajectory.

## Inspect The Output

Confirm that OpenCode completed the work and that NeMo Flow output exists:

```bash
test -f phase1-demo.txt
test -s ./.nemoflow/opencode.atof.jsonl
ls ./.nemoflow/opencode-*.atif.json
```

`opencode.atof.jsonl` contains raw NeMo Flow lifecycle events. Each
`opencode-*.atif.json` file contains one exported top-level agent trajectory.

Use these checks while debugging a local setup:

```bash
grep -E 'opencode\.chat\.message|opencode\.llm\.request|"category":"tool"' \
  ./.nemoflow/opencode.atof.jsonl

jq '.session_id // .trajectories // .' ./.nemoflow/opencode-*.atif.json

tail -n 20 ./.nemoflow/opencode-plugin.log
```

The expected ATOF output includes:

| Signal | Where It Comes From |
|---|---|
| `opencode.chat.message` | OpenCode `chat.message` hook |
| `opencode.llm.request` | OpenCode `chat.params` hook |
| Tool start and end records | OpenCode `tool.execute.before` and `tool.execute.after` hooks |
| `opencode.session.*` marks | OpenCode session lifecycle events |
| `opencode.session.flush` | Session idle, deleted, or plugin cleanup |

## How The Background Export Works

This sequence is what you should see in a successful run.

```{mermaid}
sequenceDiagram
    participant Dev as Developer
    participant OC as OpenCode
    participant Plug as NeMo Flow plugin
    participant Host as NeMo Flow plugin host
    participant NF as NeMo Flow runtime
    participant Files as .nemoflow files

    Dev->>OC: Start OpenCode in a project
    OC->>Plug: server(input, options)
    Plug->>Host: Validate and initialize plugins config
    Dev->>OC: Send a prompt or run a task
    OC->>Plug: chat.message and chat.params
    Plug->>NF: Emit session and LLM request marks
    NF->>Files: Append ATOF JSONL
    OC->>Plug: tool.execute.before and after
    Plug->>NF: Open and close tool lifecycle records
    NF->>Files: Append ATOF JSONL
    OC->>Plug: session.status idle or session.deleted
    Plug->>NF: Close session scope
    NF->>Files: Write ATIF JSON
```

## Pass-Through Checks

The plugin should not change OpenCode behavior when observability is disabled,
when the NeMo Flow runtime is unavailable, or when the generic plugin config is
invalid.

Disable the plugin:

```bash
cp opencode.json opencode.enabled.json
jq '(.plugin[0][1].enabled) = false' opencode.json > opencode.disabled.json
mv opencode.disabled.json opencode.json
rm -f ./.nemoflow/opencode.*

opencode run --title "nemo-flow disabled smoke" \
  "Reply with exactly: plugin disabled smoke."

test ! -s ./.nemoflow/opencode.atof.jsonl
test -z "$(find ./.nemoflow -name 'opencode-*.atif.json' -print -quit)"
mv opencode.enabled.json opencode.json
```

Force runtime initialization failure:

```bash
rm -f ./.nemoflow/opencode.*

NEMO_FLOW_OPENCODE_FORCE_INIT_FAILURE=1 opencode run \
  --title "nemo-flow init failure smoke" \
  "Reply with exactly: init failure smoke."

grep -i "pass-through" ./.nemoflow/opencode-plugin.log
test ! -s ./.nemoflow/opencode.atof.jsonl
```

## Limits

The current OpenCode plugin API is enough for passive observability. It is not
enough for NeMo Flow request intercepts, execution intercepts, conditional
blocking, or complete tool error spans because OpenCode does not yet expose
around-style LLM or tool hooks. Future work should add generic OpenCode plugin
hooks upstream before enabling those behaviors.
