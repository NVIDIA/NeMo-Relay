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

## What You Build

You will configure stock OpenCode to load the NeMo Flow plugin in the
background. After that, you can use OpenCode normally through the interactive
interface or `opencode run`. The plugin observes OpenCode hooks and writes
NeMo Flow ATOF and ATIF files under the OpenCode project directory.

```{mermaid}
flowchart LR
    User[Developer]
    OpenCode[Stock OpenCode]
    Plugin[NeMo Flow<br/>OpenCode plugin]
    Runtime[NeMo Flow<br/>Node.js binding]
    ATOF[(ATOF JSONL)]
    ATIF[(ATIF JSON)]

    User -->|uses normally| OpenCode
    OpenCode -->|public plugin hooks| Plugin
    Plugin -->|scopes, marks,<br/>tool lifecycle| Runtime
    Runtime -->|append events| ATOF
    Runtime -->|export on idle<br/>or deleted session| ATIF

    class User blue-lightest;
    class OpenCode green-lightest;
    class Plugin purple-lightest;
    class Runtime green-light;
    class ATOF yellow-lightest;
    class ATIF yellow-lightest;
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

When the plugin package is published, use
`@nvidia/nemoflow-opencode-plugin` in the OpenCode config instead of the local
file URL.

## Configure OpenCode

Create or update `opencode.json` in the OpenCode project directory:

```json
{
  "plugin": [
    [
      "nemo-flow-opencode",
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

The paths are resolved relative to the OpenCode project directory. If
`nemo-flow-node` is missing or cannot initialize, the plugin logs one warning
and returns no hooks, so OpenCode continues in pass-through mode.

## Run the Demo

Use this demo when you want to show the integration end to end. It uses a
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
      "file://$NEMO_FLOW_REPO/integrations/opencode-plugin",
      {
        "enabled": true,
        "atofPath": "./.nemoflow/opencode.atof.jsonl",
        "atifPath": "./.nemoflow/opencode.atif.json",
        "logPath": "./.nemoflow/opencode-plugin.log"
      }
    ]
  ]
}
JSON
```

Check that stock OpenCode sees the plugin:

```bash
opencode debug info | tee ./.nemoflow/debug-info.txt
grep "integrations/opencode-plugin" ./.nemoflow/debug-info.txt
```

Run OpenCode normally. For a repeatable demo, use `opencode run`:

```bash
opencode run \
  --title "nemo-flow opencode smoke" \
  --dangerously-skip-permissions \
  "Create phase1-demo.txt with one line: hello from NeMo Flow OpenCode. Then read it back."
```

For an interactive demo, start OpenCode and use it as you normally would:

```bash
opencode
```

Ask OpenCode to create or read a small file so the plugin can observe a
successful tool call. When the session becomes idle or is deleted, the plugin
writes the ATIF trajectory.

## Inspect the Output

Confirm that OpenCode completed the work and that NeMo Flow output exists:

```bash
test -f phase1-demo.txt
test -s ./.nemoflow/opencode.atof.jsonl
test -s ./.nemoflow/opencode.atif.json
```

`opencode.atof.jsonl` contains raw NeMo Flow lifecycle events. `opencode.atif.json`
contains the exported session trajectory when OpenCode reports the session as
idle or deleted.

Use these checks while recording a demo or debugging a local setup:

```bash
grep -E 'opencode\.chat\.message|opencode\.llm\.request|"category":"tool"' \
  ./.nemoflow/opencode.atof.jsonl

jq '.session_id // .trajectories // .' ./.nemoflow/opencode.atif.json

tail -n 20 ./.nemoflow/opencode-plugin.log
```

The expected ATOF output includes:

| Signal | Where It Comes From |
|---|---|
| `opencode.chat.message` | OpenCode `chat.message` hook |
| `opencode.llm.request` | OpenCode `chat.params` hook |
| Tool start and end records | OpenCode `tool.execute.before` and `tool.execute.after` hooks |
| `opencode.session.*` marks | OpenCode session lifecycle events |
| `opencode.session.flush` | Session idle or deleted event |

## How the Background Export Works

This sequence is what you should see in a successful run.

```{mermaid}
sequenceDiagram
    participant Dev as Developer
    participant OC as OpenCode
    participant Plug as NeMo Flow plugin
    participant NF as NeMo Flow runtime
    participant Files as .nemoflow files

    Dev->>OC: Start OpenCode in a project
    OC->>Plug: config(input)
    Plug->>Files: Write plugin diagnostics
    Dev->>OC: Send a prompt or run a task
    OC->>Plug: chat.message and chat.params
    Plug->>NF: Emit session and LLM request marks
    NF->>Files: Append ATOF JSONL
    OC->>Plug: tool.execute.before and after
    Plug->>NF: Open and close tool lifecycle records
    NF->>Files: Append ATOF JSONL
    OC->>Plug: session.status idle or session.deleted
    Plug->>NF: Flush session trajectory
    NF->>Files: Write ATIF JSON
```

## Pass-Through Checks

The plugin should not change OpenCode behavior when observability is disabled
or when the NeMo Flow runtime is unavailable.

Disable the plugin:

```bash
cp opencode.json opencode.enabled.json
jq '(.plugin[0][1].enabled) = false' opencode.json > opencode.disabled.json
mv opencode.disabled.json opencode.json
rm -f ./.nemoflow/opencode.*

opencode run --title "nemo-flow disabled smoke" \
  "Reply with exactly: plugin disabled smoke."

test ! -s ./.nemoflow/opencode.atof.jsonl
test ! -s ./.nemoflow/opencode.atif.json
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

## Demo Video Script

Use this storyboard to record a short walkthrough.

| Shot | Show | Narration |
|---|---|---|
| 1 | `opencode.json` with the plugin file URL | Stock OpenCode loads the NeMo Flow plugin through normal plugin config. |
| 2 | `opencode debug info` output | OpenCode sees the plugin without applying an OpenCode patch. |
| 3 | `opencode run` or the interactive OpenCode UI | The developer uses OpenCode normally. |
| 4 | `ls -la .nemoflow` | The plugin writes observability files in the background. |
| 5 | `grep` against `opencode.atof.jsonl` | ATOF contains session, message, LLM request, and tool lifecycle events. |
| 6 | `jq` against `opencode.atif.json` | ATIF contains the exported session trajectory. |
| 7 | Disabled or forced-failure smoke | OpenCode still runs when the plugin is disabled or pass-through. |

Keep the recording focused on the user-visible contract: install the plugin,
use OpenCode normally, and inspect `.nemoflow` output after the session.

## Limits

The current OpenCode plugin API is enough for passive observability. It is not
enough for NeMo Flow request intercepts, execution intercepts, conditional
blocking, or complete tool error spans because OpenCode does not yet expose
around-style LLM or tool hooks. Future work should add generic OpenCode plugin
hooks upstream before enabling those behaviors.
