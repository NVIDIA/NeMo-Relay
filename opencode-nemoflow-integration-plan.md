# OpenCode <> NeMo Flow Integration Plan

Reference OpenCode checkout: `reference_projects/opencode`

Reference commit: `dcfe4b0d5184cb93dd2232f1461641d6530e1abb`

## Goal

The end goal is a proper OpenCode plugin, with no NeMo Flow-maintained patch
against OpenCode.

The immediate goal is narrower than full NeMo Flow middleware support:

1. Build a standalone OpenCode observability plugin using OpenCode's existing
   public plugin API.
2. Preserve OpenCode's in-agent session, message, model, tool, and error
   context.
3. Export NeMo Flow observability data from inside OpenCode, not from a
   sidecar/proxy-only view.

The current NeMo Flow patch should be treated as a prototype and reference
implementation only. It proves where OpenCode needs plugin extension points,
but it is not the target deliverable.

The target deliverable for the first milestone is:

1. NeMo Flow ships an OpenCode plugin that uses only OpenCode's public plugin
   API.
2. OpenCode does not contain NeMo Flow-specific code, dependencies, flags, or
   config schema.
3. Users can enable the integration through normal OpenCode plugin
   configuration, without applying patches.

Future NeMo Flow intercepts and guardrails may require new OpenCode plugin
hooks, but that is a later milestone after the observability plugin is working.

The key distinction is:

- Existing OpenCode hooks are enough for a useful observability plugin.
- Existing OpenCode hooks are not enough for full NeMo Flow middleware behavior
  such as request intercepts, execution intercepts, conditional guardrails, and
  stream intercepts.
- Switchyard or other proxy interception can observe provider traffic, but it
  does not own OpenCode's agent context and hierarchy. The OpenCode plugin
  should be the source of agent/session structure.

## Current OpenCode Plugin Surface

OpenCode server plugins are loaded from `packages/opencode/src/plugin/index.ts`.
They implement the `Hooks` interface from `packages/plugin/src/index.ts`.

Current useful hooks for the first observability milestone:

| Hook | Current behavior | Useful for NeMo Flow observability | Gap for future intercepts |
| --- | --- | --- | --- |
| `config(input)` | Notifies plugins after OpenCode config is loaded. | Initialize NeMo Flow runtime/exporters from plugin options and OpenCode config context. | Enough. |
| `event({ event })` | Receives all OpenCode bus events. | Export session lifecycle, message updates, errors, idle events, and final ATIF/ATOF output. | Enough for passive observability only. |
| `chat.message(input, output)` | Called when a user message is created. | Bind session, agent, model, user message, and turn-level metadata. | Useful, but not a full LLM execution boundary. |
| `chat.params(input, output)` | Lets plugins inspect or mutate model params before the AI SDK call. | Capture provider/model/parameter metadata and approximate LLM request start. | Too narrow for full LLM request rewrite; does not wrap execution or stream lifecycle. |
| `chat.headers(input, output)` | Lets plugins add provider request headers. | Optional correlation header injection for external tracing. | Too narrow for content/message/tool/provider rewrites. |
| `tool.execute.before(input, output)` | Called before tool execution with tool name, session, call id, args. | Start a tool span and capture sanitized input. | Cannot wrap the callback, cannot reliably see thrown errors, cannot apply NeMo Flow execution intercepts. |
| `tool.execute.after(input, output)` | Called after successful tool execution. | Finish a successful tool span and capture sanitized output. | Does not run on every error path unless OpenCode manually catches; cannot alter result. |
| `permission.ask(input, output)` | Lets plugins observe or influence permission decisions. | Potential policy correlation. | Not a tool/LLM execution boundary. |
| `command.execute.before(input, output)` | Called before slash command execution. | Optional command observability. | Not core LLM/tool middleware. |
| `shell.env(input, output)` | Lets plugins mutate shell env. | Optional shell correlation. | Not core LLM/tool middleware. |
| `experimental.chat.messages.transform(input, output)` | Mutates model message history before LLM call. | Possible request shaping. | Experimental and message-only; does not cover stream lifecycle or execution intercepts. |
| `experimental.chat.system.transform(input, output)` | Mutates system prompts. | Possible prompt shaping. | Experimental and system-only. |
| `experimental.session.compacting(input, output)` | Compaction hook. | Optional lifecycle correlation. | Not core LLM/tool middleware. |
| `experimental.compaction.autocontinue(input, output)` | Compaction continuation hook. | Optional lifecycle correlation. | Not core LLM/tool middleware. |
| `experimental.text.complete(input, output)` | Text completion hook. | Optional smaller LLM observability. | Separate from main chat stream. |
| `tool.definition` | Contributes plugin-defined tools. | Useful for plugin-provided tools. | Does not wrap built-in or MCP tools. |

## What NeMo Flow Needs

NeMo Flow has two different classes of behavior. The first milestone should
focus only on observability.

| NeMo Flow behavior | Changes real execution? | Required OpenCode support |
| --- | --- | --- |
| Subscribers/exporters | No. Observes emitted runtime events. | Existing `event`, `config`, `chat.message`, `chat.params`, and tool before/after hooks are enough for an MVP. |
| Scoped lifecycle | No direct mutation, but affects event hierarchy. | Existing OpenCode session/message/tool events are enough to build useful hierarchy. |
| Sanitize guardrails | No. Redacts observability payloads only. | Can be applied inside the plugin before exporting. |
| Request intercepts | Yes. Rewrites tool args or LLM request payload. | Later milestone. Needs OpenCode to pass rewritten request/args into the real callback. |
| Execution intercepts | Yes. Wraps, replaces, retries, caches, or short-circuits execution. | Later milestone. Needs an around hook with `next`. |
| Stream execution intercepts | Yes. Wraps async LLM streams. | Later milestone. Needs an around hook for the stream producer. |
| Conditional guardrails | Yes. Blocks execution. | Later milestone. Needs a managed boundary before the real callback. |

## Observability Plugin MVP

The first plugin should be a normal OpenCode server plugin distributed like
community plugins.

It should use only the existing OpenCode plugin API:

| OpenCode hook | NeMo Flow plugin responsibility |
| --- | --- |
| `config` | Read plugin options, initialize NeMo Flow, set exporter paths, and log disabled/misconfigured state once. |
| `event` | Build session/message/error lifecycle records and flush/export when a session becomes idle or deleted. |
| `chat.message` | Create or update a turn/session record with user message, agent, and selected model metadata. |
| `chat.params` | Capture provider/model/parameter metadata near the LLM request boundary. |
| `tool.execute.before` | Start a tool span using `sessionID`, `callID`, tool name, and sanitized args. |
| `tool.execute.after` | Finish a successful tool span with title/output/metadata. |

Expected first-milestone output:

1. Session-level ATIF/ATOF records with OpenCode session IDs.
2. Message and turn metadata with agent and model context where available.
3. Tool spans for successful builtin, MCP, plugin, and task tool calls where
   OpenCode emits before/after hooks.
4. Session error records from OpenCode events.
5. A documented limitation for exact LLM stream boundaries and tool error spans
   until OpenCode exposes around-style hooks.

Switchyard can still be useful for provider-level request/response capture, but
it should not replace the OpenCode plugin for agent context. The plugin should
own hierarchy; proxy data can be correlated later if needed.

## Version Compatibility Strategy

The NeMo Flow OpenCode plugin should follow the existing OpenCode community
plugin pattern:

1. Publish or build as a normal npm package with a server plugin entrypoint,
   for example `exports["./server"]`.
2. Depend on `@opencode-ai/plugin` and, if needed, `@opencode-ai/sdk` using a
   semver range.
3. Declare supported OpenCode versions with `engines.opencode`, for example
   `">=1.3.13"` once the minimum tested version is chosen.
4. Do not pin an OpenCode source checkout as part of the plugin runtime.
5. Use CI to test the minimum supported OpenCode version and latest stable
   OpenCode.

Pinned `reference_projects/opencode` and `third_party/opencode` checkouts are
still useful for development and regression testing, but they should not become
the plugin installation model.

## Future Intercept Hooks

After the observability plugin works, NeMo Flow can evaluate whether OpenCode
needs new generic plugin hooks for real execution intercepts.

The likely missing first-party OpenCode hooks are:

```ts
llm.stream.wrap(input, next)
tool.execute.wrap(input, next)
```

The names are placeholders. The important part is the semantics: OpenCode owns
the integration point, but the plugin can wrap the real callback and decide
whether and how to call `next`.

## Future First-Party Hook Shapes

### LLM Stream Hook

```ts
type LlmStreamWrapInput = {
  sessionID: string
  parentSessionID?: string
  agent: string
  model: Model
  provider: ProviderContext
  message: UserMessage
  request: {
    system: string[]
    messages: ModelMessage[]
    tools: Record<string, Tool>
    toolChoice?: "auto" | "required" | "none"
    params: {
      temperature?: number
      topP?: number
      topK?: number
      maxOutputTokens?: number
      options: Record<string, unknown>
    }
    headers: Record<string, string>
  }
}

type LlmStreamWrapNext = (
  input: LlmStreamWrapInput,
) => AsyncIterable<LLM.Event> | Promise<AsyncIterable<LLM.Event>>

type LlmStreamWrapHook = (
  input: LlmStreamWrapInput,
  next: LlmStreamWrapNext,
) => AsyncIterable<LLM.Event> | Promise<AsyncIterable<LLM.Event>>
```

Expected semantics:

1. OpenCode builds the final request object before calling the AI SDK.
2. OpenCode calls plugins as a nested chain.
3. The NeMo Flow plugin serializes the request, runs NeMo Flow
   `llm_stream_execute`, applies request intercept results, then calls `next`.
4. OpenCode sends the final rewritten request to the provider.
5. Chunks flow back through the wrapper so NeMo Flow can emit stream start,
   chunk, end, and error events.

### Tool Execute Hook

```ts
type ToolExecuteWrapInput = {
  tool: string
  sessionID: string
  callID: string
  args: unknown
  source: "builtin" | "mcp" | "task" | "plugin"
}

type ToolExecuteWrapNext = (input: ToolExecuteWrapInput) => Promise<unknown>

type ToolExecuteWrapHook = (
  input: ToolExecuteWrapInput,
  next: ToolExecuteWrapNext,
) => Promise<unknown>
```

Expected semantics:

1. OpenCode validates tool input and constructs the tool execution context.
2. OpenCode calls the wrapper chain.
3. The NeMo Flow plugin runs `tool_call_execute`.
4. NeMo Flow request intercepts can rewrite `input.args`.
5. NeMo Flow execution intercepts can call `next`, skip `next`, retry, cache, or
   replace the result.
6. OpenCode receives the final result and continues its normal message update
   path.

## Hook Composition

OpenCode currently runs hooks sequentially in plugin load order. Around hooks
should preserve that simplicity:

```ts
function composeWrapHooks(hooks, finalNext) {
  return hooks.reduceRight(
    (next, hook) => (input) => hook(input, next),
    finalNext,
  )
}
```

No priority field is required for the first version. Load order is enough and
matches the rest of the plugin system. A priority field can be added later only
if OpenCode wants deterministic ordering independent of config order.

## Implementation Breakdown

### Phase 0: Use The Existing Patch As Reference Only

The current NeMo Flow patch is useful for learning and validation, but it
should not be the production integration strategy.

Use it to answer these questions:

1. Which OpenCode events are needed for session/message hierarchy?
2. Which existing hooks provide agent/model/tool metadata?
3. Which ATOF/ATIF output can be produced without patching OpenCode?
4. Which exact behaviors remain impossible without around-style hooks?

Do not add more NeMo Flow-specific code to OpenCode as the long-term path.

### Shared Smoke Test Setup

Use this setup for every phase demo. The goal is to show the integration as an
end user would run it, not as a patched OpenCode checkout.

Assumptions:

1. Node.js 20 or newer is installed.
2. OpenCode can reach at least one configured model provider.
3. `jq` is installed for checking JSON demo output.
4. The NeMo Flow OpenCode plugin package has been built or published.
5. The final package name is still open. The examples below use
   `@nvidia/nemoflow-opencode-plugin` as a placeholder.

Common commands:

```bash
git clone https://github.com/NVIDIA/NeMo-Flow.git
cd NeMo-Flow

npm install -g opencode-ai@latest
opencode --version

export NEMO_FLOW_OPENCODE_PLUGIN="@nvidia/nemoflow-opencode-plugin"
export NEMO_FLOW_DEMO_DIR="$PWD/tmp/opencode-nemoflow-demo"

rm -rf "$NEMO_FLOW_DEMO_DIR"
mkdir -p "$NEMO_FLOW_DEMO_DIR/.nemoflow"
cd "$NEMO_FLOW_DEMO_DIR"

cat > opencode.json <<JSON
{
  "plugin": [
    [
      "$NEMO_FLOW_OPENCODE_PLUGIN",
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

If the plugin is not published yet, replace the package name with a local file
plugin path:

```json
{
  "plugin": [
    [
      "file:///absolute/path/to/nemoflow-opencode-plugin",
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

Before each phase demo, reset the output directory:

```bash
rm -f ./.nemoflow/opencode.atof.jsonl \
  ./.nemoflow/opencode.atif.json \
  ./.nemoflow/opencode-plugin.log
```

### Phase 1: Build The Standalone Observability Plugin

Build a normal OpenCode plugin first. This should not require an OpenCode patch
or upstream OpenCode changes.

1. Create an OpenCode server plugin package for NeMo Flow.
2. Add plugin options for enabling/disabling export and configuring output
   paths.
3. Use `config`, `event`, `chat.message`, `chat.params`,
   `tool.execute.before`, and `tool.execute.after`.
4. Map OpenCode session/message/tool lifecycle data into NeMo Flow
   observability records.
5. Keep `nemo-flow-node` as a plugin dependency, not an OpenCode dependency.
6. If NeMo Flow cannot initialize, log once and make the plugin pass-through.
7. Document current limitations around exact stream lifecycle and tool error
   spans.

#### Smoke Test Guide

This demo proves the first promised feature: NeMo Flow runs as a normal
OpenCode plugin and emits useful observability output without applying an
OpenCode patch.

Run:

```bash
cd "$NEMO_FLOW_DEMO_DIR"

opencode debug info | tee ./.nemoflow/debug-info.txt

opencode run \
  --title "nemo-flow phase 1 smoke" \
  --dangerously-skip-permissions \
  "Create a file named phase1-demo.txt with one line: hello from NeMo Flow OpenCode."
```

Check:

```bash
grep "$NEMO_FLOW_OPENCODE_PLUGIN" ./.nemoflow/debug-info.txt
test -s ./.nemoflow/opencode.atof.jsonl
test -s ./.nemoflow/opencode.atif.json

tail -n 5 ./.nemoflow/opencode.atof.jsonl
jq '.sessions // .trajectories // .' ./.nemoflow/opencode.atif.json
```

Expected demo evidence:

1. `opencode debug info` lists the NeMo Flow plugin.
2. `phase1-demo.txt` exists, proving the run used normal OpenCode.
3. `.nemoflow/opencode.atof.jsonl` contains session/message/tool events.
4. `.nemoflow/opencode.atif.json` contains a session or trajectory tied to the
   OpenCode session ID.
5. No command applies `patches/opencode`.

### Phase 2: Validate Observability Behavior

Add or run tests for:

1. Plugin disabled: OpenCode behavior is unchanged.
2. NeMo Flow runtime unavailable: OpenCode behavior is unchanged and the plugin
   logs once.
3. Session created/updated/idle/deleted events produce coherent session output.
4. User message events carry session, agent, and model metadata where OpenCode
   provides it.
5. Tool before/after hooks produce successful tool spans.
6. Session errors are exported from OpenCode events.
7. ATIF/ATOF output is written when a session becomes idle or deleted.
8. The package loads through normal OpenCode plugin config.
9. Compatibility is checked against the minimum supported OpenCode version and
   latest stable OpenCode.

#### Smoke Test Guide

This demo proves the second promised feature: the plugin handles normal,
disabled, and failure-tolerant observability paths.

Run the positive path:

```bash
cd "$NEMO_FLOW_DEMO_DIR"
rm -f ./.nemoflow/*

opencode run \
  --title "nemo-flow phase 2 positive smoke" \
  --dangerously-skip-permissions \
  "Create phase2-tool-demo.txt, then read it back and summarize the content."
```

Check positive output:

```bash
test -s ./.nemoflow/opencode.atof.jsonl
test -s ./.nemoflow/opencode.atif.json

grep -E '"session|message|tool|error' ./.nemoflow/opencode.atof.jsonl | head
jq '.. | objects | select(has("sessionID") or has("session_id"))' \
  ./.nemoflow/opencode.atif.json | head
```

Run the disabled path:

```bash
cp opencode.json opencode.enabled.json
jq '(.plugin[0][1].enabled) = false' opencode.json > opencode.disabled.json
mv opencode.disabled.json opencode.json
rm -f ./.nemoflow/*

opencode run \
  --title "nemo-flow phase 2 disabled smoke" \
  "Reply with exactly: plugin disabled smoke."

ls -la ./.nemoflow
mv opencode.enabled.json opencode.json
```

Expected disabled output:

1. OpenCode still completes the run.
2. The NeMo Flow output files are absent or empty.

Run the init-failure path only if the plugin exposes a demo failure switch:

```bash
rm -f ./.nemoflow/*
NEMO_FLOW_OPENCODE_FORCE_INIT_FAILURE=1 opencode run \
  --title "nemo-flow phase 2 init failure smoke" \
  "Reply with exactly: init failure smoke."

grep -i "failed\\|disabled\\|pass-through" ./.nemoflow/opencode-plugin.log
```

Expected failure output:

1. OpenCode still completes the run.
2. The plugin log records one clear pass-through message.
3. No partial or corrupt ATIF/ATOF output is written.

### Phase 3: Retire The Patch For Observability

Once the observability plugin works:

1. Stop treating the OpenCode patch as the observability integration path.
2. Remove OpenCode-specific NeMo Flow flags, config schema, and internal plugin
   code from the OpenCode tree.
3. Update NeMo Flow docs to explain how to install and configure the OpenCode
   plugin.
4. Keep the old patch only as temporary reference material if it is still useful
   for the later intercept investigation.

#### Smoke Test Guide

This demo proves the third promised feature: the observability integration works
with a stock OpenCode install and does not depend on the NeMo Flow OpenCode
patch.

Run from a fresh directory outside the NeMo Flow repository:

```bash
export CLEAN_DEMO_DIR="$(mktemp -d)"
cd "$CLEAN_DEMO_DIR"
mkdir -p .nemoflow

npm install -g opencode-ai@latest
opencode --version | tee ./.nemoflow/opencode-version.txt
which opencode | tee ./.nemoflow/opencode-path.txt

cat > opencode.json <<JSON
{
  "plugin": [
    [
      "$NEMO_FLOW_OPENCODE_PLUGIN",
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

opencode debug info | tee ./.nemoflow/debug-info.txt

opencode run \
  --title "nemo-flow phase 3 clean install smoke" \
  --dangerously-skip-permissions \
  "Create a file named clean-install-demo.txt with the text clean OpenCode plugin demo."
```

Check:

```bash
grep "$NEMO_FLOW_OPENCODE_PLUGIN" ./.nemoflow/debug-info.txt
test -f clean-install-demo.txt
test -s ./.nemoflow/opencode.atof.jsonl
test -s ./.nemoflow/opencode.atif.json
```

Expected demo evidence:

1. The `opencode` binary comes from the normal install path, not
   `third_party/opencode`.
2. The config uses a normal plugin package spec.
3. The demo creates observability output without cloning or patching OpenCode.
4. The NeMo Flow repository is not required at runtime unless using a local file
   plugin during development.

### Phase 4: Investigate Future Intercepts

Only after the observability plugin is working, evaluate whether OpenCode needs
new generic plugin hooks for intercepts:

1. Identify exactly which NeMo Flow request intercept, execution intercept,
   stream intercept, and conditional guardrail behaviors cannot be represented
   with existing OpenCode hooks.
2. Open an upstream OpenCode proposal only for generic plugin infrastructure,
   not NeMo Flow-specific code.
3. Propose `llm.stream.wrap` and `tool.execute.wrap` or equivalent names.
4. Add tests in OpenCode proving nested order, pass-through behavior, error
   propagation, and request/result mutation semantics.
5. Extend the NeMo Flow OpenCode plugin to use those hooks only after the
   upstream API exists.

#### Smoke Test Guide

This demo proves the fourth promised feature after the future OpenCode wrapper
hooks exist: NeMo Flow can participate in real execution, not only passive
observability.

Prerequisite for this phase only:

1. Install an OpenCode build that includes the future wrapper hooks.
2. Install a NeMo Flow OpenCode plugin build with intercept-demo options.

Example config:

```json
{
  "plugin": [
    [
      "@nvidia/nemoflow-opencode-plugin",
      {
        "enabled": true,
        "atofPath": "./.nemoflow/opencode.atof.jsonl",
        "atifPath": "./.nemoflow/opencode.atif.json",
        "logPath": "./.nemoflow/opencode-plugin.log",
        "interceptDemo": {
          "rewriteToolPathFrom": "phase4-input.txt",
          "rewriteToolPathTo": "phase4-rewritten.txt",
          "blockShellPattern": "curl"
        }
      }
    ]
  ]
}
```

Run request-intercept demo:

```bash
cd "$NEMO_FLOW_DEMO_DIR"
rm -f phase4-input.txt phase4-rewritten.txt ./.nemoflow/*

opencode run \
  --title "nemo-flow phase 4 tool rewrite smoke" \
  --dangerously-skip-permissions \
  "Create a file named phase4-input.txt with one line: rewritten by intercept."
```

Check request-intercept output:

```bash
test ! -f phase4-input.txt
test -f phase4-rewritten.txt
grep -E "request_intercept|tool.execute.wrap|phase4-rewritten" \
  ./.nemoflow/opencode.atof.jsonl
```

Run guardrail demo:

```bash
opencode run \
  --title "nemo-flow phase 4 guardrail smoke" \
  --dangerously-skip-permissions \
  "Run curl https://example.com and report the status."
```

Check guardrail output:

```bash
grep -E "blocked|conditional_guardrail|blockShellPattern" \
  ./.nemoflow/opencode.atof.jsonl
```

Expected demo evidence:

1. The real tool call sees rewritten args, proven by
   `phase4-rewritten.txt`.
2. A blocked command is prevented before execution, proven by guardrail events.
3. ATOF/ATIF records include wrapper-hook lifecycle events such as request
   intercept, execution intercept, block, end, and error.
4. These behaviors are impossible to prove with proxy-only interception because
   they change or block execution inside OpenCode.

## Current Monkey Patch Versus Target Plugin

| Area | Current patch | Target observability plugin | Future intercept phase |
| --- | --- | --- | --- |
| Role | Prototype/reference only. | Production integration path for observability. | Later extension for real middleware behavior. |
| Activation | `NEMO_FLOW_ENABLED` or `experimental.nemo_flow` inside patched OpenCode. | Normal OpenCode plugin configuration. | Same plugin, using additional public hooks if available. |
| Runtime dependency | Optional `nemo-flow-node` file dependency in the OpenCode tree. | Plugin package can depend on `nemo-flow-node`; OpenCode does not need to. | Same. |
| LLM lifecycle | Direct patch in OpenCode LLM/session flow. | Approximate from `chat.message`, `chat.params`, and message/session events. | Exact stream lifecycle through `llm.stream.wrap` or equivalent. |
| Tool lifecycle | Direct patch in builtin/MCP/task tool execution. | Successful spans from `tool.execute.before/after`; error limitation documented. | Full lifecycle through `tool.execute.wrap` or equivalent. |
| Event export | Internal plugin using existing `event` hook. | Standalone plugin using existing `event` hook. | Same. |
| Request mutation | Works only where patch passes rewritten request/args back. | Not a first-milestone goal. | Officially supported by future hook contract. |
| Maintenance cost | Rebase patch whenever OpenCode internals move. | Stable public plugin API, tested against supported OpenCode versions. | Small upstream API surface if new hooks are accepted. |

## Non-Goals

The target integration should not:

1. Keep a NeMo Flow-specific OpenCode patch as the normal installation path.
2. Add `nemo-flow-node` as an OpenCode runtime dependency.
3. Add NeMo Flow-specific flags or config schema to OpenCode core.
4. Rely on private OpenCode session internals from the NeMo Flow plugin.
5. Require users to apply patches before using NeMo Flow with OpenCode.
6. Use Switchyard/proxy interception as the only source of agent hierarchy.
7. Block the observability plugin on future intercept hook design.

## Open Questions

1. What should the plugin package be called and where should it live before it
   is published?
2. What is the minimum OpenCode version for `engines.opencode`?
3. Should ATIF/ATOF output be written by default, or only when plugin options
   explicitly enable paths?
4. How should the observability plugin model approximate LLM start/end before a
   stream wrapper exists?
5. Should OpenCode expose one generic `wrap` hook with `kind: "llm" | "tool"`,
   or separate hooks? Separate hooks are clearer and easier to type.
6. Should wrapper hooks run before or after `tool.execute.before` and
   `tool.execute.after`? Recommended order: wrapper owns the whole execution,
   and OpenCode can keep before/after inside the default `next` path for
   backward compatibility.
7. Should stream chunks be exposed as OpenCode's internal AI SDK events or as a
   normalized event shape? Recommended first version: keep OpenCode's current
   stream event objects and let plugins decide how to serialize.
8. Should OpenCode provide an explicit session/agent lifecycle hook? Useful, but
   not a blocker for NeMo Flow if `chat.message`, `event`, and wrapper hooks are
   available.

## Recommended Next Step

Build the standalone NeMo Flow OpenCode observability plugin first.

The first implementation should use only existing OpenCode hooks and normal
OpenCode plugin packaging. It should prove that NeMo Flow can export useful
agent/session/tool observability without any OpenCode patch.

After that is working, use the remaining gaps to justify a small upstream
OpenCode proposal for generic intercept hooks such as `llm.stream.wrap` and
`tool.execute.wrap`.
