<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hermes Patch Maintenance Notes

These notes are for maintaining and validating the tracked Hermes patch in this
repository. They are not the primary user path for Hermes observability.

Use the current runtime paths first:

- NeMo Relay CLI wrapper: `nemo-relay hermes` manages the local gateway lifetime
  and observes Hermes through shell hooks plus gateway-routed provider traffic.
  See [`docs/nemo-relay-cli/hermes.mdx`](../../docs/nemo-relay-cli/hermes.mdx).
- Upstream Hermes plugin: Hermes' bundled `observability/nemo_relay` plugin
  emits NeMo Relay observability from inside Hermes. Observe-only plugin builds
  keep Hermes in control of LLM and tool execution.
- Adaptive execution: available only when the Hermes build includes adaptive
  middleware support and the installed NeMo Relay runtime exposes managed
  `llm.execute(...)` and `tools.execute(...)` boundaries. Verify the Hermes
  release tag before treating this as released behavior.

The maintained patch is a source-first compatibility artifact for the pinned
`third_party/hermes-agent` checkout. It wires Hermes to a NeMo Relay plugin
entry point, lifecycle hooks, and the legacy ACG override seam. At runtime, the
patch emits a NeMo Relay session scope plus manual LLM/tool lifecycle spans and
uses `AtifExporter` to materialize trajectory JSON on session finalization.

All commands assume your working directory is the NeMo Relay repo root unless a
step says otherwise.

## Prerequisites

- `git`
- `python3.11`
- `uv`
- A writable Hermes home directory at `${HERMES_HOME:-$HOME/.hermes}`
- Optional for live model validation: an Anthropic API key that you will place
  in `~/.hermes/.env`
- Optional for OpenInference validation: an OTLP gRPC backend such as Arize
  Phoenix listening on `http://127.0.0.1:4317`

Keep credentials in local `.env` files only. Do not commit `ANTHROPIC_API_KEY`
or any other secret-bearing `.env` file into the repo.

## Install

### Step 1: Prepare the pinned Hermes checkout

The maintained Hermes baseline is pinned in `third_party/sources.lock`. Clone
Hermes and detach at that exact commit before you apply the NeMo-Relay patch.

```bash
HERMES_COMMIT=$(git config -f third_party/sources.lock --get submodule.third_party/hermes-agent.commit)
git clone https://github.com/NousResearch/hermes-agent.git third_party/hermes-agent
git -C third_party/hermes-agent fetch --tags origin
git -C third_party/hermes-agent checkout --detach "$HERMES_COMMIT"
git -C third_party/hermes-agent log --oneline -1
```

Success signal: `git log --oneline -1` prints the same commit you read from
`third_party/sources.lock`.

### Step 2: Apply the tracked NeMo-Relay patch

Check that the patch applies cleanly, then apply it.

```bash
git -C third_party/hermes-agent apply --check ../../patches/hermes-agent/0001-add-nemo-relay-integration.patch
git -C third_party/hermes-agent apply ../../patches/hermes-agent/0001-add-nemo-relay-integration.patch
```

Success signal: `git apply --check` prints no errors, and the second command
returns to the shell without conflicts.

### Step 3: Bootstrap Hermes with the `nemo-relay` extra

Hermes discovers this integration through its plugin entry points, so reinstall
the editable package with the `nemo-relay` extra after the patch is applied.

```bash
cd third_party/hermes-agent
uv venv .venv --python 3.11
. .venv/bin/activate
uv pip install -e '.[nemo-relay]' --force-reinstall
uv run hermes --help
```

Success signal: `uv run hermes --help` prints the Hermes CLI help from the
activated `.venv`.

## Enable

### Step 4: Put patch runtime settings in `~/.hermes/.env`

This configuration is for the patch path only. Current upstream Hermes plugin
configuration uses Hermes' bundled `observability/nemo_relay` plugin and its
documented `HERMES_NEMO_RELAY_*` settings instead.

Hermes loads `~/.hermes/.env` first and treats it as the durable source of
truth. If a repo-local project `.env` also exists, Hermes only uses that file
as a development fallback. When `~/.hermes/.env` is present, its values win and
the project `.env` only fills in missing variables.

```bash
mkdir -p "${HERMES_HOME:-$HOME/.hermes}"
cat > "${HERMES_HOME:-$HOME/.hermes}/.env" <<'EOF'
# Optional: required only for a real Anthropic-backed agent turn.
ANTHROPIC_API_KEY=<paste-your-key-here>
HERMES_NEMO_RELAY_ENABLED=1
HERMES_NEMO_RELAY_ACG_ENABLED=1
HERMES_NEMO_RELAY_ATIF_DIR=${HERMES_HOME:-$HOME/.hermes}/atif
HERMES_NEMO_RELAY_OPENINFERENCE_ENABLED=1
HERMES_NEMO_RELAY_OPENINFERENCE_TRANSPORT=grpc
HERMES_NEMO_RELAY_OPENINFERENCE_ENDPOINT=http://127.0.0.1:4317
HERMES_NEMO_RELAY_OPENINFERENCE_SERVICE_NAME=hermes-agent
HERMES_NEMO_RELAY_OPENINFERENCE_INSTRUMENTATION_SCOPE=hermes-agent/nemo-relay/openinference
EOF
```

Use these knobs as the patch runtime contract:

- `HERMES_NEMO_RELAY_ENABLED=1` enables the integration. If it is unset,
  Hermes falls back to `nemo_relay.enabled` in `~/.hermes/config.yaml`. The
  default is off.
- `HERMES_NEMO_RELAY_ACG_ENABLED=1` turns on the ACG override path. If the
  master switch is on and this sub-toggle is unset, Hermes defaults ACG to on.
- `HERMES_NEMO_RELAY_ACG_ENABLED=0` keeps the plugin loaded but preserves native
  Hermes prompt-caching behavior.
- `HERMES_NEMO_RELAY_ATIF_DIR` overrides the ATIF output directory. If it is
  unset, Hermes falls back to `nemo_relay.atif_output_dir` in YAML and then to
  `${HERMES_HOME}/atif`.
- `HERMES_NEMO_RELAY_OPENINFERENCE_ENABLED=1` turns on OTLP export for the
  emitted NeMo-Relay events.
- `HERMES_NEMO_RELAY_OPENINFERENCE_TRANSPORT=grpc` selects the OTLP gRPC
  exporter path that Phoenix expects on port `4317`.
- `HERMES_NEMO_RELAY_OPENINFERENCE_ENDPOINT` points the plugin at the OTLP
  collector endpoint, for example `http://127.0.0.1:4317`.
- `HERMES_NEMO_RELAY_OPENINFERENCE_SERVICE_NAME` and
  `HERMES_NEMO_RELAY_OPENINFERENCE_INSTRUMENTATION_SCOPE` control how the spans
  show up in the OpenInference-aware backend.

### Step 5: Use YAML only as a fallback for non-secret settings

If you prefer to keep non-secret toggles in YAML, put them in
`~/.hermes/config.yaml`. Environment variables still take precedence over this
file.

```yaml
nemo_relay:
  enabled: true
  atif_output_dir: /absolute/path/to/atif
  acg:
    enabled: true
  openinference:
    enabled: true
    transport: grpc
    endpoint: http://127.0.0.1:4317
    service_name: hermes-agent
    instrumentation_scope: hermes-agent/nemo-relay/openinference
```

Recommendation: keep credentials and the primary on/off switches in
`~/.hermes/.env`, and reserve YAML for optional non-secret defaults.

## Smoke Validation

You are now in a patched Hermes checkout with the `nemo-relay` extra installed
and the patch runtime knobs set in `~/.hermes/.env`.

### Structural validation

First confirm that Hermes can discover the plugin entry point and import the
runtime modules that the patch added.

```bash
cd third_party/hermes-agent
. .venv/bin/activate
python - <<'PY'
import importlib.metadata as metadata

eps = metadata.entry_points()
if hasattr(eps, "select"):
    group = eps.select(group="hermes_agent.plugins")
elif isinstance(eps, dict):
    group = eps.get("hermes_agent.plugins", [])
else:
    group = [ep for ep in eps if ep.group == "hermes_agent.plugins"]

matches = [ep.value for ep in group if ep.name == "nemo_relay"]
assert matches == ["plugins.nemo_relay"], matches

import plugins.nemo_relay as plugin

assert callable(getattr(plugin, "register", None))
print("entrypoint:", matches[0])
print("register(): ok")
PY
```

Success signal:

- The snippet prints `entrypoint: plugins.nemo_relay`
- The snippet prints `register(): ok`

### Lifecycle smoke without a model key

Then run one real Hermes plugin lifecycle without calling an external model.
This exercises the maintained runtime integration in the patch: entry-point
discovery, plugin hook registration, session lifecycle, ATIF export, and
OpenInference OTLP export over gRPC.

```bash
cd third_party/hermes-agent
. .venv/bin/activate
ATIF_DIR="${HERMES_NEMO_RELAY_ATIF_DIR:-${HERMES_HOME:-$HOME/.hermes}/atif}"
mkdir -p "$ATIF_DIR"

python - <<'PY'
import uuid
from hermes_cli.plugins import discover_plugins, get_plugin_manager, invoke_hook

discover_plugins()
plugins = get_plugin_manager().list_plugins()
assert any(plugin["name"] == "nemo_relay" and plugin["enabled"] for plugin in plugins), plugins

session_id = f"phoenix-smoke-{uuid.uuid4().hex[:8]}"
model = "anthropic/claude-sonnet-4"

invoke_hook("on_session_start", session_id=session_id, model=model, platform="cli")
invoke_hook(
    "pre_api_request",
    task_id="phoenix-smoke-task",
    session_id=session_id,
    platform="cli",
    model=model,
    provider="anthropic",
    base_url="https://api.anthropic.com",
    api_mode="anthropic_messages",
    api_call_count=1,
    message_count=1,
    tool_count=1,
    approx_input_tokens=8,
    request_char_count=24,
    max_tokens=64,
)
invoke_hook(
    "post_api_request",
    task_id="phoenix-smoke-task",
    session_id=session_id,
    platform="cli",
    model=model,
    provider="anthropic",
    base_url="https://api.anthropic.com",
    api_mode="anthropic_messages",
    api_call_count=1,
    api_duration=0.123,
    finish_reason="stop",
    message_count=1,
    response_model=model,
    usage={"input_tokens": 8, "output_tokens": 3, "total_tokens": 11},
    assistant_content_chars=2,
    assistant_tool_call_count=1,
)
invoke_hook(
    "pre_tool_call",
    tool_name="echo",
    args={"text": "hello from hermes"},
    task_id="phoenix-smoke-task",
    session_id=session_id,
    tool_call_id="tool-1",
)
invoke_hook(
    "post_tool_call",
    tool_name="echo",
    result={"text": "hello from hermes"},
    task_id="phoenix-smoke-task",
    session_id=session_id,
    tool_call_id="tool-1",
)
invoke_hook("on_session_finalize", session_id=session_id, platform="cli")
print(session_id)
PY

ls -lt "$ATIF_DIR" | head
```

Success signal:

- The Python snippet prints a fresh `phoenix-smoke-...` session ID
- `ls -lt "$ATIF_DIR"` shows a fresh session JSON written by the finalize hook
- The plugin manager reports `nemo_relay` as enabled before the smoke emits any
  events

### Verify ingestion in Phoenix

If Phoenix is running locally, query its REST API for the root span using the
session name that the smoke printed.

```bash
SESSION_ID=<paste-the-phoenix-smoke-session-id>
curl -s --get 'http://127.0.0.1:6006/v1/projects/default/spans' \
  --data-urlencode "name=hermes-session-${SESSION_ID}" \
  --data 'limit=5'
```

Success signal:

- The response includes one `AGENT` span named
  `hermes-session-${SESSION_ID}`
- The trace also contains child `LLM` and `TOOL` spans beneath that root span
- The `LLM` span carries OpenInference LLM attributes such as `input.value`,
  `output.value`, and `llm.model_name`
- The `TOOL` span carries OpenInference tool attributes such as
  `tool.name`, `tool_call.function.arguments`, and `output.value`

### Optional live Anthropic smoke

When an Anthropic API key is available, run one real Hermes turn and explicitly
finalize the plugin session so the trajectory exporter flushes to disk. This
also exercises the Anthropic ACG override seam when
`HERMES_NEMO_RELAY_ACG_ENABLED=1`.

```bash
cd third_party/hermes-agent
. .venv/bin/activate
ATIF_DIR="${HERMES_NEMO_RELAY_ATIF_DIR:-${HERMES_HOME:-$HOME/.hermes}/atif}"
mkdir -p "$ATIF_DIR"

python - <<'PY'
from hermes_cli.plugins import invoke_hook
from run_agent import AIAgent

agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
    skip_context_files=True,
    skip_memory=True,
)

try:
    reply = agent.chat("Reply with exactly OK.")
    print(reply)
finally:
    invoke_hook(
        "on_session_finalize",
        session_id=agent.session_id,
        platform="cli",
    )
PY
```

### Verify exported trajectory location

When the integration is enabled, exported trajectory JSON lands in this
precedence order:

1. `HERMES_NEMO_RELAY_ATIF_DIR`
2. `nemo_relay.atif_output_dir` in `~/.hermes/config.yaml`
3. `${HERMES_HOME:-$HOME/.hermes}/atif`

After the smoke suite finishes, confirm that the expected directory contains a
fresh session JSON for the run you just exercised.

## Disable

To keep Hermes patched but turn off NeMo-Relay completely, set
`HERMES_NEMO_RELAY_ENABLED=0` in `~/.hermes/.env` or remove the `nemo_relay`
block from `~/.hermes/config.yaml`.

To keep observability installed but disable only ACG ownership, leave
`HERMES_NEMO_RELAY_ENABLED=1` and set `HERMES_NEMO_RELAY_ACG_ENABLED=0`.

After changing either switch, start a new shell or reactivate the `.venv`
before rerunning Hermes. If you want a quick post-change check, rerun the CLI
smoke command from the previous section before you continue operator work.

## Uninstall

To return this checkout to native Hermes, remove the NeMo-Relay-specific config,
delete the patched virtualenv, and reinstall Hermes without the `nemo-relay`
extra.

```bash
rm -rf third_party/hermes-agent/.venv
cd third_party/hermes-agent
uv venv .venv --python 3.11
. .venv/bin/activate
uv pip install -e . --force-reinstall
```

If you also want a clean upstream tree, reclone `third_party/hermes-agent` from
the pinned commit in `third_party/sources.lock`, skip the patch-apply step, and
leave the `HERMES_NEMO_RELAY_*` variables out of `~/.hermes/.env`.

## Patch Refresh

Patch maintenance always starts from the NeMo-Relay repo root. The checked-in
scripts are tracked `100644`, so invoke them with `bash` rather than trying to
execute them directly.

### Replay the patch against the pinned Hermes baseline

First reset the Hermes checkout to the manifest pin, then run the dry-run patch
replay:

```bash
HERMES_COMMIT=$(git config -f third_party/sources.lock --get submodule.third_party/hermes-agent.commit)
git -C third_party/hermes-agent checkout --detach "$HERMES_COMMIT"
bash ./scripts/apply-patches.sh --check
```

Success signal: the Hermes patch is processed with no `git apply` errors.

### Regenerate after intentional Hermes changes

After you finish updating files under `third_party/hermes-agent`, regenerate the
patch and immediately replay-check it again:

```bash
bash ./scripts/generate-patches.sh
bash ./scripts/apply-patches.sh --check
```

Success signal:

- `patches/hermes-agent/0001-add-nemo-relay-integration.patch` contains your new
  Hermes delta
- `bash ./scripts/apply-patches.sh --check` still returns without patch
  failures
- The structural and live smoke steps in the `Smoke Validation` section still
  work before you hand the patch to another operator
