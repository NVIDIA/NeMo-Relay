<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Plugin

This package contains Codex hook entries that forward canonical Codex hook JSON
to `nemo-relay` at `/hooks/codex`.

Codex CLI is fully supported for local sessions. Codex GUI or app sessions are
supported only when they run locally and honor the same hook/plugin config and
provider routing. Cloud or remote Codex tasks are partial or unsupported for
local gateway LLM capture.

Requires `codex-cli >= 0.129.0` (introduced the `features.hooks` flag and the
provider alias surface the gateway relies on).

## Files

- `.codex-plugin/plugin.json` describes the Codex plugin package.
- `hooks/hooks.json` contains Codex hook entries that run
  `nemo-relay plugin-shim hook codex`.
- `nemo-relay install codex` creates the local marketplace, installs the plugin,
  and persists Codex hook and provider configuration using `nemo-relay` from
  `PATH`.

## Captured Events

With `codex-cli >= 0.129.0`, the minimum supported installed hooks are
`SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`,
`PermissionRequest`, `Stop`, `PreCompact`, and `PostCompact`.

The hook template also documents events used by newer or broader host hook
surfaces, including `SessionEnd`, `PostToolUseFailure`, `SubagentStart`,
`SubagentStop`, and `Notification`. Relay forwards any delivered supported hook
as scope, tool, mark, or private LLM correlation events, but v1 does not depend
on Codex exposing those broader events.

Transparent setup injects these hooks with CLI config overrides. Plugin setup
does not install hooks from the package template directly. It writes
`features.hooks = true` in `.codex/config.toml`, configures the
`nemo-relay-openai` provider alias, and merges hook shim entries into
`.codex/hooks.json`.

Codex plugin mode uses hook-supervised lazy startup only. It does not install a
user-level daemon, launchd agent, systemd user service, scheduled task, login
item, wrapper, or persistent supervisor. The sidecar starts only when a Codex
hook invokes `nemo-relay plugin-shim hook codex`.

## Transparent Setup

Build or install the gateway binary so `nemo-relay` is on `PATH`.

Run Codex through the wrapper:

```bash
nemo-relay run -- codex
```

The wrapper starts a per-invocation gateway on a dynamic localhost port,
enables Codex hooks with CLI config overrides, injects hook commands that use
`NEMO_RELAY_GATEWAY_URL`, and points Codex at a temporary `nemo-relay-openai`
provider alias that uses the gateway URL while preserving Codex's OpenAI auth
path.

Inspect the launch without starting Codex:

```bash
nemo-relay run \
  --dry-run \
  --print \
  -- codex
```

## Shared Config

Use `.nemo-relay/config.toml` for project defaults or
`~/.config/nemo-relay/config.toml` for user defaults:

```toml
[agents.codex]
command = "codex"
```

Configure observability with `nemo-relay plugins edit --project` or
`.nemo-relay/plugins.toml`:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atif]
enabled = true
output_directory = ".nemo-relay/atif"
```

Then run:

```bash
nemo-relay run --agent codex
```

## Standalone Gateway

Use the long-running gateway only when you do not want to launch Codex through
the wrapper. Start the gateway manually:

```bash
nemo-relay --bind 127.0.0.1:4040
```

Then edit `~/.codex/config.toml` and configure local Codex to use a gateway
provider alias instead of overriding the reserved built-in `openai` provider:

```toml
model_provider = "nemo-relay-openai"

[model_providers.nemo-relay-openai]
name = "NeMo Relay OpenAI"
base_url = "http://127.0.0.1:4040"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
```

After saving the file, restart the Codex GUI or app so it reloads the provider
configuration. For CLI usage, start a new `codex` process.

Some Codex GUI or app versions appear to scope visible conversation history by
the active provider configuration. If existing conversations disappear after
switching `model_provider` to `nemo-relay-openai`, the history has not been
removed if it returns after restoring the previous provider configuration. Use
this standalone provider alias only while capturing gateway telemetry, or prefer
the transparent wrapper for CLI sessions. See the upstream Codex
[history visibility discussion](https://github.com/openai/codex/issues/15494#issuecomment-4164170537)
for context.

## Verify

Run a Codex session that starts, uses one simple tool, and ends. Confirm that
ATIF was written:

```bash
ls .nemo-relay/atif
```

For a direct endpoint smoke test against a manually started gateway:

```bash
curl -f http://127.0.0.1:4040/healthz
printf '{"session_id":"smoke-codex","hook_event_name":"sessionStart"}' \
  | NEMO_RELAY_GATEWAY_URL=http://127.0.0.1:4040 nemo-relay hook-forward codex --fail-closed
```

If hooks arrive but LLM spans are missing, confirm Codex was started by
`nemo-relay run` or that the active provider points to the gateway URL.

If LLM spans are present but attached to the top-level agent instead of a
subagent, include `x-nemo-relay-subagent-id` on gateway requests or share
`conversation_id`, `generation_id`, or `request_id` values between hook payloads
and provider requests.

## Standalone Plugin Installation

Preferred release install:

```bash
nemo-relay install codex
```

`nemo-relay install codex` writes a local Codex marketplace, registers
`nemo-relay-plugin`, enables Codex hooks, and configures the
`nemo-relay-openai` provider alias. Codex sidecar lifecycle remains
hook-supervised lazy startup only; the installer does not create a wrapper or
daemon.

The install command requires `nemo-relay` to be available on `PATH`. It does not
require launching Codex through the `nemo-relay` wrapper and does not install a
user-level daemon.

Repo marketplace discovery is also supported:

```bash
codex plugin marketplace add NVIDIA/NeMo-Relay
codex plugin add nemo-relay-plugin@nemo-relay
```

That path reads `.agents/plugins/marketplace.json` from the repository and
installs this Codex plugin from `integrations/coding-agents/codex`. Source hooks
invoke `nemo-relay plugin-shim hook codex` directly.

Treat the source marketplace path as discovery or manifest validation. For the
complete provider and generated-hook setup, remove the source-installed plugin
first and then run `nemo-relay install codex`. Keeping both the source plugin
and the generated install active can forward the same Codex hook twice.

Package or unpack the plugin so the plugin root contains:

```text
nemo-relay-plugin/
  .codex-plugin/plugin.json
  hooks/hooks.json
```

Create a local Codex marketplace and copy the plugin under that marketplace
root:

```bash
MARKETPLACE_ROOT="$HOME/.local/share/nemo-relay/codex-marketplace"
PLUGIN_ROOT="$MARKETPLACE_ROOT/plugins/nemo-relay-plugin"
mkdir -p "$MARKETPLACE_ROOT/.agents/plugins" "$MARKETPLACE_ROOT/plugins"
cp -R /path/to/nemo-relay-plugin "$PLUGIN_ROOT"
```

Create `$MARKETPLACE_ROOT/.agents/plugins/marketplace.json`:

```json
{
  "name": "nemo-relay-local",
  "interface": {
    "displayName": "NeMo Relay Local"
  },
  "plugins": [
    {
      "name": "nemo-relay-plugin",
      "source": {
        "source": "local",
        "path": "./plugins/nemo-relay-plugin"
      },
      "policy": {
        "installation": "AVAILABLE",
        "authentication": "ON_INSTALL"
      },
      "category": "Coding"
    }
  ]
}
```

Registering the local marketplace with Codex is useful for development and
manifest validation:

```bash
codex plugin marketplace add "$MARKETPLACE_ROOT"
codex plugin add nemo-relay-plugin@nemo-relay-local
```

For end-to-end installation, prefer `nemo-relay install codex`; it performs the
marketplace registration and the persistent Codex provider/hook setup together.
If you used the manual source marketplace commands above, remove that plugin
before running the full installer so source hook templates and generated
persistent hooks do not both forward the same event.

The installer writes a provider alias like:

```toml
model_provider = "nemo-relay-openai"

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
```

Run read-only plugin checks:

```bash
nemo-relay doctor --plugin codex
```

Start a normal Codex session:

```bash
codex
```

The installed hooks start the Relay sidecar lazily on
`http://127.0.0.1:47632`, and the Codex provider alias routes model traffic
through that sidecar. No launchd agent, systemd user service, scheduled task,
login item, wrapper, or persistent supervisor is installed.

To upgrade, replace the plugin directory contents with the new package for the
same host, keep the same `MARKETPLACE_ROOT`, refresh the local marketplace
registration, and rerun the top-level installer:

```bash
codex plugin remove nemo-relay-plugin@nemo-relay-local
codex plugin marketplace remove nemo-relay-local
codex plugin marketplace add "$MARKETPLACE_ROOT"
codex plugin add nemo-relay-plugin@nemo-relay-local
nemo-relay install codex
```

To uninstall, remove NeMo Relay's Codex config and hook entries, remove the
marketplace registration, and remove the generated marketplace directory:

```bash
nemo-relay uninstall codex
```

Full first-request LLM capture depends on Codex firing one of the installed
hooks, especially `SessionStart` or `UserPromptSubmit`, before its first model
provider request. If a Codex version sends the provider request first, the first
request may fail or may not be captured until the next hook starts Relay.
