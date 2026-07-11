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

Requires `codex-cli >= 0.143.0`, including the complete required lifecycle-hook
set, plugin app-server metadata, `features.hooks`, and provider alias surfaces
used by the installer.

## Files

- `.codex-plugin/plugin.json` describes the Codex plugin package.
- `.mcp.json` starts the native `nemo-relay mcp` lifecycle client and requires
  successful gateway initialization.
- `hooks/hooks.json` contains Codex hook entries that run
  `nemo-relay plugin-shim hook codex`.
- `nemo-relay install codex` creates the local marketplace, installs the plugin,
  and persists Codex provider and exact plugin-hook trust using `nemo-relay`
  from `PATH`.

## Captured Events

With `codex-cli >= 0.143.0`, persistent installation requires `SessionStart`,
`UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PermissionRequest`,
`SubagentStart`, `SubagentStop`, `Stop`, `PreCompact`, and `PostCompact`. Relay
requires exactly one enabled and trusted app-server handler for every generated
event and forwards delivered hooks as scope, tool, mark, or private LLM
correlation events. `PostToolUseFailure`, `Notification`, and `SessionEnd` are not
in the Codex 0.143 plugin hook schema and are not generated.

Each delivered `Stop` closes the active turn and writes a cumulative ATIF
snapshot. Because the plugin schema does not expose `SessionEnd`, the final
`Stop` is the final session snapshot.

Transparent setup injects hooks with CLI config overrides. Persistent setup
writes `features.hooks = true` in `.codex/config.toml`, configures the
`nemo-relay-openai` provider alias, and uses this plugin's `hooks/hooks.json` as
the sole persistent Relay hook source. It does not add Relay groups to
`.codex/hooks.json`.

Persistent installation opens the stable Codex app-server interface and
selects only hooks whose source is `plugin`, plugin ID is
`nemo-relay-plugin@nemo-relay-local`, and command exactly matches the generated
canonical Relay shim command. It requires exactly one handler for each event
in the complete 10-event supported set listed above.

Unrelated user, project, and plugin hooks are never trusted. If installation
fails after a trust write, it restores every targeted hook's prior trusted,
modified, disabled, or absent state together with Codex config and backups.
Upgrade removes legacy Relay user-hook groups while preserving unrelated hooks.

Codex plugin mode starts the native `nemo-relay mcp` subcommand through the
plugin MCP configuration. That Rust process is a lightweight lifecycle client:
it starts or reuses a detached native `nemo-relay --bind 127.0.0.1:47632`
sidecar, rejects foreign listeners, and completes the MCP handshake only after
Relay identity, version, bootstrap protocol, and effective persistent
configuration are verified. Compatibility uses a one-way fingerprint of the
resolved settings and relevant environment values without exposing secrets.
The complete dynamic plugin activation snapshot, including adjacent runtime
files and a copied managed Python environment, is limited to 100,000 filesystem
entries and 512 MiB in total, with a maximum directory traversal depth of 128.
If startup reports an activation snapshot budget error, remove unrelated files
from the manifest or load-target directory, flatten deeply nested directories,
or reduce the managed Python environment before retrying. Concurrent Codex and
Claude Code plugin processes can share the gateway and heartbeat it every 30
seconds. The sidecar
remains available for 300 idle seconds after the final client closes. If it dies
while MCP remains open, one coordinated restart is attempted. The MCP server
advertises no tools.

On Windows, Relay requests Job Object breakaway only when the host job permits
it. Under a restrictive Job Object that permits nested jobs, Relay keeps the
sidecar scoped to the host job and retains a nested cleanup job for the sidecar
process tree. The sidecar cannot outlive the host job, so the 300-second idle
reuse window can end early. If Relay cannot create or configure the cleanup job,
or the host rejects nested assignment, persistent bootstrap fails actionably
instead of running without process-tree cleanup guarantees.

Persistent mode loads only system and user Relay configuration and starts from
the user configuration directory. Project `.nemo-relay` layers remain specific
to transparent `nemo-relay run` invocations. The MCP manifest forwards approved
provider, Relay, OpenTelemetry, AWS, proxy, certificate, and config-referenced
credential variable names without storing values.

Plugin-owned hook commands pin `http://127.0.0.1:47632` explicitly, so an
ambient `NEMO_RELAY_PLUGIN_GATEWAY_URL` cannot split hook traffic from the
required MCP-managed gateway.

If the Relay version, user configuration, or forwarded credentials change, an
MCP client refuses to reuse the incompatible sidecar. `nemo-relay install codex
--force` retires an installer-owned sidecar through its private shutdown token
before refreshing the plugin.

An existing install without MCP generation fencing cannot be retired safely,
even when Codex reports that the plugin is no longer registered, because an
already-running MCP process can outlive registration. If upgrade or uninstall
reports a missing MCP generation marker, close every Codex client and
standalone `nemo-relay mcp` process, then run:

```bash
codex plugin remove nemo-relay-plugin@nemo-relay-local
codex plugin marketplace remove nemo-relay-local
```

Remove `codex-marketplace` and `codex.json` from the install directory named in
the error, then retry `nemo-relay install codex --force` to install a fenced
generation. If removal was the original goal, run `nemo-relay uninstall codex`
immediately afterward; the fenced reinstall lets Relay remove provider and hook
trust state transactionally.

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

## Configure Transparent Runs

Use `.nemo-relay/config.toml` for project defaults:

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

This example writes ATIF files under the project at `.nemo-relay/atif`.

## Configure the Persistent Plugin

Use `~/.config/nemo-relay/config.toml`, or
`$XDG_CONFIG_HOME/nemo-relay/config.toml` when `XDG_CONFIG_HOME` is set, for
persistent provider defaults. Run `nemo-relay plugins edit` without
`--project` to write user-scoped observability configuration. For example:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atif]
enabled = true
output_directory = "atif"
```

Persistent mode ignores project layers and starts the sidecar in the user Relay
configuration directory. The relative path above resolves to
`$XDG_CONFIG_HOME/nemo-relay/atif`, or `~/.config/nemo-relay/atif` when
`XDG_CONFIG_HOME` is not set.

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

Run a Codex session that starts, uses one simple tool, and ends. For a
transparent project run, confirm that ATIF was written:

```bash
ls .nemo-relay/atif
```

For the persistent user-scoped configuration above, enter:

```bash
ls "${XDG_CONFIG_HOME:-$HOME/.config}/nemo-relay/atif"
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
`nemo-relay-openai` provider alias. It writes a required MCP server entry that
invokes the resolved native `nemo-relay` binary. Installation automatically
trusts the exact plugin-owned hook definitions through `codex app-server` and
rolls back files and original trust state if activation cannot be verified.

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

Treat the source marketplace path as discovery or manifest validation. Use
`nemo-relay install codex` for the complete provider, environment-forwarding,
and verified-trust setup.

Package or unpack the plugin so the plugin root contains:

```text
nemo-relay-plugin/
  .codex-plugin/plugin.json
  .mcp.json
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

For end-to-end installation, use `nemo-relay install codex`; it performs the
marketplace registration and persistent provider/plugin-hook setup together.

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

Doctor reports the generated MCP server, native `nemo-relay mcp` support,
plugin hook installation, environment forwarding, and live Codex trust state.
In JSON mode, inspect `checks.codex_hooks_trusted` and `codex_hook_trust` for
untrusted, modified, disabled, or missing required hook entries.

Start a normal Codex session:

```bash
codex
```

Start a new CLI process after install, or restart the Codex desktop app if it
was already open, so the provider selection and hooks are reloaded.

The required plugin MCP server starts or reuses the shared native Relay gateway
on `http://127.0.0.1:47632` before Codex begins the captured turn, and the
provider alias routes model traffic through it.

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

To uninstall, remove NeMo Relay's Codex config and exact plugin-hook trust,
remove the marketplace registration, and remove the generated marketplace
directory:

```bash
nemo-relay uninstall codex
```

Codex can perform provider discovery before it launches plugin MCP servers. A
cold start can therefore produce transient `/models` connection failures that
Codex retries. Because the MCP server is required, Codex does not begin the
captured turn or send its `/responses` request until the native Relay gateway is
ready; if startup fails, the turn fails instead of silently bypassing Relay.
