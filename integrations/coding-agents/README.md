<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Coding-Agent Observability Integrations

This directory contains hook integration bundles for coding agents that should
be observed by `nemo-relay`.

The gateway combines two observability paths:

- Agent lifecycle hooks for sessions, prompts, subagents, tool calls,
  compaction, responses, and stop events.
- A passthrough LLM gateway for OpenAI-compatible and Anthropic-compatible
  provider traffic.

Hook integrations preserve each coding agent's canonical hook payload. They do
not wrap the payload in a shared NeMo Relay envelope. Gateway-specific settings
travel through the transparent wrapper, hook command arguments, HTTP headers,
environment variables, or shared TOML config.

## Packages

Each host uses a slightly different integration surface:

- `claude-code/` is a Claude Code plugin package. The
  `nemo-relay install claude-code` command installs a native MCP lifecycle
  client and hook entries targeting `POST /hooks/claude-code` through
  `nemo-relay` on `PATH`.
- `codex/` is a Codex plugin package. `nemo-relay install codex` creates the
  marketplace, installs the plugin, enables `features.hooks = true`, and
  configures a local `nemo-relay-openai` provider alias. Codex plugin delivery
  uses required native `nemo-relay mcp` lifecycle clients. Claude Code starts
  the same lifecycle client automatically from its plugin. Clients from either
  host share one Rust gateway, subject to the Windows Job Object lifetime
  caveat below, with no wrapper, login item, launchd agent, systemd user
  service, scheduled task, or persistent supervisor.
- Hermes does not require a static marketplace bundle. The
  `nemo-relay install hermes` command adds a native MCP lifecycle client,
  canonical hooks, and exact per-event trust to the user-owned Hermes config as
  one transaction.

## Transparent Setup

Build or install the gateway binary so `nemo-relay` is on `PATH`.

Prefer the wrapper. It starts a gateway on a dynamic `127.0.0.1` port, injects
temporary hook and gateway configuration, runs the agent, and shuts the gateway
down when the agent exits.

```bash
nemo-relay run -- claude
nemo-relay run -- codex
nemo-relay run -- hermes
```

When a wrapper hides the agent command name, configure that wrapper under
`[agents.<host>].command` and select it with
`--agent claude|codex|hermes`. Use `--dry-run --print` to inspect generated
config without launching.

Use `nemo-relay doctor` to inspect environment, config, agent commands, hook
readiness, observability outputs, and shell completions. Scope the report to one
agent when troubleshooting launch readiness:

```bash
nemo-relay doctor
nemo-relay doctor codex
nemo-relay doctor hermes --json
```

The command is read-only: it reports missing ATIF directories, hook files, and
agent commands instead of creating or patching them.

## Persistent Integration Installation

The `nemo-relay` CLI installs the Claude Code and Codex plugins and manages the
Hermes user integration. The CLI must already be available on `$PATH` or
`%PATH%`; you do not need a separate npm installer, release bundle, or
plugin-local Relay binary.

Persistent installation and transparent launch require Claude Code 2.1.121 or
newer, `codex-cli` 0.143.0 or newer, or Hermes Agent 0.18.2 or newer for the
selected agent.

Each plugin MCP entry—and the equivalent Hermes `mcp_servers` entry—starts
`nemo-relay mcp`, a lightweight client that starts or reuses a native
`nemo-relay --bind 127.0.0.1:47632` sidecar. Relay detaches the sidecar when
host policy permits. A restrictive Windows Job Object can limit the sidecar to
the host job. If nested assignment cannot provide the required process-tree
cleanup guarantee, bootstrap stops and explains the conflict.

The MCP process acquires the gateway before reading protocol frames and returns
its initialization response only after it verifies Relay identity, version,
and bootstrap-protocol readiness. Concurrent Codex, Claude Code, and
Hermes processes share the gateway and heartbeat it while their MCP stdio
connections remain open; the gateway exits after the final client's idle
timeout. Overlapping MCP clients coordinate one restart for the endpoint, even
when their heartbeats arrive at different times.
Codex requires MCP initialization before the captured turn. Claude Code marks
Relay MCP as `alwaysLoad`, so it also waits for the connection before session
startup. Hermes starts MCP discovery asynchronously, so an early
generation-fenced command hook waits for the MCP-owned gateway. Installed MCP
entries and hook commands carry both their generation-file path and the
immutable identity expected there, so cached host configuration cannot adopt a
replacement installation at the same path. The
MCP client advertises no tools.

MCP bootstrap is host-neutral: all three generated integrations use
`nemo-relay mcp`. Agent identity appears only in lifecycle hook commands, where
Relay needs it to translate each host's canonical payload. The old
`mcp --agent <agent>` form no longer parses. Refresh a fenced installation with
`nemo-relay install <host> --force`; if its generation marker is missing, Relay
refuses the upgrade and prints the manual cleanup steps.

Persistent mode loads system and user Relay configuration only and starts the
sidecar from the user configuration directory. This keeps relative exporter
paths stable across projects. Codex's generated MCP manifest forwards
approved provider, Relay, OpenTelemetry, AWS, proxy, certificate, and
config-referenced credential environment names without storing their values;
Claude Code supplies its normal MCP process environment. Use transparent
`nemo-relay run` for project-specific configuration. The managed sidecar
injects a forwarded provider key only for a request with provider authorization
or Relay's private per-user client proof. Codex receives that derived proof in
its managed provider headers; the installer writes that config privately and
Relay consumes the proof before middleware, telemetry, or upstream forwarding.
Claude Code and Hermes send their normal provider authorization, so an
unrelated loopback caller cannot spend forwarded keys.

Install the persistent integrations with:

```bash
nemo-relay install claude-code
nemo-relay install codex
nemo-relay install hermes
nemo-relay install all
```

For Claude Code and Codex, `nemo-relay install` writes local marketplace files,
registers the selected host plugin, and performs the required provider and hook
setup. For Hermes, `install` is the only command that updates Relay-owned user
MCP, hook, trust, and generation state; interactive `config hermes` manages
only the transparent wrapper. Use
`nemo-relay uninstall <host>` to roll back and
`nemo-relay doctor --plugin <host>` to check an installed integration.

If you are using Codex, add this repository as a marketplace for source/dev
discovery:

```bash
codex plugin marketplace add NVIDIA/NeMo-Relay
codex plugin add nemo-relay-plugin@nemo-relay
```

That path relies on `nemo-relay` being available on `PATH`. Source plugin hooks
use `nemo-relay hook-forward codex --forward-only`: they post to the gateway
started by the required MCP entry but cannot launch or recover Relay without an
installer-owned generation fence. Before posting, they authenticate the Relay
identity and verify that its user-level configuration matches. The proof and
payload use one TCP connection, preventing a replacement listener from
receiving the payload after verification.

Use the source marketplace path for discovery or manifest validation. Use
`nemo-relay install codex` for complete provider routing, environment
forwarding, and verified plugin-hook trust.

Remove the source-installed Codex plugin before you use the generated install.
If both remain active and trusted, they can forward the same lifecycle payload.

Claude Code users can add this repository as a marketplace the same way:

```bash
claude plugin marketplace add NVIDIA/NeMo-Relay \
  --sparse .claude-plugin integrations/coding-agents/claude-code
claude plugin install nemo-relay-plugin@nemo-relay --scope user
```

That path reads `.claude-plugin/marketplace.json` from the repository. Source
plugin hooks use `nemo-relay hook-forward claude --forward-only`: they post to
the gateway started by the `alwaysLoad` MCP entry but cannot launch or recover
Relay without an installer-owned generation fence. They authenticate that
gateway on the same connection used to send lifecycle data. Use
`nemo-relay install claude-code` for the complete provider-routing setup, and
remove the source-installed plugin first to avoid duplicate lifecycle events.

Hermes persistent installation is user-level:

```bash
nemo-relay install hermes
```

It writes the MCP server and trusted hooks to `$HERMES_HOME/config.yaml` or
`~/.hermes/config.yaml`. Transparent Hermes runs leave that file untouched and
export the dynamic `NEMO_RELAY_GATEWAY_URL` through a process-private
`HERMES_HOME` overlay with no fixed MCP entry.

Shared TOML config is loaded from `/etc/nemo-relay/config.toml`, then nearest
project `.nemo-relay/config.toml`, then
`$XDG_CONFIG_HOME/nemo-relay/config.toml` or
`~/.config/nemo-relay/config.toml`.

That layering applies to transparent runs. Persistent mode skips the
project layer and merges only system and user configuration.

```toml
[agents.codex]
command = "codex"

[agents.hermes]
command = "hermes"
```

Observability exporters are configured in `plugins.toml`. Run
`nemo-relay plugins edit --project` to create `.nemo-relay/plugins.toml`, or
write the plugin config directly:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atif]
enabled = true
output_directory = ".nemo-relay/atif"

[components.config.openinference]
enabled = true
endpoint = "http://127.0.0.1:4318/v1/traces"
```

During setup or launch, Relay fails closed on invalid shared TOML, malformed
plugin config, unsupported exporter settings, or unavailable exporter features.
The wrapper does not start the coding agent with a configuration that it cannot
parse, validate, or activate. After the gateway and agent are running,
exporter delivery failures follow the observability plugin policy: application
work continues while the failing ATOF, ATIF, OpenTelemetry, or OpenInference
destination records, logs, or reports the failure.

## Hook Forwarding

Transparent Claude Code and Codex hooks call
`nemo-relay hook-forward <agent>` with the canonical hook payload on standard
input. The wrapper-owned command embeds the ephemeral per-run gateway URL and
is marked as transparent so it never starts or recovers the fixed gateway.

Persistent Claude Code, Codex, and Hermes hooks call
`nemo-relay hook-forward <agent>` with the fixed gateway and an
installer-owned generation fence. They wait for and authenticate the
MCP-owned gateway, then send the payload once. They never start or recover the
gateway. Transparent Hermes hooks instead embed the wrapper's dynamic gateway
URL.

For Codex, the installed plugin file is the sole persistent Relay hook source;
installation does not add Relay groups to `~/.codex/hooks.json`.

Since hook forwarding fails open by default, gateway or sidecar outages do not
block the coding agent. The hook command exits successfully after logging the
forwarding problem, so the host agent can continue even though that hook
payload can be missing from telemetry. For wrapper-generated `hook-forward`
commands, add `--fail-closed` when policy requires hook delivery to block the
agent. For generated persistent hooks, set `NEMO_RELAY_FAIL_CLOSED=1` in the hook
execution environment. In that mode, forwarding failures return a non-zero
hook command status to the host.

These `hook-forward` options control delivery and metadata:

- `--gateway-url <url>` selects the Relay gateway that receives the payload.
- `--forward-only` allows a source plugin or custom automation to use an
  existing compatible gateway without an installer-owned generation fence. It
  verifies the gateway but never launches or recovers Relay. Generated
  installed hooks use a private generation fence instead.
- `--session-metadata '<json>'` adds structured metadata to the agent begin
  event. For example, `--session-metadata '{"user_id":"alice"}'` exposes the
  string as `user.id` on OTLP trace roots.
- `--profile <name>` records a configuration profile in session metadata.
- `--gateway-mode hook-only|passthrough|required` records the expected gateway
  behavior in session metadata.
- `--fail-closed` returns a failure when delivery fails or Relay rejects the
  hook instead of allowing the coding agent to continue.

## LLM Gateway

Complete LLM lifecycle observability requires model traffic to pass through the
gateway. Hook-only mode observes agent, subagent, and tool lifecycle, but it
cannot observe provider request and response lifecycle when the coding agent
sends model traffic directly to an upstream provider or remote service.

The gateway exposes these passthrough routes:

- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/messages`
- `POST /v1/messages/count_tokens`
- `GET /v1/models`

Transparent runs configure provider routing automatically where the launched
agent supports local routing. Standalone gateway mode requires you to point the
agent's provider base URL at the gateway manually.

## Verify Export

Complete a coding-agent turn or session that uses one tool. Then confirm that
ATIF was written:

```bash
ls .nemo-relay/atif
```

The snapshot boundary depends on the host. Claude Code writes ATIF on
`SessionEnd`. Codex writes a cumulative snapshot on each `Stop` because its
plugin schema does not expose `SessionEnd`. Hermes writes or updates the
snapshot on `on_session_end`, `on_session_finalize`, or `on_session_reset`.

Run the opt-in host E2E targets when the corresponding CLI is installed. These
targets are intentionally outside `test-rust` and mandatory CI:

```bash
just test-claude-plugin-e2e
just test-codex-plugin-e2e
just test-hermes-mcp-e2e
```

Each target uses an isolated home directory and local mock provider. The Claude
and Hermes targets each run 10 cold sessions plus two concurrent sessions and
verify MCP connection, hook delivery, provider routing, session isolation,
balanced ATOF output, and final port release.
