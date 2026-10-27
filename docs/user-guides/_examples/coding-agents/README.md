<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Two coding-agent users, one managed Relay deployment

Alice uses Claude Code and Bob uses Codex CLI. Separate Linux accounts, homes,
Relay credentials, task repositories, MCP sessions, and workers share one system
daemon and administrator-owned plugin policy. This is a shared-host deployment;
it does not simulate separate workstation networks or enterprise SSO. The normal
provider credentials remain separate from Relay route credentials.

## Prepare a fresh host

Use an ordinary Brev Ubuntu VM or an equivalent disposable Linux host with
systemd, sudo, Git, and Python 3.11+. Install a verified Relay CLI binary that
includes `daemon managed-bundle`; use the version matching the docs. Install
supported Claude Code and Codex CLI executables so each account can run its
harness. Use the support matrix's minimum versions. Provider access must permit
gateway routing. Use Claude.ai subscription sign-in for Alice and ChatGPT
sign-in for Bob. API keys or approved gateway credentials are alternatives. Do not put provider credentials in the shared
configuration. A login on another machine or under another Linux user is not
automatically available to these accounts.

Start Phoenix on loopback port 6006 on this host, or change the template's OTLP
endpoint to an approved receiving tier. The template enables full payloads only
to inspect the synthetic email redaction: use this example with synthetic tasks.
For an authenticated exporter, supply its environment to each user worker and
add its `header_env` configuration before provisioning.

Review `config.toml`, `plugins.toml`, `nemo-relay.service`, and `deploy.sh`, then:

```bash
sudo bash deploy.sh /absolute/path/to/verified/nemo-relay /absolute/path/to/python3.12
```

The script refuses to replace existing accounts and deployment files. It installs
fixed administrator-owned paths, generates an immutable bundle for both
harnesses, creates fresh per-user credentials, and starts the daemon with
`--require-worker`. Incomplete worker startup therefore fails visibly. The
generated digest is outside the bundle. A partial setup failure needs inspection
or a fresh VM rather than blindly rerunning the script.

## Prepare Bob's managed Codex integration

The provisioning script creates an administrator-owned local marketplace beside
the immutable bundle, copies `codex/plugin-v1` including hidden files, and puts
the generated provider configuration and `hooks = true` in Bob's fresh Codex
configuration. As Bob, install the managed plugin:

```bash
codex plugin marketplace add /opt/nvidia/share/nemo-relay-codex-marketplace-v1
codex plugin add nemo-relay-managed-v1@nemo-relay-managed
```

In Codex, inspect `/plugins` and accept its hook trust/approval request only for
the administrator-installed `nemo-relay-managed-v1` handlers invoking
`/opt/nvidia/bin/nemo-relay`. Prompt wording depends on Codex version; do not
disable trust checks to get past an unexpected handler. See the
[managed Codex instructions](https://docs.nvidia.com/nemo/relay/daemon/configuration). `session.sh` supplies the matching
`NEMO_RELAY_CLIENT_TOKEN` for model headers; the plugin listing alone is not
acceptance evidence. Preserve existing authentication and unrelated settings
when adapting the example to an existing account.

## Run independent sessions

Open two terminals. Log into each account, authenticate its provider without
echoing or saving a key in source, then run the installed launcher:

```bash
# Terminal A
sudo -iu alice
claude auth login
claude auth status
/opt/nvidia/share/relay-user-guide/session.sh
```

```bash
# Terminal B
sudo -iu bob
codex login --device-auth
codex login status
/opt/nvidia/share/relay-user-guide/session.sh
```

Complete the device-login URL and code in your own browser; the account or
workspace must permit device login. If Bob is already signed in, reuse that
login. With the default OpenAI upstream and no replacement provider credential,
Relay recognizes ChatGPT OAuth and selects the native ChatGPT Codex backend.
Use `exit` to return from Bob's shell to the original administrator account
before running `sudo -iu alice`; Bob is a separate, non-administrator account.

For API-key authentication instead, run the following as Bob before starting
the launcher. This selects API billing rather than his ChatGPT subscription:

```bash
IFS= read -r -s -p 'OpenAI API key: ' OPENAI_API_KEY; printf '\n'
export OPENAI_API_KEY
printenv OPENAI_API_KEY | codex login --with-api-key
```

For Alice, select the Claude.ai account in browser login. Her subscription can
route through the managed `ANTHROPIC_BASE_URL`; Relay must preserve provider
authorization and the `anthropic-beta` OAuth capability. Leave
`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, and `apiKeyHelper` unset for this
subscription path. The launcher's custom Relay token header is separate.
This was verified with Claude Code 2.1.309 and Relay 0.11.0-rc.1; do not infer
compatibility for every older supported harness version.

For API-key authentication instead, supply Alice's `ANTHROPIC_API_KEY` through
your secret manager or hidden interactive input before launching. That selects
API billing instead of subscription usage. See
[Anthropic's subscription and gateway guidance](https://code.claude.com/docs/en/llm-gateway#subscriptions-and-gateways).

In both sessions, ask the agent to read
`/opt/nvidia/share/relay-user-guide/task.txt` and complete it in its current
repository. Keep both sessions open for the administrator checks. Approve only
the local file and test operations needed for this task. Alice's launcher loads
both generated Claude artifacts; Bob's installed managed plugin and provider
settings select the same daemon. Each harness starts MCP and its own worker.

## Accept the deployment

While both sessions are active, inspect:

```bash
curl -fsS http://127.0.0.1:47632/_nemo_relay/v1/workers/status
ps -eo user,pid,args | grep -E '[n]emo-relay daemon (mcp|worker)'
sudo journalctl -u nemo-relay.service -n 50 --no-pager
```

Require two distinct ready worker IDs and worker processes owned by Alice and
Bob. The status endpoint reports assignments, not a live worker probe or a
user directory. Correlate the sessions with their run times and lifecycle logs.
In Phoenix project `default`, inspect service `relay-managed-coding-agents`:
require the model/tool activity
supported by each harness and check that `sample.employee@example.com` is
replaced with `[REDACTED]` in the supported tool-result payload from reading
`task.txt`. Inspect that tool result before relying on the model's final message. The provider conversation and source
files still contain the original synthetic email. Neither a ready worker nor a
successful task proves delivery and redaction by itself.

Review each user's code and tests. Close Alice's harness, allow up to two minutes
for accepted requests to drain, watch `worker_drain_started` then `worker_stopped`
in lifecycle logs and recheck the worker status/process list,
and confirm Bob can continue with his existing worker. Workers for two sessions
of the same user may be reused; workers for these two users must be distinct.
Telemetry labels are not enterprise identity-provider enrollment or offboarding.

Preserve organization approval requirements and use interactive sessions when
commands or plugin hooks need approval. An authenticated noninteractive Codex
check produced real Relay model spans but could not execute the task's commands;
it still exited zero. Check the produced files, tests, and exported tool results
before accepting the deployment. Do not bypass approval or hook trust checks to
make the example pass.

### Measured interactive Codex result

Bob completed this task on Brev with ChatGPT sign-in, Codex 0.164.0, and Relay
0.11.0-rc.1. His three test methods and 21 independent acceptance cases passed.
One completed Phoenix trace contained one agent, one turn, six model calls, and
eight tool calls, with all parent spans present. The `task.txt` tool result and
final handoff were redacted; no original synthetic email appeared in the task's
exported attributes. Bob's worker remained ready while his session stayed open.
This result covers the interactive Codex path.

### Measured Claude subscription result

Alice completed the task with her existing Claude.ai Enterprise login through
the managed Relay route, using Claude Code 2.1.309 and Relay 0.11.0-rc.1.
Her initial six test methods passed but independent acceptance caught two Unicode
prefix errors (19/21 cases passed). After review feedback, Claude repaired the
implementation and added regression tests: seven test methods and all 21 cases
passed. The initial task trace contained three model and five tool spans; the
repair trace contained five model and five tool spans. Each was parented under
`claude-code-turn`. The `task.txt` Read result was redacted, and the original
synthetic email was absent from both traces' attributes.

These bounded `claude -p` runs retained explicit approvals for task files and
unittest. A first attempt stopped because the administrator's allow rule used a
single slash instead of Claude's `//` syntax for absolute paths. Correcting the
rule enabled the authorized read. A later shell-based rewrite was denied;
Claude used the approved Edit tool instead. No bypass-permissions mode was used.

Claude's expected root is a bounded `claude-code-turn` Custom scope, projected
as OpenInference `CHAIN`; the pinned source deliberately omits a durable Claude
session-level Agent scope. Codex instead groups turns under `codex-task`
(`AGENT`). Both measured Claude task traces had complete model/tool parentage
under their turn roots. Missing a session-level AGENT is not itself a capture
failure.

Cold starts emitted separate `mark:agent_end_without_start` diagnostics.
Initializing MCP first preserved a `session.start` mark. The exported Claude
turn output was `{"content":[]}` despite a native final handoff, so complete
assistant-transcript capture remains unverified. Tool-result redaction is
verified separately above.

After Alice exited, her workers were removed and Bob's original worker remained
ready at that check. A later standalone Bob request succeeded, but the original
worker had already ended before that probe; it does not establish continuity
in the earlier interactive session.

## Apply the files elsewhere

The service, policy, bundle, and user-session boundaries are ordinary Relay
deployment artifacts. Brev only supplies the Linux host. Other supported
harnesses and local or remote daemon topologies are possible. For multiple
workstations, distribute policy and plugin files to every client: workers run
on those clients, and a remote daemon needs a trusted return route to them.
Use organization-managed harness policy for enforcement; Claude's `--settings`
and this reviewable launcher alone do not make its settings immutable.

The example creates disposable accounts and a system service. Use the daemon's
Linux removal runbook or delete the disposable VM after review. Keep credentials
out of screenshots and shared terminal output.


## Exact prerequisites and Brev access

Use Relay CLI 0.11.0-rc.1, Claude Code >=2.1.121, and Codex CLI >=0.143.0 for
this recipe. Install Relay from the
[installation page](https://docs.nvidia.com/nemo/relay/getting-started/installation),
Claude from its [setup instructions](https://code.claude.com/docs/en/setup), and
Codex from its [CLI instructions](https://developers.openai.com/codex/cli).
Make the harness executables visible to fresh Linux login shells (for example,
a root-owned `/usr/local/bin` installation); an administrator's private npm or
home directory is not sufficient. Check `sudo -iu alice command -v claude` and
`sudo -iu bob command -v codex` after provisioning. Avoid root-owned files in
user homes when installing harnesses.

Start Phoenix in an administrator terminal with
`uvx --from arize-phoenix phoenix serve`. Use a disposable Ubuntu systemd VM.
On Brev, connect with `brev shell YOUR_INSTANCE` and verify PID 1 is systemd.
If this opens a container, use `--host` when that host route is reachable.
Forward the UI with `brev port-forward YOUR_INSTANCE --port 6006:6006`, adding
`--host` only if Phoenix uses that same host route. The default proxy connection
can work even when the direct `--host` alias times out.

The worker status JSON supplies `worker_id`, `state`, `reference_count`,
`control_available`, and `in_flight`. It does not expose a username map. Correlate
two ready assignments with Alice/Bob process ownership and lifecycle timestamps;
never publish the client-token files. A process or status row alone is not a
provider-routing or redaction check.

## Cleanup

Close both harnesses, wait for worker shutdown, and follow
[Remove a Deployment](https://docs.nvidia.com/nemo/relay/daemon/operations#remove-a-deployment)
and the [Linux systemd runbook](https://docs.nvidia.com/nemo/relay/daemon/linux).
Remove only this example's service, managed bundle, helper/marketplace paths,
and disposable accounts after retaining needed results. Check ownership before
removing anything on a reused host; deleting the disposable VM is simpler.

The pinned Relay version is a release candidate: use `NEMO_RELAY_VERSION=0.11.0-rc.1`
with the official installer. Stable `0.11.0` assets are not available yet.

On Ubuntu images that default to Python 3.10, pass an explicit Python 3.11+
interpreter as the second argument to `deploy.sh`. For uv-managed Python, run
`uv python install 3.12` and use the path from `uv python find 3.12`. The script
validates `tomllib` before changing deployment files; sudo may use a different
PATH from your normal shell.


### Authenticated worker-isolation check

A controlled follow-up held two independent MCP sessions and required two ready
workers before making actual subscription-authenticated Claude and Codex calls.
After only Alice's session closed, her worker disappeared from status. Bob's
next exact-response request succeeded through the same remaining worker ID.
This verifies worker isolation and continued model routing with provider logins;
it does not claim that input was sent to the earlier interactive Codex UI.
The earlier standalone Bob probe and the initially failed lifecycle harness
remain separate validation records.
