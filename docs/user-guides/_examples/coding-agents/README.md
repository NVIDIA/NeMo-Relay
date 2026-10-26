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
gateway routing; use an Anthropic API key for Alice and an OpenAI API key for
Bob in the first example. A paid harness subscription alone does not establish
that API access. Do not put either key in the shared configuration.

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
IFS= read -r -s -p 'Anthropic API key: ' ANTHROPIC_API_KEY; printf '\n'
export ANTHROPIC_API_KEY
/opt/nvidia/share/relay-user-guide/session.sh
```

```bash
# Terminal B
sudo -iu bob
IFS= read -r -s -p 'OpenAI API key: ' OPENAI_API_KEY; printf '\n'
export OPENAI_API_KEY
printenv OPENAI_API_KEY | codex login --with-api-key
/opt/nvidia/share/relay-user-guide/session.sh
```

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
In Phoenix, inspect `relay-managed-coding-agents`: require the model/tool activity
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
