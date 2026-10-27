<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Two independent users with one managed coding harness

Choose **Claude Code for both users** or **Codex for both users**.
Relay can serve mixed supported harnesses; using one here reduces setup.
Each user signs into their own provider session on the host.

## Prepare a fresh host

Use Ubuntu with systemd, sudo, Git, curl, Python 3.12, and Relay 0.10.4.
Install the chosen harness where both login shells can find it.
A CPU VM is sufficient with remote inference.

```bash
sudo bash deploy.sh "$(command -v nemo-relay)" claude "$(command -v python3.12)"
# Or choose codex instead of claude.
```

The script refuses existing users and deployment paths. It creates Alice, Bob,
and a dedicated service user; installs the selected immutable bundle under
`/opt/nvidia/share`; creates private per-user tokens and task repositories;
and installs `/etc/nemo-relay` policy and a systemd service on loopback 47632.
Codex mode also prepares both users' managed settings and a local marketplace.

Use a separate VM from other examples: system plugin policy affects other Relay
processes on the host. See [removal](https://docs.nvidia.com/nemo/relay/daemon/operations#remove-a-deployment).

## Optional synthetic payload inspection

The installed default `plugins.toml` has `enable_full_payloads = false`.
To inspect redaction using only the supplied synthetic task, opt in before
starting either session:

```bash
sudo install -m 0644 plugins.synthetic.toml /etc/nemo-relay/plugins.toml
sudo systemctl restart nemo-relay.service
```

Start disposable Phoenix with `uvx --from arize-phoenix phoenix serve`.
For Brev, forward its UI using
`brev port-forward YOUR_INSTANCE --port 6006:6006`.

## Start each user

Use separate administrator terminals: `sudo -iu alice` and `sudo -iu bob`.
Run the same chosen setup inside both accounts.

### Claude Code

```bash
claude auth login
claude auth status
/opt/nvidia/share/relay-user-guide/session.sh
```

A subscription login and API authentication are separate choices. Keep provider
overrides unset when preserving a subscription login. The launcher adds only
the Relay routing header. Follow
[Anthropic's gateway requirements](https://code.claude.com/docs/en/llm-gateway#subscriptions-and-gateways)
and your organization's provider policy.

### Codex

```bash
codex plugin marketplace add /opt/nvidia/share/nemo-relay-codex-marketplace-v1
codex plugin add nemo-relay-managed-v1@nemo-relay-managed
codex login --device-auth
codex login status
/opt/nvidia/share/relay-user-guide/session.sh
```

Use an approved existing login or API key if device login is unavailable.
Review plugin/hook trust prompts and retain command approvals.

Ask the agent to read `/opt/nvidia/share/relay-user-guide/task.txt` and complete
it in the current repository. Keep both sessions open.

## Acceptance

- Two distinct ready worker IDs; processes owned separately by Alice and Bob.
- Each repository contains a correct implementation and passing tests.
- Supported model/tool spans correlate with each task or turn.
- Under the synthetic policy, task payloads contain `[REDACTED]` instead of the contact.
- Close Alice; after worker shutdown Bob can complete another captured request.

Use worker status, process ownership, operational logs, and run times to map
sessions to users. Trace UUIDs identify scopes, not employee directory identities.
Provider success or exit zero does not prove that tools executed. Run the installed
`verify_task.py` against each task repository; it checks 21 independent cases.

```bash
curl -fsS http://127.0.0.1:47632/_nemo_relay/v1/workers/status
ps -eo user,pid,args | grep -E '[n]emo-relay daemon (mcp|worker)'
sudo journalctl -u nemo-relay.service -n 50 --no-pager
```

Tokens are private per-user routing credentials, distinct from provider logins.
Removing a token file does not revoke an active daemon binding; follow the
[credential lifecycle](https://docs.nvidia.com/nemo/relay/daemon/configuration#replace-a-client-token).

## Cleanup

Close both harnesses and retain evidence. Restore metadata-only capture before
using real data. Follow the removal runbook or delete the disposable VM.
Stop only a Phoenix process you started for this example.
