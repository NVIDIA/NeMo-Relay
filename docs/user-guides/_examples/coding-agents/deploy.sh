#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Provision a NEW disposable Linux host. Existing deployments need a planned merge.
set -euo pipefail
[[ $(id -u) == 0 ]] || { echo "Run as an administrator on a fresh example VM." >&2; exit 1; }
[[ $# -ge 1 && $# -le 2 && -x $1 ]] || {
    echo "Usage: sudo bash deploy.sh /absolute/path/to/nemo-relay [python3.11-or-newer]" >&2
    exit 1
}
python_command=${2:-python3}
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
for tool in systemctl useradd runuser "$python_command" git install chmod cp cat; do
    command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 1; }
done
"$python_command" -c 'import tomllib' || { echo "Python 3.11+ is required." >&2; exit 1; }
[[ -d /run/systemd/system ]] || { echo "This example requires a Linux systemd host." >&2; exit 1; }
for path in /etc/nemo-relay /opt/nvidia/bin/nemo-relay /opt/nvidia/share/nemo-relay-managed-v1 \
    /opt/nvidia/share/nemo-relay-codex-marketplace-v1 \
    /opt/nvidia/share/relay-user-guide /etc/systemd/system/nemo-relay.service; do
    [[ ! -e $path ]] || { echo "Refusing to replace $path" >&2; exit 1; }
done
for account in alice bob nemo-relay; do
    ! id "$account" &>/dev/null || { echo "Account $account already exists; use a fresh VM." >&2; exit 1; }
done
"$1" daemon managed-bundle --help >/dev/null

install -d -m 0755 /opt/nvidia/bin /opt/nvidia/share /etc/nemo-relay
install -m 0755 "$1" /opt/nvidia/bin/nemo-relay
install -m 0644 "$source_dir/config.toml" "$source_dir/plugins.toml" /etc/nemo-relay/
useradd --system --user-group --home-dir /var/lib/nemo-relay --create-home --shell /usr/sbin/nologin nemo-relay
chmod 700 /var/lib/nemo-relay
install -d -o nemo-relay -g nemo-relay -m 0700 /var/lib/nemo-relay/.config
/opt/nvidia/bin/nemo-relay daemon managed-bundle \
    --output /opt/nvidia/share/nemo-relay-managed-v1 \
    --daemon-address http://127.0.0.1:47632 --dispatcher-command /opt/nvidia/bin/nemo-relay \
    --platform linux --agent claude --agent codex > /etc/nemo-relay/managed-bundle.sha256
chmod -R a+rX,go-w /opt/nvidia/share/nemo-relay-managed-v1
marketplace=/opt/nvidia/share/nemo-relay-codex-marketplace-v1
install -d -m 0755 "$marketplace/.agents/plugins" "$marketplace/plugins/nemo-relay-managed-v1"
install -m 0644 "$source_dir/marketplace.json" "$marketplace/.agents/plugins/marketplace.json"
cp -R /opt/nvidia/share/nemo-relay-managed-v1/codex/plugin-v1/. "$marketplace/plugins/nemo-relay-managed-v1/"
chmod -R a+rX,go-w "$marketplace"
install -d -m 0755 /opt/nvidia/share/relay-user-guide
install -m 0755 "$source_dir/session.sh" /opt/nvidia/share/relay-user-guide/
install -m 0644 "$source_dir/task.txt" /opt/nvidia/share/relay-user-guide/
for account in alice bob; do
    useradd --create-home --user-group --shell /bin/bash "$account"
    chmod 700 "/home/$account"
    runuser -l "$account" -c '/opt/nvidia/bin/nemo-relay daemon token ensure --json' >/dev/null
    runuser -l "$account" -c 'mkdir "$HOME/ticket-service"; cd "$HOME/ticket-service"; git init -q'
done
install -d -o bob -g bob -m 0700 /home/bob/.codex
install -o bob -g bob -m 0600 /opt/nvidia/share/nemo-relay-managed-v1/codex/settings-v1/config.toml /home/bob/.codex/config.toml
# The current generator omits [features]. Refuse future conflicting output before
# changing the file instead of silently generating duplicate TOML tables.
"$python_command" - <<'CONFIG'
from pathlib import Path
import tomllib
path = Path("/home/bob/.codex/config.toml")
text = path.read_text()
if "features" in tomllib.loads(text):
    raise SystemExit("Generated Codex config already has [features]; merge hooks=true before continuing.")
text += "\n[features]\nhooks = true\n"
tomllib.loads(text)
path.write_text(text)
CONFIG
runuser -u nemo-relay -- env XDG_CONFIG_HOME=/var/lib/nemo-relay/.config \
    /opt/nvidia/bin/nemo-relay daemon tls-certificate > /etc/nemo-relay/hook-tls-certificate.pem
chmod 644 /etc/nemo-relay/hook-tls-certificate.pem
install -m 0644 "$source_dir/nemo-relay.service" /etc/systemd/system/nemo-relay.service
systemctl daemon-reload
systemctl enable --now nemo-relay.service
echo "Shared daemon installed. Complete each harness's authentication and managed setup in README.md."
