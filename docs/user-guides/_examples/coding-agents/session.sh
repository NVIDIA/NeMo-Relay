#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
account=$(id -un)
[[ $account == alice || $account == bob ]] || { echo "Log in as alice or bob first." >&2; exit 1; }
bundle=/opt/nvidia/share/nemo-relay-managed-v1
digest=$(cat /etc/nemo-relay/managed-bundle.sha256)
/opt/nvidia/bin/nemo-relay doctor --managed-bundle "$bundle" --managed-bundle-sha256 "$digest"
# Read the token in the process that starts the harness; never place it in argv or logs.
export NEMO_RELAY_CLIENT_TOKEN
NEMO_RELAY_CLIENT_TOKEN=$(cat "$HOME/.config/nemo-relay/.client-token")
cd "$HOME/ticket-service"
if [[ $account == alice ]]; then
    export ANTHROPIC_CUSTOM_HEADERS="x-nemo-relay-client-token: $NEMO_RELAY_CLIENT_TOKEN"
    exec claude --plugin-dir "$bundle/claude-code/plugin-v1" \
        --settings "$bundle/claude-code/settings-v1/managed-settings.json"
else
    exec codex
fi
