#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="$example_root/scripts/run_phase2_from_env.sh"

if docker info >/dev/null 2>&1; then
  exec "$runner"
fi

user_name="$(id -un)"
if command -v sg >/dev/null && id -nG "$user_name" | tr ' ' '\n' | grep -Fxq docker; then
  export PHASE2_DOCKER_GROUP_RUNNER="$runner"
  exec sg docker -c 'exec "$PHASE2_DOCKER_GROUP_RUNNER"'
fi

echo "Docker is not accessible from the detached tmux process." >&2
echo "Add $user_name to the docker group and start a new login session before retrying." >&2
exit 2
