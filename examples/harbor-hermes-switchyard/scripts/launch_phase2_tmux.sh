#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$#" -eq 1 ]]; then
  env_file="$example_root/.env"
  session="$1"
elif [[ "$#" -eq 2 ]]; then
  env_file="$1"
  session="$2"
else
  echo "usage: $0 [env-file] tmux-session-name" >&2
  exit 2
fi
if [[ ! -f "$env_file" ]]; then
  echo "Phase 2 environment file does not exist: $env_file" >&2
  exit 2
fi
env_file="$(cd "$(dirname "$env_file")" && pwd)/$(basename "$env_file")"
if [[ ! "$session" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "tmux session name may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi
command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 2; }
"$example_root/scripts/validate_phase2_environment.sh" "$env_file"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 21
fi

# The caller must not source the protected file. Only its path is projected
# into the detached session; run_phase2_from_env.sh sources it with xtrace off.
tmux new-session -d -s "$session" \
  -e "TERMINAL_BENCH_ENV_FILE=$env_file" \
  "$example_root/scripts/run_phase2_from_env.sh"
echo "started tmux session: $session"
