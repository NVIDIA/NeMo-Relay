#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared process and readiness helpers for the manually-invoked Switchyard E2E scripts.

e2e_pids=()

e2e_add_pid() {
  e2e_pids+=("$1")
}

e2e_stop_tree() {
  local pid="$1"
  local child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    e2e_stop_tree "$child"
  done
  kill "$pid" 2>/dev/null || true
}

e2e_stop_processes() {
  local pid
  for pid in "${e2e_pids[@]}"; do
    e2e_stop_tree "$pid"
  done
  for pid in "${e2e_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  e2e_pids=()
}

e2e_wait_for() {
  local url="$1"
  local attempts="${2:-120}"
  local delay="${3:-0.25}"
  local attempt
  for attempt in $(seq 1 "$attempts"); do
    if curl --fail --silent "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done
  echo "timed out waiting for $url" >&2
  return 1
}

e2e_random_token() {
  python3 -c 'import secrets; print(secrets.token_hex(24))'
}

# The experimental Relay integration is tested against a specific public
# Switchyard topic-branch commit. Override this only when intentionally testing
# another revision.
e2e_verify_switchyard_checkout() {
  local root="$1"
  local expected="${2:-8f9db9a6a47f848cdff1d262276ba25a8ae9cbc8}"
  local actual
  actual="$(git -C "$root" rev-parse HEAD 2>/dev/null)" || {
    echo "Switchyard worktree is not a git checkout: $root" >&2
    return 1
  }
  if [[ "$actual" != "$expected" ]]; then
    echo "Switchyard checkout mismatch: expected $expected, found $actual" >&2
    echo "Set SWITCHYARD_EXPECTED_COMMIT only when intentionally testing another revision." >&2
    return 1
  fi
  printf 'Switchyard revision: %s (%s)\n' "$actual" "$(git -C "$root" show -s --format=%s HEAD)"
}

e2e_tail_logs() {
  local directory="$1"
  local log
  for log in "$directory"/*.log; do
    [[ -f "$log" ]] || continue
    echo "--- $log" >&2
    tail -100 "$log" >&2
  done
}
