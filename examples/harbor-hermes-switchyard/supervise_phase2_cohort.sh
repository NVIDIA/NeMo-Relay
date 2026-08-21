#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_root="${1:-}"
if [[ -z "$run_root" || "$run_root" != /* ]]; then
  echo "usage: $0 /absolute/phase2-run-root [coordinator options]" >&2
  exit 2
fi
shift

child_pid=""
terminate_group() {
  if [[ -z "$child_pid" ]]; then
    return
  fi
  kill -TERM -- "-$child_pid" >/dev/null 2>&1 || true
  for _ in {1..10}; do
    if ! kill -0 -- "-$child_pid" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  kill -KILL -- "-$child_pid" >/dev/null 2>&1 || true
}
terminate() {
  terminate_group
  [[ -z "$child_pid" ]] || wait "$child_pid" >/dev/null 2>&1 || true
  exit 143
}
trap terminate INT TERM

backoff_seconds="${TERMINAL_BENCH_SUPERVISOR_BACKOFF_SECONDS:-60}"
maximum_backoff_seconds="${TERMINAL_BENCH_SUPERVISOR_MAX_BACKOFF_SECONDS:-900}"
if [[ ! "$backoff_seconds" =~ ^[1-9][0-9]*$ || ! "$maximum_backoff_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "Phase 2 supervisor backoffs must be positive integers" >&2
  exit 2
fi

while true; do
  python_bin="${EVAL_PYTHON:-$example_root/.venv/bin/python}"
  "$python_bin" "$example_root/scripts/exec_process_group.py" \
    "$example_root/run_phase2_cohort.sh" "$run_root" "$@" &
  child_pid=$!
  wait "$child_pid"
  status=$?
  terminate_group
  child_pid=""
  case "$status" in
    0)
      exit 0
      ;;
    20)
      echo "[phase2-supervisor] stopping on preserved harness/integration blocker" >&2
      exit 20
      ;;
    *)
      echo "[phase2-supervisor] coordinator exited $status; resuming in ${backoff_seconds}s" >&2
      sleep "$backoff_seconds"
      if ((backoff_seconds < maximum_backoff_seconds)); then
        backoff_seconds=$((backoff_seconds * 2))
        if ((backoff_seconds > maximum_backoff_seconds)); then
          backoff_seconds=$maximum_backoff_seconds
        fi
      fi
      ;;
  esac
done
