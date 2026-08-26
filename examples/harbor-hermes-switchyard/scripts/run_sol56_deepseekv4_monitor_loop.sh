#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
status_path="/localhome/local-bbednarski/terminal-bench-artifacts/monitoring/tb21-sol56-deepseekv4-stage03-pr270-5c84c16/status.json"
interval_seconds="${SOL56_MONITOR_INTERVAL_SECONDS:-600}"
[[ "$interval_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo "SOL56_MONITOR_INTERVAL_SECONDS must be a positive integer" >&2
  exit 2
}

while true; do
  cycle_started_at="$(date +%s)"
  "$example_root/scripts/monitor_sol56_deepseekv4_runs.sh"
  if [[ -f "$status_path" ]] && [[ "$(jq -r '.all_complete' "$status_path")" == "true" ]]; then
    exit 0
  fi
  cycle_elapsed_seconds=$(( $(date +%s) - cycle_started_at ))
  cycle_sleep_seconds=$(( interval_seconds - cycle_elapsed_seconds ))
  if (( cycle_sleep_seconds > 0 )); then
    sleep "$cycle_sleep_seconds"
  fi
done
