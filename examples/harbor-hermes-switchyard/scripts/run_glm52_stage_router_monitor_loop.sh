#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
status_path="/localhome/local-bbednarski/terminal-bench-artifacts/monitoring/tb21-sol56-glm52-stage-ef05/status.json"
interval_seconds="${GLM52_STAGE_MONITOR_INTERVAL_SECONDS:-600}"
[[ "$interval_seconds" =~ ^[1-9][0-9]*$ ]] || exit 2

while true; do
  started="$(date +%s)"
  "$example_root/scripts/monitor_glm52_stage_router_campaign.sh"
  if [[ -f "$status_path" ]] \
    && jq -e '.signal_complete and .classifier_complete' "$status_path" >/dev/null \
    && [[ -f "$example_root/reports/tb21-glm52-stage-ef05-signal-vs-classifier/README.pdf" ]]; then
    exit 0
  fi
  elapsed=$(( $(date +%s) - started ))
  remaining=$(( interval_seconds - elapsed ))
  (( remaining > 0 )) && sleep "$remaining"
done
