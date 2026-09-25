#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
set +x

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${TERMINAL_BENCH_ENV_FILE:-${1:-}}"
if [[ -z "$env_file" || ! -f "$env_file" ]]; then
  echo "TERMINAL_BENCH_ENV_FILE must reference an existing environment file" >&2
  exit 2
fi
env_file="$(cd "$(dirname "$env_file")" && pwd)/$(basename "$env_file")"

"$example_root/scripts/validate_phase2_environment.sh" "$env_file"
set -a
# shellcheck disable=SC1090
source "$env_file"
set +a
set +x

mkdir -p "$TERMINAL_BENCH_RUN_ROOT"
chmod 0700 "$TERMINAL_BENCH_RUN_ROOT"
runtime_harness="$TERMINAL_BENCH_RUN_ROOT/runtime-harness"
"$EVAL_PYTHON" "$example_root/scripts/stage_phase2_runtime.py" \
  --source "$example_root" \
  --destination "$runtime_harness" \
  --plan "$TERMINAL_BENCH_RUN_ROOT/plan.json"
exec "$runtime_harness/supervise_phase2_cohort.sh" "$TERMINAL_BENCH_RUN_ROOT" \
  >>"$TERMINAL_BENCH_RUN_ROOT/supervisor.log" 2>&1
