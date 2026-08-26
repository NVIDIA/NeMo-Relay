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

runtime_harness="$TERMINAL_BENCH_RUN_ROOT/runtime-harness"
snapshot="$runtime_harness/snapshot.json"
plan="$TERMINAL_BENCH_RUN_ROOT/plan.json"
staged_runner="$runtime_harness/supervise_phase2_cohort.sh"
for required in "$snapshot" "$plan" "$staged_runner"; do
  [[ -f "$required" ]] || {
    echo "immutable staged runtime input is missing: $required" >&2
    exit 2
  }
done

"$EVAL_PYTHON" - "$snapshot" "$plan" <<'PY'
import json
import pathlib
import sys

snapshot = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
plan = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
staged_digest = snapshot.get("runtime_sources_sha256")
planned_digest = plan.get("inputs", {}).get("runtime_sources_sha256")
if not staged_digest or staged_digest != planned_digest:
    raise SystemExit("staged runtime digest does not match the immutable plan")
PY

if docker info >/dev/null 2>&1; then
  exec "$staged_runner" "$TERMINAL_BENCH_RUN_ROOT" >>"$TERMINAL_BENCH_RUN_ROOT/supervisor.log" 2>&1
fi

user_name="$(id -un)"
if command -v sg >/dev/null && id -nG "$user_name" | tr ' ' '\n' | grep -Fxq docker; then
  export PHASE2_STAGED_RUNNER="$staged_runner"
  exec sg docker -c \
    'exec "$PHASE2_STAGED_RUNNER" "$TERMINAL_BENCH_RUN_ROOT" >>"$TERMINAL_BENCH_RUN_ROOT/supervisor.log" 2>&1'
fi

echo "Docker is not accessible from the detached tmux process." >&2
echo "Add $user_name to the docker group and start a new login session before retrying." >&2
exit 2
