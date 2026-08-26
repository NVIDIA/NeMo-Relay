#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
set +x

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 ENV_FILE RUN_ROOT" >&2
  exit 2
fi
env_file="$(realpath "$1")"
run_root="$(realpath "$2")"
[[ -f "$env_file" && -f "$run_root/plan.json" && -f "$run_root/summary.json" ]] || exit 2

set -a
source "$env_file"
set +a
: "${EVAL_PYTHON:?EVAL_PYTHON is required}"
: "${PHOENIX_BASE_URL:?PHOENIX_BASE_URL is required}"
: "${PHOENIX_PROJECT:?PHOENIX_PROJECT is required}"
: "${SWITCHYARD_PROVIDER_AUTHORIZATION:?SWITCHYARD_PROVIDER_AUTHORIZATION is required}"

mapfile -t blockers < <(
  find "$run_root/tasks" -mindepth 2 -maxdepth 2 -name task-state.json -type f -print0 \
    | xargs -0 -r -n1 jq -er 'select(.status == "failed" and .failure_class == "harness_or_integration") | input_filename'
)
for state_path in "${blockers[@]}"; do
  task_root="$(dirname "$state_path")"
  attempt_name="$(jq -er '.latest_attempt' "$state_path")"
  attempt="$task_root/attempts/$attempt_name"
  [[ -d "$attempt" ]] || { echo "missing preserved attempt: $attempt" >&2; exit 2; }
  mapfile -t artifacts < <(find "$attempt/jobs" -path '*/artifacts/logs/agent/direct-hermes' -type d)
  mapfile -t jobs < <(find "$attempt/jobs" -mindepth 1 -maxdepth 1 -type d)
  if (( ${#artifacts[@]} != 1 || ${#jobs[@]} != 1 )); then
    echo "could not uniquely resolve artifact/job roots for $attempt" >&2
    exit 2
  fi
  artifact="${artifacts[0]}"
  job_dir="${jobs[0]}"
  candidate="$artifact/validation.reconciliation-candidate.json"
  upload="$artifact/phoenix-upload.reconciliation.json"
  "$EVAL_PYTHON" "$example_root/scripts/validate_run.py" \
    --artifacts "$artifact" \
    --provenance "$attempt/runtime/provenance.json" \
    --openinference "$attempt/telemetry/trajectory.openinference.json" \
    --harbor-job-dir "$job_dir" \
    --scan-root "$job_dir" \
    --secret-env SWITCHYARD_PROVIDER_AUTHORIZATION \
    --output "$candidate" >"$attempt/validation.reconciliation-candidate.log"
  jq -e '
    .status == "passed"
    and .benchmark.status == "passed"
    and .integration.status == "passed"
    and (.benchmark_task_passed | type == "boolean")
    and (
      (.terminal_agent_timeout_completion == true)
      or (.terminal_turn_budget_completion == true)
      or (.terminal_quiet_output_completion == true)
    )
  ' "$candidate" >/dev/null
  if ! jq -e '.status == "passed"' "$upload" >/dev/null 2>&1; then
    "$EVAL_PYTHON" "$example_root/scripts/upload_openinference.py" \
      --openinference "$attempt/telemetry/trajectory.openinference.json" \
      --phoenix-url "$PHOENIX_BASE_URL" \
      --project "$PHOENIX_PROJECT" \
      --output "$upload" >"$attempt/phoenix-upload.reconciliation.log"
  fi
  "$EVAL_PYTHON" "$example_root/scripts/reconcile_completed_nonpass.py" \
    --attempt-root "$attempt" \
    --validation-candidate "$candidate" \
    --phoenix-upload "$upload" \
    --validator "$example_root/scripts/validate_run.py" \
    >"$attempt/reconciliation.log"
  echo "reconciled verifier-backed terminal completion: $(basename "$task_root") attempt $attempt_name"
done
