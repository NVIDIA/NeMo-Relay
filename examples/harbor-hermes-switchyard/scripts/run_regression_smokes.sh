#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
regression_root="${1:-}"

if [[ -z "$regression_root" || "$regression_root" != /* ]]; then
  echo "usage: $0 /absolute/new-regression-root" >&2
  exit 2
fi
if [[ -e "$regression_root" ]]; then
  echo "regression root already exists: $regression_root" >&2
  exit 2
fi
mkdir -m 0700 "$regression_root"

tasks=(
  adaptive-rejection-sampler
  circuit-fibsqrt
  gpt2-codegolf
  overfull-hbox
)

for task in "${tasks[@]}"; do
  echo "Running regression smoke: $task"
  run_root="$regression_root/$task"
  inject=false
  if [[ "$task" == "circuit-fibsqrt" ]]; then
    inject=true
  fi
  TASK_NAME="$task" \
    PHOENIX_PROJECT="${PHOENIX_PROJECT:-harbor-hermes-switchyard-regression}-$task" \
    EVAL_COHORT="${EVAL_COHORT:-harbor-hermes-switchyard-regression}-$task" \
    EVAL_PHASE="regression" \
    INJECT_POST_RESPONSE_FAILURE="$inject" \
    "$example_root/run_terminal_bench.sh" "$run_root"
done

python_bin="${EVAL_PYTHON:-python3}"
"$python_bin" - "$regression_root" "${tasks[@]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
tasks = sys.argv[2:]
summaries = []
for task in tasks:
    summary_path = root / task / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "passed":
        raise SystemExit(f"regression did not pass: {task}")
    summaries.append(summary)

result = {
    "schema_version": "harbor-hermes-switchyard.regression-smokes.v1",
    "status": "passed",
    "planned": len(tasks),
    "completed": len(summaries),
    "tasks": tasks,
}
(root / "summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(result, indent=2))
PY
