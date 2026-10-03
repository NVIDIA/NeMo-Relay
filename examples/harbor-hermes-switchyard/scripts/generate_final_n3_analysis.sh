#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
set +x

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${1:-$example_root/config/final_analysis_runs.json}"
target="${2:-$example_root/reports/tb21-final-n3-opus48-baseline-v6}"
manifest="$(realpath -- "$manifest")"
target="$(realpath -m -- "$target")"
lock_path="${target}.lock"

mkdir -p "$(dirname "$target")"
exec 9>"$lock_path"
if ! flock -n 9; then
  exit 0
fi
if [[ -f "$target/aggregate-metrics.json" \
  && -f "$target/README.pdf" \
  && -f "$target/evidence/report-validation.json" \
  && -f "$target/evidence/pdf-validation.json" ]] \
  && jq -e '.status == "passed"' "$target/evidence/report-validation.json" >/dev/null \
  && jq -e '.status == "passed"' "$target/evidence/pdf-validation.json" >/dev/null; then
  exit 0
fi
if [[ -e "$target" ]]; then
  printf 'analysis target exists but is incomplete: %s\n' "$target" >&2
  exit 2
fi

dataset="$(jq -er '.dataset' "$manifest")"
baseline_group="$(jq -er '.baseline_group' "$manifest")"
sample_size="$(jq -er '.sample_size_per_group' "$manifest")"
group_count="$(jq -er '.groups | length' "$manifest")"
report_title="$(jq -er '.title // "Terminal-Bench 2.1 router comparison (N=3)"' "$manifest")"
analysis_request="$(jq -er '.analysis_request // "Compare independent repeated router configurations against the declared observed baseline."' "$manifest")"
output_placeholder="__TB_REPORT_OUTPUT_DIR__"
command=(
  "$example_root/.venv/bin/python" "$example_root/terminal-bench-report/scripts/analyze.py"
  --output-dir "$output_placeholder"
  --mode compare
  --baseline-group "$baseline_group"
  --expected-group-size "$sample_size"
  --group-compatibility-manifest "$manifest"
  --title "$report_title"
  --analysis-request "$analysis_request"
)

for ((group_index = 0; group_index < group_count; group_index++)); do
  group="$(jq -er ".groups[$group_index].id" "$manifest")"
  mapfile -t roots < <(jq -er ".groups[$group_index].run_roots[]" "$manifest")
  if (( ${#roots[@]} != sample_size )); then
    printf 'group %s has %s roots; expected %s\n' "$group" "${#roots[@]}" "$sample_size" >&2
    exit 2
  fi
  for root_index in "${!roots[@]}"; do
    root="${roots[$root_index]}"
    summary="$root/summary.json"
    plan="$root/plan.json"
    if ! jq -e '(.status == "passed") and (.planned_tasks == 89) and (.completed_tasks == 89) and ((.benchmark_pass_count + .benchmark_nonpass_count) == 89)' "$summary" >/dev/null; then
      printf 'run is not admitted as a complete 89-task result: %s\n' "$root" >&2
      exit 2
    fi
    if [[ "$(jq -er '.dataset' "$plan")" != "$dataset" ]]; then
      printf 'run dataset does not match analysis manifest: %s\n' "$root" >&2
      exit 2
    fi
    label="$group-r$((root_index + 1))"
    command+=(--run-root "$root" --label "$label" --group "$group")
  done
done

staging_parent="$(mktemp -d "$(dirname "$target")/.tb21-final-n3-building.XXXXXX")"
staging="$staging_parent/bundle"
output_replaced=false
for command_index in "${!command[@]}"; do
  if [[ "${command[$command_index]}" == "$output_placeholder" ]]; then
    command[$command_index]="$staging"
    output_replaced=true
    break
  fi
done
if [[ "$output_replaced" != true ]]; then
  rmdir "$staging_parent"
  printf 'internal error: analysis output placeholder was not found\n' >&2
  exit 2
fi
if ! "${command[@]}"; then
  failed="$(dirname "$target")/tb21-final-n3-failed-$(date -u +'%Y%m%dT%H%M%SZ')"
  mv "$staging_parent" "$failed"
  printf 'analysis generation failed; preserved staging evidence at %s\n' "$failed" >&2
  exit 2
fi
if ! jq -e '.status == "passed"' "$staging/evidence/report-validation.json" >/dev/null; then
  failed="$(dirname "$target")/tb21-final-n3-failed-$(date -u +'%Y%m%dT%H%M%SZ')"
  mv "$staging_parent" "$failed"
  printf 'analysis validation failed; preserved bundle at %s\n' "$failed" >&2
  exit 2
fi
if ! "$example_root/terminal-bench-report/scripts/render_pdf.sh" "$staging"; then
  failed="$(dirname "$target")/tb21-final-n3-failed-$(date -u +'%Y%m%dT%H%M%SZ')"
  mv "$staging_parent" "$failed"
  printf 'PDF rendering failed; preserved bundle at %s\n' "$failed" >&2
  exit 2
fi
if ! jq -e '.status == "passed"' "$staging/evidence/pdf-validation.json" >/dev/null; then
  failed="$(dirname "$target")/tb21-final-n3-failed-$(date -u +'%Y%m%dT%H%M%SZ')"
  mv "$staging_parent" "$failed"
  printf 'PDF validation failed; preserved bundle at %s\n' "$failed" >&2
  exit 2
fi
mv "$staging" "$target"
rmdir "$staging_parent"
printf 'generated validated analysis and PDF: %s\n' "$target"
