#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Guarded supervisor for the signal-only → classifier staged-router study.
# It never modifies a completed run and refuses an automatic restart when an
# attempt is classified as a harness/integration failure.

set -euo pipefail
set +x

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_root="/localhome/local-bbednarski/terminal-bench-artifacts/monitoring/tb21-sol56-glm52-stage-ef05"
signal_env="$example_root/.env.tb21-sol56-glm52-stage-ef05-signal-r1"
classifier_env="$example_root/.env.tb21-sol56-glm52-stage-ef05-classifier-r1"
signal_root="/localhome/local-bbednarski/terminal-bench-artifacts/runs/tb21-sol56-glm52-stage-ef05-signal-c24-r1"
classifier_root="/localhome/local-bbednarski/terminal-bench-artifacts/runs/tb21-sol56-glm52-stage-ef05-classifier-c24-r1"
signal_session="tb21-sol56-glm52-signal-r1"
classifier_session="tb21-sol56-glm52-classifier-r1"
report_root="$example_root/reports/tb21-glm52-stage-ef05-signal-vs-classifier"
status_path="$state_root/status.json"
log_path="$state_root/monitor.log"

mkdir -p "$state_root"
chmod 0700 "$state_root"
exec 9>"$state_root/monitor.lock"
flock -n 9 || exit 0

timestamp="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
log() { printf '[%s] %s\n' "$timestamp" "$*" | tee -a "$log_path"; }

complete() {
  local root="$1"
  local summary="$root/summary.json"
  [[ -f "$summary" ]] && jq -e \
    '(.status == "passed") and (.planned_tasks == 89) and (.completed_tasks == 89) and ((.benchmark_pass_count + .benchmark_nonpass_count) == 89)' \
    "$summary" >/dev/null
}

integration_blocker() {
  local root="$1"
  find "$root/tasks" -mindepth 2 -maxdepth 2 -name task-state.json -type f -print0 2>/dev/null \
    | xargs -0 -r -n1 jq -e 'select(.status == "failed" and .failure_class == "harness_or_integration")' >/dev/null
}

session_or_worker_live() {
  local session="$1" root="$2"
  tmux has-session -t "$session" 2>/dev/null || pgrep -f -- "$root" >/dev/null
}

resume_if_safe() {
  local name="$1" env_file="$2" session="$3" root="$4"
  if complete "$root"; then
    return
  fi
  if session_or_worker_live "$session" "$root"; then
    return
  fi
  if [[ ! -f "$root/plan.json" || ! -f "$env_file" ]]; then
    log "$name held: immutable plan or environment file is missing"
    return
  fi
  if integration_blocker "$root"; then
    log "$name held: harness/integration failure requires diagnosis"
    return
  fi
  if "$example_root/scripts/launch_phase2_tmux.sh" "$env_file" "$session" >>"$log_path" 2>&1; then
    log "$name resumed"
  else
    log "$name resume failed; preserved output for diagnosis"
  fi
}

signal_complete=false
classifier_complete=false
complete "$signal_root" && signal_complete=true
complete "$classifier_root" && classifier_complete=true

if [[ "$signal_complete" == false ]]; then
  resume_if_safe signal "$signal_env" "$signal_session" "$signal_root"
else
  decision_count="$(rg -l '"name"\s*:\s*"switchyard\.routing\.decision"' "$signal_root/tasks" --glob 'trajectory.atof.jsonl' 2>/dev/null | wc -l)"
  if (( decision_count == 0 )); then
    log "classifier held: complete signal run has no routing-decision telemetry"
  elif [[ "$classifier_complete" == false ]] && ! session_or_worker_live "$classifier_session" "$classifier_root"; then
    if [[ ! -e "$classifier_root/plan.json" ]]; then
      setup_root="$signal_root/setup-admission"
      if [[ ! -f "$setup_root/summary.json" ]] || ! jq -e '(.status == "passed") and (.passed == 89) and (.failed == 0)' "$setup_root/summary.json" >/dev/null; then
        log "classifier held: signal setup admission is not a complete 89/89 pass"
      else
        sed -i '/^TBENCH_REUSE_SETUP_EVIDENCE=/d' "$classifier_env"
        printf 'TBENCH_REUSE_SETUP_EVIDENCE=%s\n' "$setup_root" >>"$classifier_env"
        chmod 600 "$classifier_env"
        if "$example_root/scripts/launch_phase2_tmux.sh" "$classifier_env" "$classifier_session" >>"$log_path" 2>&1; then
          log "classifier launched using verified signal setup evidence"
        else
          log "classifier launch failed; preserved output for diagnosis"
        fi
      fi
    else
      resume_if_safe classifier "$classifier_env" "$classifier_session" "$classifier_root"
    fi
  fi
fi

complete "$signal_root" && signal_complete=true || signal_complete=false
complete "$classifier_root" && classifier_complete=true || classifier_complete=false

if [[ "$signal_complete" == true && "$classifier_complete" == true ]]; then
  if "$example_root/scripts/generate_glm52_stage_router_analysis.sh" \
    "$example_root/config/glm52_stage_router_analysis_runs.json" "$report_root" >>"$log_path" 2>&1; then
    log "both arms complete and validated report generated"
  else
    log "both arms complete but report generation failed; retry next cycle"
  fi
fi

jq -n \
  --arg checked_at "$timestamp" \
  --argjson signal_complete "$signal_complete" \
  --argjson classifier_complete "$classifier_complete" \
  '{schema_version:"harbor-hermes-switchyard.glm52-stage-router-monitor.v1",checked_at:$checked_at,signal_complete:$signal_complete,classifier_complete:$classifier_complete}' \
  >"$status_path.tmp"
chmod 600 "$status_path.tmp"
mv "$status_path.tmp" "$status_path"
