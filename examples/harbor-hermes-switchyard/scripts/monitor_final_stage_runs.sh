#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
set +x

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_root="/localhome/local-bbednarski/terminal-bench-artifacts/monitoring/tb21-stage05-baaf678"
status_path="$state_root/status.json"
analysis_ready_path="$state_root/analysis-ready.json"
log_path="$state_root/monitor.log"
lock_path="$state_root/monitor.lock"
analysis_inputs="$example_root/config/final_analysis_runs.json"
analysis_output="$example_root/reports/tb21-final-n3-opus48-baseline-v6"
maximum_automatic_restarts=3

mkdir -p "$state_root"
chmod 0700 "$state_root"
exec 9>"$lock_path"
if ! flock -n 9; then
  exit 0
fi

timestamp="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
records='[]'
all_complete=true

log() {
  printf '[%s] %s\n' "$timestamp" "$*" | tee -a "$log_path"
}

for run in 1 2 3; do
  env_file="$example_root/.env.tb21-c24-stage05-baaf678-r${run}"
  session="harbor-hermes-switchyard-tb21-stage05-baaf678-r${run}"
  run_root="/localhome/local-bbednarski/terminal-bench-artifacts-tb21-c24-stage05-baaf678-nemotron-ultra-opus48-r${run}"
  summary_path="$run_root/summary.json"
  plan_path="$run_root/plan.json"
  restart_count_path="$state_root/r${run}-automatic-restarts"
  restart_count=0
  [[ -f "$restart_count_path" ]] && read -r restart_count < "$restart_count_path"
  [[ "$restart_count" =~ ^[0-9]+$ ]] || restart_count=0

  completed=0
  planned=89
  pass_count=0
  nonpass_count=0
  summary_status="missing"
  if [[ -f "$summary_path" ]]; then
    completed="$(jq -r '.completed_tasks // 0' "$summary_path")"
    planned="$(jq -r '.planned_tasks // 89' "$summary_path")"
    pass_count="$(jq -r '.benchmark_pass_count // 0' "$summary_path")"
    nonpass_count="$(jq -r '.benchmark_nonpass_count // 0' "$summary_path")"
    summary_status="$(jq -r '.status // "unknown"' "$summary_path")"
  fi

  session_live=false
  if tmux has-session -t "$session" 2>/dev/null; then
    session_live=true
  fi
  mapfile -t matching_pids < <(pgrep -f -- "$run_root" || true)
  worker_count="${#matching_pids[@]}"
  action="observed"

  if [[ "$summary_status" == "passed" ]] \
    && (( planned == 89 && completed == 89 && pass_count + nonpass_count == 89 )); then
    action="complete"
  else
    if [[ "$session_live" == true ]]; then
      action="running"
    elif (( worker_count > 0 )); then
      action="held-matching-workers-without-tmux"
      log "R${run} is incomplete (${completed}/${planned}); tmux is absent but ${worker_count} matching process(es) remain, so no duplicate supervisor was launched"
    elif [[ ! -f "$plan_path" || ! -f "$summary_path" || ! -f "$env_file" ]]; then
      action="held-missing-immutable-input"
      log "R${run} is incomplete and missing its plan, summary, or protected environment; automatic resume refused"
    elif (( restart_count >= maximum_automatic_restarts )); then
      action="held-restart-limit"
      log "R${run} is incomplete and reached the ${maximum_automatic_restarts}-restart safety limit; diagnosis is required"
    else
      blocker_count="$(find "$run_root/tasks" -mindepth 2 -maxdepth 2 -name task-state.json -type f -print0 \
        | xargs -0 -r -n1 jq -r 'select(.status == "failed" and .failure_class == "harness_or_integration") | 1' \
        | wc -l)"
      if (( blocker_count > 0 )); then
        if "$example_root/scripts/reconcile_terminal_completions_from_env.sh" "$env_file" "$run_root" >>"$log_path" 2>&1; then
          log "R${run} reconciled ${blocker_count} verifier-backed terminal completion(s) before resume"
          completed="$(jq -r '.completed_tasks // 0' "$summary_path")"
          planned="$(jq -r '.planned_tasks // 89' "$summary_path")"
          pass_count="$(jq -r '.benchmark_pass_count // 0' "$summary_path")"
          nonpass_count="$(jq -r '.benchmark_nonpass_count // 0' "$summary_path")"
          summary_status="$(jq -r '.status // "unknown"' "$summary_path")"
        else
          action="held-integration-blocker"
          log "R${run} has ${blocker_count} integration blocker(s) not admitted by terminal-completion reconciliation; automatic resume refused"
        fi
      fi
      if [[ "$action" == "held-integration-blocker" ]]; then
        :
      elif [[ "$summary_status" == "passed" ]] \
        && (( planned == 89 && completed == 89 && pass_count + nonpass_count == 89 )); then
        action="completed-after-reconciliation"
      else
        "$example_root/scripts/launch_phase2_tmux.sh" "$env_file" "$session" >>"$log_path" 2>&1
        restart_count=$((restart_count + 1))
        printf '%s\n' "$restart_count" > "$restart_count_path"
        chmod 0600 "$restart_count_path"
        session_live=true
        action="resumed"
        log "R${run} was safely resumed from ${completed}/${planned}; automatic restart ${restart_count}/${maximum_automatic_restarts}"
      fi
    fi
  fi

  if [[ "$summary_status" != "passed" ]] \
    || (( planned != 89 || completed != 89 || pass_count + nonpass_count != 89 )); then
    all_complete=false
  fi

  record="$(jq -cn \
    --arg run "R${run}" \
    --arg root_label "stage05-baaf678-r${run}" \
    --arg summary_status "$summary_status" \
    --arg action "$action" \
    --argjson planned "$planned" \
    --argjson completed "$completed" \
    --argjson pass_count "$pass_count" \
    --argjson nonpass_count "$nonpass_count" \
    --argjson session_live "$session_live" \
    --argjson worker_count "$worker_count" \
    --argjson automatic_restarts "$restart_count" \
    '{run:$run,root_label:$root_label,summary_status:$summary_status,planned:$planned,completed:$completed,pass_count:$pass_count,nonpass_count:$nonpass_count,session_live:$session_live,matching_processes:$worker_count,automatic_restarts:$automatic_restarts,action:$action}')"
  records="$(jq -cn --argjson records "$records" --argjson record "$record" '$records + [$record]')"
done

status_tmp="$status_path.tmp.$$"
jq -n \
  --arg timestamp "$timestamp" \
  --argjson all_complete "$all_complete" \
  --arg analysis_inputs "config/final_analysis_runs.json" \
  --argjson runs "$records" \
  '{schema_version:"harbor-hermes-switchyard.final-run-monitor.v1",checked_at:$timestamp,all_complete:$all_complete,analysis_inputs:$analysis_inputs,runs:$runs}' \
  > "$status_tmp"
chmod 0600 "$status_tmp"
mv "$status_tmp" "$status_path"

if [[ "$all_complete" == true && ! -f "$analysis_ready_path" ]]; then
  ready_tmp="$analysis_ready_path.tmp.$$"
  jq -n \
    --arg timestamp "$timestamp" \
    --arg analysis_inputs "config/final_analysis_runs.json" \
    '{schema_version:"harbor-hermes-switchyard.analysis-ready.v1",ready_at:$timestamp,all_final_stage_runs_complete:true,analysis_inputs:$analysis_inputs}' \
    > "$ready_tmp"
  chmod 0600 "$ready_tmp"
  mv "$ready_tmp" "$analysis_ready_path"
  log "all three final staged runs reached 89/89; the N=3 comparison is ready to generate from $analysis_inputs"
fi

if [[ "$all_complete" == true \
  && ( ! -f "$analysis_output/aggregate-metrics.json" \
    || ! -f "$analysis_output/README.pdf" \
    || ! -f "$analysis_output/evidence/pdf-validation.json" ) ]]; then
  if "$example_root/scripts/generate_final_n3_analysis.sh" "$analysis_inputs" "$analysis_output" >>"$log_path" 2>&1; then
    log "generated the validated N=3 analysis and PDF at $analysis_output"
  else
    log "N=3 analysis/PDF generation failed; preserved evidence is recorded in the monitor log and the next timer cycle will retry"
  fi
fi
