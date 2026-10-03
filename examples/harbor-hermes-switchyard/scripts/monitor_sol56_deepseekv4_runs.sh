#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
set +x

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_root="/localhome/local-bbednarski/terminal-bench-artifacts/monitoring/tb21-sol56-deepseekv4-stage03-pr270-5c84c16"
status_path="$state_root/status.json"
ready_path="$state_root/analysis-ready.json"
log_path="$state_root/monitor.log"
lock_path="$state_root/monitor.lock"
analysis_manifest="$example_root/config/sol56_deepseekv4_analysis_runs.json"
analysis_output="$example_root/reports/tb21-sol56-deepseekv4-cf03-ef03-n3"
maximum_automatic_restarts=3

mkdir -p "$state_root"
chmod 0700 "$state_root"
exec 9>"$lock_path"
flock -n 9 || exit 0

timestamp="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
records='[]'
all_complete=true

log() {
  printf '[%s] %s\n' "$timestamp" "$*" | tee -a "$log_path"
}

for strategy in direct cf03 ef03; do
  for replicate in 1 2 3; do
    case "$strategy" in
      direct)
        env_file="$example_root/.env.tb21-direct-r${replicate}"
        session="hhstb21-sol56-direct-r${replicate}"
        run_root="/localhome/local-bbednarski/terminal-bench-artifacts-tb21-c24-sol56-direct-pr270-5c84c16-r${replicate}"
        ;;
      cf03)
        env_file="$example_root/.env.tb21-cf03-v3-r${replicate}"
        session="hhstb21-sol56-cf03-v3-r${replicate}"
        run_root="/localhome/local-bbednarski/terminal-bench-artifacts-tb21-c24-sol56-deepseekv4-cf03-pr270-5c84c16-v3-r${replicate}"
        ;;
      ef03)
        env_file="$example_root/.env.tb21-ef03-v3-r${replicate}"
        session="hhstb21-sol56-ef03-v3-r${replicate}"
        run_root="/localhome/local-bbednarski/terminal-bench-artifacts-tb21-c24-sol56-deepseekv4-ef03-pr270-5c84c16-v3-r${replicate}"
        ;;
    esac
    summary_path="$run_root/summary.json"
    restart_count_path="$state_root/${strategy}-r${replicate}-automatic-restarts"
    restart_count=0
    [[ -f "$restart_count_path" ]] && read -r restart_count < "$restart_count_path"
    [[ "$restart_count" =~ ^[0-9]+$ ]] || restart_count=0

    planned=89
    completed=0
    pass_count=0
    nonpass_count=0
    summary_status="missing"
    if [[ -f "$summary_path" ]]; then
      planned="$(jq -r '.planned_tasks // 89' "$summary_path")"
      completed="$(jq -r '.completed_tasks // 0' "$summary_path")"
      pass_count="$(jq -r '.benchmark_pass_count // 0' "$summary_path")"
      nonpass_count="$(jq -r '.benchmark_nonpass_count // 0' "$summary_path")"
      summary_status="$(jq -r '.status // "unknown"' "$summary_path")"
    fi

    session_live=false
    tmux has-session -t "$session" 2>/dev/null && session_live=true
    mapfile -t matching_pids < <(pgrep -f -- "$run_root" || true)
    worker_count="${#matching_pids[@]}"
    action="observed"

    if [[ "$summary_status" == "passed" ]] \
      && (( planned == 89 && completed == 89 && pass_count + nonpass_count == 89 )); then
      action="complete"
    elif [[ "$session_live" == true ]]; then
      action="running"
      all_complete=false
    elif (( worker_count > 0 )); then
      action="held-matching-workers-without-tmux"
      all_complete=false
      log "$strategy R$replicate is incomplete (${completed}/${planned}); ${worker_count} matching process(es) remain"
    elif [[ ! -f "$run_root/plan.json" || ! -f "$summary_path" || ! -f "$env_file" ]]; then
      action="held-missing-immutable-input"
      all_complete=false
      log "$strategy R$replicate is incomplete and missing its plan, summary, or protected environment"
    else
      if ! "$example_root/.venv/bin/python" "$example_root/scripts/prepare_provider_retry.py" \
        --run-root "$run_root" >>"$log_path" 2>&1; then
        action="held-provider-retry-error"
        all_complete=false
        log "$strategy R$replicate could not safely prepare a preserved provider retry"
      fi
      blocker_count="$(find "$run_root/tasks" -mindepth 2 -maxdepth 2 -name task-state.json -type f -print0 \
        | xargs -0 -r -n1 jq -r 'select(.status == "failed" and .failure_class == "harness_or_integration") | 1' \
        | wc -l)"
      if [[ "$action" != "held-provider-retry-error" ]] && (( blocker_count > 0 )); then
        if "$example_root/scripts/reconcile_terminal_completions_from_env.sh" "$env_file" "$run_root" >>"$log_path" 2>&1; then
          log "$strategy R$replicate reconciled verifier-backed terminal completion(s)"
          completed="$(jq -r '.completed_tasks // 0' "$summary_path")"
          pass_count="$(jq -r '.benchmark_pass_count // 0' "$summary_path")"
          nonpass_count="$(jq -r '.benchmark_nonpass_count // 0' "$summary_path")"
          summary_status="$(jq -r '.status // "unknown"' "$summary_path")"
        else
          action="held-integration-blocker"
          all_complete=false
          log "$strategy R$replicate has ${blocker_count} unreconciled integration blocker(s); automatic resume refused"
        fi
      fi
      if [[ "$action" != "held-integration-blocker" && "$action" != "held-provider-retry-error" ]]; then
        if [[ "$summary_status" == "passed" ]] \
          && (( planned == 89 && completed == 89 && pass_count + nonpass_count == 89 )); then
          action="completed-after-reconciliation"
        elif (( restart_count >= maximum_automatic_restarts )); then
          action="held-restart-limit"
          all_complete=false
          log "$strategy R$replicate reached the automatic restart limit after safe reconciliation; diagnosis is required"
        else
          "$example_root/scripts/launch_phase2_tmux.sh" "$env_file" "$session" >>"$log_path" 2>&1
          restart_count=$((restart_count + 1))
          printf '%s\n' "$restart_count" >"$restart_count_path"
          chmod 0600 "$restart_count_path"
          session_live=true
          action="resumed"
          all_complete=false
          log "$strategy R$replicate resumed at ${completed}/${planned}; restart ${restart_count}/${maximum_automatic_restarts}"
        fi
      fi
    fi

    record="$(jq -cn \
      --arg strategy "$strategy" \
      --arg replicate "R${replicate}" \
      --arg summary_status "$summary_status" \
      --arg action "$action" \
      --argjson planned "$planned" \
      --argjson completed "$completed" \
      --argjson pass_count "$pass_count" \
      --argjson nonpass_count "$nonpass_count" \
      --argjson session_live "$session_live" \
      --argjson worker_count "$worker_count" \
      --argjson automatic_restarts "$restart_count" \
      '{strategy:$strategy,replicate:$replicate,summary_status:$summary_status,planned:$planned,completed:$completed,pass_count:$pass_count,nonpass_count:$nonpass_count,session_live:$session_live,matching_processes:$worker_count,automatic_restarts:$automatic_restarts,action:$action}')"
    records="$(jq -cn --argjson records "$records" --argjson record "$record" '$records + [$record]')"
  done
done

status_tmp="$status_path.tmp.$$"
jq -n \
  --arg timestamp "$timestamp" \
  --argjson all_complete "$all_complete" \
  --argjson runs "$records" \
  '{schema_version:"harbor-hermes-switchyard.sol56-deepseekv4-monitor.v1",checked_at:$timestamp,interval_seconds:600,all_complete:$all_complete,runs:$runs}' \
  >"$status_tmp"
chmod 0600 "$status_tmp"
mv "$status_tmp" "$status_path"

if [[ "$all_complete" == true && ! -f "$ready_path" ]]; then
  jq -n --arg timestamp "$timestamp" \
    '{schema_version:"harbor-hermes-switchyard.analysis-ready.v1",ready_at:$timestamp,all_nine_runs_complete:true}' \
    >"$ready_path"
  chmod 0600 "$ready_path"
  log "all nine Sol 5.6 / DeepSeek V4 experiments reached 89/89"
fi

if [[ "$all_complete" == true \
  && ( ! -f "$analysis_output/aggregate-metrics.json" \
    || ! -f "$analysis_output/README.pdf" \
    || ! -f "$analysis_output/evidence/pdf-validation.json" ) ]]; then
  if "$example_root/scripts/generate_sol56_deepseekv4_n3_analysis.sh" \
    "$analysis_manifest" "$analysis_output" >>"$log_path" 2>&1; then
    log "generated the validated isolated N=3 report at $analysis_output"
  else
    log "isolated N=3 report generation failed; the next 10-minute cycle will retry"
  fi
fi
