#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

relay_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$relay_root/examples/switchyard/e2e-common.sh"
switchyard_root="${SWITCHYARD_ROOT:-$(cd "$relay_root/.." && pwd)/Switchyard-relay-cost-baseline}"
secret_file="${INFERENCEHUB_SECRETS_FILE:-$(cd "$relay_root/.." && pwd)/.inference_secrets}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
artifact_dir="${SWITCHYARD_TRAJECTORY_DIR:-$relay_root/artifacts/inferencehub-stage-router-$run_id}"
token="$(e2e_random_token)"
claude_session_id="$(python3 -c 'import uuid; print(uuid.uuid4())')"
docker_network="switchyard-inferencehub-$run_id"
phoenix_container="switchyard-inferencehub-phoenix-$run_id"
collector_container="switchyard-inferencehub-otel-$run_id"
phoenix_port="${SWITCHYARD_PHOENIX_PORT:-6010}"
keep_phoenix="${SWITCHYARD_KEEP_PHOENIX:-0}"
collector_running=0
phoenix_running=0
network_created=0

mkdir -p "$artifact_dir"/{atof,atif,phoenix,workspace,.nemo-relay,claude}
artifact_dir="$(cd "$artifact_dir" && pwd)"

cleanup() {
  local status=$?
  e2e_stop_processes
  if [[ $collector_running -eq 1 ]]; then
    docker rm -f "$collector_container" >/dev/null 2>&1 || true
  fi
  if [[ $phoenix_running -eq 1 && ( $status -ne 0 || "$keep_phoenix" != "1" ) ]]; then
    docker rm -f "$phoenix_container" >/dev/null 2>&1 || true
    phoenix_running=0
  fi
  if [[ $network_created -eq 1 && $phoenix_running -eq 0 ]]; then
    docker network rm "$docker_network" >/dev/null 2>&1 || true
  fi
  if [[ $status -ne 0 ]]; then
    echo "InferenceHub StageRouter smoke failed; artifacts preserved in $artifact_dir" >&2
    e2e_tail_logs "$artifact_dir"
  fi
}
trap cleanup EXIT

for dependency in cargo claude curl docker jq python3 sed tar; do
  command -v "$dependency" >/dev/null || {
    echo "missing required command: $dependency" >&2
    exit 1
  }
done
[[ -d "$switchyard_root" ]] || {
  echo "Switchyard worktree not found: $switchyard_root" >&2
  exit 1
}
[[ -f "$secret_file" ]] || {
  echo "InferenceHub secret file not found; set INFERENCEHUB_SECRETS_FILE" >&2
  exit 1
}

set -a
# shellcheck disable=SC1090
source "$secret_file"
set +a
[[ -n "${NV_INFERENCEHUB_KEY:-}" ]] || {
  echo "NV_INFERENCEHUB_KEY is missing or blank" >&2
  exit 1
}
export INFERENCEHUB_AUTHORIZATION="Bearer ${NV_INFERENCEHUB_KEY}"
export SWITCHYARD_AUTHORIZATION="Bearer $token"

# Verify both exact endpoints without printing credentials or response payloads.
models="$(curl --fail --silent \
  -H "Authorization: ${INFERENCEHUB_AUTHORIZATION}" \
  https://inference-api.nvidia.com/v1/models)"
for model in azure/anthropic/claude-sonnet-4-6 azure/anthropic/claude-opus-4-6; do
  jq -e --arg model "$model" '.data | any(.id == $model)' <<<"$models" >/dev/null || {
    echo "InferenceHub model is unavailable: $model" >&2
    exit 1
  }
done
unset models

cat >"$artifact_dir/pricing.json" <<'JSON'
{
  "version": 1,
  "entries": [
    {
      "provider": "inferencehub",
      "model_id": "azure/anthropic/claude-sonnet-4-6",
      "rates": {"input_per_million": 3.0, "output_per_million": 15.0, "cache_read_per_million": 0.3, "cache_write_per_million": 3.75},
      "prompt_cache": {"read_accounting": "included_in_prompt_tokens"},
      "pricing_as_of": "2026-05-27",
      "pricing_source": "Anthropic public list pricing; estimate only for InferenceHub"
    },
    {
      "provider": "inferencehub",
      "model_id": "azure/anthropic/claude-opus-4-6",
      "rates": {"input_per_million": 5.0, "output_per_million": 25.0, "cache_read_per_million": 0.5, "cache_write_per_million": 6.25},
      "prompt_cache": {"read_accounting": "included_in_prompt_tokens"},
      "pricing_as_of": "2026-05-27",
      "pricing_source": "Anthropic public list pricing; estimate only for InferenceHub"
    }
  ]
}
JSON

sed \
  -e "s|@PRICING_CATALOG@|$artifact_dir/pricing.json|g" \
  -e "s|@ATOF_DIR@|$artifact_dir/atof|g" \
  -e "s|@ATIF_DIR@|$artifact_dir/atif|g" \
  "$relay_root/examples/switchyard/inferencehub-stage-router-plugins.toml.in" \
  >"$artifact_dir/.nemo-relay/plugins.toml"
cat >"$artifact_dir/.nemo-relay/config.toml" <<EOF
[agents.claude]
command = "$(command -v claude)"
EOF

# The external Visor plugin is optional for the Switchyard-only smoke. The
# cumulative validation supplies its generated manifest without placing any
# Visor source path or dependency in this repository.
if [[ -n "${VISOR_PLUGIN_MANIFEST:-}" ]]; then
  [[ -f "$VISOR_PLUGIN_MANIFEST" ]] || {
    echo "VISOR_PLUGIN_MANIFEST does not exist" >&2
    exit 1
  }
  (
    cd "$artifact_dir"
    "$relay_root/target/debug/nemo-relay" plugins add --project "$VISOR_PLUGIN_MANIFEST"
    "$relay_root/target/debug/nemo-relay" plugins enable visor
  )
  cat >>"$artifact_dir/.nemo-relay/plugins.toml" <<'EOF'

[plugins.dynamic.config]
version = 1
evidence_enabled = true

[plugins.dynamic.config.tool_result_compression]
mode = "auto"

[plugins.dynamic.config.llm_tool_result_rewrite]
enabled = "auto"
providers = ["anthropic_messages", "openai_chat", "openai_responses"]
EOF
fi

# Use a real, deterministic tool result rather than an injected routing
# failure. Its size is intentional: it gives Visor causal compression evidence
# and leaves StageRouter enough organic Relay history for later decisions.
python3 - "$artifact_dir/workspace/TASK_CONTEXT.md" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
scenario = """# Bounded queue review fixture

The queue uses independent atomic head and tail indexes over a fixed ring.
Slots contain a pointer and a ready bit, but no generation counter. Producers
reserve tail with compare-and-swap, publish a pointer, then set ready. Consumers
reserve head, read the pointer after ready, clear ready, and reuse the slot.
The review must distinguish index reservation from item publication, identify
the ABA risk after wraparound, and state the acquire/release edges precisely.
"""
details = "\n".join(
    f"Invariant {index:03d}: a slot from generation g must never be observed as generation g+1."
    for index in range(1, 241)
)
path.write_text(scenario + "\n" + details + "\n")
PY

docker info >/dev/null
docker network create "$docker_network" >/dev/null
network_created=1
docker run --detach --rm \
  --name "$phoenix_container" \
  --network "$docker_network" \
  --network-alias phoenix \
  --publish "127.0.0.1:$phoenix_port:6006" \
  --env PHOENIX_WORKING_DIR=/mnt/data \
  --volume "$artifact_dir/phoenix:/mnt/data" \
  arizephoenix/phoenix:13.22 >"$artifact_dir/phoenix.container-id"
phoenix_running=1
e2e_wait_for "http://127.0.0.1:$phoenix_port/" 240 0.5

docker run --detach --rm \
  --name "$collector_container" \
  --network "$docker_network" \
  --publish 127.0.0.1:4318:4318 \
  --volume "$relay_root/examples/switchyard/otel-collector.yaml:/etc/otelcol-contrib/config.yaml:ro" \
  --volume "$artifact_dir:/artifacts" \
  otel/opentelemetry-collector-contrib:0.135.0 \
  --config=/etc/otelcol-contrib/config.yaml >"$artifact_dir/collector.container-id"
collector_running=1

(
  cd "$switchyard_root"
  SWITCHYARD_ATOF_BEARER_TOKEN="$token" \
    cargo run -p switchyard-server -- \
      --config "$relay_root/examples/switchyard/inferencehub-stage-router-profiles.yaml" \
      --port 4000 --atof-max-snapshot-age-millis 300000
) >"$artifact_dir/switchyard.log" 2>&1 &
e2e_add_pid "$!"
e2e_wait_for http://127.0.0.1:4000/health 240 0.5

run_query() {
  local sequence="$1"
  local label="$2"
  local query="$3"
  local mode="$4"
  local before_lines=0
  local after_lines
  local -a session_args
  if [[ "$mode" == "start" ]]; then
    session_args=(--session-id "$claude_session_id")
  else
    session_args=(--resume "$claude_session_id")
  fi
  if [[ -f "$artifact_dir/atof/trajectory.atof.jsonl" ]]; then
    before_lines="$(wc -l <"$artifact_dir/atof/trajectory.atof.jsonl" | tr -d ' ')"
  fi
  (
    cd "$artifact_dir/workspace"
    ANTHROPIC_API_KEY=inferencehub \
    CLAUDE_CONFIG_DIR="$artifact_dir/claude" \
    CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1 \
    CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 \
    CLAUDE_CODE_DISABLE_THINKING=1 \
    CLAUDE_CODE_EFFORT_LEVEL=auto \
    CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096 \
    CLAUDE_CODE_SIMPLE_SYSTEM_PROMPT=1 \
    DISABLE_INTERLEAVED_THINKING=1 \
    DISABLE_PROMPT_CACHING=1 \
      "$relay_root/target/debug/nemo-relay" run \
        --agent claude \
        --anthropic-base-url https://inference-api.nvidia.com \
        --plugin-config-path "$artifact_dir/.nemo-relay/plugins.toml" \
        -- -p --output-format json \
        --model azure/anthropic/claude-sonnet-4-6 \
        --tools Read --permission-mode bypassPermissions \
        --disable-slash-commands "${session_args[@]}" "$query"
  ) >"$artifact_dir/query-$sequence-$label.log" 2>&1
  after_lines="$(wc -l <"$artifact_dir/atof/trajectory.atof.jsonl" | tr -d ' ')"
  printf '%s\t%s\t%s\t%s\n' "$sequence" "$label" "$before_lines" "$after_lines" \
    >>"$artifact_dir/query-event-ranges.tsv"
  sed -n "$((before_lines + 1)),${after_lines}p" "$artifact_dir/atof/trajectory.atof.jsonl" \
    >"$artifact_dir/trajectory-$sequence-$label.atof.jsonl"
  curl --fail --silent http://127.0.0.1:4000/v1/atof/events \
    -H "authorization: Bearer $token" \
    -H 'content-type: application/x-ndjson' \
    --data-binary "@$artifact_dir/trajectory-$sequence-$label.atof.jsonl" \
    >"$artifact_dir/trajectory-$sequence-$label.atof-ingest.json"
}

run_query 01 context \
  "Use the Read tool to read $artifact_dir/workspace/TASK_CONTEXT.md completely. Then reply with exactly CONTEXT_READY." start
run_query 02 architecture \
  'Without calling tools, produce a rigorous review of the bounded queue in TASK_CONTEXT.md. Give a concrete wraparound ABA interleaving, separate reservation from publication linearization points, specify the minimum acquire/release ordering, and recommend the smallest defensible correction. Keep the answer under 700 words.' resume
run_query 03 status \
  'Without calling tools, reply with exactly REVIEW_COMPLETE and no other text.' resume

sleep 2
docker stop --time 10 "$collector_container" >/dev/null
collector_running=0

python3 - "$artifact_dir" "$claude_session_id" "$phoenix_port" "${VISOR_PLUGIN_MANIFEST:+true}" <<'PY'
import collections
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
claude_session_id, phoenix_port, visor_enabled = sys.argv[2:]
atof_path = root / "atof" / "trajectory.atof.jsonl"
events = [json.loads(line) for line in atof_path.read_text().splitlines() if line.strip()]
decisions = [event for event in events if event.get("name") == "switchyard.routing.decision"]
models = [event.get("data", {}).get("selected_model") for event in decisions]
expected = {
    "azure/anthropic/claude-sonnet-4-6",
    "azure/anthropic/claude-opus-4-6",
}
if not expected.issubset(set(models)):
    raise SystemExit(f"both InferenceHub routes were not observed: {models}")
if any(event.get("data", {}).get("router") != "stage_router" for event in decisions):
    raise SystemExit("non-StageRouter decision observed")
warm = [
    event for event in decisions
    if event.get("data", {}).get("router_metadata", {}).get("feature_state") == "fresh"
]
if not warm or any("snapshot_age_millis" not in event["data"]["router_metadata"] for event in warm):
    raise SystemExit("fresh decisions did not retain snapshot age metadata")
if any("cascade" in json.dumps(event).lower() for event in decisions):
    raise SystemExit("legacy Cascade terminology leaked into decisions")

optimization_marks = [event for event in events if event.get("name") == "nemo_relay.llm.optimization"]
routing_marks = [
    event for event in optimization_marks
    if event.get("data", {}).get("producer") == "switchyard"
]
if not routing_marks:
    raise SystemExit("no Switchyard optimization contributions were emitted")
for event in routing_marks:
    transition = event.get("data", {}).get("model_transition", {})
    baseline = (transition.get("baseline") or {}).get("model")
    effective = (transition.get("effective") or {}).get("model")
    if baseline != "azure/anthropic/claude-opus-4-6" or effective not in expected:
        raise SystemExit(f"invalid routed model transition: {transition}")

atif_paths = sorted((root / "atif").glob("trajectory-*.atif.json"))
trajectories = []
tokens_saved = 0
summary_count = 0
for path in atif_paths:
    payload = json.loads(path.read_text())
    for step in payload.get("steps", []):
        optimization = (((step.get("metrics") or {}).get("extra") or {}).get("nemo_relay") or {}).get("optimization")
        if optimization:
            summary_count += 1
            tokens_saved += (optimization.get("tokens_saved") or {}).get("total_tokens") or 0
    trajectories.append({
        "file": path.name,
        "step_count": len(payload.get("steps", [])),
        "model_names": sorted({step.get("model_name") for step in payload.get("steps", []) if step.get("model_name")}),
        "final_metrics": payload.get("final_metrics"),
    })
if not atif_paths or summary_count == 0:
    raise SystemExit("ATIF did not retain optimization summaries")
if visor_enabled == "true" and tokens_saved <= 0:
    raise SystemExit("Visor was enabled but downstream ATIF token savings were not positive")

otel_path = root / "trajectory.otel.json"
if not otel_path.exists() or otel_path.stat().st_size == 0:
    raise SystemExit("OTEL collector did not write a trajectory")

by_reason = collections.Counter(event.get("data", {}).get("reason_code") for event in decisions)
summary = {
    "harness": "claude-code-cli",
    "claude_session_id": claude_session_id,
    "decision_count": len(decisions),
    "route_models": models,
    "route_counts": dict(collections.Counter(models)),
    "reason_counts": dict(by_reason),
    "fresh_decision_count": len(warm),
    "optimization_mark_count": len(optimization_marks),
    "switchyard_contribution_count": len(routing_marks),
    "downstream_total_tokens_saved": tokens_saved,
    "atif": trajectories,
    "otel_file": otel_path.name,
    "phoenix_url": f"http://127.0.0.1:{phoenix_port}",
    "visor_enabled": visor_enabled == "true",
}
(root / "trajectory-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

readme = f"""# InferenceHub Claude 4.6 StageRouter trajectory

This bundle is a real Claude Code CLI session routed by Switchyard StageRouter
through NeMo Relay. It uses InferenceHub's
`azure/anthropic/claude-sonnet-4-6` efficient target and
`azure/anthropic/claude-opus-4-6` capable baseline/target. No synthetic OOM or
random router is used.

## Fixed trajectory

1. Read a 240-invariant queue fixture with the real Claude Code `Read` tool.
2. Review its ABA and memory-ordering failure mode under a strict reasoning prompt.
3. Return a mechanical completion status.

The first request starts cold on the efficient picker default. Later decisions
use exact-session ATOF state and the StageRouter classifier. `router_metadata`
records `feature_state`, monotonic snapshot age, age limit, event count, and
turn depth. Turn lifecycle events cannot make old material evidence fresh.

## Files

| Path | Meaning |
| --- | --- |
| `trajectory-summary.json` | Machine-checked route, freshness, contribution, token, and exporter totals |
| `atof/trajectory.atof.jsonl` | Canonical Relay lifecycle, routing, Visor, and optimization events |
| `atif/*.atif.json` | Direct ATIF trajectories visualizable by Phoenix |
| `trajectory.otel.json` | OTLP JSON exported by the collector |
| `query-*.log` | Claude Code/Relay output for each fixed query |
| `switchyard.log` | Switchyard server and classifier diagnostics |
| `pricing.json` | Repriceable public-list estimate catalog used by Relay |
| `phoenix/` | Persistent Phoenix state for the captured run |

## Accounting interpretation

Switchyard contributions compare the explicit Opus capable baseline only with
the model Relay actually dispatched. Visor contributions, when enabled, retain
explicit prompt/total token savings. Relay combines those contributions once at
LLM close. The token counts are the durable evidence; monetary values are
repriceable estimates based on Anthropic public list prices dated 2026-05-27,
not a claim about internal InferenceHub billing.

Phoenix URL during a kept-local run: `{summary['phoenix_url']}`
"""
(root / "README.md").write_text(readme)
print(json.dumps(summary, indent=2))
PY

tar -C "$(dirname "$artifact_dir")" -czf "$artifact_dir.tar.gz" "$(basename "$artifact_dir")"
echo "InferenceHub StageRouter trajectory passed"
echo "Artifacts: $artifact_dir"
echo "Bundle: $artifact_dir.tar.gz"
