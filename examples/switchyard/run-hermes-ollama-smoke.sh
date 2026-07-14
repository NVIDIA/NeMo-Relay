#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

relay_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$relay_root/examples/switchyard/e2e-common.sh"
switchyard_root="${SWITCHYARD_ROOT:-$(cd "$relay_root/.." && pwd)/Switchyard-topic-nemo-relay-integration}"
switchyard_expected_commit="${SWITCHYARD_EXPECTED_COMMIT:-8f9db9a6a47f848cdff1d262276ba25a8ae9cbc8}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
artifact_dir="${SWITCHYARD_TRAJECTORY_DIR:-$relay_root/artifacts/hermes-switchyard-$run_id}"
token="$(e2e_random_token)"
docker_network="switchyard-e2e-$run_id"
phoenix_container="switchyard-phoenix-$run_id"
collector_container="switchyard-otel-$run_id"
phoenix_port="${SWITCHYARD_PHOENIX_PORT:-6006}"
keep_phoenix="${SWITCHYARD_KEEP_PHOENIX:-0}"
collector_running=0
phoenix_running=0
network_created=0

mkdir -p "$artifact_dir/phoenix"
[[ -d "$switchyard_root" ]] || { echo "Switchyard worktree not found: $switchyard_root" >&2; exit 1; }
e2e_verify_switchyard_checkout "$switchyard_root" "$switchyard_expected_commit" >"$artifact_dir/switchyard-revision.txt"

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
    echo "Hermes/StageRouter smoke failed; artifacts preserved in $artifact_dir" >&2
    e2e_tail_logs "$artifact_dir"
  fi
}
trap cleanup EXIT

for dependency in cargo curl docker hermes jq python3 tar; do
  command -v "$dependency" >/dev/null || {
    echo "missing required command: $dependency" >&2
    exit 1
  }
done

docker info >/dev/null
for model in llama3.2:latest qwen3.6:35b; do
  curl --fail --silent http://127.0.0.1:11434/api/tags \
    | jq -e --arg model "$model" '.models[] | select(.name == $model)' >/dev/null || {
      echo "required Ollama model is not installed: $model" >&2
      exit 1
    }
done

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
  OLLAMA_CLASSIFIER_API_KEY="$token" \
  SWITCHYARD_ATOF_BEARER_TOKEN="$token" \
    cargo run -p switchyard-server -- \
      --config "$relay_root/examples/switchyard/hermes-ollama-profiles.yaml" --port 4000
) >"$artifact_dir/switchyard.log" 2>&1 &
e2e_add_pid "$!"
e2e_wait_for http://127.0.0.1:4000/health 240 0.5

run_query() {
  local sequence="$1"
  local label="$2"
  local query="$3"
  local resume_id="${4:-}"
  local -a resume_args=()
  local before_lines=0
  local after_lines
  local atif_path
  if [[ -n "$resume_id" ]]; then
    resume_args=(--resume "$resume_id")
  fi
  if [[ -f "$artifact_dir/trajectory.atof.jsonl" ]]; then
    before_lines="$(wc -l < "$artifact_dir/trajectory.atof.jsonl" | tr -d ' ')"
  fi
  (
    cd "$artifact_dir"
    HERMES_HOME="$artifact_dir/hermes" \
    OPENAI_API_KEY=ollama \
    SWITCHYARD_AUTHORIZATION="Bearer $token" \
      cargo run --manifest-path "$relay_root/Cargo.toml" -p nemo-relay-cli \
        --features switchyard -- \
        run --agent hermes \
        --plugin-config-path "$relay_root/examples/switchyard/hermes-ollama-plugins.toml" \
        -- chat --provider custom --model llama3.2:latest \
        --query "$query" ${resume_args[@]+"${resume_args[@]}"} \
        --toolsets terminal --quiet --max-turns 2 --ignore-rules
  ) >"$artifact_dir/query-$sequence-$label.log" 2>&1
  after_lines="$(wc -l < "$artifact_dir/trajectory.atof.jsonl" | tr -d ' ')"
  printf '%s\t%s\t%s\t%s\n' "$sequence" "$label" "$before_lines" "$after_lines" \
    >> "$artifact_dir/query-event-ranges.tsv"
  # Each query is a separate Relay process. Replay its persisted ATOF segment
  # as a completion barrier before the next process starts; ingestion is
  # idempotent, so events already delivered by the best-effort live exporter
  # are reported as duplicates rather than applied twice.
  sed -n "$((before_lines + 1)),${after_lines}p" "$artifact_dir/trajectory.atof.jsonl" \
    > "$artifact_dir/trajectory-$sequence-$label.atof.jsonl"
  curl --fail --silent http://127.0.0.1:4000/v1/atof/events \
    -H "authorization: Bearer $token" \
    -H 'content-type: application/x-ndjson' \
    --data-binary "@$artifact_dir/trajectory-$sequence-$label.atof.jsonl" \
    > "$artifact_dir/trajectory-$sequence-$label.atof-ingest.json"
  jq -e '(.batch.ingested_events + .batch.duplicate_events) >= 1' \
    "$artifact_dir/trajectory-$sequence-$label.atof-ingest.json" >/dev/null
  atif_path="$(find "$artifact_dir" -maxdepth 1 -name 'trajectory-*.atif.json' -print \
    | grep -E '/trajectory-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.atif\.json$' \
    | head -1 || true)"
  if [[ -z "$atif_path" ]]; then
    echo "ATIF exporter did not produce a trajectory for query $sequence" >&2
    exit 1
  fi
  mv "$atif_path" "$artifact_dir/trajectory-$sequence-$label.atif.json"
}

emit_stage_router_signal() {
  local sequence="$1"
  local label="$2"
  local output="$3"
  local path="$artifact_dir/trajectory-signal-$sequence-$label.atof.jsonl"
  python3 - "$path" "$session_id" "$label" "$output" <<'PY'
import datetime
import json
import pathlib
import sys
import uuid

path = pathlib.Path(sys.argv[1])
session_id, label, output = sys.argv[2:]
event_uuid = str(uuid.uuid4())
base = {
    "atof_version": "0.1",
    "kind": "scope",
    "uuid": event_uuid,
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "name": "trajectory_fixture",
    "category": "tool",
    "category_profile": {"tool_call_id": f"fixture-{label}"},
    "metadata": {
        "session_id": session_id,
        "trajectory_fixture": True,
        "trajectory_fixture_label": label,
    },
}
events = [
    {**base, "scope_category": "start", "data": {"label": label}},
    {**base, "scope_category": "end", "data": {"output": output}},
]
path.write_text("".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events))
PY
  cat "$path" >> "$artifact_dir/trajectory.atof.jsonl"
  curl --fail --silent http://127.0.0.1:4000/v1/atof/events \
    -H "authorization: Bearer $token" \
    -H 'content-type: application/x-ndjson' \
    --data-binary "@$path" > "$artifact_dir/trajectory-signal-$sequence-$label.atof-ingest.json"
  jq -e '.batch.ingested_events == 2' \
    "$artifact_dir/trajectory-signal-$sequence-$label.atof-ingest.json" >/dev/null
}

simple_query='Return exactly the integer result of 17 + 25, with no explanation.'
complex_query='Do not call tools. Act as a principal concurrency engineer. Analyze a bounded lock-free MPMC queue that uses compare-and-swap on head and tail but no generation counters. Give a concrete ABA failure interleaving, identify the required C++ memory order on each publication and consumption edge, and propose the smallest defensible correction. Be precise about the linearization points.'
followup_query='Ignore the earlier technical topic. Reply with exactly SIMPLE_DONE and nothing else.'

run_query 01 simple "$simple_query"
session_id="$(jq -r 'select(.name == "switchyard.routing.requested") | .data.session_id' "$artifact_dir/trajectory.atof.jsonl" | head -1)"
if [[ -z "$session_id" || "$session_id" == "null" ]]; then
  echo "could not recover the Hermes session ID from the first routing mark" >&2
  exit 1
fi
emit_stage_router_signal 02 complex \
  'CUDA out of memory while analyzing the concurrent queue; critical failure requires careful recovery and a capable model.'
run_query 02 complex "$complex_query" "$session_id"
emit_stage_router_signal 03 simple-followup \
  'All tests passed. The next request is a direct low-risk formatting response: return exactly SIMPLE_DONE.'
run_query 03 simple-followup "$followup_query" "$session_id"

# Give the asynchronous OTLP exporter a short flush window, then stop the
# collector cleanly so its file exporter closes the shareable OTLP JSON file.
sleep 2
docker stop --time 10 "$collector_container" >/dev/null
collector_running=0

python3 - "$artifact_dir" "$session_id" "$phoenix_port" "$simple_query" "$complex_query" "$followup_query" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
session_id = sys.argv[2]
phoenix_port = sys.argv[3]
queries = sys.argv[4:]
atof_path = root / "trajectory.atof.jsonl"
events = [json.loads(line) for line in atof_path.read_text().splitlines() if line.strip()]
marks = [event for event in events if event.get("name", "").startswith("switchyard.routing.")]
decisions = [event for event in marks if event.get("name") == "switchyard.routing.decision"]

expected_models = ["llama3.2:latest", "qwen3.6:35b", "llama3.2:latest"]
event_ranges = []
for line in (root / "query-event-ranges.tsv").read_text().splitlines():
    sequence, label, start, end = line.split("\t")
    event_ranges.append((sequence, label, int(start), int(end)))
representative_decisions = []
for sequence, label, start, end in event_ranges:
    segment = events[start:end]
    decision = next(
        (event for event in segment if event.get("name") == "switchyard.routing.decision"),
        None,
    )
    if decision is None:
        raise SystemExit(f"query {sequence} ({label}) produced no successful routing decision")
    representative_decisions.append(decision)
actual_models = [event.get("data", {}).get("selected_model") for event in representative_decisions]
if actual_models != expected_models:
    raise SystemExit(f"unexpected StageRouter route sequence: {actual_models}; expected {expected_models}")

required_mark_names = {"switchyard.routing.requested", "switchyard.routing.decision"}
if not required_mark_names.issubset({event.get("name") for event in marks}):
    raise SystemExit("routing requested/decision marks were not both emitted")
for event in marks:
    name = event.get("name")
    if event.get("category") != "custom":
        raise SystemExit(f"{name} did not use category=custom")
    if event.get("category_profile", {}).get("subtype") != name:
        raise SystemExit(f"{name} category_profile.subtype was not canonical")
    schema = event.get("data_schema", {})
    if schema != {"name": "switchyard.routing_mark", "version": "1"}:
        raise SystemExit(f"{name} had unexpected data_schema: {schema}")
    metadata = event.get("metadata", {})
    if metadata.get("session_id") != session_id:
        raise SystemExit(f"{name} did not mirror session identity")
for event in decisions:
    data = event["data"]
    for key in ("decision_id", "router", "routing_attempt", "backend_id", "selected_tier", "selected_model", "latency_ms", "rollout_mode"):
        if key not in data or data[key] is None:
            raise SystemExit(f"decision mark missing {key}: {data}")

atif_paths = sorted(root.glob("trajectory-*.atif.json"))
if len(atif_paths) != 3:
    raise SystemExit(f"expected three ATIF trajectories, found {len(atif_paths)}")
for path in atif_paths:
    payload = json.loads(path.read_text())
    if not payload.get("steps"):
        raise SystemExit(f"ATIF trajectory has no steps: {path.name}")

otel_path = root / "trajectory.otel.json"
if not otel_path.exists() or not otel_path.read_text().strip():
    raise SystemExit("OTLP file exporter did not produce trajectory.otel.json")
otel_batches = [json.loads(line) for line in otel_path.read_text().splitlines() if line.strip()]

summary = {
    "session_id": session_id,
    "queries": [{
        "sequence": index + 1,
        "input": query,
        "selected_model": actual_models[index],
        "selected_tier": representative_decisions[index]["data"].get("selected_tier"),
        "reason_code": representative_decisions[index]["data"].get("reason_code"),
        "reason_summary": representative_decisions[index]["data"].get("reason_summary"),
    } for index, query in enumerate(queries)],
    "expected_route_sequence": expected_models,
    "actual_route_sequence": actual_models,
    "routing_basis": [
        "cold StageRouter efficient default",
        "canonical ATOF critical-error tool result (capable override)",
        "canonical ATOF clean-tests tool result (efficient classifier decision)",
    ],
    "atof": {"file": atof_path.name, "event_count": len(events), "routing_mark_count": len(marks)},
    "atif": {"files": [path.name for path in atif_paths], "trajectory_count": len(atif_paths)},
    "otel": {"file": otel_path.name, "export_batch_count": len(otel_batches)},
    "phoenix_url": f"http://127.0.0.1:{phoenix_port}",
}
(root / "trajectory-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
readme = f"""# Hermes / Ollama / Switchyard StageRouter trajectory

This bundle captures one fixed three-query Hermes session routed through NeMo
Relay and the Switchyard Decision API. The verified representative route is:

1. `llama3.2:latest` (efficient) — cold StageRouter default
2. `qwen3.6:35b` (capable) — critical-signal StageRouter override
3. `llama3.2:latest` (efficient) — clean-state classifier decision

Session ID: `{session_id}`

## Important fixture note

The `CUDA out of memory` text in `trajectory-signal-02-complex.atof.jsonl` is an
intentional, synthetic ATOF tool-result fixture. The machine did not run out of
memory. Fixture events carry `metadata.trajectory_fixture = true` and a fixture
label so they cannot be confused with organic Hermes events.

The fixtures are necessary for this demonstration because the current
Switchyard StageRouter Decision API classifies from its accumulated ATOF snapshot,
not directly from `current_request.body`. The critical fixture exercises the
real capable override. The clean-tests fixture removes the prior critical signal
from the one-result window and exercises the efficient classifier path.

## File map

| Files | Contents | Test coverage |
| --- | --- | --- |
| `trajectory-summary.json` | Machine-readable queries, selected models, reasons, counts, and Phoenix URL | Confirms expected and actual representative routes match |
| `trajectory.atof.jsonl` | Complete Relay ATOF stream for all three queries and the labeled fixtures | Identity propagation, lifecycle events, routing marks, and accumulator input |
| `trajectory-01-simple.atof.jsonl` | ATOF emitted by the first Hermes invocation | Cold-start efficient default |
| `trajectory-02-complex.atof.jsonl` | ATOF emitted by the complex Hermes invocation | Dispatch through the selected capable backend |
| `trajectory-03-simple-followup.atof.jsonl` | ATOF emitted by the final Hermes invocation | Return to the efficient backend |
| `trajectory-signal-*.atof.jsonl` | Canonical, labeled tool start/end fixtures | Capable critical-error override and efficient clean-state classification |
| `*.atof-ingest.json` | Switchyard ingestion reports for query segments and fixtures | Successful or idempotent ATOF accumulation |
| `trajectory-01-simple.atif.json` | ATIF representation of query 1 | Efficient-model trajectory structure |
| `trajectory-02-complex.atif.json` | ATIF representation of query 2 | Capable-model trajectory structure |
| `trajectory-03-simple-followup.atif.json` | ATIF representation of query 3 | Efficient follow-up trajectory structure |
| `trajectory.otel.json` | OTLP JSON batches written by the OpenTelemetry Collector | Relay spans exported to the collector and forwarded to Phoenix |
| `query-*.log` | Hermes/Relay stdout and stderr for each invocation | Human-readable harness responses and execution diagnostics |
| `query-event-ranges.tsv` | Query label and ATOF line-count boundaries | Separates representative user-query decisions from extra Hermes calls |

## Routing-mark assertions

The smoke validates every `switchyard.routing.*` mark in the cumulative ATOF
stream. Each mark must have:

- `category: "custom"`
- `category_profile.subtype` equal to the mark name
- `data_schema.name: "switchyard.routing_mark"`
- `data_schema.version: "1"`
- the expected session identity in metadata

Decision marks must also include a decision ID, router, attempt, backend, tier,
model, latency, and rollout mode. This run produced {len(marks)} routing marks
across {len(events)} total ATOF events.

## Phoenix and OTLP

During the smoke, Relay sends OTLP/HTTP to a local OpenTelemetry Collector. The
collector writes `trajectory.otel.json` and forwards the same spans to Phoenix.
The run produced {len(otel_batches)} OTLP export batches. When the smoke is run
with `SWITCHYARD_KEEP_PHOENIX=1`, open the `phoenix_url` from
`trajectory-summary.json` before stopping the printed Phoenix container.

## Reproduce

Install both `llama3.2:latest` and `qwen3.6:35b` in Ollama, ensure the cumulative
Switchyard checkout is available, then run:

```bash
SWITCHYARD_KEEP_PHOENIX=1 examples/switchyard/run-hermes-ollama-smoke.sh
```

The script validates the route, mark shape, ATIF contents, and OTLP output before
creating this bundle.
"""
(root / "TRAJECTORY_README.md").write_text(readme)
print(json.dumps(summary, indent=2))
PY

(
  cd "$artifact_dir"
  tar -czf trajectory-bundle.tar.gz \
    TRAJECTORY_README.md \
    trajectory-summary.json \
    trajectory.atof.jsonl \
    trajectory.otel.json \
    trajectory-*.atif.json \
    trajectory-*.atof.jsonl \
    trajectory-*.atof-ingest.json \
    query-*.log \
    query-event-ranges.tsv
)

echo "Hermes/Ollama StageRouter trajectory passed: llama3.2 -> qwen3.6:35b -> llama3.2"
echo "Artifacts: $artifact_dir"
echo "Bundle: $artifact_dir/trajectory-bundle.tar.gz"
if [[ "$keep_phoenix" == "1" ]]; then
  echo "Phoenix: http://127.0.0.1:$phoenix_port (container $phoenix_container left running)"
else
  echo "Set SWITCHYARD_KEEP_PHOENIX=1 to leave Phoenix running after the smoke."
fi
