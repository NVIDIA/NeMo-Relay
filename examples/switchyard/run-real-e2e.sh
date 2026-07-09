#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

relay_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$relay_root/examples/switchyard/e2e-common.sh"
switchyard_root="${SWITCHYARD_ROOT:-$(cd "$relay_root/.." && pwd)/Switchyard-topic-nemo-relay-integration}"
switchyard_expected_commit="${SWITCHYARD_EXPECTED_COMMIT:-5e61cb71ea94fe4f0d365bbc788c9011d42af2e4}"
work_dir="$(mktemp -d)"
upstream_log="$work_dir/upstream.jsonl"
token="$(e2e_random_token)"

[[ -d "$switchyard_root" ]] || { echo "Switchyard worktree not found: $switchyard_root" >&2; exit 1; }
e2e_verify_switchyard_checkout "$switchyard_root" "$switchyard_expected_commit"

cleanup() {
  local status=$?
  e2e_stop_processes
  if [[ $status -eq 0 ]]; then
    rm -rf "$work_dir"
  else
    echo "E2E logs preserved in $work_dir" >&2
    e2e_tail_logs "$work_dir"
  fi
}
trap cleanup EXIT

python3 "$relay_root/examples/switchyard/fake_upstream.py" \
  --port 4101 --log "$upstream_log" >"$work_dir/upstream.log" 2>&1 &
e2e_add_pid "$!"

(
  cd "$switchyard_root"
  SWITCHYARD_ATOF_BEARER_TOKEN="$token" cargo run -p switchyard-server -- \
    --config "$relay_root/examples/switchyard/real-e2e-profiles.yaml" --port 4000
) >"$work_dir/switchyard.log" 2>&1 &
e2e_add_pid "$!"

(
  cd "$work_dir"
  SWITCHYARD_AUTHORIZATION="Bearer $token" cargo run \
    --manifest-path "$relay_root/Cargo.toml" -p nemo-relay-cli -- \
    --plugin-config-path "$relay_root/examples/switchyard/real-e2e-plugins.toml" \
    --bind 127.0.0.1:4041
) >"$work_dir/relay.log" 2>&1 &
e2e_add_pid "$!"

e2e_wait_for http://127.0.0.1:4000/health
e2e_wait_for http://127.0.0.1:4041/healthz

request() {
  local request_id="$1"
  local stream="$2"
  curl --fail --silent --no-buffer http://127.0.0.1:4041/v1/chat/completions \
    -H 'content-type: application/json' \
    -H 'x-nemo-relay-session-id: e2e-session' \
    -H "x-nemo-relay-request-id: $request_id" \
    --data-binary "{\"model\":\"client-model\",\"stream\":$stream,\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"
}

request cold-request false >"$work_dir/cold.json"

for payload in \
  '{"hook_event_name":"on_session_start","session_id":"e2e-session"}' \
  '{"hook_event_name":"pre_tool_call","session_id":"e2e-session","tool_name":"Bash","tool_input":{"command":"test"},"extra":{"task_id":"task-1","tool_call_id":"call-1"}}' \
  '{"hook_event_name":"post_tool_call","session_id":"e2e-session","tool_name":"Bash","tool_input":{"command":"test"},"tool_response":{"output":"CUDA out of memory"},"extra":{"task_id":"task-1","tool_call_id":"call-1"}}'
do
  curl --fail --silent http://127.0.0.1:4041/hooks/hermes \
    -H 'content-type: application/json' --data-binary "$payload" >/dev/null
done

sleep 1
request warm-request false >"$work_dir/warm.json"
request stream-request true >"$work_dir/stream.sse"

python3 - "$upstream_log" "$work_dir/stream.sse" <<'PY'
import json
import pathlib
import sys

records = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines()]
models = [record["body"]["model"] for record in records]
if models != ["provider/weak", "provider/strong", "provider/strong"]:
    raise SystemExit(f"unexpected cold/warm/stream route sequence: {models}")
stream = pathlib.Path(sys.argv[2]).read_text()
if "fake" not in stream or "[DONE]" not in stream:
    raise SystemExit(f"unexpected SSE output: {stream}")
print(f"real Switchyard E2E passed: {models}")
PY
