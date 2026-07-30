#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

relay_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$relay_root/examples/switchyard/e2e-common.sh"
work_dir="$(mktemp -d)"
upstream_log="$work_dir/upstream.jsonl"

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

for dependency in cargo curl python3; do
  command -v "$dependency" >/dev/null || {
    echo "missing required command: $dependency" >&2
    exit 1
  }
done

python3 "$relay_root/examples/switchyard/fake_upstream.py" \
  --port 4101 --log "$upstream_log" >"$work_dir/upstream.log" 2>&1 &
e2e_add_pid "$!"

(
  cd "$relay_root"
  cargo run -p nemo-relay-cli --features switchyard -- \
    --plugin-config-path "$relay_root/examples/switchyard/classifier-plugins.toml" \
    --bind 127.0.0.1:4041
) >"$work_dir/relay.log" 2>&1 &
relay_pid="$!"
e2e_add_pid "$relay_pid"

e2e_wait_for http://127.0.0.1:4041/healthz 240 0.25 "$relay_pid"

curl --fail --silent http://127.0.0.1:4041/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-nemo-relay-session-id: classifier-buffered' \
  --data-binary \
    '{"model":"client/model","stream":false,"messages":[{"role":"user","content":"classify this"}]}' \
  >"$work_dir/buffered.json"

curl --fail --silent --no-buffer http://127.0.0.1:4041/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-nemo-relay-session-id: classifier-streaming' \
  --data-binary \
    '{"model":"client/model","stream":true,"messages":[{"role":"user","content":"classify this stream"}]}' \
  >"$work_dir/stream.sse"

python3 - "$upstream_log" "$work_dir/buffered.json" "$work_dir/stream.sse" <<'PY'
import json
import pathlib
import sys

records = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines()]
models = [record["body"]["model"] for record in records]
expected = [
    "provider/classifier",
    "provider/fast",
    "provider/classifier",
    "provider/fast",
]
if models != expected:
    raise SystemExit(f"unexpected classifier call sequence: {models}")
for record in records:
    body = record["body"]
    if body["model"] == "provider/classifier":
        if not isinstance(body.get("response_format"), dict):
            raise SystemExit(f"classifier request omitted response_format: {body}")
buffered = json.loads(pathlib.Path(sys.argv[2]).read_text())
if buffered["model"] != "provider/fast":
    raise SystemExit(f"classifier did not select the weak target: {buffered}")
stream = pathlib.Path(sys.argv[3]).read_text()
if "fake" not in stream or "[DONE]" not in stream:
    raise SystemExit(f"unexpected SSE output: {stream}")
print(f"in-process Switchyard classifier E2E passed: {models}")
PY
