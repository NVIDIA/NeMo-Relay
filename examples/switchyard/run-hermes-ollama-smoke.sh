#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

relay_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
switchyard_root="${SWITCHYARD_ROOT:-$(cd "$relay_root/.." && pwd)/Switchyard-relay-cumulative}"
work_dir="$(mktemp -d)"
token="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
switchyard_pid=""

cleanup() {
  local status=$?
  if [[ -n "$switchyard_pid" ]]; then
    kill "$switchyard_pid" 2>/dev/null || true
    wait "$switchyard_pid" 2>/dev/null || true
  fi
  if [[ $status -eq 0 ]]; then
    rm -rf "$work_dir"
  else
    echo "Hermes smoke logs preserved in $work_dir" >&2
    tail -100 "$work_dir/switchyard.log" 2>/dev/null || true
    tail -100 "$work_dir/relay-hermes.log" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for dependency in curl hermes jq python3; do
  command -v "$dependency" >/dev/null || {
    echo "missing required command: $dependency" >&2
    exit 1
  }
done

curl --fail --silent http://127.0.0.1:11434/api/tags \
  | jq -e '.models[] | select(.name == "qwen3.6:35b")' >/dev/null

(
  cd "$switchyard_root"
  SWITCHYARD_ATOF_BEARER_TOKEN="$token" cargo run -p switchyard-server -- \
    --config "$relay_root/examples/switchyard/hermes-ollama-profiles.yaml" --port 4000
) >"$work_dir/switchyard.log" 2>&1 &
switchyard_pid="$!"

for _ in $(seq 1 120); do
  if curl --fail --silent http://127.0.0.1:4000/health >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done
curl --fail --silent http://127.0.0.1:4000/health >/dev/null

(
  cd "$work_dir"
  HERMES_HOME="$work_dir/hermes" \
  OPENAI_API_KEY=ollama \
  SWITCHYARD_AUTHORIZATION="Bearer $token" \
    cargo run --manifest-path "$relay_root/Cargo.toml" -p nemo-relay-cli -- \
      run --agent hermes \
      --plugin-config-path "$relay_root/examples/switchyard/hermes-ollama-plugins.toml" \
      -- chat --provider custom --model qwen3.6:35b \
      --query 'Reply with exactly: HERMES_RELAY_SWITCHYARD_OK' \
      --quiet --max-turns 1 --ignore-rules
) >"$work_dir/relay-hermes.log" 2>&1

grep -q 'HERMES_RELAY_SWITCHYARD_OK' "$work_dir/relay-hermes.log"
echo "Hermes/Ollama smoke passed through Relay and Switchyard"
