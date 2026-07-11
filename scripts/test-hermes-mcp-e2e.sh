#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
keep_work="${NEMO_RELAY_E2E_KEEP_WORK:-0}"
cold_runs="${NEMO_RELAY_HERMES_E2E_COLD_RUNS:-10}"

if ! command -v hermes >/dev/null 2>&1; then
    echo "SKIP: hermes is not installed"
    exit 0
fi
cargo build -p nemo-relay-cli --bin nemo-relay

work="$(mktemp -d)"
provider_pid=""
background_pids=("")

cleanup() {
    for pid in "${background_pids[@]}"; do
        [[ -n "$pid" ]] || continue
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    if [[ -n "$provider_pid" ]]; then
        kill "$provider_pid" 2>/dev/null || true
        wait "$provider_pid" 2>/dev/null || true
    fi
    for owner in "${XDG_CONFIG_HOME:-}/nemo-relay/bootstrap"/sidecar-*.owner.json; do
        [[ -f "$owner" ]] || continue
        python3 - "$owner" <<'PY' || true
import json
import sys
import urllib.request
from pathlib import Path

owner = json.loads(Path(sys.argv[1]).read_text())
request = urllib.request.Request(
    f"{owner['url']}/bootstrap/shutdown",
    headers={"x-nemo-relay-bootstrap-token": owner["shutdown_token"]},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=2):
        pass
except OSError:
    pass
PY
    done
    if [[ "$keep_work" == "1" ]]; then
        echo "Hermes MCP E2E work directory preserved at $work" >&2
        return
    fi
    rm -rf "$work"
}
trap cleanup EXIT

while IFS='=' read -r name _; do
    if [[ "$name" == NEMO_RELAY_* ]]; then
        unset "$name"
    fi
done < <(env)

export HOME="$work/home"
export HERMES_HOME="$work/hermes"
export XDG_CONFIG_HOME="$work/xdg"
export XDG_DATA_HOME="$work/data"
export XDG_RUNTIME_DIR="$work/runtime"
export TMPDIR="$work/tmp"
export PATH="$repo_root/target/debug:$PATH"
export OPENAI_API_KEY="relay-hermes-e2e-key"
export OPENAI_BASE_URL="http://127.0.0.1:47632/v1"
export NEMO_RELAY_GATEWAY_URL="http://127.0.0.1:1"
# Hermes drains some shell hooks after the foreground CLI has exited. Keep a short grace period so
# one lifecycle cannot be split across two gateway generations; production retains the gateway for
# 300 seconds.
export NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS=5
export DISABLE_AUTOUPDATER=1

mkdir -p \
    "$HOME" \
    "$HERMES_HOME" \
    "$XDG_CONFIG_HOME/nemo-relay" \
    "$XDG_DATA_HOME" \
    "$XDG_RUNTIME_DIR" \
    "$TMPDIR" \
    "$work/atof" \
    "$work/provider-barrier" \
    "$work/workspace"

provider_ready="$work/provider-ready.json"
provider_log="$work/provider-requests.jsonl"
python3 "$repo_root/scripts/test-support/codex_mock_provider.py" \
    --ready-file "$provider_ready" \
    --log-file "$provider_log" \
    --barrier-dir "$work/provider-barrier" &
provider_pid=$!

for _ in $(seq 1 100); do
    [[ -s "$provider_ready" ]] && break
    sleep 0.05
done
[[ -s "$provider_ready" ]]
provider_address="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["address"])' "$provider_ready")"

cat >"$XDG_CONFIG_HOME/nemo-relay/config.toml" <<EOF
[upstream]
openai_base_url = "http://$provider_address/v1"

[agents.hermes]
command = "hermes"
hooks_path = "$HERMES_HOME/config.yaml"
EOF

cat >"$XDG_CONFIG_HOME/nemo-relay/plugins.toml" <<EOF
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = true
output_directory = "$work/atof"
filename = "events.jsonl"
mode = "append"
EOF

nemo-relay install hermes
nemo-relay doctor --plugin hermes --json >"$work/doctor.json"

python3 - "$HERMES_HOME" "$work/doctor.json" "$repo_root/target/debug/nemo-relay" <<'PY'
import json
import sys
from pathlib import Path

home, doctor_path, relay = map(Path, sys.argv[1:])
config = (home / "config.yaml").read_text()
assert "mcp_servers:" in config and "nemo-relay:" in config, config
assert str(relay.resolve()) in config, config
assert "- mcp" in config and "- --agent" not in config, config
assert "NEMO_RELAY_GATEWAY_BIND: 127.0.0.1:47632" in config, config
assert "OPENAI_API_KEY: ${OPENAI_API_KEY}" in config, config
generation = home / ".nemo-relay-generation"
assert f"NEMO_RELAY_MCP_GENERATION_FILE: {generation}" in config, config
assert generation == home / ".nemo-relay-generation", generation
assert generation.is_file(), generation
generation_token = generation.read_text().splitlines()[0].strip()
assert f"NEMO_RELAY_MCP_GENERATION: {generation_token}" in config, config

allowlist = json.loads((home / "shell-hooks-allowlist.json").read_text())
commands = {
    entry["command"]
    for entry in allowlist["approvals"]
    if "hook-forward hermes" in entry.get("command", "")
}
assert len(commands) == 1, commands
command = commands.pop()
assert f"--generation-token {generation_token}" in command, command
approvals = [entry for entry in allowlist["approvals"] if entry.get("command") == command]
assert len(approvals) == 13, approvals
assert len({entry["event"] for entry in approvals}) == 13, approvals
assert config.count("hook-forward hermes") == 13, config

doctor = json.loads(doctor_path.read_text())
hermes = next(agent for agent in doctor["agents"] if agent["name"] == "hermes")
assert hermes["status"] == "pass", hermes
assert "MCP lifecycle" in hermes["annotation"], hermes
PY

wait_for_relay_port_release() {
    python3 - <<'PY'
import socket
import time

deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        if sock.connect_ex(("127.0.0.1", 47632)) != 0:
            raise SystemExit(0)
    time.sleep(0.1)
raise SystemExit("Relay port 47632 did not become free")
PY
}

run_hermes() {
    run_id="$1"
    output="$work/hermes-$run_id.stdout"
    stderr="$work/hermes-$run_id.stderr"
    (
        cd "$work/workspace"
        hermes -z "ping" --provider openai-api --model gpt-4o-mini
    ) >"$output" 2>"$stderr"
    python3 - "$output" "$stderr" <<'PY'
import sys
from pathlib import Path

output, stderr = map(Path, sys.argv[1:])
assert output.read_text().strip().lower() == "pong", (output.read_text(), stderr.read_text())
PY
}

wait_for_relay_port_release
for run_id in $(seq 1 "$cold_runs"); do
    run_hermes "$run_id"
    wait_for_relay_port_release
done

touch "$work/provider-barrier/enabled"
run_hermes concurrent-a &
background_pids+=("$!")
run_hermes concurrent-b &
background_pids+=("$!")

python3 - "$work/provider-barrier/arrivals" <<'PY'
import socket
import sys
import time
from pathlib import Path

arrivals = Path(sys.argv[1])
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    if arrivals.exists() and int(arrivals.read_text() or "0") >= 2:
        with socket.socket() as sock:
            sock.settimeout(0.2)
            assert sock.connect_ex(("127.0.0.1", 47632)) == 0, "shared Relay gateway is not alive"
        raise SystemExit(0)
    time.sleep(0.05)
raise SystemExit("concurrent Hermes requests did not reach the provider barrier")
PY
touch "$work/provider-barrier/release"

for pid in "${background_pids[@]}"; do
    [[ -n "$pid" ]] || continue
    wait "$pid"
done
background_pids=("")
wait_for_relay_port_release

python3 - "$provider_log" "$work/atof/events.jsonl" "$cold_runs" <<'PY'
import collections
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

provider_log, atof_path = map(Path, sys.argv[1:3])
cold_runs = int(sys.argv[3])
expected_runs = cold_runs + 2
requests = [json.loads(line) for line in provider_log.read_text().splitlines() if line.strip()]
completions = [
    row for row in requests if urlparse(row["path"]).path.endswith("/chat/completions")
]
assert len(completions) == expected_runs, completions
assert all(row["authorization"] == "Bearer relay-hermes-e2e-key" for row in completions), completions

events = [json.loads(line) for line in atof_path.read_text().splitlines() if line.strip()]
assert events and all(event.get("atof_version") == "0.1" for event in events), events
scope_counts = collections.defaultdict(collections.Counter)
for event in events:
    if event.get("kind") == "scope":
        scope_counts[event["uuid"]][event["scope_category"]] += 1
for scope_id, counts in scope_counts.items():
    assert counts == {"start": 1, "end": 1}, (scope_id, counts)

turn_starts = [
    event
    for event in events
    if event.get("kind") == "scope"
    and event.get("scope_category") == "start"
    and event.get("name") == "hermes-turn"
]
llm_starts = [
    event
    for event in events
    if event.get("kind") == "scope"
    and event.get("scope_category") == "start"
    and event.get("name") == "openai.chat_completions"
]
assert len(turn_starts) == expected_runs, turn_starts
assert len(llm_starts) == expected_runs, llm_starts
session_ids = [event.get("metadata", {}).get("session_id") for event in turn_starts]
assert None not in session_ids and len(set(session_ids)) == expected_runs, session_ids
llm_parents = [event.get("parent_uuid") for event in llm_starts]
assert None not in llm_parents and len(set(llm_parents)) == expected_runs, llm_parents
PY

echo "Hermes MCP E2E passed: $cold_runs cold runs and 2 concurrent runs"
