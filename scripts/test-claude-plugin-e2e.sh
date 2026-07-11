#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v claude >/dev/null 2>&1; then
    echo "SKIP: claude is not installed"
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
    if [[ -d "$work/install" ]]; then
        nemo-relay uninstall claude-code --install-dir "$work/install" >/dev/null 2>&1 || true
    fi
    if [[ -n "$provider_pid" ]]; then
        kill "$provider_pid" 2>/dev/null || true
        wait "$provider_pid" 2>/dev/null || true
    fi
    if [[ "${RELAY_E2E_KEEP_WORK:-0}" == "1" ]]; then
        echo "Claude Code E2E workspace retained at $work" >&2
    else
        rm -rf "$work"
    fi
}
trap cleanup EXIT

while IFS='=' read -r name _; do
    if [[ "$name" == NEMO_RELAY_* ]]; then
        unset "$name"
    fi
done < <(env)

export HOME="$work/home"
export XDG_CONFIG_HOME="$work/xdg"
export XDG_DATA_HOME="$work/data"
export XDG_RUNTIME_DIR="$work/runtime"
export TMPDIR="$work/tmp"
export PATH="$repo_root/target/debug:$PATH"
export ANTHROPIC_API_KEY="relay-claude-e2e-key"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export DISABLE_AUTOUPDATER=1
export NEMO_RELAY_GATEWAY_URL="http://127.0.0.1:1"
export NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS=1

mkdir -p \
    "$HOME" \
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
anthropic_base_url = "http://$provider_address"
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

nemo-relay install claude-code --install-dir "$work/install" --skip-doctor
plugin_root="$work/install/claude-code-marketplace/plugins/nemo-relay-plugin"
claude plugin validate "$plugin_root" --strict
nemo-relay doctor --plugin claude-code --install-dir "$work/install"

python3 - "$plugin_root" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

plugin_root = Path(sys.argv[1])
plugins = json.loads(subprocess.check_output(["claude", "plugin", "list", "--json"]))
relay = [item for item in plugins if item.get("id") == "nemo-relay-plugin@nemo-relay-local"]
assert len(relay) == 1, relay
server = relay[0]["mcpServers"]["nemo-relay"]
assert server["args"] == ["mcp"], server
assert server["env"]["NEMO_RELAY_GATEWAY_BIND"] == "127.0.0.1:47632", server
generation = Path(server["env"]["NEMO_RELAY_MCP_GENERATION_FILE"])
assert generation == plugin_root / ".nemo-relay-generation", generation
assert generation.is_file(), generation
generation_token = server["env"]["NEMO_RELAY_MCP_GENERATION"]
assert generation_token == generation.read_text().splitlines()[0].strip(), server
assert server["alwaysLoad"] is True, server
PY

wait_for_relay_port_release() {
    python3 - <<'PY'
import socket
import time

deadline = time.monotonic() + 8
while time.monotonic() < deadline:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        if sock.connect_ex(("127.0.0.1", 47632)) != 0:
            raise SystemExit(0)
    time.sleep(0.1)
raise SystemExit("Relay port 47632 did not become free")
PY
}

run_claude() {
    run_id="$1"
    output="$work/claude-$run_id.json"
    stderr="$work/claude-$run_id.stderr"
    debug="$work/claude-$run_id.debug.log"
    (
        cd "$work/workspace"
        claude -p "ping" \
            --output-format json \
            --model claude-sonnet-4-5 \
            --no-session-persistence \
            --tools "" \
            --debug-file "$debug"
    ) >"$output" 2>"$stderr"
    python3 - "$output" "$stderr" "$debug" <<'PY'
import json
import sys
from pathlib import Path

output, stderr, debug = map(Path, sys.argv[1:])
result = json.loads(output.read_text())
assert result["subtype"] == "success", (result, stderr.read_text())
assert result["result"] == "pong", result
log = debug.read_text()
assert log.count("Hook SessionStart:startup") == 1, log
assert log.count("Hook UserPromptSubmit") == 1, log
assert log.count('Hook Stop (Stop) success') == 1, log
assert log.count("SessionEnd:other") == 1, log
assert log.count('MCP server "plugin:nemo-relay-plugin:nemo-relay": Successfully connected') == 1, log
assert '"hasTools":false' in log, log
PY
}

run_transparent_claude() {
    output="$work/claude-transparent.json"
    stderr="$work/claude-transparent.stderr"
    debug="$work/claude-transparent.debug.log"
    (
        cd "$work/workspace"
        nemo-relay run \
            --config "$XDG_CONFIG_HOME/nemo-relay/config.toml" \
            -- \
            claude \
            --settings "$work/claude-user-settings.json" \
            -p "ping" \
            --output-format json \
            --no-session-persistence \
            --tools "" \
            --debug-file "$debug"
    ) >"$output" 2>"$stderr"
    python3 - "$output" "$stderr" "$debug" <<'PY'
import json
import sys
from pathlib import Path

output, stderr, debug = map(Path, sys.argv[1:])
result = json.loads(output.read_text())
assert result["subtype"] == "success", (result, stderr.read_text())
assert result["result"] == "pong", result
log = debug.read_text()
assert 1 <= log.count("Hook SessionStart:startup") <= 2, log
assert 1 <= log.count("Hook UserPromptSubmit") <= 2, log
assert 1 <= log.count('Hook Stop (Stop) success') <= 2, log
assert 1 <= log.count("SessionEnd:other") <= 2, log
assert log.count('MCP server "plugin:nemo-relay-plugin:nemo-relay": Successfully connected') == 1, log
PY
}

# The transparent wrapper preserves the explicit Claude settings source. The installed Relay MCP
# borrows the dynamic gateway, while its persistent hooks exit without duplicating ATOF delivery.
cat >"$work/claude-user-settings.json" <<'EOF'
{
  "model": "claude-haiku-4-5",
  "enabledPlugins": {
    "nemo-relay-plugin@nemo-relay-local": true
  }
}
EOF
cp "$HOME/.claude/settings.json" "$work/claude-settings-before-transparent.json"
cp "$work/claude-user-settings.json" "$work/claude-user-settings-before-transparent.json"
: >"$provider_log"
events="$work/atof/events.jsonl"
rm -f "$events"
wait_for_relay_port_release
run_transparent_claude
wait_for_relay_port_release
cmp "$HOME/.claude/settings.json" "$work/claude-settings-before-transparent.json"
cmp "$work/claude-user-settings.json" "$work/claude-user-settings-before-transparent.json"
python3 - "$provider_log" "$events" <<'PY'
import json
import sys
from urllib.parse import urlparse

requests = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
messages = [row for row in requests if urlparse(row["path"]).path.endswith("/messages")]
assert len(messages) == 1, requests
assert messages[0]["model"] == "claude-haiku-4-5", messages
events = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
turn_starts = [
    event for event in events
    if event.get("kind") == "scope"
    and event.get("name") == "claude-code-turn"
    and event.get("scope_category") == "start"
]
turn_ends = [
    event for event in events
    if event.get("kind") == "scope"
    and event.get("name") == "claude-code-turn"
    and event.get("scope_category") == "end"
]
assert len(turn_starts) == len(turn_ends) == 1, (turn_starts, turn_ends)
PY
nemo-relay doctor --plugin claude-code --install-dir "$work/install"

wait_for_relay_port_release
: >"$provider_log"
rm -f "$events"
for run_id in $(seq 1 10); do
    run_claude "$run_id"
    wait_for_relay_port_release
done

touch "$work/provider-barrier/enabled"
run_claude concurrent-a &
background_pids+=("$!")
run_claude concurrent-b &
background_pids+=("$!")

python3 - "$work/provider-barrier/arrivals" <<'PY'
import sys
import time
from pathlib import Path

arrivals = Path(sys.argv[1])
deadline = time.monotonic() + 20
while time.monotonic() < deadline:
    if arrivals.exists() and int(arrivals.read_text() or "0") >= 2:
        raise SystemExit(0)
    time.sleep(0.05)
raise SystemExit("concurrent Claude requests did not reach the provider barrier")
PY
touch "$work/provider-barrier/release"

for pid in "${background_pids[@]}"; do
    [[ -n "$pid" ]] || continue
    wait "$pid"
done
background_pids=("")
wait_for_relay_port_release

python3 - "$provider_log" "$work/atof/events.jsonl" "$work" <<'PY'
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

provider_log, atof_path, work = map(Path, sys.argv[1:])
requests = [json.loads(line) for line in provider_log.read_text().splitlines()]
messages = [row for row in requests if urlparse(row["path"]).path.endswith("/messages")]
assert len(messages) == 12, messages
assert all(row["x_api_key"] == "relay-claude-e2e-key" for row in messages), messages

events = [json.loads(line) for line in atof_path.read_text().splitlines()]
turn_starts = [
    event
    for event in events
    if event.get("kind") == "scope"
    and event.get("name") == "claude-code-turn"
    and event.get("scope_category") == "start"
]
turn_ends = [
    event
    for event in events
    if event.get("kind") == "scope"
    and event.get("name") == "claude-code-turn"
    and event.get("scope_category") == "end"
]
llm_starts = [
    event
    for event in events
    if event.get("kind") == "scope"
    and event.get("name") == "anthropic.messages"
    and event.get("scope_category") == "start"
]
llm_ends = [
    event
    for event in events
    if event.get("kind") == "scope"
    and event.get("name") == "anthropic.messages"
    and event.get("scope_category") == "end"
]
assert len(turn_starts) == len(turn_ends) == 12, (len(turn_starts), len(turn_ends))
assert len(llm_starts) == len(llm_ends) == 12, (len(llm_starts), len(llm_ends))
session_ids = {event["metadata"]["session_id"] for event in turn_starts}
assert len(session_ids) == 12, session_ids

debug_logs = [
    path for path in work.glob("claude-*.debug.log")
    if path.name != "claude-transparent.debug.log"
]
assert len(debug_logs) == 12, debug_logs
PY

echo "Claude Code plugin E2E passed: 10 cold runs and 2 concurrent runs"
