#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v claude >/dev/null 2>&1; then
    echo "SKIP: claude is not installed"
    exit 0
fi

cargo build -p nemo-relay-cli --bin nemo-relay --features __test-cli-port-override

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
    return 0
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
export ANTHROPIC_AUTH_TOKEN="relay-claude-e2e-token"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export DISABLE_AUTOUPDATER=1
export NEMO_RELAY_GATEWAY_URL="http://127.0.0.1:1"
export NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS=1
gateway_port="$(python3 -c 'import socket; sock = socket.socket(); sock.bind(("127.0.0.1", 0)); print(sock.getsockname()[1]); sock.close()')"
export NEMO_RELAY_TEST_GATEWAY_BIND="127.0.0.1:$gateway_port"

mkdir -p \
    "$HOME" \
    "$XDG_CONFIG_HOME/nemo-relay" \
    "$XDG_DATA_HOME" \
    "$XDG_RUNTIME_DIR" \
    "$TMPDIR" \
    "$work/atof" \
    "$work/provider-barrier" \
    "$work/workspace"

cat >"$HOME/.claude.json" <<'EOF'
{
  "hasCompletedOnboarding": true,
  "theme": "dark"
}
EOF

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
version = 4

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
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
import os
import subprocess
import sys
from pathlib import Path

plugin_root = Path(sys.argv[1])
plugins = json.loads(subprocess.check_output(["claude", "plugin", "list", "--json"]))
relay = [item for item in plugins if item.get("id") == "nemo-relay-plugin@nemo-relay-local"]
assert len(relay) == 1, relay
server = relay[0]["mcpServers"]["nemo-relay"]
assert server["args"] == ["mcp"], server
assert server["env"]["NEMO_RELAY_GATEWAY_BIND"] == os.environ["NEMO_RELAY_TEST_GATEWAY_BIND"], server
generation = Path(server["env"]["NEMO_RELAY_MCP_GENERATION_FILE"])
assert generation == plugin_root / ".nemo-relay-generation", generation
assert generation.is_file(), generation
generation_token = server["env"]["NEMO_RELAY_MCP_GENERATION"]
assert generation_token == generation.read_text().splitlines()[0].strip(), server
assert server["alwaysLoad"] is True, server
PY

wait_for_relay_port_release() {
    python3 - "$gateway_port" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
deadline = time.monotonic() + 8
while time.monotonic() < deadline:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        if sock.connect_ex(("127.0.0.1", port)) != 0:
            raise SystemExit(0)
    time.sleep(0.1)
raise SystemExit(f"Relay port {port} did not become free")
PY
    return 0
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
    return 0
}

run_transparent_claude() {
    output="$work/claude-transparent.terminal"
    debug="$work/claude-transparent.debug.log"
    python3 - \
        "$output" \
        "$debug" \
        "$work/workspace" \
        "$XDG_CONFIG_HOME/nemo-relay/config.toml" \
        "$work/claude-user-settings.json" <<'PY'
import os
import pty
import select
import signal
import shutil
import subprocess
import sys
import termios
import time
import fcntl
from pathlib import Path

output, debug, workspace, config, settings = map(Path, sys.argv[1:])
relay = shutil.which("nemo-relay")
claude = shutil.which("claude")
assert relay and claude, (relay, claude)
master, slave = pty.openpty()


def make_controlling_terminal():
    os.setsid()
    fcntl.ioctl(slave, termios.TIOCSCTTY, 0)


process = subprocess.Popen(
    [
        relay,
        "run",
        "--config",
        str(config),
        "--",
        claude,
        "--settings",
        str(settings),
        "--permission-mode",
        "manual",
        "--debug-file",
        str(debug),
        "relay-e2e-tool",
    ],
    cwd=workspace,
    stdin=slave,
    stdout=slave,
    stderr=slave,
    preexec_fn=make_controlling_terminal,
)
os.close(slave)
terminal = bytearray()
deadline = time.monotonic() + 30
sent_exit = False
trusted_workspace = False
confirmed_api_key = False
try:
    while process.poll() is None and time.monotonic() < deadline:
        readable, _, _ = select.select([master], [], [], 0.2)
        if readable:
            try:
                terminal.extend(os.read(master, 65536))
            except OSError:
                break
        if not trusted_workspace and b"Accessing" in terminal and b"Quick" in terminal:
            os.write(master, b"\r")
            trusted_workspace = True
        if not confirmed_api_key and b"Detected" in terminal and b"ANTHROPIC_API_KEY" in terminal:
            os.write(master, b"\x1b[A\r")
            confirmed_api_key = True
        if not sent_exit and b"pong" in terminal.lower():
            os.write(master, b"/exit\r")
            sent_exit = True
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
finally:
    os.close(master)
    output.write_bytes(terminal)
assert sent_exit, terminal.decode(errors="replace")
assert process.returncode == 0, (process.returncode, terminal.decode(errors="replace"))

log = debug.read_text()
assert 1 <= log.count("Hook SessionStart:startup") <= 2, log
assert 1 <= log.count("Hook UserPromptSubmit") <= 2, log
assert 1 <= log.count('Hook Stop (Stop) success') <= 2, log
assert 1 <= log.count("SessionEnd:") <= 2, log
assert "Hook PreToolUse" in log, log
assert "Hook PermissionRequest" in log, log
assert "Hook PostToolUse" in log, log
assert log.count('MCP server "plugin:nemo-relay-plugin:nemo-relay": Successfully connected') == 1, log
PY
    return 0
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
messages = [
    row
    for row in requests
    if urlparse(row["path"]).path.endswith("/messages") and row["tools"]
]
assert len(messages) == 2, requests
assert messages[0]["model"] == "claude-haiku-4-5", messages
assert [message["has_tool_result"] for message in messages] == [False, True], messages
assert any(tool["name"] == "Bash" for tool in messages[0]["tools"]), messages
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
tool_starts = [
    event for event in events
    if event.get("kind") == "scope"
    and event.get("category") == "tool"
    and event.get("scope_category") == "start"
]
tool_ends = [
    event for event in events
    if event.get("kind") == "scope"
    and event.get("category") == "tool"
    and event.get("scope_category") == "end"
]
assert len(tool_starts) == len(tool_ends) == 1, (tool_starts, tool_ends)
PY
nemo-relay doctor --plugin claude-code --install-dir "$work/install"

if [[ "${RELAY_E2E_TRANSPARENT_ONLY:-0}" == "1" ]]; then
    exit 0
fi

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
