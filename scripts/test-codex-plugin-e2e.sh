#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v codex >/dev/null 2>&1; then
    echo "SKIP: codex is not installed"
    exit 0
fi

cargo build -p nemo-relay-cli --bin nemo-relay

work="$(mktemp -d)"
provider_pid=""
background_pids=("")
find_sidecar_file() {
    python3 - "${TMPDIR:-$work}" "${XDG_CONFIG_HOME:-$work}" "$1" <<'PY'
import sys
from pathlib import Path

matches = [
    path
    for root in sys.argv[1:3]
    for path in Path(root).rglob(sys.argv[3])
    if path.is_file()
]
if matches:
    print(max(matches, key=lambda path: path.stat().st_mtime_ns))
PY
}

read_sidecar_pid() {
    python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    print(json.load(source)["pid"])
PY
}

cleanup() {
    codex_pgids=("")
    for pgid_file in "$work"/codex-*.pgid; do
        [[ -f "$pgid_file" ]] || continue
        pgid="$(cat "$pgid_file" 2>/dev/null || true)"
        [[ "$pgid" =~ ^[0-9]+$ ]] || continue
        codex_pgids+=("$pgid")
        kill -TERM -- "-$pgid" 2>/dev/null || true
    done
    for pid in "${background_pids[@]}"; do
        [[ -n "$pid" ]] || continue
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    for pgid in "${codex_pgids[@]}"; do
        [[ -n "$pgid" ]] || continue
        for _ in $(seq 1 50); do
            kill -0 -- "-$pgid" 2>/dev/null || break
            sleep 0.02
        done
        kill -KILL -- "-$pgid" 2>/dev/null || true
    done
    if [[ -n "$provider_pid" ]]; then
        kill "$provider_pid" 2>/dev/null || true
        wait "$provider_pid" 2>/dev/null || true
    fi
    pid_file="$(find_sidecar_file 'sidecar-*.owner.json')"
    if [[ -n "$pid_file" && -f "$pid_file" ]]; then
        sidecar_pid="$(read_sidecar_pid "$pid_file" 2>/dev/null || true)"
        if [[ "$sidecar_pid" =~ ^[0-9]+$ ]]; then
            kill "$sidecar_pid" 2>/dev/null || true
            for _ in $(seq 1 50); do
                kill -0 "$sidecar_pid" 2>/dev/null || break
                sleep 0.02
            done
            kill -KILL "$sidecar_pid" 2>/dev/null || true
        fi
    fi
    if [[ "${RELAY_E2E_KEEP_WORK:-0}" == "1" ]]; then
        echo "Codex E2E workspace retained at $work" >&2
    else
        rm -rf "$work"
    fi
}
trap cleanup EXIT

# Remove inherited Relay settings before defining the test-owned environment.
while IFS='=' read -r name _; do
    if [[ "$name" == NEMO_RELAY_* ]]; then
        unset "$name"
    fi
done < <(env)
if env | grep -q '^NEMO_RELAY_'; then
    echo "failed to clear ambient NEMO_RELAY_* variables" >&2
    exit 1
fi

# Keep a conflicting hook target set so the persistent plugin must preserve its
# explicitly installed hook endpoint instead of inheriting an ambient target.
export NEMO_RELAY_GATEWAY_URL="http://127.0.0.1:1"

export HOME="$work/home"
export CODEX_HOME="$work/codex-home"
export XDG_CONFIG_HOME="$work/xdg"
export XDG_DATA_HOME="$work/data"
export TMPDIR="$work/tmp"
export PATH="$repo_root/target/debug:$PATH"
export OPENAI_API_KEY="relay-e2e-key"
export NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS=1
mkdir -p "$HOME" "$CODEX_HOME" "$XDG_CONFIG_HOME/nemo-relay" "$XDG_DATA_HOME" "$TMPDIR"

provider_ready="$work/provider-ready.json"
provider_log="$work/provider-requests.jsonl"
provider_barrier="$work/provider-barrier"
python3 "$repo_root/scripts/test-support/codex_mock_provider.py" \
    --ready-file "$provider_ready" \
    --log-file "$provider_log" \
    --barrier-dir "$provider_barrier" &
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
EOF

cat >"$XDG_CONFIG_HOME/nemo-relay/plugins.toml" <<'EOF'
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 2

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = "atof"
filename = "events.jsonl"
mode = "append"
EOF

wait_for_relay_port_release() {
    python3 - <<'PY'
import socket
import time

deadline = time.monotonic() + 6
while time.monotonic() < deadline:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        if sock.connect_ex(("127.0.0.1", 47632)) != 0:
            raise SystemExit(0)
    time.sleep(0.1)
raise SystemExit("Relay port 47632 did not become free")
PY
}

wait_for_mcp_initialize() {
    output_path="$1"
    process_pid="$2"
    python3 - "$output_path" "$process_pid" <<'PY'
import os
import sys
import time
from pathlib import Path

output_path = Path(sys.argv[1])
process_pid = int(sys.argv[2])
deadline = time.monotonic() + 25
while time.monotonic() < deadline:
    try:
        if '"serverInfo"' in output_path.read_text(encoding="utf-8", errors="replace"):
            raise SystemExit(0)
    except FileNotFoundError:
        pass
    try:
        os.kill(process_pid, 0)
    except ProcessLookupError:
        raise SystemExit(1)
    time.sleep(0.05)
raise SystemExit(1)
PY
}

wait_for_process_exit() {
    process_pid="$1"
    for _ in $(seq 1 200); do
        kill -0 "$process_pid" 2>/dev/null || return 0
        sleep 0.05
    done
    return 1
}

run_mcp_once() {
    stdout_path="$1"
    stderr_path="$2"
    request_id="$3"
    python3 - "$stdout_path" "$stderr_path" "$request_id" <<'PY'
import os
import signal
import subprocess
import sys

stdout_path, stderr_path, request_id = sys.argv[1:]
message = (
    '{"jsonrpc":"2.0","id":'
    + request_id
    + ',"method":"initialize","params":{"protocolVersion":"2025-06-18"}}\n'
).encode()
with open(stdout_path, "wb") as stdout, open(stderr_path, "wb") as stderr:
    process = subprocess.Popen(
        ["nemo-relay", "mcp"],
        stdin=subprocess.PIPE,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    try:
        process.communicate(message, timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise SystemExit(124)
raise SystemExit(process.returncode)
PY
}

stop_owned_sidecar() {
    owner_path="$1"
    python3 - "$owner_path" <<'PY'
import http.client
import json
import sys
from urllib.parse import urlsplit

with open(sys.argv[1], encoding="utf-8") as source:
    owner = json.load(source)
url = urlsplit(owner["url"])
connection = http.client.HTTPConnection(url.hostname, url.port, timeout=2)
connection.request(
    "POST",
    "/bootstrap/shutdown",
    headers={"X-NeMo-Relay-Bootstrap-Token": owner["shutdown_token"]},
)
response = connection.getresponse()
response.read()
assert response.status == 204, response.status
PY
}

wait_for_relay_port_release
install_dir="$work/plugins"
nemo-relay install codex --install-dir "$install_dir"
nemo-relay doctor --plugin codex --install-dir "$install_dir"

run_codex_ping() {
    stdout="$work/codex-$1.stdout"
    stderr="$work/codex-$1.stderr"
    pgid_path="$work/codex-$1.pgid"
    if ! python3 - "$stdout" "$stderr" "$pgid_path" <<'PY'
import os
from pathlib import Path
import signal
import subprocess
import sys

stdout_path, stderr_path, pgid_path = sys.argv[1:]


def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


try:
    with open(stdout_path, "wb") as stdout, open(stderr_path, "wb") as stderr:
        process = subprocess.Popen(
            ["codex", "exec", "--skip-git-repo-check", "ping"],
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        temporary = Path(f"{pgid_path}.tmp")
        temporary.write_text(str(process.pid), encoding="utf-8")
        temporary.replace(pgid_path)
        try:
            returncode = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            stop_process_group(process)
            raise SystemExit(124)
finally:
    Path(pgid_path).unlink(missing_ok=True)
raise SystemExit(returncode)
PY
    then
        echo "Codex run $1 failed" >&2
        cat "$stdout" >&2
        cat "$stderr" >&2
        return 1
    fi
    grep -qi "pong" "$stdout"
    if ! python3 - "$stderr" <<'PY'
import re
import sys
from collections import Counter

with open(sys.argv[1], encoding="utf-8", errors="replace") as source:
    lines = source.read().splitlines()
failure = re.compile(r"\b(error|failed|failure|panic(?:ked)?|refused|unable|timed?\s*out)\b", re.I)
models_retry = re.compile(
    r"connection refused|connect error|failed to (?:connect|send)|error sending request|retry",
    re.I,
)
unexpected = []
for line in lines:
    if not failure.search(line):
        continue
    if "/models" in line.lower() and models_retry.search(line):
        continue
    lowered = line.lower()
    if (
        "failed to warm featured plugin ids cache" in lowered
        and "chatgpt.com/backend-api/plugins/featured" in lowered
    ):
        continue
    if (
        "codex_core::shell_snapshot: failed to delete shell snapshot" in lowered
        and "kind: notfound" in lowered
        and "no such file or directory" in lowered
    ):
        continue
    unexpected.append(line)
if unexpected:
    print("\n".join(unexpected), file=sys.stderr)
    raise SystemExit(1)

ansi = re.compile(r"\x1b\[[0-9;]*m")
started = Counter()
completed = Counter()
for raw_line in lines:
    line = ansi.sub("", raw_line).strip()
    match = re.search(r"(?:^|\s)hook: (SessionStart|UserPromptSubmit|Stop)( Completed)?$", line)
    if not match:
        continue
    target = completed if match.group(2) else started
    target[match.group(1)] += 1
expected = Counter({"SessionStart": 1, "UserPromptSubmit": 1, "Stop": 1})
if started != expected or completed != expected:
    print(
        f"unexpected Codex hook counts: started={started}, completed={completed}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
    then
        echo "Codex run $1 reported an unexpected error" >&2
        cat "$stderr" >&2
        return 1
    fi
}

run_transparent_codex_ping() {
    stdout="$work/codex-transparent.stdout"
    stderr="$work/codex-transparent.stderr"
    if ! python3 - "$stdout" "$stderr" "$XDG_CONFIG_HOME/nemo-relay/config.toml" "$transparent_project" <<'PY'
import subprocess
import sys

stdout_path, stderr_path, relay_config, project = sys.argv[1:]
with open(stdout_path, "wb") as stdout, open(stderr_path, "wb") as stderr:
    process = subprocess.run(
        [
            "nemo-relay",
            "run",
            "--config",
            relay_config,
            "--",
            "codex",
            "--profile",
            "relay-user-profile",
            "exec",
            "--skip-git-repo-check",
            "ping",
        ],
        stdout=stdout,
        stderr=stderr,
        cwd=project,
        timeout=60,
        check=False,
    )
raise SystemExit(process.returncode)
PY
    then
        echo "transparent Codex run with persistent plugin installed failed" >&2
        cat "$stdout" >&2
        cat "$stderr" >&2
        return 1
    fi
    grep -qi "pong" "$stdout"
    python3 - "$stderr" <<'PY'
import re
import sys
from collections import Counter

lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
ansi = re.compile(r"\x1b\[[0-9;]*m")
started = Counter()
completed = Counter()
for raw_line in lines:
    line = ansi.sub("", raw_line).strip()
    match = re.search(r"(?:^|\s)hook: (SessionStart|UserPromptSubmit|Stop)( Completed)?$", line)
    if not match:
        continue
    (completed if match.group(2) else started)[match.group(1)] += 1
expected = Counter({"SessionStart": 1, "UserPromptSubmit": 1, "Stop": 1})
# The installed plugin remains enabled and its process-local hook exits without forwarding. Codex
# can therefore report both that hook and the wrapper-owned hook, while the ATOF assertions below
# still require exactly one delivered lifecycle stream.
assert all(count <= 2 for count in started.values()), (started, lines)
assert all(count <= 2 for count in completed.values()), (completed, lines)
if started or completed:
    assert set(started) == set(expected) and set(completed) == set(expected), (started, completed, lines)
    assert started == completed, (started, completed, lines)
PY
}

# Transparent mode preserves the selected profile. The installed plugin remains configured, but its
# MCP borrows the wrapper-owned dynamic gateway and its persistent hooks become process-local no-ops.
cat >"$CODEX_HOME/relay-user-profile.config.toml" <<'EOF'
model = "gpt-5.1-codex"
model_reasoning_effort = "low"
EOF
cp "$CODEX_HOME/config.toml" "$work/codex-config-before-transparent.toml"
cp "$CODEX_HOME/relay-user-profile.config.toml" "$work/codex-profile-before-transparent.toml"
: >"$provider_log"
transparent_project="$work/transparent-project"
mkdir -p "$transparent_project"
events="$transparent_project/atof/events.jsonl"
rm -f "$events"
wait_for_relay_port_release
run_transparent_codex_ping
wait_for_relay_port_release
cmp "$CODEX_HOME/config.toml" "$work/codex-config-before-transparent.toml"
cmp "$CODEX_HOME/relay-user-profile.config.toml" "$work/codex-profile-before-transparent.toml"
[[ -z "$(find_sidecar_file 'sidecar-*.owner.json')" ]]
python3 - "$provider_log" "$events" <<'PY'
import json
import sys

requests = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
responses = [row for row in requests if row["method"] == "POST" and row["path"].endswith("/responses")]
assert len(responses) == 1, requests
assert responses[0]["model"] == "gpt-5.1-codex", responses
events = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
turn_starts = [
    event for event in events
    if event.get("kind") == "scope"
    and event.get("name") == "codex-turn"
    and event.get("scope_category") == "start"
]
turn_ends = [
    event for event in events
    if event.get("kind") == "scope"
    and event.get("name") == "codex-turn"
    and event.get("scope_category") == "end"
]
assert len(turn_starts) == len(turn_ends) == 1, (turn_starts, turn_ends)
assert turn_starts[0].get("data", {}).get("hook_event_name", "").lower() == "userpromptsubmit", turn_starts
assert turn_ends[0].get("metadata", {}).get("hook_event_name", "").lower() == "stop", turn_ends
PY
nemo-relay doctor --plugin codex --install-dir "$install_dir"

# Exercise incompatible configuration handling before collecting acceptance events.
export NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS=300
holder_fifo="$work/mcp-holder.stdin"
holder_stdout="$work/mcp-holder.stdout"
holder_stderr="$work/mcp-holder.stderr"
mkfifo "$holder_fifo"
exec 9<>"$holder_fifo"
nemo-relay mcp 9>&- <"$holder_fifo" >"$holder_stdout" 2>"$holder_stderr" &
holder_pid=$!
background_pids+=("$holder_pid")
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' >&9
if ! wait_for_mcp_initialize "$holder_stdout" "$holder_pid"; then
    cat "$holder_stderr" >&2
    exit 1
fi
old_sidecar_pid_file="$(find_sidecar_file 'sidecar-*.owner.json')"
[[ -s "$old_sidecar_pid_file" ]]
old_sidecar_pid="$(read_sidecar_pid "$old_sidecar_pid_file")"
kill -0 "$old_sidecar_pid"
exec 9>&-
if ! wait_for_process_exit "$holder_pid"; then
    echo "original MCP client did not exit after its stdin closed" >&2
    exit 1
fi
wait "$holder_pid"
background_pids=("")
kill -0 "$old_sidecar_pid"

export OPENAI_API_KEY="relay-e2e-key-rotated"
mismatch_stdout="$work/mcp-mismatch.stdout"
mismatch_stderr="$work/mcp-mismatch.stderr"
if run_mcp_once "$mismatch_stdout" "$mismatch_stderr" 2; then
    echo "MCP unexpectedly reused a sidecar with an incompatible credential fingerprint" >&2
    exit 1
fi
grep -qi "different version or persistent configuration" "$mismatch_stderr"

nemo-relay install codex --force --install-dir "$install_dir"
for _ in $(seq 1 100); do
    kill -0 "$old_sidecar_pid" 2>/dev/null || break
    sleep 0.05
done
if kill -0 "$old_sidecar_pid" 2>/dev/null; then
    echo "forced Codex reinstall did not retire the owned sidecar" >&2
    exit 1
fi
wait_for_relay_port_release

replacement_stdout="$work/mcp-replacement.stdout"
replacement_stderr="$work/mcp-replacement.stderr"
run_mcp_once "$replacement_stdout" "$replacement_stderr" 3
grep -q '"serverInfo"' "$replacement_stdout"
replacement_pid_file="$(find_sidecar_file 'sidecar-*.owner.json')"
replacement_owner_file="$(find_sidecar_file 'sidecar-*.owner.json')"
[[ -s "$replacement_pid_file" && -s "$replacement_owner_file" ]]
replacement_pid="$(read_sidecar_pid "$replacement_pid_file")"
[[ "$replacement_pid" != "$old_sidecar_pid" ]]
kill -0 "$replacement_pid"
stop_owned_sidecar "$replacement_owner_file"
wait_for_relay_port_release
rm -f "$replacement_owner_file" "$replacement_pid_file"

# The acceptance counts below cover only real Codex runs, not bootstrap probes.
: >"$provider_log"
events="$XDG_CONFIG_HOME/nemo-relay/atof/events.jsonl"
rm -f "$events"
export NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS=1

for iteration in $(seq 1 10); do
    run_codex_ping "cold-$iteration"
    wait_for_relay_port_release
done

export NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS=300
touch "$provider_barrier/enabled"
run_codex_ping concurrent-1 &
first_pid=$!
background_pids+=("$first_pid")
run_codex_ping concurrent-2 &
second_pid=$!
background_pids+=("$second_pid")

python3 - "$provider_barrier/arrivals" <<'PY'
import sys
import time
from pathlib import Path

arrivals_path = Path(sys.argv[1])
deadline = time.monotonic() + 25
while time.monotonic() < deadline:
    try:
        arrivals = int(arrivals_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        arrivals = 0
    if arrivals >= 2:
        raise SystemExit(0)
    time.sleep(0.05)
raise SystemExit("concurrent Codex requests did not reach the provider within 25 seconds")
PY
[[ "$(cat "$provider_barrier/arrivals")" -eq 2 ]]
sidecar_pid_file="$(find_sidecar_file 'sidecar-*.owner.json')"
[[ -s "$sidecar_pid_file" ]]
shared_sidecar_pid="$(read_sidecar_pid "$sidecar_pid_file")"
[[ "$shared_sidecar_pid" =~ ^[0-9]+$ ]]
kill -0 "$shared_sidecar_pid"
touch "$provider_barrier/release"
wait "$first_pid"
wait "$second_pid"
background_pids=("")
[[ "$(read_sidecar_pid "$sidecar_pid_file")" == "$shared_sidecar_pid" ]]
kill -0 "$shared_sidecar_pid"
python3 - <<'PY'
import socket

with socket.socket() as sock:
    sock.settimeout(0.2)
    assert sock.connect_ex(("127.0.0.1", 47632)) == 0, "shared Relay gateway stopped early"
PY

owner_file="$(find_sidecar_file 'sidecar-*.owner.json')"
stop_owned_sidecar "$owner_file"
if ! wait_for_process_exit "$shared_sidecar_pid"; then
    echo "shared Relay gateway did not exit after the shutdown handshake" >&2
    exit 1
fi
wait_for_relay_port_release
rm -f "$owner_file" "$sidecar_pid_file"

python3 - "$provider_log" "$events" <<'PY'
import collections
import json
import sys

request_path, event_path = sys.argv[1:]
with open(request_path, encoding="utf-8") as source:
    requests = [json.loads(line) for line in source if line.strip()]
response_requests = [
    item for item in requests
    if item.get("method") == "POST" and item.get("path", "").endswith("/responses")
]
model_requests = [
    item for item in requests
    if item.get("method") == "GET" and item.get("path", "").endswith("/models")
]
assert len(response_requests) == 12, (
    f"expected one provider response per Codex run, got {len(response_requests)}; "
    f"all provider requests: {requests}"
)
assert all(item["authorization"] == "Bearer relay-e2e-key-rotated" for item in response_requests), response_requests
assert all(item["relay_client_token"] is None for item in requests), requests
provider_response_ids = {item.get("response_id") for item in response_requests}
assert None not in provider_response_ids, response_requests
assert len(provider_response_ids) == 12, (
    f"expected 12 unique provider response IDs, got {provider_response_ids}"
)
assert len(response_requests) + len(model_requests) == len(requests), requests

with open(event_path, encoding="utf-8") as source:
    events = [json.loads(line) for line in source if line.strip()]
assert events and all(event.get("atof_version") == "0.1" for event in events)

scope_counts = collections.defaultdict(collections.Counter)
for event in events:
    if event.get("kind") == "scope":
        scope_counts[event["uuid"]][event["scope_category"]] += 1
for scope_id, counts in scope_counts.items():
    assert counts == {"start": 1, "end": 1}, f"unbalanced or duplicate scope {scope_id}: {counts}"

turn_starts = [
    event
    for event in events
    if event.get("kind") == "scope"
    and event.get("scope_category") == "start"
    and event.get("category") == "custom"
    and event.get("name") == "codex-turn"
]
turn_ends = [
    event
    for event in events
    if event.get("kind") == "scope"
    and event.get("scope_category") == "end"
    and event.get("category") == "custom"
    and event.get("name") == "codex-turn"
]
session_ids = [event.get("metadata", {}).get("session_id") for event in turn_starts]
session_counts = collections.Counter(session_ids)
summary = sorted(
    {
        (
            event.get("kind"),
            event.get("scope_category"),
            event.get("category"),
            event.get("name"),
        )
        for event in events
    }
)
assert len(turn_starts) == 12, (
    f"expected exactly one Codex turn start per run, got {len(turn_starts)}; "
    f"sessions: {session_counts}; event shapes: {summary}"
)
assert None not in session_ids
assert len(set(session_ids)) == 12, f"Codex sessions were not isolated: {session_ids}"
assert len(turn_ends) == 12, f"expected exactly one Codex turn end per run, got {len(turn_ends)}"
assert {event["uuid"] for event in turn_starts} == {event["uuid"] for event in turn_ends}
assert all(
    event.get("data", {}).get("hook_event_name", "").lower() == "userpromptsubmit"
    for event in turn_starts
), turn_starts
assert all(
    event.get("metadata", {}).get("hook_event_name", "").lower() == "userpromptsubmit"
    for event in turn_starts
), turn_starts
# Stop closes the turn, but the semantic output remains the final provider response.
assert all(
    event.get("metadata", {}).get("hook_event_name", "").lower() == "stop"
    for event in turn_ends
), turn_ends
assert all("pong" in json.dumps(event.get("data")) for event in turn_ends), turn_ends
turn_start_sessions = {
    event["uuid"]: event.get("metadata", {}).get("session_id") for event in turn_starts
}
assert all(
    event.get("metadata", {}).get("session_id") == turn_start_sessions[event["uuid"]]
    for event in turn_ends
), turn_ends

llm_starts = [
    event for event in events
    if event.get("category") == "llm" and event.get("scope_category") == "start"
]
llm_ends = [
    event for event in events
    if event.get("category") == "llm" and event.get("scope_category") == "end"
]
assert len(llm_starts) == 12, f"expected 12 LLM starts, got {len(llm_starts)}"
assert len(llm_ends) == 12, f"expected 12 LLM ends, got {len(llm_ends)}"
assert all("pong" in json.dumps(event) for event in llm_ends), llm_ends
llm_start_by_uuid = {event["uuid"]: event for event in llm_starts}
llm_end_by_uuid = {event["uuid"]: event for event in llm_ends}
assert len(llm_start_by_uuid) == 12, f"duplicate LLM starts: {llm_starts}"
assert llm_start_by_uuid.keys() == llm_end_by_uuid.keys(), (
    f"unmatched LLM scopes: starts={llm_start_by_uuid.keys()}, ends={llm_end_by_uuid.keys()}"
)
llm_starts_by_turn = collections.defaultdict(list)
llm_ends_by_turn = collections.defaultdict(list)
for event in llm_starts:
    llm_starts_by_turn[event.get("parent_uuid")].append(event)
for event in llm_ends:
    llm_ends_by_turn[event.get("parent_uuid")].append(event)
turn_ids = {event["uuid"] for event in turn_starts}
assert set(llm_starts_by_turn) == turn_ids, (
    f"LLM starts were not attached to every turn: {llm_starts_by_turn}"
)
assert set(llm_ends_by_turn) == turn_ids, (
    f"LLM ends were not attached to every turn: {llm_ends_by_turn}"
)
assert all(len(children) == 1 for children in llm_starts_by_turn.values()), llm_starts_by_turn
assert all(len(children) == 1 for children in llm_ends_by_turn.values()), llm_ends_by_turn


def strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)
    elif isinstance(value, str):
        yield value


def response_id(event):
    matches = provider_response_ids.intersection(strings(event))
    assert len(matches) == 1, (
        f"expected exactly one provider response ID in event, got {matches}: {event}"
    )
    return next(iter(matches))


turn_end_by_uuid = {event["uuid"]: event for event in turn_ends}
captured_response_ids = set()
for turn_id in turn_ids:
    llm_start = llm_starts_by_turn[turn_id][0]
    llm_end = llm_ends_by_turn[turn_id][0]
    assert llm_start["uuid"] == llm_end["uuid"]
    captured_id = response_id(llm_end)
    assert response_id(turn_end_by_uuid[turn_id]) == captured_id
    captured_response_ids.add(captured_id)
assert captured_response_ids == provider_response_ids, (
    f"captured/provider response IDs differ: captured={captured_response_ids}, "
    f"provider={provider_response_ids}"
)
print(
    f"validated 12 captured turns and 12 provider responses; "
    f"{len(model_requests)} /models requests reached Relay"
)
PY

nemo-relay uninstall codex --install-dir "$install_dir"
echo "Codex plugin E2E passed: 10 cold and 2 concurrent runs each invoked and completed SessionStart, UserPromptSubmit, and Stop exactly once; every Stop closed one ATOF turn with the matching provider response; pre-MCP /models retries, featured-plugin cache warnings, and concurrent shell-snapshot cleanup races were ignored"
