#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="${1:-}"
image="${PHASE1_COMPAT_IMAGE:-python:3.11-bookworm}"
platform="${PHASE1_COMPAT_PLATFORM:-linux/amd64}"
hermes_repository="${HERMES_REPOSITORY:-https://github.com/bbednarski9/hermes-agent.git}"
hermes_ref="${HERMES_REF:-feat/relay-native-plugin-init}"
hermes_commit="${HERMES_COMMIT:-efb63e714abc436af88af9b0d6734751c199aa6d}"

if [[ -z "$run_root" || "$run_root" != /* ]]; then
  echo "usage: $0 /absolute/prepared-run-root" >&2
  exit 2
fi
for required in \
  "$run_root/runtime/plugins.toml" \
  "$run_root/runtime/provenance.json" \
  "$run_root/runtime/switchyard-plugin/relay-plugin.toml"; do
  [[ -f "$required" ]] || { echo "missing prepared runtime file: $required" >&2; exit 1; }
done
command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }
docker info >/dev/null
case "$platform" in
  linux/amd64) expected_architecture=x86_64 ;;
  linux/arm64) expected_architecture=aarch64 ;;
  *) echo "PHASE1_COMPAT_PLATFORM must be linux/amd64 or linux/arm64" >&2; exit 2 ;;
esac
prepared_architecture="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["nemo_relay"].get("architecture", "x86_64"))' "$run_root/runtime/provenance.json")"
[[ "$prepared_architecture" == "$expected_architecture" ]] || {
  echo "prepared Relay architecture $prepared_architecture does not match $platform" >&2
  exit 2
}

artifacts="$run_root/artifacts/offline-compatibility"
if [[ -e "$artifacts" ]]; then
  [[ -d "$artifacts" ]] || { echo "artifact path is not a directory: $artifacts" >&2; exit 1; }
  [[ -z "$(find "$artifacts" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    echo "offline compatibility artifacts already exist: $artifacts" >&2
    exit 1
  }
else
  mkdir -m 0700 "$artifacts"
fi

docker run --rm \
  --platform "$platform" \
  --volume "$example_root:/example:ro" \
  --volume "$run_root/runtime:/runtime:ro" \
  --volume "$run_root/runtime/switchyard-plugin:/opt/relay-plugins/nvidia.switchyard:ro" \
  --volume "$artifacts:/logs/agent/direct-hermes" \
  "$image" \
  bash -lc '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    export HERMES_HOME=/tmp/hermes
    export HERMES_NEMO_RELAY_PLUGINS_TOML=/runtime/plugins.toml
    export SWITCHYARD_PROVIDER_AUTHORIZATION="Bearer phase1-offline-secret-value"
    apt-get update
    apt-get install -y --no-install-recommends build-essential ca-certificates curl git ripgrep xz-utils
    git clone --no-tags --branch "'"$hermes_ref"'" "'"$hermes_repository"'" /tmp/hermes-agent-src
    git -C /tmp/hermes-agent-src fetch --depth 1 origin "'"$hermes_commit"'"
    git -C /tmp/hermes-agent-src checkout --detach "'"$hermes_commit"'"
    # The installer treats ffmpeg as optional, but a root-owned Debian smoke
    # otherwise installs ~500 MB of unrelated TTS/video packages. Advertise a
    # command only during installation; it is not on PATH for the runtime.
    mkdir /tmp/hermes-install-path
    ln -s /bin/true /tmp/hermes-install-path/ffmpeg
    HERMES_INSTALL_DIR=/tmp/hermes-agent-src \
      PATH=/tmp/hermes-install-path:$PATH \
      bash /tmp/hermes-agent-src/scripts/install.sh \
        --skip-setup --skip-browser --no-skills \
        --dir /tmp/hermes-agent-src \
        --branch "'"$hermes_ref"'" \
        --commit "'"$hermes_commit"'" --force-commit
    test "$(git -C /tmp/hermes-agent-src rev-parse HEAD)" = "'"$hermes_commit"'"
    cd /tmp/hermes-agent-src
    UV_PROJECT_ENVIRONMENT=/tmp/hermes-agent-src/venv \
      /tmp/hermes/bin/uv sync --frozen --extra all
    cd /
    /tmp/hermes-agent-src/venv/bin/python -c \
      "import importlib.metadata as m; assert m.version(\"nemo-relay\") == \"0.7.0\""
    relay_wheel="$(find /runtime/wheels -maxdepth 1 -type f -name "nemo_relay-0.7.0-*.whl" -print)"
    test -n "$relay_wheel"
    expected_wheel_sha="$(python3 -c "import json; print(json.load(open(\"/runtime/provenance.json\"))[\"nemo_relay\"][\"wheel_sha256\"])")"
    test "$(sha256sum "$relay_wheel" | cut -d" " -f1)" = "$expected_wheel_sha"
    /tmp/hermes/bin/uv pip install \
      --python /tmp/hermes-agent-src/venv/bin/python \
      --force-reinstall --no-deps "$relay_wheel"
    python3 /example/scripts/fake_openai_upstream.py \
      --token phase1-offline-secret-value \
      --request-log /logs/agent/direct-hermes/provider-requests.jsonl &
    provider_pid=$!
    python3 /example/scripts/fake_otlp_collector.py \
      --request-log /logs/agent/direct-hermes/otlp-requests.jsonl &
    otlp_pid=$!
    cleanup() {
      kill "$provider_pid" "$otlp_pid" >/dev/null 2>&1 || true
      wait "$provider_pid" "$otlp_pid" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT
    for endpoint in http://127.0.0.1:8000/healthz http://127.0.0.1:4318/healthz; do
      for _ in $(seq 1 50); do
        curl --fail --silent "$endpoint" >/dev/null && break
        sleep 0.1
      done
      curl --fail --silent "$endpoint" >/dev/null
    done
    PYTHONPATH=/tmp/hermes-agent-src \
      /tmp/hermes-agent-src/venv/bin/python \
      /example/scripts/offline_compatibility_smoke.py \
        --plugins /runtime/plugins.toml \
        --artifacts /logs/agent/direct-hermes \
        --request-log /logs/agent/direct-hermes/provider-requests.jsonl
    test -s /logs/agent/direct-hermes/otlp-requests.jsonl
    if grep -R -F phase1-offline-secret-value /logs/agent/direct-hermes >/dev/null; then
      echo "offline secret leaked into persisted evidence" >&2
      exit 1
    fi
  '

python3 - "$artifacts" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
result = json.loads((root / "offline-smoke.json").read_text())
if result.get("status") != "passed":
    raise SystemExit("offline compatibility smoke did not pass")
print(json.dumps(result, indent=2))
PY
