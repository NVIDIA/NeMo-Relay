#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_root="${1:-}"
task_name="${TASK_NAME:-adaptive-rejection-sampler}"
target_model="${TARGET_MODEL:-}"
upstream_base_url="${UPSTREAM_BASE_URL:-}"
upstream_auth_env="${UPSTREAM_AUTH_ENV:-SWITCHYARD_PROVIDER_AUTHORIZATION}"
phoenix_base="${PHOENIX_BASE_URL:-}"
phoenix_project="${PHOENIX_PROJECT:-harbor-hermes-switchyard-phase1}"
eval_cohort="${EVAL_COHORT:-harbor-hermes-switchyard-phase1}"
harbor_bin="${HARBOR_BIN:-harbor}"
python_bin="${PHASE1_PYTHON:-python3}"
switchyard_bundle="${SWITCHYARD_BUNDLE:-}"
relay_wheel="${RELAY_WHEEL:-}"
agent_timeout_multiplier="${AGENT_TIMEOUT_MULTIPLIER:-3}"
agent_setup_timeout_multiplier="${AGENT_SETUP_TIMEOUT_MULTIPLIER:-6}"
environment_build_timeout_multiplier="${ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER:-6}"
collector_image="${OTEL_COLLECTOR_IMAGE:-otel/opentelemetry-collector-contrib:0.135.0}"
inject_post_response_failure="${INJECT_POST_RESPONSE_FAILURE:-false}"

if [[ -z "$run_root" || "$run_root" != /* ]]; then
  echo "usage: $0 /absolute/new-run-root" >&2
  exit 2
fi
if [[ -e "$run_root" ]]; then
  echo "run root already exists: $run_root" >&2
  exit 2
fi
for required in "$target_model" "$upstream_base_url" "$phoenix_base"; do
  [[ -n "$required" ]] || {
    echo "TARGET_MODEL, UPSTREAM_BASE_URL, and PHOENIX_BASE_URL are required" >&2
    exit 2
  }
done
for dependency in curl docker "$harbor_bin" "$python_bin"; do
  command -v "$dependency" >/dev/null || {
    echo "missing required command: $dependency" >&2
    exit 1
  }
done
if [[ -z "${!upstream_auth_env:-}" ]]; then
  echo "required provider authorization environment variable is unset: $upstream_auth_env" >&2
  exit 2
fi

docker info >/dev/null
curl --fail --silent --show-error --max-time 10 "$phoenix_base" >/dev/null

temporary_build=""
collector_name=""
collector_running=0
cleanup() {
  local status=$?
  if [[ "$collector_running" == 1 ]]; then
    docker stop --time 10 "$collector_name" >/dev/null 2>&1 || true
  fi
  if [[ -n "$temporary_build" && -d "$temporary_build" ]]; then
    rm -rf "$temporary_build"
  fi
  return "$status"
}
trap cleanup EXIT

if [[ -z "$switchyard_bundle" ]]; then
  temporary_build="$(mktemp -d "$(dirname "$run_root")/.phase1-switchyard-build.XXXXXX")"
  switchyard_bundle="$temporary_build/bundle"
  "$example_root/scripts/build_switchyard_plugin.sh" "$switchyard_bundle"
fi

free_port="$($python_bin - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
openinference_endpoint="http://host.docker.internal:$free_port/v1/traces"
upstream_host="$($python_bin -c 'import sys; from urllib.parse import urlsplit; print(urlsplit(sys.argv[1]).hostname or "")' "$upstream_base_url")"

prepare_args=(
  "$example_root/scripts/prepare_runtime.py"
  --run-root "$run_root"
  --switchyard-bundle "$switchyard_bundle"
  --upstream-base-url "$upstream_base_url"
  --upstream-auth-env "$upstream_auth_env"
  --target-model "$target_model"
  --openinference-endpoint "$openinference_endpoint"
  --phoenix-project "$phoenix_project"
  --eval-cohort "$eval_cohort"
)
if [[ -n "$relay_wheel" ]]; then
  prepare_args+=(--relay-wheel "$relay_wheel")
fi
"$python_bin" "${prepare_args[@]}" >"$run_root.prepare.log"

relay_wheel_sha256="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["nemo_relay"]["wheel_sha256"])' "$run_root/runtime/provenance.json")"
relay_wheel_path="$($python_bin -c 'import json,pathlib,sys; p=json.load(open(sys.argv[1])); print(pathlib.Path(sys.argv[1]).parent / "wheels" / p["nemo_relay"]["wheel"])' "$run_root/runtime/provenance.json")"

"$python_bin" "$example_root/scripts/verify_harbor_hermes_compat.py" \
  --bridge "$example_root/agents/harbor_hermes_agent.py" \
  --relay-config "$run_root/runtime/plugins.toml" \
  --output "$run_root/artifacts/harbor-hermes-compatibility.json" \
  >"$run_root/compatibility.log"

mkdir -m 0700 "$run_root/telemetry"
collector_name="harbor-hermes-switchyard-$($python_bin -c 'import uuid; print(uuid.uuid4().hex[:12])')"
docker run --detach --rm \
  --name "$collector_name" \
  --publish "127.0.0.1:$free_port:4318" \
  --volume "$example_root/config/otel-collector.yaml:/etc/otelcol-contrib/config.yaml:ro" \
  --volume "$run_root/telemetry:/artifacts" \
  "$collector_image" \
  --config=/etc/otelcol-contrib/config.yaml >"$run_root/collector.container-id"
collector_running=1

export PYTHONPATH="$example_root/agents${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY="relay-managed-placeholder"
job_name="phase1-${task_name}-$(date -u +%Y%m%dT%H%M%SZ)"
agent_hosts=(--allow-agent-host host.docker.internal)
if [[ -n "$upstream_host" && "$upstream_host" != "host.docker.internal" ]]; then
  agent_hosts+=(--allow-agent-host "$upstream_host")
fi
agent_kwargs=()
validation_expectations=()
if [[ "$inject_post_response_failure" == "true" ]]; then
  agent_kwargs+=(--ak inject_post_response_failure=true)
  validation_expectations+=(--expect-late-failure)
elif [[ "$inject_post_response_failure" != "false" ]]; then
  echo "INJECT_POST_RESPONSE_FAILURE must be true or false" >&2
  exit 2
fi
(
  "$harbor_bin" run \
    --dataset terminal-bench@2.0 \
    --include-task-name "$task_name" \
    --n-tasks 1 \
    --agent harbor_hermes_agent:HarborHermesAgent \
    --model "openai/$target_model" \
    --ak "repository_url=https://github.com/bbednarski9/hermes-agent.git" \
    --ak "repository_ref=feat/relay-native-plugin-init" \
    --ak "commit=a07830e086b3055e313b74cc0c8fd5326a4c2c00" \
    --ak "relay_config_path=$run_root/runtime/plugins.toml" \
    --ak "switchyard_bundle_dir=$run_root/runtime/switchyard-plugin" \
    --ak "relay_wheel_path=$relay_wheel_path" \
    --ak "relay_wheel_sha256=$relay_wheel_sha256" \
    "${agent_kwargs[@]}" \
    --ae "$upstream_auth_env=${!upstream_auth_env}" \
    --ae OPENAI_API_KEY=relay-managed-placeholder \
    "${agent_hosts[@]}" \
    --artifact /logs/agent/direct-hermes \
    --agent-include-logs 'direct-hermes/**' \
    --agent-include-logs hermes-session.jsonl \
    --agent-include-logs hermes.txt \
    --job-name "$job_name" \
    --jobs-dir "$run_root/jobs" \
    --n-concurrent 1 \
    --n-attempts 1 \
    --agent-timeout-multiplier "$agent_timeout_multiplier" \
    --agent-setup-timeout-multiplier "$agent_setup_timeout_multiplier" \
    --environment-build-timeout-multiplier "$environment_build_timeout_multiplier" \
    --force-build \
    --yes
) >"$run_root/harbor.log" 2>&1

docker stop --time 10 "$collector_name" >/dev/null
collector_running=0

direct_result="$($python_bin - "$run_root/jobs/$job_name" <<'PY'
import pathlib
import sys

matches = sorted(pathlib.Path(sys.argv[1]).glob("**/direct-hermes-result.json"))
if len(matches) != 1:
    raise SystemExit(f"expected one direct Hermes result, found {len(matches)}")
print(matches[0])
PY
)"
if [[ -z "$direct_result" ]]; then
  echo "direct Hermes result discovery returned an empty path" >&2
  exit 1
fi
artifact_root="$(dirname "$direct_result")"
openinference="$run_root/telemetry/trajectory.openinference.json"

validation_args=(
  "$example_root/scripts/validate_run.py"
  --artifacts "$artifact_root"
  --provenance "$run_root/runtime/provenance.json"
  --openinference "$openinference"
  --harbor-job-dir "$run_root/jobs/$job_name"
  --scan-root "$run_root/jobs/$job_name"
  --secret-env "$upstream_auth_env"
  --output "$artifact_root/validation.json"
  "${validation_expectations[@]}"
)
"$python_bin" "${validation_args[@]}" >"$run_root/validation.log"

"$python_bin" "$example_root/scripts/upload_openinference.py" \
  --openinference "$openinference" \
  --phoenix-url "$phoenix_base" \
  --project "$phoenix_project" \
  --output "$artifact_root/phoenix-upload.json" \
  >"$run_root/phoenix-upload.log"

"$python_bin" - "$artifact_root" "$run_root" "$job_name" "$task_name" <<'PY'
import json
import pathlib
import sys

artifacts = pathlib.Path(sys.argv[1])
run_root = pathlib.Path(sys.argv[2])
summary = {
    "schema_version": "harbor-hermes-switchyard.task-summary.v1",
    "status": "passed",
    "job_name": sys.argv[3],
    "task_name": sys.argv[4],
    "artifacts": str(artifacts),
    "validation": json.loads((artifacts / "validation.json").read_text()),
    "phoenix_upload": json.loads((artifacts / "phoenix-upload.json").read_text()),
}
if summary["validation"].get("status") != "passed" or summary["phoenix_upload"].get("status") != "passed":
    raise SystemExit("task evidence gates did not pass")
(run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "Phase 1 task passed: $task_name"
echo "Run root: $run_root"
echo "Artifacts: $artifact_root"
