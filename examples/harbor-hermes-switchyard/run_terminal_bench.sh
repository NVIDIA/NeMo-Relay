#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_root="${1:-}"
task_name="${TASK_NAME:-adaptive-rejection-sampler}"
upstream_auth_env="SWITCHYARD_PROVIDER_AUTHORIZATION"
fail_closed_openai_base_url="http://127.0.0.1:9/v1"
phoenix_base="${PHOENIX_BASE_URL:-}"
phoenix_project="${PHOENIX_PROJECT:-harbor-hermes-switchyard-phase1}"
eval_cohort="${EVAL_COHORT:-harbor-hermes-switchyard-phase1}"
default_harbor_bin="$example_root/.venv/bin/harbor"
default_python_bin="$example_root/.venv/bin/python"
harbor_bin="${HARBOR_BIN:-$default_harbor_bin}"
python_bin="${EVAL_PYTHON:-$default_python_bin}"
expected_harbor_version="0.18.0"
eval_phase="${EVAL_PHASE:-phase1}"
tbench_dataset_path="${TBENCH_DATASET_PATH:-}"
switchyard_bundle="${SWITCHYARD_BUNDLE:-}"
relay_wheel="${RELAY_WHEEL:-}"
relay_architecture="${RELAY_ARCHITECTURE:-x86_64}"
plugin_config_template="${PLUGIN_CONFIG_TEMPLATE:-$example_root/config/plugins.toml.in}"
agent_timeout_multiplier="${AGENT_TIMEOUT_MULTIPLIER:-3}"
agent_setup_timeout_multiplier="${AGENT_SETUP_TIMEOUT_MULTIPLIER:-6}"
environment_build_timeout_multiplier="${ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER:-6}"
collector_image="${OTEL_COLLECTOR_IMAGE:-otel/opentelemetry-collector-contrib:0.135.0}"
inject_post_response_failure="${INJECT_POST_RESPONSE_FAILURE:-false}"
hermetic_runtime_dir="${HERMETIC_RUNTIME_DIR:-}"
hermetic_runtime_sha256="${HERMETIC_RUNTIME_SHA256:-}"
harbor_force_build="${HARBOR_FORCE_BUILD:-true}"

if [[ -z "$run_root" || "$run_root" != /* ]]; then
  echo "usage: $0 /absolute/new-run-root" >&2
  exit 2
fi
if [[ -e "$run_root" ]]; then
  echo "run root already exists: $run_root" >&2
  exit 2
fi
for required in "$plugin_config_template" "$phoenix_base"; do
  [[ -n "$required" ]] || {
    echo "PLUGIN_CONFIG_TEMPLATE and PHOENIX_BASE_URL are required" >&2
    exit 2
  }
done
if [[ ! -f "$plugin_config_template" ]]; then
  echo "plugin configuration template is missing: $plugin_config_template" >&2
  exit 2
fi
for dependency in curl docker "$harbor_bin" "$python_bin"; do
  command -v "$dependency" >/dev/null || {
    echo "missing required command: $dependency" >&2
    exit 1
  }
done
observed_harbor_version="$($python_bin -c 'import importlib.metadata; print(importlib.metadata.version("harbor"))')"
if [[ "$observed_harbor_version" != "$expected_harbor_version" ]]; then
  echo "Harbor $expected_harbor_version is required; $python_bin provides $observed_harbor_version" >&2
  exit 2
fi
observed_harbor_cli_version="$($harbor_bin --version)"
if [[ "$observed_harbor_cli_version" != "$expected_harbor_version" ]]; then
  echo "Harbor CLI $expected_harbor_version is required; $harbor_bin reports $observed_harbor_cli_version" >&2
  exit 2
fi
if [[ ! "$eval_phase" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "EVAL_PHASE must contain only lowercase letters, digits, and hyphens" >&2
  exit 2
fi
dataset_args=(--dataset terminal-bench@2.0)
if [[ -n "$tbench_dataset_path" ]]; then
  if [[ "$tbench_dataset_path" != /* || ! -d "$tbench_dataset_path" ]]; then
    echo "TBENCH_DATASET_PATH must be an absolute local dataset directory" >&2
    exit 2
  fi
  dataset_args=(--path "$tbench_dataset_path")
elif [[ "$eval_phase" == "phase2" ]]; then
  echo "Phase 2 requires TBENCH_DATASET_PATH and never resolves the remote registry" >&2
  exit 2
fi
if [[ -z "${!upstream_auth_env:-}" ]]; then
  echo "required provider authorization environment variable is unset: $upstream_auth_env" >&2
  exit 2
fi
if [[ "$relay_architecture" != "x86_64" && "$relay_architecture" != "aarch64" ]]; then
  echo "RELAY_ARCHITECTURE must be x86_64 or aarch64" >&2
  exit 2
fi
if [[ ( -n "$hermetic_runtime_dir" && -z "$hermetic_runtime_sha256" ) || \
      ( -z "$hermetic_runtime_dir" && -n "$hermetic_runtime_sha256" ) ]]; then
  echo "HERMETIC_RUNTIME_DIR and HERMETIC_RUNTIME_SHA256 must be supplied together" >&2
  exit 2
fi
if [[ -n "$hermetic_runtime_dir" ]]; then
  if [[ "$hermetic_runtime_dir" != /* || ! -f "$hermetic_runtime_dir/payload.json" ]]; then
    echo "HERMETIC_RUNTIME_DIR must be an absolute prepared runtime directory" >&2
    exit 2
  fi
  if [[ ! "$hermetic_runtime_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "HERMETIC_RUNTIME_SHA256 must be a lowercase SHA-256 digest" >&2
    exit 2
  fi
fi
if [[ "$harbor_force_build" != "true" && "$harbor_force_build" != "false" ]]; then
  echo "HARBOR_FORCE_BUILD must be true or false" >&2
  exit 2
fi

docker info >/dev/null
curl --fail --silent --show-error \
  --connect-timeout 5 --max-time 10 \
  --retry 2 --retry-all-errors --retry-delay 2 \
  "$phoenix_base" >/dev/null

temporary_build=""
temporary_secret_dir=""
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
  if [[ -n "$temporary_secret_dir" && -d "$temporary_secret_dir" ]]; then
    rm -rf "$temporary_secret_dir"
  fi
  return "$status"
}
trap cleanup EXIT

host_temporary_root="$(cd "${TMPDIR:-/tmp}" && pwd -P)"
case "$host_temporary_root/" in
  "$run_root/"*)
    echo "Host temporary directory must be outside the task run root" >&2
    exit 2
    ;;
esac
temporary_secret_dir="$(mktemp -d "$host_temporary_root/harbor-phase2-secret.XXXXXX")"
chmod 0700 "$temporary_secret_dir"
provider_authorization_file="$temporary_secret_dir/switchyard-provider-authorization"
(umask 077; printf '%s' "${!upstream_auth_env}" >"$provider_authorization_file")
chmod 0600 "$provider_authorization_file"
provider_authorization_target="/run/secrets/switchyard-provider-authorization"
mounts_json="$($python_bin -c '
import json, sys
mounts = [{
    "type": "bind",
    "source": sys.argv[1],
    "target": sys.argv[2],
    "read_only": True,
    "bind": {"create_host_path": False},
}]
if sys.argv[3]:
    mounts.append({
        "type": "bind",
        "source": sys.argv[3],
        "target": "/opt/hermes-runtime",
        "read_only": True,
        "bind": {"create_host_path": False},
    })
print(json.dumps(mounts, separators=(",", ":")))
' "$provider_authorization_file" "$provider_authorization_target" "$hermetic_runtime_dir")"

if [[ -z "$switchyard_bundle" ]]; then
  temporary_build="$(mktemp -d "$(dirname "$run_root")/.switchyard-build.XXXXXX")"
  switchyard_bundle="$temporary_build/bundle"
  SWITCHYARD_TARGET_ARCHITECTURE="$relay_architecture" \
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

prepare_args=(
  "$example_root/scripts/prepare_runtime.py"
  --run-root "$run_root"
  --switchyard-bundle "$switchyard_bundle"
  --relay-architecture "$relay_architecture"
  --plugin-config-template "$plugin_config_template"
  --openinference-endpoint "$openinference_endpoint"
  --phoenix-project "$phoenix_project"
  --eval-cohort "$eval_cohort"
)
if [[ -n "$relay_wheel" ]]; then
  prepare_args+=(--relay-wheel "$relay_wheel")
fi
"$python_bin" "${prepare_args[@]}" >"$run_root.prepare.log"

hermes_caller_model="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["routing"]["hermes_caller_model"])' "$run_root/runtime/provenance.json")"

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
job_name="${eval_phase}-${task_name}-$(date -u +%Y%m%dT%H%M%SZ)"
agent_hosts=(--allow-agent-host host.docker.internal)
while IFS= read -r upstream_host; do
  if [[ -n "$upstream_host" && "$upstream_host" != "host.docker.internal" ]]; then
    agent_hosts+=(--allow-agent-host "$upstream_host")
  fi
done < <("$python_bin" -c '
import json, sys
from urllib.parse import urlsplit
values = json.load(open(sys.argv[1]))["routing"]
print("\n".join(sorted({urlsplit(value).hostname for key, value in values.items() if key.endswith("_base_url")})))
' "$run_root/runtime/provenance.json")
agent_kwargs=()
if [[ "$inject_post_response_failure" == "true" ]]; then
  agent_kwargs+=(--ak inject_post_response_failure=true)
elif [[ "$inject_post_response_failure" != "false" ]]; then
  echo "INJECT_POST_RESPONSE_FAILURE must be true or false" >&2
  exit 2
fi
if [[ -n "$hermetic_runtime_dir" ]]; then
  agent_kwargs+=(
    --ak "hermetic_runtime_dir=$hermetic_runtime_dir"
    --ak "hermetic_runtime_sha256=$hermetic_runtime_sha256"
  )
fi
harbor_build_args=()
if [[ "$harbor_force_build" == "true" ]]; then
  harbor_build_args+=(--force-build)
fi
(
  "$harbor_bin" run \
    "${dataset_args[@]}" \
    --include-task-name "$task_name" \
    --n-tasks 1 \
    --agent harbor_hermes_agent:HarborHermesAgent \
    --model "openai/$hermes_caller_model" \
    --ak "repository_url=https://github.com/bbednarski9/hermes-agent.git" \
    --ak "repository_ref=feat/relay-native-plugin-init" \
    --ak "commit=a3d472f0e6bdc376df87b1436a461c4796db6747" \
    --ak "relay_config_path=$run_root/runtime/plugins.toml" \
    --ak "switchyard_bundle_dir=$run_root/runtime/switchyard-plugin" \
    --ak "relay_wheel_path=$relay_wheel_path" \
    --ak "relay_wheel_sha256=$relay_wheel_sha256" \
    --ak "relay_architecture=$relay_architecture" \
    "${agent_kwargs[@]}" \
    --ae 'OPENAI_API_KEY=${OPENAI_API_KEY}' \
    --ae "OPENAI_BASE_URL=$fail_closed_openai_base_url" \
    --ae 'OPENROUTER_API_KEY=relay-intercepted' \
    --ae "OPENROUTER_BASE_URL=$fail_closed_openai_base_url" \
    --mounts "$mounts_json" \
    "${agent_hosts[@]}" \
    --artifact /logs/agent/direct-hermes \
    --agent-include-logs hermes-session.jsonl \
    --agent-include-logs hermes.txt \
    --job-name "$job_name" \
    --jobs-dir "$run_root/jobs" \
    --n-concurrent 1 \
    --n-attempts 1 \
    --agent-timeout-multiplier "$agent_timeout_multiplier" \
    --agent-setup-timeout-multiplier "$agent_setup_timeout_multiplier" \
    --environment-build-timeout-multiplier "$environment_build_timeout_multiplier" \
    "${harbor_build_args[@]}" \
    --yes
) >"$run_root/harbor.log" 2>&1

docker stop --time 10 "$collector_name" >/dev/null
collector_running=0

direct_result="$($python_bin - "$run_root/jobs/$job_name" <<'PY'
import pathlib
import sys

matches = sorted(
    pathlib.Path(sys.argv[1]).glob(
        "*/artifacts/logs/agent/direct-hermes/direct-hermes-result.json"
    )
)
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
)
if [[ "$inject_post_response_failure" == "true" ]]; then
  validation_args+=(--expect-late-failure)
fi
"$python_bin" "${validation_args[@]}" >"$run_root/validation.log"

if ! "$python_bin" "$example_root/scripts/upload_openinference.py" \
  --openinference "$openinference" \
  --phoenix-url "$phoenix_base" \
  --project "$phoenix_project" \
  --output "$artifact_root/phoenix-upload.json" \
  >"$run_root/phoenix-upload.log"; then
  "$python_bin" - "$artifact_root/phoenix-upload.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({"status": "failed", "error": "Phoenix upload command failed"}, indent=2) + "\n")
PY
fi

"$python_bin" - "$artifact_root" "$run_root" "$job_name" "$task_name" <<'PY'
import json
import pathlib
import sys

artifacts = pathlib.Path(sys.argv[1])
run_root = pathlib.Path(sys.argv[2])
summary = {
    "schema_version": "harbor-hermes-switchyard.task-summary.v1",
    "job_name": sys.argv[3],
    "task_name": sys.argv[4],
    "artifacts": str(artifacts),
    "validation": json.loads((artifacts / "validation.json").read_text()),
    "phoenix_upload": json.loads((artifacts / "phoenix-upload.json").read_text()),
}
integration = summary["validation"].setdefault("integration", {"status": "passed", "errors": [], "warnings": []})
integration_errors = list(integration.get("errors", []))
integration["phoenix_upload"] = summary["phoenix_upload"]
if summary["phoenix_upload"].get("status") != "passed":
    integration_errors.append("Phoenix upload did not pass")
integration["errors"] = sorted(set(integration_errors))
integration["status"] = "passed" if not integration["errors"] else "failed"
(artifacts / "validation.json").write_text(json.dumps(summary["validation"], indent=2, sort_keys=True) + "\n")
summary["benchmark_completion"] = summary["validation"].get("benchmark", {})
summary["integration_validation"] = integration
benchmark_complete = summary["benchmark_completion"].get(
    "status", summary["validation"].get("status")
) == "passed"
summary["status"] = "passed" if benchmark_complete else "failed"
if summary["status"] != "passed":
    raise SystemExit("benchmark completion did not pass")
(run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "Phase 1 task passed: $task_name"
echo "Run root: $run_root"
echo "Artifacts: $artifact_root"
