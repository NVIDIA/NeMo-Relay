#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
set +x

env_file="${1:-}"
if [[ -z "$env_file" || ! -f "$env_file" ]]; then
  echo "usage: $0 .env" >&2
  exit 2
fi
env_file="$(cd "$(dirname "$env_file")" && pwd)/$(basename "$env_file")"
if mode="$(stat -f '%Lp' "$env_file" 2>/dev/null)"; then
  :
else
  mode="$(stat -c '%a' "$env_file")"
fi
if [[ "$mode" != "600" ]]; then
  echo "Phase 2 environment file must have mode 0600: $env_file" >&2
  exit 2
fi
if grep -Eq '^(INFERENCE_SECRETS_FILE|NV_INFERENCEHUB_ENDPOINT|NV_INFERENCEHUB_KEY|STRONG_MODEL|WEAK_MODEL|UPSTREAM_BASE_URL|UPSTREAM_AUTH_ENV)=' "$env_file"; then
  echo "legacy secret or provider-routing overrides are not supported in Phase 2" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

required_values=(
  EXAMPLE_ROOT TERMINAL_BENCH_RUN_ID TERMINAL_BENCH_RUN_ROOT TERMINAL_BENCH_ADMISSION_ROOT
  TERMINAL_BENCH_BOOTSTRAP_ROOT
  HARBOR_BIN EVAL_PYTHON TBENCH_DATASET_PATH SWITCHYARD_BUNDLE RELAY_WHEEL
  RELAY_ARCHITECTURE PLUGIN_CONFIG_TEMPLATE TERMINAL_BENCH_SMOKE_EVIDENCE
  TERMINAL_BENCH_OFFLINE_EVIDENCE PHOENIX_BASE_URL PHOENIX_PROJECT EVAL_COHORT
  TBENCH_SAMPLE_COUNT TBENCH_CANARY_TASK TBENCH_CONCURRENCY
  TBENCH_SETUP_CONCURRENCY TBENCH_SETUP_BATCH_SIZE TBENCH_SETUP_MAX_INFRA_ATTEMPTS
  TBENCH_PARALLEL_MAX_MEMORY_GB TBENCH_DOCKER_MEMORY_RESERVE_GB
  TBENCH_MINIMUM_FREE_GB SWITCHYARD_PROVIDER_AUTHORIZATION
)
for name in "${required_values[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "required Phase 2 variable is unset: $name" >&2
    exit 2
  fi
done
for name in EXAMPLE_ROOT TERMINAL_BENCH_RUN_ROOT TERMINAL_BENCH_ADMISSION_ROOT \
  TERMINAL_BENCH_BOOTSTRAP_ROOT HARBOR_BIN EVAL_PYTHON \
  TBENCH_DATASET_PATH SWITCHYARD_BUNDLE RELAY_WHEEL PLUGIN_CONFIG_TEMPLATE \
  TERMINAL_BENCH_SMOKE_EVIDENCE TERMINAL_BENCH_OFFLINE_EVIDENCE; do
  if [[ "${!name}" != /* ]]; then
    echo "Phase 2 path must be absolute: $name" >&2
    exit 2
  fi
done
for name in TBENCH_SAMPLE_COUNT TBENCH_CONCURRENCY TBENCH_SETUP_CONCURRENCY \
  TBENCH_SETUP_BATCH_SIZE TBENCH_SETUP_MAX_INFRA_ATTEMPTS TBENCH_PARALLEL_MAX_MEMORY_GB \
  TBENCH_DOCKER_MEMORY_RESERVE_GB TBENCH_MINIMUM_FREE_GB; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Phase 2 capacity value must be a positive integer: $name" >&2
    exit 2
  fi
done
case "$RELAY_ARCHITECTURE" in
  x86_64|aarch64) ;;
  *) echo "RELAY_ARCHITECTURE must be x86_64 or aarch64" >&2; exit 2 ;;
esac
for path in "$EXAMPLE_ROOT" "$TBENCH_DATASET_PATH" "$SWITCHYARD_BUNDLE"; do
  [[ -d "$path" ]] || { echo "required Phase 2 directory is missing" >&2; exit 2; }
done
for path in "$HARBOR_BIN" "$EVAL_PYTHON" "$RELAY_WHEEL" "$PLUGIN_CONFIG_TEMPLATE"; do
  [[ -f "$path" ]] || { echo "required Phase 2 file is missing" >&2; exit 2; }
done

"$EVAL_PYTHON" - "$PLUGIN_CONFIG_TEMPLATE" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
plugins = config.get("plugins", {}).get("dynamic", [])
if len(plugins) != 1:
    raise SystemExit("plugin config must contain one dynamic plugin")
targets = plugins[0].get("config", {}).get("targets", {})
if set(targets) != {"strong", "weak", "judge"}:
    raise SystemExit("plugin config must contain strong, weak, and judge targets")
for target in targets.values():
    if target.get("header_env") != {"authorization": "SWITCHYARD_PROVIDER_AUTHORIZATION"}:
        raise SystemExit("plugin config must reference SWITCHYARD_PROVIDER_AUTHORIZATION")
PY

echo "Phase 2 environment validation passed (secret values withheld)"
