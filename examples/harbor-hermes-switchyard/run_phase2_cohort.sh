#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_root="${1:-${TERMINAL_BENCH_RUN_ROOT:-}}"
if [[ -z "$run_root" || "$run_root" != /* ]]; then
  echo "usage: $0 /absolute/phase2-run-root [coordinator options]" >&2
  exit 2
fi
shift

dataset="${TBENCH_DATASET:-terminal-bench@2.0}"
dataset_name="${dataset%@*}"
dataset_export_root="${TBENCH_DATASET_EXPORT_ROOT:-$(dirname "$run_root")/harbor-datasets}"
dataset_root="${TBENCH_DATASET_PATH:-$dataset_export_root/$dataset_name}"
harbor_bin="${HARBOR_BIN:-$example_root/.venv/bin/harbor}"
python_bin="${EVAL_PYTHON:-$example_root/.venv/bin/python}"
smoke_evidence="${TERMINAL_BENCH_SMOKE_EVIDENCE:-}"
offline_evidence="${TERMINAL_BENCH_OFFLINE_EVIDENCE:-}"
phoenix_url="${PHOENIX_BASE_URL:-}"
phoenix_project="${PHOENIX_PROJECT:-}"
eval_cohort="${EVAL_COHORT:-}"
switchyard_bundle="${SWITCHYARD_BUNDLE:-}"
relay_wheel="${RELAY_WHEEL:-}"
relay_architecture="${RELAY_ARCHITECTURE:-x86_64}"
plugin_config_template="${PLUGIN_CONFIG_TEMPLATE:-$example_root/config/plugins.toml.in}"
sample_count="${TBENCH_SAMPLE_COUNT:-89}"
# An explicitly blank value disables canary-first scheduling. An unset value
# keeps the conservative default.
canary_task="${TBENCH_CANARY_TASK-adaptive-rejection-sampler}"
concurrency="${TBENCH_CONCURRENCY:-4}"
setup_concurrency="${TBENCH_SETUP_CONCURRENCY:-2}"
setup_batch_size="${TBENCH_SETUP_BATCH_SIZE:-89}"
setup_max_infra_attempts="${TBENCH_SETUP_MAX_INFRA_ATTEMPTS:-4}"
parallel_memory_gb="${TBENCH_PARALLEL_MAX_MEMORY_GB:-2}"
docker_memory_reserve_gb="${TBENCH_DOCKER_MEMORY_RESERVE_GB:-4}"
minimum_free_gb="${TBENCH_MINIMUM_FREE_GB:-100}"
bootstrap_root="${TERMINAL_BENCH_BOOTSTRAP_ROOT:-${TERMINAL_BENCH_ADMISSION_ROOT:-$(dirname "$run_root")}/bootstrap}"

for required in \
  "$harbor_bin" \
  "$python_bin" \
  "$smoke_evidence" \
  "$offline_evidence" \
  "$dataset_root" \
  "$plugin_config_template" \
  "$switchyard_bundle/relay-plugin.toml" \
  "$relay_wheel"; do
  [[ -e "$required" ]] || {
    echo "required Phase 2 input is missing: $required" >&2
    exit 2
  }
done
if [[ -z "${SWITCHYARD_PROVIDER_AUTHORIZATION:-}" ]]; then
  echo "SWITCHYARD_PROVIDER_AUTHORIZATION must be set by the protected Phase 2 environment file" >&2
  exit 2
fi
for label in phoenix_url phoenix_project eval_cohort; do
  [[ -n "${!label}" ]] || {
    echo "${label^^} must be set" >&2
    exit 2
  }
done

if [[ "$dataset_root" != /* || ! -d "$dataset_root" ]]; then
  echo "TBENCH_DATASET_PATH must select an existing absolute local dataset directory" >&2
  echo "Phase 2 never downloads or resolves a dataset through the Harbor registry" >&2
  exit 2
fi

exec "$python_bin" "$example_root/scripts/run_phase2_cohort.py" \
  --run-root "$run_root" \
  --dataset "$dataset" \
  --dataset-root "$dataset_root" \
  --sample-count "$sample_count" \
  --canary-task "$canary_task" \
  --concurrency "$concurrency" \
  --setup-concurrency "$setup_concurrency" \
  --setup-batch-size "$setup_batch_size" \
  --setup-max-infra-attempts "$setup_max_infra_attempts" \
  --parallel-max-memory-gb "$parallel_memory_gb" \
  --docker-memory-reserve-gb "$docker_memory_reserve_gb" \
  --minimum-free-gb "$minimum_free_gb" \
  --smoke-evidence "$smoke_evidence" \
  --offline-evidence "$offline_evidence" \
  --plugin-config-template "$plugin_config_template" \
  --task-runner "$example_root/run_terminal_bench.sh" \
  --harbor-bin "$harbor_bin" \
  --python-bin "$python_bin" \
  --phoenix-url "$phoenix_url" \
  --phoenix-project "$phoenix_project" \
  --eval-cohort "$eval_cohort" \
  --switchyard-bundle "$switchyard_bundle" \
  --relay-wheel "$relay_wheel" \
  --relay-architecture "$relay_architecture" \
  --bootstrap-root "$bootstrap_root" \
  "$@"
