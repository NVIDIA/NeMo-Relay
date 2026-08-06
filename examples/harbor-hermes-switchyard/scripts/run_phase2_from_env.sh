#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
set +x

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${PHASE2_ENV_FILE:-${1:-}}"
if [[ -z "$env_file" || "$env_file" != /* ]]; then
  echo "PHASE2_ENV_FILE must be an absolute path" >&2
  exit 2
fi

"$example_root/scripts/validate_phase2_environment.sh" "$env_file"
set -a
# shellcheck disable=SC1090
source "$env_file"
set +a
set +x

mkdir -p "$PHASE2_RUN_ROOT"
chmod 0700 "$PHASE2_RUN_ROOT"
exec "$example_root/supervise_phase2_cohort.sh" "$PHASE2_RUN_ROOT" \
  >>"$PHASE2_RUN_ROOT/supervisor.log" 2>&1
