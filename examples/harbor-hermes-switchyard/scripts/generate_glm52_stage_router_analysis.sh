#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${1:-$example_root/config/glm52_stage_router_analysis_runs.json}"
target="${2:-$example_root/reports/tb21-glm52-stage-ef05-signal-vs-classifier}"

exec "$example_root/scripts/generate_final_n3_analysis.sh" "$manifest" "$target"
