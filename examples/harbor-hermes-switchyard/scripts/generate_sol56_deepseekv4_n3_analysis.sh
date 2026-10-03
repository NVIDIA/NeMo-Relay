#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

example_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${1:-$example_root/config/sol56_deepseekv4_analysis_runs.json}"
target="${2:-$example_root/reports/tb21-sol56-deepseekv4-cf03-ef03-n3}"

exec "$example_root/scripts/generate_final_n3_analysis.sh" "$manifest" "$target"
