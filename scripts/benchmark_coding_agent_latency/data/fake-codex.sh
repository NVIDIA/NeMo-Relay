#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if [ "${1:-}" = "--version" ]; then
    printf 'codex-cli 0.143.0\n'
    exit 0
fi
printf '%s' "$NEMO_RELAY_GATEWAY_URL" > "$BENCHMARK_GATEWAY_FILE"
while [ ! -f "$BENCHMARK_STOP_FILE" ]; do
    sleep 0.1
done
