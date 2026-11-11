#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Commit hooks inherit repository-local Git environment variables. cargo-deny's
# advisory refresh invokes Git in a separate checkout; inheriting these values
# can reset the caller's branch/index instead of that advisory checkout.
while IFS= read -r git_local_variable; do
    unset "$git_local_variable"
done < <(git rev-parse --local-env-vars)

exec cargo deny --locked --all-features check
