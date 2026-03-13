#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Apply NVMagic integration patches to third-party submodules.
#
# Usage:
#   ./scripts/apply-patches.sh          # apply all patches
#   ./scripts/apply-patches.sh --check  # dry-run (verify patches apply)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECK_FLAG=""

if [[ "${1:-}" == "--check" ]]; then
    CHECK_FLAG="--check"
    echo "Dry-run mode: verifying patches apply cleanly..."
fi

apply_patches() {
    local submodule="$1"
    local patch_dir="$REPO_ROOT/patches/$submodule"
    local target_dir="$REPO_ROOT/third_party/$submodule"

    if [[ ! -d "$target_dir" ]]; then
        echo "SKIP: $target_dir does not exist (run 'git submodule update --init')"
        return
    fi

    if [[ ! -d "$patch_dir" ]]; then
        echo "SKIP: no patches for $submodule"
        return
    fi

    local count=0
    for patch in "$patch_dir"/*.patch; do
        [[ -f "$patch" ]] || continue
        echo "  Applying $(basename "$patch") to $submodule..."
        git -C "$target_dir" apply $CHECK_FLAG "$patch"
        count=$((count + 1))
    done

    if [[ $count -eq 0 ]]; then
        echo "  No .patch files found in $patch_dir"
    else
        echo "  $count patch(es) applied to $submodule"
    fi
}

echo "Applying patches..."
for patch_dir in "$REPO_ROOT"/patches/*/; do
    submodule="$(basename "$patch_dir")"
    apply_patches "$submodule"
done
echo "Done."
