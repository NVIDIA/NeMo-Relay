#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Apply Nexus integration patches to local third-party checkouts.
#
# Usage:
#   ./scripts/apply-patches.sh          # apply all patches
#   ./scripts/apply-patches.sh --check  # dry-run (verify patches apply)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST_FILE="$REPO_ROOT/third_party/sources.lock"
CHECK_FLAG=""

if [[ "${1:-}" == "--check" ]]; then
    CHECK_FLAG="--check"
    echo "Dry-run mode: verifying patches apply cleanly..."
fi

apply_patches() {
    local path="$1"
    local name="$2"
    local patch_dir="$REPO_ROOT/patches/$name"
    local target_dir="$REPO_ROOT/$path"

    if [[ ! -d "$target_dir" ]]; then
        echo "SKIP: $target_dir does not exist (run './scripts/bootstrap-third-party.sh')"
        return
    fi

    if [[ ! -d "$patch_dir" ]]; then
        echo "SKIP: no patches for $name"
        return
    fi

    local count=0
    for patch in "$patch_dir"/*.patch; do
        [[ -f "$patch" ]] || continue
        echo "  Applying $(basename "$patch") to $name..."
        git -C "$target_dir" apply $CHECK_FLAG "$patch"
        count=$((count + 1))
    done

    if [[ $count -eq 0 ]]; then
        echo "  No .patch files found in $patch_dir"
    else
        echo "  $count patch(es) applied to $name"
    fi
}

echo "Applying patches..."
while read -r section_key path; do
    manifest_name="${section_key#submodule.}"
    manifest_name="${manifest_name%.path}"
    name="$(basename "$manifest_name")"
    apply_patches "$path" "$name"
done < <(git config -f "$MANIFEST_FILE" --get-regexp '^submodule\..*\.path$')
echo "Done."
