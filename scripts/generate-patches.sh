#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Regenerate patches from the current working tree of third-party submodules.
#
# Usage:
#   ./scripts/generate-patches.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

generate_patches() {
    local submodule="$1"
    local target_dir="$REPO_ROOT/third_party/$submodule"
    local patch_dir="$REPO_ROOT/patches/$submodule"

    if [[ ! -d "$target_dir/.git" ]] && [[ ! -f "$target_dir/.git" ]]; then
        echo "SKIP: $target_dir is not a git repo"
        return
    fi

    # Check for any changes (tracked modifications + untracked files)
    local has_tracked has_untracked
    has_tracked="$(git -C "$target_dir" diff HEAD --name-only 2>/dev/null)"
    has_untracked="$(git -C "$target_dir" ls-files --others --exclude-standard 2>/dev/null)"

    if [[ -z "$has_tracked" ]] && [[ -z "$has_untracked" ]]; then
        echo "SKIP: $submodule has no changes"
        return
    fi

    mkdir -p "$patch_dir"
    local patch_file="$patch_dir/0001-add-nvmagic-integration.patch"

    # Combine tracked diffs and new file diffs
    {
        # Modified/deleted tracked files
        if [[ -n "$has_tracked" ]]; then
            git -C "$target_dir" diff HEAD
        fi
        # New untracked files
        if [[ -n "$has_untracked" ]]; then
            while IFS= read -r f; do
                git -C "$target_dir" diff --no-index /dev/null "$f" 2>/dev/null || true
            done <<< "$has_untracked"
        fi
    } > "$patch_file"

    echo "Generated $patch_file ($(wc -l < "$patch_file") lines)"
}

echo "Generating patches..."
for dir in "$REPO_ROOT"/third_party/*/; do
    submodule="$(basename "$dir")"
    generate_patches "$submodule"
done
echo "Done."
