#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Regenerate patches from the current working tree of local third-party checkouts.
#
# Usage:
#   ./scripts/third-party/generate-patches.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MANIFEST_FILE="$REPO_ROOT/third_party/sources.lock"

verify_patch_applies_to_clean_head() {
    local repo_dir="$1"
    local patch_file="$2"

    (
        local temp_root temp_worktree
        temp_root="$(mktemp -d)"
        temp_worktree="$temp_root/worktree"
        trap 'git -C "$repo_dir" worktree remove --force "$temp_worktree" >/dev/null 2>&1 || true; rm -rf "$temp_root"' EXIT

        git -C "$repo_dir" worktree add --detach "$temp_worktree" >/dev/null
        git -C "$temp_worktree" apply --check "$patch_file"
    )
    return $?
}

generate_patches() {
    local path="$1"
    local name="$2"
    local target_dir="$REPO_ROOT/$path"
    local patch_dir="$REPO_ROOT/patches/$name"

    if [[ ! -d "$target_dir/.git" ]] && [[ ! -f "$target_dir/.git" ]]; then
        echo "SKIP: $target_dir is not a git repo"
        return
    fi

    # Check for any changes (tracked modifications + untracked files)
    local has_tracked has_untracked
    has_tracked="$(git -C "$target_dir" diff HEAD --name-only 2>/dev/null)"
    has_untracked="$(git -C "$target_dir" ls-files --others --exclude-standard 2>/dev/null)"

    if [[ -z "$has_tracked" ]] && [[ -z "$has_untracked" ]]; then
        echo "SKIP: $name has no changes"
        return
    fi

    mkdir -p "$patch_dir"
    local patch_file="$patch_dir/0001-add-nemo-relay-integration.patch"

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

    if [[ -s "$patch_file" ]]; then
        verify_patch_applies_to_clean_head "$target_dir" "$patch_file"
    fi

    echo "Generated $patch_file ($(wc -l < "$patch_file") lines)"
}

echo "Generating patches..."
while read -r section_key path; do
    manifest_name="${section_key#submodule.}"
    manifest_name="${manifest_name%.path}"
    name="$(basename "$manifest_name")"
    generate_patches "$path" "$name"
done < <(git config -f "$MANIFEST_FILE" --get-regexp '^submodule\..*\.path$')
echo "Done."
