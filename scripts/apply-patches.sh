#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Apply NeMo Flow integration patches to local third-party checkouts.
#
# Usage:
#   ./scripts/apply-patches.sh          # apply all patches
#   ./scripts/apply-patches.sh --check  # dry-run against a clean detached HEAD checkout

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST_FILE="$REPO_ROOT/third_party/sources.lock"
CHECK_FLAG=""

if [[ "${1:-}" == "--check" ]]; then
    CHECK_FLAG="--check"
    echo "Dry-run mode: verifying patches apply cleanly to a detached checkout of each submodule HEAD..."
fi

with_detached_head_worktree() {
    local repo_dir="$1"
    local callback="$2"
    local callback_arg="$3"

    (
        local temp_root temp_worktree
        temp_root="$(mktemp -d)"
        temp_worktree="$temp_root/worktree"
        trap 'git -C "$repo_dir" worktree remove --force "$temp_worktree" >/dev/null 2>&1 || true; rm -rf "$temp_root"' EXIT

        git -C "$repo_dir" worktree add --detach "$temp_worktree" >/dev/null
        "$callback" "$temp_worktree" "$callback_arg"
    )
}

apply_patch_dir() {
    local worktree_dir="$1"
    local patch_dir="$2"
    local patch
    local count=0

    for patch in "$patch_dir"/*.patch; do
        [[ -f "$patch" ]] || continue
        echo "  Applying $(basename "$patch")..."
        git -C "$worktree_dir" apply $CHECK_FLAG "$patch"
        count=$((count + 1))
    done

    if [[ $count -eq 0 ]]; then
        echo "  No .patch files found in $patch_dir"
    else
        echo "  $count patch(es) processed"
    fi
}

apply_patches() {
    local path="$1"
    local name="$2"
    local patch_dir="$REPO_ROOT/patches/$name"
    local target_dir="$REPO_ROOT/$path"

    if [[ ! -d "$patch_dir" ]]; then
        echo "SKIP: no patches for $name"
        return
    fi

    if [[ ! -d "$target_dir" ]]; then
        echo "SKIP: $target_dir does not exist (run './scripts/bootstrap-third-party.sh')"
        return
    fi

    if [[ ! -d "$target_dir/.git" ]] && [[ ! -f "$target_dir/.git" ]]; then
        echo "SKIP: $target_dir is not a git checkout"
        return
    fi

    echo "$name:"
    if [[ -n "$CHECK_FLAG" ]]; then
        if [[ -n "$(git -C "$target_dir" status --porcelain)" ]]; then
            echo "  NOTE: local changes detected; checking patch applicability against a clean detached HEAD checkout."
        fi
        with_detached_head_worktree "$target_dir" apply_patch_dir "$patch_dir"
    else
        if [[ -n "$(git -C "$target_dir" status --porcelain)" ]]; then
            echo "ERROR: $target_dir has local changes; refusing to apply patches on top of a dirty checkout." >&2
            echo "  Commit/stash/reset the local changes first, or run './scripts/apply-patches.sh --check' to validate patches against clean HEAD." >&2
            return 1
        fi
        apply_patch_dir "$target_dir" "$patch_dir"
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
