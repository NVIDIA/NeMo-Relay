#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Bootstrap local third-party upstream checkouts from the tracked manifest.
#
# Usage:
#   ./scripts/bootstrap-third-party.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST_FILE="$REPO_ROOT/third_party/sources.lock"

if [[ ! -f "$MANIFEST_FILE" ]]; then
    echo "ERROR: manifest not found: $MANIFEST_FILE" >&2
    exit 1
fi

bootstrap_checkout() {
    local path="$1"
    local url="$2"
    local commit="$3"
    local target_dir="$REPO_ROOT/$path"

    if [[ ! -e "$target_dir" ]]; then
        echo "Cloning $path from $url..."
        git clone "$url" "$target_dir"
    elif [[ ! -d "$target_dir/.git" ]] && [[ ! -f "$target_dir/.git" ]]; then
        echo "SKIP: $path exists but is not a git checkout" >&2
        return
    fi

    if [[ "$(git -C "$target_dir" remote get-url origin 2>/dev/null || true)" != "$url" ]]; then
        echo "WARN: $path origin does not match manifest URL" >&2
        echo "  manifest: $url" >&2
        echo "  origin:   $(git -C "$target_dir" remote get-url origin 2>/dev/null || echo "<none>")" >&2
    fi

    echo "Fetching $path..."
    git -C "$target_dir" fetch --tags origin

    local current_commit
    current_commit="$(git -C "$target_dir" rev-parse HEAD)"
    if [[ "$current_commit" == "$commit" ]]; then
        echo "OK: $path already at $commit"
        return
    fi

    if [[ -n "$(git -C "$target_dir" status --porcelain)" ]]; then
        echo "SKIP: $path has local changes; leaving checkout at $current_commit" >&2
        return
    fi

    echo "Checking out $path at $commit..."
    git -C "$target_dir" checkout --detach "$commit"
}

echo "Bootstrapping third-party checkouts from $MANIFEST_FILE..."
while read -r section_key path; do
    url="$(git config -f "$MANIFEST_FILE" --get "${section_key%.path}.url")"
    commit="$(git config -f "$MANIFEST_FILE" --get "${section_key%.path}.commit")"
    bootstrap_checkout "$path" "$url" "$commit"
done < <(git config -f "$MANIFEST_FILE" --get-regexp '^submodule\..*\.path$')
echo "Done."
