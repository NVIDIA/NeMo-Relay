#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Public bundle acquisition for the Ubuntu x86_64 Brev example. Other platforms
# can use Relay's GitHub CLI installer to select the matching published asset.
set -euo pipefail
[[ $# == 1 ]] || { echo 'Usage: bash install-plugin.sh NEW_DIRECTORY' >&2; exit 1; }
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
    echo 'This helper requires Linux x86_64 with glibc; use the Relay plugin installer on other platforms.' >&2
    exit 1
}
getconf GNU_LIBC_VERSION >/dev/null || { echo 'A glibc host is required.' >&2; exit 1; }
for tool in curl sha256sum tar; do
    command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 1; }
done
[[ ! -e $1 ]] || { echo 'Bundle directory exists: choose a new directory.' >&2; exit 1; }
mkdir -p -m 0700 -- "$1"
switchyard_bundle_dir=$(cd -- "$1" && pwd)
asset=switchyard-plugin-0.3.2-linux-x86_64.tar.gz
release=https://github.com/NVIDIA/NeMo-Relay-Plugins/releases/download/switchyard-plugin-0.3.2
curl -fsSL --connect-timeout 10 --max-time 300 "$release/$asset" -o "$switchyard_bundle_dir/$asset"
curl -fsSL --connect-timeout 10 --max-time 300 "$release/$asset.sha256" -o "$switchyard_bundle_dir/$asset.sha256"
(cd -- "$switchyard_bundle_dir" && sha256sum --check --status "$asset.sha256") || {
    echo 'Bundle checksum failed; do not load this artifact.' >&2; exit 1;
}
tar -xzf "$switchyard_bundle_dir/$asset" -C "$switchyard_bundle_dir"
manifest="$switchyard_bundle_dir/switchyard-plugin/relay-plugin.toml"
[[ -f $manifest ]] || { echo 'Expected manifest missing; inspect the downloaded bundle.' >&2; exit 1; }
printf '%s\n' "$manifest"
