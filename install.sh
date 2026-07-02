#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

REPOSITORY="NVIDIA/NeMo-Relay"
GITHUB_URL="https://github.com/${REPOSITORY}"
GITHUB_API_URL="https://api.github.com/repos/${REPOSITORY}"

usage() {
    cat <<'EOF'
Install the NeMo Relay CLI from GitHub Releases.

Usage:
  install.sh [--install-dir DIR]
  install.sh --help

Environment:
  NEMO_RELAY_VERSION   Release to install, for example 0.5.0 or v0.5.0.
                       Defaults to the latest stable release.

Options:
  --install-dir DIR    Destination directory (default: $HOME/.local/bin).
  -h, --help           Show this help text.

Examples:
  curl -fsSL https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/main/install.sh | sh
  curl -fsSL https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/main/install.sh | NEMO_RELAY_VERSION=0.5.0 sh
  curl -fsSL https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/main/install.sh | sh -s -- --install-dir "$HOME/bin"
EOF
}

error() {
    printf 'nemo-relay installer: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || error "required command not found: $1"
}

curl_with_timeouts() {
    curl -fsSL --connect-timeout 10 --max-time 300 "$@"
}

version="${NEMO_RELAY_VERSION:-}"
install_dir="${HOME:+${HOME}/.local/bin}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --install-dir)
            [ "$#" -ge 2 ] || error "--install-dir requires a directory"
            install_dir=$2
            shift 2
            ;;
        --install-dir=*)
            install_dir=${1#*=}
            shift
            ;;
        --)
            shift
            ;;
        -*)
            error "unknown option: $1"
            ;;
        *)
            error "unexpected argument: $1"
            ;;
    esac
done

[ -n "$install_dir" ] || error "install directory must not be empty"
require_command curl
require_command uname
require_command mktemp

if [ -z "$version" ]; then
    printf 'Finding the latest stable NeMo Relay release...\n'
    release_json=$(curl_with_timeouts \
        -H 'Accept: application/vnd.github+json' \
        -H 'User-Agent: nemo-relay-install-script' \
        "${GITHUB_API_URL}/releases/latest") || error "could not resolve the latest stable release"
    version=$(printf '%s\n' "$release_json" | sed -n 's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p')
    [ -n "$version" ] || error "latest release response did not contain a tag name"
fi

version=${version#v}
printf '%s\n' "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-(alpha|beta|rc)\.[0-9]+)?$' || \
    error "unsupported version '${version}'; expected 0.5.0 or a prerelease such as 0.5.0-rc.1"

os=$(uname -s)
arch=$(uname -m)

case "${os}:${arch}" in
    Linux:x86_64|Linux:amd64)
        target="x86_64-unknown-linux-musl"
        ;;
    Linux:aarch64|Linux:arm64)
        target="aarch64-unknown-linux-musl"
        ;;
    Darwin:aarch64|Darwin:arm64)
        target="aarch64-apple-darwin"
        ;;
    *)
        error "unsupported platform ${os}/${arch}. Supported platforms: Linux x86_64, Linux ARM64, and macOS ARM64. For other platforms, use 'cargo install nemo-relay-cli'."
        ;;
esac

asset="nemo-relay-cli-${target}-${version}"
asset_url="${GITHUB_URL}/releases/download/${version}/${asset}"
checksum_url="${asset_url}.sha256"

mkdir -p "$install_dir" || error "could not create install directory: ${install_dir}"
[ -d "$install_dir" ] || error "install path is not a directory: ${install_dir}"
[ -w "$install_dir" ] || error "install directory is not writable: ${install_dir}"

download_file=$(mktemp "${install_dir}/.nemo-relay.download.XXXXXX") || \
    error "could not create a temporary file in ${install_dir}"
checksum_file=$(mktemp "${install_dir}/.nemo-relay.checksum.XXXXXX") || {
    rm -f "$download_file"
    error "could not create a temporary file in ${install_dir}"
}

cleanup() {
    rm -f "$download_file" "$checksum_file"
}
trap cleanup EXIT HUP INT TERM

printf 'Downloading NeMo Relay CLI %s for %s...\n' "$version" "$target"
curl_with_timeouts -o "$download_file" "$asset_url" || error "could not download ${asset_url}"
curl_with_timeouts -o "$checksum_file" "$checksum_url" || error "could not download ${checksum_url}"

expected_checksum=$(sed -n '1{s/[[:space:]].*//;p;}' "$checksum_file" | tr 'A-F' 'a-f')
printf '%s\n' "$expected_checksum" | grep -Eq '^[0-9a-f]{64}$' || \
    error "invalid checksum file for ${asset}"

if command -v sha256sum >/dev/null 2>&1; then
    actual_checksum=$(sha256sum "$download_file" | sed -n '1{s/[[:space:]].*//;p;}')
elif command -v shasum >/dev/null 2>&1; then
    actual_checksum=$(shasum -a 256 "$download_file" | sed -n '1{s/[[:space:]].*//;p;}')
elif command -v openssl >/dev/null 2>&1; then
    actual_checksum=$(openssl dgst -sha256 "$download_file" | sed 's/^.*= //')
else
    error "no SHA-256 utility found; install sha256sum, shasum, or openssl"
fi
actual_checksum=$(printf '%s\n' "$actual_checksum" | tr 'A-F' 'a-f')

[ "$actual_checksum" = "$expected_checksum" ] || error "checksum verification failed for ${asset}"

chmod 0755 "$download_file" || error "could not make the downloaded binary executable"
destination="${install_dir}/nemo-relay"
mv -f "$download_file" "$destination" || error "could not install ${destination}"

printf 'Installed NeMo Relay CLI %s to %s\n' "$version" "$destination"
case ":${PATH:-}:" in
    *":${install_dir}:"*)
        ;;
    *)
        printf 'Add %s to PATH to run nemo-relay from your shell.\n' "$install_dir"
        ;;
esac
