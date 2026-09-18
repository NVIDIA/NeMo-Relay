#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 <raw-semver-tag>" >&2
    exit 1
fi
tag="$1"
if [[ ! "$tag" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-((alpha|beta|rc)\.[0-9]+))?$ ]]; then
    echo "Error: tag must use raw SemVer, such as 0.1.0 or 0.1.0-rc.1" >&2
    exit 1
fi
repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
github_repository='NVIDIA/NeMo-Relay'

deployment_status() { case "$1" in 200) printf '☑' ;; 404) printf '○' ;; *) printf '%s' "$1" ;; esac; }
request_status() {
    local status=''
    sleep 0.1
    status="$(curl --location --silent --show-error --output /dev/null --write-out '%{http_code}' --user-agent 'Mozilla/5.0' "$1" 2>/dev/null)" || status="${status:-000}"
    printf '%s\n' "$status"
}
pypi_status() {
    local package="$1" version="$2" response='' body='' status=''
    sleep 0.1
    response="$(curl --location --silent --show-error --header 'Accept: application/vnd.pypi.simple.v1+json' --write-out $'\n%{http_code}' "https://pypi.org/simple/${package}/" 2>/dev/null)" || { printf '000\n'; return; }
    status="${response##*$'\n'}"; body="${response%$'\n'*}"
    if [[ "$status" != 200 ]]; then printf '%s\n' "$status"; return; fi
    jq -e --arg version "$version" '.versions | index($version) != null' >/dev/null <<<"$body" && printf '200\n' || printf '404\n'
}
check_cargo=true; check_python=true; check_node=true
print_pipeline() {
    local runs run name conclusion url
    printf '## GitHub Release Pipeline\n\n| Workflow | Tag | Run | Status |\n| --- | --- | --- | --- |\n'
    if ! gh api "repos/${github_repository}/git/ref/tags/${tag}" >/dev/null 2>&1; then printf '| all workflows | `%s` | — | ○ tag not found |\n' "$tag"; return; fi
    runs="$(gh run list --repo "$github_repository" --branch "$tag" --event push --limit 100 --json databaseId,workflowName,status,conclusion,url 2>/dev/null)" || { printf '| all workflows | `%s` | — | 000 |\n' "$tag"; return; }
    if ! jq -e 'length > 0' >/dev/null <<<"$runs"; then printf '| all workflows | `%s` | — | ○ run not found |\n' "$tag"; return; fi
    while IFS= read -r run; do
        name="$(jq -r '.workflowName' <<<"$run")"; conclusion="$(jq -r '.conclusion // .status' <<<"$run")"; url="$(jq -r '.url' <<<"$run")"
        [[ "$name" == 'Request NVSkills CI' ]] && continue
        [[ "$conclusion" == success ]] && conclusion='☑ success'
        printf '| `%s` | `%s` | [%s](%s) | %s |\n' "$name" "$tag" "$(jq -r '.databaseId' <<<"$run")" "$url" "$conclusion"
        [[ "$conclusion" != failure ]] && continue
        case "$name" in 'Publish (crates.io)') check_cargo=false ;; 'Publish (PyPI)') check_python=false ;; 'Publish (npm)') check_node=false ;; esac
    done < <(jq -c '.[]' <<<"$runs")
}
print_result() { local status; status="$(request_status "$4")"; printf '| %s | `%s` | `%s` | %s |\n' "$1" "$2" "$3" "$(deployment_status "$status")"; }

python_version="$(just --quiet semver-to-pep440 "$tag")"
cargo_packages="$(just --quiet published-cargo-packages)"
python_packages=(nemo-relay nemo-relay-plugin nemo-relay-cli-bin)
node_packages=(nemo-relay-node nemo-relay-node-linux-x64-gnu nemo-relay-node-linux-arm64-gnu nemo-relay-node-linux-x64-musl nemo-relay-node-linux-arm64-musl nemo-relay-node-darwin-arm64 nemo-relay-node-win32-x64-msvc nemo-relay-node-win32-arm64-msvc nemo-relay-openclaw)
print_pipeline
printf '\n## Package Deployments\n\n| Ecosystem | Package | Version | Status |\n| --- | --- | --- | --- |\n'
if "$check_cargo"; then while IFS= read -r package; do print_result Cargo "$package" "$tag" "https://crates.io/api/v1/crates/${package}/${tag}"; done <<<"$cargo_packages"; else printf '| Cargo | — | — | skipped: publication workflow failed |\n'; fi
if "$check_python"; then for package in "${python_packages[@]}"; do printf '| Python | `%s` | `%s` | %s |\n' "$package" "$python_version" "$(deployment_status "$(pypi_status "$package" "$python_version")")"; done; else printf '| Python | — | — | skipped: publication workflow failed |\n'; fi
if "$check_node"; then for package in "${node_packages[@]}"; do print_result 'Node.js' "$package" "$tag" "https://registry.npmjs.org/${package}/${tag}"; done; else printf '| Node.js | — | — | skipped: publication workflow failed |\n'; fi
