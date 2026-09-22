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
pipeline_status() {
    case "$1" in
        success) printf '☑ success' ;;
        skipped) printf '○ skipped' ;;
        '') printf '000' ;;
        *) printf '%s' "$1" ;;
    esac
}
print_job_row() {
    local workflow="$1" job="$2" name result url database_id
    name="$(jq -r '.name' <<<"$job")"
    result="$(jq -r '.conclusion // .status // ""' <<<"$job")"
    url="$(jq -r '.url' <<<"$job")"
    database_id="$(jq -r '.databaseId' <<<"$job")"
    printf '| `%s` | `%s` | [%s](%s) | %s |\n' "$workflow" "$name" "$database_id" "$url" "$(pipeline_status "$result")"
    [[ "$result" == success ]] && return
    case "$name" in
        'Publish (crates.io)') check_cargo=false ;;
        'Publish (PyPI)') check_python=false ;;
        'Publish (npm)') check_node=false ;;
    esac
}
print_release_jobs() {
    local runs run workflow database_id job job_name go_jobs build_run_id='' docs_run_id='' main_jobs='' docs_jobs='' docs_job_name='Release docs version'
    printf '## GitHub Release Jobs\n\n| Workflow | Job | Job ID | Status |\n| --- | --- | --- | --- |\n'
    if ! gh api "repos/${github_repository}/git/ref/tags/${tag}" >/dev/null 2>&1; then printf '| all workflows | `tag %s` | — | ○ tag not found |\n' "$tag"; return; fi
    runs="$(gh run list --repo "$github_repository" --branch "$tag" --event push --limit 100 --json databaseId,workflowName 2>/dev/null)" || { printf '| all workflows | release workflow discovery | — | 000 |\n'; return; }
    if ! jq -e 'length > 0' >/dev/null <<<"$runs"; then printf '| all workflows | release workflow discovery | — | ○ run not found |\n'; return; fi
    for workflow in 'Build pull request' 'Fern Docs'; do
        run="$(jq -c --arg workflow "$workflow" 'first(.[] | select(.workflowName == $workflow)) // empty' <<<"$runs")"
        if [[ -z "$run" ]]; then
            printf '| `%s` | release workflow | — | ○ run not found |\n' "$workflow"
            continue
        fi
        database_id="$(jq -r '.databaseId' <<<"$run")"
        case "$workflow" in
            'Build pull request') build_run_id="$database_id" ;;
            'Fern Docs') docs_run_id="$database_id" ;;
        esac
    done

    if [[ -n "$build_run_id" ]]; then
        main_jobs="$(gh run view "$build_run_id" --repo "$github_repository" --json jobs 2>/dev/null)" || main_jobs=''
        if [[ -z "$main_jobs" ]]; then
            printf '| `Build pull request` | release jobs | — | 000 |\n'
        else
            for job_name in 'CI Pipeline' 'Release Distribution Artifacts' 'Publish (crates.io)' 'Publish (PyPI)' 'Publish (npm)'; do
                job="$(jq -c --arg name "$job_name" 'first(.jobs[] | select(.name == $name)) // empty' <<<"$main_jobs")"
                if [[ -z "$job" ]]; then
                    printf '| `Build pull request` | `%s` | — | ○ job not found |\n' "$job_name"
                else
                    print_job_row 'Build pull request' "$job"
                fi
            done
            go_jobs="$(jq -c '[.jobs[] | select(.name | startswith("Go / Test ("))] | sort_by(.name)' <<<"$main_jobs")"
            if ! jq -e 'length > 0' >/dev/null <<<"$go_jobs"; then
                printf '| `Build pull request` | `Go / Test (*)` | — | ○ job not found |\n'
            else
                while IFS= read -r job; do
                    print_job_row 'Build pull request' "$job"
                done < <(jq -c '.[]' <<<"$go_jobs")
            fi
        fi
    fi

    [[ "$tag" == *-alpha.* ]] && docs_job_name='Skip nightly alpha docs'
    if [[ -n "$docs_run_id" ]]; then
        docs_jobs="$(gh run view "$docs_run_id" --repo "$github_repository" --json jobs 2>/dev/null)" || docs_jobs=''
        if [[ -z "$docs_jobs" ]]; then
            printf '| `Fern Docs` | `%s` | — | 000 |\n' "$docs_job_name"
        else
            run="$(jq -c --arg name "$docs_job_name" 'first(.jobs[] | select(.name == $name)) // empty' <<<"$docs_jobs")"
            if [[ -z "$run" ]]; then
                printf '| `Fern Docs` | `%s` | — | ○ job not found |\n' "$docs_job_name"
            else
                print_job_row 'Fern Docs' "$run"
            fi
        fi
    fi
}
print_result() { local status; status="$(request_status "$4")"; printf '| %s | `%s` | `%s` | %s |\n' "$1" "$2" "$3" "$(deployment_status "$status")"; }

python_version="$(just --quiet semver-to-pep440 "$tag")"
cargo_packages="$(just --quiet published-cargo-packages)"
python_packages=(nemo-relay nemo-relay-plugin nemo-relay-cli-bin)
read -r module_keyword go_module < go/nemo_relay/go.mod
go_module_prefix="github.com/${github_repository}/"
if [[ "$module_keyword" != module || "$go_module" != "${go_module_prefix}"* ]]; then
    echo "Error: unsupported Go module declaration in go/nemo_relay/go.mod" >&2
    exit 1
fi
go_module_path="${go_module#"$go_module_prefix"}"
go_package="${go_module##*/}"
python_executable="$(uv python find)"
node_platforms="$("$python_executable" scripts/package-node-bin.py --print-platforms)"
node_packages=(nemo-relay-node)
while IFS=$'\t' read -r platform package; do
    if [[ -z "$platform" || -z "$package" ]]; then
        echo "Error: invalid platform entry from package-node-bin.py" >&2
        exit 1
    fi
    node_packages+=("$package")
done <<<"$node_platforms"
node_packages+=(nemo-relay-openclaw)
print_release_jobs
printf '\n## Package Deployments\n\n| Ecosystem | Package | Version | Status |\n| --- | --- | --- | --- |\n'
if "$check_cargo"; then while IFS= read -r package; do print_result Cargo "$package" "$tag" "https://crates.io/api/v1/crates/${package}/${tag}"; done <<<"$cargo_packages"; else printf '| Cargo | — | — | skipped: publication job did not succeed |\n'; fi
print_result Go "$go_package" "$tag" "https://raw.githubusercontent.com/${github_repository}/${tag}/${go_module_path}/go.mod"
if "$check_python"; then for package in "${python_packages[@]}"; do printf '| Python | `%s` | `%s` | %s |\n' "$package" "$python_version" "$(deployment_status "$(pypi_status "$package" "$python_version")")"; done; else printf '| Python | — | — | skipped: publication job did not succeed |\n'; fi
if "$check_node"; then for package in "${node_packages[@]}"; do print_result 'Node.js' "$package" "$tag" "https://registry.npmjs.org/${package}/${tag}"; done; else printf '| Node.js | — | — | skipped: publication job did not succeed |\n'; fi
