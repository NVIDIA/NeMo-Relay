#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
installer="${repo_root}/install.sh"
test_root=$(mktemp -d)
live_install_dir="${test_root}/live-bin"
tests_run=0

cleanup() {
    rm -rf "$test_root"
}
trap cleanup EXIT HUP INT TERM

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

run_command() {
    if run_output=$("$@" 2>&1); then
        run_status=0
    else
        run_status=$?
    fi
}

assert_success() {
    [ "$run_status" -eq 0 ] || fail "expected success, got ${run_status}: ${run_output}"
}

assert_failure() {
    [ "$run_status" -ne 0 ] || fail "expected failure: ${run_output}"
}

assert_contains() {
    printf '%s\n' "$1" | grep -F "$2" >/dev/null || fail "expected '$2' in: $1"
}

assert_no_temporary_files() {
    set -- "$1"/.nemo-relay.*
    [ ! -e "$1" ] || fail "temporary installer file was not cleaned up: $1"
}

test_interface_validation() {
    tests_run=$((tests_run + 1))

    run_command sh "$installer" --help
    assert_success
    assert_contains "$run_output" "Usage:"

    run_command sh "$installer" --unknown
    assert_failure
    assert_contains "$run_output" "unknown option"

    run_command sh "$installer" 0.5.0
    assert_failure
    assert_contains "$run_output" "unexpected argument"

    run_command env NEMO_RELAY_VERSION=not-a-version sh "$installer"
    assert_failure
    assert_contains "$run_output" "unsupported version"

    run_command env -u HOME NEMO_RELAY_VERSION=0.3.0 sh "$installer"
    assert_failure
    assert_contains "$run_output" "install directory must not be empty"
}

test_live_latest_and_pinned_replacement() {
    tests_run=$((tests_run + 1))

    run_command env -u NEMO_RELAY_VERSION sh "$installer" --install-dir "$live_install_dir"
    assert_success
    [ -x "${live_install_dir}/nemo-relay" ] || fail "latest install did not create an executable"
    latest_version=$("${live_install_dir}/nemo-relay" --version)
    assert_contains "$latest_version" "nemo-relay "
    assert_no_temporary_files "$live_install_dir"

    run_command env NEMO_RELAY_VERSION=0.3.0 sh "$installer" --install-dir "$live_install_dir"
    assert_success
    pinned_version=$("${live_install_dir}/nemo-relay" --version)
    assert_contains "$pinned_version" "nemo-relay 0.3.0"
    assert_no_temporary_files "$live_install_dir"
}

test_live_asset_404_preserves_existing_binary() {
    tests_run=$((tests_run + 1))

    run_command env NEMO_RELAY_VERSION=999.999.999 sh "$installer" --install-dir "$live_install_dir"
    assert_failure
    assert_contains "$run_output" "could not download https://github.com/NVIDIA/NeMo-Relay/releases/download/999.999.999/"
    preserved_version=$("${live_install_dir}/nemo-relay" --version)
    assert_contains "$preserved_version" "nemo-relay 0.3.0"
    assert_no_temporary_files "$live_install_dir"
}

test_live_network_failure() {
    tests_run=$((tests_run + 1))
    network_failure_dir="${test_root}/network-failure-bin"

    run_command env -u NEMO_RELAY_VERSION \
        HTTPS_PROXY=http://127.0.0.1:1 \
        https_proxy=http://127.0.0.1:1 \
        ALL_PROXY= \
        all_proxy= \
        NO_PROXY= \
        no_proxy= \
        sh "$installer" --install-dir "$network_failure_dir"
    assert_failure
    assert_contains "$run_output" "could not resolve the latest stable release"
    [ ! -e "${network_failure_dir}/nemo-relay" ] || fail "binary installed after network failure"
}

test_interface_validation
test_live_latest_and_pinned_replacement
test_live_asset_404_preserves_existing_binary
test_live_network_failure

printf 'PASS: %s live and non-mocked installer groups\n' "$tests_run"
