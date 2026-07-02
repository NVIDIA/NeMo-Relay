#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
installer="${repo_root}/install.sh"
test_root=$(mktemp -d)
original_path=$PATH
tests_run=0

cleanup() {
    rm -rf "$test_root"
}
trap cleanup EXIT HUP INT TERM

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
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

assert_file_contains() {
    grep -F "$2" "$1" >/dev/null || fail "expected '$2' in $1"
}

assert_no_temporary_files() {
    set -- "$1"/.nemo-relay.*
    [ ! -e "$1" ] || fail "temporary installer file was not cleaned up: $1"
}

make_mock_commands() {
    mock_bin=$1
    mkdir -p "$mock_bin"

    cat >"${mock_bin}/uname" <<'EOF'
#!/bin/sh
case "${1:-}" in
    -s) printf '%s\n' "$MOCK_UNAME_S" ;;
    -m) printf '%s\n' "$MOCK_UNAME_M" ;;
    *) exit 1 ;;
esac
EOF

    cat >"${mock_bin}/curl" <<'EOF'
#!/bin/sh
output=""
url=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o)
            output=$2
            shift 2
            ;;
        -H)
            shift 2
            ;;
        -*)
            shift
            ;;
        *)
            url=$1
            shift
            ;;
    esac
done

printf '%s\n' "$url" >>"$MOCK_CURL_LOG"
case "$url" in
    */releases/latest)
        printf '{"url":"mock","tag_name":"%s","prerelease":false}\n' "$MOCK_LATEST_VERSION"
        ;;
    *.sha256)
        [ "${MOCK_CHECKSUM_DOWNLOAD_FAIL:-0}" != 1 ] || exit 22
        printf '%s  %s\n' "$MOCK_EXPECTED_CHECKSUM" "${url##*/}" >"$output"
        ;;
    *)
        printf '#!/bin/sh\nprintf "mock nemo-relay\\n"\n' >"$output"
        ;;
esac
EOF

    cat >"${mock_bin}/sha256sum" <<'EOF'
#!/bin/sh
printf '%s  %s\n' "$MOCK_ACTUAL_CHECKSUM" "$1"
EOF

    chmod +x "${mock_bin}/uname" "${mock_bin}/curl" "${mock_bin}/sha256sum"
}

new_case() {
    tests_run=$((tests_run + 1))
    case_root="${test_root}/case-${tests_run}"
    home_dir="${case_root}/home"
    mock_bin="${case_root}/bin"
    curl_log="${case_root}/curl.log"
    mkdir -p "$home_dir"
    : >"$curl_log"
    make_mock_commands "$mock_bin"

    MOCK_UNAME_S=Linux
    MOCK_UNAME_M=x86_64
    MOCK_LATEST_VERSION=0.5.0
    MOCK_EXPECTED_CHECKSUM=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    MOCK_ACTUAL_CHECKSUM=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    MOCK_CHECKSUM_DOWNLOAD_FAIL=0
    NEMO_RELAY_VERSION=""
    HOME=$home_dir
    PATH="${mock_bin}:${original_path}"
    MOCK_CURL_LOG=$curl_log
    export MOCK_UNAME_S MOCK_UNAME_M MOCK_LATEST_VERSION
    export MOCK_EXPECTED_CHECKSUM MOCK_ACTUAL_CHECKSUM MOCK_CHECKSUM_DOWNLOAD_FAIL
    export NEMO_RELAY_VERSION HOME PATH MOCK_CURL_LOG
}

run_installer() {
    if run_output=$(sh "$installer" "$@" 2>&1); then
        run_status=0
    else
        run_status=$?
    fi
}

test_latest_linux_x86_64() {
    new_case
    run_installer
    assert_success
    [ -x "${HOME}/.local/bin/nemo-relay" ] || fail "latest install did not create an executable"
    assert_file_contains "$curl_log" "/releases/latest"
    assert_file_contains "$curl_log" "/0.5.0/nemo-relay-cli-x86_64-unknown-linux-musl-0.5.0"
    assert_no_temporary_files "${HOME}/.local/bin"
}

test_positional_version_precedence_and_v_normalization() {
    new_case
    NEMO_RELAY_VERSION=0.4.0
    export NEMO_RELAY_VERSION
    run_installer v0.5.0
    assert_success
    assert_file_contains "$curl_log" "/0.5.0/nemo-relay-cli-x86_64-unknown-linux-musl-0.5.0"
}

test_environment_version_and_linux_arm64() {
    new_case
    NEMO_RELAY_VERSION=v0.5.0
    MOCK_UNAME_M=aarch64
    custom_dir="${case_root}/custom-bin"
    export NEMO_RELAY_VERSION MOCK_UNAME_M
    run_installer --install-dir "$custom_dir"
    assert_success
    [ -x "${custom_dir}/nemo-relay" ] || fail "custom install directory was not used"
    assert_file_contains "$curl_log" "nemo-relay-cli-aarch64-unknown-linux-musl-0.5.0"
}

test_macos_arm64() {
    new_case
    MOCK_UNAME_S=Darwin
    MOCK_UNAME_M=arm64
    export MOCK_UNAME_S MOCK_UNAME_M
    run_installer 0.5.0
    assert_success
    assert_file_contains "$curl_log" "nemo-relay-cli-aarch64-apple-darwin-0.5.0"
}

test_unsupported_platform() {
    new_case
    MOCK_UNAME_S=Darwin
    MOCK_UNAME_M=x86_64
    NEMO_RELAY_VERSION=0.5.0
    export MOCK_UNAME_S MOCK_UNAME_M NEMO_RELAY_VERSION
    run_installer
    assert_failure
    assert_contains "$run_output" "unsupported platform Darwin/x86_64"
    [ ! -s "$curl_log" ] || fail "unsupported platform attempted a download"
}

test_checksum_mismatch_preserves_existing_binary() {
    new_case
    install_dir="${HOME}/.local/bin"
    mkdir -p "$install_dir"
    printf 'existing binary\n' >"${install_dir}/nemo-relay"
    MOCK_ACTUAL_CHECKSUM=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    export MOCK_ACTUAL_CHECKSUM
    run_installer 0.5.0
    assert_failure
    assert_contains "$run_output" "checksum verification failed"
    assert_file_contains "${install_dir}/nemo-relay" "existing binary"
    assert_no_temporary_files "$install_dir"
}

test_missing_checksum_fails_closed() {
    new_case
    MOCK_CHECKSUM_DOWNLOAD_FAIL=1
    export MOCK_CHECKSUM_DOWNLOAD_FAIL
    run_installer 0.5.0
    assert_failure
    assert_contains "$run_output" "could not download"
    [ ! -e "${HOME}/.local/bin/nemo-relay" ] || fail "binary installed without a checksum"
    assert_no_temporary_files "${HOME}/.local/bin"
}

test_replace_existing_binary() {
    new_case
    install_dir="${HOME}/.local/bin"
    mkdir -p "$install_dir"
    printf 'existing binary\n' >"${install_dir}/nemo-relay"
    run_installer 0.5.0
    assert_success
    assert_file_contains "${install_dir}/nemo-relay" "mock nemo-relay"
    assert_no_temporary_files "$install_dir"
}

test_help_and_invalid_inputs() {
    new_case
    run_installer --help
    assert_success
    assert_contains "$run_output" "Usage:"
    [ ! -s "$curl_log" ] || fail "help attempted a download"

    run_installer --unknown
    assert_failure
    assert_contains "$run_output" "unknown option"

    run_installer not-a-version
    assert_failure
    assert_contains "$run_output" "unsupported version"

    HOME=""
    export HOME
    run_installer 0.5.0
    assert_failure
    assert_contains "$run_output" "install directory must not be empty"

    run_installer 0.5.0 --install-dir "${case_root}/no-home-bin"
    assert_success
}

test_latest_linux_x86_64
test_positional_version_precedence_and_v_normalization
test_environment_version_and_linux_arm64
test_macos_arm64
test_unsupported_platform
test_checksum_mismatch_preserves_existing_binary
test_missing_checksum_fails_closed
test_replace_existing_binary
test_help_and_invalid_inputs

printf 'PASS: %s installer scenarios\n' "$tests_run"
