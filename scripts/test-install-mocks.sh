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
    return 0
}
trap cleanup EXIT HUP INT TERM

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
    return 1
}

assert_success() {
    [ "$run_status" -eq 0 ] || fail "expected success, got ${run_status}: ${run_output}"
    return 0
}

assert_failure() {
    [ "$run_status" -ne 0 ] || fail "expected failure: ${run_output}"
    return 0
}

assert_contains() {
    assert_actual=$1
    assert_expected=$2
    printf '%s\n' "$assert_actual" | grep -F "$assert_expected" >/dev/null || fail "expected '$assert_expected' in: $assert_actual"
    return 0
}

assert_file_contains() {
    assert_file=$1
    assert_expected=$2
    grep -F "$assert_expected" "$assert_file" >/dev/null || fail "expected '$assert_expected' in $assert_file"
    return 0
}

assert_no_temporary_files() {
    assert_directory=$1
    set -- "$assert_directory"/.nemo-relay.*
    assert_temporary_file=$1
    [ ! -e "$assert_temporary_file" ] || fail "temporary installer file was not cleaned up: $assert_temporary_file"
    return 0
}

make_mock_commands() {
    mock_commands_dir=$1
    mkdir -p "$mock_commands_dir"

    cat >"${mock_commands_dir}/uname" <<'EOF'
#!/bin/sh
case "${1:-}" in
    -s) printf '%s\n' "$MOCK_UNAME_S" ;;
    -m) printf '%s\n' "$MOCK_UNAME_M" ;;
    *) exit 1 ;;
esac
EOF

    cat >"${mock_commands_dir}/curl" <<'EOF'
#!/bin/sh
output=""
url=""
connect_timeout=""
max_time=""
authorization=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o) output=$2; shift 2 ;;
        -H)
            [ "$2" = "Authorization: Bearer ${MOCK_GH_TOKEN:-}" ] && authorization=1
            shift 2
            ;;
        --connect-timeout) connect_timeout=$2; shift 2 ;;
        --max-time) max_time=$2; shift 2 ;;
        -*) shift ;;
        *) url=$1; shift ;;
    esac
done

[ "$connect_timeout" = 10 ] || exit 97
[ "$max_time" = 300 ] || exit 98
printf '%s\n' "$url" >>"$MOCK_CURL_LOG"
case "$url" in
    */releases/latest)
        [ -z "${MOCK_GH_TOKEN:-}" ] || [ "$authorization" = 1 ] || exit 99
        printf '%s\n' "$MOCK_API_RESPONSE"
        ;;
    *.sha256)
        [ "${MOCK_CHECKSUM_MISSING:-0}" != 1 ] || exit 22
        printf '%s  %s\n' "$MOCK_EXPECTED_CHECKSUM" "${url##*/}" >"$output"
        ;;
    *)
        printf '#!/bin/sh\nprintf "mock nemo-relay\\n"\n' >"$output"
        ;;
esac
EOF

    cat >"${mock_commands_dir}/sha256sum" <<'EOF'
#!/bin/sh
printf '%s  %s\n' "$MOCK_ACTUAL_CHECKSUM" "$1"
EOF

    cat >"${mock_commands_dir}/cygpath" <<'EOF'
#!/bin/sh
case "${1:-}" in
    -u) printf '%s\n' "$2" ;;
    -w) printf 'C:%s\n' "${2#/}" ;;
    *) exit 1 ;;
esac
EOF

    cat >"${mock_commands_dir}/powershell.exe" <<'EOF'
#!/bin/sh
[ -n "${NEMO_RELAY_INSTALL_DIR:-}" ] || exit 99
printf '%s\n' "$NEMO_RELAY_INSTALL_DIR" >>"$MOCK_POWERSHELL_LOG"
EOF

    chmod +x "${mock_commands_dir}/uname" "${mock_commands_dir}/curl" "${mock_commands_dir}/sha256sum" \
        "${mock_commands_dir}/cygpath" "${mock_commands_dir}/powershell.exe"
    return 0
}

new_case() {
    tests_run=$((tests_run + 1))
    case_root="${test_root}/case-${tests_run}"
    home_dir="${case_root}/home"
    mock_bin="${case_root}/bin"
    curl_log="${case_root}/curl.log"
    powershell_log="${case_root}/powershell.log"
    mkdir -p "$home_dir"
    : >"$curl_log"
    : >"$powershell_log"
    make_mock_commands "$mock_bin"

    MOCK_UNAME_S=Linux
    MOCK_UNAME_M=x86_64
    MOCK_API_RESPONSE='{"tag_name":"0.5.0"}'
    MOCK_EXPECTED_CHECKSUM=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    MOCK_ACTUAL_CHECKSUM=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    MOCK_CHECKSUM_MISSING=0
    MOCK_GH_TOKEN=mock-github-token
    NEMO_RELAY_VERSION=0.5.0
    HOME=$home_dir
    PATH="${mock_bin}:${original_path}"
    MOCK_CURL_LOG=$curl_log
    MOCK_POWERSHELL_LOG=$powershell_log
    GH_TOKEN=$MOCK_GH_TOKEN
    export MOCK_UNAME_S MOCK_UNAME_M MOCK_API_RESPONSE
    export MOCK_EXPECTED_CHECKSUM MOCK_ACTUAL_CHECKSUM MOCK_CHECKSUM_MISSING MOCK_GH_TOKEN
    export GH_TOKEN NEMO_RELAY_VERSION HOME PATH MOCK_CURL_LOG MOCK_POWERSHELL_LOG
    return 0
}

run_installer() {
    if run_output=$(sh "$installer" "$@" 2>&1); then
        run_status=0
    else
        run_status=$?
    fi
    return 0
}

test_linux_arm64_mapping() {
    new_case
    MOCK_UNAME_M=aarch64
    export MOCK_UNAME_M
    run_installer
    assert_success
    assert_file_contains "$curl_log" "nemo-relay-cli-aarch64-unknown-linux-musl-0.5.0"
    return 0
}

test_macos_arm64_mapping() {
    new_case
    MOCK_UNAME_S=Darwin
    MOCK_UNAME_M=arm64
    export MOCK_UNAME_S MOCK_UNAME_M
    run_installer
    assert_success
    assert_file_contains "$curl_log" "nemo-relay-cli-aarch64-apple-darwin-0.5.0"
    return 0
}

test_git_bash_windows_x86_64_mapping_and_path_update() {
    new_case
    MOCK_UNAME_S=MINGW64_NT-10.0
    MOCK_UNAME_M=x86_64
    LOCALAPPDATA="${HOME}/AppData/Local"
    export MOCK_UNAME_S MOCK_UNAME_M LOCALAPPDATA
    run_installer
    assert_success
    assert_file_contains "$curl_log" "nemo-relay-cli-x86_64-pc-windows-msvc-0.5.0.exe"
    [ -f "${LOCALAPPDATA}/nemo-relay/bin/nemo-relay.exe" ] || fail "Windows install did not create nemo-relay.exe"
    assert_file_contains "$powershell_log" "C:${LOCALAPPDATA#/}/nemo-relay/bin"
    return 0
}

test_git_bash_windows_arm64_mapping() {
    new_case
    MOCK_UNAME_S=MSYS_NT-10.0
    MOCK_UNAME_M=arm64
    LOCALAPPDATA="${HOME}/AppData/Local"
    export MOCK_UNAME_S MOCK_UNAME_M LOCALAPPDATA
    run_installer --install-dir "${HOME}/custom-bin"
    assert_success
    assert_file_contains "$curl_log" "nemo-relay-cli-aarch64-pc-windows-msvc-0.5.0.exe"
    [ -f "${HOME}/custom-bin/nemo-relay.exe" ] || fail "Windows ARM64 install did not create nemo-relay.exe"
    assert_file_contains "$powershell_log" "C:${HOME#/}/custom-bin"
    return 0
}

test_unsupported_platform() {
    new_case
    MOCK_UNAME_S=Darwin
    MOCK_UNAME_M=x86_64
    export MOCK_UNAME_S MOCK_UNAME_M
    run_installer
    assert_failure
    assert_contains "$run_output" "unsupported platform Darwin/x86_64"
    [ ! -s "$curl_log" ] || fail "unsupported platform attempted a download"
    return 0
}

test_malformed_release_response() {
    new_case
    NEMO_RELAY_VERSION=""
    MOCK_API_RESPONSE='{"not_tag_name":"0.5.0"}'
    export NEMO_RELAY_VERSION MOCK_API_RESPONSE
    run_installer
    assert_failure
    assert_contains "$run_output" "latest release response did not contain a tag name"
    return 0
}

test_missing_checksum_fails_closed() {
    new_case
    MOCK_CHECKSUM_MISSING=1
    export MOCK_CHECKSUM_MISSING
    run_installer
    assert_failure
    assert_contains "$run_output" "could not download"
    [ ! -e "${HOME}/.local/bin/nemo-relay" ] || fail "binary installed without a checksum"
    assert_no_temporary_files "${HOME}/.local/bin"
    return 0
}

test_checksum_mismatch_preserves_existing_binary() {
    new_case
    install_dir="${HOME}/.local/bin"
    mkdir -p "$install_dir"
    printf 'existing binary\n' >"${install_dir}/nemo-relay"
    MOCK_ACTUAL_CHECKSUM=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    export MOCK_ACTUAL_CHECKSUM
    run_installer
    assert_failure
    assert_contains "$run_output" "checksum verification failed"
    assert_file_contains "${install_dir}/nemo-relay" "existing binary"
    assert_no_temporary_files "$install_dir"
    return 0
}

test_linux_arm64_mapping
test_macos_arm64_mapping
test_git_bash_windows_x86_64_mapping_and_path_update
test_git_bash_windows_arm64_mapping
test_unsupported_platform
test_malformed_release_response
test_missing_checksum_fails_closed
test_checksum_mismatch_preserves_existing_binary

printf 'PASS: %s mock-only installer scenarios\n' "$tests_run"
