// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn wrapper_probe_uses_last_host_token_and_validates_opaque_wrappers() {
    assert_eq!(
        version_probe_argv(
            CodingAgent::Codex,
            &command_argv("npm exec --package @openai/codex -- codex exec")
        ),
        [
            "npm",
            "exec",
            "--package",
            "@openai/codex",
            "--",
            "codex",
            "--version"
        ]
    );
    assert_eq!(
        version_probe_argv(
            CodingAgent::Codex,
            &command_argv("custom-codex-wrapper --profile dev")
        ),
        ["custom-codex-wrapper", "--profile", "dev", "--version"]
    );
}

#[test]
fn platform_resolution_supports_explicit_paths_and_windows_pathext() {
    let temp = tempfile::tempdir().unwrap();
    let shim = temp.path().join("codex.CMD");
    std::fs::write(&shim, "").unwrap();

    assert_eq!(
        resolve_executable_for_platform(
            "codex",
            Some(temp.path().as_os_str()),
            Some(std::ffi::OsStr::new(".EXE;.CMD")),
            true,
        ),
        Some(shim.clone())
    );
    assert_eq!(
        resolve_executable_for_platform(
            shim.to_str().unwrap(),
            None,
            Some(std::ffi::OsStr::new(".EXE;.CMD")),
            true,
        ),
        Some(shim)
    );
}

#[cfg(windows)]
#[test]
fn windows_command_shim_preserves_metacharacter_arguments() {
    let temp = tempfile::tempdir().unwrap();
    let shim = temp.path().join("agent shim.cmd");
    let marker = temp.path().join("completed.txt");
    std::fs::write(
        &shim,
        "@echo off\r\n\
         @if not \"%~1\"==\"space & value\" exit /b 11\r\n\
         @if not \"%~2\"==\"caret^value\" exit /b 12\r\n\
         @if not \"%~3\"==\"%%TOKEN%%\" exit /b 13\r\n\
         @echo ok>\"%NEMO_RELAY_ARGV_MARKER%\"\r\n",
    )
    .unwrap();
    let argv = vec![
        shim.display().to_string(),
        "space & value".into(),
        "caret^value".into(),
        "%TOKEN%".into(),
    ];
    let status = std_command(&argv)
        .env("NEMO_RELAY_ARGV_MARKER", &marker)
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(std::fs::read_to_string(marker).unwrap().trim(), "ok");
}
