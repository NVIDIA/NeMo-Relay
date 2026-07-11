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

#[test]
fn windows_command_line_quotes_program_and_arguments() {
    assert!(is_windows_command_script(Path::new("codex.CMD")));
    assert!(!is_windows_command_script(Path::new("codex.exe")));
    let command = windows_command_line(
        Path::new(r"C:\Program Files\Relay & Co\codex.cmd"),
        &["exec".into(), "a & b".into(), "%TOKEN%".into()],
    );
    assert!(command.contains(r#""C:\Program Files\Relay ^& Co\codex.cmd""#));
    assert!(command.contains(r#""a ^& b""#));
    assert!(command.contains("%%TOKEN%%"));
}
