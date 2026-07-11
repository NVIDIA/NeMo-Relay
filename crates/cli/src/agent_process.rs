// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared coding-agent command parsing, discovery, and process construction.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::config::CodingAgent;

/// Parses the intentionally simple command strings accepted by `[agents.*].command`.
///
/// Complex shell expressions belong after `nemo-relay run --`; configuration values are argv
/// prefixes and therefore use whitespace separation consistently in launch and diagnostics.
pub(crate) fn command_argv(command: &str) -> Vec<String> {
    command.split_whitespace().map(ToOwned::to_owned).collect()
}

/// Builds the host version probe while preserving a configured wrapper prefix.
///
/// The last recognizable host token wins so package selectors such as
/// `npm exec --package @openai/codex -- codex` do not truncate the probe at the package name.
/// Opaque wrappers must expose the selected host's version when passed `--version`.
pub(crate) fn version_probe_argv(agent: CodingAgent, argv: &[String]) -> Vec<String> {
    let mut probe = argv
        .iter()
        .rposition(|argument| CodingAgent::infer(argument) == Some(agent))
        .map_or_else(|| argv.to_vec(), |index| argv[..=index].to_vec());
    if probe.is_empty() {
        probe.push(agent.executable().into());
    }
    probe.push("--version".into());
    probe
}

/// Resolves a command using the current platform's executable conventions.
pub(crate) fn resolve_executable(command: &str) -> Option<PathBuf> {
    resolve_executable_for_platform(
        command,
        std::env::var_os("PATH").as_deref(),
        std::env::var_os("PATHEXT").as_deref(),
        cfg!(windows),
    )
}

/// Resolves a command against an explicit PATH. This keeps setup detection deterministic in tests.
pub(crate) fn resolve_executable_in_path(command: &str, path: Option<&OsStr>) -> Option<PathBuf> {
    resolve_executable_for_platform(
        command,
        path,
        std::env::var_os("PATHEXT").as_deref(),
        cfg!(windows),
    )
}

pub(crate) fn resolve_executable_for_platform(
    command: &str,
    path: Option<&OsStr>,
    path_ext: Option<&OsStr>,
    windows: bool,
) -> Option<PathBuf> {
    if command.is_empty() {
        return None;
    }
    let command_path = Path::new(command);
    let extensions = executable_extensions(command_path, path_ext, windows);
    if command_path.is_absolute() || command_path.components().count() > 1 {
        return resolve_candidate(command_path, &extensions);
    }
    path.into_iter()
        .flat_map(std::env::split_paths)
        .find_map(|directory| resolve_candidate(&directory.join(command), &extensions))
}

fn executable_extensions(command: &Path, path_ext: Option<&OsStr>, windows: bool) -> Vec<OsString> {
    if !windows || command.extension().is_some() {
        return vec![OsString::new()];
    }
    path_ext
        .and_then(OsStr::to_str)
        .filter(|value| !value.is_empty())
        .unwrap_or(".EXE;.CMD;.BAT;.COM")
        .split(';')
        .filter(|extension| !extension.is_empty())
        .map(OsString::from)
        .collect()
}

fn resolve_candidate(base: &Path, extensions: &[OsString]) -> Option<PathBuf> {
    extensions.iter().find_map(|extension| {
        let candidate = if extension.is_empty() {
            base.to_path_buf()
        } else {
            let mut value = base.as_os_str().to_os_string();
            value.push(extension);
            PathBuf::from(value)
        };
        candidate.is_file().then_some(candidate)
    })
}

/// Creates a synchronous command, including the `cmd.exe` bridge required by Windows shims.
pub(crate) fn std_command(argv: &[String]) -> Command {
    debug_assert!(!argv.is_empty());
    let program = resolve_executable(&argv[0]).unwrap_or_else(|| PathBuf::from(&argv[0]));
    #[cfg(windows)]
    if is_windows_command_script(&program) {
        let mut command =
            Command::new(std::env::var_os("COMSPEC").unwrap_or_else(|| OsString::from("cmd.exe")));
        command
            .args(["/d", "/s", "/c"])
            .arg(windows_command_line(&program, &argv[1..]));
        return command;
    }
    let mut command = Command::new(program);
    command.args(&argv[1..]);
    command
}

/// Creates an asynchronous command with the same platform behavior as [`std_command`].
pub(crate) fn tokio_command(argv: &[String]) -> tokio::process::Command {
    debug_assert!(!argv.is_empty());
    let program = resolve_executable(&argv[0]).unwrap_or_else(|| PathBuf::from(&argv[0]));
    #[cfg(windows)]
    if is_windows_command_script(&program) {
        let mut command = tokio::process::Command::new(
            std::env::var_os("COMSPEC").unwrap_or_else(|| OsString::from("cmd.exe")),
        );
        command
            .args(["/d", "/s", "/c"])
            .arg(windows_command_line(&program, &argv[1..]));
        return command;
    }
    let mut command = tokio::process::Command::new(program);
    command.args(&argv[1..]);
    command
}

#[cfg(any(windows, test))]
pub(crate) fn is_windows_command_script(program: &Path) -> bool {
    program
        .extension()
        .and_then(OsStr::to_str)
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("cmd") || extension.eq_ignore_ascii_case("bat")
        })
}

#[cfg(any(windows, test))]
pub(crate) fn windows_command_line(program: &Path, args: &[String]) -> String {
    std::iter::once(crate::plugin_host::shell_quote_arg_for_platform(
        &program.display().to_string(),
        true,
    ))
    .chain(
        args.iter()
            .map(|argument| crate::plugin_host::shell_quote_arg_for_platform(argument, true)),
    )
    .collect::<Vec<_>>()
    .join(" ")
}

#[cfg(test)]
#[path = "../tests/coverage/agent_process_tests.rs"]
mod tests;
