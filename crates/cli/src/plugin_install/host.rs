// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host CLI discovery and marketplace registration commands.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::config::PluginHost;

use super::state::{PluginInstallOptions, PluginLayout};
use super::{MARKETPLACE_NAME, PLUGIN_NAME, RELAY_COMMAND, host_cli};

pub(super) fn run_host_marketplace_registration(
    host: PluginHost,
    layout: &PluginLayout,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<(), String> {
    run_command(
        host_cli(host),
        &[
            "plugin".into(),
            "marketplace".into(),
            "add".into(),
            layout.marketplace_root.display().to_string(),
        ],
        options,
        runner,
    )
}

pub(super) fn run_host_plugin_registration(
    host: PluginHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<(), String> {
    match host {
        PluginHost::Codex => run_command(
            host_cli(host),
            &[
                "plugin".into(),
                "add".into(),
                format!("{PLUGIN_NAME}@{MARKETPLACE_NAME}"),
            ],
            options,
            runner,
        ),
        PluginHost::ClaudeCode => run_command(
            host_cli(host),
            &[
                "plugin".into(),
                "install".into(),
                format!("{PLUGIN_NAME}@{MARKETPLACE_NAME}"),
                "--scope".into(),
                "user".into(),
            ],
            options,
            runner,
        ),
        PluginHost::All => unreachable!("all is expanded before host registration"),
    }
}

pub(super) fn run_host_plugin_removal(
    host: PluginHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<(), String> {
    match host {
        PluginHost::Codex => run_command(
            host_cli(host),
            &[
                "plugin".into(),
                "remove".into(),
                format!("{PLUGIN_NAME}@{MARKETPLACE_NAME}"),
            ],
            options,
            runner,
        )?,
        PluginHost::ClaudeCode => run_command(
            host_cli(host),
            &["plugin".into(), "uninstall".into(), PLUGIN_NAME.into()],
            options,
            runner,
        )?,
        PluginHost::All => unreachable!("all is expanded before host unregistration"),
    }
    Ok(())
}

pub(super) fn run_host_marketplace_removal(
    host: PluginHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<(), String> {
    run_command(
        host_cli(host),
        &[
            "plugin".into(),
            "marketplace".into(),
            "remove".into(),
            MARKETPLACE_NAME.into(),
        ],
        options,
        runner,
    )
}

pub(super) fn require_relay(
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<PathBuf, String> {
    if options.dry_run {
        return Ok(PathBuf::from(RELAY_COMMAND));
    }
    runner
        .resolve_executable(RELAY_COMMAND)?
        .ok_or_else(|| "required `nemo-relay` executable was not found on PATH".into())
}

pub(super) fn validate_relay_plugin_shim(
    relay: &Path,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<(), String> {
    if options.dry_run {
        return Ok(());
    }
    let args = ["plugin-shim".into(), "hook".into(), "--help".into()];
    let status = runner.run_quiet(relay, &args)?;
    if status == 0 {
        Ok(())
    } else {
        Err(format!(
            "{} failed with exit code {status}; installed hooks require `nemo-relay plugin-shim hook` support",
            format_command(&relay.display().to_string(), &args)
        ))
    }
}

pub(super) fn require_host_cli(
    host: PluginHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<(), String> {
    if options.dry_run {
        return Ok(());
    }
    let cli = host_cli(host);
    runner
        .resolve_executable(cli)?
        .map(|_| ())
        .ok_or_else(|| format!("required `{cli}` CLI was not found on PATH"))
}

pub(super) fn run_command(
    program: &str,
    args: &[String],
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<(), String> {
    if options.dry_run {
        println!("{}", format_command(program, args));
        return Ok(());
    }
    let resolved = runner
        .resolve_executable(program)?
        .ok_or_else(|| format!("required `{program}` executable was not found on PATH"))?;
    run_path_command(&resolved, args, options, runner)
}

pub(super) fn run_path_command(
    program: &Path,
    args: &[String],
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<(), String> {
    if options.dry_run {
        println!("{}", format_command(&program.display().to_string(), args));
        return Ok(());
    }
    let status = runner.run(program, args)?;
    if status == 0 {
        Ok(())
    } else {
        Err(format!(
            "{} failed with exit code {status}",
            format_command(&program.display().to_string(), args)
        ))
    }
}

pub(super) fn format_command(program: &str, args: &[String]) -> String {
    let mut parts = vec![program.to_string()];
    parts.extend(args.iter().cloned());
    format!(
        "$ {}",
        parts
            .iter()
            .map(|part| shell_quote(part))
            .collect::<Vec<_>>()
            .join(" ")
    )
}

fn shell_quote(raw: &str) -> String {
    if raw.chars().all(|ch| {
        ch.is_ascii_alphanumeric()
            || matches!(ch, '/' | '\\' | ':' | '.' | '_' | '-' | '=' | '@' | '+')
    }) {
        raw.into()
    } else {
        let mut escaped = String::new();
        for ch in raw.chars() {
            if matches!(ch, '"' | '\\' | '$' | '`') {
                escaped.push('\\');
            }
            escaped.push(ch);
        }
        format!("\"{escaped}\"")
    }
}

pub(super) trait CommandRunner {
    fn resolve_executable(&self, command: &str) -> Result<Option<PathBuf>, String>;
    fn run(&self, program: &Path, args: &[String]) -> Result<i32, String>;
    fn run_quiet(&self, program: &Path, args: &[String]) -> Result<i32, String>;
}

pub(super) struct RealCommandRunner;

impl CommandRunner for RealCommandRunner {
    fn resolve_executable(&self, command: &str) -> Result<Option<PathBuf>, String> {
        Ok(find_executable(command))
    }

    fn run(&self, program: &Path, args: &[String]) -> Result<i32, String> {
        #[cfg(windows)]
        if is_windows_command_script(program) {
            let status = Command::new(env::var_os("COMSPEC").unwrap_or_else(|| "cmd.exe".into()))
                .args(["/d", "/s", "/c"])
                .arg(windows_command_line(program, args))
                .status()
                .map_err(|error| format!("failed to run {}: {error}", program.display()))?;
            return Ok(status.code().unwrap_or(1));
        }

        let status = Command::new(program)
            .args(args)
            .status()
            .map_err(|error| format!("failed to run {}: {error}", program.display()))?;
        Ok(status.code().unwrap_or(1))
    }

    fn run_quiet(&self, program: &Path, args: &[String]) -> Result<i32, String> {
        #[cfg(windows)]
        if is_windows_command_script(program) {
            let status = Command::new(env::var_os("COMSPEC").unwrap_or_else(|| "cmd.exe".into()))
                .args(["/d", "/s", "/c"])
                .arg(windows_command_line(program, args))
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map_err(|error| format!("failed to run {}: {error}", program.display()))?;
            return Ok(status.code().unwrap_or(1));
        }

        let status = Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map_err(|error| format!("failed to run {}: {error}", program.display()))?;
        Ok(status.code().unwrap_or(1))
    }
}

fn find_executable(command: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    let candidates = env::split_paths(&path);
    let extensions = executable_extensions(command);
    for dir in candidates {
        for extension in &extensions {
            let candidate = dir.join(format!("{command}{extension}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn executable_extensions(command: &str) -> Vec<String> {
    if cfg!(windows) && Path::new(command).extension().is_none() {
        env::var("PATHEXT")
            .unwrap_or_else(|_| ".EXE;.CMD;.BAT;.COM".into())
            .split(';')
            .map(str::to_string)
            .collect()
    } else {
        vec![String::new()]
    }
}

#[cfg(windows)]
fn is_windows_command_script(program: &Path) -> bool {
    program
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("cmd") || extension.eq_ignore_ascii_case("bat")
        })
}

#[cfg(windows)]
fn windows_command_line(program: &Path, args: &[String]) -> String {
    std::iter::once(windows_command_argument(&program.display().to_string()))
        .chain(args.iter().map(|arg| windows_command_argument(arg)))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(windows)]
fn windows_command_argument(argument: &str) -> String {
    format!("\"{}\"", argument.replace('"', "\\\""))
}
