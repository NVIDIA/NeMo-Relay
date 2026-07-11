// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host CLI discovery and marketplace registration commands.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use semver::Version;
use serde_json::Value;

#[cfg(test)]
use serde_json::json;

use crate::config::PluginHost;

use super::state::PluginInstallOptions;
use super::{MARKETPLACE_NAME, PLUGIN_NAME, RELAY_COMMAND, host_cli};

pub(super) fn run_host_marketplace_registration(
    host: PluginHost,
    marketplace_root: &Path,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<(), String> {
    run_command(
        host_cli(host),
        &[
            "plugin".into(),
            "marketplace".into(),
            "add".into(),
            marketplace_root.display().to_string(),
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

#[derive(Debug, Clone)]
pub(super) struct HostRegistrationReport {
    pub(super) host_plugin_registered: bool,
    pub(super) host_marketplace_registered: bool,
}

impl HostRegistrationReport {
    pub(super) fn ok(&self) -> bool {
        self.host_plugin_registered && self.host_marketplace_registered
    }

    #[cfg(test)]
    pub(super) fn to_json(&self) -> Value {
        json!({
            "ok": self.ok(),
            "host_plugin_registered": self.host_plugin_registered,
            "host_marketplace_registered": self.host_marketplace_registered
        })
    }
}

#[cfg(test)]
pub(super) fn validate_host_registration(
    host: PluginHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<HostRegistrationReport, String> {
    let report = host_registration_report(host, options, runner)?;
    if report.ok() {
        Ok(report)
    } else {
        let mut missing = Vec::new();
        if !report.host_plugin_registered {
            missing.push(format!("{PLUGIN_NAME}@{MARKETPLACE_NAME} host plugin"));
        }
        if !report.host_marketplace_registered {
            missing.push(format!("{MARKETPLACE_NAME} host marketplace"));
        }
        Err(format!(
            "{} plugin host registration is incomplete: missing {}",
            host_cli(host),
            missing.join(", ")
        ))
    }
}

pub(super) fn host_registration_report(
    host: PluginHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<HostRegistrationReport, String> {
    if options.dry_run {
        return Ok(HostRegistrationReport {
            host_plugin_registered: true,
            host_marketplace_registered: true,
        });
    }
    require_host_cli(host, options, runner)?;
    Ok(match host {
        PluginHost::ClaudeCode => HostRegistrationReport {
            host_plugin_registered: claude_plugin_registered(options, runner)?,
            host_marketplace_registered: claude_marketplace_registered(options, runner)?,
        },
        PluginHost::Codex => HostRegistrationReport {
            host_plugin_registered: codex_plugin_registered(options, runner)?,
            host_marketplace_registered: codex_marketplace_registered(options, runner)?,
        },
        PluginHost::All => unreachable!("all is expanded before host registration checks"),
    })
}

fn claude_plugin_registered(
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<bool, String> {
    let output = run_capture_command(
        "claude",
        &["plugin".into(), "list".into(), "--json".into()],
        options,
        runner,
    )?;
    let plugins = parse_json_command_output("claude plugin list --json", output)?;
    Ok(plugins
        .as_array()
        .is_some_and(|plugins| plugins.iter().any(plugin_entry_matches)))
}

fn claude_marketplace_registered(
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<bool, String> {
    let output = run_capture_command(
        "claude",
        &[
            "plugin".into(),
            "marketplace".into(),
            "list".into(),
            "--json".into(),
        ],
        options,
        runner,
    )?;
    let marketplaces = parse_json_command_output("claude plugin marketplace list --json", output)?;
    Ok(marketplaces
        .as_array()
        .is_some_and(|marketplaces| marketplaces.iter().any(marketplace_entry_matches)))
}

fn codex_plugin_registered(
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<bool, String> {
    // Codex `plugin list` has no `--json` flag (unlike Claude Code).
    let output = run_capture_command("codex", &["plugin".into(), "list".into()], options, runner)?;
    let plugin_id = format!("{PLUGIN_NAME}@{MARKETPLACE_NAME}");
    Ok(output
        .stdout
        .lines()
        .any(|line| codex_plugin_line_installed(line, &plugin_id)))
}

fn codex_plugin_line_installed(line: &str, plugin_id: &str) -> bool {
    let mut columns = line.split_whitespace();
    if columns.next() != Some(plugin_id) {
        return false;
    }
    columns
        .next()
        .is_some_and(|status| status.starts_with("installed"))
}

fn codex_marketplace_registered(
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<bool, String> {
    let output = run_capture_command(
        "codex",
        &["plugin".into(), "marketplace".into(), "list".into()],
        options,
        runner,
    )?;
    Ok(output
        .stdout
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .any(|name| name == MARKETPLACE_NAME))
}

fn plugin_entry_matches(entry: &Value) -> bool {
    let plugin_id = format!("{PLUGIN_NAME}@{MARKETPLACE_NAME}");
    string_field(entry, "id") == Some(plugin_id.as_str())
        || string_field(entry, "pluginId") == Some(plugin_id.as_str())
        || (string_field(entry, "name") == Some(PLUGIN_NAME)
            && string_field(entry, "marketplaceName") == Some(MARKETPLACE_NAME))
}

fn marketplace_entry_matches(entry: &Value) -> bool {
    string_field(entry, "name") == Some(MARKETPLACE_NAME)
        || string_field(entry, "id") == Some(MARKETPLACE_NAME)
}

fn string_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn parse_json_command_output(command: &str, output: CommandOutput) -> Result<Value, String> {
    serde_json::from_str::<Value>(&output.stdout)
        .map_err(|error| format!("failed to parse `{command}` output as JSON: {error}"))
}

pub(super) fn require_relay(
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<PathBuf, String> {
    if options.dry_run {
        return Ok(PathBuf::from(RELAY_COMMAND));
    }
    runner
        .current_executable()
        .map(|path| path.canonicalize().unwrap_or(path))
        .map(crate::plugin_shim::portable_executable_path)
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

pub(super) fn validate_relay_mcp(
    relay: &Path,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<(), String> {
    if options.dry_run {
        return Ok(());
    }
    let args = ["mcp".into(), "--help".into()];
    let status = runner.run_quiet(relay, &args)?;
    if status == 0 {
        Ok(())
    } else {
        Err(format!(
            "{} failed with exit code {status}; coding-agent plugins require native `nemo-relay mcp` support",
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

pub(super) fn validate_codex_version(
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<(), String> {
    if options.dry_run {
        return Ok(());
    }
    let output = run_capture_command("codex", &["--version".into()], options, runner)?;
    let version = parse_codex_version(&output.stdout).ok_or_else(|| {
        format!(
            "could not parse `codex --version` output {:?}; the Relay plugin requires codex-cli 0.143.0 or newer",
            output.stdout.trim()
        )
    })?;
    if version >= Version::new(0, 143, 0) {
        Ok(())
    } else {
        Err(format!(
            "codex-cli {version} is unsupported; the Relay plugin requires codex-cli 0.143.0 or newer"
        ))
    }
}

fn parse_codex_version(raw: &str) -> Option<Version> {
    Version::parse(raw.trim().strip_prefix("codex-cli ")?).ok()
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

pub(super) fn run_capture_command(
    program: &str,
    args: &[String],
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<CommandOutput, String> {
    if options.dry_run {
        println!("{}", format_command(program, args));
        // Keep dry-run capture output syntactically valid for future callers that parse stdout.
        return Ok(CommandOutput::success("null\n".into()));
    }
    let resolved = runner
        .resolve_executable(program)?
        .ok_or_else(|| format!("required `{program}` executable was not found on PATH"))?;
    let output = runner.run_capture(&resolved, args)?;
    if output.status == 0 {
        Ok(output)
    } else {
        let stderr = output.stderr.trim();
        let detail = if stderr.is_empty() {
            String::new()
        } else {
            format!(": {stderr}")
        };
        Err(format!(
            "{} failed with exit code {}{detail}",
            format_command(&resolved.display().to_string(), args),
            output.status
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

#[derive(Debug, Clone)]
pub(super) struct CommandOutput {
    pub(super) status: i32,
    pub(super) stdout: String,
    pub(super) stderr: String,
}

impl CommandOutput {
    pub(super) fn success(stdout: String) -> Self {
        Self {
            status: 0,
            stdout,
            stderr: String::new(),
        }
    }
}

pub(super) trait CommandRunner {
    fn current_executable(&self) -> Result<PathBuf, String>;
    fn resolve_executable(&self, command: &str) -> Result<Option<PathBuf>, String>;
    fn run(&self, program: &Path, args: &[String]) -> Result<i32, String>;
    fn run_quiet(&self, program: &Path, args: &[String]) -> Result<i32, String>;
    fn run_capture(&self, program: &Path, args: &[String]) -> Result<CommandOutput, String>;
}

pub(super) struct RealCommandRunner;

impl CommandRunner for RealCommandRunner {
    fn current_executable(&self) -> Result<PathBuf, String> {
        env::current_exe()
            .map_err(|error| format!("failed to resolve current nemo-relay executable: {error}"))
    }

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

    fn run_capture(&self, program: &Path, args: &[String]) -> Result<CommandOutput, String> {
        #[cfg(windows)]
        if is_windows_command_script(program) {
            let output = Command::new(env::var_os("COMSPEC").unwrap_or_else(|| "cmd.exe".into()))
                .args(["/d", "/s", "/c"])
                .arg(windows_command_line(program, args))
                .output()
                .map_err(|error| format!("failed to run {}: {error}", program.display()))?;
            return Ok(command_output(output));
        }

        let output = Command::new(program)
            .args(args)
            .output()
            .map_err(|error| format!("failed to run {}: {error}", program.display()))?;
        Ok(command_output(output))
    }
}

fn command_output(output: std::process::Output) -> CommandOutput {
    CommandOutput {
        status: output.status.code().unwrap_or(1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
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
