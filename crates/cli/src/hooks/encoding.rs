// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hook definition and portable command encoding.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::agents::CodingAgent;

#[cfg(test)]
pub(crate) fn generated_hooks(agent: CodingAgent, command: &str) -> Value {
    generated_policy_hooks(agent, &GeneratedHookCommands::new(command, command))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GeneratedHookCommands {
    fail_open: String,
    fail_closed: String,
    legacy: Option<String>,
}

impl GeneratedHookCommands {
    pub(crate) fn new(fail_open: impl Into<String>, fail_closed: impl Into<String>) -> Self {
        Self {
            fail_open: fail_open.into(),
            fail_closed: fail_closed.into(),
            legacy: None,
        }
    }

    pub(crate) fn for_event(&self, event: &str) -> &str {
        if event_requires_fail_closed(event) {
            &self.fail_closed
        } else {
            &self.fail_open
        }
    }

    pub(crate) fn legacy(&self) -> Option<&str> {
        self.legacy.as_deref()
    }
}

pub(crate) fn generated_policy_hooks(
    agent: CodingAgent,
    commands: &GeneratedHookCommands,
) -> Value {
    grouped_hooks(agent.hook_events(), commands)
}

pub(crate) fn persistent_hook_forward_commands(
    relay: &Path,
    agent: CodingAgent,
    generation_file: &Path,
    _generation_token: &str,
) -> Result<GeneratedHookCommands, String> {
    hook_commands(
        agent,
        relay,
        &hook_config_arguments(agent, &persistent_hook_config_path(generation_file), false),
    )
}

#[cfg(test)]
pub(crate) fn transparent_hook_forward_commands(
    relay: &Path,
    agent: CodingAgent,
    gateway_url: &str,
) -> Result<GeneratedHookCommands, String> {
    hook_commands(
        agent,
        relay,
        &hook_config_arguments(agent, Path::new(gateway_url), true),
    )
}

pub(crate) fn transparent_hook_forward_commands_with_config(
    relay: &Path,
    agent: CodingAgent,
    hook_config: &Path,
) -> Result<GeneratedHookCommands, String> {
    #[cfg(windows)]
    if agent.hooks_use_powershell() {
        return transparent_powershell_hook_scripts(agent, relay, hook_config);
    }
    hook_commands(
        agent,
        relay,
        &hook_config_arguments(agent, hook_config, true),
    )
}

#[cfg(test)]
pub(crate) fn transparent_hook_forward_commands_for_platform(
    relay: &Path,
    agent: CodingAgent,
    gateway_url: &str,
    windows: bool,
) -> GeneratedHookCommands {
    hook_commands_for_platform(
        agent,
        relay,
        &hook_config_arguments(agent, Path::new(gateway_url), true),
        windows,
    )
}

#[cfg(test)]
pub(crate) fn persistent_hook_forward_commands_for_platform(
    relay: &Path,
    agent: CodingAgent,
    generation_file: &Path,
    _generation_token: &str,
    windows: bool,
) -> GeneratedHookCommands {
    hook_commands_for_platform(
        agent,
        relay,
        &hook_config_arguments(agent, &persistent_hook_config_path(generation_file), false),
        windows,
    )
}

pub(crate) fn persistent_hook_config_path(generation_file: &Path) -> PathBuf {
    generation_file.with_file_name(".nemo-relay-hook-config.json")
}

pub(super) fn hook_config_arguments(
    agent: CodingAgent,
    hook_config: &Path,
    transparent_run: bool,
) -> Vec<String> {
    let mut arguments = vec![
        "hook-forward".into(),
        agent.as_arg().into(),
        "--hook-config".into(),
        hook_config.display().to_string(),
    ];
    if transparent_run {
        arguments.push("--transparent-run".into());
    }
    arguments
}

fn hook_commands(
    agent: CodingAgent,
    relay: &Path,
    arguments: &[String],
) -> Result<GeneratedHookCommands, String> {
    let mut commands = GeneratedHookCommands::new(
        hook_command(agent, relay, &with_failure_policy(arguments, "--fail-open"))?,
        hook_command(
            agent,
            relay,
            &with_failure_policy(arguments, "--fail-closed"),
        )?,
    );
    commands.legacy = Some(hook_command(agent, relay, arguments)?);
    Ok(commands)
}

#[cfg(test)]
fn hook_commands_for_platform(
    agent: CodingAgent,
    relay: &Path,
    arguments: &[String],
    windows: bool,
) -> GeneratedHookCommands {
    let mut commands = GeneratedHookCommands::new(
        hook_command_for_platform(
            agent,
            relay,
            &with_failure_policy(arguments, "--fail-open"),
            windows,
        ),
        hook_command_for_platform(
            agent,
            relay,
            &with_failure_policy(arguments, "--fail-closed"),
            windows,
        ),
    );
    commands.legacy = Some(hook_command_for_platform(agent, relay, arguments, windows));
    commands
}

fn with_failure_policy(arguments: &[String], policy: &str) -> Vec<String> {
    arguments
        .iter()
        .cloned()
        .chain(std::iter::once(policy.to_string()))
        .collect()
}

pub(super) fn hook_command(
    agent: CodingAgent,
    relay: &Path,
    arguments: &[String],
) -> Result<String, String> {
    let command = render_hook_command(agent, relay, arguments, cfg!(windows));
    #[cfg(windows)]
    validate_windows_hook_command(&command)?;
    Ok(command)
}

#[cfg(any(windows, test))]
pub(super) fn validate_windows_hook_command(command: &str) -> Result<(), String> {
    let length = command.encode_utf16().count();
    if length > MAX_WINDOWS_HOOK_COMMAND_UTF16_UNITS {
        return Err(format!(
            "generated Windows coding-agent hook command is {length} characters and exceeds the {MAX_WINDOWS_HOOK_COMMAND_UTF16_UNITS}-character safety limit; shorten the Relay or hook configuration path"
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn hook_command_for_platform(
    agent: CodingAgent,
    relay: &Path,
    arguments: &[String],
    windows: bool,
) -> String {
    render_hook_command(agent, relay, arguments, windows)
}

fn powershell_hook_script(relay: &Path, arguments: &[String]) -> String {
    let fail_open = arguments
        .last()
        .is_some_and(|argument| argument == "--fail-open");
    let arguments = std::iter::once(relay_for_command(relay, true).display().to_string())
        .chain(arguments.iter().cloned())
        .map(|argument| powershell_literal(&argument))
        .collect::<Vec<_>>()
        .join(" ");
    let invocation = if fail_open {
        format!("try {{ & {arguments}; exit $LASTEXITCODE }} catch {{ exit 0 }}")
    } else {
        format!("& {arguments}; exit $LASTEXITCODE")
    };
    format!("$ErrorActionPreference = 'Stop'; {invocation}")
}

/// Keep the repeated session-hook overrides below cmd.exe's 8191-character limit.
/// The scripts live beside the process-private hook configuration and are cleaned up with it.
#[cfg(windows)]
fn transparent_powershell_hook_scripts(
    agent: CodingAgent,
    relay: &Path,
    hook_config: &Path,
) -> Result<GeneratedHookCommands, String> {
    let arguments = hook_config_arguments(agent, hook_config, true);
    let mut commands = Vec::new();
    for policy in ["--fail-open", "--fail-closed"] {
        let path = hook_config.with_file_name(format!("{policy}.ps1"));
        let command = format!(
            "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File {}; exit $LASTEXITCODE",
            powershell_literal(&path.display().to_string()),
        );
        validate_windows_hook_command(&command)?;
        let script = powershell_hook_script(relay, &with_failure_policy(&arguments, policy));
        // Windows PowerShell needs a BOM to read non-ASCII paths as UTF-8.
        crate::filesystem::atomic_write_private(&path, format!("\u{feff}{script}").as_bytes())?;
        commands.push(command);
    }
    Ok(GeneratedHookCommands::new(&commands[0], &commands[1]))
}

fn powershell_literal(argument: &str) -> String {
    format!(
        "'{}'",
        argument
            .replace('\'', "''")
            .replace('‘', "‘‘")
            .replace('’', "’’")
            .replace('‚', "‚‚")
            .replace('‛', "‛‛")
    )
}

fn render_hook_command(
    agent: CodingAgent,
    relay: &Path,
    arguments: &[String],
    windows: bool,
) -> String {
    // Claude Code executes command hooks through Bash, including on Windows.
    if windows && agent.hooks_use_bash() {
        return std::iter::once(bash_windows_path(&relay.display().to_string()))
            .chain(arguments.iter().enumerate().map(|(index, argument)| {
                if index > 0 && arguments[index - 1] == "--hook-config" {
                    bash_windows_path(argument)
                } else {
                    argument.clone()
                }
            }))
            .map(|argument| crate::process::shell_quote_arg_for_platform(&argument, false))
            .collect::<Vec<_>>()
            .join(" ");
    }
    if windows && agent.hooks_use_powershell() {
        use base64::Engine;
        // Codex uses PowerShell for Windows session hooks. Encode literal arguments,
        // then preserve the native exit code in the outer PowerShell hook runner too.
        let script = powershell_hook_script(relay, arguments);
        let bytes = script
            .encode_utf16()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        return format!(
            "powershell.exe -NoProfile -NonInteractive -EncodedCommand {encoded}; exit $LASTEXITCODE"
        );
    }
    let relay = relay_for_command(relay, windows);
    let command = std::iter::once(relay.display().to_string())
        .chain(arguments.iter().cloned())
        .map(|argument| crate::process::shell_quote_arg_for_platform(&argument, windows))
        .collect::<Vec<_>>()
        .join(" ");
    if windows {
        format!("\"{command}\"")
    } else {
        command
    }
}

// Verbatim drive and UNC prefixes are Windows filesystem syntax, not Bash paths.
fn bash_windows_path(raw: &str) -> String {
    let path = if let Some(unc) = raw.strip_prefix(r"\\?\UNC\") {
        format!("//{unc}")
    } else {
        raw.strip_prefix(r"\\?\").unwrap_or(raw).to_string()
    };
    path.replace('\\', "/")
}

#[cfg(windows)]
fn relay_for_command(relay: &Path, windows: bool) -> std::path::PathBuf {
    if windows {
        crate::process::short_windows_path(relay).unwrap_or_else(|| relay.to_path_buf())
    } else {
        relay.to_path_buf()
    }
}

#[cfg(not(windows))]
fn relay_for_command(relay: &Path, _windows: bool) -> std::path::PathBuf {
    relay.to_path_buf()
}

// `cmd.exe` accepts at most 8,191 characters. Leave room for `/C` and host-added text.
#[cfg(any(windows, test))]
const MAX_WINDOWS_HOOK_COMMAND_UTF16_UNITS: usize = 8_000;

fn grouped_hooks(events: &[&str], commands: &GeneratedHookCommands) -> Value {
    let hooks: serde_json::Map<String, Value> = events
        .iter()
        .map(|event| {
            let mut group = serde_json::Map::new();
            if event_matches_tools(event) {
                group.insert("matcher".into(), json!("*"));
            }
            group.insert(
                "hooks".into(),
                json!([{"type": "command", "command": commands.for_event(event), "timeout": 30}]),
            );
            (
                (*event).to_string(),
                Value::Array(vec![Value::Object(group)]),
            )
        })
        .collect();
    json!({ "hooks": Value::Object(hooks) })
}

pub(crate) fn event_matches_tools(event: &str) -> bool {
    matches!(
        event,
        "PreToolUse" | "PostToolUse" | "PostToolUseFailure" | "PermissionRequest"
    )
}

pub(crate) fn event_requires_fail_closed(event: &str) -> bool {
    matches!(
        event,
        "PreToolUse"
            | "PermissionRequest"
            | "pre_tool_call"
            | "tool_call"
            | "toolCall"
            | "user_bash"
            | "userBash"
    )
}
