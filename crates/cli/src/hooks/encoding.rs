// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hook definition and portable command encoding.

use std::path::Path;

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
        relay,
        &hook_config_arguments(agent, &config_path(generation_file)),
    )
}

#[cfg(test)]
pub(crate) fn transparent_hook_forward_commands(
    relay: &Path,
    agent: CodingAgent,
    gateway_url: &str,
) -> Result<GeneratedHookCommands, String> {
    hook_commands(relay, &hook_config_arguments(agent, Path::new(gateway_url)))
}

pub(crate) fn transparent_hook_forward_commands_with_config(
    relay: &Path,
    agent: CodingAgent,
    hook_config: &Path,
) -> Result<GeneratedHookCommands, String> {
    hook_commands(relay, &hook_config_arguments(agent, hook_config))
}

#[cfg(test)]
pub(crate) fn transparent_hook_forward_commands_for_platform(
    relay: &Path,
    agent: CodingAgent,
    gateway_url: &str,
    windows: bool,
) -> GeneratedHookCommands {
    hook_commands_for_platform(
        relay,
        &hook_config_arguments(agent, Path::new(gateway_url)),
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
        relay,
        &hook_config_arguments(agent, &config_path(generation_file)),
        windows,
    )
}

fn config_path(generation_file: &Path) -> std::path::PathBuf {
    generation_file.with_file_name(".nemo-relay-hook-config.json")
}

pub(super) fn hook_config_arguments(agent: CodingAgent, hook_config: &Path) -> Vec<String> {
    vec![
        "hook-forward".into(),
        agent.as_arg().into(),
        "--hook-config".into(),
        hook_config.display().to_string(),
    ]
}

fn hook_commands(relay: &Path, arguments: &[String]) -> Result<GeneratedHookCommands, String> {
    let mut commands = GeneratedHookCommands::new(
        hook_command(relay, &with_failure_policy(arguments, "--fail-open"))?,
        hook_command(relay, &with_failure_policy(arguments, "--fail-closed"))?,
    );
    commands.legacy = Some(hook_command(relay, arguments)?);
    Ok(commands)
}

#[cfg(test)]
fn hook_commands_for_platform(
    relay: &Path,
    arguments: &[String],
    windows: bool,
) -> GeneratedHookCommands {
    let mut commands = GeneratedHookCommands::new(
        hook_command_for_platform(
            relay,
            &with_failure_policy(arguments, "--fail-open"),
            windows,
        ),
        hook_command_for_platform(
            relay,
            &with_failure_policy(arguments, "--fail-closed"),
            windows,
        ),
    );
    commands.legacy = Some(hook_command_for_platform(relay, arguments, windows));
    commands
}

fn with_failure_policy(arguments: &[String], policy: &str) -> Vec<String> {
    arguments
        .iter()
        .cloned()
        .chain(std::iter::once(policy.to_string()))
        .collect()
}

pub(super) fn hook_command(relay: &Path, arguments: &[String]) -> Result<String, String> {
    let command = render_hook_command(relay, arguments, cfg!(windows));
    #[cfg(windows)]
    if command.encode_utf16().count() > MAX_WINDOWS_HOOK_COMMAND_UTF16_UNITS {
        return Err(format!(
            "generated Windows coding-agent hook command is {} characters and exceeds the {MAX_WINDOWS_HOOK_COMMAND_UTF16_UNITS}-character safety limit; shorten the Relay or hook configuration path",
            command.encode_utf16().count()
        ));
    }
    Ok(command)
}

#[cfg(test)]
pub(super) fn hook_command_for_platform(
    relay: &Path,
    arguments: &[String],
    windows: bool,
) -> String {
    render_hook_command(relay, arguments, windows)
}

fn render_hook_command(relay: &Path, arguments: &[String], windows: bool) -> String {
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
#[cfg(windows)]
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
    matches!(event, "PreToolUse" | "PermissionRequest" | "pre_tool_call")
}
