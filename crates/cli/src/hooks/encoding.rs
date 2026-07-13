// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hook definition and portable command encoding.

use std::path::Path;

use serde_json::{Value, json};

use crate::agents::CodingAgent;

#[cfg(any(windows, test))]
use base64::Engine;

pub(crate) fn generated_hooks(agent: CodingAgent, command: &str) -> Value {
    if agent.uses_direct_hook_entries() {
        direct_hooks(agent.hook_events(), command)
    } else {
        grouped_hooks(agent.hook_events(), command)
    }
}

/// Canonical persistent hook command used by every supported host.
pub(crate) fn persistent_hook_forward_command(
    relay: &Path,
    agent: CodingAgent,
    generation_file: &Path,
    generation_token: &str,
) -> Result<String, String> {
    hook_command(
        relay,
        &persistent_hook_arguments(agent, generation_file, generation_token),
    )
}

/// Canonical transparent hook command. It embeds the process-private dynamic gateway so hook hosts
/// that filter inherited environment variables cannot redirect delivery to the fixed endpoint.
pub(crate) fn transparent_hook_forward_command(
    relay: &Path,
    agent: CodingAgent,
    gateway_url: &str,
) -> Result<String, String> {
    hook_command(relay, &transparent_hook_arguments(agent, gateway_url))
}

#[cfg(test)]
pub(crate) fn transparent_hook_forward_command_for_platform(
    relay: &Path,
    agent: CodingAgent,
    gateway_url: &str,
    windows: bool,
) -> String {
    hook_command_for_platform(
        relay,
        &transparent_hook_arguments(agent, gateway_url),
        windows,
    )
}

#[cfg(test)]
pub(crate) fn persistent_hook_forward_command_for_platform(
    relay: &Path,
    agent: CodingAgent,
    generation_file: &Path,
    generation_token: &str,
    windows: bool,
) -> String {
    hook_command_for_platform(
        relay,
        &persistent_hook_arguments(agent, generation_file, generation_token),
        windows,
    )
}

pub(super) fn transparent_hook_arguments(agent: CodingAgent, gateway_url: &str) -> Vec<String> {
    vec![
        "hook-forward".into(),
        agent.as_arg().into(),
        "--gateway-url".into(),
        gateway_url.into(),
        "--transparent-run".into(),
    ]
}

pub(super) fn persistent_hook_arguments(
    agent: CodingAgent,
    generation_file: &Path,
    generation_token: &str,
) -> Vec<String> {
    vec![
        "hook-forward".into(),
        agent.as_arg().into(),
        "--gateway-url".into(),
        crate::bootstrap::DEFAULT_URL.into(),
        "--generation-file".into(),
        generation_file.display().to_string(),
        "--generation-token".into(),
        generation_token.into(),
    ]
}

pub(super) fn hook_command(relay: &Path, arguments: &[String]) -> Result<String, String> {
    #[cfg(windows)]
    {
        return encoded_windows_hook_command(&windows_powershell_launcher()?, relay, arguments);
    }
    #[cfg(not(windows))]
    {
        Ok(posix_hook_command(relay, arguments))
    }
}

#[cfg(test)]
pub(super) fn hook_command_for_platform(
    relay: &Path,
    arguments: &[String],
    windows: bool,
) -> String {
    if windows {
        return encoded_windows_hook_command(
            "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
            relay,
            arguments,
        )
        .expect("test hook command must fit within the Windows command-line limit");
    }
    posix_hook_command(relay, arguments)
}

#[cfg(any(not(windows), test))]
pub(super) fn posix_hook_command(relay: &Path, arguments: &[String]) -> String {
    std::iter::once(relay.display().to_string())
        .chain(arguments.iter().cloned())
        .map(|argument| crate::agents::shell_quote_arg_for_platform(&argument, false))
        .collect::<Vec<_>>()
        .join(" ")
}

// `cmd.exe` accepts at most 8,191 characters. Leave room for `/C` and the executable path added
// by the hook host instead of generating a command that will be truncated at runtime.
#[cfg(any(windows, test))]
const MAX_WINDOWS_HOOK_COMMAND_UTF16_UNITS: usize = 8_000;

/// Encode a native Relay invocation so Windows hook hosts can pass it through `cmd.exe /C` as one
/// argument without corrupting quotes in canonical paths. Windows PowerShell is part of the
/// supported Windows platform; it only launches the Rust binary and preserves its standard I/O.
#[cfg(any(windows, test))]
pub(crate) fn encoded_windows_hook_command(
    powershell: &str,
    relay: &Path,
    arguments: &[String],
) -> Result<String, String> {
    const PREFIX: &str = "$ErrorActionPreference='Stop'; & ";
    const SUFFIX: &str = "; if ($null -eq $LASTEXITCODE) { exit 1 }; exit $LASTEXITCODE";

    let invocation = std::iter::once(relay.display().to_string())
        .chain(arguments.iter().cloned())
        .map(|argument| format!("'{}'", argument.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(" ");
    let script = format!("{PREFIX}{invocation}{SUFFIX}");
    let bytes = script
        .encode_utf16()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    let command =
        format!("{powershell} -NoLogo -NoProfile -NonInteractive -EncodedCommand {encoded}");
    if command.encode_utf16().count() > MAX_WINDOWS_HOOK_COMMAND_UTF16_UNITS {
        return Err(format!(
            "generated Windows coding-agent hook command exceeds the {MAX_WINDOWS_HOOK_COMMAND_UTF16_UNITS}-character safety limit; shorten the Relay or plugin installation path"
        ));
    }
    Ok(command)
}

#[cfg(windows)]
pub(super) fn windows_powershell_launcher() -> Result<String, String> {
    let powershell = windows_powershell_path()?;
    if !Path::new(&powershell).is_file() {
        return Err(format!(
            "trusted Windows PowerShell launcher is missing at {powershell}; install Windows PowerShell before configuring coding-agent hooks"
        ));
    }
    Ok(powershell)
}

#[cfg(windows)]
pub(super) fn windows_powershell_path() -> Result<String, String> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;

    let mut buffer = vec![0_u16; 260];
    let length = loop {
        // SAFETY: `buffer` is writable for its declared length and remains live for the call.
        let length = unsafe { GetSystemDirectoryW(buffer.as_mut_ptr(), buffer.len() as u32) };
        if length == 0 {
            return Err(format!(
                "failed to resolve the trusted Windows system directory: {}",
                std::io::Error::last_os_error()
            ));
        }
        if (length as usize) < buffer.len() {
            break length as usize;
        }
        buffer.resize(length as usize + 1, 0);
    };
    let system = std::path::PathBuf::from(std::ffi::OsString::from_wide(&buffer[..length]));
    let powershell = system.join("WindowsPowerShell/v1.0/powershell.exe");
    let powershell = powershell
        .into_os_string()
        .into_string()
        .map_err(|_| "trusted Windows PowerShell path is not valid Unicode".to_string())?
        .replace('\\', "/");
    if !safe_windows_launcher_token(&powershell) {
        return Err(format!(
            "trusted Windows PowerShell path {powershell} contains characters that cannot be represented safely in coding-agent hook commands"
        ));
    }
    Ok(powershell)
}

#[cfg(any(windows, test))]
pub(super) fn safe_windows_launcher_token(launcher: &str) -> bool {
    !launcher.is_empty()
        && launcher.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '/' | ':' | '.' | '_' | '-')
        })
        && launcher
            .to_ascii_lowercase()
            .ends_with("/system32/windowspowershell/v1.0/powershell.exe")
}

/// Decode only the exact PowerShell envelope emitted by [`encoded_windows_hook_command`].
///
/// Hermes uses this to migrate and replace Relay-owned hooks whose generation arguments change.
#[cfg(test)]
pub(crate) fn decode_windows_hook_command(command: &str) -> Option<Vec<String>> {
    const COMMAND_SEPARATOR: &str = " -NoLogo -NoProfile -NonInteractive -EncodedCommand ";
    const SCRIPT_PREFIX: &str = "$ErrorActionPreference='Stop'; & ";
    const SCRIPT_SUFFIX: &str = "; if ($null -eq $LASTEXITCODE) { exit 1 }; exit $LASTEXITCODE";

    if command.encode_utf16().count() > MAX_WINDOWS_HOOK_COMMAND_UTF16_UNITS {
        return None;
    }
    let (launcher, encoded) = command.split_once(COMMAND_SEPARATOR)?;
    if !safe_windows_launcher_token(launcher) {
        return None;
    }
    #[cfg(windows)]
    if !launcher.eq_ignore_ascii_case(&windows_powershell_path().ok()?) {
        return None;
    }
    if encoded.is_empty() || encoded.chars().any(char::is_whitespace) {
        return None;
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .ok()?;
    let pairs = bytes.chunks_exact(2);
    if !pairs.remainder().is_empty() {
        return None;
    }
    let script = String::from_utf16(
        &pairs
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>(),
    )
    .ok()?;
    let invocation = script
        .strip_prefix(SCRIPT_PREFIX)?
        .strip_suffix(SCRIPT_SUFFIX)?;
    parse_powershell_single_quoted_arguments(invocation)
}

#[cfg(test)]
pub(super) fn parse_powershell_single_quoted_arguments(mut raw: &str) -> Option<Vec<String>> {
    let mut arguments = Vec::new();
    while !raw.is_empty() {
        raw = raw.strip_prefix('\'')?;
        let mut argument = String::new();
        loop {
            let quote = raw.find('\'')?;
            argument.push_str(&raw[..quote]);
            raw = &raw[quote + 1..];
            if let Some(rest) = raw.strip_prefix('\'') {
                argument.push('\'');
                raw = rest;
            } else {
                break;
            }
        }
        arguments.push(argument);
        if raw.is_empty() {
            break;
        }
        raw = raw.strip_prefix(' ')?;
        if raw.is_empty() {
            return None;
        }
    }
    (!arguments.is_empty()).then_some(arguments)
}

pub(super) fn direct_hooks(events: &[&str], command: &str) -> Value {
    let hooks: serde_json::Map<String, Value> = events
        .iter()
        .map(|event| {
            (
                (*event).to_string(),
                json!([{
                    "command": command,
                    "timeout": 30
                }]),
            )
        })
        .collect();
    json!({ "hooks": Value::Object(hooks) })
}

// Generates hook groups for Claude/Codex events and adds a wildcard matcher to tool events when
// the target agent requires matcher-scoped tool hooks. Non-tool events omit matchers so they fire
// for the full lifecycle.
pub(super) fn grouped_hooks(events: &[&str], command: &str) -> Value {
    let hooks: serde_json::Map<String, Value> = events
        .iter()
        .map(|event| {
            let mut group = serde_json::Map::new();
            if event_matches_tools(event) {
                group.insert("matcher".into(), json!("*"));
            }
            group.insert(
                "hooks".into(),
                json!([{
                    "type": "command",
                    "command": command,
                    "timeout": 30
                }]),
            );
            (
                (*event).to_string(),
                Value::Array(vec![Value::Object(group)]),
            )
        })
        .collect();
    json!({ "hooks": Value::Object(hooks) })
}

// Identifies hook events that should receive wildcard tool matchers. The list includes current
// Claude/Codex spellings.
pub(crate) fn event_matches_tools(event: &str) -> bool {
    matches!(
        event,
        "PreToolUse" | "PostToolUse" | "PostToolUseFailure" | "PermissionRequest"
    )
}
