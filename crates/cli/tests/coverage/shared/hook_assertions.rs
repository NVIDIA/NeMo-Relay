// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;

pub(crate) fn decode_windows_hook_command(command: &str) -> Option<Vec<String>> {
    if let Some(encoded) =
        command.strip_prefix("powershell.exe -NoProfile -NonInteractive -EncodedCommand ")
    {
        use base64::Engine;
        let encoded = encoded.strip_suffix("; exit $LASTEXITCODE")?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .ok()?;
        let units = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        let script = String::from_utf16(&units).ok()?;
        let invocation = script.strip_prefix("$ErrorActionPreference = 'Stop'; ")?;
        let arguments = invocation
            .strip_prefix("try { & ")
            .and_then(|invocation| {
                invocation.strip_suffix("; exit $LASTEXITCODE } catch { exit 0 }")
            })
            .or_else(|| {
                invocation
                    .strip_prefix("& ")?
                    .strip_suffix("; exit $LASTEXITCODE")
            })?;
        return decode_powershell_literals(arguments);
    }
    let command = command
        .strip_prefix('"')
        .and_then(|command| command.strip_suffix('"'))
        .unwrap_or(command);
    shell_words::split(command).ok().map(|arguments| {
        arguments
            .into_iter()
            .map(|argument| argument.replace("^%", "%"))
            .collect()
    })
}

/// Decode the single-quoted, space-separated literals generated for PowerShell.
fn decode_powershell_literals(arguments: &str) -> Option<Vec<String>> {
    let mut chars = arguments.chars().peekable();
    let mut decoded = Vec::new();
    while chars.next() == Some('\'') {
        let mut argument = String::new();
        loop {
            let ch = chars.next()?;
            if matches!(ch, '\'' | '‘' | '’' | '‚' | '‛') {
                if chars.peek() == Some(&ch) {
                    chars.next();
                } else if ch == '\'' {
                    break;
                } else {
                    return None;
                }
            }
            argument.push(ch);
        }
        decoded.push(argument);
        match chars.next() {
            None => return Some(decoded),
            Some(' ') => {}
            _ => return None,
        }
    }
    None
}

pub(crate) fn command_has_arguments(command: &str, expected: &[&str]) -> bool {
    let arguments =
        decode_windows_hook_command(command).or_else(|| shell_words::split(command).ok());
    arguments.is_some_and(|arguments| {
        arguments.windows(expected.len()).any(|window| {
            window
                .iter()
                .map(String::as_str)
                .eq(expected.iter().copied())
        })
    })
}

pub(crate) fn value_has_command_arguments(value: &Value, expected: &[&str]) -> bool {
    match value {
        Value::String(_) => false,
        Value::Array(values) => values
            .iter()
            .any(|value| value_has_command_arguments(value, expected)),
        Value::Object(values) => values.iter().any(|(name, value)| {
            if name == "command" {
                value
                    .as_str()
                    .is_some_and(|command| command_has_arguments(command, expected))
            } else {
                value_has_command_arguments(value, expected)
            }
        }),
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

#[test]
fn command_matching_requires_complete_arguments() {
    assert!(command_has_arguments(
        "'/opt/NeMo Relay/nemo-relay' hook-forward codex --transparent-run",
        &["hook-forward", "codex", "--transparent-run"]
    ));
    assert!(!command_has_arguments(
        "nemo-relay hook-forward codex --transparent-run-disabled",
        &["hook-forward", "codex", "--transparent-run"]
    ));
}

#[test]
fn structured_matching_ignores_non_command_metadata() {
    let value = serde_json::json!({
        "description": "nemo-relay hook-forward codex --transparent-run",
        "handler": {
            "command": "nemo-relay hook-forward codex --transparent-run-disabled"
        }
    });
    assert!(!value_has_command_arguments(
        &value,
        &["hook-forward", "codex", "--transparent-run"]
    ));
}
