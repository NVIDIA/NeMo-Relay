// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;

pub(crate) fn command_has_arguments(command: &str, expected: &[&str]) -> bool {
    let arguments = crate::installer::decode_windows_hook_command(command)
        .or_else(|| shell_words::split(command).ok());
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
