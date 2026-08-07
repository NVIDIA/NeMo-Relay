// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn argv(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

#[test]
fn invocation_diagnostic_detects_supported_agent_names_aliases_and_paths() {
    let cases = [
        (CodingAgent::ClaudeCode, "claude"),
        (CodingAgent::ClaudeCode, "/opt/bin/claude-code.exe"),
        (CodingAgent::Codex, r"C:\\tools\\CODEX.CMD"),
        (CodingAgent::Hermes, "hermes-agent"),
    ];

    for (agent, executable) in cases {
        assert!(
            DuplicateAgentExecutable::detect(
                agent,
                &argv(&[executable, "synthetic argument"]),
                InvocationForm::Run,
            )
            .is_some(),
            "expected {executable:?} to duplicate {agent:?}"
        );
    }
}

#[test]
fn invocation_diagnostic_only_inspects_the_first_post_boundary_token() {
    for command in [
        argv(&[]),
        argv(&["-p", "claude appears later"]),
        argv(&["my-wrapper", "claude"]),
        argv(&["codex", "claude"]),
    ] {
        assert!(
            DuplicateAgentExecutable::detect(
                CodingAgent::ClaudeCode,
                &command,
                InvocationForm::Run,
            )
            .is_none(),
            "unexpected duplicate for {command:?}"
        );
    }
}

#[test]
fn invocation_diagnostic_doctor_redacts_arguments_unless_explicitly_requested() {
    let diagnostic = DuplicateAgentExecutable::detect(
        CodingAgent::ClaudeCode,
        &argv(&["/opt/bin/claude-code", "-p", "private synthetic value"]),
        InvocationForm::Run,
    )
    .unwrap();

    let safe = diagnostic.format_doctor(false);
    assert!(safe.contains("code = possible_duplicate_agent_executable"));
    assert!(safe.contains("observed = nemo-relay run --agent claude -- claude"));
    assert!(safe.contains("<arguments redacted>"));
    assert!(!safe.contains("/opt/bin/claude-code"));
    assert!(!safe.contains("private synthetic value"));

    let full = diagnostic.format_doctor(true);
    assert!(full.contains("/opt/bin/claude-code"));
    assert!(full.contains("private synthetic value"));
    assert!(full.contains("recommended = nemo-relay run --agent claude -- -p"));
}

#[test]
fn invocation_diagnostic_uses_the_shortcut_command_shape() {
    let diagnostic = DuplicateAgentExecutable::detect(
        CodingAgent::Hermes,
        &argv(&["hermes-agent", "chat"]),
        InvocationForm::Shortcut,
    )
    .unwrap();

    let output = diagnostic.format_doctor(false);
    assert!(output.contains("observed = nemo-relay hermes -- hermes"));
    assert!(output.contains("recommended = nemo-relay hermes -- '<arguments redacted>'"));
}
