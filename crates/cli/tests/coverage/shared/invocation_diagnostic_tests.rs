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
        (CodingAgent::ClaudeCode, "claude-code"),
        (CodingAgent::ClaudeCode, "/opt/bin/claude"),
        (CodingAgent::ClaudeCode, "/opt/bin/claude-code.exe"),
        (CodingAgent::Codex, "codex"),
        (CodingAgent::Codex, r"C:\\tools\\CODEX.CMD"),
        (CodingAgent::Codex, r"C:\\tools\\codex.com"),
        (CodingAgent::Hermes, "hermes"),
        (CodingAgent::Hermes, "hermes-agent"),
        (CodingAgent::Hermes, "/opt/bin/hermes-agent.bat"),
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
fn invocation_diagnostic_warning_redacts_arguments_and_points_to_dry_run() {
    let command = argv(&["/opt/bin/claude-code", "-p", "private synthetic value"]);
    let diagnostic =
        DuplicateAgentExecutable::detect(CodingAgent::ClaudeCode, &command, InvocationForm::Run)
            .unwrap();

    let output = diagnostic.format_warning();
    assert!(output.contains("Diagnostic: possible_duplicate_agent_executable"));
    assert!(output.contains("Observed: nemo-relay run --agent claude -- claude"));
    assert!(output.contains("Recommended: nemo-relay run --agent claude --"));
    assert!(
        output.contains(
            "Inspect without launching: nemo-relay run --agent claude --dry-run -- claude"
        )
    );
    assert!(output.contains("<arguments redacted>"));
    assert!(!output.contains("/opt/bin/claude-code"));
    assert!(!output.contains("private synthetic value"));
}

#[test]
fn invocation_diagnostic_uses_the_shortcut_command_shape() {
    let command = argv(&["hermes-agent", "chat"]);
    let diagnostic =
        DuplicateAgentExecutable::detect(CodingAgent::Hermes, &command, InvocationForm::Shortcut)
            .unwrap();

    let output = diagnostic.format_warning();
    let redacted =
        crate::process::shell_quote_arg_for_platform("<arguments redacted>", cfg!(windows));
    assert!(output.contains("Observed: nemo-relay hermes -- hermes"));
    assert!(output.contains(&format!("Recommended: nemo-relay hermes -- {redacted}")));
    assert!(output.contains("Inspect without launching: nemo-relay hermes --dry-run -- hermes"));
}
