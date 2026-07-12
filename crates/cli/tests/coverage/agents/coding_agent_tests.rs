// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn agent_descriptors_are_complete_and_unique() {
    let arguments = CodingAgent::ALL.map(CodingAgent::as_arg);
    let install_arguments = CodingAgent::ALL.map(CodingAgent::install_arg);
    let executables = CodingAgent::ALL.map(CodingAgent::executable);
    let hook_paths = CodingAgent::ALL.map(CodingAgent::hook_path);

    assert_eq!(arguments, ["claude", "codex", "hermes"]);
    assert_eq!(install_arguments, ["claude-code", "codex", "hermes"]);
    assert_eq!(executables, ["claude", "codex", "hermes"]);
    assert_eq!(
        hook_paths,
        ["/hooks/claude-code", "/hooks/codex", "/hooks/hermes"]
    );
    assert_eq!(CodingAgent::ClaudeCode.label(), "Claude Code");
    assert_eq!(CodingAgent::Codex.label(), "Codex");
    assert_eq!(CodingAgent::Hermes.label(), "Hermes Agent");
    assert_eq!(CodingAgent::ClaudeCode.hook_events().len(), 14);
    assert_eq!(CodingAgent::Codex.hook_events().len(), 10);
    assert_eq!(CodingAgent::Hermes.hook_events().len(), 13);
    assert!(!CodingAgent::ClaudeCode.uses_direct_hook_entries());
    assert!(!CodingAgent::Codex.uses_direct_hook_entries());
    assert!(CodingAgent::Hermes.uses_direct_hook_entries());
    for agent in CodingAgent::ALL {
        let events = agent.hook_events();
        assert!(events.iter().all(|event| !event.is_empty()));
        assert_eq!(
            events
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            events.len(),
            "{agent:?} declares duplicate lifecycle events"
        );
    }
}

#[test]
fn centralized_minimum_versions_accept_stable_boundaries() {
    let cases = [
        (CodingAgent::ClaudeCode, "2.1.121 (Claude Code)"),
        (CodingAgent::Codex, "codex-cli 0.143.0"),
        (CodingAgent::Hermes, "Hermes Agent v0.18.2 (2026.7.7.2)"),
    ];

    for (agent, output) in cases {
        assert_eq!(
            agent.validate_version_output(output).unwrap(),
            agent.minimum_version()
        );
    }
}

#[test]
fn centralized_minimum_versions_reject_old_prerelease_and_malformed_output() {
    let cases = [
        (CodingAgent::ClaudeCode, "2.1.120 (Claude Code)"),
        (CodingAgent::ClaudeCode, "2.1.121-beta.1 (Claude Code)"),
        (CodingAgent::ClaudeCode, "2.1.121 (Other Agent)"),
        (CodingAgent::Codex, "codex-cli 0.142.9"),
        (CodingAgent::Codex, "codex-cli 0.143.0-alpha.1"),
        (CodingAgent::Hermes, "Hermes Agent v0.18.1"),
        (CodingAgent::Hermes, "Hermes Agent v0.18.2-rc.1"),
    ];

    for (agent, output) in cases {
        assert!(
            agent.validate_version_output(output).is_err(),
            "{agent:?}: {output}"
        );
    }
    for agent in CodingAgent::ALL {
        assert!(agent.validate_version_output("unknown version").is_err());
        assert!(agent.validate_version_output("").is_err());
    }
}

#[test]
fn agent_inference_accepts_supported_binary_aliases() {
    assert_eq!(
        CodingAgent::infer("/opt/bin/claude"),
        Some(CodingAgent::ClaudeCode)
    );
    assert_eq!(
        CodingAgent::infer("claude-code"),
        Some(CodingAgent::ClaudeCode)
    );
    assert_eq!(CodingAgent::infer("codex"), Some(CodingAgent::Codex));
    assert_eq!(CodingAgent::infer("CODEX.EXE"), Some(CodingAgent::Codex));
    assert_eq!(
        CodingAgent::infer(r"C:\\tools\\codex.cmd"),
        Some(CodingAgent::Codex)
    );
    assert_eq!(CodingAgent::infer("@openai/codex"), None);
    assert_eq!(
        CodingAgent::infer("hermes-agent"),
        Some(CodingAgent::Hermes)
    );
    assert_eq!(CodingAgent::infer("unknown"), None);
}
