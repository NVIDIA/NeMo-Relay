// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn agent_descriptors_are_complete_and_unique() {
    let arguments = CodingAgent::ALL.map(CodingAgent::as_arg);
    let install_arguments = CodingAgent::ALL.map(CodingAgent::install_arg);
    let executables = CodingAgent::ALL.map(CodingAgent::executable);
    let hook_paths = CodingAgent::ALL.map(CodingAgent::hook_path);

    assert_eq!(arguments, ["claude", "codex", "pi"]);
    assert_eq!(install_arguments, ["claude-code", "codex", "pi"]);
    assert_eq!(executables, ["claude", "codex", "pi"]);
    assert_eq!(
        hook_paths,
        ["/hooks/claude-code", "/hooks/codex", "/hooks/pi"]
    );
    assert_eq!(CodingAgent::ClaudeCode.label(), "Claude Code");
    assert_eq!(CodingAgent::Codex.label(), "Codex");
    assert_eq!(CodingAgent::ClaudeCode.hook_events().len(), 14);
    assert_eq!(CodingAgent::Codex.hook_events().len(), 11);
    assert_eq!(CodingAgent::Pi.label(), "pi");
    assert_eq!(CodingAgent::Pi.hook_events().len(), 15);
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
        (CodingAgent::Codex, "codex-cli 0.150.0"),
        // pi prints the bare version and nothing else, so there is no product token to match --
        // an accept path neither of the others exercises.
        (CodingAgent::Pi, "1.6.0"),
    ];

    for (agent, output) in cases {
        assert_eq!(
            agent.validate_version_output(output).unwrap(),
            agent.minimum_version()
        );
    }
}

#[test]
fn pi_reports_unverified_releases_above_its_minimum() {
    let pi = CodingAgent::Pi;
    let verified = pi.validate_version_output("1.6.0").unwrap();
    assert_eq!(pi.unverified_version(&verified), None);
    for untested in ["1.6.0+custom", "1.6.1", "1.7.0", "2.0.0"] {
        let version = pi.validate_version_output(untested).unwrap();
        let warning = pi.unverified_version(&version).unwrap();
        assert!(warning.contains(untested));
        assert!(warning.contains("1.6.0"));
    }

    // Claude Code and Codex declare no verified band, because their minors are additive.
    // Without this, adding a band to the descriptor would silently start warning for them.
    for (agent, newer) in [
        (CodingAgent::ClaudeCode, "99.0.0 (Claude Code)"),
        (CodingAgent::Codex, "codex-cli 99.0.0"),
    ] {
        let version = agent.validate_version_output(newer).unwrap();
        assert_eq!(agent.unverified_version(&version), None, "{agent:?}");
    }
}

#[test]
fn centralized_minimum_versions_reject_old_prerelease_and_malformed_output() {
    let cases = [
        (CodingAgent::ClaudeCode, "2.1.120 (Claude Code)"),
        (CodingAgent::ClaudeCode, "2.1.121-beta.1 (Claude Code)"),
        (CodingAgent::ClaudeCode, "2.1.121 (Other Agent)"),
        (CodingAgent::Codex, "codex-cli 0.149.1"),
        (CodingAgent::Codex, "codex-cli 0.150.0-alpha.1"),
        (CodingAgent::Pi, "0.84.0"),
        (CodingAgent::Pi, "1.5.1"),
        (CodingAgent::Pi, "1.6.0-rc.1"),
        // A prefixed line is not something pi emits, so it is a parse failure rather than an
        // old-version rejection.
        (CodingAgent::Pi, "pi 1.6.0"),
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
    assert_eq!(
        CodingAgent::infer(r"C:\\tools\\codex.bat"),
        Some(CodingAgent::Codex)
    );
    assert_eq!(
        CodingAgent::infer(r"C:\\tools\\codex.com"),
        Some(CodingAgent::Codex)
    );
    assert_eq!(CodingAgent::infer("@openai/codex"), None);
    assert_eq!(CodingAgent::infer("hermes"), None);
    assert_eq!(CodingAgent::infer("hermes-agent"), None);
    assert_eq!(CodingAgent::infer("unknown"), None);
}
