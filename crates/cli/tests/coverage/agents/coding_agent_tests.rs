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
    assert_eq!(CodingAgent::Codex.hook_events().len(), 10);
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
        (CodingAgent::Codex, "codex-cli 0.143.0"),
        // pi prints the bare version and nothing else, so there is no product token to match --
        // an accept path neither of the others exercises.
        (CodingAgent::Pi, "0.84.0"),
    ];

    for (agent, output) in cases {
        assert_eq!(
            agent.validate_version_output(output).unwrap(),
            agent.minimum_version()
        );
    }
}

// The floor alone said "supported" for any stable version above it, which is right for a host
// whose minors are additive and wrong for one that can move a hook shape in a minor. pi is the
// second kind: above 0.84.x it is untested, and the symptom of a broken hook shape is missing
// spans rather than an error, so silence is the worst possible report.
#[test]
fn a_minor_above_the_verified_band_is_reported_as_unverified_without_being_rejected() {
    let newer = CodingAgent::Pi.validate_version_output("0.85.0").unwrap();
    let note = CodingAgent::Pi
        .unverified_version(&newer)
        .expect("a newer pi minor must be reported");
    assert!(
        note.contains("0.84"),
        "the note should name the verified band: {note}"
    );

    // Still accepted: refusing would make a user downgrade pi to use Relay at all.
    assert!(CodingAgent::Pi.validate_version_output("0.85.0").is_ok());
    assert!(CodingAgent::Pi.validate_version_output("1.0.0").is_ok());

    // Inside the band, nothing to say -- including a patch above the floor.
    for inside in ["0.84.0", "0.84.7"] {
        let version = CodingAgent::Pi.validate_version_output(inside).unwrap();
        assert_eq!(
            CodingAgent::Pi.unverified_version(&version),
            None,
            "{inside}"
        );
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
        (CodingAgent::Codex, "codex-cli 0.142.9"),
        (CodingAgent::Codex, "codex-cli 0.143.0-alpha.1"),
        (CodingAgent::Pi, "0.83.9"),
        (CodingAgent::Pi, "0.84.0-alpha.1"),
        // A prefixed line is not something pi emits, so it is a parse failure rather than an
        // old-version rejection.
        (CodingAgent::Pi, "pi 0.84.0"),
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
