// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn parses_bare_semver_version_output() {
    // `pi --version` prints only the version, with no product prefix, unlike
    // `codex-cli 0.143.0` or `2.1.121 (Claude Code)`.
    assert_eq!(parse_version("0.84.0"), Some(Version::new(0, 84, 0)));
    assert_eq!(parse_version("  0.84.0  "), Some(Version::new(0, 84, 0)));
}

#[test]
fn rejects_prefixed_or_empty_version_output() {
    assert_eq!(parse_version("pi 0.84.0"), None);
    assert_eq!(parse_version(""), None);
    assert_eq!(parse_version("not-a-version"), None);
}

#[test]
fn descriptor_routes_to_the_pi_hook_endpoint() {
    assert_eq!(DESCRIPTOR.hook_path, "/hooks/pi");
    assert_eq!(DESCRIPTOR.executable, "pi");
}

#[test]
fn hook_events_use_pi_vocabulary_not_codex_vocabulary() {
    // pi hooks originate in a NeMo Relay-authored extension, so the descriptor
    // lists pi's own hook names rather than PreToolUse/PostToolUse.
    assert!(DESCRIPTOR.hook_events.contains(&"tool_call"));
    assert!(DESCRIPTOR.hook_events.contains(&"agent_settled"));
    assert!(!DESCRIPTOR.hook_events.contains(&"PreToolUse"));
}
