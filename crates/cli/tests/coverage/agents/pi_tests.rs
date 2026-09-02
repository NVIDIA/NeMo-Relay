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
    // Never forwarded: it fires before validation and for calls that never execute.
    assert!(!DESCRIPTOR.hook_events.contains(&"tool_execution_start"));
}

// The list gates nothing inbound -- an unrecognized event name becomes a mark rather than an
// error -- so its whole value is being an accurate inventory of what the extension posts. Pinning
// the exact set is what makes a hook added on one side and forgotten on the other visible here
// rather than in a trace nobody reads.
#[test]
fn hook_events_inventory_matches_what_the_extension_posts() {
    let mut declared = DESCRIPTOR.hook_events.to_vec();
    declared.sort_unstable();
    assert_eq!(
        declared,
        [
            "agent_end",
            "agent_settled",
            "agent_start",
            // Not pi hook names. The extension synthesizes these three: two report a decision it
            // took (redirect the model's provider, apply a rewrite), and `user_bash_end` closes
            // the inline-shell span because pi reports no completion for it.
            "model_redirect",
            "session_before_compact",
            "session_compact",
            "session_shutdown",
            "session_start",
            "tool_arguments_transformed",
            "tool_call",
            "tool_execution_end",
            "turn_end",
            "turn_start",
            "user_bash",
            "user_bash_end",
        ]
    );
}
