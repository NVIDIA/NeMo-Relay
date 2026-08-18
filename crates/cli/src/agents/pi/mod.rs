// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! pi coding-agent identity and compatibility policy.
//!
//! pi has no native hook-configuration file and its external stream is
//! observation-only, so hook calls originate inside a pi *extension* that posts
//! to `/hooks/pi`. The extension is the only component that can gate a tool call
//! before it runs, which is why the hook event names below are pi's own
//! extension hook names rather than the `PreToolUse`/`PostToolUse` vocabulary
//! Codex and Claude Code use.

use semver::Version;

use super::AgentDescriptor;

pub(crate) mod doctor;
pub(crate) mod launch;

pub(super) const DESCRIPTOR: AgentDescriptor = AgentDescriptor {
    argument: "pi",
    install_argument: "pi",
    label: "pi",
    executable: "pi",
    hook_path: "/hooks/pi",
    version_product: "pi",
    // pi ships breaking changes through minor releases and has no major-release
    // channel, so this floor is the version the integration was verified
    // against rather than a lower bound that is expected to keep holding.
    minimum_version: (0, 84, 0),
    hook_events: &[
        "session_start",
        "session_shutdown",
        "agent_start",
        "agent_end",
        "agent_settled",
        "turn_start",
        "turn_end",
        "tool_call",
        "tool_execution_start",
        "tool_execution_end",
    ],
};

/// `pi --version` prints a bare semver line with no product prefix.
pub(super) fn parse_version(raw: &str) -> Option<Version> {
    Version::parse(raw.trim()).ok()
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/pi_tests.rs"]
mod tests;
