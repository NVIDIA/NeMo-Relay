// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use semver::Version;

use super::AgentDescriptor;

pub(super) mod assets;
pub(super) mod host;

pub(super) const DESCRIPTOR: AgentDescriptor = AgentDescriptor {
    argument: "claude",
    install_argument: "claude-code",
    label: "Claude Code",
    executable: "claude",
    hook_path: "/hooks/claude-code",
    version_product: "Claude Code",
    minimum_version: (2, 1, 121),
    hook_events: &[
        "SessionStart",
        "UserPromptSubmit",
        "UserPromptExpansion",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PermissionRequest",
        "SubagentStart",
        "SubagentStop",
        "Notification",
        "Stop",
        "PreCompact",
        "PostCompact",
        "SessionEnd",
    ],
    direct_hook_entries: false,
};

pub(super) fn parse_version(raw: &str) -> Option<Version> {
    Version::parse(raw.strip_suffix(" (Claude Code)")?).ok()
}
