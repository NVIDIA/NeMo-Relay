// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use semver::Version;

use super::AgentDescriptor;

pub(super) mod app_server;
pub(super) mod host;

pub(super) const DESCRIPTOR: AgentDescriptor = AgentDescriptor {
    argument: "codex",
    install_argument: "codex",
    label: "Codex",
    executable: "codex",
    hook_path: "/hooks/codex",
    version_product: "codex-cli",
    minimum_version: (0, 143, 0),
    hook_events: &[
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PermissionRequest",
        "SubagentStart",
        "SubagentStop",
        "Stop",
        "PreCompact",
        "PostCompact",
    ],
    direct_hook_entries: false,
};

pub(super) fn parse_version(raw: &str) -> Option<Version> {
    Version::parse(raw.strip_prefix("codex-cli ")?).ok()
}
