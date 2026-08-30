// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use semver::Version;

use super::AgentDescriptor;

pub(super) mod app_server;
pub(super) mod assets;
pub(crate) mod doctor;
pub(super) mod host;
pub(crate) mod install;
pub(crate) mod launch;

pub(super) const DESCRIPTOR: AgentDescriptor = AgentDescriptor {
    argument: "codex",
    install_argument: "codex",
    label: "Codex",
    executable: "codex",
    hook_path: "/hooks/codex",
    version_product: "codex-cli",
    minimum_version: (0, 143, 0),
    verified_through: None,
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
};

pub(super) fn parse_version(raw: &str) -> Option<Version> {
    Version::parse(raw.strip_prefix("codex-cli ")?).ok()
}

pub(super) fn versioned_gateway_url(gateway_url: &str) -> String {
    let gateway_url = gateway_url.trim_end_matches('/');
    if gateway_url.ends_with("/v1") {
        gateway_url.to_string()
    } else {
        format!("{gateway_url}/v1")
    }
}
