// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use semver::Version;

use super::AgentDescriptor;

pub(super) const DESCRIPTOR: AgentDescriptor = AgentDescriptor {
    argument: "hermes",
    install_argument: "hermes",
    label: "Hermes Agent",
    executable: "hermes",
    hook_path: "/hooks/hermes",
    version_product: "Hermes Agent",
    minimum_version: (0, 18, 2),
    hook_events: &[
        "on_session_start",
        "on_session_end",
        "on_session_finalize",
        "on_session_reset",
        "pre_llm_call",
        "post_llm_call",
        "pre_api_request",
        "post_api_request",
        "api_request_error",
        "pre_tool_call",
        "post_tool_call",
        "subagent_start",
        "subagent_stop",
    ],
    direct_hook_entries: true,
};

pub(super) fn parse_version(raw: &str) -> Option<Version> {
    Version::parse(
        raw.strip_prefix("Hermes Agent v")?
            .split_whitespace()
            .next()?,
    )
    .ok()
}

mod config;
mod files;
mod integration;
mod trust;

pub(crate) use integration::*;
