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

pub(crate) mod alignment;
pub(crate) mod assets;
pub(crate) mod doctor;
pub(crate) mod install;
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
    // Which is why the floor alone was a lie by omission: it accepted 0.85.0 as
    // "supported" for a host that can move a hook shape in a minor. Below the floor
    // is an error, above this is a warning -- untested, not broken, and blocking a
    // launch over it would make the user downgrade pi to use Relay at all.
    verified_through: Some((0, 84)),
    // The hooks the extension actually posts to `/hooks/pi`, which is narrower than the set it
    // registers with pi. `tool_execution_start` is registered but never forwarded -- it fires
    // before validation and for calls that never execute, so it is used only to remember a tool
    // name for the matching end -- and listing it here described a hook the gateway never sees.
    hook_events: &[
        "session_start",
        "session_shutdown",
        "session_before_compact",
        "session_compact",
        "agent_start",
        "agent_end",
        "agent_settled",
        "turn_start",
        "turn_end",
        "tool_call",
        "tool_execution_end",
        // Not a pi hook name: posted after a request intercept's rewrite is applied to a tool
        // call, so the trace records that the arguments the tool ran were not the ones proposed.
        "tool_arguments_transformed",
        // The bang-prefixed inline shell gate. `user_bash` is a pi hook; `user_bash_end` is not --
        // pi reports no completion for inline shell, so the extension synthesizes the close.
        "user_bash",
        "user_bash_end",
        // Not a pi hook name: the extension posts this after deciding whether to point the active
        // model's provider at the gateway, so a trace with no LLM spans carries its own reason.
        "model_redirect",
    ],
};

/// `pi --version` prints a bare semver line with no product prefix.
pub(super) fn parse_version(raw: &str) -> Option<Version> {
    Version::parse(raw.trim()).ok()
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/pi_tests.rs"]
mod tests;
