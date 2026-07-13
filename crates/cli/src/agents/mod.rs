// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical coding-agent identity and compatibility policy.

pub(crate) mod hermes;
pub(crate) mod host;
pub(crate) mod install;
pub(crate) mod shared;

use clap::ValueEnum;
use semver::Version;

/// Coding-agent hosts supported by the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum CodingAgent {
    /// `claude-code` remains an input alias for older Relay configuration.
    #[value(name = "claude", alias = "claude-code")]
    ClaudeCode,
    Codex,
    Hermes,
}

#[derive(Debug, Clone, Copy)]
struct AgentDescriptor {
    argument: &'static str,
    install_argument: &'static str,
    label: &'static str,
    executable: &'static str,
    hook_path: &'static str,
    version_product: &'static str,
    minimum_version: (u64, u64, u64),
    version_format: VersionFormat,
    hook_events: &'static [&'static str],
    direct_hook_entries: bool,
}

#[derive(Debug, Clone, Copy)]
enum VersionFormat {
    Codex,
    ClaudeCode,
    Hermes,
}

// Claude Code validates plugin hooks.json against a strict event-name whitelist. Every event in
// this descriptor must therefore exist in the prescribed minimum Claude Code release.
const CLAUDE_CODE_HOOK_EVENTS: &[&str] = &[
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
];

const CODEX_HOOK_EVENTS: &[&str] = &[
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
];

const HERMES_HOOK_EVENTS: &[&str] = &[
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "pre_llm_call",
    "post_llm_call",
    "pre_api_request",
    "post_api_request",
    // Observer-only failure telemetry closes failed provider attempts.
    "api_request_error",
    "pre_tool_call",
    "post_tool_call",
    "subagent_start",
    "subagent_stop",
];

const CLAUDE_CODE: AgentDescriptor = AgentDescriptor {
    argument: "claude",
    install_argument: "claude-code",
    label: "Claude Code",
    executable: "claude",
    hook_path: "/hooks/claude-code",
    version_product: "Claude Code",
    minimum_version: (2, 1, 121),
    version_format: VersionFormat::ClaudeCode,
    hook_events: CLAUDE_CODE_HOOK_EVENTS,
    direct_hook_entries: false,
};

const CODEX: AgentDescriptor = AgentDescriptor {
    argument: "codex",
    install_argument: "codex",
    label: "Codex",
    executable: "codex",
    hook_path: "/hooks/codex",
    version_product: "codex-cli",
    minimum_version: (0, 143, 0),
    version_format: VersionFormat::Codex,
    hook_events: CODEX_HOOK_EVENTS,
    direct_hook_entries: false,
};

const HERMES: AgentDescriptor = AgentDescriptor {
    argument: "hermes",
    install_argument: "hermes",
    label: "Hermes Agent",
    executable: "hermes",
    hook_path: "/hooks/hermes",
    version_product: "Hermes Agent",
    minimum_version: (0, 18, 2),
    version_format: VersionFormat::Hermes,
    hook_events: HERMES_HOOK_EVENTS,
    direct_hook_entries: true,
};

impl CodingAgent {
    pub(crate) const ALL: [Self; 3] = [Self::ClaudeCode, Self::Codex, Self::Hermes];

    const fn descriptor(self) -> AgentDescriptor {
        match self {
            Self::ClaudeCode => CLAUDE_CODE,
            Self::Codex => CODEX,
            Self::Hermes => HERMES,
        }
    }

    /// Canonical CLI spelling used in generated commands and configuration.
    pub(crate) const fn as_arg(self) -> &'static str {
        self.descriptor().argument
    }

    /// Canonical spelling accepted by persistent integration commands.
    pub(crate) const fn install_arg(self) -> &'static str {
        self.descriptor().install_argument
    }

    /// Human-readable product name used in diagnostics.
    pub(crate) const fn label(self) -> &'static str {
        self.descriptor().label
    }

    /// Default executable name used for discovery and transparent launch.
    pub(crate) const fn executable(self) -> &'static str {
        self.descriptor().executable
    }

    /// Stable gateway endpoint used by lifecycle hooks.
    pub(crate) const fn hook_path(self) -> &'static str {
        self.descriptor().hook_path
    }

    /// Complete lifecycle event set installed for this host.
    pub(crate) const fn hook_events(self) -> &'static [&'static str] {
        self.descriptor().hook_events
    }

    /// Hermes stores direct command entries; plugin hosts use nested command-hook groups.
    pub(crate) const fn uses_direct_hook_entries(self) -> bool {
        self.descriptor().direct_hook_entries
    }

    pub(crate) fn minimum_version(self) -> Version {
        let (major, minor, patch) = self.descriptor().minimum_version;
        Version::new(major, minor, patch)
    }

    pub(crate) fn version_requirement(self) -> String {
        let descriptor = self.descriptor();
        format!(
            "{} {} or newer",
            descriptor.version_product,
            self.minimum_version()
        )
    }

    /// Parses and validates the first version line emitted by the host CLI.
    pub(crate) fn validate_version_output(self, raw: &str) -> Result<Version, String> {
        let first_line = raw.lines().next().unwrap_or_default().trim();
        let version = self.parse_version(first_line).ok_or_else(|| {
            format!(
                "could not parse `{} --version` output {:?}; NeMo Relay requires {}",
                self.executable(),
                raw.trim(),
                self.version_requirement()
            )
        })?;
        if version < self.minimum_version() || !version.pre.is_empty() {
            return Err(format!(
                "{} {version} is unsupported; NeMo Relay requires {}",
                self.descriptor().version_product,
                self.version_requirement()
            ));
        }
        Ok(version)
    }

    fn parse_version(self, raw: &str) -> Option<Version> {
        let descriptor = self.descriptor();
        let token = match descriptor.version_format {
            VersionFormat::Codex => raw.strip_prefix("codex-cli ")?,
            VersionFormat::ClaudeCode => raw.strip_suffix(" (Claude Code)")?,
            VersionFormat::Hermes => raw
                .strip_prefix("Hermes Agent v")?
                .split_whitespace()
                .next()?,
        };
        Version::parse(token).ok()
    }

    /// Infers a host from an executable basename.
    pub(crate) fn infer(command: &str) -> Option<Self> {
        let command = command.trim_matches(['"', '\'']);
        if command.starts_with('@') {
            return None;
        }
        let name = command
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(command)
            .to_ascii_lowercase();
        let name = [".exe", ".cmd", ".bat", ".com"]
            .into_iter()
            .find_map(|suffix| name.strip_suffix(suffix))
            .unwrap_or(&name);
        match name {
            "claude" | "claude-code" => Some(Self::ClaudeCode),
            "codex" => Some(Self::Codex),
            "hermes" | "hermes-agent" => Some(Self::Hermes),
            _ => None,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/agents/coding_agent_tests.rs"]
mod tests;
