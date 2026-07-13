// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical coding-agent identity and compatibility policy.

pub(crate) mod claude;
pub(crate) mod codex;
pub(crate) mod hermes;
pub(crate) mod host;
pub(crate) mod install;
pub(crate) mod shared;

use semver::Version;

/// Coding-agent hosts supported by the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CodingAgent {
    /// `claude-code` remains an input alias for older Relay configuration.
    ClaudeCode,
    Codex,
    Hermes,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AgentDescriptor {
    argument: &'static str,
    install_argument: &'static str,
    label: &'static str,
    executable: &'static str,
    hook_path: &'static str,
    version_product: &'static str,
    minimum_version: (u64, u64, u64),
    hook_events: &'static [&'static str],
    direct_hook_entries: bool,
}

impl CodingAgent {
    pub(crate) const ALL: [Self; 3] = [Self::ClaudeCode, Self::Codex, Self::Hermes];

    const fn descriptor(self) -> AgentDescriptor {
        match self {
            Self::ClaudeCode => claude::DESCRIPTOR,
            Self::Codex => codex::DESCRIPTOR,
            Self::Hermes => hermes::DESCRIPTOR,
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
        match self {
            Self::ClaudeCode => claude::parse_version(raw),
            Self::Codex => codex::parse_version(raw),
            Self::Hermes => hermes::parse_version(raw),
        }
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
