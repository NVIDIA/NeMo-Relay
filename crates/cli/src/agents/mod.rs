// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical coding-agent identity and compatibility policy.

pub(crate) mod claude;
pub(crate) mod codex;
pub(crate) mod hermes;
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

pub(crate) fn marketplace_manifest(
    agent: CodingAgent,
    marketplace: &str,
    plugin: &str,
) -> serde_json::Value {
    match agent {
        CodingAgent::Codex => codex::assets::marketplace_manifest(marketplace, plugin),
        CodingAgent::ClaudeCode => claude::assets::marketplace_manifest(marketplace, plugin),
        CodingAgent::Hermes => unreachable!("Hermes does not install a marketplace plugin"),
    }
}

pub(crate) fn plugin_manifest(agent: CodingAgent, plugin: &str) -> serde_json::Value {
    match agent {
        CodingAgent::Codex => codex::assets::plugin_manifest(plugin),
        CodingAgent::ClaudeCode => claude::assets::plugin_manifest(plugin),
        CodingAgent::Hermes => unreachable!("Hermes does not install a marketplace plugin"),
    }
}

pub(crate) fn plugin_mcp_config(
    agent: CodingAgent,
    server: serde_json::Value,
) -> Result<serde_json::Value, String> {
    match agent {
        CodingAgent::Codex => codex::assets::mcp_config(server),
        CodingAgent::ClaudeCode => Ok(claude::assets::mcp_config(server)),
        CodingAgent::Hermes => unreachable!("Hermes does not install a marketplace plugin"),
    }
}

#[cfg(test)]
pub(crate) fn codex_mcp_env_vars_from(
    environment: impl IntoIterator<Item = String>,
    config: Option<&serde_json::Value>,
) -> Vec<String> {
    codex::assets::mcp_env_vars_from(environment, config)
}

pub(crate) fn prepare_launch(
    agent: CodingAgent,
    launch: &mut crate::process::PreparedAgentLaunch,
    gateway_url: &str,
    resolved: &crate::configuration::ResolvedConfig,
    dry_run: bool,
) -> Result<(), crate::error::CliError> {
    match agent {
        CodingAgent::Codex => codex::launch::prepare(launch, gateway_url),
        CodingAgent::ClaudeCode => claude::launch::prepare(launch, gateway_url, dry_run),
        CodingAgent::Hermes => hermes::launch::prepare(
            launch,
            resolved.agents.hermes.hooks_path.as_deref(),
            dry_run,
        ),
    }
}

pub(crate) use claude::host::{ClaudeSetupSnapshot, restore_claude_setup, snapshot_claude_setup};
pub(crate) use codex::host::{CodexSetupSnapshot, restore_codex_setup, snapshot_codex_setup};
pub(crate) use shared::host::portable_executable_path;
pub(crate) use shared::host::shell_quote_arg_for_platform;
#[cfg(test)]
pub(crate) use shared::host::strip_windows_verbatim_prefix;

use std::path::Path;

use serde_json::{Value, json};

use claude::host::claude_settings_base_url;
use codex::host::{
    codex_hook_trust_report, codex_hook_trust_report_with_generation, codex_hooks_installed,
    codex_hooks_installed_with_generation, codex_provider_installed, empty_codex_hook_trust_report,
    install_codex_with_generation, uninstall_codex,
};
use shared::host::{current_exe, healthz, print_check, print_info};

#[cfg(test)]
pub(super) use crate::bootstrap::DEFAULT_URL;

pub(crate) fn install_codex_plugin_with_generation(
    gateway_url: &str,
    plugin_root: &Path,
    generation_token: Option<&str>,
) -> Result<(), String> {
    install_codex_with_generation(
        gateway_url,
        &plugin_root.join("hooks").join("hooks.json"),
        generation_token,
    )
    .map(|_| ())
}

pub(crate) fn stop_plugin_gateway() -> Result<(), String> {
    crate::bootstrap::state::stop_owned_and_reset(crate::bootstrap::DEFAULT_URL)
}

pub(crate) fn uninstall_codex_plugin(gateway_url: &str, plugin_root: &Path) -> Result<(), String> {
    uninstall_codex(gateway_url, &plugin_root.join("hooks").join("hooks.json")).map(|_| ())
}

pub(crate) fn enable_claude_provider(gateway_url: &str) -> Result<(), String> {
    claude::host::enable_claude_provider(gateway_url)
}

pub(crate) fn restore_claude_provider(gateway_url: &str) -> Result<(), String> {
    claude::host::restore_claude_provider(gateway_url)
}

pub(crate) fn doctor_plugin(
    agent: CodingAgent,
    gateway_url: &str,
    plugin_root: &Path,
) -> Result<(), String> {
    doctor_plugin_with_generation(agent, gateway_url, plugin_root, None)
}

pub(crate) fn doctor_plugin_with_generation(
    agent: CodingAgent,
    gateway_url: &str,
    plugin_root: &Path,
    generation_token: Option<&str>,
) -> Result<(), String> {
    if doctor_ok(
        agent,
        gateway_url,
        Some(&plugin_root.join("hooks").join("hooks.json")),
        generation_token,
    )? {
        Ok(())
    } else {
        Err(format!("{} plugin doctor checks failed", agent.as_arg()))
    }
}

pub(crate) fn doctor_plugin_json(
    agent: CodingAgent,
    gateway_url: &str,
    plugin_root: &Path,
) -> Result<Value, String> {
    let plugin_binary = current_exe().ok().is_some_and(|path| path.exists());
    let sidecar_running = healthz(gateway_url);
    let (checks, ok, codex_trust) = match agent {
        CodingAgent::ClaudeCode => {
            let provider = claude_settings_base_url().as_deref() == Some(gateway_url);
            (
                json!({
                    "plugin_binary": plugin_binary,
                    "sidecar_running": sidecar_running,
                    "claude_provider_routing": provider
                }),
                plugin_binary && provider,
                None,
            )
        }
        CodingAgent::Codex => {
            let plugin_hooks_path = plugin_root.join("hooks").join("hooks.json");
            let provider = codex_provider_installed(gateway_url);
            let hooks = codex_hooks_installed(&plugin_hooks_path)?;
            let trust = if hooks {
                codex_hook_trust_report(&plugin_hooks_path)?
            } else {
                empty_codex_hook_trust_report()
            };
            let hooks_trusted = trust.ready();
            (
                json!({
                    "plugin_binary": plugin_binary,
                    "sidecar_running": sidecar_running,
                    "codex_provider_alias": provider,
                    "codex_hooks": hooks,
                    "codex_hooks_trusted": hooks_trusted
                }),
                plugin_binary && provider && hooks && hooks_trusted,
                Some(trust),
            )
        }
        other => {
            return Err(format!(
                "plugin doctor supports claude and codex, got {}",
                other.as_arg()
            ));
        }
    };
    let mut report = json!({
        "ok": ok,
        "sidecar_health": if sidecar_running {
            "running"
        } else {
            "not_running_mcp_start"
        },
        "checks": checks
    });
    if let Some(trust) = codex_trust {
        report["codex_hook_trust"] = trust.to_json();
    }
    Ok(report)
}

fn doctor_ok(
    agent: CodingAgent,
    gateway_url: &str,
    plugin_hooks_path: Option<&Path>,
    generation_token: Option<&str>,
) -> Result<bool, String> {
    let mut ok = true;
    ok &= print_check(
        "plugin binary",
        current_exe().ok().is_some_and(|path| path.exists()),
    );
    if healthz(gateway_url) {
        print_info("sidecar health", "running");
    } else {
        print_info(
            "sidecar health",
            "not running; the plugin MCP starts it when the host launches",
        );
    }
    match agent {
        CodingAgent::ClaudeCode => {
            ok &= print_check(
                "claude provider routing",
                claude_settings_base_url().as_deref() == Some(gateway_url),
            );
        }
        CodingAgent::Codex => {
            let plugin_hooks_path = plugin_hooks_path
                .ok_or_else(|| "Codex plugin hooks path is required for doctor".to_string())?;
            let provider = codex_provider_installed(gateway_url);
            let hooks = codex_hooks_installed_with_generation(plugin_hooks_path, generation_token)?;
            ok &= print_check("codex provider alias", provider);
            ok &= print_check("codex hooks", hooks);
            let trust = if hooks {
                codex_hook_trust_report_with_generation(plugin_hooks_path, generation_token)?
            } else {
                empty_codex_hook_trust_report()
            };
            ok &= print_check("codex hooks trusted and enabled", trust.ready());
            if !trust.ready() {
                print_info("codex hook trust", &trust.summary());
            }
        }
        other => {
            return Err(format!(
                "plugin doctor supports claude and codex, got {}",
                other.as_arg()
            ));
        }
    }
    Ok(ok)
}

#[cfg(test)]
use crate::bootstrap::*;
#[cfg(test)]
use crate::hooks::generated_hooks;
#[cfg(test)]
use claude::host::*;
#[cfg(test)]
use codex::app_server::*;
#[cfg(test)]
use codex::host::*;
#[cfg(test)]
use shared::host::*;

#[cfg(test)]
#[path = "../../tests/coverage/agents/plugin_host_tests.rs"]
mod host_tests;

#[cfg(test)]
#[path = "../../tests/coverage/agents/coding_agent_tests.rs"]
mod tests;
