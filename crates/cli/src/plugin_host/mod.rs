// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transactional host configuration for packaged coding-agent plugins.

mod claude;
mod codex;
mod codex_app_server;
mod shared;

pub(crate) use claude::{ClaudeSetupSnapshot, restore_claude_setup, snapshot_claude_setup};
pub(crate) use codex::{CodexSetupSnapshot, restore_codex_setup, snapshot_codex_setup};
pub(crate) use shared::portable_executable_path;
pub(crate) use shared::shell_quote_arg_for_platform;
pub(crate) use shared::shell_quote_for_platform;
#[cfg(test)]
pub(crate) use shared::strip_windows_verbatim_prefix;

use std::path::Path;

use serde_json::{Value, json};

use claude::claude_settings_base_url;
use codex::{codex_hook_trust_report, empty_codex_hook_trust_report};
use codex::{codex_hooks_installed, codex_provider_installed, install_codex, uninstall_codex};
use shared::{current_exe, healthz, print_check, print_info};

use crate::config::CodingAgent;

#[cfg(test)]
pub(super) use crate::sidecar::DEFAULT_URL;

pub(crate) fn install_codex_plugin(gateway_url: &str, plugin_root: &Path) -> Result<(), String> {
    install_codex(gateway_url, &plugin_root.join("hooks").join("hooks.json")).map(|_| ())
}

pub(crate) fn stop_plugin_gateway(agent: CodingAgent) -> Result<(), String> {
    crate::sidecar::stop_owned_sidecar(agent)
}

pub(crate) fn uninstall_codex_plugin(gateway_url: &str, plugin_root: &Path) -> Result<(), String> {
    uninstall_codex(gateway_url, &plugin_root.join("hooks").join("hooks.json")).map(|_| ())
}

pub(crate) fn enable_claude_provider(gateway_url: &str) -> Result<(), String> {
    claude::enable_claude_provider(gateway_url)
}

pub(crate) fn restore_claude_provider(gateway_url: &str) -> Result<(), String> {
    claude::restore_claude_provider(gateway_url)
}

pub(crate) fn doctor_plugin(
    agent: CodingAgent,
    gateway_url: &str,
    plugin_root: &Path,
) -> Result<(), String> {
    if doctor_ok(
        agent,
        gateway_url,
        Some(&plugin_root.join("hooks").join("hooks.json")),
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
            let hooks = codex_hooks_installed(plugin_hooks_path)?;
            ok &= print_check("codex provider alias", provider);
            ok &= print_check("codex hooks", hooks);
            let trust = if hooks {
                codex_hook_trust_report(plugin_hooks_path)?
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
use crate::installer::generated_hooks;
#[cfg(test)]
use crate::sidecar::*;
#[cfg(test)]
use claude::*;
#[cfg(test)]
use codex::*;
#[cfg(test)]
use codex_app_server::*;
#[cfg(test)]
use shared::*;

#[cfg(test)]
#[path = "../../tests/coverage/plugin_host_tests.rs"]
mod tests;
