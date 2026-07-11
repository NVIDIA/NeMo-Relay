// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Platform-neutral launcher and hook shim for packaged coding-agent plugins.

mod claude;
mod codex;
mod codex_app_server;
mod command;
mod shared;

pub(crate) use claude::{ClaudeSetupSnapshot, restore_claude_setup, snapshot_claude_setup};
pub(crate) use codex::{CodexSetupSnapshot, restore_codex_setup, snapshot_codex_setup};
pub(crate) use shared::portable_executable_path;
pub(crate) use shared::shell_quote_for_platform;
#[cfg(test)]
pub(crate) use shared::strip_windows_verbatim_prefix;

pub(crate) use command::PluginShimCommand;

use std::env;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use serde_json::{Value, json};

use claude::{claude_provider, claude_settings_base_url};
use codex::{codex_hook_trust_report, empty_codex_hook_trust_report};
use codex::{codex_hooks_installed, codex_provider_installed, install_codex, uninstall_codex};
use command::{
    PluginShimDoctorCommand, PluginShimInstallCommand, PluginShimProviderAction,
    PluginShimProviderCommand, PluginShimSubcommand, PluginShimUninstallCommand,
};
#[cfg(test)]
use shared::MAX_HOOK_RESPONSE_BYTES;
use shared::{
    ExecOrStatus, HookForwardError, current_exe, fail_closed, gateway_url, healthz, home_dir,
    plugin_idle_timeout, post_hook, print_check, print_info, relay_binary,
};

use crate::config::{CodingAgent, ServerArgs};
use crate::error::CliError;

pub(super) use crate::sidecar::{DEFAULT_BIND, DEFAULT_URL};
const DEFAULT_HOOK_STDIN_BYTES: usize = crate::config::DEFAULT_MAX_HOOK_PAYLOAD_BYTES;

pub(crate) fn run(command: PluginShimCommand) -> Result<ExitCode, CliError> {
    match command.command {
        PluginShimSubcommand::Serve(command) => serve(command.args),
        PluginShimSubcommand::Hook(command) => hook(command.agent, command.gateway_url.as_deref()),
        PluginShimSubcommand::Install(command) => install(command),
        PluginShimSubcommand::Uninstall(command) => uninstall(command),
        PluginShimSubcommand::Provider(command) => provider(command),
        PluginShimSubcommand::Doctor(command) => doctor(command),
    }
    .map_err(CliError::Install)
}

fn serve(args: Vec<String>) -> Result<ExitCode, String> {
    let relay = relay_binary()?;
    let bind = env::var("NEMO_RELAY_PLUGIN_BIND").unwrap_or_else(|_| DEFAULT_BIND.into());
    let mut command = Command::new(relay);
    command.arg("--bind").arg(bind).args(args);
    command.env(
        "NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS",
        plugin_idle_timeout()?.as_secs().to_string(),
    );
    command
        .exec_or_status()
        .map_err(|error| format!("failed to start nemo-relay sidecar: {error}"))
}

fn hook(agent: CodingAgent, explicit_gateway_url: Option<&str>) -> Result<ExitCode, String> {
    let url = gateway_url(agent, explicit_gateway_url);
    let plugin_launch = crate::sidecar::loopback_bind(&url).and_then(|bind| {
        crate::sidecar::resolve_plugin_gateway(agent, &ServerArgs::default(), bind)
            .map_err(|error| error.to_string())
    });
    let max_hook_payload_bytes = plugin_launch
        .as_ref()
        .map_or(DEFAULT_HOOK_STDIN_BYTES, |launch| {
            launch.max_hook_payload_bytes
        });
    let mut input = std::io::stdin();
    let mut output = std::io::stdout();
    hook_with_io(
        HookInvocation {
            agent,
            gateway_url: Some(&url),
            max_payload_bytes: max_hook_payload_bytes,
            preflight_gateway: matches!(agent, CodingAgent::Codex),
        },
        &mut input,
        &mut output,
        |_agent, _url| match plugin_launch.as_ref() {
            Ok(launch) => launch.gateway.ensure().map(|_| ()),
            Err(error) => Err(error.clone()),
        },
        post_hook,
        fail_closed,
    )
}

#[derive(Clone, Copy)]
struct HookInvocation<'a> {
    agent: CodingAgent,
    gateway_url: Option<&'a str>,
    max_payload_bytes: usize,
    preflight_gateway: bool,
}

fn hook_with_io<R, W, E, P, F>(
    invocation: HookInvocation<'_>,
    input: &mut R,
    output: &mut W,
    mut ensure_sidecar: E,
    mut post_hook: P,
    fail_closed: F,
) -> Result<ExitCode, String>
where
    R: Read,
    W: Write,
    E: FnMut(CodingAgent, &str) -> Result<(), String>,
    P: FnMut(CodingAgent, &str, &[u8]) -> Result<Vec<u8>, HookForwardError>,
    F: FnOnce() -> bool,
{
    let agent = invocation.agent;
    let url = gateway_url(agent, invocation.gateway_url);
    let fail_closed = fail_closed();
    let mut payload = Vec::new();
    if let Err(error) = read_hook_payload(input, &mut payload, invocation.max_payload_bytes) {
        if fail_closed {
            return Err(error);
        }
        eprintln!("{error}");
        return Ok(ExitCode::SUCCESS);
    }
    if payload.iter().all(u8::is_ascii_whitespace) {
        payload = b"{}".to_vec();
    }
    let preflight = if invocation.preflight_gateway {
        ensure_sidecar(agent, &url)
    } else {
        Ok(())
    };
    let forwarded = match preflight {
        Err(error) => Err(HookForwardError::not_retryable(format!(
            "gateway identity preflight failed: {error}"
        ))),
        Ok(()) => match post_hook(agent, &url, &payload) {
            Err(error) if error.is_retryable() => match ensure_sidecar(agent, &url) {
                Ok(()) => post_hook(agent, &url, &payload),
                Err(start_error) => Err(HookForwardError::not_retryable(format!(
                    "{error}; sidecar bootstrap failed: {start_error}"
                ))),
            },
            result => result,
        },
    };
    match forwarded {
        Ok(body) => {
            if !body.is_empty() {
                output
                    .write_all(&body)
                    .map_err(|error| format!("failed to write hook response: {error}"))?;
            }
            Ok(ExitCode::SUCCESS)
        }
        Err(error) if fail_closed => Err(error.to_string()),
        Err(error) => {
            eprintln!("{error}");
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn read_hook_payload<R: Read>(
    input: &mut R,
    payload: &mut Vec<u8>,
    limit: usize,
) -> Result<(), String> {
    input
        .take(limit.saturating_add(1) as u64)
        .read_to_end(payload)
        .map_err(|error| format!("failed to read hook payload: {error}"))?;
    if payload.len() > limit {
        return Err(format!("hook payload exceeds the {limit}-byte limit"));
    }
    Ok(())
}

fn install(command: PluginShimInstallCommand) -> Result<ExitCode, String> {
    match command.agent {
        CodingAgent::Codex => install_codex(&command.gateway_url, &plugin_hooks_path_from_env()?),
        CodingAgent::Hermes => {
            let home = home_dir()?;
            let relay = relay_binary()?;
            crate::hermes::install_persistent(&crate::hermes::user_config_path(&home), &relay)
                .map_err(|error| error.to_string())?;
            Ok(ExitCode::SUCCESS)
        }
        other => Err(format!(
            "plugin install supports codex and hermes, got {}",
            other.as_arg()
        )),
    }
}

fn uninstall(command: PluginShimUninstallCommand) -> Result<ExitCode, String> {
    match command.agent {
        CodingAgent::Codex => uninstall_codex(&command.gateway_url, &plugin_hooks_path_from_env()?),
        CodingAgent::Hermes => {
            let home = home_dir()?;
            crate::hermes::uninstall_persistent(&crate::hermes::user_config_path(&home))
                .map_err(|error| error.to_string())?;
            Ok(ExitCode::SUCCESS)
        }
        other => Err(format!(
            "plugin uninstall supports codex and hermes, got {}",
            other.as_arg()
        )),
    }
}

fn provider(command: PluginShimProviderCommand) -> Result<ExitCode, String> {
    match command.agent {
        CodingAgent::ClaudeCode => claude_provider(command.action, &command.gateway_url),
        other => Err(format!(
            "plugin provider supports claude, got {}",
            other.as_arg()
        )),
    }
}

fn doctor(command: PluginShimDoctorCommand) -> Result<ExitCode, String> {
    let plugin_hooks = matches!(command.agent, CodingAgent::Codex)
        .then(plugin_hooks_path_from_env)
        .transpose()?;
    Ok(
        if doctor_ok(command.agent, &command.gateway_url, plugin_hooks.as_deref())? {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        },
    )
}

pub(crate) fn install_codex_plugin(gateway_url: &str, plugin_root: &Path) -> Result<(), String> {
    install_codex(gateway_url, &plugin_root.join("hooks").join("hooks.json")).map(|_| ())
}

pub(crate) fn stop_plugin_gateway(agent: CodingAgent) -> Result<(), String> {
    crate::sidecar::stop_owned_sidecar(agent)
}

pub(crate) fn codex_plugin_hook_command(relay: &std::path::Path) -> String {
    codex::codex_plugin_hook_command(relay)
}

pub(crate) fn uninstall_codex_plugin(gateway_url: &str, plugin_root: &Path) -> Result<(), String> {
    uninstall_codex(gateway_url, &plugin_root.join("hooks").join("hooks.json")).map(|_| ())
}

pub(crate) fn enable_claude_provider(gateway_url: &str) -> Result<(), String> {
    claude_provider(PluginShimProviderAction::Enable, gateway_url).map(|_| ())
}

pub(crate) fn restore_claude_provider(gateway_url: &str) -> Result<(), String> {
    claude_provider(PluginShimProviderAction::Restore, gateway_url).map(|_| ())
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
        } else if matches!(agent, CodingAgent::Codex) {
            "not_running_mcp_start"
        } else {
            "not_running_lazy_start"
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
    } else if matches!(agent, CodingAgent::Codex) {
        print_info(
            "sidecar health",
            "not running; the required plugin MCP starts it before the captured turn",
        );
    } else {
        print_info(
            "sidecar health",
            "not running; the plugin MCP or first hook starts it lazily",
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

fn plugin_hooks_path_from_env() -> Result<PathBuf, String> {
    env::var_os("PLUGIN_ROOT")
        .map(PathBuf::from)
        .map(|root| root.join("hooks").join("hooks.json"))
        .ok_or_else(|| "PLUGIN_ROOT is required for Codex plugin hook setup".into())
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
#[path = "../../tests/coverage/plugin_shim_tests.rs"]
mod tests;
