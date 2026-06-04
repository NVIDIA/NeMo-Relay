// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Platform-neutral launcher and hook shim for packaged coding-agent plugins.

mod claude;
mod codex;
mod command;
mod shared;

pub(crate) use command::PluginShimCommand;

use std::env;
use std::io::{Read, Write};
use std::process::{Command, ExitCode};

use claude::{claude_provider, claude_settings_base_url};
use codex::{codex_hooks_installed, codex_provider_installed, install_codex, uninstall_codex};
use command::{
    PluginShimDoctorCommand, PluginShimInstallCommand, PluginShimProviderAction,
    PluginShimProviderCommand, PluginShimSubcommand, PluginShimUninstallCommand,
};
use shared::{
    ExecOrStatus, current_exe, fail_closed, gateway_url, healthz, plugin_idle_timeout, post_hook,
    print_check, print_info, relay_binary,
};

use crate::config::CodingAgent;
use crate::error::CliError;

pub(super) const DEFAULT_BIND: &str = "127.0.0.1:47632";
pub(super) const DEFAULT_URL: &str = "http://127.0.0.1:47632";
pub(super) const HEALTHZ_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);
pub(super) const STALE_LOCK_AFTER: std::time::Duration = std::time::Duration::from_secs(10);

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
    command.env("NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS", plugin_idle_timeout());
    command
        .exec_or_status()
        .map_err(|error| format!("failed to start nemo-relay sidecar: {error}"))
}

fn hook(agent: CodingAgent, explicit_gateway_url: Option<&str>) -> Result<ExitCode, String> {
    let url = gateway_url(agent, explicit_gateway_url);
    let mut payload = Vec::new();
    std::io::stdin()
        .read_to_end(&mut payload)
        .map_err(|error| format!("failed to read hook payload: {error}"))?;
    if payload.iter().all(u8::is_ascii_whitespace) {
        payload = b"{}".to_vec();
    }
    shared::ensure_sidecar(agent, &url);
    match post_hook(agent, &url, &payload) {
        Ok(body) => {
            if !body.is_empty() {
                std::io::stdout()
                    .write_all(&body)
                    .map_err(|error| format!("failed to write hook response: {error}"))?;
            }
            Ok(ExitCode::SUCCESS)
        }
        Err(error) if fail_closed() => Err(error),
        Err(error) => {
            eprintln!("{error}");
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn install(command: PluginShimInstallCommand) -> Result<ExitCode, String> {
    match command.agent {
        CodingAgent::Codex => install_codex(&command.gateway_url),
        other => Err(format!(
            "plugin install supports codex, got {}",
            other.as_arg()
        )),
    }
}

fn uninstall(command: PluginShimUninstallCommand) -> Result<ExitCode, String> {
    match command.agent {
        CodingAgent::Codex => uninstall_codex(&command.gateway_url),
        other => Err(format!(
            "plugin uninstall supports codex, got {}",
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
    Ok(if doctor_ok(command.agent, &command.gateway_url)? {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

pub(crate) fn install_codex_plugin(gateway_url: &str) -> Result<(), String> {
    install_codex(gateway_url).map(|_| ())
}

pub(crate) fn uninstall_codex_plugin(gateway_url: &str) -> Result<(), String> {
    uninstall_codex(gateway_url).map(|_| ())
}

pub(crate) fn enable_claude_provider(gateway_url: &str) -> Result<(), String> {
    claude_provider(PluginShimProviderAction::Enable, gateway_url).map(|_| ())
}

pub(crate) fn restore_claude_provider(gateway_url: &str) -> Result<(), String> {
    claude_provider(PluginShimProviderAction::Restore, gateway_url).map(|_| ())
}

pub(crate) fn doctor_plugin(agent: CodingAgent, gateway_url: &str) -> Result<(), String> {
    if doctor_ok(agent, gateway_url)? {
        Ok(())
    } else {
        Err(format!("{} plugin doctor checks failed", agent.as_arg()))
    }
}

fn doctor_ok(agent: CodingAgent, gateway_url: &str) -> Result<bool, String> {
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
            "not running; hooks start it lazily on first use",
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
            ok &= print_check(
                "codex provider alias",
                codex_provider_installed(gateway_url),
            );
            ok &= print_check("codex hooks", codex_hooks_installed(gateway_url)?);
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
use claude::*;
#[cfg(test)]
use codex::*;
#[cfg(test)]
use shared::*;

#[cfg(test)]
#[path = "../../tests/coverage/plugin_shim_tests.rs"]
mod tests;
