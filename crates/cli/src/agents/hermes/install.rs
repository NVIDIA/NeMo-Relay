// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::ExitCode;

use crate::agents::CodingAgent;
use crate::error::CliError;
use crate::installation::marketplace::HostPluginReadiness;
use crate::installation::marketplace::host::{
    CommandRunner, RealCommandRunner, require_host_cli, require_relay, validate_host_version,
    validate_relay_hook_forward, validate_relay_mcp,
};
use crate::installation::marketplace::state::PluginInstallOptions;
use crate::installation::{InstallRequest, UninstallRequest};

pub(crate) fn install(command: InstallRequest) -> Result<ExitCode, CliError> {
    let options = options(command.dry_run, command.skip_doctor, command.force);
    let runner = RealCommandRunner;
    require_host_cli(CodingAgent::Hermes, &options, &runner).map_err(CliError::Install)?;
    validate_host_version(CodingAgent::Hermes, &options, &runner).map_err(CliError::Install)?;
    let relay = require_relay(&options, &runner).map_err(CliError::Install)?;
    validate_relay_hook_forward(&relay, &options, &runner).map_err(CliError::Install)?;
    validate_relay_mcp(&relay, &options, &runner).map_err(CliError::Install)?;
    let config = config_path().map_err(CliError::Install)?;
    if options.dry_run {
        println!("configure Hermes MCP and hooks at {}", config.display());
        return Ok(ExitCode::SUCCESS);
    }
    super::install_persistent(&config, &relay)
        .map_err(|error| CliError::Install(error.to_string()))?;
    if !options.skip_doctor {
        super::diagnose_persistent(&config).map_err(CliError::Install)?;
    }
    println!("installed Hermes integration");
    Ok(ExitCode::SUCCESS)
}

pub(crate) fn uninstall(command: UninstallRequest) -> Result<ExitCode, CliError> {
    let config = config_path().map_err(CliError::Install)?;
    if command.dry_run {
        println!(
            "remove Relay-owned Hermes MCP and hooks from {}",
            config.display()
        );
        return Ok(ExitCode::SUCCESS);
    }
    super::uninstall_persistent(&config).map_err(|error| CliError::Install(error.to_string()))?;
    println!("uninstalled Hermes integration");
    Ok(ExitCode::SUCCESS)
}

pub(crate) fn config_path() -> Result<PathBuf, String> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .map(|home| super::user_config_path(&home))
        .ok_or_else(|| "cannot determine home directory (set HOME or USERPROFILE)".into())
}

fn options(dry_run: bool, skip_doctor: bool, force: bool) -> PluginInstallOptions {
    PluginInstallOptions {
        install_dir: PathBuf::new(),
        operation_lock_dir: PathBuf::new(),
        force,
        dry_run,
        skip_doctor,
    }
}

pub(crate) fn doctor(
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<(), String> {
    let report = doctor_json_value(options, runner)?;
    for check in report["readiness_checks"]
        .as_array()
        .expect("Hermes readiness checks are an array")
    {
        println!(
            "{}: {} ({})",
            check["name"].as_str().unwrap_or_default(),
            if check["ok"] == serde_json::json!(true) {
                "ok"
            } else {
                "failed"
            },
            check["details"].as_str().unwrap_or_default()
        );
    }
    (report["ok"] == serde_json::json!(true))
        .then_some(())
        .ok_or_else(|| {
            format!(
                "Hermes integration doctor checks failed; remediation: {}",
                report["remediation"].as_str().unwrap_or_default()
            )
        })
}

pub(crate) fn doctor_json_value(
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<serde_json::Value, String> {
    let config = config_path()?;
    let readiness = collect_readiness(&config, options, runner);
    Ok(serde_json::json!({
        "ok": readiness.ok(),
        "host": readiness.host,
        "remediation": readiness.remediation,
        "config": config,
        "readiness_checks": readiness.checks
    }))
}

pub(crate) fn collect_readiness(
    config: &std::path::Path,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> HostPluginReadiness {
    let mut readiness = HostPluginReadiness {
        host: CodingAgent::Hermes.install_arg().into(),
        remediation: format!(
            "nemo-relay install {} --force",
            CodingAgent::Hermes.install_arg()
        ),
        state_path: config.to_path_buf(),
        marketplace: None,
        plugin: None,
        checks: Vec::new(),
        relay: None,
        host_plugin_registered: None,
        host_marketplace_registered: None,
        plugin_setup: None,
    };

    let host_cli = require_host_cli(CodingAgent::Hermes, options, runner);
    readiness.push(
        "Host CLI",
        host_cli
            .as_ref()
            .map(|_| "hermes is available".into())
            .map_err(Clone::clone),
    );
    let version = validate_host_version(CodingAgent::Hermes, options, runner);
    if version.is_err() {
        readiness.remediation = format!(
            "upgrade to {}, then run `nemo-relay install {} --force`",
            CodingAgent::Hermes.version_requirement(),
            CodingAgent::Hermes.install_arg()
        );
    }
    readiness.push(
        "Hermes Agent version",
        version.map(|_| format!("{} is installed", CodingAgent::Hermes.version_requirement())),
    );

    let relay = super::configured_relay_executable(config);
    readiness.push(
        "Configured Relay binary",
        relay
            .as_ref()
            .map(|path| format!("found at {}", path.display()))
            .map_err(Clone::clone),
    );
    match relay {
        Ok(relay) => {
            readiness.relay = Some(relay.clone());
            readiness.push(
                "Relay hook support",
                validate_relay_hook_forward(&relay, options, runner)
                    .map(|_| "hook-forward is supported".into()),
            );
            readiness.push(
                "Relay MCP support",
                validate_relay_mcp(&relay, options, runner)
                    .map(|_| "native mcp subcommand is supported".into()),
            );
        }
        Err(error) => {
            let unavailable = || format!("cannot verify configured Relay capabilities: {error}");
            readiness.push("Relay hook support", Err(unavailable()));
            readiness.push("Relay MCP support", Err(unavailable()));
        }
    }
    readiness.push(
        "Hermes MCP, hooks, and trust",
        super::diagnose_persistent(config),
    );
    readiness
}
