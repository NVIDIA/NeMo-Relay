// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local marketplace installer for Claude Code and Codex plugins.

mod host;
mod marketplace;
mod setup;
mod state;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde_json::{Value, json};

use crate::config::{InstallCommand, PluginHost, UninstallCommand};
use crate::error::CliError;

use host::{
    CommandRunner, RealCommandRunner, require_host_cli, require_relay,
    run_host_marketplace_registration, run_host_marketplace_removal, run_host_plugin_registration,
    run_host_plugin_removal, validate_relay_plugin_shim,
};
use marketplace::write_plugin_marketplace;
use setup::{
    PluginSetupRunner, RealPluginSetupRunner, run_plugin_doctor, run_plugin_doctor_json,
    run_plugin_setup, run_plugin_uninstall,
};
use state::{
    CanonicalizeOrSelf, HostRegistrationProgress, HostSelectionMode, PluginInstallOptions,
    PluginLayout, PluginState, default_install_dir, mark_plugin_setup_installed, read_state,
    remove_path, state_path, write_state, write_state_for_host,
};

pub(super) const DEFAULT_GATEWAY_URL: &str = "http://127.0.0.1:47632";
pub(super) const MARKETPLACE_NAME: &str = "nemo-relay-local";
pub(super) const PLUGIN_NAME: &str = "nemo-relay-plugin";
pub(super) const RELAY_COMMAND: &str = "nemo-relay";

pub(crate) fn install(command: InstallCommand) -> Result<ExitCode, CliError> {
    let options = PluginInstallOptions {
        install_dir: command
            .install_dir
            .unwrap_or_else(default_install_dir)
            .canonicalize_or_self(),
        force: command.force,
        dry_run: command.dry_run,
        skip_doctor: command.skip_doctor,
        json: false,
    };
    run_for_hosts(
        command.host,
        HostSelectionMode::Install,
        &options,
        |host, options, runner, setup_runner| install_host(host, options, runner, setup_runner),
    )
}

pub(crate) fn uninstall(command: UninstallCommand) -> Result<ExitCode, CliError> {
    let options = PluginInstallOptions {
        install_dir: command
            .install_dir
            .unwrap_or_else(default_install_dir)
            .canonicalize_or_self(),
        force: false,
        dry_run: command.dry_run,
        skip_doctor: true,
        json: false,
    };
    run_for_hosts(
        command.host,
        HostSelectionMode::InstalledState,
        &options,
        |host, options, runner, setup_runner| uninstall_host(host, options, runner, setup_runner),
    )
}

pub(crate) fn doctor(
    host: PluginHost,
    install_dir: Option<PathBuf>,
    json: bool,
) -> Result<ExitCode, CliError> {
    let options = PluginInstallOptions {
        install_dir: install_dir
            .unwrap_or_else(default_install_dir)
            .canonicalize_or_self(),
        force: false,
        dry_run: false,
        skip_doctor: true,
        json,
    };
    if json {
        return doctor_json(host, &options);
    }
    run_for_hosts(
        host,
        HostSelectionMode::InstalledState,
        &options,
        |host, options, runner, setup_runner| doctor_host(host, options, runner, setup_runner),
    )
}

fn run_for_hosts<F>(
    host: PluginHost,
    mode: HostSelectionMode,
    options: &PluginInstallOptions,
    mut action: F,
) -> Result<ExitCode, CliError>
where
    F: FnMut(
        PluginHost,
        &PluginInstallOptions,
        &dyn CommandRunner,
        &dyn PluginSetupRunner,
    ) -> Result<(), String>,
{
    let runner = RealCommandRunner;
    let setup_runner = RealPluginSetupRunner;
    let hosts = select_hosts(host, mode, options, &runner)?;
    if hosts.is_empty() {
        return Err(CliError::Install(match host {
            PluginHost::All => match mode {
                HostSelectionMode::Install => {
                    "no supported Claude Code or Codex host CLI was detected".into()
                }
                HostSelectionMode::InstalledState => {
                    "no installed Claude Code or Codex plugin state was found".into()
                }
            },
            _ => "no supported plugin host selected".into(),
        }));
    }
    for host in hosts {
        action(host, options, &runner, &setup_runner).map_err(CliError::Install)?;
    }
    Ok(ExitCode::SUCCESS)
}

fn doctor_json(host: PluginHost, options: &PluginInstallOptions) -> Result<ExitCode, CliError> {
    let runner = RealCommandRunner;
    let setup_runner = RealPluginSetupRunner;
    let hosts = select_hosts(host, HostSelectionMode::InstalledState, options, &runner)?;
    if hosts.is_empty() {
        return Err(CliError::Install(match host {
            PluginHost::All => "no installed Claude Code or Codex plugin state was found".into(),
            _ => "no supported plugin host selected".into(),
        }));
    }
    let reports = hosts
        .into_iter()
        .map(|host| doctor_host_json_value(host, options, &runner, &setup_runner))
        .collect::<Result<Vec<_>, _>>()
        .map_err(CliError::Install)?;
    if matches!(host, PluginHost::All) {
        print_json(&json!({
            "schema_version": 1,
            "plugins": reports
        }))
    } else {
        print_json(&with_schema(
            reports.into_iter().next().expect("hosts is not empty"),
        ))
    }
    .map_err(CliError::Install)?;
    Ok(ExitCode::SUCCESS)
}

fn select_hosts(
    host: PluginHost,
    mode: HostSelectionMode,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<Vec<PluginHost>, CliError> {
    if host != PluginHost::All {
        return Ok(vec![host]);
    }
    let mut hosts = Vec::new();
    for candidate in [PluginHost::Codex, PluginHost::ClaudeCode] {
        let selected = match mode {
            HostSelectionMode::Install => runner
                .resolve_executable(host_cli(candidate))
                .map_err(CliError::Install)?
                .is_some(),
            HostSelectionMode::InstalledState => {
                state_path(candidate, &options.install_dir).exists()
            }
        };
        if selected {
            hosts.push(candidate);
        }
    }
    Ok(hosts)
}

fn install_host(
    host: PluginHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    let relay = require_relay(options, runner)?;
    validate_relay_plugin_shim(&relay, options, runner)?;
    require_host_cli(host, options, runner)?;
    let layout = PluginLayout::new(host, &options.install_dir);
    if options.force {
        force_cleanup_existing_install(host, &layout, options, runner, setup_runner)?;
    }
    write_plugin_marketplace(host, &layout, options)?;
    if let Err(error) = write_state(&layout, options) {
        if let Err(cleanup_error) = remove_path(&layout.marketplace_root, options) {
            return Err(format!(
                "{error}; additionally failed to remove generated marketplace {}: {cleanup_error}",
                layout.marketplace_root.display()
            ));
        }
        return Err(error);
    }
    let mut registration = HostRegistrationProgress::default();
    let mut setup_attempted = false;
    let result = (|| {
        run_host_marketplace_registration(host, &layout, options, runner)?;
        registration.host_marketplace_added = true;
        run_host_plugin_registration(host, options, runner)?;
        registration.host_plugin_added = true;
        setup_attempted = true;
        run_plugin_setup(host, options, setup_runner)?;
        mark_plugin_setup_installed(host, &layout, options)?;
        if !options.skip_doctor {
            run_plugin_doctor(host, options, setup_runner)?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        if let Err(rollback_error) = rollback_install(
            host,
            &layout,
            registration,
            setup_attempted,
            options,
            runner,
            setup_runner,
        ) {
            return Err(format!(
                "{error}; additionally failed to roll back install: {rollback_error}"
            ));
        }
        return Err(error);
    }
    println!(
        "installed {} plugin marketplace at {}",
        host_label(host),
        layout.marketplace_root.display()
    );
    Ok(())
}

fn uninstall_host(
    host: PluginHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    uninstall_host_with_setup_override(host, options, runner, setup_runner, false)
}

fn uninstall_host_with_setup_override(
    host: PluginHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
    force_plugin_setup_uninstall: bool,
) -> Result<(), String> {
    let state = read_state(host, &options.install_dir).unwrap_or_else(|| {
        let layout = PluginLayout::new(host, &options.install_dir);
        PluginState {
            marketplace_root: layout.marketplace_root,
            plugin_root: layout.plugin_root,
            host_plugin_removed: false,
            host_marketplace_removed: false,
            plugin_setup_installed: true,
        }
    });
    if let Err(error) = require_relay(options, runner)
        .and_then(|relay| validate_relay_plugin_shim(&relay, options, runner))
    {
        eprintln!("warning: skipping nemo-relay validation during uninstall: {error}");
    }
    let mut state = state;
    if force_plugin_setup_uninstall && !state.plugin_setup_installed {
        state.plugin_setup_installed = true;
        write_state_for_host(host, &state, &options.install_dir, options)?;
    }
    run_host_unregistration(host, &mut state, &options.install_dir, options, runner)?;
    if force_plugin_setup_uninstall || state.plugin_setup_installed {
        run_plugin_uninstall(host, options, setup_runner)?;
        state.plugin_setup_installed = false;
        write_state_for_host(host, &state, &options.install_dir, options)?;
    }
    remove_path(&state.marketplace_root, options)?;
    remove_path(&state_path(host, &options.install_dir), options)?;
    println!("uninstalled {} plugin", host_label(host));
    Ok(())
}

fn doctor_host(
    host: PluginHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    let relay = require_relay(options, runner)?;
    validate_relay_plugin_shim(&relay, options, runner)?;
    let state = read_state(host, &options.install_dir)
        .ok_or_else(|| format!("no installed {} plugin state found", host_label(host)))?;
    if options.json {
        print_json(&with_schema(doctor_host_json_value(
            host,
            options,
            runner,
            setup_runner,
        )?))?;
        return Ok(());
    }
    println!("nemo-relay: {}", relay.display());
    println!("host: {}", host_arg(host));
    println!("marketplace: {}", state.marketplace_root.display());
    println!("plugin: {}", state.plugin_root.display());
    run_plugin_doctor(host, options, setup_runner)
}

fn doctor_host_json_value(
    host: PluginHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<Value, String> {
    let relay = require_relay(options, runner)?;
    validate_relay_plugin_shim(&relay, options, runner)?;
    let state = read_state(host, &options.install_dir)
        .ok_or_else(|| format!("no installed {} plugin state found", host_label(host)))?;
    let plugin = run_plugin_doctor_json(host, setup_runner)?;
    Ok(json!({
        "host": host_arg(host),
        "nemo_relay": relay,
        "marketplace": state.marketplace_root,
        "plugin": state.plugin_root,
        "checks": plugin
    }))
}

fn force_cleanup_existing_install(
    host: PluginHost,
    layout: &PluginLayout,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    if layout.state_path.exists() {
        uninstall_host(host, options, runner, setup_runner)?;
    } else {
        let mut state = PluginState {
            marketplace_root: layout.marketplace_root.clone(),
            plugin_root: layout.plugin_root.clone(),
            host_plugin_removed: false,
            host_marketplace_removed: false,
            plugin_setup_installed: false,
        };
        run_host_unregistration(host, &mut state, &options.install_dir, options, runner)?;
        remove_path(&layout.marketplace_root, options)?;
        remove_path(&layout.state_path, options)?;
    }
    Ok(())
}

fn rollback_install(
    host: PluginHost,
    layout: &PluginLayout,
    registration: HostRegistrationProgress,
    setup_attempted: bool,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    if setup_attempted {
        return uninstall_host_with_setup_override(host, options, runner, setup_runner, true);
    }
    let mut state = read_state(host, &options.install_dir).unwrap_or_else(|| PluginState {
        marketplace_root: layout.marketplace_root.clone(),
        plugin_root: layout.plugin_root.clone(),
        host_plugin_removed: false,
        host_marketplace_removed: false,
        plugin_setup_installed: false,
    });
    if registration.any_added() {
        state.host_plugin_removed |= !registration.host_plugin_added;
        state.host_marketplace_removed |= !registration.host_marketplace_added;
        write_state_for_host(host, &state, &options.install_dir, options)?;
        run_host_unregistration(host, &mut state, &options.install_dir, options, runner)?;
    }
    remove_path(&layout.marketplace_root, options)?;
    remove_path(&layout.state_path, options)
}

fn run_host_unregistration(
    host: PluginHost,
    state: &mut PluginState,
    install_dir: &Path,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<(), String> {
    if !state.host_plugin_removed {
        require_host_cli(host, options, runner)?;
        run_host_plugin_removal(host, options, runner)?;
        state.host_plugin_removed = true;
        write_state_for_host(host, state, install_dir, options)?;
    }
    if !state.host_marketplace_removed {
        require_host_cli(host, options, runner)?;
        run_host_marketplace_removal(host, options, runner)?;
        state.host_marketplace_removed = true;
        write_state_for_host(host, state, install_dir, options)?;
    }
    Ok(())
}

fn host_arg(host: PluginHost) -> &'static str {
    match host {
        PluginHost::Codex => "codex",
        PluginHost::ClaudeCode => "claude-code",
        PluginHost::All => "all",
    }
}

fn host_label(host: PluginHost) -> &'static str {
    match host {
        PluginHost::Codex => "Codex",
        PluginHost::ClaudeCode => "Claude Code",
        PluginHost::All => "all",
    }
}

fn host_cli(host: PluginHost) -> &'static str {
    match host {
        PluginHost::Codex => "codex",
        PluginHost::ClaudeCode => "claude",
        PluginHost::All => unreachable!("all is expanded before host CLI resolution"),
    }
}

fn print_json(value: &Value) -> Result<(), String> {
    let rendered = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    println!("{rendered}");
    Ok(())
}

fn with_schema(mut value: Value) -> Value {
    if let Some(object) = value.as_object_mut() {
        object.insert("schema_version".into(), json!(1));
    }
    value
}

#[cfg(test)]
use marketplace::*;
#[cfg(test)]
use state::*;

#[cfg(test)]
#[path = "../../tests/coverage/plugin_install_tests.rs"]
mod tests;
