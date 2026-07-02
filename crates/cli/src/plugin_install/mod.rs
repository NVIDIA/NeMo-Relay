// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local marketplace installer for Claude Code and Codex plugins.

mod host;
mod marketplace;
mod setup;
mod state;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use serde::Serialize;
use serde_json::{Value, json};

use crate::config::{InstallCommand, PluginHost, UninstallCommand};
use crate::error::CliError;

use host::{
    CommandRunner, RealCommandRunner, host_registration_report, require_host_cli, require_relay,
    run_host_marketplace_registration, run_host_marketplace_removal, run_host_plugin_registration,
    run_host_plugin_removal, validate_relay_plugin_shim,
};
use marketplace::{marketplace_manifest, plugin_manifest, write_plugin_marketplace};
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
const DEFAULT_HOST_PLUGIN_READINESS_TIMEOUT: Duration = Duration::from_secs(5);

/// One non-mutating readiness check for an installed coding-agent plugin.
///
/// This is deliberately independent from the CLI doctor's status type so the installer can
/// expose its checks to both the focused and top-level doctor paths without coupling their
/// rendering concerns.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct HostPluginReadinessCheck {
    pub(crate) name: String,
    pub(crate) ok: bool,
    pub(crate) details: String,
}

/// Readiness state for one persisted host-plugin installation.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct HostPluginReadiness {
    pub(crate) host: String,
    pub(crate) remediation: String,
    pub(crate) state_path: PathBuf,
    pub(crate) marketplace: Option<PathBuf>,
    pub(crate) plugin: Option<PathBuf>,
    pub(crate) checks: Vec<HostPluginReadinessCheck>,
    #[serde(skip_serializing)]
    pub(crate) relay: Option<PathBuf>,
    #[serde(skip_serializing)]
    pub(crate) host_plugin_registered: Option<bool>,
    #[serde(skip_serializing)]
    pub(crate) host_marketplace_registered: Option<bool>,
    #[serde(skip_serializing)]
    pub(crate) plugin_setup: Option<Value>,
}

impl HostPluginReadiness {
    pub(crate) fn ok(&self) -> bool {
        self.checks.iter().all(|check| check.ok)
    }

    fn push(&mut self, name: impl Into<String>, result: Result<String, String>) {
        match result {
            Ok(details) => self.checks.push(HostPluginReadinessCheck {
                name: name.into(),
                ok: true,
                details,
            }),
            Err(details) => self.checks.push(HostPluginReadinessCheck {
                name: name.into(),
                ok: false,
                details,
            }),
        }
    }
}

struct PendingHostPluginReadiness {
    host: PluginHost,
    state_path: PathBuf,
    receiver: Receiver<HostPluginReadiness>,
}

/// Collects default-location host-plugin readiness without printing or mutating state.
///
/// Only hosts with a persisted install-state record are included. This keeps ordinary
/// transparent-run users from failing the top-level doctor merely because they have not opted
/// into the persistent host-plugin workflow.
pub(crate) fn collect_default_host_plugin_readiness() -> Vec<HostPluginReadiness> {
    let install_dir = default_install_dir().canonicalize_or_self();
    let pending = [PluginHost::Codex, PluginHost::ClaudeCode]
        .into_iter()
        .filter(|host| state_path(*host, &install_dir).exists())
        .map(|host| spawn_default_host_plugin_readiness(host, install_dir.clone()))
        .collect::<Vec<_>>();
    pending
        .into_iter()
        .map(|pending| {
            receive_host_plugin_readiness(pending, DEFAULT_HOST_PLUGIN_READINESS_TIMEOUT)
        })
        .collect()
}

fn spawn_default_host_plugin_readiness(
    host: PluginHost,
    install_dir: PathBuf,
) -> PendingHostPluginReadiness {
    let state_path = state_path(host, &install_dir);
    let (sender, receiver) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let options = PluginInstallOptions {
            install_dir,
            force: false,
            dry_run: false,
            skip_doctor: true,
        };
        let runner = RealCommandRunner;
        let setup_runner = RealPluginSetupRunner;
        let readiness = collect_host_plugin_readiness(host, &options, &runner, &setup_runner);
        let _ = sender.send(readiness);
    });
    PendingHostPluginReadiness {
        host,
        state_path,
        receiver,
    }
}

fn receive_host_plugin_readiness(
    pending: PendingHostPluginReadiness,
    timeout: Duration,
) -> HostPluginReadiness {
    match pending.receiver.recv_timeout(timeout) {
        Ok(readiness) => readiness,
        Err(mpsc::RecvTimeoutError::Timeout) => failed_host_plugin_readiness(
            pending.host,
            pending.state_path,
            "timed out while collecting host-plugin readiness",
        ),
        Err(mpsc::RecvTimeoutError::Disconnected) => failed_host_plugin_readiness(
            pending.host,
            pending.state_path,
            "host-plugin readiness collector stopped unexpectedly",
        ),
    }
}

fn failed_host_plugin_readiness(
    host: PluginHost,
    state_path: PathBuf,
    details: impl Into<String>,
) -> HostPluginReadiness {
    let layout = PluginLayout::new(host, state_path.parent().unwrap_or_else(|| Path::new(".")));
    let mut readiness = HostPluginReadiness {
        host: host_arg(host).to_string(),
        remediation: format!("nemo-relay install {} --force", host_arg(host)),
        state_path,
        marketplace: Some(layout.marketplace_root),
        plugin: Some(layout.plugin_root),
        checks: Vec::new(),
        relay: None,
        host_plugin_registered: None,
        host_marketplace_registered: None,
        plugin_setup: None,
    };
    readiness.push("Host readiness", Err(details.into()));
    readiness
}

pub(crate) fn install(command: InstallCommand) -> Result<ExitCode, CliError> {
    let options = PluginInstallOptions {
        install_dir: command
            .install_dir
            .unwrap_or_else(default_install_dir)
            .canonicalize_or_self(),
        force: command.force,
        dry_run: command.dry_run,
        skip_doctor: command.skip_doctor,
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
    let ready = reports
        .iter()
        .all(|report| report.get("ok").and_then(Value::as_bool) == Some(true));
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
    Ok(if ready {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
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
    let readiness = collect_host_plugin_readiness(host, options, runner, setup_runner);
    println!("host: {}", readiness.host);
    println!("state: {}", readiness.state_path.display());
    if let Some(path) = &readiness.marketplace {
        println!("marketplace: {}", path.display());
    }
    if let Some(path) = &readiness.plugin {
        println!("plugin: {}", path.display());
    }
    for check in &readiness.checks {
        let marker = if check.ok { "ok" } else { "failed" };
        println!("{}: {marker} ({})", check.name, check.details);
    }
    readiness.ok().then_some(()).ok_or_else(|| {
        format!(
            "{} plugin doctor checks failed; run `nemo-relay install {} --force` to repair the installation",
            host_label(host),
            host_arg(host)
        )
    })
}

fn doctor_host_json_value(
    host: PluginHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<Value, String> {
    let readiness = collect_host_plugin_readiness(host, options, runner, setup_runner);
    let host_registration_ok = readiness.host_plugin_registered == Some(true)
        && readiness.host_marketplace_registered == Some(true);
    Ok(json!({
        "ok": readiness.ok(),
        "host": readiness.host,
        "remediation": readiness.remediation,
        "nemo_relay": readiness.relay,
        "marketplace": readiness.marketplace,
        "plugin": readiness.plugin,
        "host_registration": {
            "ok": host_registration_ok,
            "host_plugin_registered": readiness.host_plugin_registered,
            "host_marketplace_registered": readiness.host_marketplace_registered
        },
        "checks": readiness.plugin_setup,
        "state_path": readiness.state_path,
        "readiness_checks": readiness.checks
    }))
}

fn collect_host_plugin_readiness(
    host: PluginHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> HostPluginReadiness {
    let state_path = state_path(host, &options.install_dir);
    let state = read_state(host, &options.install_dir);
    let layout = PluginLayout::new(host, &options.install_dir);
    let marketplace = state
        .as_ref()
        .map(|state| state.marketplace_root.clone())
        .or_else(|| state_path.exists().then(|| layout.marketplace_root.clone()));
    let plugin = state
        .as_ref()
        .map(|state| state.plugin_root.clone())
        .or_else(|| state_path.exists().then(|| layout.plugin_root.clone()));
    let mut readiness = HostPluginReadiness {
        host: host_arg(host).to_string(),
        remediation: format!("nemo-relay install {} --force", host_arg(host)),
        state_path: state_path.clone(),
        marketplace,
        plugin,
        checks: Vec::new(),
        relay: None,
        host_plugin_registered: None,
        host_marketplace_registered: None,
        plugin_setup: None,
    };

    readiness.push(
        "Install state",
        state
            .as_ref()
            .map(|_| format!("valid state at {}", state_path.display()))
            .ok_or_else(|| format!("missing or invalid state at {}", state_path.display())),
    );
    if let Some(marketplace) = readiness.marketplace.as_ref() {
        let manifest = marketplace_manifest_path(host, marketplace);
        readiness.push(
            "Generated marketplace",
            generated_manifest_check(&manifest, &marketplace_manifest(host), "marketplace"),
        );
    }
    if let Some(plugin) = readiness.plugin.as_ref() {
        let manifest = plugin_manifest_path(host, plugin);
        readiness.push(
            "Generated plugin",
            generated_manifest_check(&manifest, &plugin_manifest(host), "plugin"),
        );
    }

    let relay = require_relay(options, runner);
    readiness.push(
        "Relay binary",
        relay
            .as_ref()
            .map(|path| format!("found at {}", path.display()))
            .map_err(Clone::clone),
    );
    if let Ok(relay) = relay {
        readiness.relay = Some(relay.clone());
        readiness.push(
            "Relay hook support",
            validate_relay_plugin_shim(&relay, options, runner)
                .map(|_| "plugin-shim hook is supported".into()),
        );
    }

    let host_cli_check = require_host_cli(host, options, runner);
    readiness.push(
        "Host CLI",
        host_cli_check
            .as_ref()
            .map(|_| format!("{} is available", host_cli(host)))
            .map_err(Clone::clone),
    );
    if host_cli_check.is_ok() {
        match host_registration_report(host, options, runner) {
            Ok(report) => {
                readiness.host_plugin_registered = Some(report.host_plugin_registered);
                readiness.host_marketplace_registered = Some(report.host_marketplace_registered);
                readiness.push(
                    "Host registration",
                    report
                        .ok()
                        .then_some("plugin and marketplace registered".into())
                        .ok_or_else(|| "plugin or marketplace registration is incomplete".into()),
                );
                readiness.push(
                    "Host plugin registration",
                    report
                        .host_plugin_registered
                        .then_some("registered".into())
                        .ok_or_else(|| "nemo-relay host plugin is not registered".into()),
                );
                readiness.push(
                    "Host marketplace registration",
                    report
                        .host_marketplace_registered
                        .then_some("registered".into())
                        .ok_or_else(|| "nemo-relay marketplace is not registered".into()),
                );
            }
            Err(error) => readiness.push("Host registration", Err(error)),
        }
    }

    match run_plugin_doctor_json(host, setup_runner) {
        Ok(plugin_report) => {
            append_plugin_setup_checks(&mut readiness, &plugin_report);
            readiness.plugin_setup = Some(plugin_report);
        }
        Err(error) => readiness.push("Host setup", Err(error)),
    }
    readiness
}

fn append_plugin_setup_checks(readiness: &mut HostPluginReadiness, report: &Value) {
    if let Some(health) = report.get("sidecar_health").and_then(Value::as_str) {
        readiness.push("Sidecar health", Ok(health.to_string()));
    }
    if let Some(checks) = report.get("checks").and_then(Value::as_object) {
        for (name, value) in checks {
            if name == "sidecar_running" {
                continue;
            }
            let details = name.replace('_', " ");
            readiness.push(
                details,
                value
                    .as_bool()
                    .filter(|ok| *ok)
                    .map(|_| "configured".into())
                    .ok_or_else(|| "not configured".into()),
            );
        }
    }
}

fn without_version(mut value: Value) -> Value {
    if let Some(object) = value.as_object_mut() {
        object.remove("version");
    }
    value
}

fn generated_manifest_check(path: &Path, expected: &Value, label: &str) -> Result<String, String> {
    let raw = std::fs::read_to_string(path).map_err(|error| {
        format!(
            "missing or unreadable {label} manifest {}: {error}",
            path.display()
        )
    })?;
    let actual = serde_json::from_str::<Value>(&raw)
        .map_err(|error| format!("invalid {label} manifest {}: {error}", path.display()))?;
    if without_version(actual) == without_version(expected.clone()) {
        Ok(format!("valid at {}", path.display()))
    } else {
        Err(format!(
            "unexpected {label} manifest contents at {}",
            path.display()
        ))
    }
}

fn marketplace_manifest_path(host: PluginHost, root: &Path) -> PathBuf {
    match host {
        PluginHost::Codex => root
            .join(".agents")
            .join("plugins")
            .join("marketplace.json"),
        PluginHost::ClaudeCode => root.join(".claude-plugin").join("marketplace.json"),
        PluginHost::All => unreachable!("all is expanded before layout resolution"),
    }
}

fn plugin_manifest_path(host: PluginHost, root: &Path) -> PathBuf {
    match host {
        PluginHost::Codex => root.join(".codex-plugin").join("plugin.json"),
        PluginHost::ClaudeCode => root.join(".claude-plugin").join("plugin.json"),
        PluginHost::All => unreachable!("all is expanded before layout resolution"),
    }
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
use setup::setup_action_description;
#[cfg(test)]
use state::*;

#[cfg(test)]
#[path = "../../tests/coverage/plugin_install_tests.rs"]
mod tests;
