// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local marketplace installer for Claude Code and Codex plugins.

mod host;
mod marketplace;
mod operation_lock;
mod setup;
mod state;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::{Value, json};

use crate::configuration::CodingAgent;
use crate::error::CliError;
use crate::installation::IntegrationHost;
use crate::installation::generation::{
    GENERATION_FILE_NAME, GenerationRetirement, InstallGeneration,
};
use crate::installation::{InstallRequest, UninstallRequest};

use host::{
    CommandRunner, RealCommandRunner, host_registration_report, require_host_cli, require_relay,
    run_host_marketplace_registration, run_host_marketplace_removal, run_host_plugin_registration,
    run_host_plugin_removal, validate_relay_hook_forward, validate_relay_mcp,
};
use marketplace::{
    marketplace_manifest, plugin_hooks, plugin_manifest, plugin_mcp_config,
    write_plugin_marketplace, write_plugin_marketplace_for_generation,
};
use operation_lock::{DEFAULT_OPERATION_LOCK_TIMEOUT, PluginOperationLock};
use setup::{
    PluginSetupRunner, PluginSetupSnapshot, RealPluginSetupRunner, run_plugin_doctor_json,
    run_plugin_doctor_with_generation, run_plugin_setup_with_generation, run_plugin_uninstall,
};
#[cfg(test)]
use setup::{run_plugin_doctor, run_plugin_setup};
use state::{
    CanonicalizeOrSelf, HostRegistrationProgress, HostSelectionMode, PluginInstallOptions,
    PluginLayout, PluginState, default_install_dir, mark_plugin_setup_installed, read_state,
    remove_path, state_path, write_state, write_state_for_host,
};

pub(super) use crate::bootstrap::DEFAULT_URL as DEFAULT_GATEWAY_URL;
pub(super) const MARKETPLACE_NAME: &str = "nemo-relay-local";
pub(super) const PLUGIN_NAME: &str = "nemo-relay-plugin";
pub(super) const RELAY_COMMAND: &str = "nemo-relay";
const DEFAULT_HOST_PLUGIN_READINESS_TIMEOUT: Duration = Duration::from_secs(5);

fn default_operation_lock_dir() -> Result<PathBuf, String> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .map(CanonicalizeOrSelf::canonicalize_or_self)
        .map(|home| home.join(".nemo-relay").join("plugin-operations"))
        .ok_or_else(|| {
            "cannot determine the per-user plugin operation lock directory; set HOME or USERPROFILE"
                .into()
        })
}

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

/// Readiness state for one persistent coding-agent integration.
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
    host: IntegrationHost,
    state_path: PathBuf,
    receiver: Receiver<HostPluginReadiness>,
}

/// Collects default-location persistent-integration readiness without printing or mutating state.
///
/// Only hosts with a persisted install-state record are included. This keeps ordinary
/// transparent-run users from failing the top-level doctor merely because they have not opted
/// into a persistent coding-agent integration.
pub(crate) fn collect_default_host_plugin_readiness() -> Vec<HostPluginReadiness> {
    let install_dir = default_install_dir().canonicalize_or_self();
    let mut hosts = [IntegrationHost::Codex, IntegrationHost::ClaudeCode]
        .into_iter()
        .filter(|host| state_path(*host, &install_dir).exists())
        .collect::<Vec<_>>();
    if hermes_config_path().is_ok_and(|path| crate::agents::hermes::persistent_state_exists(&path))
    {
        hosts.push(IntegrationHost::Hermes);
    }
    let pending = hosts
        .into_iter()
        .map(|host| spawn_default_host_plugin_readiness(host, install_dir.clone()))
        .collect::<Vec<_>>();
    let deadline = Instant::now() + DEFAULT_HOST_PLUGIN_READINESS_TIMEOUT;
    pending
        .into_iter()
        .map(|pending| {
            receive_host_plugin_readiness(
                pending,
                deadline.saturating_duration_since(Instant::now()),
            )
        })
        .collect()
}

fn spawn_default_host_plugin_readiness(
    host: IntegrationHost,
    install_dir: PathBuf,
) -> PendingHostPluginReadiness {
    let state_path = match host {
        IntegrationHost::Hermes => hermes_config_path()
            .unwrap_or_else(|_| state_path(IntegrationHost::Hermes, &install_dir)),
        _ => state_path(host, &install_dir),
    };
    let worker_state_path = state_path.clone();
    let (sender, receiver) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let options = PluginInstallOptions {
            install_dir,
            operation_lock_dir: PathBuf::new(),
            force: false,
            dry_run: false,
            skip_doctor: true,
        };
        let runner = RealCommandRunner;
        let setup_runner = RealPluginSetupRunner;
        let readiness = match host {
            IntegrationHost::Hermes => {
                collect_hermes_host_readiness(&worker_state_path, &options, &runner)
            }
            _ => collect_host_plugin_readiness(host, &options, &runner, &setup_runner),
        };
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
            "timed out while collecting persistent-integration readiness",
        ),
        Err(mpsc::RecvTimeoutError::Disconnected) => failed_host_plugin_readiness(
            pending.host,
            pending.state_path,
            "persistent-integration readiness collector stopped unexpectedly",
        ),
    }
}

fn failed_host_plugin_readiness(
    host: IntegrationHost,
    state_path: PathBuf,
    details: impl Into<String>,
) -> HostPluginReadiness {
    let (marketplace, plugin) = match host {
        IntegrationHost::Hermes => (None, None),
        _ => {
            let layout =
                PluginLayout::new(host, state_path.parent().unwrap_or_else(|| Path::new(".")));
            (Some(layout.marketplace_root), Some(layout.plugin_root))
        }
    };
    let mut readiness = HostPluginReadiness {
        host: host.as_arg().to_string(),
        remediation: format!("nemo-relay install {} --force", host.as_arg()),
        state_path,
        marketplace,
        plugin,
        checks: Vec::new(),
        relay: None,
        host_plugin_registered: None,
        host_marketplace_registered: None,
        plugin_setup: None,
    };
    readiness.push("Host readiness", Err(details.into()));
    readiness
}

pub(crate) fn install(command: InstallRequest) -> Result<ExitCode, CliError> {
    let operation_lock_dir = if command.dry_run {
        PathBuf::new()
    } else {
        default_operation_lock_dir().map_err(CliError::Install)?
    };
    let options = PluginInstallOptions {
        install_dir: command
            .install_dir
            .unwrap_or_else(default_install_dir)
            .canonicalize_or_self(),
        operation_lock_dir,
        force: command.force,
        dry_run: command.dry_run,
        skip_doctor: command.skip_doctor,
    };
    run_for_hosts(
        command.host,
        HostSelectionMode::Install,
        &options,
        |host, options, runner, setup_runner| match host {
            IntegrationHost::Hermes => install_hermes_host(options, runner),
            _ => install_host(host, options, runner, setup_runner),
        },
    )
}

pub(crate) fn uninstall(command: UninstallRequest) -> Result<ExitCode, CliError> {
    let operation_lock_dir = if command.dry_run {
        PathBuf::new()
    } else {
        default_operation_lock_dir().map_err(CliError::Install)?
    };
    let options = PluginInstallOptions {
        install_dir: command
            .install_dir
            .unwrap_or_else(default_install_dir)
            .canonicalize_or_self(),
        operation_lock_dir,
        force: false,
        dry_run: command.dry_run,
        skip_doctor: true,
    };
    run_for_hosts(
        command.host,
        HostSelectionMode::InstalledState,
        &options,
        |host, options, runner, setup_runner| match host {
            IntegrationHost::Hermes => uninstall_hermes_host(options),
            _ => uninstall_host(host, options, runner, setup_runner),
        },
    )
}

pub(crate) fn doctor(
    host: IntegrationHost,
    install_dir: Option<PathBuf>,
    json: bool,
) -> Result<ExitCode, CliError> {
    let options = PluginInstallOptions {
        install_dir: install_dir
            .unwrap_or_else(default_install_dir)
            .canonicalize_or_self(),
        operation_lock_dir: PathBuf::new(),
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
        |host, options, runner, setup_runner| match host {
            IntegrationHost::Hermes => doctor_hermes_host(options, runner),
            _ => doctor_host(host, options, runner, setup_runner),
        },
    )
}

fn run_for_hosts<F>(
    host: IntegrationHost,
    mode: HostSelectionMode,
    options: &PluginInstallOptions,
    mut action: F,
) -> Result<ExitCode, CliError>
where
    F: FnMut(
        IntegrationHost,
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
            IntegrationHost::All => match mode {
                HostSelectionMode::Install => {
                    "no supported Claude Code, Codex, or Hermes host CLI was detected".into()
                }
                HostSelectionMode::InstalledState => {
                    "no installed Claude Code, Codex, or Hermes integration state was found".into()
                }
            },
            _ => "no supported integration host selected".into(),
        }));
    }
    for host in hosts {
        action(host, options, &runner, &setup_runner).map_err(CliError::Install)?;
    }
    Ok(ExitCode::SUCCESS)
}

fn doctor_json(
    host: IntegrationHost,
    options: &PluginInstallOptions,
) -> Result<ExitCode, CliError> {
    let runner = RealCommandRunner;
    let setup_runner = RealPluginSetupRunner;
    let hosts = select_hosts(host, HostSelectionMode::InstalledState, options, &runner)?;
    if hosts.is_empty() {
        return Err(CliError::Install(match host {
            IntegrationHost::All => {
                "no installed Claude Code, Codex, or Hermes integration state was found".into()
            }
            _ => "no supported integration host selected".into(),
        }));
    }
    let reports = hosts
        .into_iter()
        .map(|host| match host {
            IntegrationHost::Hermes => doctor_hermes_json_value(options, &runner),
            _ => doctor_host_json_value(host, options, &runner, &setup_runner),
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(CliError::Install)?;
    let ready = reports
        .iter()
        .all(|report| report.get("ok").and_then(Value::as_bool) == Some(true));
    if matches!(host, IntegrationHost::All) {
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
    host: IntegrationHost,
    mode: HostSelectionMode,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<Vec<IntegrationHost>, CliError> {
    if host != IntegrationHost::All {
        return Ok(vec![host]);
    }
    let mut hosts = Vec::new();
    for candidate in [
        IntegrationHost::Codex,
        IntegrationHost::ClaudeCode,
        IntegrationHost::Hermes,
    ] {
        let selected = match mode {
            HostSelectionMode::Install => runner
                .resolve_executable(candidate.executable().expect("concrete plugin host"))
                .map_err(CliError::Install)?
                .is_some(),
            HostSelectionMode::InstalledState => match candidate {
                IntegrationHost::Hermes => hermes_config_path()
                    .is_ok_and(|path| crate::agents::hermes::persistent_state_exists(&path)),
                _ => state_path(candidate, &options.install_dir).exists(),
            },
        };
        if selected {
            hosts.push(candidate);
        }
    }
    Ok(hosts)
}

fn hermes_config_path() -> Result<PathBuf, String> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .map(|home| crate::agents::hermes::user_config_path(&home))
        .ok_or_else(|| "cannot determine home directory (set HOME or USERPROFILE)".into())
}

fn install_hermes_host(
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<(), String> {
    require_host_cli(IntegrationHost::Hermes, options, runner)?;
    host::validate_host_version(IntegrationHost::Hermes, options, runner)?;
    let relay = require_relay(options, runner)?;
    validate_relay_hook_forward(&relay, options, runner)?;
    validate_relay_mcp(&relay, options, runner)?;
    let config = hermes_config_path()?;
    if options.dry_run {
        println!("configure Hermes MCP and hooks at {}", config.display());
        return Ok(());
    }
    crate::agents::hermes::install_persistent(&config, &relay)
        .map_err(|error| error.to_string())?;
    if !options.skip_doctor {
        crate::agents::hermes::diagnose_persistent(&config)?;
    }
    println!("installed Hermes integration");
    Ok(())
}

fn uninstall_hermes_host(options: &PluginInstallOptions) -> Result<(), String> {
    let config = hermes_config_path()?;
    if options.dry_run {
        println!(
            "remove Relay-owned Hermes MCP and hooks from {}",
            config.display()
        );
        return Ok(());
    }
    crate::agents::hermes::uninstall_persistent(&config).map_err(|error| error.to_string())?;
    println!("uninstalled Hermes integration");
    Ok(())
}

fn doctor_hermes_host(
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<(), String> {
    let report = doctor_hermes_json_value(options, runner)?;
    for check in report["readiness_checks"]
        .as_array()
        .expect("Hermes readiness checks are an array")
    {
        println!(
            "{}: {} ({})",
            check["name"].as_str().unwrap_or_default(),
            if check["ok"] == json!(true) {
                "ok"
            } else {
                "failed"
            },
            check["details"].as_str().unwrap_or_default()
        );
    }
    (report["ok"] == json!(true)).then_some(()).ok_or_else(|| {
        format!(
            "Hermes integration doctor checks failed; remediation: {}",
            report["remediation"].as_str().unwrap_or_default()
        )
    })
}

fn doctor_hermes_json_value(
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<Value, String> {
    let config = hermes_config_path()?;
    let readiness = collect_hermes_host_readiness(&config, options, runner);
    Ok(json!({
        "ok": readiness.ok(),
        "host": readiness.host,
        "remediation": readiness.remediation,
        "config": config,
        "readiness_checks": readiness.checks
    }))
}

fn collect_hermes_host_readiness(
    config: &Path,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> HostPluginReadiness {
    let mut readiness = HostPluginReadiness {
        host: IntegrationHost::Hermes.as_arg().into(),
        remediation: format!(
            "nemo-relay install {} --force",
            IntegrationHost::Hermes.as_arg()
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

    let host_cli = require_host_cli(IntegrationHost::Hermes, options, runner);
    readiness.push(
        "Host CLI",
        host_cli
            .as_ref()
            .map(|_| "hermes is available".into())
            .map_err(Clone::clone),
    );
    let version = host::validate_host_version(IntegrationHost::Hermes, options, runner);
    if version.is_err() {
        readiness.remediation = format!(
            "upgrade to {}, then run `nemo-relay install {} --force`",
            CodingAgent::Hermes.version_requirement(),
            IntegrationHost::Hermes.as_arg()
        );
    }
    readiness.push(
        "Hermes Agent version",
        version.map(|_| format!("{} is installed", CodingAgent::Hermes.version_requirement())),
    );

    let relay = crate::agents::hermes::configured_relay_executable(config);
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
        crate::agents::hermes::diagnose_persistent(config),
    );
    readiness
}

fn install_host(
    host: IntegrationHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    install_host_with_operation_timeout(
        host,
        options,
        runner,
        setup_runner,
        DEFAULT_OPERATION_LOCK_TIMEOUT,
    )
}

fn install_host_with_operation_timeout(
    host: IntegrationHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
    lock_timeout: Duration,
) -> Result<(), String> {
    let _operation_lock = (!options.dry_run)
        .then(|| {
            PluginOperationLock::acquire(
                host,
                &options.operation_lock_dir,
                &options.install_dir,
                lock_timeout,
            )
        })
        .transpose()?;
    install_host_locked(host, options, runner, setup_runner)
}

fn install_host_locked(
    host: IntegrationHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    let relay = require_relay(options, runner)?;
    validate_relay_hook_forward(&relay, options, runner)?;
    validate_relay_mcp(&relay, options, runner)?;
    require_host_cli(host, options, runner)?;
    host::validate_host_version(host, options, runner)?;
    let layout = PluginLayout::new(host, &options.install_dir);
    let plugin_preflight = if !options.dry_run {
        Some(prepare_plugin_install(host, &layout, options, runner)?)
    } else {
        None
    };
    if !options.force
        && plugin_preflight
            .as_ref()
            .is_some_and(|preflight| preflight.previous_install_exists)
    {
        return Err(existing_plugin_install_requires_force_error(host));
    }
    let mut force_snapshot = None;
    let mut replacement_generation_lock = None;
    let staged = if options.force && !options.dry_run {
        let preflight = plugin_preflight.expect("MCP plugin force install has preflight state");
        let initialize_generation_lock = match preflight.generation_retirement.as_ref() {
            Some(retirement) => !retirement.uses_lock_path(&layout.generation_lock)?,
            None => true,
        };
        let staged =
            stage_plugin_marketplace(host, &relay, &layout, initialize_generation_lock, options)?;
        if initialize_generation_lock {
            replacement_generation_lock = match acquire_replacement_generation_lock(
                host,
                &staged.layout.generation_fence,
                &layout.generation_lock,
                staged.generation_lock_created,
            ) {
                Ok(lock) => Some(lock),
                Err(error) => {
                    staged.cleanup();
                    if staged.generation_lock_created {
                        remove_generation_lock_best_effort(&layout.generation_lock);
                    }
                    return Err(error);
                }
            };
        }
        match begin_force_replacement(host, &layout, preflight, options, runner, setup_runner) {
            Ok(mut snapshot) => {
                if let Err(error) = setup_runner.refresh_gateway() {
                    staged.cleanup();
                    return restore_force_replacement_after_error(
                        host,
                        &layout,
                        &mut snapshot,
                        options,
                        runner,
                        setup_runner,
                        error,
                    );
                }
                force_snapshot = Some(snapshot);
                Some(staged)
            }
            Err(error) => {
                staged.cleanup();
                return Err(error);
            }
        }
    } else {
        None
    };
    if options.force && staged.is_none() {
        force_cleanup_existing_install(host, &layout, options, runner, setup_runner)?;
    }
    if let Some(staged) = staged.as_ref() {
        if let Err(error) = staged.promote(&layout) {
            staged.cleanup();
            return restore_force_replacement_after_error(
                host,
                &layout,
                force_snapshot.as_mut().expect("force snapshot exists"),
                options,
                runner,
                setup_runner,
                error,
            );
        }
        force_snapshot
            .as_mut()
            .expect("force snapshot exists")
            .replacement_promoted = true;
        if let Some(lock) = replacement_generation_lock.as_mut()
            && let Err(error) = lock.retarget_promoted_marker(&layout.generation_fence)
        {
            staged.cleanup();
            return restore_force_replacement_after_error(
                host,
                &layout,
                force_snapshot.as_mut().expect("force snapshot exists"),
                options,
                runner,
                setup_runner,
                error,
            );
        }
        staged.cleanup();
    } else {
        let generation_lock_created =
            !options.dry_run && generation_lock_is_absent(&layout.generation_lock);
        if let Err(error) = write_plugin_marketplace(host, &layout, &relay, options) {
            let cleanup_error = (!options.dry_run)
                .then(|| remove_path(&layout.marketplace_root, options).err())
                .flatten();
            if generation_lock_created {
                remove_generation_lock_best_effort(&layout.generation_lock);
            }
            return match cleanup_error {
                Some(cleanup_error) => Err(format!(
                    "{error}; additionally failed to remove the incomplete marketplace: {cleanup_error}"
                )),
                None => Err(error),
            };
        }
        if !options.dry_run {
            replacement_generation_lock = match acquire_replacement_generation_lock(
                host,
                &layout.generation_fence,
                &layout.generation_lock,
                generation_lock_created,
            ) {
                Ok(lock) => Some(lock),
                Err(error) => {
                    let cleanup_error = remove_path(&layout.marketplace_root, options).err();
                    if generation_lock_created {
                        remove_generation_lock_best_effort(&layout.generation_lock);
                    }
                    return match cleanup_error {
                        Some(cleanup_error) => Err(format!(
                            "{error}; additionally failed to remove the incomplete marketplace: {cleanup_error}"
                        )),
                        None => Err(error),
                    };
                }
            };
        }
    }
    if let Err(error) = write_state(&layout, options) {
        let _replacement_retirement = if force_snapshot.is_some() {
            let existing_retirement = replacement_generation_lock
                .as_mut()
                .map(ReplacementGenerationLock::retirement_mut)
                .or_else(|| {
                    force_snapshot
                        .as_mut()
                        .and_then(|snapshot| snapshot.generation_retirement.as_mut())
                });
            match retire_replacement_before_rollback(
                host,
                &layout,
                options,
                setup_runner,
                existing_retirement,
            ) {
                Ok(retirement) => retirement,
                Err(retirement_error) => {
                    return Err(format!(
                        "{error}; refusing destructive rollback because the replacement MCP generation could not be retired: {retirement_error}"
                    ));
                }
            }
        } else {
            None
        };
        let cleanup_error = remove_path(&layout.marketplace_root, options).err();
        let restore_error = force_snapshot.as_mut().and_then(|snapshot| {
            restore_force_replacement(host, &layout, snapshot, options, runner, setup_runner).err()
        });
        let errors = [cleanup_error, restore_error]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if !errors.is_empty() {
            return Err(format!("{error}; additionally {}", errors.join("; ")));
        }
        return Err(error);
    }
    let mut registration = HostRegistrationProgress::default();
    let mut registration_state_uncertain = false;
    let mut setup_installed = false;
    let result = (|| {
        let generation_token = replacement_generation_lock
            .as_ref()
            .map(ReplacementGenerationLock::retirement)
            .or_else(|| {
                force_snapshot
                    .as_ref()
                    .and_then(|snapshot| snapshot.generation_retirement.as_ref())
            })
            .map(GenerationRetirement::active_visible_token)
            .transpose()?;
        if let Err(error) =
            run_host_marketplace_registration(host, &layout.marketplace_root, options, runner)
        {
            registration_state_uncertain = true;
            return Err(error);
        }
        registration.host_marketplace_added = true;
        if let Err(error) = run_host_plugin_registration(host, options, runner) {
            registration_state_uncertain = true;
            return Err(error);
        }
        registration.host_plugin_added = true;
        if !matches!(host, IntegrationHost::Codex) {
            setup_installed = true;
        }
        run_plugin_setup_with_generation(
            host,
            &layout,
            options,
            setup_runner,
            generation_token.as_deref(),
        )?;
        setup_installed = true;
        mark_plugin_setup_installed(host, &layout, options)?;
        if !options.skip_doctor {
            run_plugin_doctor_with_generation(
                host,
                &layout.plugin_root,
                options,
                setup_runner,
                generation_token.as_deref(),
            )?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        if registration_state_uncertain {
            let observed = host_registration_report(host, options, runner).map_err(|report_error| {
                format!(
                    "{error}; refusing destructive rollback because the host registration state could not be verified after a registration command failed: {report_error}"
                )
            })?;
            registration.host_plugin_added |= observed.host_plugin_registered;
            registration.host_marketplace_added |= observed.host_marketplace_registered;
        }
        let replacement_may_be_live = force_snapshot.is_some() || registration.host_plugin_added;
        let _replacement_retirement = if replacement_may_be_live {
            let existing_retirement = replacement_generation_lock
                .as_mut()
                .map(ReplacementGenerationLock::retirement_mut)
                .or_else(|| {
                    force_snapshot
                        .as_mut()
                        .and_then(|snapshot| snapshot.generation_retirement.as_mut())
                });
            match retire_replacement_before_rollback(
                host,
                &layout,
                options,
                setup_runner,
                existing_retirement,
            ) {
                Ok(retirement) => retirement,
                Err(retirement_error) => {
                    return Err(format!(
                        "{error}; refusing destructive rollback because the replacement MCP generation could not be retired: {retirement_error}"
                    ));
                }
            }
        } else {
            None
        };
        let rollback_error = rollback_install(
            host,
            &layout,
            registration,
            setup_installed,
            options,
            runner,
            setup_runner,
        )
        .err();
        let restore_error = force_snapshot.as_mut().and_then(|snapshot| {
            restore_force_replacement(host, &layout, snapshot, options, runner, setup_runner).err()
        });
        let rollback_errors = [
            rollback_error.map(|error| format!("failed to roll back install: {error}")),
            restore_error.map(|error| format!("failed to restore previous install: {error}")),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        if !rollback_errors.is_empty() {
            return Err(format!(
                "{error}; additionally {}",
                rollback_errors.join("; ")
            ));
        }
        return Err(error);
    }
    if let Some(snapshot) = force_snapshot {
        snapshot.commit(&layout.generation_lock);
    }
    println!(
        "installed {} plugin marketplace at {}",
        host.label(),
        layout.marketplace_root.display()
    );
    Ok(())
}

fn uninstall_host(
    host: IntegrationHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    uninstall_host_with_operation_timeout(
        host,
        options,
        runner,
        setup_runner,
        DEFAULT_OPERATION_LOCK_TIMEOUT,
    )
}

fn uninstall_host_with_operation_timeout(
    host: IntegrationHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
    lock_timeout: Duration,
) -> Result<(), String> {
    let _operation_lock = (!options.dry_run)
        .then(|| {
            PluginOperationLock::acquire(
                host,
                &options.operation_lock_dir,
                &options.install_dir,
                lock_timeout,
            )
        })
        .transpose()?;
    uninstall_host_locked(host, options, runner, setup_runner)
}

fn uninstall_host_locked(
    host: IntegrationHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    let state = read_state(host, &options.install_dir);
    let layout = PluginLayout::new(host, &options.install_dir);
    if let Some(state) = state.as_ref() {
        layout.validate_persisted_state(state)?;
    }
    let plugin_root = state
        .as_ref()
        .map(|state| state.plugin_root.as_path())
        .unwrap_or(&layout.plugin_root);
    let local_install_exists = state.is_some() || layout.marketplace_root.exists();
    let mut generation_retirement = retire_installed_generation(
        host,
        plugin_root,
        &layout.generation_lock,
        local_install_exists,
        options,
        runner,
    )?;
    if let Some(retirement) = generation_retirement.as_mut() {
        retirement.invalidate_for_replacement().map_err(|error| {
            format!(
                "failed to retire installed MCP generation before uninstalling {}: {error}",
                plugin_root.display()
            )
        })?;
    }
    if !options.dry_run
        && let Err(error) = setup_runner.refresh_gateway()
    {
        if let Some(retirement) = generation_retirement.as_mut()
            && let Err(restore_error) = retirement.restore_after_rollback()
        {
            return Err(format!(
                "{error}; additionally failed to restore the installed MCP generation: {restore_error}"
            ));
        }
        return Err(error);
    }
    let retired_lock = generation_retirement
        .as_ref()
        .map(|retirement| retirement.lock_path().to_owned());
    if let Some(retirement) = generation_retirement.as_mut() {
        retirement.release_legacy_lock_for_tree_mutation()?;
        retirement.commit_replacement();
    }
    let result = uninstall_host_with_setup_override(host, options, runner, setup_runner, false);
    if result.is_ok() {
        drop(generation_retirement);
        if let Some(lock_path) = retired_lock {
            remove_generation_lock_best_effort(&lock_path);
        }
    }
    result
}

fn retire_installed_generation(
    host: IntegrationHost,
    plugin_root: &Path,
    expected_generation_lock: &Path,
    local_install_exists: bool,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<Option<GenerationRetirement>, String> {
    if options.dry_run {
        return Ok(None);
    }
    let generation_fence = plugin_root.join(GENERATION_FILE_NAME);
    let mut existing_install = local_install_exists;
    if !generation_fence.exists() {
        let registration = host_registration_report(host, options, runner)?;
        existing_install |=
            registration.host_plugin_registered || registration.host_marketplace_registered;
        if existing_install && !legacy_plugin_without_mcp(host, plugin_root)? {
            return Err(missing_generation_fence_error(host, &generation_fence));
        }
    }
    let retirement =
        GenerationRetirement::acquire_for_plugin(&generation_fence, expected_generation_lock)
            .map_err(|cause| invalid_generation_fence_error(host, &generation_fence, &cause))?;
    if retirement.is_none() && !existing_install {
        let registration = host_registration_report(host, options, runner)?;
        existing_install =
            registration.host_plugin_registered || registration.host_marketplace_registered;
    }
    if retirement.is_none() && existing_install && !legacy_plugin_without_mcp(host, plugin_root)? {
        return Err(missing_generation_fence_error(host, &generation_fence));
    }
    Ok(retirement)
}

fn retire_replacement_before_rollback(
    host: IntegrationHost,
    layout: &PluginLayout,
    options: &PluginInstallOptions,
    setup_runner: &dyn PluginSetupRunner,
    existing_retirement: Option<&mut GenerationRetirement>,
) -> Result<Option<GenerationRetirement>, String> {
    if options.dry_run {
        return Ok(None);
    }
    if let Some(retirement) = existing_retirement
        && retirement.uses_lock_path(&layout.generation_lock)?
    {
        let visible = retirement.retire_visible_replacement().map_err(|error| {
            format!(
                "failed to retire replacement MCP generation {} before rollback: {error}",
                layout.generation_fence.display()
            )
        })?;
        if let Err(error) = setup_runner.refresh_gateway() {
            return match retirement.restore_visible_replacement(visible) {
                Ok(()) => Err(error),
                Err(restore_error) => Err(format!(
                    "{error}; additionally failed to restore the replacement MCP generation after rollback refresh failed: {restore_error}"
                )),
            };
        }
        return Ok(None);
    }
    let mut retirement =
        GenerationRetirement::acquire_for_plugin(&layout.generation_fence, &layout.generation_lock)
            .map_err(|cause| {
                invalid_generation_fence_error(host, &layout.generation_fence, &cause)
            })?
            .ok_or_else(|| missing_generation_fence_error(host, &layout.generation_fence))?;
    retirement.invalidate_for_replacement().map_err(|error| {
        format!(
            "failed to retire replacement MCP generation {} before rollback: {error}",
            layout.generation_fence.display()
        )
    })?;
    if let Err(error) = setup_runner.refresh_gateway() {
        return match retirement.restore_after_rollback() {
            Ok(()) => Err(error),
            Err(restore_error) => Err(format!(
                "{error}; additionally failed to restore the replacement MCP generation after rollback refresh failed: {restore_error}"
            )),
        };
    }
    retirement.commit_replacement();
    Ok(Some(retirement))
}

fn existing_plugin_install_requires_force_error(host: IntegrationHost) -> String {
    format!(
        "an existing fenced {} plugin install was found; rerun `nemo-relay install {} --force` to replace it safely",
        host.label(),
        host.as_arg()
    )
}

fn missing_generation_fence_error(host: IntegrationHost, generation_fence: &Path) -> String {
    unsafe_generation_fence_error(
        host,
        &format!("is missing at {}", generation_fence.display()),
    )
}

fn invalid_generation_fence_error(
    host: IntegrationHost,
    generation_fence: &Path,
    cause: &str,
) -> String {
    unsafe_generation_fence_error(
        host,
        &format!(
            "at {} is invalid or unreadable: {cause}",
            generation_fence.display()
        ),
    )
}

fn unsafe_generation_fence_error(host: IntegrationHost, problem: &str) -> String {
    match host {
        IntegrationHost::Codex => format!(
            "cannot safely replace or uninstall an existing Codex plugin because its MCP generation marker {problem}; close all Codex clients and standalone `nemo-relay mcp` processes, run `codex plugin remove nemo-relay-plugin@nemo-relay-local` and `codex plugin marketplace remove nemo-relay-local`, remove the stale marketplace and state from the selected install directory, then run `nemo-relay install codex --force` to create a fenced install (and `nemo-relay uninstall codex` afterward if removal was intended)"
        ),
        IntegrationHost::ClaudeCode => format!(
            "cannot safely replace or uninstall an existing Claude Code plugin because its MCP generation marker {problem}; close all Claude Code clients and standalone `nemo-relay mcp` processes, run `claude plugin uninstall nemo-relay-plugin` and `claude plugin marketplace remove nemo-relay-local`, remove the stale marketplace and state from the selected install directory, then run `nemo-relay install claude-code --force` to create a fenced install (and `nemo-relay uninstall claude-code` afterward if removal was intended)"
        ),
        IntegrationHost::Hermes | IntegrationHost::All => {
            unreachable!("all is expanded before generation validation")
        }
    }
}

fn legacy_plugin_without_mcp(host: IntegrationHost, plugin_root: &Path) -> Result<bool, String> {
    if !matches!(host, IntegrationHost::ClaudeCode) || plugin_root.join(".mcp.json").exists() {
        return Ok(false);
    }
    let manifest_path = plugin_manifest_path(host, plugin_root);
    let raw = match fs::read_to_string(&manifest_path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(format!(
                "failed to inspect legacy plugin manifest {}: {error}",
                manifest_path.display()
            ));
        }
    };
    let manifest = serde_json::from_str::<Value>(&raw).map_err(|error| {
        format!(
            "failed to inspect legacy plugin manifest {}: {error}",
            manifest_path.display()
        )
    })?;
    Ok(manifest.get("mcpServers").is_none())
}

fn uninstall_host_with_setup_override(
    host: IntegrationHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
    force_plugin_setup_uninstall: bool,
) -> Result<(), String> {
    let layout = PluginLayout::new(host, &options.install_dir);
    let state = read_state(host, &options.install_dir).unwrap_or_else(|| PluginState {
        marketplace_root: layout.marketplace_root.clone(),
        plugin_root: layout.plugin_root.clone(),
        host_plugin_removed: false,
        host_marketplace_removed: false,
        plugin_setup_installed: true,
    });
    layout.validate_persisted_state(&state)?;
    if let Err(error) = require_relay(options, runner)
        .and_then(|relay| validate_relay_hook_forward(&relay, options, runner))
    {
        eprintln!("warning: skipping nemo-relay validation during uninstall: {error}");
    }
    let mut state = state;
    if force_plugin_setup_uninstall && !state.plugin_setup_installed {
        state.plugin_setup_installed = true;
        write_state_for_host(host, &state, &options.install_dir, options)?;
    }
    if force_plugin_setup_uninstall || state.plugin_setup_installed {
        run_plugin_uninstall(host, &state.plugin_root, options, setup_runner)?;
        state.plugin_setup_installed = false;
        write_state_for_host(host, &state, &options.install_dir, options)?;
    }
    run_host_unregistration(host, &mut state, &options.install_dir, options, runner)?;
    remove_path(&state.marketplace_root, options)?;
    remove_path(&state_path(host, &options.install_dir), options)?;
    println!("uninstalled {} plugin", host.label());
    Ok(())
}

fn doctor_host(
    host: IntegrationHost,
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
            "{} plugin doctor checks failed; remediation: {}",
            host.label(),
            readiness.remediation
        )
    })
}

fn doctor_host_json_value(
    host: IntegrationHost,
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
    host: IntegrationHost,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> HostPluginReadiness {
    let state_path = state_path(host, &options.install_dir);
    let state = read_state(host, &options.install_dir);
    let layout = PluginLayout::new(host, &options.install_dir);
    let setup_plugin_root = state
        .as_ref()
        .map(|state| state.plugin_root.clone())
        .unwrap_or_else(|| layout.plugin_root.clone());
    let marketplace = state
        .as_ref()
        .map(|state| state.marketplace_root.clone())
        .or_else(|| state_path.exists().then(|| layout.marketplace_root.clone()));
    let plugin = state
        .as_ref()
        .map(|state| state.plugin_root.clone())
        .or_else(|| state_path.exists().then(|| layout.plugin_root.clone()));
    let mut readiness = HostPluginReadiness {
        host: host.as_arg().to_string(),
        remediation: format!("nemo-relay install {} --force", host.as_arg()),
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
            validate_relay_hook_forward(&relay, options, runner)
                .map(|_| "hook-forward is supported".into()),
        );
        if let Some(plugin) = readiness.plugin.as_ref() {
            let generation_fence =
                plugin.join(crate::installation::generation::GENERATION_FILE_NAME);
            readiness.push(
                "Generated hooks",
                InstallGeneration::capture(generation_fence.clone()).and_then(|generation| {
                    let expected =
                        plugin_hooks(host, &relay, &generation_fence, generation.token())?;
                    generated_manifest_check(
                        &plugin.join("hooks").join("hooks.json"),
                        &expected,
                        "hooks",
                    )
                }),
            );
        }
        readiness.push(
            "Relay MCP support",
            validate_relay_mcp(&relay, options, runner)
                .map(|_| "native mcp subcommand is supported".into()),
        );
        if let Some(plugin) = readiness.plugin.as_ref() {
            let generation_fence =
                plugin.join(crate::installation::generation::GENERATION_FILE_NAME);
            let mcp_config = plugin_mcp_config_path(plugin);
            readiness.push(
                "MCP generation fence",
                InstallGeneration::capture(generation_fence.clone())
                    .map(|_| format!("valid generation at {}", generation_fence.display())),
            );
            let check =
                InstallGeneration::capture(generation_fence.clone()).and_then(|generation| {
                    plugin_mcp_config(host, &relay, &generation_fence, generation.token()).and_then(
                        |expected| generated_mcp_config_check(host, &mcp_config, &expected),
                    )
                });
            readiness.push("Generated MCP server", check);
        }
    }

    let host_cli_check = require_host_cli(host, options, runner);
    readiness.push(
        "Host CLI",
        host_cli_check
            .as_ref()
            .map(|_| {
                format!(
                    "{} is available",
                    host.executable().expect("concrete plugin host")
                )
            })
            .map_err(Clone::clone),
    );
    if host_cli_check.is_ok() {
        let agent = host
            .agent()
            .expect("all is expanded before readiness checks");
        let version = host::validate_host_version(host, options, runner);
        if version.is_err() {
            readiness.remediation = format!(
                "upgrade to {}, then run `nemo-relay install {} --force`",
                agent.version_requirement(),
                host.as_arg()
            );
        }
        readiness.push(
            format!("{} version", agent.label()),
            version.map(|_| format!("{} is installed", agent.version_requirement())),
        );
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

    match run_plugin_doctor_json(host, &setup_plugin_root, setup_runner) {
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

fn generated_mcp_config_check(
    host: IntegrationHost,
    path: &Path,
    expected: &Value,
) -> Result<String, String> {
    generated_mcp_config_check_for_platform(host, path, expected, cfg!(windows))
}

fn generated_mcp_config_check_for_platform(
    host: IntegrationHost,
    path: &Path,
    expected: &Value,
    windows: bool,
) -> Result<String, String> {
    let raw = std::fs::read_to_string(path).map_err(|error| {
        format!(
            "missing or unreadable MCP server manifest {}: {error}",
            path.display()
        )
    })?;
    let actual = serde_json::from_str::<Value>(&raw)
        .map_err(|error| format!("invalid MCP server manifest {}: {error}", path.display()))?;
    if actual == *expected {
        return Ok(format!("valid at {}", path.display()));
    }
    if !matches!(host, IntegrationHost::Codex) {
        return Err(format!(
            "unexpected MCP server manifest contents at {}; run `nemo-relay install {} --force`",
            path.display(),
            host.as_arg()
        ));
    }
    let expected_server = &expected["nemo-relay"];
    let actual_server = &actual["nemo-relay"];
    let Some(expected_vars) = mcp_env_var_names(expected_server) else {
        return Err(format!(
            "unexpected MCP server manifest contents at {}; run `nemo-relay install {} --force`",
            path.display(),
            host.as_arg()
        ));
    };
    let Some(actual_vars) = mcp_env_var_names(actual_server) else {
        return Err(format!(
            "unexpected MCP server manifest contents at {}; run `nemo-relay install {} --force`",
            path.display(),
            host.as_arg()
        ));
    };
    let duplicate = actual_vars.iter().enumerate().any(|(index, name)| {
        actual_vars[..index].iter().any(|other| {
            crate::mcp_environment::forwarded_names_match_for_platform(name, other, windows)
        })
    });
    if duplicate
        || actual_vars.iter().any(|name| {
            !expected_vars.iter().any(|expected| {
                crate::mcp_environment::forwarded_names_match_for_platform(name, expected, windows)
            }) && !crate::mcp_environment::previously_forwardable_name_for_platform(name, windows)
        })
    {
        return Err(format!(
            "unexpected MCP server manifest contents at {}; run `nemo-relay install {} --force`",
            path.display(),
            host.as_arg()
        ));
    }
    let missing = expected_vars
        .iter()
        .filter(|expected| {
            !actual_vars.iter().any(|actual| {
                crate::mcp_environment::forwarded_names_match_for_platform(
                    expected, actual, windows,
                )
            })
        })
        .map(String::as_str)
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "MCP server at {} is missing forwarded environment variables: {}; run `nemo-relay install {} --force`",
            path.display(),
            missing.join(", "),
            host.as_arg()
        ));
    }
    let mut expected_without_vars = expected.clone();
    let mut actual_without_vars = actual.clone();
    let expected_server = expected_without_vars
        .get_mut("nemo-relay")
        .and_then(Value::as_object_mut);
    let actual_server = actual_without_vars
        .get_mut("nemo-relay")
        .and_then(Value::as_object_mut);
    if let (Some(expected_server), Some(actual_server)) = (expected_server, actual_server) {
        expected_server.remove("env_vars");
        actual_server.remove("env_vars");
        if actual_without_vars == expected_without_vars {
            return Ok(format!("valid at {}", path.display()));
        }
    }
    Err(format!(
        "unexpected MCP server manifest contents at {}; run `nemo-relay install {} --force`",
        path.display(),
        host.as_arg()
    ))
}

fn mcp_env_var_names(server: &Value) -> Option<Vec<String>> {
    server
        .get("env_vars")?
        .as_array()?
        .iter()
        .map(|value| value.as_str().map(str::to_owned))
        .collect()
}

fn marketplace_manifest_path(host: IntegrationHost, root: &Path) -> PathBuf {
    match host {
        IntegrationHost::Codex => root
            .join(".agents")
            .join("plugins")
            .join("marketplace.json"),
        IntegrationHost::ClaudeCode => root.join(".claude-plugin").join("marketplace.json"),
        IntegrationHost::Hermes | IntegrationHost::All => {
            unreachable!("all is expanded before layout resolution")
        }
    }
}

fn plugin_manifest_path(host: IntegrationHost, root: &Path) -> PathBuf {
    match host {
        IntegrationHost::Codex => root.join(".codex-plugin").join("plugin.json"),
        IntegrationHost::ClaudeCode => root.join(".claude-plugin").join("plugin.json"),
        IntegrationHost::Hermes | IntegrationHost::All => {
            unreachable!("all is expanded before layout resolution")
        }
    }
}

fn plugin_mcp_config_path(root: &Path) -> PathBuf {
    root.join(".mcp.json")
}

struct StagedPluginMarketplace {
    layout: PluginLayout,
    parent: PathBuf,
    generation_lock_created: bool,
}

struct ReplacementGenerationLock {
    retirement: Option<GenerationRetirement>,
    lock_path: PathBuf,
    remove_lock_if_unreferenced: bool,
}

fn acquire_replacement_generation_lock(
    host: IntegrationHost,
    marker_path: &Path,
    expected_generation_lock: &Path,
    remove_lock_if_unreferenced: bool,
) -> Result<ReplacementGenerationLock, String> {
    match GenerationRetirement::acquire_for_plugin(marker_path, expected_generation_lock) {
        Ok(Some(retirement)) => Ok(ReplacementGenerationLock::new(
            retirement,
            remove_lock_if_unreferenced,
        )),
        Ok(None) => Err(missing_generation_fence_error(host, marker_path)),
        Err(cause) => Err(invalid_generation_fence_error(host, marker_path, &cause)),
    }
}

impl ReplacementGenerationLock {
    fn new(retirement: GenerationRetirement, remove_lock_if_unreferenced: bool) -> Self {
        Self {
            lock_path: retirement.lock_path().to_owned(),
            retirement: Some(retirement),
            remove_lock_if_unreferenced,
        }
    }

    fn retirement_mut(&mut self) -> &mut GenerationRetirement {
        self.retirement
            .as_mut()
            .expect("replacement generation transaction remains present")
    }

    fn retirement(&self) -> &GenerationRetirement {
        self.retirement
            .as_ref()
            .expect("replacement generation transaction remains present")
    }

    fn retarget_promoted_marker(&mut self, marker_path: &Path) -> Result<(), String> {
        self.retirement_mut().retarget_promoted_marker(marker_path)
    }
}

impl Drop for ReplacementGenerationLock {
    fn drop(&mut self) {
        let remove_owned_lock = self.remove_lock_if_unreferenced
            && self.retirement.as_ref().is_some_and(|retirement| {
                match fs::metadata(retirement.marker_path()) {
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
                    Err(error) => {
                        eprintln!(
                            "warning: retaining MCP generation lock {} because marker {} could not be inspected: {error}",
                            self.lock_path.display(),
                            retirement.marker_path().display()
                        );
                        false
                    }
                    Ok(_) => match retirement.visible_marker_uses_transaction_lock() {
                        Ok(referenced) => !referenced,
                        Err(error) => {
                            eprintln!(
                                "warning: retaining MCP generation lock {} because its marker reference could not be verified: {error}",
                                self.lock_path.display()
                            );
                            false
                        }
                    },
                }
            });
        drop(self.retirement.take());
        if remove_owned_lock {
            remove_generation_lock_best_effort(&self.lock_path);
        }
    }
}

impl StagedPluginMarketplace {
    fn promote(&self, target: &PluginLayout) -> Result<(), String> {
        fs::rename(&self.layout.marketplace_root, &target.marketplace_root).map_err(|error| {
            format!(
                "failed to promote staged marketplace {} to {}: {error}",
                self.layout.marketplace_root.display(),
                target.marketplace_root.display()
            )
        })
    }

    fn cleanup(&self) {
        let _ = fs::remove_dir_all(&self.parent);
    }
}

struct PluginInstallPreflight {
    persisted: Option<PluginState>,
    state_bytes: Option<Vec<u8>>,
    previous_marketplace_root: PathBuf,
    previous_plugin_root: PathBuf,
    previous_generation_fence: PathBuf,
    plugin_registered: bool,
    marketplace_registered: bool,
    previous_setup_installed: bool,
    previous_install_exists: bool,
    generation_retirement: Option<GenerationRetirement>,
}

fn prepare_plugin_install(
    host: IntegrationHost,
    layout: &PluginLayout,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
) -> Result<PluginInstallPreflight, String> {
    let persisted = read_state(host, &options.install_dir);
    if let Some(state) = persisted.as_ref() {
        layout.validate_persisted_state(state)?;
    }
    let registration = host_registration_report(host, options, runner)?;
    let plugin_registered = registration.host_plugin_registered;
    let marketplace_registered = registration.host_marketplace_registered;
    let state_bytes = match fs::read(&layout.state_path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(format!(
                "failed to snapshot {}: {error}",
                layout.state_path.display()
            ));
        }
    };
    let previous_setup_installed = persisted
        .as_ref()
        .is_some_and(|state| state.plugin_setup_installed)
        || plugin_registered;
    let previous_marketplace_root = persisted
        .as_ref()
        .map(|state| state.marketplace_root.clone())
        .unwrap_or_else(|| layout.marketplace_root.clone());
    let previous_plugin_root = persisted
        .as_ref()
        .map(|state| state.plugin_root.clone())
        .unwrap_or_else(|| layout.plugin_root.clone());
    let previous_generation_fence = previous_plugin_root.join(GENERATION_FILE_NAME);
    let local_install_exists = match host {
        IntegrationHost::Codex => layout.marketplace_root.exists(),
        IntegrationHost::ClaudeCode => {
            plugin_manifest_path(host, &previous_plugin_root).exists()
                || previous_plugin_root.join(".mcp.json").exists()
                || previous_generation_fence.exists()
        }
        IntegrationHost::Hermes | IntegrationHost::All => {
            unreachable!("all is expanded before install preflight")
        }
    };
    let previous_install_exists = state_bytes.is_some()
        || local_install_exists
        || plugin_registered
        || marketplace_registered;
    let generation_retirement = if previous_install_exists {
        if !previous_generation_fence.exists() {
            if legacy_plugin_without_mcp(host, &previous_plugin_root)? {
                None
            } else {
                return Err(missing_generation_fence_error(
                    host,
                    &previous_generation_fence,
                ));
            }
        } else {
            Some(
                GenerationRetirement::acquire_for_plugin(
                    &previous_generation_fence,
                    &layout.generation_lock,
                )
                .map_err(|cause| {
                    invalid_generation_fence_error(host, &previous_generation_fence, &cause)
                })?
                .ok_or_else(|| missing_generation_fence_error(host, &previous_generation_fence))?,
            )
        }
    } else {
        None
    };
    Ok(PluginInstallPreflight {
        persisted,
        state_bytes,
        previous_marketplace_root,
        previous_plugin_root,
        previous_generation_fence,
        plugin_registered,
        marketplace_registered,
        previous_setup_installed,
        previous_install_exists,
        generation_retirement,
    })
}

struct ForceInstallSnapshot {
    state_bytes: Option<Vec<u8>>,
    setup_snapshot: Option<PluginSetupSnapshot>,
    original_marketplace_root: PathBuf,
    original_plugin_root: PathBuf,
    original_generation_fence: PathBuf,
    plugin_registered: bool,
    marketplace_registered: bool,
    backup_marketplace_root: PathBuf,
    backup_plugin_root: Option<PathBuf>,
    marketplace_moved: bool,
    plugin_moved: bool,
    replacement_promoted: bool,
    generation_retirement: Option<GenerationRetirement>,
}

impl ForceInstallSnapshot {
    fn plugin_moves_with_marketplace(&self) -> bool {
        self.original_plugin_root
            .starts_with(&self.original_marketplace_root)
    }

    fn commit(mut self, replacement_lock: &Path) {
        let obsolete_lock = self.generation_retirement.as_ref().and_then(|retirement| {
            retirement
                .uses_lock_path(replacement_lock)
                .ok()
                .filter(|same| !same)
                .map(|_| retirement.lock_path().to_owned())
        });
        if let Some(retirement) = self.generation_retirement.as_mut() {
            retirement.commit_replacement();
        }
        if self.marketplace_moved {
            match fs::remove_dir_all(&self.backup_marketplace_root) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => eprintln!(
                    "warning: failed to remove replaced marketplace backup {}: {error}",
                    self.backup_marketplace_root.display()
                ),
            }
        }
        if self.plugin_moved
            && let Some(backup_plugin_root) = self.backup_plugin_root.as_ref()
        {
            match fs::remove_dir_all(backup_plugin_root) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => eprintln!(
                    "warning: failed to remove replaced plugin backup {}: {error}",
                    backup_plugin_root.display()
                ),
            }
        }
        drop(self.generation_retirement.take());
        if let Some(lock_path) = obsolete_lock {
            remove_generation_lock_best_effort(&lock_path);
        }
    }
}

fn remove_generation_lock_best_effort(path: &Path) {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => eprintln!(
            "warning: failed to remove retired MCP generation lock {}: {error}",
            path.display()
        ),
    }
}

fn generation_lock_is_absent(path: &Path) -> bool {
    matches!(
        fs::symlink_metadata(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound
    )
}

fn stage_plugin_marketplace(
    host: IntegrationHost,
    relay: &Path,
    target: &PluginLayout,
    initialize_generation_lock: bool,
    options: &PluginInstallOptions,
) -> Result<StagedPluginMarketplace, String> {
    let parent = options.install_dir.join(format!(
        ".{}-install-stage-{}",
        host.as_arg(),
        uuid::Uuid::now_v7()
    ));
    stage_plugin_marketplace_at(
        host,
        relay,
        target,
        initialize_generation_lock,
        options,
        parent,
    )
}

fn stage_plugin_marketplace_at(
    host: IntegrationHost,
    relay: &Path,
    target: &PluginLayout,
    initialize_generation_lock: bool,
    options: &PluginInstallOptions,
    parent: PathBuf,
) -> Result<StagedPluginMarketplace, String> {
    let layout = PluginLayout::new(host, &parent);
    let generation_lock_created =
        initialize_generation_lock && generation_lock_is_absent(&target.generation_lock);
    if let Err(error) = write_plugin_marketplace_for_generation(
        host,
        &layout,
        relay,
        &target.generation_fence,
        &target.generation_lock,
        initialize_generation_lock,
        options,
    ) {
        let _ = fs::remove_dir_all(&parent);
        if generation_lock_created {
            remove_generation_lock_best_effort(&target.generation_lock);
        }
        return Err(error);
    }
    Ok(StagedPluginMarketplace {
        layout,
        parent,
        generation_lock_created,
    })
}

fn begin_force_replacement(
    host: IntegrationHost,
    layout: &PluginLayout,
    preflight: PluginInstallPreflight,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<ForceInstallSnapshot, String> {
    let PluginInstallPreflight {
        persisted,
        state_bytes,
        previous_marketplace_root,
        previous_plugin_root,
        previous_generation_fence,
        plugin_registered,
        marketplace_registered,
        previous_setup_installed,
        previous_install_exists: _,
        generation_retirement,
    } = preflight;
    let setup_snapshot = setup_runner.snapshot(host)?;
    let backup_parent = previous_marketplace_root
        .parent()
        .unwrap_or(&options.install_dir);
    let backup_marketplace_root = backup_parent.join(format!(
        ".{}-marketplace-backup-{}",
        host.as_arg(),
        uuid::Uuid::now_v7()
    ));
    let backup_plugin_root =
        (!previous_plugin_root.starts_with(&previous_marketplace_root)).then(|| {
            previous_plugin_root
                .parent()
                .unwrap_or(&options.install_dir)
                .join(format!(
                    ".{}-plugin-backup-{}",
                    host.as_arg(),
                    uuid::Uuid::now_v7()
                ))
        });
    let mut snapshot = ForceInstallSnapshot {
        state_bytes,
        setup_snapshot,
        original_marketplace_root: previous_marketplace_root,
        original_plugin_root: previous_plugin_root,
        original_generation_fence: previous_generation_fence,
        plugin_registered,
        marketplace_registered,
        backup_marketplace_root,
        backup_plugin_root,
        marketplace_moved: false,
        plugin_moved: false,
        replacement_promoted: false,
        generation_retirement,
    };
    let mut cleanup_state = persisted.unwrap_or_else(|| PluginState {
        marketplace_root: layout.marketplace_root.clone(),
        plugin_root: layout.plugin_root.clone(),
        host_plugin_removed: !plugin_registered,
        host_marketplace_removed: !marketplace_registered,
        plugin_setup_installed: previous_setup_installed,
    });
    cleanup_state.host_plugin_removed = !plugin_registered;
    cleanup_state.host_marketplace_removed = !marketplace_registered;
    let result = (|| {
        if cleanup_state.plugin_setup_installed {
            run_plugin_uninstall(
                host,
                &cleanup_state.plugin_root,
                options,
                setup_runner,
            )?;
            cleanup_state.plugin_setup_installed = false;
        }
        run_host_unregistration(
            host,
            &mut cleanup_state,
            &options.install_dir,
            options,
            runner,
        )
    })()
    .and_then(|()| {
        if let Some(retirement) = snapshot.generation_retirement.as_mut() {
            retirement.invalidate_for_replacement().map_err(|error| {
                format!(
                    "failed to retire previous MCP generation {} before replacement: {error}",
                    snapshot.original_generation_fence.display()
                )
            })?;
            retirement
                .release_legacy_lock_for_tree_mutation()
                .map_err(|error| {
                    format!(
                        "failed to release previous MCP generation {} before moving its plugin tree: {error}",
                        snapshot.original_generation_fence.display()
                    )
                })?;
        }
        if snapshot.original_marketplace_root.exists() {
            fs::rename(
                &snapshot.original_marketplace_root,
                &snapshot.backup_marketplace_root,
            )
            .map_err(|error| {
                format!(
                    "failed to preserve existing marketplace {}: {error}",
                    snapshot.original_marketplace_root.display()
                )
            })?;
            snapshot.marketplace_moved = true;
        }
        if !snapshot.plugin_moves_with_marketplace() && snapshot.original_plugin_root.exists() {
            let backup_plugin_root = snapshot
                .backup_plugin_root
                .as_ref()
                .expect("separate original plugin root has a backup path");
            fs::rename(&snapshot.original_plugin_root, backup_plugin_root).map_err(|error| {
                format!(
                    "failed to preserve existing plugin root {} containing generation marker {}: {error}",
                    snapshot.original_plugin_root.display(),
                    snapshot.original_generation_fence.display()
                )
            })?;
            snapshot.plugin_moved = true;
        }
        Ok(())
    });
    if let Err(error) = result {
        return restore_force_replacement_after_error(
            host,
            layout,
            &mut snapshot,
            options,
            runner,
            setup_runner,
            error,
        );
    }
    Ok(snapshot)
}

fn restore_force_replacement_after_error<T>(
    host: IntegrationHost,
    layout: &PluginLayout,
    snapshot: &mut ForceInstallSnapshot,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
    original_error: String,
) -> Result<T, String> {
    match restore_force_replacement(host, layout, snapshot, options, runner, setup_runner) {
        Ok(()) => Err(original_error),
        Err(rollback_error) => Err(format!(
            "{original_error}; additionally failed to restore previous install: {rollback_error}"
        )),
    }
}

fn restore_force_replacement(
    host: IntegrationHost,
    layout: &PluginLayout,
    snapshot: &mut ForceInstallSnapshot,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    let mut errors = Vec::new();
    if snapshot.replacement_promoted {
        match host_registration_report(host, options, runner) {
            Ok(report) => {
                if report.host_plugin_registered
                    && let Err(error) = run_host_plugin_removal(host, options, runner)
                {
                    errors.push(error);
                }
                if report.host_marketplace_registered
                    && let Err(error) = run_host_marketplace_removal(host, options, runner)
                {
                    errors.push(error);
                }
            }
            Err(error) => errors.push(error),
        }
        if let Err(error) = remove_path(&layout.marketplace_root, options) {
            errors.push(error);
        }
        snapshot.replacement_promoted = false;
    }
    if snapshot.marketplace_moved {
        if let Err(error) = fs::rename(
            &snapshot.backup_marketplace_root,
            &snapshot.original_marketplace_root,
        ) {
            errors.push(format!(
                "failed to restore marketplace {}: {error}",
                snapshot.original_marketplace_root.display()
            ));
        } else {
            snapshot.marketplace_moved = false;
        }
    }
    if snapshot.plugin_moved
        && let Some(backup_plugin_root) = snapshot.backup_plugin_root.as_ref()
    {
        if let Err(error) = fs::rename(backup_plugin_root, &snapshot.original_plugin_root) {
            errors.push(format!(
                "failed to restore plugin root {} containing generation marker {}: {error}",
                snapshot.original_plugin_root.display(),
                snapshot.original_generation_fence.display()
            ));
        } else {
            snapshot.plugin_moved = false;
        }
    }
    if let Some(retirement) = snapshot.generation_retirement.as_mut()
        && let Err(error) = retirement.restore_after_rollback()
    {
        errors.push(error);
    }
    match host_registration_report(host, options, runner) {
        Ok(report) => {
            if report.host_plugin_registered
                && !snapshot.plugin_registered
                && let Err(error) = run_host_plugin_removal(host, options, runner)
            {
                errors.push(error);
            }
            if report.host_marketplace_registered
                && !snapshot.marketplace_registered
                && let Err(error) = run_host_marketplace_removal(host, options, runner)
            {
                errors.push(error);
            }
            if snapshot.marketplace_registered
                && !report.host_marketplace_registered
                && let Err(error) = run_host_marketplace_registration(
                    host,
                    &snapshot.original_marketplace_root,
                    options,
                    runner,
                )
            {
                errors.push(error);
            }
            if snapshot.plugin_registered
                && !report.host_plugin_registered
                && let Err(error) = run_host_plugin_registration(host, options, runner)
            {
                errors.push(error);
            }
        }
        Err(error) => errors.push(error),
    }
    if let Some(setup_snapshot) = snapshot.setup_snapshot.as_ref()
        && let Err(error) = setup_runner.restore_snapshot(setup_snapshot)
    {
        errors.push(error);
    }
    if let Some(bytes) = snapshot.state_bytes.as_deref() {
        if let Some(parent) = layout.state_path.parent()
            && let Err(error) = fs::create_dir_all(parent)
        {
            errors.push(format!("failed to create {}: {error}", parent.display()));
        }
        if let Err(error) = fs::write(&layout.state_path, bytes) {
            errors.push(format!(
                "failed to restore {}: {error}",
                layout.state_path.display()
            ));
        }
    } else if let Err(error) = fs::remove_file(&layout.state_path)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        errors.push(format!(
            "failed to remove {}: {error}",
            layout.state_path.display()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

fn force_cleanup_existing_install(
    host: IntegrationHost,
    layout: &PluginLayout,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    if layout.state_path.exists() {
        uninstall_host_locked(host, options, runner, setup_runner)?;
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
    host: IntegrationHost,
    layout: &PluginLayout,
    registration: HostRegistrationProgress,
    setup_installed: bool,
    options: &PluginInstallOptions,
    runner: &dyn CommandRunner,
    setup_runner: &dyn PluginSetupRunner,
) -> Result<(), String> {
    if setup_installed {
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
    host: IntegrationHost,
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
#[path = "../../../tests/coverage/agents/plugin_install_tests.rs"]
mod tests;
