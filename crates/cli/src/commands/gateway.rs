// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::path::Path;
use std::process::ExitCode;
use std::thread;
use std::time::{Duration, Instant};

use clap::{Args, Subcommand};
use listeners::{Listener, Process, Protocol};
#[cfg(any(unix, windows))]
use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, Signal, System};

use crate::error::CliError;

use super::serve::ServerArgs;

#[derive(Debug, Clone, Args)]
pub(crate) struct GatewayCommand {
    #[command(subcommand)]
    command: GatewaySubcommand,
}

#[derive(Debug, Clone, Subcommand)]
enum GatewaySubcommand {
    /// Start the gateway with the same server configuration as a bare daemon invocation.
    Start,
    /// Stop the gateway process listening at the configured loopback endpoint.
    Stop {
        /// Terminate immediately instead of requesting graceful shutdown (the Windows default).
        #[arg(long)]
        force: bool,
    },
}

impl GatewayCommand {
    /// Returns whether this command only stops an existing gateway.
    pub(crate) fn is_stop(&self) -> bool {
        matches!(self.command, GatewaySubcommand::Stop { .. })
    }
}

/// Executes a gateway lifecycle command.
pub(crate) async fn execute(
    command: GatewayCommand,
    server: &ServerArgs,
    bootstrap_shutdown_token: Option<String>,
) -> Result<ExitCode, CliError> {
    match command.command {
        GatewaySubcommand::Start => super::serve_gateway(server, bootstrap_shutdown_token).await,
        GatewaySubcommand::Stop { force } => stop(stop_bind(server), force),
    }
}

const STOP_TIMEOUT: Duration = Duration::from_secs(5);
const STOP_POLL_INTERVAL: Duration = Duration::from_millis(50);

pub(super) fn stop_bind(server: &ServerArgs) -> SocketAddr {
    server
        .bind
        .unwrap_or_else(|| crate::configuration::GatewayConfig::default().bind)
}

fn stop(bind: SocketAddr, force: bool) -> Result<ExitCode, CliError> {
    validate_stop_bind(bind)?;
    let Some(target) = resolve_relay_listener(bind)? else {
        return Ok(ExitCode::SUCCESS);
    };

    // Resolve again immediately before signaling. This avoids acting on a PID that stopped owning
    // the endpoint between discovery and action.
    let Some(current) = resolve_relay_listener(bind)? else {
        return Ok(ExitCode::SUCCESS);
    };
    if current.pid != target.pid {
        return Err(CliError::Launch(format!(
            "gateway listener at {bind} changed from PID {} to PID {} before shutdown",
            target.pid, current.pid
        )));
    }

    signal_process(target.pid, force).map_err(|error| {
        CliError::Launch(format!(
            "failed to stop gateway process {} at {bind}: {error}",
            target.pid
        ))
    })?;

    wait_for_listener_exit(bind, target.pid)?;
    Ok(ExitCode::SUCCESS)
}

pub(super) fn validate_stop_bind(bind: SocketAddr) -> Result<(), CliError> {
    if !bind.ip().is_loopback() {
        return Err(CliError::Config(format!(
            "gateway stop requires a loopback address, got {bind}"
        )));
    }
    if bind.port() == 0 {
        return Err(CliError::Config(
            "gateway stop requires a concrete nonzero port".into(),
        ));
    }
    Ok(())
}

fn resolve_relay_listener(bind: SocketAddr) -> Result<Option<Process>, CliError> {
    let listeners = listeners::get_all().map_err(|error| {
        CliError::Launch(format!(
            "failed to resolve the gateway process listening at {bind}: {error}"
        ))
    })?;
    select_relay_listener(bind, listeners)
}

pub(super) fn select_relay_listener(
    bind: SocketAddr,
    listeners: impl IntoIterator<Item = Listener>,
) -> Result<Option<Process>, CliError> {
    let mut matches = listeners
        .into_iter()
        .filter(|listener| listener.socket == bind && listener.protocol == Protocol::TCP);
    let Some(first) = matches.next() else {
        return Ok(None);
    };
    if let Some(other) = matches.find(|listener| listener.process.pid != first.process.pid) {
        return Err(CliError::Launch(format!(
            "multiple processes listen at gateway address {bind}: PIDs {} and {}",
            first.process.pid, other.process.pid
        )));
    }
    if !is_relay_process(&first.process) {
        return Err(CliError::Launch(format!(
            "refusing to stop non-Relay process '{}' at {bind}",
            first.process.name
        )));
    }
    Ok(Some(first.process))
}

fn is_relay_process(process: &Process) -> bool {
    Path::new(&process.name)
        .file_stem()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("nemo-relay"))
        || Path::new(&process.path)
            .file_stem()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("nemo-relay"))
}

fn wait_for_listener_exit(bind: SocketAddr, pid: u32) -> Result<(), CliError> {
    let deadline = Instant::now() + STOP_TIMEOUT;
    loop {
        match resolve_relay_listener(bind)? {
            None => return Ok(()),
            Some(process) if process.pid != pid => {
                return Err(CliError::Launch(format!(
                    "a different process replaced gateway PID {pid} at {bind} during shutdown"
                )));
            }
            Some(_) if Instant::now() < deadline => thread::sleep(STOP_POLL_INTERVAL),
            Some(_) => {
                return Err(CliError::Launch(format!(
                    "gateway process {pid} at {bind} did not stop"
                )));
            }
        }
    }
}

#[cfg(any(unix, windows))]
pub(super) fn signal_process(pid: u32, force: bool) -> Result<(), String> {
    let system = System::new_with_specifics(
        RefreshKind::nothing().with_processes(ProcessRefreshKind::nothing().without_tasks()),
    );
    let root = Pid::from_u32(pid);
    #[cfg(unix)]
    if force {
        signal_descendants(&system, root, Signal::Kill)?;
    }
    #[cfg(windows)]
    signal_descendants(&system, root, Signal::Kill)?;
    signal_pid(&system, root, stop_signal(force))
}

#[cfg(any(unix, windows))]
fn signal_descendants(system: &System, parent: Pid, signal: Signal) -> Result<(), String> {
    for pid in system
        .processes()
        .iter()
        .filter_map(|(pid, process)| (process.parent() == Some(parent)).then_some(*pid))
        .collect::<Vec<_>>()
    {
        signal_descendants(system, pid, signal)?;
        signal_pid(system, pid, signal)?;
    }
    Ok(())
}

#[cfg(any(unix, windows))]
fn signal_pid(system: &System, pid: Pid, signal: Signal) -> Result<(), String> {
    let Some(process) = system.process(pid) else {
        return Ok(());
    };
    process
        .kill_with(signal)
        .ok_or_else(|| format!("the platform cannot signal PID {}", pid.as_u32()))?
        .then_some(())
        .ok_or_else(|| format!("could not signal PID {}", pid.as_u32()))
}

#[cfg(unix)]
fn stop_signal(force: bool) -> Signal {
    if force {
        Signal::Kill
    } else {
        Signal::Interrupt
    }
}

#[cfg(windows)]
fn stop_signal(_force: bool) -> Signal {
    Signal::Kill
}
