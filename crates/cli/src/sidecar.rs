// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared native gateway sidecar lifecycle.

mod health;
mod process;
mod state;

use std::env;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{OnceLock, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use crate::config::{CodingAgent, ServerArgs, resolve_persistent_server_config};
use crate::error::CliError;
use crate::file_io::{LockAttempt, try_lock_exclusive};
#[cfg(test)]
pub(crate) use health::healthz_compatible;
use health::{
    RelayHealth, probe_after_lock as probe_relay_health_after_lock,
    probe_with_instance as probe_relay_health_with_instance,
};
pub(crate) use health::{healthz, loopback_bind};
use process::DetachedSidecarProcess;
#[cfg(all(windows, not(test)))]
use process::SidecarJob;
#[cfg(all(test, windows))]
pub(crate) use process::SidecarJob;
pub(crate) use process::configure_detached_sidecar;
pub(crate) use process::join_sidecar_job_from_env;
#[cfg(all(test, unix))]
pub(crate) use process::terminate_sidecar_process_tree;
#[cfg(test)]
pub(crate) use process::{
    WINDOWS_CREATE_BREAKAWAY_FROM_JOB, WINDOWS_CREATE_NEW_PROCESS_GROUP, WINDOWS_CREATE_NO_WINDOW,
    WINDOWS_JOB_OBJECT_LIMIT_BREAKAWAY_OK, WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_CLOSE,
    WINDOWS_JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK, terminate_unready_sidecar,
    windows_sidecar_creation_flags,
};
pub(crate) use state::BOOTSTRAP_STATE_DIR_ENV;
pub(crate) use state::EndpointLease;
use state::{
    RecoveryEpoch, create_private_runtime_dir, open_lock as open_sidecar_lock, read_owner_record,
    read_ready_file as read_sidecar_ready_file, read_recovery_epoch, runtime_dir,
    write_recovery_epoch,
};
pub(crate) use state::{
    lock_endpoint as lock_sidecar_endpoint, lock_path as sidecar_lock_path,
    owner_path as sidecar_owner_path, owner_paths as sidecar_owner_paths,
    owner_pid_path as sidecar_owner_pid_path, pid_path as sidecar_pid_path,
    state_dir as sidecar_state_dir, validate_owner as validate_sidecar_owner,
};
#[cfg(test)]
pub(crate) use state::{
    lock_endpoint_for as lock_sidecar_endpoint_for, lock_name as sidecar_lock_name,
    runtime_dir_for, stop_owned_record as stop_owned_sidecar_record,
    write_owner as write_sidecar_owner,
};
pub(crate) use state::{
    publish_owner_from_env as publish_sidecar_owner_from_env, stop_owned as stop_owned_sidecar,
};

pub(crate) const DEFAULT_BIND: &str = "127.0.0.1:47632";
pub(crate) const DEFAULT_URL: &str = "http://127.0.0.1:47632";
pub(crate) const HEALTHZ_TIMEOUT: Duration = Duration::from_millis(500);

pub(super) const SIDECAR_LOCK_TIMEOUT: Duration = Duration::from_secs(20);
const SIDECAR_START_TIMEOUT: Duration = Duration::from_secs(10);
pub(crate) const BOOTSTRAP_PROTOCOL_VERSION: u64 = 2;
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GatewayEndpoint {
    pub(crate) address: SocketAddr,
    pub(crate) url: String,
    pub(crate) instance_id: String,
}

pub(crate) struct GatewayAcquisition {
    pub(crate) endpoint: GatewayEndpoint,
    pub(crate) lease: Option<EndpointLease>,
    pub(crate) spec: GatewaySpec,
}

/// Complete launch and compatibility contract for one shared gateway.
///
/// Callers construct this once and use the same value for discovery, startup, health checks, and
/// recovery. Keeping those inputs together prevents a gateway from being started with one
/// identity and later adopted under another.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GatewaySpec {
    agent: CodingAgent,
    bind: SocketAddr,
    sidecar_args: Vec<OsString>,
    bootstrap_fingerprint: Option<String>,
    user_config_scope: bool,
}

impl GatewaySpec {
    pub(crate) fn new(agent: CodingAgent, bind: SocketAddr) -> Self {
        Self {
            agent,
            bind,
            sidecar_args: Vec::new(),
            bootstrap_fingerprint: None,
            user_config_scope: false,
        }
    }

    pub(crate) fn with_launch_args(mut self, args: Vec<OsString>) -> Self {
        self.sidecar_args = args;
        self
    }

    pub(crate) fn with_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.bootstrap_fingerprint = Some(fingerprint.into());
        self
    }

    pub(crate) fn with_user_config_scope(mut self) -> Self {
        self.user_config_scope = true;
        self
    }

    pub(crate) fn bind(&self) -> SocketAddr {
        self.bind
    }

    pub(crate) fn ensure(&self) -> Result<GatewayEndpoint, String> {
        ensure_gateway(self)
    }

    pub(crate) fn acquire(&self) -> Result<GatewayAcquisition, String> {
        acquire_gateway(self)
    }

    pub(crate) fn recover(&self, expected_instance: &str) -> Result<GatewayEndpoint, String> {
        recover_gateway(self, expected_instance)
    }

    pub(crate) fn healthy_instance(&self, url: &str) -> Option<String> {
        health::compatible_instance_id(url, self.bootstrap_fingerprint.as_deref())
    }
}

fn acquire_gateway(spec: &GatewaySpec) -> Result<GatewayAcquisition, String> {
    if spec.bind.port() == 0 {
        let endpoint = spec.ensure()?;
        let effective_spec = GatewaySpec {
            bind: endpoint.address,
            ..spec.clone()
        };
        let state = sidecar_state_dir()?;
        let lease = EndpointLease::acquire(&state, &endpoint.url)?;
        record_gateway_epoch(
            &effective_spec,
            &state,
            &endpoint.url,
            lease.fresh_epoch(),
            &endpoint,
        )?;
        return Ok(GatewayAcquisition {
            endpoint,
            lease: Some(lease),
            spec: effective_spec,
        });
    }
    let url = format!("http://{}", spec.bind);
    let state = sidecar_state_dir()?;
    let lease = EndpointLease::acquire(&state, &url)?;
    let endpoint = if lease.fresh_epoch() {
        spec.ensure()?
    } else if let Some(epoch) = read_recovery_epoch(&state, &url)? {
        spec.recover(&epoch.instance_id)?
    } else {
        spec.ensure()?
    };
    record_gateway_epoch(spec, &state, &url, lease.fresh_epoch(), &endpoint)?;
    Ok(GatewayAcquisition {
        endpoint,
        lease: Some(lease),
        spec: spec.clone(),
    })
}

fn recover_gateway(spec: &GatewaySpec, expected_instance: &str) -> Result<GatewayEndpoint, String> {
    debug_assert_ne!(spec.bind.port(), 0, "acquisition resolves automatic ports");
    let agent = spec.agent;
    let url = format!("http://{}", spec.bind);
    let runtime = runtime_dir();
    let state = sidecar_state_dir()?;
    create_private_runtime_dir(&runtime).map_err(|error| {
        sidecar_start_error(
            agent,
            &runtime,
            &format!("failed to create {}: {error}", runtime.display()),
        )
    })?;
    create_private_runtime_dir(&state).map_err(|error| {
        sidecar_start_error(
            agent,
            &runtime,
            &format!("failed to create {}: {error}", state.display()),
        )
    })?;
    let lock = lock_sidecar_endpoint(&state, &url)?;
    let (health, instance_id) =
        probe_relay_health_after_lock(&url, spec.bootstrap_fingerprint.as_deref());
    match health {
        RelayHealth::Compatible => {
            let endpoint = compatible_endpoint(spec.bind, url.clone(), instance_id)?;
            reconcile_gateway_epoch(&state, &url, false, &endpoint.instance_id)?;
            return Ok(endpoint);
        }
        RelayHealth::Incompatible => return Err(incompatible_relay_error(agent, &url)),
        RelayHealth::Foreign => return Err(foreign_listener_error(&url)),
        RelayHealth::Unavailable => {}
    }
    let mut epoch = read_recovery_epoch(&state, &url)?
        .ok_or_else(|| "shared gateway recovery epoch is missing".to_string())?;
    if epoch.instance_id != expected_instance {
        return Err(format!(
            "shared Relay gateway recovery moved from instance {expected_instance} to {}",
            epoch.instance_id
        ));
    }
    if epoch.restarts >= 1 || epoch.pending {
        return Err(
            "shared Relay gateway became unhealthy after its endpoint-scoped restart".into(),
        );
    }
    epoch.restarts = 1;
    epoch.pending = true;
    write_recovery_epoch(&state, &epoch)?;
    let endpoint = start_sidecar_bind(spec, &runtime, &state, Some(lock))
        .map_err(|error| sidecar_start_error(agent, &runtime, &error))?;
    record_gateway_epoch(spec, &state, &url, false, &endpoint)?;
    Ok(endpoint)
}

fn record_gateway_epoch(
    spec: &GatewaySpec,
    state: &Path,
    url: &str,
    fresh: bool,
    endpoint: &GatewayEndpoint,
) -> Result<(), String> {
    let _lock = lock_sidecar_endpoint(state, url)?;
    let current = spec.healthy_instance(url).ok_or_else(|| {
        "shared Relay gateway disappeared before recovery was recorded".to_string()
    })?;
    let instance_id = if current == endpoint.instance_id {
        &endpoint.instance_id
    } else {
        &current
    };
    reconcile_gateway_epoch(state, url, fresh, instance_id)
}

fn reconcile_gateway_epoch(
    state: &Path,
    url: &str,
    fresh: bool,
    instance_id: &str,
) -> Result<(), String> {
    let mut epoch = read_recovery_epoch(state, url)?;
    if fresh || epoch.is_none() {
        return write_recovery_epoch(state, &RecoveryEpoch::new(url, instance_id));
    }
    let epoch = epoch.as_mut().expect("gateway epoch is present");
    if epoch.instance_id == instance_id {
        epoch.pending = false;
        return write_recovery_epoch(state, epoch);
    }
    if epoch.pending || epoch.restarts == 0 {
        epoch.instance_id = instance_id.into();
        epoch.restarts = 1;
        epoch.pending = false;
        return write_recovery_epoch(state, epoch);
    }
    Err("shared Relay gateway was replaced again after its endpoint-scoped restart".into())
}

/// Persistent plugin gateway settings shared by MCP bootstrap and hook recovery.
pub(crate) struct PluginGatewaySpec {
    pub(crate) gateway: GatewaySpec,
    pub(crate) max_hook_payload_bytes: usize,
}

pub(crate) fn resolve_plugin_gateway(
    agent: CodingAgent,
    server_args: &ServerArgs,
    bind: SocketAddr,
) -> Result<PluginGatewaySpec, CliError> {
    let mut persistent_args = server_args.clone();
    persistent_args.bind = Some(bind);
    let resolved = resolve_persistent_server_config(&persistent_args)?;
    let bootstrap_fingerprint = resolved
        .bootstrap_fingerprint
        .expect("persistent gateway resolution sets a bootstrap fingerprint");
    let max_hook_payload_bytes = resolved.gateway.max_hook_payload_bytes;
    let sidecar_args = [
        ("--openai-base-url", resolved.gateway.openai_base_url),
        ("--anthropic-base-url", resolved.gateway.anthropic_base_url),
        (
            "--max-hook-payload-bytes",
            resolved.gateway.max_hook_payload_bytes.to_string(),
        ),
        (
            "--max-passthrough-body-bytes",
            resolved.gateway.max_passthrough_body_bytes.to_string(),
        ),
    ]
    .into_iter()
    .flat_map(|(flag, value)| [OsString::from(flag), OsString::from(value)])
    .collect();
    Ok(PluginGatewaySpec {
        gateway: GatewaySpec::new(agent, bind)
            .with_launch_args(sidecar_args)
            .with_fingerprint(bootstrap_fingerprint)
            .with_user_config_scope(),
        max_hook_payload_bytes,
    })
}

#[cfg(test)]
pub(crate) fn ensure_sidecar_bind(
    agent: CodingAgent,
    bind: SocketAddr,
) -> Result<GatewayEndpoint, String> {
    GatewaySpec::new(agent, bind).ensure()
}

fn ensure_gateway(spec: &GatewaySpec) -> Result<GatewayEndpoint, String> {
    let agent = spec.agent;
    let bind = spec.bind;
    let bootstrap_fingerprint = spec.bootstrap_fingerprint.as_deref();
    if !bind.ip().is_loopback() {
        return Err(format!(
            "plugin sidecars require a loopback bind address, got {bind}"
        ));
    }
    let url = format!("http://{bind}");
    let runtime = runtime_dir();
    create_private_runtime_dir(&runtime).map_err(|error| {
        sidecar_start_error(
            agent,
            &runtime,
            &format!("failed to create {}: {error}", runtime.display()),
        )
    })?;
    let state = sidecar_state_dir()?;
    create_private_runtime_dir(&state).map_err(|error| {
        sidecar_start_error(
            agent,
            &runtime,
            &format!("failed to create {}: {error}", state.display()),
        )
    })?;
    if bind.port() == 0 {
        return start_sidecar_bind(spec, &runtime, &state, None)
            .map_err(|error| sidecar_start_error(agent, &runtime, &error));
    }
    let lock_path = sidecar_lock_path(&state, &url);
    let (initial_health, initial_instance) =
        probe_relay_health_with_instance(&url, bootstrap_fingerprint);
    match initial_health {
        RelayHealth::Compatible => {
            return compatible_endpoint(bind, url, initial_instance);
        }
        RelayHealth::Incompatible | RelayHealth::Foreign | RelayHealth::Unavailable => {}
    }
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|error| {
            sidecar_start_error(
                agent,
                &runtime,
                &format!(
                    "failed to open sidecar lock {}: {error}",
                    lock_path.display()
                ),
            )
        })?;
    let lock_deadline = Instant::now() + SIDECAR_LOCK_TIMEOUT;
    loop {
        match try_lock_exclusive(&lock) {
            Ok(LockAttempt::Acquired) => break,
            Ok(LockAttempt::Contended) => {
                if Instant::now() >= lock_deadline {
                    return Err(sidecar_start_error(
                        agent,
                        &runtime,
                        "sidecar lock timed out",
                    ));
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(error) => {
                return Err(sidecar_start_error(
                    agent,
                    &runtime,
                    &format!("failed to acquire sidecar lock: {error}"),
                ));
            }
        }
    }
    let (health, instance_id) = probe_relay_health_after_lock(&url, bootstrap_fingerprint);
    match health {
        RelayHealth::Compatible => {
            return compatible_endpoint(bind, url, instance_id);
        }
        RelayHealth::Incompatible => return Err(incompatible_relay_error(agent, &url)),
        RelayHealth::Foreign => return Err(foreign_listener_error(&url)),
        RelayHealth::Unavailable => {}
    }
    let result = start_sidecar_bind(spec, &runtime, &state, Some(lock));
    result.map_err(|error| sidecar_start_error(agent, &runtime, &error))
}

fn compatible_endpoint(
    address: SocketAddr,
    url: String,
    instance_id: Option<String>,
) -> Result<GatewayEndpoint, String> {
    let instance_id = instance_id.ok_or_else(|| foreign_listener_error(&url))?;
    Ok(GatewayEndpoint {
        address,
        url,
        instance_id,
    })
}

fn foreign_listener_error(url: &str) -> String {
    format!(
        "{url} is occupied by a service that is not a compatible NeMo Relay gateway; stop that service or configure another port"
    )
}

fn incompatible_relay_error(agent: CodingAgent, url: &str) -> String {
    let remediation = match agent {
        CodingAgent::Codex => "run `nemo-relay install codex --force`",
        CodingAgent::ClaudeCode => "run `nemo-relay install claude-code --force`",
        CodingAgent::Hermes => "run `nemo-relay install hermes --force`",
    };
    format!(
        "{url} is occupied by NeMo Relay with a different version or persistent configuration; stop it, wait for its idle shutdown, or {remediation} before retrying"
    )
}

fn sidecar_start_error(agent: CodingAgent, runtime: &Path, error: &str) -> String {
    let log_path = runtime.join(format!("{}-sidecar.log", agent.as_arg()));
    format!("{error}; inspect {}", log_path.display())
}

#[cfg(all(test, unix))]
pub(super) fn start_sidecar(agent: CodingAgent, url: &str, runtime: &Path) -> Result<(), String> {
    let spec = GatewaySpec::new(agent, loopback_bind(url)?);
    start_sidecar_bind(&spec, runtime, runtime, None).map(|_| ())
}

struct ArmedSidecarChild {
    process: Option<DetachedSidecarProcess>,
    startup_pid_path: PathBuf,
    state: PathBuf,
    pid: u32,
}

impl ArmedSidecarChild {
    fn new(
        child: Child,
        startup_pid_path: PathBuf,
        state: &Path,
        #[cfg(windows)] prepared_job: Option<SidecarJob>,
    ) -> Self {
        let pid = child.id();
        Self {
            process: Some(DetachedSidecarProcess::new(
                child,
                #[cfg(windows)]
                prepared_job,
            )),
            startup_pid_path,
            state: state.to_path_buf(),
            pid,
        }
    }

    fn id(&self) -> u32 {
        self.pid
    }

    fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.process
            .as_mut()
            .expect("armed sidecar process is present")
            .try_wait()
    }

    fn disarm(mut self) -> DetachedSidecarProcess {
        self.process
            .take()
            .expect("armed sidecar process is present")
    }
}

impl Drop for ArmedSidecarChild {
    fn drop(&mut self) {
        if let Some(mut process) = self.process.take() {
            // The launcher may exit before readiness while leaving descendants behind. Always
            // target the detached process group even when the direct child has already exited.
            process.terminate();
            cleanup_sidecar_records_for_pid(&self.state, self.pid);
        }
        let _ = fs::remove_file(&self.startup_pid_path);
    }
}

fn cleanup_sidecar_records_for_pid(runtime: &Path, pid: u32) {
    let Ok(paths) = sidecar_owner_paths(runtime) else {
        return;
    };
    for owner_path in paths {
        let Ok(Some(owner)) = read_owner_record(&owner_path) else {
            continue;
        };
        if owner.pid != pid {
            continue;
        }
        let _ = fs::remove_file(&owner_path);
        let _ = fs::remove_file(sidecar_owner_pid_path(runtime, &owner_path, &owner.url));
    }
}

fn start_sidecar_bind(
    spec: &GatewaySpec,
    runtime: &Path,
    state: &Path,
    mut startup_lock: Option<fs::File>,
) -> Result<GatewayEndpoint, String> {
    let agent = spec.agent;
    let bind = spec.bind;
    let sidecar_args = &spec.sidecar_args;
    let bootstrap_fingerprint = spec.bootstrap_fingerprint.as_deref();
    let requested_url = format!("http://{bind}");
    if bind.port() != 0 {
        let (health, instance_id) =
            probe_relay_health_with_instance(&requested_url, bootstrap_fingerprint);
        match health {
            RelayHealth::Compatible => {
                return compatible_endpoint(bind, requested_url, instance_id);
            }
            RelayHealth::Incompatible => {
                return Err(incompatible_relay_error(agent, &requested_url));
            }
            RelayHealth::Foreign => return Err(foreign_listener_error(&requested_url)),
            RelayHealth::Unavailable => {}
        }
    }
    let relay = relay_binary()?;
    let log_path = runtime.join(format!("{}-sidecar.log", agent.as_arg()));
    let ready_path = runtime.join(format!(
        "{}-sidecar-{}-{}.ready.json",
        agent.as_arg(),
        std::process::id(),
        uuid::Uuid::now_v7()
    ));
    let _ = fs::remove_file(&ready_path);
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|error| format!("failed to open {}: {error}", log_path.display()))?;
    let err_log = log
        .try_clone()
        .map_err(|error| format!("failed to clone sidecar log handle: {error}"))?;
    let idle_timeout = plugin_idle_timeout()?;
    let shutdown_token = uuid::Uuid::now_v7().to_string();
    #[cfg(windows)]
    let sidecar_job = SidecarJob::create()?;
    let mut command = Command::new(relay);
    command
        .arg("--bind")
        .arg(bind.to_string())
        .arg("--ready-file")
        .arg(&ready_path)
        .args(sidecar_args)
        .env(
            "NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS",
            idle_timeout.as_secs().to_string(),
        )
        .env(
            crate::config::BOOTSTRAP_FINGERPRINT_ENV,
            bootstrap_fingerprint.unwrap_or_default(),
        )
        .env(BOOTSTRAP_STATE_DIR_ENV, state)
        .env("NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN", &shutdown_token)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(err_log));
    #[cfg(windows)]
    sidecar_job.configure_child(&mut command);
    if spec.user_config_scope {
        command.env("NEMO_RELAY_CONFIG_SCOPE", "user");
        if let Some(config_dir) = crate::config::user_config_dir() {
            fs::create_dir_all(&config_dir).map_err(|error| {
                format!(
                    "failed to create plugin sidecar working directory {}: {error}",
                    config_dir.display()
                )
            })?;
            command.current_dir(config_dir);
        }
    }
    configure_detached_sidecar(&mut command);
    let child = command
        .spawn()
        .map_err(|error| format!("failed to spawn nemo-relay sidecar: {error}"))?;
    let startup_pid_path = ready_path.with_extension("pid");
    let _ = fs::write(&startup_pid_path, child.id().to_string());
    let mut child = ArmedSidecarChild::new(
        child,
        startup_pid_path,
        state,
        #[cfg(windows)]
        Some(sidecar_job),
    );
    let deadline = Instant::now() + SIDECAR_START_TIMEOUT;
    while Instant::now() < deadline {
        match read_sidecar_ready_file(&ready_path) {
            Ok(Some(endpoint))
                if (bind.port() == 0 || endpoint.address == bind)
                    && health::compatible_instance_id(&endpoint.url, bootstrap_fingerprint)
                        .as_deref()
                        == Some(endpoint.instance_id.as_str()) =>
            {
                let ownership_lock = match startup_lock.take() {
                    Some(lock) => lock,
                    None => lock_sidecar_endpoint(state, &endpoint.url)?,
                };
                let owner_path = sidecar_owner_path(state, &endpoint.url);
                let pid_path = sidecar_pid_path(state, &endpoint.url);
                let pid = child.id();
                validate_sidecar_owner(
                    &owner_path,
                    &pid_path,
                    pid,
                    &endpoint.url,
                    &shutdown_token,
                    bootstrap_fingerprint,
                )?;
                if let Err(error) = handoff_detached_sidecar_to_reaper(
                    child.disarm(),
                    owner_path.clone(),
                    pid_path.clone(),
                    sidecar_lock_path(state, &endpoint.url),
                ) {
                    let _ = fs::remove_file(&owner_path);
                    let _ = fs::remove_file(&pid_path);
                    let _ = fs::remove_file(&ready_path);
                    return Err(error);
                }
                drop(ownership_lock);
                let _ = fs::remove_file(&ready_path);
                return Ok(endpoint);
            }
            Ok(_) => {}
            Err(error) => {
                let _ = fs::remove_file(&ready_path);
                return Err(error);
            }
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                let _ = fs::remove_file(&ready_path);
                let (health, instance_id) =
                    probe_relay_health_with_instance(&requested_url, bootstrap_fingerprint);
                if bind.port() != 0 && health == RelayHealth::Compatible {
                    return compatible_endpoint(bind, requested_url, instance_id);
                }
                return Err(format!(
                    "nemo-relay sidecar exited before becoming ready at {requested_url}: {status}"
                ));
            }
            Ok(None) => {}
            Err(error) => {
                let _ = fs::remove_file(&ready_path);
                return Err(format!(
                    "failed to inspect nemo-relay sidecar process: {error}"
                ));
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
    let _ = fs::remove_file(&ready_path);
    Err(format!(
        "nemo-relay sidecar did not become ready at {requested_url}; terminated startup process"
    ))
}

struct SidecarReapRequest {
    process: DetachedSidecarProcess,
    exited: bool,
    owner_path: PathBuf,
    pid_path: PathBuf,
    lock_path: PathBuf,
}

static SIDECAR_REAPER: OnceLock<Result<mpsc::Sender<SidecarReapRequest>, String>> = OnceLock::new();

fn sidecar_reaper_sender() -> Result<&'static mpsc::Sender<SidecarReapRequest>, String> {
    match SIDECAR_REAPER.get_or_init(|| {
        let (sender, receiver) = mpsc::channel();
        thread::Builder::new()
            .name("nemo-relay-sidecar-reaper".into())
            .spawn(move || run_sidecar_reaper(receiver))
            .map(|_| sender)
            .map_err(|error| format!("failed to start nemo-relay sidecar reaper: {error}"))
    }) {
        Ok(sender) => Ok(sender),
        Err(error) => Err(error.clone()),
    }
}

#[cfg(test)]
pub(super) fn handoff_sidecar_to_reaper(
    child: Child,
    owner_path: PathBuf,
    pid_path: PathBuf,
    lock_path: PathBuf,
) -> Result<(), String> {
    let process = DetachedSidecarProcess::new(
        child,
        #[cfg(windows)]
        None,
    );
    handoff_detached_sidecar_to_reaper(process, owner_path, pid_path, lock_path)
}

fn handoff_detached_sidecar_to_reaper(
    process: DetachedSidecarProcess,
    owner_path: PathBuf,
    pid_path: PathBuf,
    lock_path: PathBuf,
) -> Result<(), String> {
    let sender = match sidecar_reaper_sender() {
        Ok(sender) => sender,
        Err(error) => {
            return terminate_reaper_handoff(process, &pid_path, error);
        }
    };
    let request = SidecarReapRequest {
        process,
        exited: false,
        owner_path,
        pid_path,
        lock_path,
    };
    match sender.send(request) {
        Ok(()) => Ok(()),
        Err(error) => terminate_reaper_handoff(
            error.0.process,
            &error.0.pid_path,
            "nemo-relay sidecar reaper stopped unexpectedly".into(),
        ),
    }
}

fn terminate_reaper_handoff(
    mut process: DetachedSidecarProcess,
    pid_path: &Path,
    error: String,
) -> Result<(), String> {
    process.terminate();
    let _ = fs::remove_file(pid_path);
    Err(error)
}

fn run_sidecar_reaper(receiver: mpsc::Receiver<SidecarReapRequest>) {
    let mut children = Vec::new();
    loop {
        match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(request) => children.push(request),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) if children.is_empty() => return,
            Err(mpsc::RecvTimeoutError::Disconnected) => {}
        }
        while let Ok(request) = receiver.try_recv() {
            children.push(request);
        }
        let mut index = 0;
        while index < children.len() {
            if !children[index].exited {
                match children[index].process.try_wait() {
                    Ok(Some(_)) => children[index].exited = true,
                    Ok(None) => {
                        index += 1;
                        continue;
                    }
                    Err(error) => {
                        eprintln!("failed to inspect nemo-relay sidecar process: {error}");
                        index += 1;
                        continue;
                    }
                }
            }
            children[index].process.terminate_retained_descendants();
            if cleanup_reaped_sidecar(&children[index]) {
                let request = children.swap_remove(index);
                drop(request);
            } else {
                index += 1;
            }
        }
    }
}

fn cleanup_reaped_sidecar(request: &SidecarReapRequest) -> bool {
    let lock = match open_sidecar_lock(&request.lock_path) {
        Ok(lock) => lock,
        Err(error) => {
            eprintln!("failed to clean up nemo-relay sidecar ownership: {error}");
            return true;
        }
    };
    match try_lock_exclusive(&lock) {
        Ok(LockAttempt::Acquired) => {}
        Ok(LockAttempt::Contended) => return false,
        Err(error) => {
            eprintln!("failed to lock nemo-relay sidecar ownership for cleanup: {error}");
            return true;
        }
    }
    let pid = request.process.id();
    if read_owner_record(&request.owner_path)
        .ok()
        .flatten()
        .is_some_and(|owner| owner.pid == pid)
    {
        let _ = fs::remove_file(&request.owner_path);
    }
    if fs::read_to_string(&request.pid_path)
        .ok()
        .is_some_and(|value| value.trim() == pid.to_string())
    {
        let _ = fs::remove_file(&request.pid_path);
    }
    true
}

pub(super) fn relay_binary() -> Result<PathBuf, String> {
    if let Ok(path) = env::var("NEMO_RELAY_PLUGIN_BINARY") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
        return Err(format!(
            "NEMO_RELAY_PLUGIN_BINARY does not exist: {}",
            path.display()
        ));
    }
    current_exe()
}

pub(super) fn current_exe() -> Result<PathBuf, String> {
    env::current_exe().map_err(|error| format!("failed to resolve current executable: {error}"))
}

pub(crate) fn plugin_idle_timeout() -> Result<Duration, String> {
    let raw = env::var(crate::config::PLUGIN_IDLE_TIMEOUT_ENV).unwrap_or_else(|_| "300".into());
    let seconds = raw.parse::<u64>().map_err(|error| {
        format!(
            "{} must be a positive integer: {error}",
            crate::config::PLUGIN_IDLE_TIMEOUT_ENV
        )
    })?;
    if seconds == 0 {
        return Err(format!(
            "{} must be greater than 0",
            crate::config::PLUGIN_IDLE_TIMEOUT_ENV
        ));
    }
    Ok(Duration::from_secs(seconds))
}

pub(crate) fn plugin_heartbeat_interval() -> Result<Duration, String> {
    Ok((plugin_idle_timeout()? / 3).clamp(Duration::from_millis(100), Duration::from_secs(30)))
}

#[cfg(test)]
#[path = "../tests/coverage/sidecar_tests.rs"]
mod tests;
