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
    RelayHealth, probe as probe_relay_health, probe_after_lock as probe_relay_health_after_lock,
};
pub(crate) use health::{healthz, loopback_authority, loopback_bind, parse_loopback_url};
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
pub(crate) use state::{BOOTSTRAP_AGENT_ENV, BOOTSTRAP_STATE_DIR_ENV};
use state::{
    create_private_runtime_dir, open_lock as open_sidecar_lock, read_owner_record,
    read_ready_file as read_sidecar_ready_file, runtime_dir,
};
pub(crate) use state::{
    lock_endpoint as lock_sidecar_endpoint, lock_path as sidecar_lock_path,
    owner_path as sidecar_owner_path, owner_paths as sidecar_owner_paths,
    pid_path as sidecar_pid_path, state_dir as sidecar_state_dir,
    validate_owner as validate_sidecar_owner,
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
pub(crate) const BOOTSTRAP_PROTOCOL_VERSION: u64 = 1;
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GatewayEndpoint {
    pub(crate) address: SocketAddr,
    pub(crate) url: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GatewayBootstrap {
    pub(crate) endpoint: GatewayEndpoint,
    pub(crate) started: bool,
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

    pub(crate) fn ensure(&self) -> Result<GatewayBootstrap, String> {
        ensure_gateway(self)
    }

    pub(crate) fn is_healthy(&self, url: &str) -> bool {
        probe_relay_health(url, self.bootstrap_fingerprint.as_deref()) == RelayHealth::Compatible
    }
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
    GatewaySpec::new(agent, bind)
        .ensure()
        .map(|bootstrap| bootstrap.endpoint)
}

fn ensure_gateway(spec: &GatewaySpec) -> Result<GatewayBootstrap, String> {
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
            &url,
            &runtime,
            &format!("failed to create {}: {error}", runtime.display()),
        )
    })?;
    let state = sidecar_state_dir()?;
    create_private_runtime_dir(&state).map_err(|error| {
        sidecar_start_error(
            agent,
            &url,
            &runtime,
            &format!("failed to create {}: {error}", state.display()),
        )
    })?;
    if bind.port() == 0 {
        return start_sidecar_bind(spec, &runtime, &state, None)
            .map_err(|error| sidecar_start_error(agent, &url, &runtime, &error));
    }
    let lock_path = sidecar_lock_path(&state, &url);
    let initial_health = probe_relay_health(&url, bootstrap_fingerprint);
    match initial_health {
        RelayHealth::Compatible => {
            return Ok(GatewayBootstrap {
                endpoint: GatewayEndpoint { address: bind, url },
                started: false,
            });
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
                &url,
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
                if probe_relay_health(&url, bootstrap_fingerprint) == RelayHealth::Compatible {
                    return Ok(GatewayBootstrap {
                        endpoint: GatewayEndpoint { address: bind, url },
                        started: false,
                    });
                }
                if Instant::now() >= lock_deadline {
                    return Err(sidecar_start_error(
                        agent,
                        &url,
                        &runtime,
                        "sidecar lock timed out",
                    ));
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(error) => {
                return Err(sidecar_start_error(
                    agent,
                    &url,
                    &runtime,
                    &format!("failed to acquire sidecar lock: {error}"),
                ));
            }
        }
    }
    match probe_relay_health_after_lock(&url, bootstrap_fingerprint) {
        RelayHealth::Compatible => {
            return Ok(GatewayBootstrap {
                endpoint: GatewayEndpoint { address: bind, url },
                started: false,
            });
        }
        RelayHealth::Incompatible => return Err(incompatible_relay_error(agent, &url)),
        RelayHealth::Foreign => return Err(foreign_listener_error(&url)),
        RelayHealth::Unavailable => {}
    }
    let result = start_sidecar_bind(spec, &runtime, &state, Some(lock));
    result.map_err(|error| sidecar_start_error(agent, &url, &runtime, &error))
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

fn sidecar_start_error(agent: CodingAgent, url: &str, runtime: &Path, error: &str) -> String {
    let log_path = runtime.join(format!("{}-sidecar.log", agent.as_arg()));
    let manual = parse_loopback_url(url)
        .map(|(host, port)| format!("nemo-relay --bind {}", loopback_authority(&host, port)))
        .unwrap_or_else(|_| "nemo-relay --bind 127.0.0.1:47632".into());
    format!(
        "{error}; inspect {}; or start the gateway manually with `{manual}`",
        log_path.display()
    )
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
    agent: CodingAgent,
    pid: u32,
}

impl ArmedSidecarChild {
    fn new(
        child: Child,
        startup_pid_path: PathBuf,
        state: &Path,
        agent: CodingAgent,
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
            agent,
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
            cleanup_sidecar_records_for_pid(&self.state, self.agent, self.pid);
        }
        let _ = fs::remove_file(&self.startup_pid_path);
    }
}

fn cleanup_sidecar_records_for_pid(runtime: &Path, agent: CodingAgent, pid: u32) {
    let Ok(paths) = sidecar_owner_paths(runtime, agent) else {
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
        let _ = fs::remove_file(sidecar_pid_path(runtime, agent, &owner.url));
    }
}

fn start_sidecar_bind(
    spec: &GatewaySpec,
    runtime: &Path,
    state: &Path,
    mut startup_lock: Option<fs::File>,
) -> Result<GatewayBootstrap, String> {
    let agent = spec.agent;
    let bind = spec.bind;
    let sidecar_args = &spec.sidecar_args;
    let bootstrap_fingerprint = spec.bootstrap_fingerprint.as_deref();
    let requested_url = format!("http://{bind}");
    if bind.port() != 0 {
        match probe_relay_health(&requested_url, bootstrap_fingerprint) {
            RelayHealth::Compatible => {
                return Ok(GatewayBootstrap {
                    endpoint: GatewayEndpoint {
                        address: bind,
                        url: requested_url,
                    },
                    started: false,
                });
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
        .env(BOOTSTRAP_AGENT_ENV, agent.as_arg())
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
        agent,
        #[cfg(windows)]
        Some(sidecar_job),
    );
    let deadline = Instant::now() + SIDECAR_START_TIMEOUT;
    while Instant::now() < deadline {
        match read_sidecar_ready_file(&ready_path) {
            Ok(Some(endpoint))
                if (bind.port() == 0 || endpoint.address == bind)
                    && probe_relay_health(&endpoint.url, bootstrap_fingerprint)
                        == RelayHealth::Compatible =>
            {
                let ownership_lock = match startup_lock.take() {
                    Some(lock) => lock,
                    None => lock_sidecar_endpoint(state, &endpoint.url)?,
                };
                let owner_path = sidecar_owner_path(state, agent, &endpoint.url);
                let pid_path = sidecar_pid_path(state, agent, &endpoint.url);
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
                return Ok(GatewayBootstrap {
                    endpoint,
                    started: true,
                });
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
                if bind.port() != 0
                    && probe_relay_health(&requested_url, bootstrap_fingerprint)
                        == RelayHealth::Compatible
                {
                    return Ok(GatewayBootstrap {
                        endpoint: GatewayEndpoint {
                            address: bind,
                            url: requested_url,
                        },
                        started: false,
                    });
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
