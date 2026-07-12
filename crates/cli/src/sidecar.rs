// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin bootstrap coordinator for the existing Relay gateway server.

mod health;
mod process;
mod state;

use std::env;
use std::ffi::OsString;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::config::{ServerArgs, resolve_persistent_server_config};
use crate::error::CliError;
use health::{RelayHealth, probe_with_instance as probe_relay_health_with_instance};
pub(crate) use health::{
    VerifiedHttpError, VerifiedHttpResponse, authenticated_instance_id, healthz,
    healthz_compatible, loopback_bind,
};
#[cfg(test)]
pub(crate) use process::{
    WINDOWS_CREATE_BREAKAWAY_FROM_JOB, WINDOWS_CREATE_NEW_PROCESS_GROUP, WINDOWS_CREATE_NO_WINDOW,
    WINDOWS_JOB_OBJECT_LIMIT_BREAKAWAY_OK, WINDOWS_JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK,
    windows_creation_flags as windows_sidecar_creation_flags,
};
pub(crate) use state::BOOTSTRAP_STATE_DIR_ENV;
#[cfg(test)]
pub(crate) use state::lock_name as sidecar_lock_name;
pub(crate) use state::{
    publish_owner_from_env as publish_sidecar_owner_from_env, state_dir as sidecar_state_dir,
    stop_owned_and_reset as stop_owned_sidecar_and_reset,
};

pub(crate) const DEFAULT_BIND: &str = "127.0.0.1:47632";
pub(crate) const DEFAULT_URL: &str = "http://127.0.0.1:47632";
pub(crate) const HEALTHZ_TIMEOUT: Duration = Duration::from_millis(500);
pub(crate) const BOOTSTRAP_PROTOCOL_VERSION: u64 = 2;

pub(super) const SIDECAR_LOCK_TIMEOUT: Duration = Duration::from_secs(20);
const SIDECAR_START_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GatewayEndpoint {
    pub(crate) address: SocketAddr,
    pub(crate) url: String,
    pub(crate) instance_id: String,
}

/// Inputs required to identify and, when absent, start one persistent gateway.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GatewaySpec {
    bind: SocketAddr,
    launch_args: Vec<OsString>,
    bootstrap_fingerprint: Option<String>,
    user_config_scope: bool,
}

impl GatewaySpec {
    pub(crate) fn new(bind: SocketAddr) -> Self {
        Self {
            bind,
            launch_args: Vec::new(),
            bootstrap_fingerprint: None,
            user_config_scope: false,
        }
    }

    pub(crate) fn with_launch_args(mut self, args: Vec<OsString>) -> Self {
        self.launch_args = args;
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

    /// Return the compatible gateway already bound at this endpoint, or start the existing server
    /// under a per-user lock and wait for authenticated readiness.
    pub(crate) fn acquire(&self) -> Result<GatewayEndpoint, String> {
        acquire_gateway(self)
    }

    pub(crate) fn recover(&self, expected_instance: &str) -> Result<GatewayEndpoint, String> {
        recover_gateway(self, expected_instance)
    }

    pub(crate) fn healthy_instance(&self, url: &str) -> Option<String> {
        health::compatible_instance_id(url, self.bootstrap_fingerprint.as_deref())
    }

    pub(crate) fn existing_healthy_instance(&self, url: &str) -> Result<Option<String>, String> {
        match probe_relay_health_with_instance(url, self.bootstrap_fingerprint.as_deref()) {
            (RelayHealth::Compatible, instance_id) => Ok(instance_id),
            (RelayHealth::Unavailable, _) => Ok(None),
            (RelayHealth::Incompatible, _) => Err(incompatible_relay_error(url)),
            (RelayHealth::Foreign, _) => Err(foreign_listener_error(url)),
        }
    }

    pub(crate) fn post_verified(
        &self,
        url: &str,
        path: &str,
        headers: &[(String, String)],
        body: &[u8],
        timeout: Duration,
        max_response_bytes: usize,
    ) -> Result<VerifiedHttpResponse, VerifiedHttpError> {
        let Some(fingerprint) = self.bootstrap_fingerprint.as_deref() else {
            return Err(VerifiedHttpError::missing_fingerprint());
        };
        health::post_verified(
            url,
            fingerprint,
            path,
            headers,
            body,
            timeout,
            max_response_bytes,
        )
    }
}

fn acquire_gateway(spec: &GatewaySpec) -> Result<GatewayEndpoint, String> {
    if !spec.bind.ip().is_loopback() {
        return Err(format!(
            "plugin gateways require a loopback bind address, got {}",
            spec.bind
        ));
    }
    let url = format!("http://{}", spec.bind);
    if spec.bind.port() != 0 {
        match probe_relay_health_with_instance(&url, spec.bootstrap_fingerprint.as_deref()) {
            (RelayHealth::Compatible, instance_id) => {
                return compatible_endpoint(spec.bind, url, instance_id);
            }
            (RelayHealth::Incompatible, _) => return Err(incompatible_relay_error(&url)),
            // A gateway may already be binding while another MCP process owns the
            // startup lock. Serialize before deciding whether either state is a
            // genuine conflict.
            (RelayHealth::Foreign | RelayHealth::Unavailable, _) => {}
        }
    }

    let state = sidecar_state_dir()?;
    state::create_private_dir(&state)?;
    let _startup_lock = state::lock_endpoint(&state, &url)?;
    if spec.bind.port() == 0 {
        return start_gateway(spec, &state);
    }
    match probe_relay_health_with_instance(&url, spec.bootstrap_fingerprint.as_deref()) {
        (RelayHealth::Compatible, instance_id) => compatible_endpoint(spec.bind, url, instance_id),
        (RelayHealth::Incompatible, _) => Err(incompatible_relay_error(&url)),
        (RelayHealth::Foreign, _) => Err(foreign_listener_error(&url)),
        (RelayHealth::Unavailable, _) => start_gateway(spec, &state),
    }
}

fn recover_gateway(spec: &GatewaySpec, expected_instance: &str) -> Result<GatewayEndpoint, String> {
    let requested_url = format!("http://{}", spec.bind);
    let state = sidecar_state_dir()?;
    state::create_private_dir(&state)?;
    let _startup_lock = state::lock_endpoint(&state, &requested_url)?;

    if spec.bind.port() != 0 {
        match probe_relay_health_with_instance(
            &requested_url,
            spec.bootstrap_fingerprint.as_deref(),
        ) {
            (RelayHealth::Compatible, instance_id) => {
                return compatible_endpoint(spec.bind, requested_url, instance_id);
            }
            (RelayHealth::Incompatible, _) => return Err(incompatible_relay_error(&requested_url)),
            (RelayHealth::Foreign, _) => return Err(foreign_listener_error(&requested_url)),
            (RelayHealth::Unavailable, _) => {}
        }
    }

    if let Some(previous) = state::read_recovery(&state, &requested_url)?
        && previous.from_instance == expected_instance
    {
        if !previous.endpoint_url.is_empty()
            && !previous.to_instance.is_empty()
            && spec.healthy_instance(&previous.endpoint_url).as_deref()
                == Some(previous.to_instance.as_str())
        {
            let address = loopback_bind(&previous.endpoint_url)?;
            return compatible_endpoint(address, previous.endpoint_url, Some(previous.to_instance));
        }
        return Err("shared Relay gateway became unhealthy after its coordinated restart".into());
    }

    // Record the attempt while holding the startup lock. If the replacement
    // dies before readiness, another overlapping MCP must not start a second
    // replacement.
    state::write_recovery(
        &state,
        &requested_url,
        &state::RecoveryRecord {
            from_instance: expected_instance.into(),
            endpoint_url: String::new(),
            to_instance: String::new(),
        },
    )?;
    let endpoint = start_gateway(spec, &state)?;
    state::write_recovery(
        &state,
        &requested_url,
        &state::RecoveryRecord {
            from_instance: expected_instance.into(),
            endpoint_url: endpoint.url.clone(),
            to_instance: endpoint.instance_id.clone(),
        },
    )?;
    Ok(endpoint)
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

fn incompatible_relay_error(url: &str) -> String {
    format!(
        "{url} is occupied by NeMo Relay with a different version or persistent configuration; stop it, wait for idle shutdown, or reinstall the integration with --force"
    )
}

fn start_gateway(spec: &GatewaySpec, state: &Path) -> Result<GatewayEndpoint, String> {
    let relay = relay_binary()?;
    let ready_path = state.join(format!(
        "gateway-{}-{}.ready.json",
        std::process::id(),
        uuid::Uuid::now_v7()
    ));
    let _ = fs::remove_file(&ready_path);
    let shutdown_token = uuid::Uuid::now_v7().to_string();
    let mut command = Command::new(relay);
    command
        .arg("--bind")
        .arg(spec.bind.to_string())
        .arg("--ready-file")
        .arg(&ready_path)
        .args(&spec.launch_args)
        .env(
            crate::config::PLUGIN_IDLE_TIMEOUT_ENV,
            plugin_idle_timeout()?.as_secs().to_string(),
        )
        .env(
            crate::config::BOOTSTRAP_FINGERPRINT_ENV,
            spec.bootstrap_fingerprint.as_deref().unwrap_or_default(),
        )
        .env(BOOTSTRAP_STATE_DIR_ENV, state)
        .env("NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN", &shutdown_token)
        .env_remove(crate::install_generation::GENERATION_FILE_ENV)
        .env_remove(crate::install_generation::GENERATION_TOKEN_ENV)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if spec.user_config_scope {
        command.env("NEMO_RELAY_CONFIG_SCOPE", "user");
        if let Some(config_dir) = crate::config::user_config_dir() {
            fs::create_dir_all(&config_dir).map_err(|error| {
                format!(
                    "failed to create gateway working directory {}: {error}",
                    config_dir.display()
                )
            })?;
            command.current_dir(config_dir);
        }
    }
    process::configure_detached(&mut command);
    let child = process::spawn_detached(&mut command)
        .map_err(|error| format!("failed to spawn nemo-relay gateway: {error}"))?;
    let mut child = ArmedChild::new(child);
    let deadline = Instant::now() + SIDECAR_START_TIMEOUT;
    while Instant::now() < deadline {
        if let Some(endpoint) = read_ready_file(&ready_path)?
            && (endpoint.address == spec.bind
                || (spec.bind.port() == 0 && endpoint.address.ip() == spec.bind.ip()))
            && spec.healthy_instance(&endpoint.url).as_deref()
                == Some(endpoint.instance_id.as_str())
        {
            let mut detached = child.disarm();
            let _ = thread::Builder::new()
                .name("nemo-relay-gateway-wait".into())
                .spawn(move || {
                    let _ = detached.wait();
                });
            let _ = fs::remove_file(&ready_path);
            return Ok(endpoint);
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                let _ = fs::remove_file(&ready_path);
                return Err(format!(
                    "nemo-relay gateway exited before becoming ready at http://{}: {status}",
                    spec.bind
                ));
            }
            Ok(None) => {}
            Err(error) => {
                let _ = fs::remove_file(&ready_path);
                return Err(format!(
                    "failed to inspect nemo-relay gateway process: {error}"
                ));
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
    let _ = fs::remove_file(&ready_path);
    Err(format!(
        "nemo-relay gateway did not become ready at http://{}",
        spec.bind
    ))
}

struct ArmedChild(Option<Child>);

impl ArmedChild {
    fn new(child: Child) -> Self {
        Self(Some(child))
    }

    fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.0
            .as_mut()
            .expect("armed gateway child is present")
            .try_wait()
    }

    fn disarm(mut self) -> Child {
        self.0.take().expect("armed gateway child is present")
    }
}

impl Drop for ArmedChild {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            process::terminate_tree(&mut child);
        }
    }
}

#[derive(Deserialize)]
struct ReadyRecord {
    service: String,
    version: String,
    bootstrap_protocol: u64,
    address: SocketAddr,
    instance_id: String,
}

fn read_ready_file(path: &Path) -> Result<Option<GatewayEndpoint>, String> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(format!(
                "failed to read gateway readiness {}: {error}",
                path.display()
            ));
        }
    };
    let record = serde_json::from_slice::<ReadyRecord>(&bytes).map_err(|error| {
        format!(
            "failed to parse gateway readiness {}: {error}",
            path.display()
        )
    })?;
    if record.service != "nemo-relay"
        || record.version != env!("CARGO_PKG_VERSION")
        || record.bootstrap_protocol != BOOTSTRAP_PROTOCOL_VERSION
        || record.instance_id.is_empty()
    {
        return Err(format!(
            "gateway readiness {} has an incompatible identity",
            path.display()
        ));
    }
    Ok(Some(GatewayEndpoint {
        address: record.address,
        url: format!("http://{}", record.address),
        instance_id: record.instance_id,
    }))
}

/// Persistent plugin settings shared by MCP bootstrap and forward-only hooks.
pub(crate) struct PluginGatewaySpec {
    pub(crate) gateway: GatewaySpec,
    pub(crate) max_hook_payload_bytes: usize,
}

pub(crate) fn resolve_plugin_gateway(
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
    let launch_args = [
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
        gateway: GatewaySpec::new(bind)
            .with_launch_args(launch_args)
            .with_fingerprint(bootstrap_fingerprint)
            .with_user_config_scope(),
        max_hook_payload_bytes,
    })
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
