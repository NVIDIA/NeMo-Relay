// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-user ownership, locking, and readiness state for managed sidecars.

use std::env;
use std::fs::{self, OpenOptions};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use reqwest::Url;
use serde::{Deserialize, Serialize};

use crate::config::CodingAgent;
use crate::file_io::{LockAttempt, atomic_write, try_lock_exclusive};

use super::health::{RelayHealth, probe, request_shutdown};
use super::{BOOTSTRAP_PROTOCOL_VERSION, GatewayEndpoint, SIDECAR_LOCK_TIMEOUT};

pub(crate) const BOOTSTRAP_AGENT_ENV: &str = "NEMO_RELAY_BOOTSTRAP_AGENT";
pub(crate) const BOOTSTRAP_STATE_DIR_ENV: &str = "NEMO_RELAY_BOOTSTRAP_STATE_DIR";

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct OwnerRecord {
    service: String,
    version: String,
    bootstrap_protocol: u64,
    pub(super) pid: u32,
    pub(super) url: String,
    shutdown_token: String,
    bootstrap_fingerprint: Option<String>,
}

impl OwnerRecord {
    fn new(pid: u32, url: &str, shutdown_token: &str, bootstrap_fingerprint: Option<&str>) -> Self {
        Self {
            service: "nemo-relay".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            bootstrap_protocol: BOOTSTRAP_PROTOCOL_VERSION,
            pid,
            url: url.into(),
            shutdown_token: shutdown_token.into(),
            bootstrap_fingerprint: bootstrap_fingerprint.map(str::to_owned),
        }
    }

    fn matches(
        &self,
        pid: u32,
        url: &str,
        shutdown_token: &str,
        bootstrap_fingerprint: Option<&str>,
    ) -> bool {
        self.service == "nemo-relay"
            && self.version == env!("CARGO_PKG_VERSION")
            && self.bootstrap_protocol == BOOTSTRAP_PROTOCOL_VERSION
            && self.pid == pid
            && self.url == url
            && self.shutdown_token == shutdown_token
            && self.bootstrap_fingerprint.as_deref() == bootstrap_fingerprint
    }
}

#[derive(Debug, Deserialize)]
struct ReadyRecord {
    service: String,
    version: String,
    bootstrap_protocol: u64,
    address: String,
}

pub(super) fn read_owner_record(path: &Path) -> Result<Option<OwnerRecord>, String> {
    let raw = match fs::read(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(format!(
                "failed to read sidecar ownership {}: {error}",
                path.display()
            ));
        }
    };
    serde_json::from_slice(&raw)
        .map(Some)
        .map_err(|error| format!("invalid sidecar ownership file {}: {error}", path.display()))
}

pub(super) fn read_ready_file(path: &Path) -> Result<Option<GatewayEndpoint>, String> {
    let raw = match fs::read(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(format!(
                "failed to read sidecar readiness file {}: {error}",
                path.display()
            ));
        }
    };
    let record = serde_json::from_slice::<ReadyRecord>(&raw)
        .map_err(|error| format!("invalid sidecar readiness file {}: {error}", path.display()))?;
    if record.service != "nemo-relay"
        || record.version != env!("CARGO_PKG_VERSION")
        || record.bootstrap_protocol != BOOTSTRAP_PROTOCOL_VERSION
    {
        return Err(format!(
            "incompatible sidecar readiness file {}",
            path.display()
        ));
    }
    let address = record
        .address
        .parse::<SocketAddr>()
        .map_err(|error| format!("invalid sidecar address in {}: {error}", path.display()))?;
    Ok(Some(GatewayEndpoint {
        address,
        url: format!("http://{address}"),
    }))
}

pub(crate) fn owner_path(runtime: &Path, agent: CodingAgent, url: &str) -> PathBuf {
    runtime.join(format!(
        "{}-sidecar-{}.owner.json",
        agent.as_arg(),
        lock_name(url)
    ))
}

pub(crate) fn pid_path(runtime: &Path, agent: CodingAgent, url: &str) -> PathBuf {
    runtime.join(format!("{}-sidecar-{}.pid", agent.as_arg(), lock_name(url)))
}

pub(crate) fn lock_path(runtime: &Path, url: &str) -> PathBuf {
    runtime.join(format!("{}-sidecar.lock", lock_name(url)))
}

pub(crate) fn state_dir() -> Result<PathBuf, String> {
    crate::config::user_config_dir()
        .map(|path| path.join("bootstrap"))
        .ok_or_else(|| {
            "cannot determine the per-user NeMo Relay bootstrap state directory; set HOME or USERPROFILE"
                .into()
        })
}

pub(crate) fn lock_endpoint(runtime: &Path, url: &str) -> Result<fs::File, String> {
    lock_endpoint_for(runtime, url, SIDECAR_LOCK_TIMEOUT)
}

pub(crate) fn lock_endpoint_for(
    runtime: &Path,
    url: &str,
    timeout: Duration,
) -> Result<fs::File, String> {
    let path = lock_path(runtime, url);
    let lock = open_lock(&path)?;
    let deadline = Instant::now() + timeout;
    loop {
        match try_lock_exclusive(&lock) {
            Ok(LockAttempt::Acquired) => return Ok(lock),
            Ok(LockAttempt::Contended) => {
                if Instant::now() >= deadline {
                    return Err(format!(
                        "timed out waiting for sidecar lock {}",
                        path.display()
                    ));
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(error) => {
                return Err(format!(
                    "failed to acquire sidecar lock {}: {error}",
                    path.display()
                ));
            }
        }
    }
}

pub(super) fn open_lock(path: &Path) -> Result<fs::File, String> {
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| format!("failed to open sidecar lock {}: {error}", path.display()))
}

pub(crate) fn write_owner(
    path: &Path,
    pid: u32,
    url: &str,
    shutdown_token: &str,
    bootstrap_fingerprint: Option<&str>,
) -> Result<(), String> {
    let record = OwnerRecord::new(pid, url, shutdown_token, bootstrap_fingerprint);
    let bytes = serde_json::to_vec(&record)
        .map_err(|error| format!("failed to encode sidecar ownership: {error}"))?;
    atomic_write(path, &bytes)
}

pub(crate) fn publish_owner_from_env(address: SocketAddr) -> Result<(), String> {
    let state = env::var_os(BOOTSTRAP_STATE_DIR_ENV);
    let agent = env::var(BOOTSTRAP_AGENT_ENV).ok();
    let token = env::var("NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN").ok();
    let bootstrap_fingerprint = env::var(crate::config::BOOTSTRAP_FINGERPRINT_ENV)
        .ok()
        .filter(|fingerprint| !fingerprint.is_empty());
    if state.is_none() && agent.is_none() && token.is_none() {
        return Ok(());
    }
    let state = state
        .map(PathBuf::from)
        .ok_or_else(|| format!("{BOOTSTRAP_STATE_DIR_ENV} is required for managed bootstrap"))?;
    if !state.is_absolute() {
        return Err(format!(
            "{BOOTSTRAP_STATE_DIR_ENV} must be an absolute path, got {}",
            state.display()
        ));
    }
    let agent = match agent.as_deref() {
        Some("codex") => CodingAgent::Codex,
        Some("claude-code") => CodingAgent::ClaudeCode,
        Some(other) => return Err(format!("unsupported bootstrap agent {other}")),
        None => {
            return Err(format!(
                "{BOOTSTRAP_AGENT_ENV} is required for managed bootstrap"
            ));
        }
    };
    let token = token.filter(|token| !token.is_empty()).ok_or_else(|| {
        "NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN is required for managed bootstrap".to_string()
    })?;
    if !address.ip().is_loopback() {
        return Err(format!(
            "managed bootstrap ownership requires a loopback address, got {address}"
        ));
    }
    create_private_runtime_dir(&state).map_err(|error| {
        format!(
            "failed to create bootstrap state directory {}: {error}",
            state.display()
        )
    })?;
    let url = format!("http://{address}");
    let pid = std::process::id();
    let owner_path = owner_path(&state, agent, &url);
    let pid_path = pid_path(&state, agent, &url);
    atomic_write(&pid_path, pid.to_string().as_bytes())?;
    if let Err(error) = write_owner(
        &owner_path,
        pid,
        &url,
        &token,
        bootstrap_fingerprint.as_deref(),
    ) {
        let _ = fs::remove_file(&pid_path);
        return Err(error);
    }
    Ok(())
}

pub(crate) fn validate_owner(
    owner_path: &Path,
    pid_path: &Path,
    pid: u32,
    url: &str,
    shutdown_token: &str,
    bootstrap_fingerprint: Option<&str>,
) -> Result<(), String> {
    let owner = read_owner_record(owner_path)?.ok_or_else(|| {
        format!(
            "failed to read sidecar ownership {}: file does not exist",
            owner_path.display()
        )
    })?;
    let valid = owner.matches(pid, url, shutdown_token, bootstrap_fingerprint)
        && fs::read_to_string(pid_path)
            .ok()
            .is_some_and(|value| value.trim() == pid.to_string());
    if valid {
        Ok(())
    } else {
        Err(format!(
            "sidecar ownership {} does not match the ready process",
            owner_path.display()
        ))
    }
}

pub(crate) fn stop_owned(agent: CodingAgent) -> Result<(), String> {
    let runtime = state_dir()?;
    let mut errors = Vec::new();
    for owner_path in owner_paths(&runtime, agent)? {
        if let Err(error) = stop_owned_record(agent, &runtime, &owner_path) {
            errors.push(error);
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

pub(crate) fn owner_paths(runtime: &Path, agent: CodingAgent) -> Result<Vec<PathBuf>, String> {
    let entries = match fs::read_dir(runtime) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(format!(
                "failed to enumerate sidecar ownership in {}: {error}",
                runtime.display()
            ));
        }
    };
    let prefix = format!("{}-sidecar-", agent.as_arg());
    let legacy = format!("{}-sidecar.owner.json", agent.as_arg());
    let mut paths = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            ((name == legacy || (name.starts_with(&prefix) && name.ends_with(".owner.json")))
                && entry.file_type().ok()?.is_file())
            .then(|| entry.path())
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

pub(crate) fn stop_owned_record(
    agent: CodingAgent,
    runtime: &Path,
    owner_path: &Path,
) -> Result<(), String> {
    let Some(initial_owner) = read_owner_record(owner_path)? else {
        return Ok(());
    };
    let _lock = lock_endpoint(runtime, &initial_owner.url)?;
    let Some(owner) = read_owner_record(owner_path)? else {
        return Ok(());
    };
    let url = owner.url.as_str();
    let token = (!owner.shutdown_token.is_empty())
        .then_some(owner.shutdown_token.as_str())
        .ok_or_else(|| {
            format!(
                "sidecar ownership {} has no shutdown token",
                owner_path.display()
            )
        })?;
    let bootstrap_fingerprint = owner
        .bootstrap_fingerprint
        .as_deref()
        .filter(|fingerprint| !fingerprint.is_empty())
        .ok_or_else(|| {
            format!(
                "sidecar ownership {} has no authenticated bootstrap fingerprint",
                owner_path.display()
            )
        })?;
    let legacy = owner_path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == format!("{}-sidecar.owner.json", agent.as_arg()));
    let pid_path = if legacy {
        runtime.join(format!("{}-sidecar.pid", agent.as_arg()))
    } else {
        pid_path(runtime, agent, url)
    };
    match probe(url, Some(bootstrap_fingerprint)) {
        RelayHealth::Unavailable => {
            let _ = fs::remove_file(owner_path);
            let _ = fs::remove_file(&pid_path);
            return Ok(());
        }
        RelayHealth::Compatible => {}
        RelayHealth::Incompatible | RelayHealth::Foreign => {
            return Err(format!(
                "refusing to stop {url}: sidecar ownership points to a foreign listener"
            ));
        }
    }
    request_shutdown(url, token)?;
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if probe(url, Some(bootstrap_fingerprint)) == RelayHealth::Unavailable {
            let _ = fs::remove_file(owner_path);
            let _ = fs::remove_file(&pid_path);
            return Ok(());
        }
        thread::sleep(Duration::from_millis(50));
    }
    Err(format!(
        "timed out waiting for managed sidecar at {url} to stop"
    ))
}

pub(super) fn runtime_dir() -> PathBuf {
    runtime_dir_for(
        env::var_os("XDG_RUNTIME_DIR"),
        env::var_os("TMPDIR"),
        env::var_os("TEMP"),
        env::temp_dir(),
        verified_runtime_user(),
        None,
    )
}

#[cfg(unix)]
fn verified_runtime_user() -> Option<std::ffi::OsString> {
    // SAFETY: `geteuid` has no preconditions and does not dereference pointers.
    Some(format!("uid-{}", unsafe { libc::geteuid() }).into())
}

#[cfg(not(unix))]
fn verified_runtime_user() -> Option<std::ffi::OsString> {
    env::var_os("USERNAME").or_else(|| env::var_os("USER"))
}

pub(crate) fn runtime_dir_for(
    xdg_runtime_dir: Option<std::ffi::OsString>,
    tmpdir: Option<std::ffi::OsString>,
    temp: Option<std::ffi::OsString>,
    temp_dir: PathBuf,
    user: Option<std::ffi::OsString>,
    username: Option<std::ffi::OsString>,
) -> PathBuf {
    if let Some(base) = xdg_runtime_dir {
        return PathBuf::from(base).join("nemo-relay-plugin");
    }
    PathBuf::from(tmpdir.or(temp).unwrap_or_else(|| temp_dir.into_os_string()))
        .join(runtime_user_segment(user, username))
        .join("nemo-relay-plugin")
}

pub(super) fn create_private_runtime_dir(path: &Path) -> std::io::Result<()> {
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

pub(crate) fn lock_name(url: &str) -> String {
    let raw = Url::parse(url)
        .ok()
        .and_then(|parsed| {
            let host = parsed.host_str()?;
            let port = parsed.port_or_known_default()?;
            Some(format!("{host}-{port}"))
        })
        .unwrap_or_else(|| url.to_string());
    sanitize_filesystem_segment(&raw)
}

fn runtime_user_segment(
    user: Option<std::ffi::OsString>,
    username: Option<std::ffi::OsString>,
) -> String {
    let raw = user
        .or(username)
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "unknown-user".into());
    sanitize_filesystem_segment(&raw)
}

fn sanitize_filesystem_segment(raw: &str) -> String {
    let sanitized: String = raw
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "unknown".into()
    } else {
        sanitized
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/sidecar_state_tests.rs"]
mod tests;
