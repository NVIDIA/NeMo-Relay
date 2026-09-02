// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-user startup lock and ownership record for the shared gateway.

use std::env;
use std::fs::{self, OpenOptions};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use reqwest::Url;
use serde::{Deserialize, Serialize};

use crate::filesystem::{LockAttempt, atomic_write, try_lock_exclusive};

use super::{BOOTSTRAP_LOCK_TIMEOUT, BOOTSTRAP_PROTOCOL_VERSION};
use crate::gateway::client::{RelayHealth, probe, request_shutdown};

pub(crate) const BOOTSTRAP_STATE_DIR_ENV: &str = "NEMO_RELAY_BOOTSTRAP_STATE_DIR";
pub(crate) const BOOTSTRAP_SHUTDOWN_TOKEN_ENV: &str = "NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN";
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
const UNHEALTHY_GATEWAY_TERMINATION_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub(super) struct OwnerRecord {
    service: String,
    version: String,
    bootstrap_protocol: u64,
    pid: u32,
    #[serde(default)]
    process_identity: Option<u64>,
    url: String,
    shutdown_token: String,
    bootstrap_fingerprint: Option<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq, Serialize)]
pub(super) struct RecoveryRecord {
    pub(super) from_instance: String,
    pub(super) endpoint_url: String,
    pub(super) to_instance: String,
}

impl OwnerRecord {
    fn new(pid: u32, url: &str, shutdown_token: &str, fingerprint: Option<&str>) -> Self {
        Self {
            service: "nemo-relay".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            bootstrap_protocol: BOOTSTRAP_PROTOCOL_VERSION,
            pid,
            process_identity: process_identity(pid),
            url: url.into(),
            shutdown_token: shutdown_token.into(),
            bootstrap_fingerprint: fingerprint.map(str::to_owned),
        }
    }

    fn valid_for(&self, url: &str) -> bool {
        self.service == "nemo-relay"
            && self.bootstrap_protocol == BOOTSTRAP_PROTOCOL_VERSION
            && self.url == url
            && !self.shutdown_token.is_empty()
            && self
                .bootstrap_fingerprint
                .as_deref()
                .is_some_and(|fingerprint| !fingerprint.is_empty())
    }
}

/// Removes this process's ownership record when the gateway server exits.
#[derive(Debug)]
pub(crate) struct OwnerGuard {
    path: PathBuf,
    record: OwnerRecord,
}

impl Drop for OwnerGuard {
    fn drop(&mut self) {
        let _ = remove_if_matches(&self.path, &self.record);
    }
}

pub(crate) fn state_dir() -> Result<PathBuf, String> {
    crate::configuration::user_config_dir()
        .map(|path| path.join("bootstrap"))
        .ok_or_else(|| {
            "cannot determine the per-user NeMo Relay bootstrap state directory; set HOME or USERPROFILE"
                .into()
        })
}

pub(crate) fn create_private_dir(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .map_err(|error| format!("failed to secure {}: {error}", path.display()))?;
    }
    Ok(())
}

pub(crate) fn owner_path(state: &Path, url: &str) -> PathBuf {
    state.join(format!("sidecar-{}.owner.json", lock_name(url)))
}

pub(crate) fn lock_path(state: &Path, url: &str) -> PathBuf {
    state.join(format!("gateway-{}.lock", lock_name(url)))
}

fn recovery_path(state: &Path, url: &str) -> PathBuf {
    state.join(format!("gateway-{}.recovery.json", lock_name(url)))
}

pub(super) fn read_recovery(state: &Path, url: &str) -> Result<Option<RecoveryRecord>, String> {
    let path = recovery_path(state, url);
    match fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map(Some).map_err(|error| {
            format!(
                "failed to parse gateway recovery {}: {error}",
                path.display()
            )
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!(
            "failed to read gateway recovery {}: {error}",
            path.display()
        )),
    }
}

pub(super) fn write_recovery(
    state: &Path,
    url: &str,
    record: &RecoveryRecord,
) -> Result<(), String> {
    let path = recovery_path(state, url);
    let bytes = serde_json::to_vec(record)
        .map_err(|error| format!("failed to encode gateway recovery: {error}"))?;
    atomic_write(&path, &bytes)
}

pub(crate) fn lock_endpoint(state: &Path, url: &str) -> Result<fs::File, String> {
    lock_endpoint_for(state, url, BOOTSTRAP_LOCK_TIMEOUT)
}

pub(crate) fn lock_endpoint_for(
    state: &Path,
    url: &str,
    timeout: Duration,
) -> Result<fs::File, String> {
    create_private_dir(state)?;
    let path = lock_path(state, url);
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|error| format!("failed to open gateway lock {}: {error}", path.display()))?;
    let deadline = Instant::now() + timeout;
    loop {
        match try_lock_exclusive(&lock) {
            Ok(LockAttempt::Acquired) => return Ok(lock),
            Ok(LockAttempt::Contended) if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(50));
            }
            Ok(LockAttempt::Contended) => {
                return Err(format!(
                    "timed out waiting for gateway startup lock {}",
                    path.display()
                ));
            }
            Err(error) => {
                return Err(format!(
                    "failed to acquire gateway startup lock {}: {error}",
                    path.display()
                ));
            }
        }
    }
}

pub(crate) fn publish_owner_from_env(
    address: SocketAddr,
    shutdown_token: Option<&str>,
) -> Result<Option<OwnerGuard>, String> {
    let state = env::var_os(BOOTSTRAP_STATE_DIR_ENV);
    if state.is_none() && shutdown_token.is_none() {
        return Ok(None);
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
    let token = shutdown_token
        .filter(|token| !token.is_empty())
        .ok_or_else(|| {
            format!("{BOOTSTRAP_SHUTDOWN_TOKEN_ENV} is required for managed bootstrap")
        })?;
    if !address.ip().is_loopback() {
        return Err(format!(
            "managed bootstrap ownership requires a loopback address, got {address}"
        ));
    }
    create_private_dir(&state)?;
    let url = format!("http://{address}");
    let fingerprint = env::var(crate::configuration::BOOTSTRAP_FINGERPRINT_ENV)
        .ok()
        .filter(|value| !value.is_empty());
    let record = OwnerRecord::new(std::process::id(), &url, token, fingerprint.as_deref());
    let path = owner_path(&state, &url);
    write_owner_record(&path, &record)?;
    Ok(Some(OwnerGuard { path, record }))
}

pub(crate) fn stop_owned_and_reset(url: &str) -> Result<(), String> {
    let state = state_dir()?;
    if !state.exists() {
        return Ok(());
    }
    let _lock = lock_endpoint(&state, url)?;
    stop_owned_and_reset_locked(&state, url).map(|_| ())
}

/// Stops a version-mismatched, Relay-owned gateway while the caller holds the
/// endpoint startup lock.
///
/// Returns `true` when an owned gateway was stopped or a stale owned record was
/// removed. Same-version, invalid, and absent records return `false` so callers
/// can preserve the original incompatible-gateway error.
pub(crate) fn stop_version_mismatched_owned_gateway_locked(
    state: &Path,
    url: &str,
) -> Result<bool, String> {
    let path = owner_path(state, url);
    let Some(owner) = read_owner_record(&path)? else {
        return Ok(false);
    };
    if !owner.valid_for(url) || owner.version == env!("CARGO_PKG_VERSION") {
        return Ok(false);
    }
    stop_owned_gateway_locked(&path, &owner, url)
}

/// Terminates a managed gateway whose health endpoint is unavailable so its
/// listener cannot block a replacement process from binding the endpoint.
///
/// The caller holds the endpoint startup lock. A valid ownership record is the
/// authority for signalling the process; listeners without one remain untouched.
pub(crate) fn stop_unhealthy_owned_gateway_locked(state: &Path, url: &str) -> Result<bool, String> {
    let path = owner_path(state, url);
    let Some(owner) = read_owner_record(&path)? else {
        return Ok(false);
    };
    if !owner.valid_for(url) {
        return Ok(false);
    }
    if probe(url, owner.bootstrap_fingerprint.as_deref()) == RelayHealth::Compatible {
        return Ok(false);
    }
    if !owner_process_identity_matches(&owner) {
        remove_if_matches(&path, &owner)?;
        return Ok(false);
    }

    let was_running = terminate_owned_gateway_process(owner.pid)?;
    remove_if_matches(&path, &owner)?;
    Ok(was_running)
}

fn stop_owned_and_reset_locked(state: &Path, url: &str) -> Result<bool, String> {
    let path = owner_path(state, url);
    let Some(owner) = read_owner_record(&path)? else {
        return Ok(false);
    };
    if !owner.valid_for(url) {
        return Err(format!(
            "refusing to stop gateway from invalid ownership record {}",
            path.display()
        ));
    }
    stop_owned_gateway_locked(&path, &owner, url)
}

fn stop_owned_gateway_locked(path: &Path, owner: &OwnerRecord, url: &str) -> Result<bool, String> {
    match probe(url, owner.bootstrap_fingerprint.as_deref()) {
        RelayHealth::Unavailable => {
            remove_if_matches(path, owner)?;
            return Ok(true);
        }
        RelayHealth::Compatible => {}
        RelayHealth::Incompatible | RelayHealth::Foreign => {
            return Err(format!(
                "refusing to stop an unverified process at managed gateway URL {url}"
            ));
        }
    }
    request_shutdown(
        url,
        owner
            .bootstrap_fingerprint
            .as_deref()
            .expect("validated owner record has a bootstrap fingerprint"),
        &owner.shutdown_token,
    )?;
    let deadline = Instant::now() + SHUTDOWN_TIMEOUT;
    loop {
        match probe(url, owner.bootstrap_fingerprint.as_deref()) {
            RelayHealth::Unavailable => break,
            RelayHealth::Compatible if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(50));
            }
            RelayHealth::Compatible => {
                return Err(format!("managed Relay gateway at {url} did not stop"));
            }
            RelayHealth::Incompatible | RelayHealth::Foreign => {
                return Err(format!(
                    "a different process replaced the managed Relay gateway at {url} during shutdown"
                ));
            }
        }
    }
    remove_if_matches(path, owner)?;
    Ok(true)
}

#[cfg(unix)]
fn terminate_owned_gateway_process(pid: u32) -> Result<bool, String> {
    let Ok(pid) = i32::try_from(pid) else {
        return Ok(false);
    };
    if !process_is_running(pid) {
        return Ok(false);
    }

    log_unhealthy_gateway_termination_started(pid);
    let target = signal_gateway_process_group(pid, libc::SIGTERM)?;
    if wait_for_termination(target, UNHEALTHY_GATEWAY_TERMINATION_TIMEOUT) {
        return Ok(true);
    }

    log_unhealthy_gateway_termination_escalated(pid);
    signal_gateway_termination_target(target, libc::SIGKILL)?;
    if wait_for_termination(target, UNHEALTHY_GATEWAY_TERMINATION_TIMEOUT) {
        Ok(true)
    } else {
        Err(format!(
            "managed Relay gateway process {pid} did not terminate"
        ))
    }
}

#[cfg(target_os = "linux")]
fn process_identity(pid: u32) -> Option<u64> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // `comm` is parenthesized and may contain spaces, so process fields only
    // after the final `)` character.
    stat.rsplit_once(')')?
        .1
        .split_whitespace()
        // Field 22 is process start time in clock ticks. The first field here
        // is field 3 (`state`).
        .nth(19)?
        .parse()
        .ok()
}

#[cfg(target_os = "macos")]
fn process_identity(pid: u32) -> Option<u64> {
    let pid = i32::try_from(pid).ok()?;
    // SAFETY: `info` is initialized and passed with its exact size for the
    // documented PROC_PIDTBSDINFO query.
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let result = unsafe {
        libc::proc_pidinfo(
            pid,
            libc::PROC_PIDTBSDINFO,
            0,
            (&raw mut info).cast(),
            i32::try_from(std::mem::size_of::<libc::proc_bsdinfo>()).ok()?,
        )
    };
    (result == std::mem::size_of::<libc::proc_bsdinfo>() as i32)
        .then(|| {
            info.pbi_start_tvsec
                .checked_mul(1_000_000)?
                .checked_add(info.pbi_start_tvusec)
        })
        .flatten()
}

#[cfg(windows)]
fn process_identity(pid: u32) -> Option<u64> {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
    use windows_sys::Win32::System::Threading::{
        GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    // A handle keeps the queried process instance stable while its creation
    // identity is read, even if this PID is recycled concurrently.
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if handle.is_null() {
        return None;
    }
    let mut creation: FILETIME = unsafe { std::mem::zeroed() };
    let mut exit: FILETIME = unsafe { std::mem::zeroed() };
    let mut kernel: FILETIME = unsafe { std::mem::zeroed() };
    let mut user: FILETIME = unsafe { std::mem::zeroed() };
    let result =
        unsafe { GetProcessTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user) };
    unsafe { CloseHandle(handle) };
    (result != 0)
        .then(|| (u64::from(creation.dwHighDateTime) << 32) | u64::from(creation.dwLowDateTime))
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn process_identity(_pid: u32) -> Option<u64> {
    None
}

fn owner_process_identity_matches(owner: &OwnerRecord) -> bool {
    owner.process_identity.is_some_and(|identity| {
        process_identity(owner.pid).is_some_and(|current| current == identity)
    })
}

#[cfg(unix)]
#[derive(Clone, Copy)]
enum GatewayTerminationTarget {
    ProcessGroup(i32),
    Process(i32),
}

#[cfg(unix)]
fn signal_gateway_process_group(pid: i32, signal: i32) -> Result<GatewayTerminationTarget, String> {
    // Detached gateways call setsid, making their PID the process-group ID.
    // Fall back to the direct PID for an older or otherwise non-detached sidecar.
    let group_result = unsafe { libc::kill(-pid, signal) };
    if group_result == 0 {
        return Ok(GatewayTerminationTarget::ProcessGroup(pid));
    }
    let group_error = std::io::Error::last_os_error();
    if group_error.raw_os_error() != Some(libc::ESRCH) {
        return Err(format!(
            "failed to signal managed Relay gateway process {pid}: {group_error}"
        ));
    }
    if unsafe { libc::kill(pid, signal) } == 0 {
        Ok(GatewayTerminationTarget::Process(pid))
    } else {
        let error = std::io::Error::last_os_error();
        (error.raw_os_error() == Some(libc::ESRCH))
            .then_some(GatewayTerminationTarget::Process(pid))
            .ok_or_else(|| format!("failed to signal managed Relay gateway process {pid}: {error}"))
    }
}

#[cfg(unix)]
fn signal_gateway_termination_target(
    target: GatewayTerminationTarget,
    signal: i32,
) -> Result<(), String> {
    let (pid, target_pid) = match target {
        GatewayTerminationTarget::ProcessGroup(pid) => (pid, -pid),
        GatewayTerminationTarget::Process(pid) => (pid, pid),
    };
    if unsafe { libc::kill(target_pid, signal) } == 0 {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    (error.raw_os_error() == Some(libc::ESRCH))
        .then_some(())
        .ok_or_else(|| format!("failed to signal managed Relay gateway process {pid}: {error}"))
}

#[cfg(unix)]
fn process_is_running(pid: i32) -> bool {
    (unsafe { libc::kill(pid, 0) }) == 0
        || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(unix)]
fn wait_for_termination(target: GatewayTerminationTarget, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while termination_target_is_running(target) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(50));
    }
    !termination_target_is_running(target)
}

#[cfg(unix)]
fn termination_target_is_running(target: GatewayTerminationTarget) -> bool {
    match target {
        GatewayTerminationTarget::ProcessGroup(pid) => process_is_running(-pid),
        GatewayTerminationTarget::Process(pid) => process_is_running(pid),
    }
}

#[cfg(windows)]
fn terminate_owned_gateway_process(pid: u32) -> Result<bool, String> {
    if !windows_process_is_running(pid) {
        return Ok(false);
    }

    log_unhealthy_gateway_termination_started(pid as i32);
    let _graceful = std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T"])
        .status();
    if wait_for_windows_process_exit(pid, UNHEALTHY_GATEWAY_TERMINATION_TIMEOUT) {
        return Ok(true);
    }

    log_unhealthy_gateway_termination_escalated(pid as i32);
    let _forced = std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .status()
        .map_err(|error| {
            format!("failed to force-kill managed Relay gateway process {pid}: {error}")
        })?;
    if wait_for_windows_process_exit(pid, UNHEALTHY_GATEWAY_TERMINATION_TIMEOUT) {
        Ok(true)
    } else {
        Err(format!(
            "managed Relay gateway process {pid} did not terminate"
        ))
    }
}

#[cfg(windows)]
fn windows_process_is_running(pid: u32) -> bool {
    process_identity(pid).is_some()
}

#[cfg(windows)]
fn wait_for_windows_process_exit(pid: u32, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while windows_process_is_running(pid) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(50));
    }
    !windows_process_is_running(pid)
}

fn log_unhealthy_gateway_termination_started(pid: i32) {
    log::warn!(
        target: "nemo_relay.bootstrap",
        event = "unhealthy_gateway_termination_started",
        process_id = pid;
        "Terminating unhealthy managed gateway"
    );
}

fn log_unhealthy_gateway_termination_escalated(pid: i32) {
    log::warn!(
        target: "nemo_relay.bootstrap",
        event = "unhealthy_gateway_termination_escalated",
        process_id = pid;
        "Force-killing unhealthy managed gateway after graceful termination timeout"
    );
}

#[cfg(not(any(unix, windows)))]
fn terminate_owned_gateway_process(pid: u32) -> Result<bool, String> {
    Err(format!(
        "cannot terminate unhealthy managed Relay gateway process {pid} on this platform"
    ))
}

fn write_owner_record(path: &Path, record: &OwnerRecord) -> Result<(), String> {
    let bytes = serde_json::to_vec(record)
        .map_err(|error| format!("failed to encode gateway ownership: {error}"))?;
    atomic_write(path, &bytes)
}

pub(super) fn read_owner_record(path: &Path) -> Result<Option<OwnerRecord>, String> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map(Some).map_err(|error| {
            format!(
                "failed to parse gateway ownership {}: {error}",
                path.display()
            )
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!(
            "failed to read gateway ownership {}: {error}",
            path.display()
        )),
    }
}

fn remove_if_matches(path: &Path, expected: &OwnerRecord) -> Result<(), String> {
    if read_owner_record(path)?.as_ref() != Some(expected) {
        return Ok(());
    }
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "failed to remove gateway ownership {}: {error}",
            path.display()
        )),
    }
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
    let sanitized = raw
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "unknown".into()
    } else {
        sanitized
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/bootstrap_state_tests.rs"]
mod tests;
