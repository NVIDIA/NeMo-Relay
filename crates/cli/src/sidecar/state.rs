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

use crate::file_io::{LockAttempt, atomic_write, try_lock_exclusive, try_lock_shared};

use super::health::{RelayHealth, probe, request_shutdown};
use super::{BOOTSTRAP_PROTOCOL_VERSION, GatewayEndpoint, SIDECAR_LOCK_TIMEOUT};

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
    instance_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct RecoveryEpoch {
    service: String,
    bootstrap_protocol: u64,
    url: String,
    cohort_id: String,
    pub(super) instance_id: String,
    pub(super) restarts: u8,
    pub(super) pending: bool,
}

impl RecoveryEpoch {
    pub(super) fn new(url: &str, cohort_id: &str, instance_id: &str) -> Self {
        Self {
            service: "nemo-relay".into(),
            bootstrap_protocol: BOOTSTRAP_PROTOCOL_VERSION,
            url: url.into(),
            cohort_id: cohort_id.into(),
            instance_id: instance_id.into(),
            restarts: 0,
            pending: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RecoveryCohort {
    service: String,
    bootstrap_protocol: u64,
    url: String,
    cohort_id: String,
}

impl RecoveryCohort {
    fn new(url: &str) -> Self {
        Self {
            service: "nemo-relay".into(),
            bootstrap_protocol: BOOTSTRAP_PROTOCOL_VERSION,
            url: url.into(),
            cohort_id: uuid::Uuid::now_v7().to_string(),
        }
    }
}

/// Process-lifetime registration used to delimit one shared recovery epoch.
pub(crate) struct EndpointLease {
    file: Option<fs::File>,
    path: PathBuf,
    runtime: PathBuf,
    url: String,
    cohort_id: String,
    fresh_epoch: bool,
}

impl EndpointLease {
    pub(crate) fn acquire(runtime: &Path, url: &str) -> Result<Self, String> {
        create_private_runtime_dir(runtime).map_err(|error| {
            format!(
                "failed to create bootstrap state directory {}: {error}",
                runtime.display()
            )
        })?;
        let registry_path = lease_registry_path(runtime, url);
        let _registry = lock_file_for(&registry_path, SIDECAR_LOCK_TIMEOUT)?;
        let cohort_id = current_or_create_recovery_cohort(runtime, url)?;
        let all_leases_prefix = format!("{}-lease-", lock_name(url));
        let prefix = format!("{all_leases_prefix}{cohort_id}-");
        let mut active = false;
        let entries = fs::read_dir(runtime).map_err(|error| {
            format!(
                "failed to inspect bootstrap leases in {}: {error}",
                runtime.display()
            )
        })?;
        for entry in entries.filter_map(Result::ok) {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if !name.starts_with(&all_leases_prefix) || !name.ends_with(".lock") {
                continue;
            }
            let path = entry.path();
            let file = match OpenOptions::new().read(true).write(true).open(&path) {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => {
                    return Err(format!(
                        "failed to inspect bootstrap lease {}: {error}",
                        path.display()
                    ));
                }
            };
            match try_lock_exclusive(&file) {
                Ok(LockAttempt::Acquired) => {
                    drop(file);
                    let _ = fs::remove_file(path);
                }
                Ok(LockAttempt::Contended) if name.starts_with(&prefix) => active = true,
                Ok(LockAttempt::Contended) => {}
                Err(error) => {
                    return Err(format!(
                        "failed to inspect bootstrap lease {}: {error}",
                        path.display()
                    ));
                }
            }
        }
        let path = runtime.join(format!(
            "{prefix}{}-{}.lock",
            std::process::id(),
            uuid::Uuid::now_v7()
        ));
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|error| {
                format!(
                    "failed to create bootstrap lease {}: {error}",
                    path.display()
                )
            })?;
        match try_lock_exclusive(&file) {
            Ok(LockAttempt::Acquired) => {}
            Ok(LockAttempt::Contended) => {
                return Err(format!(
                    "new bootstrap lease {} was unexpectedly contended",
                    path.display()
                ));
            }
            Err(error) => {
                return Err(format!(
                    "failed to lock bootstrap lease {}: {error}",
                    path.display()
                ));
            }
        }
        if !active {
            let _ = fs::remove_file(recovery_path(runtime, url));
        }
        Ok(Self {
            file: Some(file),
            path,
            runtime: runtime.to_path_buf(),
            url: url.to_string(),
            cohort_id,
            fresh_epoch: !active,
        })
    }

    pub(crate) fn fresh_epoch(&self) -> bool {
        self.fresh_epoch
    }

    pub(crate) fn cohort_id(&self) -> &str {
        &self.cohort_id
    }
}

impl Drop for EndpointLease {
    fn drop(&mut self) {
        // Acquisition and release share this registry lock. This closes the handoff race where a
        // new lease could observe the old lease as active immediately before it disappeared and
        // become the sole client of an exhausted recovery epoch.
        let Ok(_registry) = lock_file_for(
            &lease_registry_path(&self.runtime, &self.url),
            SIDECAR_LOCK_TIMEOUT,
        ) else {
            // Releasing without the registry would reintroduce the last-client handoff race.
            // Conservatively retain the advisory lock until process exit; a later process can
            // clean up the lease file after the operating system releases the descriptor.
            if let Some(file) = self.file.take() {
                std::mem::forget(file);
            }
            return;
        };
        drop(self.file.take());
        let _ = fs::remove_file(&self.path);
    }
}

fn lease_registry_path(runtime: &Path, url: &str) -> PathBuf {
    runtime.join(format!("{}-leases.lock", lock_name(url)))
}

fn recovery_cohort_path(runtime: &Path, url: &str) -> PathBuf {
    runtime.join(format!("sidecar-{}.cohort.json", lock_name(url)))
}

fn current_or_create_recovery_cohort(runtime: &Path, url: &str) -> Result<String, String> {
    if let Some(cohort) = read_recovery_cohort(runtime, url)? {
        return Ok(cohort.cohort_id);
    }
    let cohort = RecoveryCohort::new(url);
    write_recovery_cohort(runtime, &cohort)?;
    Ok(cohort.cohort_id)
}

fn read_recovery_cohort(runtime: &Path, url: &str) -> Result<Option<RecoveryCohort>, String> {
    let path = recovery_cohort_path(runtime, url);
    let raw = match fs::read(&path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(format!(
                "failed to read gateway recovery cohort {}: {error}",
                path.display()
            ));
        }
    };
    let cohort = serde_json::from_slice::<RecoveryCohort>(&raw).map_err(|error| {
        format!(
            "invalid gateway recovery cohort {}: {error}",
            path.display()
        )
    })?;
    if cohort.service != "nemo-relay"
        || cohort.bootstrap_protocol != BOOTSTRAP_PROTOCOL_VERSION
        || cohort.url != url
        || cohort.cohort_id.is_empty()
    {
        return Err(format!(
            "incompatible gateway recovery cohort {}",
            path.display()
        ));
    }
    Ok(Some(cohort))
}

fn write_recovery_cohort(runtime: &Path, cohort: &RecoveryCohort) -> Result<(), String> {
    let path = recovery_cohort_path(runtime, &cohort.url);
    let bytes = serde_json::to_vec(cohort)
        .map_err(|error| format!("failed to encode gateway recovery cohort: {error}"))?;
    atomic_write(&path, &bytes)
}

pub(super) fn validate_recovery_cohort(
    runtime: &Path,
    url: &str,
    expected: &str,
) -> Result<(), String> {
    match read_recovery_cohort(runtime, url)? {
        Some(cohort) if cohort.cohort_id == expected => Ok(()),
        _ => Err(format!(
            "shared Relay gateway lifecycle at {url} was retired by an integration update"
        )),
    }
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
        || record.instance_id.is_empty()
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
        instance_id: record.instance_id,
    }))
}

pub(crate) fn owner_path(runtime: &Path, url: &str) -> PathBuf {
    runtime.join(format!("sidecar-{}.owner.json", lock_name(url)))
}

pub(crate) fn pid_path(runtime: &Path, url: &str) -> PathBuf {
    runtime.join(format!("sidecar-{}.pid", lock_name(url)))
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

pub(crate) fn lock_endpoint_shared(runtime: &Path, url: &str) -> Result<fs::File, String> {
    let path = lock_path(runtime, url);
    lock_file_for_mode(&path, SIDECAR_LOCK_TIMEOUT, false)
}

pub(crate) fn lock_endpoint_for(
    runtime: &Path,
    url: &str,
    timeout: Duration,
) -> Result<fs::File, String> {
    let path = lock_path(runtime, url);
    lock_file_for(&path, timeout)
}

fn lock_file_for(path: &Path, timeout: Duration) -> Result<fs::File, String> {
    lock_file_for_mode(path, timeout, true)
}

fn lock_file_for_mode(path: &Path, timeout: Duration, exclusive: bool) -> Result<fs::File, String> {
    let lock = open_lock(path)?;
    let deadline = Instant::now() + timeout;
    loop {
        let result = if exclusive {
            try_lock_exclusive(&lock)
        } else {
            try_lock_shared(&lock)
        };
        match result {
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

pub(super) fn recovery_path(runtime: &Path, url: &str) -> PathBuf {
    runtime.join(format!("sidecar-{}.recovery.json", lock_name(url)))
}

pub(super) fn read_recovery_epoch(
    runtime: &Path,
    url: &str,
    cohort_id: &str,
) -> Result<Option<RecoveryEpoch>, String> {
    let path = recovery_path(runtime, url);
    let raw = match fs::read(&path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(format!(
                "failed to read gateway recovery epoch {}: {error}",
                path.display()
            ));
        }
    };
    let epoch = serde_json::from_slice::<RecoveryEpoch>(&raw)
        .map_err(|error| format!("invalid gateway recovery epoch {}: {error}", path.display()))?;
    if epoch.service != "nemo-relay"
        || epoch.bootstrap_protocol != BOOTSTRAP_PROTOCOL_VERSION
        || epoch.url != url
        || epoch.cohort_id != cohort_id
        || epoch.instance_id.is_empty()
        || epoch.restarts > 1
    {
        return Err(format!(
            "incompatible gateway recovery epoch {}",
            path.display()
        ));
    }
    Ok(Some(epoch))
}

pub(super) fn write_recovery_epoch(runtime: &Path, epoch: &RecoveryEpoch) -> Result<(), String> {
    let path = recovery_path(runtime, &epoch.url);
    let bytes = serde_json::to_vec(epoch)
        .map_err(|error| format!("failed to encode gateway recovery epoch: {error}"))?;
    atomic_write(&path, &bytes)
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
    let token = env::var("NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN").ok();
    let bootstrap_fingerprint = env::var(crate::config::BOOTSTRAP_FINGERPRINT_ENV)
        .ok()
        .filter(|fingerprint| !fingerprint.is_empty());
    if state.is_none() && token.is_none() {
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
    let owner_path = owner_path(&state, &url);
    let pid_path = pid_path(&state, &url);
    for stale_path in owner_paths(&state)? {
        if stale_path == owner_path {
            continue;
        }
        let Ok(Some(stale)) = read_owner_record(&stale_path) else {
            continue;
        };
        if stale.url == url {
            let stale_pid = owner_pid_path(&state, &stale_path, &url);
            let _ = fs::remove_file(stale_path);
            let _ = fs::remove_file(stale_pid);
        }
    }
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

/// Stop a plugin-owned endpoint and begin a fresh recovery cohort as one serialized operation.
///
/// The endpoint and lease-registry locks prevent a heartbeat or hook from recovering the old
/// gateway between shutdown and cohort rotation. If shutdown fails, both lifecycle files are
/// restored byte-for-byte so the still-active clients retain their original recovery budget.
pub(crate) fn stop_owned_and_reset(target_url: &str) -> Result<(), String> {
    stop_owned_and_reset_after_rotation(target_url, || {})
}

fn stop_owned_and_reset_after_rotation(
    target_url: &str,
    after_rotation: impl FnOnce(),
) -> Result<(), String> {
    let runtime = state_dir()?;
    create_private_runtime_dir(&runtime).map_err(|error| {
        format!(
            "failed to create bootstrap state directory {}: {error}",
            runtime.display()
        )
    })?;
    let _endpoint = lock_endpoint(&runtime, target_url)?;
    let _registry = lock_file_for(
        &lease_registry_path(&runtime, target_url),
        SIDECAR_LOCK_TIMEOUT,
    )?;
    let cohort_path = recovery_cohort_path(&runtime, target_url);
    let epoch_path = recovery_path(&runtime, target_url);
    let cohort_snapshot = read_optional_bytes(&cohort_path)?;
    let epoch_snapshot = read_optional_bytes(&epoch_path)?;
    let replacement = RecoveryCohort::new(target_url);
    let result = write_recovery_cohort(&runtime, &replacement)
        .and_then(|()| remove_optional_state_file(&epoch_path))
        .and_then(|()| {
            after_rotation();
            stop_owned_matching_records(&runtime, target_url, true)
        });
    if let Err(error) = result {
        let mut rollback_errors = Vec::new();
        if let Err(restore_error) = restore_optional_bytes(&cohort_path, cohort_snapshot.as_deref())
        {
            rollback_errors.push(restore_error);
        }
        if let Err(restore_error) = restore_optional_bytes(&epoch_path, epoch_snapshot.as_deref()) {
            rollback_errors.push(restore_error);
        }
        return if rollback_errors.is_empty() {
            Err(error)
        } else {
            Err(format!(
                "{error}; additionally failed to restore gateway recovery state: {}",
                rollback_errors.join("; ")
            ))
        };
    }
    Ok(())
}

#[cfg(test)]
fn stop_owned(target_url: &str) -> Result<(), String> {
    let runtime = state_dir()?;
    stop_owned_matching_records(&runtime, target_url, false)
}

fn stop_owned_matching_records(
    runtime: &Path,
    target_url: &str,
    endpoint_is_locked: bool,
) -> Result<(), String> {
    let mut errors = Vec::new();
    for owner_path in owner_paths(runtime)? {
        match read_owner_record(&owner_path) {
            Ok(Some(owner)) if owner.url == target_url => {
                let result = if endpoint_is_locked {
                    stop_owned_record_after_lock(runtime, &owner_path)
                } else {
                    stop_owned_record(runtime, &owner_path)
                };
                if let Err(error) = result {
                    errors.push(error);
                }
            }
            Ok(_) => {}
            Err(error) => errors.push(error),
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

fn read_optional_bytes(path: &Path) -> Result<Option<Vec<u8>>, String> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("failed to read {}: {error}", path.display())),
    }
}

fn restore_optional_bytes(path: &Path, bytes: Option<&[u8]>) -> Result<(), String> {
    match bytes {
        Some(bytes) => atomic_write(path, bytes),
        None => remove_optional_state_file(path),
    }
}

fn remove_optional_state_file(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("failed to remove {}: {error}", path.display())),
    }
}

pub(crate) fn owner_paths(runtime: &Path) -> Result<Vec<PathBuf>, String> {
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
    let mut paths = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            (is_sidecar_owner_name(name) && entry.file_type().ok()?.is_file()).then(|| entry.path())
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn is_sidecar_owner_name(name: &str) -> bool {
    name == "sidecar.owner.json"
        || (name.starts_with("sidecar-") && name.ends_with(".owner.json"))
        || (name.ends_with("-sidecar.owner.json"))
        || (name.contains("-sidecar-") && name.ends_with(".owner.json"))
}

pub(crate) fn owner_pid_path(runtime: &Path, owner_path: &Path, url: &str) -> PathBuf {
    let name = owner_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if name == "sidecar.owner.json" {
        return runtime.join("sidecar.pid");
    }
    if name.ends_with("-sidecar.owner.json") {
        return owner_path.with_file_name(name.replace(".owner.json", ".pid"));
    }
    if name.contains("-sidecar-") && !name.starts_with("sidecar-") {
        return owner_path.with_file_name(name.replace(".owner.json", ".pid"));
    }
    pid_path(runtime, url)
}

pub(crate) fn stop_owned_record(runtime: &Path, owner_path: &Path) -> Result<(), String> {
    let Some(initial_owner) = read_owner_record(owner_path)? else {
        return Ok(());
    };
    let _lock = lock_endpoint(runtime, &initial_owner.url)?;
    stop_owned_record_after_lock(runtime, owner_path)
}

fn stop_owned_record_after_lock(runtime: &Path, owner_path: &Path) -> Result<(), String> {
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
    let pid_path = owner_pid_path(runtime, owner_path, url);
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
