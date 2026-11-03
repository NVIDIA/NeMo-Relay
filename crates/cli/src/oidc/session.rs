// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The stored sign-in: one owner-only file per issuer and client, and the lock around it.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{OidcSource, auth_error};
use crate::error::CliError;

const SESSIONS_DIR: &str = "oidc";
/// A refresh holds the lock for at most two issuer requests (discovery and token), so a waiter
/// gives up only after a concurrent refresh would have finished or failed.
const LOCK_WAIT: Duration = Duration::from_secs(35);

/// One stored sign-in. Do not add `Debug`: it holds tokens.
#[derive(Serialize, Deserialize)]
pub(crate) struct Session {
    pub(crate) issuer: String,
    pub(crate) client_id: String,
    pub(crate) access_token: String,
    pub(crate) refresh_token: Option<String>,
    /// Seconds since the Unix epoch.
    pub(crate) expires_at: u64,
}

/// Returns `None` when no sign-in is stored for this issuer and client.
pub(crate) fn read_session(path: &Path, source: &OidcSource) -> Result<Option<Session>, CliError> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(CliError::Io(error)),
    };
    let session: Session = serde_json::from_slice(&bytes).map_err(|_| {
        auth_error(format!(
            "the sign-in file {} is unreadable; run `nemo-relay auth logout` and sign in again",
            path.display()
        ))
    })?;
    let same_source = session.issuer == source.issuer() && session.client_id == source.client_id;
    Ok(same_source.then_some(session))
}

pub(crate) fn write_session(path: &Path, session: &Session) -> Result<(), CliError> {
    let bytes = serde_json::to_vec(session)
        .map_err(|error| auth_error(format!("failed to encode the sign-in: {error}")))?;
    crate::filesystem::atomic_write_private(path, &bytes)
        .map_err(|error| auth_error(format!("failed to save the sign-in: {error}")))
}

pub(crate) fn ensure_sessions_dir(dir: &Path) -> Result<(), CliError> {
    if dir.exists() {
        return Ok(());
    }
    if let Some(parent) = dir.parent() {
        crate::filesystem::create_private_dir_all(parent)?;
    }
    match crate::filesystem::create_private_dir(dir) {
        Err(error) if error.kind() != std::io::ErrorKind::AlreadyExists => Err(CliError::Io(error)),
        _ => Ok(()),
    }
}

/// One file per issuer and client, so sign-ins to several providers do not collide.
pub(crate) fn session_path(dir: &Path, source: &OidcSource) -> PathBuf {
    let digest = Sha256::digest(format!("{}\n{}", source.issuer(), source.client_id));
    let name: String = digest[..16]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    dir.join(format!("{name}.json"))
}

pub(crate) fn sessions_dir() -> Result<PathBuf, CliError> {
    crate::configuration::user_config_dir()
        .map(|dir| dir.join(SESSIONS_DIR))
        .ok_or_else(|| {
            CliError::Config(
                "cannot determine the per-user NeMo Relay config directory; set HOME or XDG_CONFIG_HOME"
                    .into(),
            )
        })
}

/// Serializes every write of one sign-in across processes, so a rotated refresh token is never
/// used twice and a refresh cannot restore a sign-in that `logout` removed.
pub(super) struct SessionLock(std::fs::File);

impl SessionLock {
    pub(super) async fn acquire(session: &Path) -> Result<Self, CliError> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(session.with_extension("lock"))?;
        let deadline = tokio::time::Instant::now() + LOCK_WAIT;
        while crate::filesystem::try_lock_exclusive(&file)?
            == crate::filesystem::LockAttempt::Contended
        {
            if tokio::time::Instant::now() >= deadline {
                return Err(auth_error(
                    "another sign-in update did not finish in time".into(),
                ));
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        Ok(Self(file))
    }
}

impl Drop for SessionLock {
    fn drop(&mut self) {
        let _ = crate::filesystem::unlock_file(&self.0);
    }
}
