// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serialized, rollback-capable filesystem operations for the Hermes integration.

use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::CliError;
use crate::file_io::{LockAttempt, try_lock_exclusive};
use crate::install_generation::GENERATION_FILE_NAME;

const ALLOWLIST_FILE_NAME: &str = "shell-hooks-allowlist.json";
const INSTALL_LOCK_FILE_NAME: &str = ".nemo-relay-operation.lock";
const INSTALL_LOCK_RETRY: Duration = Duration::from_millis(25);
pub(super) const INSTALL_LOCK_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PersistentPaths {
    pub(super) config: PathBuf,
    pub(super) allowlist: PathBuf,
    pub(super) generation: PathBuf,
}

impl PersistentPaths {
    pub(super) fn for_config(config: PathBuf) -> Result<Self, CliError> {
        let home = config.parent().ok_or_else(|| {
            CliError::Install(format!(
                "Hermes config path {} has no parent directory",
                config.display()
            ))
        })?;
        Ok(Self {
            allowlist: home.join(ALLOWLIST_FILE_NAME),
            generation: home.join(GENERATION_FILE_NAME),
            config,
        })
    }

    pub(super) fn all(&self) -> [PathBuf; 3] {
        [
            self.config.clone(),
            self.allowlist.clone(),
            self.generation.clone(),
        ]
    }
}

pub(super) fn acquire_install_lock(config: &Path, timeout: Duration) -> Result<File, String> {
    let home = config.parent().ok_or_else(|| {
        format!(
            "Hermes config path {} has no parent directory",
            config.display()
        )
    })?;
    acquire_lock_file(
        &home.join(INSTALL_LOCK_FILE_NAME),
        timeout,
        "another Hermes integration update",
    )
}

/// Uses Hermes's own sibling allowlist lock so Relay cannot lose an unrelated approval that
/// Hermes records concurrently.
pub(super) fn acquire_allowlist_lock(allowlist: &Path, timeout: Duration) -> Result<File, String> {
    let mut lock = allowlist.as_os_str().to_os_string();
    lock.push(".lock");
    acquire_lock_file(
        &PathBuf::from(lock),
        timeout,
        "a Hermes shell-hook approval update",
    )
}

fn acquire_lock_file(path: &Path, timeout: Duration, contention: &str) -> Result<File, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let mut options = OpenOptions::new();
    options.create(true).truncate(false).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(path).map_err(|error| {
        format!(
            "failed to open Hermes install lock {}: {error}",
            path.display()
        )
    })?;
    let deadline = Instant::now() + timeout;
    loop {
        match try_lock_exclusive(&file) {
            Ok(LockAttempt::Acquired) => return Ok(file),
            Ok(LockAttempt::Contended) if Instant::now() < deadline => {
                thread::sleep(
                    INSTALL_LOCK_RETRY.min(deadline.saturating_duration_since(Instant::now())),
                );
            }
            Ok(LockAttempt::Contended) => {
                return Err(format!(
                    "timed out waiting for {contention} at {}; wait for it to finish and retry",
                    path.display()
                ));
            }
            Err(error) => {
                return Err(format!(
                    "failed to lock Hermes integration state {}: {error}",
                    path.display()
                ));
            }
        }
    }
}

pub(super) fn read_optional_utf8(path: &Path) -> Result<Option<String>, CliError> {
    match fs::read_to_string(path) {
        Ok(raw) => Ok(Some(raw)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(CliError::Install(format!(
            "failed to read {}: {error}",
            path.display()
        ))),
    }
}

pub(super) fn replace_optional_file<W>(
    path: &Path,
    bytes: Option<&[u8]>,
    write: &mut W,
) -> Result<(), String>
where
    W: FnMut(&Path, &[u8]) -> Result<(), String>,
{
    match bytes {
        Some(bytes) => write(path, bytes),
        None => remove_optional_file(path),
    }
}

pub(super) fn remove_optional_file(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("failed to remove {}: {error}", path.display())),
    }
}

pub(super) struct FileSnapshot {
    path: PathBuf,
    bytes: Option<Vec<u8>>,
}

impl FileSnapshot {
    pub(super) fn capture(path: &Path) -> Result<Self, CliError> {
        match fs::read(path) {
            Ok(bytes) => Ok(Self {
                path: path.to_path_buf(),
                bytes: Some(bytes),
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self {
                path: path.to_path_buf(),
                bytes: None,
            }),
            Err(error) => Err(CliError::Install(format!(
                "failed to snapshot {}: {error}",
                path.display()
            ))),
        }
    }

    pub(super) fn restore<W>(&self, write: &mut W) -> Result<(), String>
    where
        W: FnMut(&Path, &[u8]) -> Result<(), String>,
    {
        if let Some(bytes) = self.bytes.as_deref() {
            return write(&self.path, bytes);
        }
        remove_optional_file(&self.path)
    }
}
