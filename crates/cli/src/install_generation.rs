// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private install-generation fencing for lifecycle-bound Codex MCP supervisors.

use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use crate::file_io::{LockAttempt, try_lock_exclusive, try_lock_shared};

pub(crate) const GENERATION_FILE_ENV: &str = "NEMO_RELAY_MCP_GENERATION_FILE";
pub(crate) const GENERATION_FILE_NAME: &str = ".nemo-relay-generation";
const MAX_GENERATION_TOKEN_BYTES: usize = 128;
const RETIRED_GENERATION_PREFIX: &str = "retired:";
const DEFAULT_GENERATION_LOCK_TIMEOUT: Duration = Duration::from_secs(5);
const GENERATION_LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct InstallGeneration {
    path: PathBuf,
    token: String,
}

impl InstallGeneration {
    pub(crate) fn capture_from_env() -> Result<Option<Self>, String> {
        env::var_os(GENERATION_FILE_ENV)
            .map(PathBuf::from)
            .map(Self::capture)
            .transpose()
    }

    pub(crate) fn capture(path: PathBuf) -> Result<Self, String> {
        let file = open_generation(&path)?;
        lock_shared_with_timeout(&file, &path, DEFAULT_GENERATION_LOCK_TIMEOUT)?;
        let token = read_generation_file(&file, &path)?;
        let current = read_generation_path(&path)?;
        if token != current {
            return Err(retired_generation_error(&path));
        }
        Ok(Self { path, token })
    }

    pub(crate) fn verify_current(&self) -> Result<(), String> {
        let file = open_generation(&self.path).map_err(|_| retired_generation_error(&self.path))?;
        lock_shared_with_timeout(&file, &self.path, DEFAULT_GENERATION_LOCK_TIMEOUT)
            .map_err(|_| retired_generation_error(&self.path))?;
        let locked = read_generation_file(&file, &self.path)
            .map_err(|_| retired_generation_error(&self.path))?;
        let current =
            read_generation_path(&self.path).map_err(|_| retired_generation_error(&self.path))?;
        if locked != self.token || current != self.token {
            return Err(retired_generation_error(&self.path));
        }
        Ok(())
    }
}

pub(crate) struct GenerationRetirement {
    lock: Option<File>,
    path: PathBuf,
    original: GenerationMarker,
    changed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum GenerationMarker {
    Active(String),
    Retired(String),
}

impl GenerationMarker {
    fn token(&self) -> &str {
        match self {
            Self::Active(token) | Self::Retired(token) => token,
        }
    }

    fn encoded(&self) -> String {
        match self {
            Self::Active(token) => format!("{token}\n"),
            Self::Retired(token) => format!("{RETIRED_GENERATION_PREFIX}{token}\n"),
        }
    }

    fn is_retired(&self) -> bool {
        matches!(self, Self::Retired(_))
    }
}

impl GenerationRetirement {
    pub(crate) fn acquire(path: &Path) -> Result<Option<Self>, String> {
        Self::acquire_with_timeout(path, DEFAULT_GENERATION_LOCK_TIMEOUT)
    }

    pub(crate) fn acquire_with_timeout(
        path: &Path,
        timeout: Duration,
    ) -> Result<Option<Self>, String> {
        let file = match OpenOptions::new().read(true).write(true).open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(format!(
                    "failed to open MCP install generation {}: {error}",
                    path.display()
                ));
            }
        };
        lock_exclusive_with_timeout(&file, path, timeout)?;
        let original = read_generation_marker(&file, path)?;
        Ok(Some(Self {
            lock: Some(file),
            path: path.to_owned(),
            original,
            changed: false,
        }))
    }

    /// Persistently invalidate this generation and release its exclusive lock before moving the
    /// plugin tree. A retired marker cannot be adopted by an old or newly launched MCP process,
    /// but remains recognizable so an interrupted uninstall can resume.
    pub(crate) fn invalidate_for_replacement(&mut self) -> Result<(), String> {
        if self.original.is_retired() || self.changed {
            self.lock = None;
            return Ok(());
        }
        let retired = GenerationMarker::Retired(self.original.token().to_owned());
        let file = self.lock.as_mut().ok_or_else(|| {
            format!(
                "MCP install generation {} is not locked",
                self.path.display()
            )
        })?;
        self.changed = true;
        replace_generation_marker(file, &self.path, &retired, "invalidate")?;
        self.lock = None;
        Ok(())
    }

    /// Restore an invalidated marker before a rolled-back plugin is registered again.
    pub(crate) fn restore_after_rollback(&mut self) -> Result<(), String> {
        if !self.changed {
            self.lock = None;
            return Ok(());
        }
        let mut file = match self.lock.take() {
            Some(file) => file,
            None => OpenOptions::new()
                .write(true)
                .open(&self.path)
                .map_err(|error| {
                    format!(
                        "failed to reopen MCP install generation {} for rollback: {error}",
                        self.path.display()
                    )
                })?,
        };
        replace_generation_marker(&mut file, &self.path, &self.original, "restore")?;
        self.changed = false;
        Ok(())
    }
}

fn replace_generation_marker(
    file: &mut File,
    path: &Path,
    marker: &GenerationMarker,
    operation: &str,
) -> Result<(), String> {
    write_generation_marker(file, marker).map_err(|error| {
        format!(
            "failed to {operation} MCP install generation {}: {error}",
            path.display()
        )
    })
}

fn write_generation_marker(file: &mut File, marker: &GenerationMarker) -> std::io::Result<()> {
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(marker.encoded().as_bytes())?;
    file.sync_all()
}

fn lock_shared_with_timeout(file: &File, path: &Path, timeout: Duration) -> Result<(), String> {
    lock_with_timeout(file, path, timeout, false)
}

fn lock_exclusive_with_timeout(file: &File, path: &Path, timeout: Duration) -> Result<(), String> {
    lock_with_timeout(file, path, timeout, true)
}

fn lock_with_timeout(
    file: &File,
    path: &Path,
    timeout: Duration,
    exclusive: bool,
) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    loop {
        let result = if exclusive {
            try_lock_exclusive(file)
        } else {
            try_lock_shared(file)
        };
        match result {
            Ok(LockAttempt::Acquired) => return Ok(()),
            Ok(LockAttempt::Contended) => {
                if Instant::now() >= deadline {
                    return Err(format!(
                        "timed out waiting for MCP install generation lock {}",
                        path.display()
                    ));
                }
                thread::sleep(GENERATION_LOCK_RETRY_INTERVAL.min(timeout));
            }
            Err(error) => {
                return Err(format!(
                    "failed to lock MCP install generation {}: {error}",
                    path.display()
                ));
            }
        }
    }
}

pub(crate) fn write_new_generation(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    fs::write(path, format!("{}\n", uuid::Uuid::now_v7()))
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn open_generation(path: &Path) -> Result<File, String> {
    OpenOptions::new().read(true).open(path).map_err(|error| {
        format!(
            "failed to open MCP install generation {}: {error}",
            path.display()
        )
    })
}

fn read_generation_path(path: &Path) -> Result<String, String> {
    let file = open_generation(path)?;
    read_generation_file(&file, path)
}

fn read_generation_file(file: &File, path: &Path) -> Result<String, String> {
    match read_generation_marker(file, path)? {
        GenerationMarker::Active(token) => Ok(token),
        GenerationMarker::Retired(_) => Err(retired_generation_error(path)),
    }
}

fn read_generation_marker(file: &File, path: &Path) -> Result<GenerationMarker, String> {
    let mut raw = String::new();
    file.take(MAX_GENERATION_TOKEN_BYTES.saturating_add(1) as u64)
        .read_to_string(&mut raw)
        .map_err(|error| {
            format!(
                "failed to read MCP install generation {}: {error}",
                path.display()
            )
        })?;
    if raw.len() > MAX_GENERATION_TOKEN_BYTES {
        return Err(format!(
            "MCP install generation {} exceeds the {MAX_GENERATION_TOKEN_BYTES}-byte limit",
            path.display()
        ));
    }
    let token = raw.trim();
    if token.is_empty() {
        return Err(format!(
            "MCP install generation {} is empty",
            path.display()
        ));
    }
    match token.strip_prefix(RETIRED_GENERATION_PREFIX) {
        Some("") => Err(format!(
            "MCP install generation {} has a retired marker without a token",
            path.display()
        )),
        Some(token) => Ok(GenerationMarker::Retired(token.to_owned())),
        None => Ok(GenerationMarker::Active(token.to_owned())),
    }
}

fn retired_generation_error(path: &Path) -> String {
    format!(
        "Codex plugin MCP install generation at {} has been retired",
        path.display()
    )
}

#[cfg(test)]
#[path = "../tests/coverage/install_generation_tests.rs"]
mod tests;
