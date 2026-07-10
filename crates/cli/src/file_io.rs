// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small, platform-aware filesystem primitives shared by CLI subsystems.

use std::fs::{self, File};
use std::io;
use std::path::Path;
#[cfg(windows)]
use std::path::PathBuf;

use fs2::FileExt;

/// Result of one nonblocking advisory-file-lock attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LockAttempt {
    Acquired,
    Contended,
}

/// Attempt an exclusive advisory lock without waiting.
pub(crate) fn try_lock_exclusive(file: &File) -> io::Result<LockAttempt> {
    normalize_lock_attempt(FileExt::try_lock_exclusive(file))
}

/// Attempt a shared advisory lock without waiting.
pub(crate) fn try_lock_shared(file: &File) -> io::Result<LockAttempt> {
    normalize_lock_attempt(FileExt::try_lock_shared(file))
}

fn normalize_lock_attempt(result: io::Result<()>) -> io::Result<LockAttempt> {
    match result {
        Ok(()) => Ok(LockAttempt::Acquired),
        Err(error) if lock_is_contended(&error) => Ok(LockAttempt::Contended),
        Err(error) => Err(error),
    }
}

fn lock_is_contended(error: &io::Error) -> bool {
    if error.kind() == io::ErrorKind::WouldBlock {
        return true;
    }
    #[cfg(windows)]
    {
        error.raw_os_error() == Some(windows_sys::Win32::Foundation::ERROR_LOCK_VIOLATION as i32)
    }
    #[cfg(not(windows))]
    {
        false
    }
}

/// Atomically replace `path` with `bytes`, creating its parent directory when needed.
pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let tmp = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(|value| value.to_str())
            .map(|value| format!("{value}."))
            .unwrap_or_default()
    ));
    fs::write(&tmp, bytes)
        .map_err(|error| format!("failed to write {}: {error}", tmp.display()))?;
    replace_file(&tmp, path)
}

#[cfg(not(windows))]
fn replace_file(tmp: &Path, path: &Path) -> Result<(), String> {
    fs::rename(tmp, path).map_err(|error| format!("failed to replace {}: {error}", path.display()))
}

#[cfg(windows)]
fn replace_file(tmp: &Path, path: &Path) -> Result<(), String> {
    if !path.exists() {
        return fs::rename(tmp, path)
            .map_err(|error| format!("failed to replace {}: {error}", path.display()));
    }

    let backup = replace_backup_path(path);
    match fs::remove_file(&backup) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "failed to remove stale replacement backup {}: {error}",
                backup.display()
            ));
        }
    }

    match fs::rename(path, &backup) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return fs::rename(tmp, path)
                .map_err(|error| format!("failed to replace {}: {error}", path.display()));
        }
        Err(error) => {
            return Err(format!(
                "failed to prepare replacement for {}: {error}",
                path.display()
            ));
        }
    }

    match fs::rename(tmp, path) {
        Ok(()) => {
            let _ = fs::remove_file(&backup);
            Ok(())
        }
        Err(error) => match fs::rename(&backup, path) {
            Ok(()) => Err(format!("failed to replace {}: {error}", path.display())),
            Err(restore_error) => Err(format!(
                "failed to replace {}: {error}; additionally failed to restore {}: {restore_error}",
                path.display(),
                backup.display()
            )),
        },
    }
}

#[cfg(windows)]
fn replace_backup_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("config");
    path.with_file_name(format!(".{file_name}.nemo-relay-replace.tmp"))
}

#[cfg(test)]
#[path = "../tests/coverage/file_io_tests.rs"]
mod tests;
