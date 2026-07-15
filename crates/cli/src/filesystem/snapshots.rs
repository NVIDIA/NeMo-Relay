// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional-file snapshots and stable backup-file management.

use std::fs;
use std::path::{Path, PathBuf};

use super::atomic_write_with_permissions;
#[cfg(windows)]
use super::{atomic_write_with_windows_dacl, read_windows_dacl};

pub(crate) fn backup(path: &Path) -> Result<(), String> {
    let backup = backup_path(path);
    if backup.exists() {
        return Ok(());
    }
    if path.exists() {
        let bytes = fs::read(path)
            .map_err(|error| format!("failed to read {} for backup: {error}", path.display()))?;
        #[cfg(windows)]
        {
            let dacl = read_windows_dacl(path).map_err(|error| {
                format!(
                    "failed to read access control for {}: {error}",
                    path.display()
                )
            })?;
            atomic_write_with_windows_dacl(&backup, &bytes, &dacl)?;
        }
        #[cfg(not(windows))]
        {
            let permissions = fs::metadata(path)
                .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
                .permissions();
            atomic_write_with_permissions(&backup, &bytes, Some(&permissions))?;
        }
    }
    Ok(())
}

pub(crate) fn remove_backup(path: &Path) -> Result<(), String> {
    let backup = backup_path(path);
    match fs::remove_file(&backup) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("failed to remove {}: {error}", backup.display())),
    }
}

pub(crate) fn backup_path(path: &Path) -> PathBuf {
    let mut extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_string();
    if extension.is_empty() {
        extension = "nemo-relay.bak".into();
    } else {
        extension.push_str(".nemo-relay.bak");
    }
    path.with_extension(extension)
}

pub(crate) struct FileSnapshot {
    path: PathBuf,
    bytes: Option<Vec<u8>>,
    permissions: Option<fs::Permissions>,
    #[cfg(windows)]
    dacl: Option<Vec<u8>>,
}

pub(crate) fn snapshot_optional_file(path: &Path) -> Result<FileSnapshot, String> {
    match fs::read(path) {
        Ok(bytes) => Ok(FileSnapshot {
            path: path.to_path_buf(),
            bytes: Some(bytes),
            permissions: fs::metadata(path).ok().map(|value| value.permissions()),
            #[cfg(windows)]
            dacl: Some(read_windows_dacl(path).map_err(|error| {
                format!(
                    "failed to read access control for {}: {error}",
                    path.display()
                )
            })?),
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(FileSnapshot {
            path: path.to_path_buf(),
            bytes: None,
            permissions: None,
            #[cfg(windows)]
            dacl: None,
        }),
        Err(error) => Err(format!("failed to read {}: {error}", path.display())),
    }
}

pub(crate) fn restore_file_snapshot(snapshot: &FileSnapshot) -> Result<(), String> {
    if let Some(bytes) = snapshot.bytes.as_deref() {
        #[cfg(windows)]
        if let Some(dacl) = snapshot.dacl.as_deref() {
            return atomic_write_with_windows_dacl(&snapshot.path, bytes, dacl);
        }
        return atomic_write_with_permissions(&snapshot.path, bytes, snapshot.permissions.as_ref());
    }
    match fs::remove_file(&snapshot.path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "failed to remove {}: {error}",
            snapshot.path.display()
        )),
    }
}
