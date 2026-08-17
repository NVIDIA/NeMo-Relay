// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional-file snapshots and stable backup-file management.

use std::fs;
use std::path::{Path, PathBuf};

use super::atomic::{atomic_write, atomic_write_with_permissions};
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
    symlink_target: Option<PathBuf>,
    #[cfg(windows)]
    dacl: Option<Vec<u8>>,
}

pub(crate) fn snapshot_optional_file(path: &Path) -> Result<FileSnapshot, String> {
    let symlink_target = symlink_target(path)?;
    match fs::read(path) {
        Ok(bytes) => Ok(FileSnapshot {
            path: path.to_path_buf(),
            bytes: Some(bytes),
            permissions: fs::metadata(path).ok().map(|value| value.permissions()),
            symlink_target,
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
            symlink_target,
            #[cfg(windows)]
            dacl: None,
        }),
        Err(error) => Err(format!("failed to read {}: {error}", path.display())),
    }
}

pub(crate) fn restore_file_snapshot(snapshot: &FileSnapshot) -> Result<(), String> {
    let path = restore_path(snapshot)?;
    if let Some(bytes) = snapshot.bytes.as_deref() {
        #[cfg(windows)]
        if let Some(dacl) = snapshot.dacl.as_deref() {
            return atomic_write_with_windows_dacl(&path, bytes, dacl);
        }
        return atomic_write_with_permissions(&path, bytes, snapshot.permissions.as_ref());
    }
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("failed to remove {}: {error}", path.display())),
    }
}

pub(crate) fn atomic_write_preserving_symlink(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let path = resolved_symlink_target_path(path)?;
    atomic_write(&path, bytes)
}

pub(crate) fn remove_file_preserving_symlink(path: &Path) -> Result<(), String> {
    let path = resolved_symlink_target_path(path)?;
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("failed to remove {}: {error}", path.display())),
    }
}

fn restore_path(snapshot: &FileSnapshot) -> Result<PathBuf, String> {
    if let Some(link_target) = snapshot.symlink_target.as_deref() {
        ensure_symlink_path(&snapshot.path, link_target)?;
        resolved_symlink_target_path(&snapshot.path)
    } else {
        Ok(snapshot.path.clone())
    }
}

fn symlink_target(path: &Path) -> Result<Option<PathBuf>, String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => fs::read_link(path)
            .map(Some)
            .map_err(|error| format!("failed to read symlink {}: {error}", path.display())),
        Ok(_) => Ok(None),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!(
            "failed to inspect {} for symlink metadata: {error}",
            path.display()
        )),
    }
}

fn resolved_symlink_target_path(path: &Path) -> Result<PathBuf, String> {
    let mut current = path.to_path_buf();
    for _ in 0..40 {
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                let target = fs::read_link(&current).map_err(|error| {
                    format!("failed to read symlink {}: {error}", current.display())
                })?;
                current = if target.is_absolute() {
                    target
                } else {
                    current
                        .parent()
                        .unwrap_or_else(|| Path::new("."))
                        .join(target)
                };
            }
            Ok(_) => return Ok(current),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(current),
            Err(error) => {
                return Err(format!(
                    "failed to inspect {} for symlink metadata: {error}",
                    current.display()
                ));
            }
        }
    }
    Err(format!(
        "too many symlink hops while resolving {}",
        path.display()
    ))
}

pub(crate) fn ensure_symlink_path(path: &Path, target: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            let current = fs::read_link(path)
                .map_err(|error| format!("failed to read symlink {}: {error}", path.display()))?;
            if current == target {
                return Ok(());
            }
            fs::remove_file(path)
                .map_err(|error| format!("failed to remove {}: {error}", path.display()))?;
        }
        Ok(_) => {
            fs::remove_file(path)
                .map_err(|error| format!("failed to remove {}: {error}", path.display()))?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "failed to inspect {} for symlink metadata: {error}",
                path.display()
            ));
        }
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    create_symlink(target, path)
}

#[cfg(unix)]
fn create_symlink(target: &Path, path: &Path) -> Result<(), String> {
    std::os::unix::fs::symlink(target, path)
        .map_err(|error| format!("failed to create symlink {}: {error}", path.display()))
}

#[cfg(windows)]
fn create_symlink(target: &Path, path: &Path) -> Result<(), String> {
    std::os::windows::fs::symlink_file(target, path)
        .map_err(|error| format!("failed to create symlink {}: {error}", path.display()))
}

#[cfg(not(any(unix, windows)))]
fn create_symlink(_target: &Path, path: &Path) -> Result<(), String> {
    Err(format!(
        "failed to create symlink {}: unsupported platform",
        path.display()
    ))
}
