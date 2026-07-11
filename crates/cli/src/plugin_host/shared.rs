// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Filesystem and platform helpers shared by host configuration.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use toml_edit::{DocumentMut, Item, Table};

pub(super) use crate::file_io::atomic_write;
pub(super) use crate::sidecar::{current_exe, healthz};

pub(crate) fn shell_quote(path: &Path) -> String {
    shell_quote_for_platform(path, cfg!(windows))
}

pub(crate) fn shell_quote_for_platform(path: &Path, windows: bool) -> String {
    shell_quote_arg_for_platform(&path.display().to_string(), windows)
}

pub(crate) fn shell_quote_arg_for_platform(raw: &str, windows: bool) -> String {
    if windows {
        return cmd_quote_arg(raw);
    }
    posix_quote_arg(raw)
}

fn posix_quote_arg(raw: &str) -> String {
    if raw.is_empty() {
        "''".into()
    } else if raw
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | ':' | '.' | '_' | '-'))
    {
        raw.to_string()
    } else {
        format!("'{}'", raw.replace('\'', "'\\''"))
    }
}

fn cmd_quote_arg(raw: &str) -> String {
    if raw.is_empty() {
        "\"\"".into()
    } else if raw.chars().all(|ch| {
        ch.is_ascii_alphanumeric()
            || matches!(ch, '/' | '\\' | ':' | '.' | '_' | '-' | '=' | '@' | '+')
    }) {
        raw.to_string()
    } else {
        let mut escaped = String::new();
        for ch in raw.chars() {
            match ch {
                // cmd expands percent variables even inside quotes. Insert a zero-length
                // substring expansion before the literal percent, matching Rust's hardened
                // batch-file encoder, so values such as `%USERPROFILE%` remain literal.
                '%' => escaped.push_str("%%cd:~,%%"),
                // Double quotes are represented by a paired quote inside a quoted cmd token.
                '"' => escaped.push_str("\"\""),
                _ => escaped.push(ch),
            }
        }
        // cmd metacharacters such as &, |, <, >, and ^ are literal inside this quote pair. A
        // caret inside the quotes would become part of the argument, so do not add one.
        format!("\"{escaped}\"")
    }
}

pub(super) fn ensure_table<'a>(doc: &'a mut DocumentMut, name: &str) -> &'a mut Table {
    if !doc.as_table().contains_key(name) || !doc[name].is_table() {
        doc[name] = Item::Table(Table::new());
    }
    doc[name].as_table_mut().expect("table was just inserted")
}

pub(super) fn read_json_object(path: &Path) -> Result<Value, String> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let value = serde_json::from_str::<Value>(&raw)
        .map_err(|error| format!("invalid JSON in {}: {error}", path.display()))?;
    if value.is_object() {
        Ok(value)
    } else {
        Err(format!("{} must contain a JSON object", path.display()))
    }
}

pub(super) fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    atomic_write(path, &bytes)
}

pub(super) fn backup(path: &Path) -> Result<(), String> {
    let backup = backup_path(path);
    if backup.exists() {
        return Ok(());
    }
    if path.exists() {
        fs::copy(path, &backup).map_err(|error| {
            format!(
                "failed to back up {} to {}: {error}",
                path.display(),
                backup.display()
            )
        })?;
    }
    Ok(())
}

pub(super) fn remove_backup(path: &Path) -> Result<(), String> {
    let backup = backup_path(path);
    match fs::remove_file(&backup) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("failed to remove {}: {error}", backup.display())),
    }
}

pub(super) fn backup_path(path: &Path) -> PathBuf {
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

pub(super) fn home_dir() -> Result<PathBuf, String> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or_else(|| "cannot determine home directory (set HOME or USERPROFILE)".into())
}

pub(super) fn print_check(label: &str, ok: bool) -> bool {
    println!("{} {label}", if ok { "ok" } else { "missing" });
    ok
}

pub(super) fn print_info(label: &str, message: &str) {
    println!("info {label}: {message}");
}

pub(super) struct FileSnapshot {
    path: PathBuf,
    bytes: Option<Vec<u8>>,
}

pub(super) fn snapshot_optional_file(path: &Path) -> Result<FileSnapshot, String> {
    match fs::read(path) {
        Ok(bytes) => Ok(FileSnapshot {
            path: path.to_path_buf(),
            bytes: Some(bytes),
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(FileSnapshot {
            path: path.to_path_buf(),
            bytes: None,
        }),
        Err(error) => Err(format!("failed to read {}: {error}", path.display())),
    }
}

pub(super) fn restore_file_snapshot(snapshot: &FileSnapshot) -> Result<(), String> {
    if let Some(bytes) = snapshot.bytes.as_deref() {
        return atomic_write(&snapshot.path, bytes);
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

#[cfg(windows)]
pub(crate) fn portable_executable_path(path: PathBuf) -> PathBuf {
    use std::ffi::OsString;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};

    let encoded = path.as_os_str().encode_wide().collect::<Vec<_>>();
    strip_windows_verbatim_prefix(&encoded)
        .map(|value| OsString::from_wide(&value))
        .map(PathBuf::from)
        .unwrap_or(path)
}

#[cfg(not(windows))]
pub(crate) fn portable_executable_path(path: PathBuf) -> PathBuf {
    path
}

#[cfg(any(test, windows))]
pub(crate) fn strip_windows_verbatim_prefix(encoded: &[u16]) -> Option<Vec<u16>> {
    const VERBATIM_PREFIX: &[u16] = &[b'\\' as u16, b'\\' as u16, b'?' as u16, b'\\' as u16];
    const VERBATIM_UNC_PREFIX: &[u16] = &[
        b'\\' as u16,
        b'\\' as u16,
        b'?' as u16,
        b'\\' as u16,
        b'U' as u16,
        b'N' as u16,
        b'C' as u16,
        b'\\' as u16,
    ];

    if let Some(rest) = encoded.strip_prefix(VERBATIM_UNC_PREFIX) {
        let mut normalized = vec![b'\\' as u16, b'\\' as u16];
        normalized.extend_from_slice(rest);
        Some(normalized)
    } else {
        encoded.strip_prefix(VERBATIM_PREFIX).map(ToOwned::to_owned)
    }
}
