// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Filesystem, hook transport, and process helpers shared by plugin shims.

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Duration;

use serde_json::{Value, json};
use toml_edit::{DocumentMut, Item, Table};

use crate::config::CodingAgent;
pub(super) use crate::file_io::atomic_write;
use crate::sidecar::{DEFAULT_URL, loopback_authority, parse_loopback_url};
pub(super) use crate::sidecar::{current_exe, healthz, plugin_idle_timeout, relay_binary};

pub(super) const MAX_HOOK_RESPONSE_BYTES: usize = 1024 * 1024;

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
    if raw.chars().all(|ch| {
        ch.is_ascii_alphanumeric()
            || matches!(ch, '/' | '\\' | ':' | '.' | '_' | '-' | '=' | '@' | '+')
    }) {
        raw.to_string()
    } else {
        let mut escaped = String::new();
        for ch in raw.chars() {
            match ch {
                '%' => escaped.push_str("%%"),
                '"' | '^' | '&' | '|' | '<' | '>' => {
                    escaped.push('^');
                    escaped.push(ch);
                }
                _ => escaped.push(ch),
            }
        }
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct HookForwardError {
    message: String,
    retryable: bool,
}

impl HookForwardError {
    pub(super) fn retryable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: true,
        }
    }

    pub(super) fn not_retryable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
        }
    }

    pub(super) fn is_retryable(&self) -> bool {
        self.retryable
    }
}

impl std::fmt::Display for HookForwardError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

pub(super) fn post_hook(
    agent: CodingAgent,
    url: &str,
    payload: &[u8],
) -> Result<Vec<u8>, HookForwardError> {
    let hook_path = match agent {
        CodingAgent::ClaudeCode => "/hooks/claude-code",
        CodingAgent::Codex => "/hooks/codex",
        CodingAgent::Hermes => "/hooks/hermes",
    };
    let (host, port) = parse_loopback_url(url).map_err(HookForwardError::not_retryable)?;
    let addrs = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|error| HookForwardError::retryable(format!("hook forward failed: {error}")))?;
    let mut stream = None;
    let mut connect_error = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, Duration::from_secs(2)) {
            Ok(candidate) => {
                stream = Some(candidate);
                break;
            }
            Err(error) => connect_error = Some(error),
        }
    }
    let Some(mut stream) = stream else {
        let detail = connect_error
            .map(|error| error.to_string())
            .unwrap_or_else(|| "no loopback address resolved".into());
        return Err(HookForwardError::retryable(format!(
            "hook forward failed before sending request bytes: {detail}"
        )));
    };
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| {
            HookForwardError::not_retryable(format!("failed to set read timeout: {error}"))
        })?;
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| {
            HookForwardError::not_retryable(format!("failed to set write timeout: {error}"))
        })?;
    let request = format!(
        "POST {hook_path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        loopback_authority(&host, port),
        payload.len()
    );
    stream
        .write_all(request.as_bytes())
        .and_then(|_| stream.write_all(payload))
        .map_err(|error| {
            HookForwardError::not_retryable(format!("hook forward failed: {error}"))
        })?;
    let mut response = Vec::new();
    stream
        .take(MAX_HOOK_RESPONSE_BYTES.saturating_add(1) as u64)
        .read_to_end(&mut response)
        .map_err(|error| {
            HookForwardError::not_retryable(format!("hook forward failed: {error}"))
        })?;
    if response.len() > MAX_HOOK_RESPONSE_BYTES {
        return Err(HookForwardError::not_retryable(format!(
            "hook forward response exceeds the {MAX_HOOK_RESPONSE_BYTES}-byte limit"
        )));
    }
    parse_http_response(&response).map_err(HookForwardError::not_retryable)
}

pub(super) fn parse_http_response(response: &[u8]) -> Result<Vec<u8>, String> {
    let Some(split) = response.windows(4).position(|window| window == b"\r\n\r\n") else {
        return Err("hook forward failed: malformed HTTP response".into());
    };
    let headers = &response[..split];
    let body = response[split + 4..].to_vec();
    let status_line = headers
        .split(|byte| *byte == b'\n')
        .next()
        .and_then(|line| std::str::from_utf8(line).ok())
        .unwrap_or_default();
    let status_code = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u16>().ok());
    if status_code.is_some_and(|code| (200..=299).contains(&code)) {
        Ok(body)
    } else {
        Err(format!(
            "nemo-relay hook forward failed with {}",
            status_line.trim()
        ))
    }
}

pub(super) fn gateway_url(agent: CodingAgent, explicit: Option<&str>) -> String {
    if let Some(url) = explicit {
        return url.to_string();
    }
    if matches!(agent, CodingAgent::ClaudeCode | CodingAgent::Hermes)
        && let Ok(url) = env::var("NEMO_RELAY_GATEWAY_URL")
    {
        return url;
    }
    env::var("NEMO_RELAY_PLUGIN_GATEWAY_URL").unwrap_or_else(|_| DEFAULT_URL.into())
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

pub(super) fn fail_closed() -> bool {
    env::var("NEMO_RELAY_FAIL_CLOSED").ok().as_deref() == Some("1")
}

pub(super) trait ExecOrStatus {
    fn exec_or_status(&mut self) -> std::io::Result<ExitCode>;
}

#[cfg(unix)]
impl ExecOrStatus for Command {
    fn exec_or_status(&mut self) -> std::io::Result<ExitCode> {
        use std::os::unix::process::CommandExt;
        let error = self.exec();
        Err(error)
    }
}

#[cfg(not(unix))]
impl ExecOrStatus for Command {
    fn exec_or_status(&mut self) -> std::io::Result<ExitCode> {
        let status = self.status()?;
        Ok(status
            .code()
            .and_then(|code| u8::try_from(code).ok())
            .map(ExitCode::from)
            .unwrap_or(ExitCode::FAILURE))
    }
}
