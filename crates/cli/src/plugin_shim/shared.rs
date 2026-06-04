// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared plugin-shim filesystem, sidecar, HTTP, and formatting helpers.

use std::env;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::thread;
use std::time::Duration;

use serde_json::{Value, json};
use toml_edit::{DocumentMut, Item, Table};

use crate::config::CodingAgent;

use super::{DEFAULT_URL, HEALTHZ_TIMEOUT, STALE_LOCK_AFTER};

pub(super) fn ensure_sidecar(agent: CodingAgent, url: &str) {
    if healthz(url) {
        return;
    }
    let runtime = runtime_dir();
    let _ = fs::create_dir_all(&runtime);
    let lock = runtime.join(format!("{}-sidecar.lock", agent.as_arg()));
    let mut acquired = false;
    for _ in 0..40 {
        match fs::create_dir(&lock) {
            Ok(()) => {
                acquired = true;
                break;
            }
            Err(_) if healthz(url) => return,
            Err(_) if repair_stale_lock(&lock) => continue,
            Err(_) => thread::sleep(Duration::from_millis(50)),
        }
    }
    if !acquired {
        eprintln!("nemo-relay sidecar lock timed out");
        return;
    }
    let result = start_sidecar(agent, url, &runtime);
    let _ = fs::remove_dir(&lock);
    if let Err(error) = result {
        eprintln!("{error}");
    }
}

pub(super) fn repair_stale_lock(lock: &Path) -> bool {
    repair_stale_lock_after(lock, STALE_LOCK_AFTER)
}

pub(super) fn repair_stale_lock_after(lock: &Path, stale_after: Duration) -> bool {
    if !lock.exists() || !lock_is_old(lock, stale_after) {
        return false;
    }
    match fs::remove_dir_all(lock) {
        Ok(()) => return true,
        Err(error) => eprintln!("failed to repair stale nemo-relay sidecar lock: {error}"),
    }
    false
}

pub(super) fn lock_is_old(lock: &Path, stale_after: Duration) -> bool {
    lock.metadata()
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|elapsed| elapsed >= stale_after)
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

pub(super) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
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
pub(super) fn replace_file(tmp: &Path, path: &Path) -> Result<(), String> {
    fs::rename(tmp, path).map_err(|error| format!("failed to replace {}: {error}", path.display()))
}

#[cfg(windows)]
pub(super) fn replace_file(tmp: &Path, path: &Path) -> Result<(), String> {
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
pub(super) fn replace_backup_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("config");
    path.with_file_name(format!(".{file_name}.nemo-relay-replace.tmp"))
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

pub(super) fn start_sidecar(agent: CodingAgent, url: &str, runtime: &Path) -> Result<(), String> {
    if healthz(url) {
        return Ok(());
    }
    let (_, port) = parse_loopback_url(url)?;
    let bind = format!("127.0.0.1:{port}");
    let relay = relay_binary()?;
    let log_path = runtime.join(format!("{}-sidecar.log", agent.as_arg()));
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|error| format!("failed to open {}: {error}", log_path.display()))?;
    let err_log = log
        .try_clone()
        .map_err(|error| format!("failed to clone sidecar log handle: {error}"))?;
    let mut child = Command::new(relay)
        .arg("--bind")
        .arg(bind)
        .env("NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS", plugin_idle_timeout())
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(err_log))
        .spawn()
        .map_err(|error| format!("failed to spawn nemo-relay sidecar: {error}"))?;
    let pid_path = runtime.join(format!("{}-sidecar.pid", agent.as_arg()));
    let _ = fs::write(&pid_path, child.id().to_string());
    for _ in 0..50 {
        if healthz(url) {
            return Ok(());
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                let _ = fs::remove_file(&pid_path);
                return Err(format!(
                    "nemo-relay sidecar exited before becoming ready at {url}: {status}"
                ));
            }
            Ok(None) => {}
            Err(error) => {
                let _ = fs::remove_file(&pid_path);
                return Err(format!(
                    "failed to inspect nemo-relay sidecar process: {error}"
                ));
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
    terminate_unready_sidecar(child, &pid_path, url)
}

pub(super) fn terminate_unready_sidecar(
    mut child: std::process::Child,
    pid_path: &Path,
    url: &str,
) -> Result<(), String> {
    match child.try_wait() {
        Ok(Some(status)) => {
            let _ = fs::remove_file(pid_path);
            return Err(format!(
                "nemo-relay sidecar exited before becoming ready at {url}: {status}"
            ));
        }
        Ok(None) => {}
        Err(error) => {
            let _ = fs::remove_file(pid_path);
            return Err(format!(
                "failed to inspect nemo-relay sidecar process: {error}"
            ));
        }
    }
    if let Err(error) = child.kill() {
        let _ = fs::remove_file(pid_path);
        return Err(format!(
            "nemo-relay sidecar did not become ready at {url}; failed to terminate startup process: {error}"
        ));
    }
    let _ = child.wait();
    let _ = fs::remove_file(pid_path);
    Err(format!(
        "nemo-relay sidecar did not become ready at {url}; terminated startup process"
    ))
}

pub(super) fn post_hook(agent: CodingAgent, url: &str, payload: &[u8]) -> Result<Vec<u8>, String> {
    let hook_path = match agent {
        CodingAgent::ClaudeCode => "/hooks/claude-code",
        CodingAgent::Codex => "/hooks/codex",
        _ => {
            return Err(format!(
                "plugin shim hook forwarding supports claude and codex, got {}",
                agent.as_arg()
            ));
        }
    };
    let (host, port) = parse_loopback_url(url)?;
    let addrs = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|error| format!("hook forward failed: {error}"))?;
    let mut stream = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, Duration::from_secs(2)) {
            Ok(candidate) => {
                stream = Some(candidate);
                break;
            }
            Err(_) => continue,
        }
    }
    let Some(mut stream) = stream else {
        return Err("hook forward failed: connection timed out".into());
    };
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| format!("failed to set read timeout: {error}"))?;
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| format!("failed to set write timeout: {error}"))?;
    let request = format!(
        "POST {hook_path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    stream
        .write_all(request.as_bytes())
        .and_then(|_| stream.write_all(payload))
        .map_err(|error| format!("hook forward failed: {error}"))?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|error| format!("hook forward failed: {error}"))?;
    parse_http_response(&response)
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

pub(super) fn healthz(url: &str) -> bool {
    let Ok((host, port)) = parse_loopback_url(url) else {
        return false;
    };
    let Ok(addrs) = (host.as_str(), port).to_socket_addrs() else {
        return false;
    };
    let mut stream = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, HEALTHZ_TIMEOUT) {
            Ok(candidate) => {
                stream = Some(candidate);
                break;
            }
            Err(_) => continue,
        }
    }
    let Some(mut stream) = stream else {
        return false;
    };
    if stream.set_read_timeout(Some(HEALTHZ_TIMEOUT)).is_err()
        || stream.set_write_timeout(Some(HEALTHZ_TIMEOUT)).is_err()
    {
        return false;
    }
    let request =
        format!("GET /healthz HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = [0_u8; 32];
    stream
        .read(&mut response)
        .ok()
        .is_some_and(|count| response[..count].starts_with(b"HTTP/1.1 200"))
}

pub(super) fn parse_loopback_url(url: &str) -> Result<(String, u16), String> {
    let without_scheme = url
        .strip_prefix("http://")
        .ok_or_else(|| format!("plugin shim only supports http loopback URLs: {url}"))?;
    let authority = without_scheme.split('/').next().unwrap_or(without_scheme);
    let (host, port) = authority
        .rsplit_once(':')
        .ok_or_else(|| format!("missing port in gateway URL: {url}"))?;
    if host != "127.0.0.1" && host != "localhost" {
        return Err(format!(
            "plugin shim only supports loopback gateway URLs: {url}"
        ));
    }
    let port = port
        .parse::<u16>()
        .map_err(|error| format!("invalid gateway port in {url}: {error}"))?;
    Ok((host.to_string(), port))
}

pub(super) fn gateway_url(agent: CodingAgent, explicit: Option<&str>) -> String {
    if let Some(url) = explicit {
        return url.to_string();
    }
    if matches!(agent, CodingAgent::ClaudeCode)
        && let Ok(url) = env::var("NEMO_RELAY_GATEWAY_URL")
    {
        return url;
    }
    env::var("NEMO_RELAY_PLUGIN_GATEWAY_URL").unwrap_or_else(|_| DEFAULT_URL.into())
}

pub(super) fn relay_binary() -> Result<PathBuf, String> {
    if let Ok(path) = env::var("NEMO_RELAY_PLUGIN_BINARY") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
        return Err(format!(
            "NEMO_RELAY_PLUGIN_BINARY does not exist: {}",
            path.display()
        ));
    }
    current_exe()
}

pub(super) fn current_exe() -> Result<PathBuf, String> {
    env::current_exe().map_err(|error| format!("failed to resolve current executable: {error}"))
}

pub(super) fn runtime_dir() -> PathBuf {
    env::var_os("XDG_RUNTIME_DIR")
        .or_else(|| env::var_os("TMPDIR"))
        .or_else(|| env::var_os("TEMP"))
        .map(PathBuf::from)
        .unwrap_or_else(env::temp_dir)
        .join("nemo-relay-plugin")
}

pub(super) fn plugin_idle_timeout() -> String {
    env::var("NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS").unwrap_or_else(|_| "300".into())
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
