// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::ffi::{OsStr, OsString};
use std::io::{Read, Write};
use std::net::TcpListener;

struct EnvScope {
    _guard: std::sync::MutexGuard<'static, ()>,
    previous: Vec<(&'static str, Option<OsString>)>,
}

impl EnvScope {
    fn set(values: &[(&'static str, Option<&OsStr>)]) -> Self {
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = values
            .iter()
            .map(|(name, _)| (*name, std::env::var_os(name)))
            .collect();
        for (name, value) in values {
            // SAFETY: The process-wide environment lock is held for this scope.
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
        Self {
            _guard: guard,
            previous,
        }
    }
}

impl Drop for EnvScope {
    fn drop(&mut self) {
        for (name, value) in self.previous.drain(..) {
            // SAFETY: The process-wide environment lock remains held during restoration.
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}

fn read_headers(stream: &mut std::net::TcpStream) -> String {
    let mut bytes = Vec::new();
    let mut byte = [0_u8; 1];
    while !bytes.ends_with(b"\r\n\r\n") {
        stream.read_exact(&mut byte).unwrap();
        bytes.push(byte[0]);
    }
    String::from_utf8(bytes).unwrap()
}

fn header(request: &str, name: &str) -> String {
    request
        .lines()
        .find_map(|line| {
            let (candidate, value) = line.split_once(':')?;
            candidate
                .eq_ignore_ascii_case(name)
                .then(|| value.trim().to_string())
        })
        .unwrap()
}

#[test]
fn owner_records_are_versioned_endpoint_scoped_and_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let url = "http://127.0.0.1:47632";
    let path = owner_path(dir.path(), url);
    let record = OwnerRecord::new(42, url, "shutdown", Some("fingerprint"));

    write_owner_record(&path, &record).unwrap();

    assert_eq!(read_owner_record(&path).unwrap(), Some(record.clone()));
    assert!(record.valid_for(url));
    assert!(!record.valid_for("http://127.0.0.1:47633"));
    assert!(owner_path(dir.path(), url).ends_with("sidecar-127.0.0.1-47632.owner.json"));
    assert_eq!(lock_name("not a url/with spaces"), "not_a_url_with_spaces");
}

#[test]
fn recovery_records_preserve_pending_and_ready_attempts() {
    let dir = tempfile::tempdir().unwrap();
    let url = "http://127.0.0.1:47632";
    let pending = RecoveryRecord {
        from_instance: "first".into(),
        endpoint_url: String::new(),
        to_instance: String::new(),
    };
    write_recovery(dir.path(), url, &pending).unwrap();
    assert_eq!(read_recovery(dir.path(), url).unwrap(), Some(pending));

    let ready = RecoveryRecord {
        from_instance: "first".into(),
        endpoint_url: url.into(),
        to_instance: "second".into(),
    };
    write_recovery(dir.path(), url, &ready).unwrap();
    assert_eq!(read_recovery(dir.path(), url).unwrap(), Some(ready));
}

#[test]
fn startup_lock_serializes_competing_mcp_processes() {
    let dir = tempfile::tempdir().unwrap();
    let url = "http://127.0.0.1:47632";
    let owner = lock_endpoint(dir.path(), url).unwrap();

    let error = lock_endpoint_for(dir.path(), url, Duration::from_millis(25)).unwrap_err();
    assert!(error.contains("timed out waiting"), "{error}");

    drop(owner);
    lock_endpoint_for(dir.path(), url, Duration::from_millis(25)).unwrap();
}

#[test]
fn managed_owner_environment_is_validated_before_writing() {
    let dir = tempfile::tempdir().unwrap();
    let relative = OsStr::new("relative");
    let absolute = dir.path().as_os_str();
    let address = "127.0.0.1:47632".parse().unwrap();

    let _scope = EnvScope::set(&[
        (BOOTSTRAP_STATE_DIR_ENV, Some(relative)),
        (
            "NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN",
            Some(OsStr::new("token")),
        ),
    ]);
    let error = publish_owner_from_env(address).unwrap_err();
    assert!(error.contains("absolute path"), "{error}");
    drop(_scope);

    let _scope = EnvScope::set(&[
        (BOOTSTRAP_STATE_DIR_ENV, Some(absolute)),
        ("NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN", None),
    ]);
    let error = publish_owner_from_env(address).unwrap_err();
    assert!(error.contains("SHUTDOWN_TOKEN"), "{error}");
    drop(_scope);

    let _scope = EnvScope::set(&[
        (BOOTSTRAP_STATE_DIR_ENV, Some(absolute)),
        (
            "NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN",
            Some(OsStr::new("token")),
        ),
    ]);
    let error = publish_owner_from_env("0.0.0.0:47632".parse().unwrap()).unwrap_err();
    assert!(error.contains("loopback"), "{error}");
}

#[test]
fn server_owner_guard_cleans_only_its_own_record() {
    let dir = tempfile::tempdir().unwrap();
    let address = "127.0.0.1:47632".parse().unwrap();
    let _scope = EnvScope::set(&[
        (BOOTSTRAP_STATE_DIR_ENV, Some(dir.path().as_os_str())),
        (
            "NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN",
            Some(OsStr::new("first-token")),
        ),
        (
            crate::configuration::BOOTSTRAP_FINGERPRINT_ENV,
            Some(OsStr::new("fingerprint")),
        ),
    ]);
    let guard = publish_owner_from_env(address).unwrap().unwrap();
    let path = owner_path(dir.path(), "http://127.0.0.1:47632");
    assert!(path.exists());

    let replacement = OwnerRecord::new(
        std::process::id(),
        "http://127.0.0.1:47632",
        "replacement-token",
        Some("fingerprint"),
    );
    write_owner_record(&path, &replacement).unwrap();
    drop(guard);

    assert_eq!(read_owner_record(&path).unwrap(), Some(replacement));
}

#[test]
fn stopping_an_absent_or_stale_owned_gateway_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config");
    let _scope = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(config.as_os_str())),
        ("HOME", Some(dir.path().as_os_str())),
        ("USERPROFILE", None),
    ]);
    let url = "http://127.0.0.1:9";

    stop_owned_and_reset(url).unwrap();
    let state = state_dir().unwrap();
    create_private_dir(&state).unwrap();
    let path = owner_path(&state, url);
    let owner = OwnerRecord::new(42, url, "shutdown", Some("fingerprint"));
    write_owner_record(&path, &owner).unwrap();

    stop_owned_and_reset(url).unwrap();
    assert!(!path.exists());
}

#[test]
fn authenticated_owned_gateway_is_shut_down_and_cleaned_up() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config");
    let _scope = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(config.as_os_str())),
        ("HOME", Some(dir.path().as_os_str())),
        ("USERPROFILE", None),
    ]);
    let key = crate::configuration::BootstrapChallengeKey::load().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let state = state_dir().unwrap();
    create_private_dir(&state).unwrap();
    let path = owner_path(&state, &url);
    let owner = OwnerRecord::new(42, &url, "shutdown-token", Some("fingerprint"));
    write_owner_record(&path, &owner).unwrap();

    let server = std::thread::spawn(move || {
        let (mut health, _) = listener.accept().unwrap();
        let request = read_headers(&mut health);
        let nonce = header(&request, "x-nemo-relay-bootstrap-nonce");
        let proof = key.proof("fingerprint", &nonce);
        let body = format!(
            "{{\"status\":\"ok\",\"service\":\"nemo-relay\",\"version\":\"{}\",\"bootstrap_protocol\":{},\"instance_id\":\"test-instance\"}}",
            env!("CARGO_PKG_VERSION"),
            BOOTSTRAP_PROTOCOL_VERSION
        );
        health
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nX-NeMo-Relay-Bootstrap-Proof: {proof}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .unwrap();

        let (mut shutdown, _) = listener.accept().unwrap();
        let request = read_headers(&mut shutdown);
        assert!(request.starts_with("POST /bootstrap/shutdown HTTP/1.1"));
        assert_eq!(
            header(&request, "x-nemo-relay-bootstrap-token"),
            "shutdown-token"
        );
        shutdown
            .write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .unwrap();
    });

    stop_owned_and_reset(&url).unwrap();
    server.join().unwrap();
    assert!(!path.exists());
}
