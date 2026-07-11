// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{OsStr, OsString};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::thread;
use std::time::Duration;

use tempfile::tempdir;

use super::*;

#[test]
fn endpoint_leases_reset_recovery_only_after_the_last_client_closes() {
    let runtime = tempfile::tempdir().unwrap();
    let url = "http://127.0.0.1:47632";
    let first = EndpointLease::acquire(runtime.path(), url).unwrap();
    assert!(first.fresh_epoch());
    let mut epoch = RecoveryEpoch::new(url, "gateway-1");
    epoch.restarts = 1;
    write_recovery_epoch(runtime.path(), &epoch).unwrap();

    let second = EndpointLease::acquire(runtime.path(), url).unwrap();
    assert!(!second.fresh_epoch());
    assert!(read_recovery_epoch(runtime.path(), url).unwrap().is_some());
    drop(first);

    let third = EndpointLease::acquire(runtime.path(), url).unwrap();
    assert!(!third.fresh_epoch());
    drop(second);
    drop(third);

    let next_epoch = EndpointLease::acquire(runtime.path(), url).unwrap();
    assert!(next_epoch.fresh_epoch());
    assert!(read_recovery_epoch(runtime.path(), url).unwrap().is_none());
}

#[test]
fn recovery_epoch_rejects_identity_and_restart_count_drift() {
    let runtime = tempfile::tempdir().unwrap();
    let url = "http://127.0.0.1:47632";
    let path = recovery_path(runtime.path(), url);
    std::fs::write(
        &path,
        serde_json::to_vec(&serde_json::json!({
            "service": "foreign",
            "bootstrap_protocol": BOOTSTRAP_PROTOCOL_VERSION,
            "url": url,
            "instance_id": "gateway-1",
            "restarts": 0,
            "pending": false,
        }))
        .unwrap(),
    )
    .unwrap();
    assert!(
        read_recovery_epoch(runtime.path(), url)
            .unwrap_err()
            .contains("incompatible")
    );

    let mut epoch = RecoveryEpoch::new(url, "gateway-1");
    epoch.restarts = 2;
    write_recovery_epoch(runtime.path(), &epoch).unwrap();
    assert!(
        read_recovery_epoch(runtime.path(), url)
            .unwrap_err()
            .contains("incompatible")
    );
}

const SHUTDOWN_TOKEN_ENV: &str = "NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN";

struct Environment {
    _lock: std::sync::MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl Environment {
    fn isolated() -> Self {
        let lock = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let keys = [
            "HOME",
            "USERPROFILE",
            "XDG_CONFIG_HOME",
            BOOTSTRAP_STATE_DIR_ENV,
            SHUTDOWN_TOKEN_ENV,
            crate::config::BOOTSTRAP_FINGERPRINT_ENV,
        ];
        let saved = keys
            .into_iter()
            .map(|key| (key, std::env::var_os(key)))
            .collect();
        Self { _lock: lock, saved }
    }

    fn set(&self, key: &'static str, value: impl AsRef<OsStr>) {
        // SAFETY: This scope holds the process-wide environment test lock.
        unsafe { std::env::set_var(key, value) };
    }

    fn remove(&self, key: &'static str) {
        // SAFETY: This scope holds the process-wide environment test lock.
        unsafe { std::env::remove_var(key) };
    }

    fn clear_managed_bootstrap(&self) {
        self.remove(BOOTSTRAP_STATE_DIR_ENV);
        self.remove(SHUTDOWN_TOKEN_ENV);
        self.remove(crate::config::BOOTSTRAP_FINGERPRINT_ENV);
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        // SAFETY: Restoration happens while the process-wide environment test lock is held.
        unsafe {
            for (key, value) in self.saved.drain(..).rev() {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

fn configure_managed_bootstrap(environment: &Environment, state: &Path) {
    environment.set(BOOTSTRAP_STATE_DIR_ENV, state);
    environment.set(SHUTDOWN_TOKEN_ENV, "shutdown-token");
    environment.set(crate::config::BOOTSTRAP_FINGERPRINT_ENV, "fingerprint");
}

fn read_http_headers(stream: &mut TcpStream) -> String {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut request = Vec::new();
    let mut buffer = [0_u8; 1024];
    while !request.windows(4).any(|window| window == b"\r\n\r\n") {
        let read = stream.read(&mut buffer).unwrap();
        if read == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..read]);
    }
    String::from_utf8(request).unwrap()
}

fn request_header(request: &str, name: &str) -> String {
    request
        .lines()
        .find_map(|line| {
            let (candidate, value) = line.split_once(':')?;
            candidate
                .eq_ignore_ascii_case(name)
                .then(|| value.trim().to_string())
        })
        .unwrap_or_else(|| panic!("missing {name} in request: {request}"))
}

#[test]
fn state_files_distinguish_absence_from_read_failure() {
    let dir = tempdir().unwrap();
    let missing = dir.path().join("missing.json");

    assert!(read_owner_record(&missing).unwrap().is_none());
    assert!(read_ready_file(&missing).unwrap().is_none());

    let owner_error = read_owner_record(dir.path()).unwrap_err();
    assert!(owner_error.contains("failed to read sidecar ownership"));
    let ready_error = read_ready_file(dir.path()).unwrap_err();
    assert!(ready_error.contains("failed to read sidecar readiness file"));
}

#[test]
fn state_directory_requires_a_user_configuration_home() {
    let environment = Environment::isolated();
    environment.remove("HOME");
    environment.remove("USERPROFILE");
    environment.remove("XDG_CONFIG_HOME");

    let error = state_dir().unwrap_err();

    assert!(error.contains("cannot determine"), "{error}");
}

#[test]
fn managed_owner_environment_is_validated_before_writing_state() {
    let dir = tempdir().unwrap();
    let environment = Environment::isolated();
    environment.clear_managed_bootstrap();

    environment.set(BOOTSTRAP_STATE_DIR_ENV, "relative-state");
    environment.set(SHUTDOWN_TOKEN_ENV, "shutdown-token");
    let error = publish_owner_from_env("127.0.0.1:47632".parse().unwrap()).unwrap_err();
    assert!(error.contains("must be an absolute path"), "{error}");

    environment.set(BOOTSTRAP_STATE_DIR_ENV, dir.path());
    environment.set(SHUTDOWN_TOKEN_ENV, "");
    let error = publish_owner_from_env("127.0.0.1:47632".parse().unwrap()).unwrap_err();
    assert!(error.contains(SHUTDOWN_TOKEN_ENV), "{error}");

    environment.set(SHUTDOWN_TOKEN_ENV, "shutdown-token");
    let error = publish_owner_from_env("0.0.0.0:47632".parse().unwrap()).unwrap_err();
    assert!(error.contains("requires a loopback address"), "{error}");

    let state_file = dir.path().join("state-file");
    std::fs::write(&state_file, "not a directory").unwrap();
    environment.set(BOOTSTRAP_STATE_DIR_ENV, &state_file);
    let error = publish_owner_from_env("127.0.0.1:47632".parse().unwrap()).unwrap_err();
    assert!(
        error.contains("failed to create bootstrap state directory"),
        "{error}"
    );
}

#[test]
fn managed_owner_is_endpoint_scoped_without_a_host_identity() {
    let dir = tempdir().unwrap();
    let state = dir.path().join("state");
    let environment = Environment::isolated();
    configure_managed_bootstrap(&environment, &state);
    let address = "127.0.0.1:47633".parse().unwrap();

    publish_owner_from_env(address).unwrap();

    let url = format!("http://{address}");
    validate_owner(
        &owner_path(&state, &url),
        &pid_path(&state, &url),
        std::process::id(),
        &url,
        "shutdown-token",
        Some("fingerprint"),
    )
    .unwrap();
}

#[test]
fn publishing_endpoint_owner_migrates_legacy_agent_scoped_records() {
    let dir = tempdir().unwrap();
    let state = dir.path().join("state");
    std::fs::create_dir_all(&state).unwrap();
    let environment = Environment::isolated();
    configure_managed_bootstrap(&environment, &state);
    let address = "127.0.0.1:47636".parse().unwrap();
    let url = format!("http://{address}");
    let legacy_owner = state.join(format!("codex-sidecar-{}.owner.json", lock_name(&url)));
    let legacy_pid = state.join(format!("codex-sidecar-{}.pid", lock_name(&url)));
    write_owner(
        &legacy_owner,
        41,
        &url,
        "old-token",
        Some("old-fingerprint"),
    )
    .unwrap();
    std::fs::write(&legacy_pid, "41").unwrap();

    publish_owner_from_env(address).unwrap();

    assert!(!legacy_owner.exists());
    assert!(!legacy_pid.exists());
    assert!(owner_path(&state, &url).exists());
    assert!(pid_path(&state, &url).exists());
}

#[test]
fn failed_owner_publish_removes_the_partial_pid_record() {
    let dir = tempdir().unwrap();
    let state = dir.path().join("state");
    std::fs::create_dir_all(&state).unwrap();
    let environment = Environment::isolated();
    configure_managed_bootstrap(&environment, &state);
    let address = "127.0.0.1:47632".parse().unwrap();
    let url = format!("http://{address}");
    let owner = owner_path(&state, &url);
    let pid = pid_path(&state, &url);
    std::fs::create_dir_all(&owner).unwrap();
    #[cfg(windows)]
    {
        let file_name = owner.file_name().unwrap().to_string_lossy();
        let stale_backup = owner.with_file_name(format!(".{file_name}.nemo-relay-replace.tmp"));
        std::fs::create_dir_all(stale_backup).unwrap();
    }

    let error = publish_owner_from_env(address).unwrap_err();

    assert!(error.contains("failed to"), "{error}");
    assert!(!pid.exists());
}

#[test]
fn owner_validation_reports_a_missing_record() {
    let dir = tempdir().unwrap();
    let owner = dir.path().join("missing-owner.json");
    let pid = dir.path().join("missing-owner.pid");

    let error = validate_owner(
        &owner,
        &pid,
        42,
        "http://127.0.0.1:47632",
        "shutdown-token",
        Some("fingerprint"),
    )
    .unwrap_err();

    assert!(error.contains("file does not exist"), "{error}");
}

#[test]
fn owner_enumeration_distinguishes_missing_and_unreadable_directories() {
    let dir = tempdir().unwrap();
    assert!(owner_paths(&dir.path().join("missing")).unwrap().is_empty());

    let file = dir.path().join("not-a-directory");
    std::fs::write(&file, "file").unwrap();
    let error = owner_paths(&file).unwrap_err();
    assert!(error.contains("failed to enumerate"), "{error}");
}

#[test]
fn stale_owner_cleanup_requires_shutdown_credentials() {
    let dir = tempdir().unwrap();
    let url = "http://127.0.0.1:9";
    let owner = owner_path(dir.path(), url);

    assert!(
        stop_owned_record(dir.path(), &owner).is_ok(),
        "an already absent owner is clean"
    );

    write_owner(&owner, 42, url, "", Some("fingerprint")).unwrap();
    let error = stop_owned_record(dir.path(), &owner).unwrap_err();
    assert!(error.contains("has no shutdown token"), "{error}");

    write_owner(&owner, 42, url, "shutdown-token", None).unwrap();
    let error = stop_owned_record(dir.path(), &owner).unwrap_err();
    assert!(
        error.contains("no authenticated bootstrap fingerprint"),
        "{error}"
    );
}

#[test]
fn unavailable_owned_sidecar_is_removed_without_sending_shutdown() {
    let dir = tempdir().unwrap();
    let url = "http://127.0.0.1:9";
    let owner = owner_path(dir.path(), url);
    let pid = pid_path(dir.path(), url);
    write_owner(&owner, 42, url, "shutdown-token", Some("fingerprint")).unwrap();
    std::fs::write(&pid, "42").unwrap();

    stop_owned_record(dir.path(), &owner).unwrap();

    assert!(!owner.exists());
    assert!(!pid.exists());
}

#[test]
fn unavailable_legacy_owner_removes_the_legacy_pid_record() {
    let dir = tempdir().unwrap();
    let url = "http://127.0.0.1:9";
    let owner = dir.path().join("codex-sidecar.owner.json");
    let pid = dir.path().join("codex-sidecar.pid");
    write_owner(&owner, 42, url, "shutdown-token", Some("fingerprint")).unwrap();
    std::fs::write(&pid, "42").unwrap();

    stop_owned_record(dir.path(), &owner).unwrap();

    assert!(!owner.exists());
    assert!(!pid.exists());
}

#[test]
fn authenticated_owned_sidecar_is_shut_down_and_cleaned_up() {
    let dir = tempdir().unwrap();
    let environment = Environment::isolated();
    environment.set("XDG_CONFIG_HOME", dir.path().join("config"));
    environment.set("HOME", dir.path());
    environment.remove("USERPROFILE");
    let key = crate::config::BootstrapChallengeKey::load().unwrap();
    let reloaded_key = crate::config::BootstrapChallengeKey::load().unwrap();
    assert!(reloaded_key.verify(
        "fingerprint",
        "test-nonce",
        &key.proof("fingerprint", "test-nonce")
    ));
    let client_token = key.client_token();
    assert!(reloaded_key.verify_client_token(&client_token));
    assert!(!reloaded_key.verify_client_token("hmac-sha256:00"));
    assert_ne!(
        client_token,
        key.proof("fingerprint", "test-nonce"),
        "health and client proofs must remain domain-separated"
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let owner = owner_path(dir.path(), &url);
    let pid = pid_path(dir.path(), &url);
    write_owner(&owner, 42, &url, "shutdown-token", Some("fingerprint")).unwrap();
    std::fs::write(&pid, "42").unwrap();
    let server = thread::spawn(move || {
        for _ in 0..2 {
            let (mut health, _) = listener.accept().unwrap();
            let request = read_http_headers(&mut health);
            let fingerprint = request_header(&request, "x-nemo-relay-bootstrap-fingerprint");
            let nonce = request_header(&request, "x-nemo-relay-bootstrap-nonce");
            let proof = key.proof(&fingerprint, &nonce);
            let body = format!(
                r#"{{"status":"ok","service":"nemo-relay","version":"{}","bootstrap_protocol":{},"instance_id":"test-instance"}}"#,
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
        }

        let (mut shutdown, _) = listener.accept().unwrap();
        let request = read_http_headers(&mut shutdown);
        assert!(request.starts_with("POST /bootstrap/shutdown HTTP/1.1"));
        assert_eq!(
            request_header(&request, "x-nemo-relay-bootstrap-token"),
            "shutdown-token"
        );
        shutdown
            .write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .unwrap();
    });

    assert_eq!(probe(&url, Some("fingerprint")), RelayHealth::Compatible);
    stop_owned_record(dir.path(), &owner).unwrap();

    server.join().unwrap();
    assert!(!owner.exists());
    assert!(!pid.exists());
}

#[test]
fn stop_owned_aggregates_record_errors() {
    let dir = tempdir().unwrap();
    let environment = Environment::isolated();
    environment.set("XDG_CONFIG_HOME", dir.path());
    environment.set("HOME", dir.path());
    environment.remove("USERPROFILE");
    let runtime = state_dir().unwrap();
    std::fs::create_dir_all(&runtime).unwrap();
    let url = "http://127.0.0.1:9";
    let owner = owner_path(&runtime, url);
    write_owner(&owner, 42, url, "", Some("fingerprint")).unwrap();

    let error = stop_owned(url).unwrap_err();

    assert!(error.contains("has no shutdown token"), "{error}");
}

#[test]
fn stop_owned_succeeds_when_no_records_exist() {
    let dir = tempdir().unwrap();
    let environment = Environment::isolated();
    environment.set("XDG_CONFIG_HOME", dir.path());
    environment.set("HOME", dir.path());
    environment.remove("USERPROFILE");

    stop_owned(crate::sidecar::DEFAULT_URL).unwrap();
}

#[test]
fn stop_owned_leaves_other_managed_endpoints_untouched() {
    let dir = tempdir().unwrap();
    let environment = Environment::isolated();
    environment.set("XDG_CONFIG_HOME", dir.path());
    environment.set("HOME", dir.path());
    environment.remove("USERPROFILE");
    let runtime = state_dir().unwrap();
    std::fs::create_dir_all(&runtime).unwrap();
    let other_url = "http://127.0.0.1:47633";
    let other_owner = owner_path(&runtime, other_url);
    write_owner(&other_owner, 42, other_url, "", Some("fingerprint")).unwrap();

    stop_owned(crate::sidecar::DEFAULT_URL).unwrap();

    assert!(other_owner.exists());
}
