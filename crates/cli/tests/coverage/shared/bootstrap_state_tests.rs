// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::test_support::{EnvScope, accept_bounded, header, read_headers};
use std::ffi::OsStr;
use std::io::Write;
use std::net::TcpListener;

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

#[cfg(any(target_os = "linux", target_os = "macos", windows))]
#[test]
fn live_owner_record_uses_a_process_instance_identity() {
    let owner = OwnerRecord::new(
        std::process::id(),
        "http://127.0.0.1:47632",
        "shutdown",
        Some("fingerprint"),
    );

    assert!(owner.process_identity.is_some());
    assert!(owner_process_identity_matches(&owner));
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
    let error = publish_owner_from_env(address, Some("token")).unwrap_err();
    assert!(error.contains("absolute path"), "{error}");
    drop(_scope);

    let _scope = EnvScope::set(&[
        (BOOTSTRAP_STATE_DIR_ENV, Some(absolute)),
        ("NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN", None),
    ]);
    let error = publish_owner_from_env(address, None).unwrap_err();
    assert!(error.contains("SHUTDOWN_TOKEN"), "{error}");
    drop(_scope);

    let _scope = EnvScope::set(&[
        (BOOTSTRAP_STATE_DIR_ENV, Some(absolute)),
        (
            "NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN",
            Some(OsStr::new("token")),
        ),
    ]);
    let error =
        publish_owner_from_env("0.0.0.0:47632".parse().unwrap(), Some("token")).unwrap_err();
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
    let guard = publish_owner_from_env(address, Some("first-token"))
        .unwrap()
        .unwrap();
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
fn version_mismatched_owned_gateway_is_shut_down_and_cleaned_up() {
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
    let mut owner = OwnerRecord::new(42, &url, "shutdown-token", Some("fingerprint"));
    owner.version = "previous-version".into();
    write_owner_record(&path, &owner).unwrap();

    let server = std::thread::spawn(move || {
        let mut health = accept_bounded(&listener);
        let request = read_headers(&mut health);
        let nonce = header(&request, "x-nemo-relay-bootstrap-nonce");
        let proof = key.proof("fingerprint", &nonce);
        let body = format!(
            "{{\"status\":\"ok\",\"service\":\"nemo-relay\",\"version\":\"{}\",\"bootstrap_protocol\":{},\"instance_id\":\"test-instance\"}}",
            "previous-version", BOOTSTRAP_PROTOCOL_VERSION
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

        let mut shutdown = accept_bounded(&listener);
        let challenge = read_headers(&mut shutdown);
        let nonce = header(&challenge, "x-nemo-relay-bootstrap-nonce");
        let proof = key.proof("fingerprint", &nonce);
        shutdown
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nX-NeMo-Relay-Bootstrap-Proof: {proof}\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .unwrap();
        let request = read_headers(&mut shutdown);
        assert!(request.starts_with("POST /bootstrap/shutdown HTTP/1.1"));
        assert_eq!(
            header(&request, "x-nemo-relay-bootstrap-token"),
            "shutdown-token"
        );
        // Close the listener before acknowledging shutdown so the verifier's
        // immediate health probe cannot race this fixture's teardown.
        drop(listener);
        shutdown
            .write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .unwrap();
    });

    let _lock = lock_endpoint(&state, &url).unwrap();
    assert!(stop_version_mismatched_owned_gateway_locked(&state, &url).unwrap());
    server.join().unwrap();
    assert!(!path.exists());
}

#[test]
fn same_version_or_invalid_owned_gateway_is_not_stopped_for_replacement() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config");
    let _scope = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(config.as_os_str())),
        ("HOME", Some(dir.path().as_os_str())),
        ("USERPROFILE", None),
    ]);
    let url = "http://127.0.0.1:47632";
    let state = state_dir().unwrap();
    create_private_dir(&state).unwrap();
    let path = owner_path(&state, url);
    let owner = OwnerRecord::new(i32::MAX as u32, url, "shutdown-token", Some("fingerprint"));
    write_owner_record(&path, &owner).unwrap();

    let _lock = lock_endpoint(&state, url).unwrap();
    assert!(!stop_version_mismatched_owned_gateway_locked(&state, url).unwrap());
    assert!(path.exists());
    drop(_lock);

    let mut invalid = owner;
    invalid.bootstrap_protocol = BOOTSTRAP_PROTOCOL_VERSION.saturating_sub(1);
    write_owner_record(&path, &invalid).unwrap();
    let _lock = lock_endpoint(&state, url).unwrap();
    assert!(!stop_version_mismatched_owned_gateway_locked(&state, url).unwrap());
    assert!(path.exists());
}

#[test]
fn stale_unhealthy_gateway_owner_is_removed() {
    let dir = tempfile::tempdir().unwrap();
    let url = "http://127.0.0.1:9";
    let path = owner_path(dir.path(), url);
    let owner = OwnerRecord::new(u32::MAX, url, "shutdown-token", Some("fingerprint"));
    write_owner_record(&path, &owner).unwrap();

    assert!(!stop_unhealthy_owned_gateway_locked(dir.path(), url).unwrap());
    assert!(!path.exists());
}

#[cfg(unix)]
#[test]
fn unhealthy_owned_gateway_is_force_killed_after_the_grace_period() {
    use std::os::unix::process::CommandExt;

    let dir = tempfile::tempdir().unwrap();
    let url = "http://127.0.0.1:9";
    let path = owner_path(dir.path(), url);
    let child_pid_path = dir.path().join("child.pid");
    let mut command = std::process::Command::new("sh");
    command.args([
        "-c",
        "sh -c 'trap \"\" TERM; while :; do sleep 60; done' & echo $! > \"$1\"; wait",
        "sh",
        child_pid_path.to_str().unwrap(),
    ]);
    // SAFETY: The child calls only async-signal-safe `setsid` before exec.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
    let mut child = command.spawn().unwrap();
    let child_pid = loop {
        if let Ok(value) = std::fs::read_to_string(&child_pid_path) {
            break value.trim().parse::<i32>().unwrap();
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let owner = OwnerRecord::new(child.id(), url, "shutdown-token", Some("fingerprint"));
    write_owner_record(&path, &owner).unwrap();
    let waiter = std::thread::spawn(move || child.wait());

    assert!(stop_unhealthy_owned_gateway_locked(dir.path(), url).unwrap());
    assert!(!waiter.join().unwrap().unwrap().success());
    let deadline = Instant::now() + Duration::from_secs(1);
    while process_is_running(child_pid) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(!process_is_running(child_pid));
    assert!(!path.exists());
}
