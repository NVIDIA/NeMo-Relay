// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::test_support::{EnvScope, accept_bounded, header, read_headers};
use std::ffi::OsStr;
use std::io::Write;
use std::net::TcpListener;

#[test]
fn owner_record_read_errors_and_invalid_ownership_preserve_state() {
    let directory = tempfile::tempdir().unwrap();
    let url = "http://127.0.0.1:9";
    let path = owner_path(directory.path(), url);
    assert_eq!(read_owner_record(&path).unwrap(), None);
    assert!(!stop_owned_and_reset_locked(directory.path(), url).unwrap());
    assert!(!stop_version_mismatched_owned_gateway_locked(directory.path(), url).unwrap());
    assert!(!stop_unhealthy_owned_gateway_locked(directory.path(), url).unwrap());

    std::fs::write(&path, b"not-json").unwrap();
    assert!(
        read_owner_record(&path)
            .unwrap_err()
            .contains("failed to parse gateway ownership")
    );
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();
    assert!(
        read_owner_record(&path)
            .unwrap_err()
            .contains("failed to read gateway ownership")
    );
    std::fs::remove_dir(&path).unwrap();

    let valid = OwnerRecord::new(u32::MAX, url, "token", Some("fingerprint"));
    for invalid in [
        OwnerRecord {
            service: "foreign".into(),
            ..valid.clone()
        },
        OwnerRecord {
            bootstrap_protocol: 0,
            ..valid.clone()
        },
        OwnerRecord {
            shutdown_token: String::new(),
            ..valid.clone()
        },
        OwnerRecord {
            bootstrap_fingerprint: None,
            ..valid.clone()
        },
        OwnerRecord {
            bootstrap_fingerprint: Some(String::new()),
            ..valid.clone()
        },
    ] {
        write_owner_record(&path, &invalid).unwrap();
        assert!(
            stop_owned_and_reset_locked(directory.path(), url)
                .unwrap_err()
                .contains("invalid ownership record")
        );
        assert!(!stop_unhealthy_owned_gateway_locked(directory.path(), url).unwrap());
        assert_eq!(read_owner_record(&path).unwrap(), Some(invalid));
    }
}

#[test]
fn managed_owner_requires_state_and_owner_guard_removes_its_unchanged_record() {
    let directory = tempfile::tempdir().unwrap();
    let address = "127.0.0.1:47632".parse().unwrap();
    let _environment = EnvScope::set(&[
        (BOOTSTRAP_STATE_DIR_ENV, None),
        (
            crate::configuration::BOOTSTRAP_FINGERPRINT_ENV,
            Some(OsStr::new("fingerprint")),
        ),
    ]);
    assert!(publish_owner_from_env(address, None).unwrap().is_none());
    assert!(
        publish_owner_from_env(address, Some("token"))
            .unwrap_err()
            .contains(BOOTSTRAP_STATE_DIR_ENV)
    );
    unsafe {
        std::env::set_var(BOOTSTRAP_STATE_DIR_ENV, directory.path());
    }
    let guard = publish_owner_from_env(address, Some("token"))
        .unwrap()
        .unwrap();
    let path = owner_path(directory.path(), "http://127.0.0.1:47632");
    let record = read_owner_record(&path).unwrap().unwrap();
    assert!(record.valid_for("http://127.0.0.1:47632"));
    drop(guard);
    assert!(!path.exists());
    remove_if_matches(&path, &record).unwrap();
}

#[test]
fn startup_lock_reports_an_unopenable_lock_file() {
    let directory = tempfile::tempdir().unwrap();
    let url = "http://127.0.0.1:9";
    std::fs::create_dir(lock_path(directory.path(), url)).unwrap();
    assert!(
        lock_endpoint_for(directory.path(), url, Duration::ZERO)
            .unwrap_err()
            .contains("failed to open gateway lock")
    );
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
fn bootstrap_state_reports_malformed_recovery_and_non_directory_state_paths() {
    let dir = tempfile::tempdir().unwrap();
    let url = "http://127.0.0.1:47632";
    std::fs::write(recovery_path(dir.path(), url), b"not-json").unwrap();
    let error = read_recovery(dir.path(), url).unwrap_err();
    assert!(
        error.contains("failed to parse gateway recovery"),
        "{error}"
    );
    std::fs::remove_file(recovery_path(dir.path(), url)).unwrap();
    std::fs::create_dir(recovery_path(dir.path(), url)).unwrap();
    let error = read_recovery(dir.path(), url).unwrap_err();
    assert!(error.contains("failed to read gateway recovery"), "{error}");

    let file = dir.path().join("not-a-directory");
    std::fs::write(&file, b"occupied").unwrap();
    let error = create_private_dir(&file).unwrap_err();
    assert!(error.contains("failed to create"), "{error}");
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
    use std::io::Read;
    use std::os::unix::process::{CommandExt, ExitStatusExt};

    let spawn_group_member = |group| {
        let mut child = std::process::Command::new("sh")
            .args(["-c", "trap \"\" TERM; printf ready; exec sleep 60"])
            .process_group(group)
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        // The readiness message follows the trap, which exec preserves for sleep.
        let mut ready = [0; 5];
        child.stdout.take().unwrap().read_exact(&mut ready).unwrap();
        assert_eq!(&ready, b"ready");
        child
    };

    let dir = tempfile::tempdir().unwrap();
    let url = "http://127.0.0.1:9";
    let path = owner_path(dir.path(), url);
    // Own and reap both group members so shutdown does not depend on the OS
    // reaping orphaned shell descendants before the termination deadline.
    let mut gateway = spawn_group_member(0);
    let process_group = i32::try_from(gateway.id()).unwrap();
    let mut group_member = spawn_group_member(process_group);
    let owner = OwnerRecord::new(gateway.id(), url, "shutdown-token", Some("fingerprint"));
    write_owner_record(&path, &owner).unwrap();
    let waiter = std::thread::spawn(move || (gateway.wait(), group_member.wait()));

    let started = Instant::now();
    let stopped = stop_unhealthy_owned_gateway_locked(dir.path(), url);
    if stopped.is_err() {
        // SAFETY: This is the private process group created by this test.
        unsafe { libc::kill(-process_group, libc::SIGKILL) };
    }
    let (gateway_status, member_status) = waiter.join().unwrap();
    assert!(stopped.unwrap());
    assert!(started.elapsed() >= UNHEALTHY_GATEWAY_TERMINATION_TIMEOUT);
    assert_eq!(gateway_status.unwrap().signal(), Some(libc::SIGKILL));
    assert_eq!(member_status.unwrap().signal(), Some(libc::SIGKILL));
    assert!(!process_is_running(-process_group));
    assert!(!path.exists());
}
