// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsString;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;
use std::thread;

use super::*;
use serde_json::json;

struct EnvVarRestore {
    key: &'static str,
    previous: Option<OsString>,
}

impl EnvVarRestore {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: Callers hold the process-wide environment test lock.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        // SAFETY: Callers retain the process-wide environment test lock until after this guard.
        unsafe {
            match self.previous.take() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

#[test]
fn gateway_spec_is_the_complete_compatibility_contract() {
    let bind = "127.0.0.1:47632".parse().unwrap();
    let first = GatewaySpec::new(CodingAgent::Codex, bind)
        .with_launch_args(vec![
            OsString::from("--openai-base-url"),
            OsString::from("mock"),
        ])
        .with_fingerprint("fingerprint-a");
    let same = GatewaySpec::new(CodingAgent::Codex, bind)
        .with_launch_args(vec![
            OsString::from("--openai-base-url"),
            OsString::from("mock"),
        ])
        .with_fingerprint("fingerprint-a");
    let different = GatewaySpec::new(CodingAgent::Codex, bind)
        .with_launch_args(vec![
            OsString::from("--openai-base-url"),
            OsString::from("other"),
        ])
        .with_fingerprint("fingerprint-a");

    assert_eq!(first, same);
    assert_ne!(first, different);
    assert_eq!(first.bind(), bind);
}

#[test]
fn gateway_spec_rejects_non_loopback_bind_before_launch() {
    let error = GatewaySpec::new(CodingAgent::Codex, "0.0.0.0:47632".parse().unwrap())
        .ensure()
        .unwrap_err();

    assert!(error.contains("require a loopback bind address"), "{error}");
}

#[test]
fn typed_owner_record_round_trips_and_rejects_identity_drift() {
    let dir = tempfile::tempdir().unwrap();
    let owner = dir.path().join("owner.json");
    let pid = dir.path().join("owner.pid");
    std::fs::write(&pid, "42").unwrap();
    write_sidecar_owner(
        &owner,
        42,
        "http://127.0.0.1:47632",
        "shutdown-token",
        Some("fingerprint"),
    )
    .unwrap();

    validate_sidecar_owner(
        &owner,
        &pid,
        42,
        "http://127.0.0.1:47632",
        "shutdown-token",
        Some("fingerprint"),
    )
    .unwrap();
    let error = validate_sidecar_owner(
        &owner,
        &pid,
        42,
        "http://127.0.0.1:47632",
        "different-token",
        Some("fingerprint"),
    )
    .unwrap_err();
    assert!(error.contains("does not match the ready process"));
}

#[test]
fn typed_owner_record_rejects_missing_required_fields() {
    let dir = tempfile::tempdir().unwrap();
    let owner = dir.path().join("owner.json");
    let pid = dir.path().join("owner.pid");
    std::fs::write(&owner, serde_json::to_vec(&json!({"pid": 42})).unwrap()).unwrap();
    std::fs::write(&pid, "42").unwrap();

    let error = validate_sidecar_owner(
        &owner,
        &pid,
        42,
        "http://127.0.0.1:47632",
        "shutdown-token",
        Some("fingerprint"),
    )
    .unwrap_err();

    assert!(error.contains("invalid sidecar ownership file"), "{error}");
}

#[test]
fn readiness_file_requires_exact_protocol_identity() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ready.json");
    std::fs::write(
        &path,
        serde_json::to_vec(&json!({
            "service": "nemo-relay",
            "version": env!("CARGO_PKG_VERSION"),
            "bootstrap_protocol": BOOTSTRAP_PROTOCOL_VERSION,
            "address": "127.0.0.1:47777",
            "instance_id": "test-instance"
        }))
        .unwrap(),
    )
    .unwrap();

    let endpoint = read_sidecar_ready_file(&path).unwrap().unwrap();
    assert_eq!(endpoint.address, "127.0.0.1:47777".parse().unwrap());

    std::fs::write(
        &path,
        serde_json::to_vec(&json!({
            "service": "nemo-relay",
            "version": env!("CARGO_PKG_VERSION"),
            "bootstrap_protocol": BOOTSTRAP_PROTOCOL_VERSION + 1,
            "address": "127.0.0.1:47777",
            "instance_id": "test-instance"
        }))
        .unwrap(),
    )
    .unwrap();
    let error = read_sidecar_ready_file(&path).unwrap_err();
    assert!(error.contains("incompatible sidecar readiness file"));
}

fn one_health_response(body: String) -> (SocketAddr, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0_u8; 2048];
        let _ = stream.read(&mut request).unwrap();
        stream
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .unwrap();
    });
    (address, server)
}

fn unused_loopback_address() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    address
}

#[test]
fn direct_sidecar_start_classifies_existing_listeners_before_spawning() {
    let dir = tempfile::tempdir().unwrap();
    let compatible_body = format!(
        r#"{{"status":"ok","service":"nemo-relay","version":"{}","bootstrap_protocol":{},"instance_id":"test-instance"}}"#,
        env!("CARGO_PKG_VERSION"),
        BOOTSTRAP_PROTOCOL_VERSION
    );
    let (address, server) = one_health_response(compatible_body);
    let endpoint = start_sidecar_bind(
        &GatewaySpec::new(CodingAgent::Codex, address),
        dir.path(),
        dir.path(),
        None,
    )
    .unwrap();
    assert_eq!(endpoint.address, address);
    server.join().unwrap();

    let incompatible_body = format!(
        r#"{{"status":"ok","service":"nemo-relay","version":"other","bootstrap_protocol":{}}}"#,
        BOOTSTRAP_PROTOCOL_VERSION
    );
    let (address, server) = one_health_response(incompatible_body);
    let error = start_sidecar_bind(
        &GatewaySpec::new(CodingAgent::Codex, address),
        dir.path(),
        dir.path(),
        None,
    )
    .unwrap_err();
    assert!(error.contains("different version"), "{error}");
    server.join().unwrap();

    let (address, server) = one_health_response("{}".into());
    let error = start_sidecar_bind(
        &GatewaySpec::new(CodingAgent::Codex, address),
        dir.path(),
        dir.path(),
        None,
    )
    .unwrap_err();
    assert!(error.contains("not a compatible NeMo Relay"), "{error}");
    server.join().unwrap();
}

#[test]
fn sidecar_record_cleanup_is_scoped_to_the_exited_pid() {
    let dir = tempfile::tempdir().unwrap();
    let invalid = sidecar_owner_path(dir.path(), "http://127.0.0.1:47630");
    std::fs::write(&invalid, "not-json").unwrap();
    let matching = sidecar_owner_path(dir.path(), "http://127.0.0.1:47631");
    write_sidecar_owner(
        &matching,
        42,
        "http://127.0.0.1:47631",
        "token",
        Some("fingerprint"),
    )
    .unwrap();
    let matching_pid = sidecar_pid_path(dir.path(), "http://127.0.0.1:47631");
    std::fs::write(&matching_pid, "42").unwrap();
    let other = sidecar_owner_path(dir.path(), "http://127.0.0.1:47632");
    write_sidecar_owner(
        &other,
        43,
        "http://127.0.0.1:47632",
        "token",
        Some("fingerprint"),
    )
    .unwrap();

    cleanup_sidecar_records_for_pid(dir.path(), 42);

    assert!(!matching.exists());
    assert!(!matching_pid.exists());
    assert!(invalid.exists());
    assert!(other.exists());

    let not_a_directory = dir.path().join("runtime-file");
    std::fs::write(&not_a_directory, "file").unwrap();
    cleanup_sidecar_records_for_pid(&not_a_directory, 42);
}

fn exited_command() -> Command {
    #[cfg(windows)]
    {
        let mut command = Command::new("cmd");
        command.args(["/C", "exit 7"]);
        command
    }
    #[cfg(not(windows))]
    {
        let mut command = Command::new("sh");
        command.args(["-c", "exit 7"]);
        command
    }
}

#[test]
fn an_already_exited_unready_sidecar_reports_its_status() {
    let dir = tempfile::tempdir().unwrap();
    let pid_path = dir.path().join("startup.pid");
    let mut child = exited_command().spawn().unwrap();
    let pid = child.id();
    assert!(!child.wait().unwrap().success());
    std::fs::write(&pid_path, pid.to_string()).unwrap();

    let error = terminate_unready_sidecar(child, &pid_path, DEFAULT_URL).unwrap_err();

    assert!(error.contains("exited before becoming ready"), "{error}");
    assert!(!pid_path.exists());
}

#[test]
fn reaper_cleanup_treats_an_unopenable_lock_as_terminal() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = exited_command().spawn().unwrap();
    assert!(!child.wait().unwrap().success());
    let request = SidecarReapRequest {
        process: DetachedSidecarProcess::new(
            child,
            #[cfg(windows)]
            None,
        ),
        exited: true,
        owner_path: dir.path().join("owner.json"),
        pid_path: dir.path().join("owner.pid"),
        lock_path: dir.path().join("missing-parent").join("owner.lock"),
    };

    assert!(cleanup_reaped_sidecar(&request));
}

#[test]
fn zero_idle_timeout_is_rejected_by_timeout_and_heartbeat_resolution() {
    let _lock = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let key = crate::config::PLUGIN_IDLE_TIMEOUT_ENV;
    let _environment = EnvVarRestore::set(key, "0");

    let timeout_error = plugin_idle_timeout().unwrap_err();
    let heartbeat_error = plugin_heartbeat_interval().unwrap_err();

    assert!(timeout_error.contains("must be greater than 0"));
    assert!(heartbeat_error.contains("must be greater than 0"));
}

#[test]
fn gateway_start_reports_an_uncreatable_runtime_directory() {
    let dir = tempfile::tempdir().unwrap();
    let runtime_base = dir.path().join("runtime");
    let runtime = runtime_base.join("nemo-relay-plugin");
    std::fs::create_dir_all(&runtime_base).unwrap();
    std::fs::write(&runtime, "file").unwrap();
    let _lock = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _runtime = EnvVarRestore::set("XDG_RUNTIME_DIR", runtime_base.to_str().unwrap());
    let _config = EnvVarRestore::set("XDG_CONFIG_HOME", dir.path().to_str().unwrap());
    let bind = unused_loopback_address();

    let error = GatewaySpec::new(CodingAgent::Codex, bind)
        .ensure()
        .unwrap_err();

    assert!(error.contains("failed to create"), "{error}");
    assert!(error.contains(&runtime.display().to_string()), "{error}");
    assert!(error.contains("inspect"), "{error}");
}

#[test]
fn gateway_start_reports_an_uncreatable_state_directory() {
    let dir = tempfile::tempdir().unwrap();
    let runtime_base = dir.path().join("runtime");
    let config_base = dir.path().join("config");
    let _lock = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _runtime = EnvVarRestore::set("XDG_RUNTIME_DIR", runtime_base.to_str().unwrap());
    let _config = EnvVarRestore::set("XDG_CONFIG_HOME", config_base.to_str().unwrap());
    let state = sidecar_state_dir().unwrap();
    std::fs::create_dir_all(state.parent().unwrap()).unwrap();
    std::fs::write(&state, "file").unwrap();
    let bind = unused_loopback_address();

    let error = GatewaySpec::new(CodingAgent::Codex, bind)
        .ensure()
        .unwrap_err();

    assert!(error.contains("failed to create"), "{error}");
    assert!(error.contains(&state.display().to_string()), "{error}");
}

#[test]
fn gateway_start_reports_an_unopenable_endpoint_lock() {
    let dir = tempfile::tempdir().unwrap();
    let runtime_base = dir.path().join("runtime");
    let config_base = dir.path().join("config");
    let _lock = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _runtime = EnvVarRestore::set("XDG_RUNTIME_DIR", runtime_base.to_str().unwrap());
    let _config = EnvVarRestore::set("XDG_CONFIG_HOME", config_base.to_str().unwrap());
    let state = sidecar_state_dir().unwrap();
    let bind = unused_loopback_address();
    let url = format!("http://{bind}");
    let endpoint_lock = sidecar_lock_path(&state, &url);
    std::fs::create_dir_all(&endpoint_lock).unwrap();

    let error = GatewaySpec::new(CodingAgent::Codex, bind)
        .ensure()
        .unwrap_err();

    assert!(error.contains("failed to open sidecar lock"), "{error}");
    assert!(
        error.contains(&endpoint_lock.display().to_string()),
        "{error}"
    );
}
