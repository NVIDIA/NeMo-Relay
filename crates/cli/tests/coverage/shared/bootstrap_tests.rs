// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::test_support::{EnvScope, accept_bounded, header, read_headers};
use std::ffi::OsStr;
use std::io::Write;
use std::net::TcpListener;
use std::process::Command;

#[test]
fn failed_reaper_spawn_terminates_and_reaps_the_retained_child() {
    let child = Command::new(std::env::current_exe().unwrap())
        .arg("--list")
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let terminated = std::sync::atomic::AtomicBool::new(false);

    let error = hand_off_to_reaper_with(
        child,
        |_| Err(std::io::Error::other("thread limit")),
        |child| {
            terminated.store(true, std::sync::atomic::Ordering::SeqCst);
            let _ = child.kill();
            child.wait().unwrap();
        },
    )
    .unwrap_err();

    assert!(terminated.load(std::sync::atomic::Ordering::SeqCst));
    assert!(error.contains("failed to start gateway reaper thread"));
}

#[test]
fn persistent_gateway_requires_a_loopback_endpoint() {
    let non_loopback = GatewaySpec::new("0.0.0.0:47632".parse().unwrap())
        .acquire()
        .unwrap_err();
    assert!(non_loopback.contains("loopback"), "{non_loopback}");
}

#[test]
fn compatible_gateway_is_reused_without_starting_another_process() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config");
    let _environment = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(config.as_os_str())),
        ("HOME", Some(temp.path().as_os_str())),
        ("USERPROFILE", None),
    ]);
    let key = crate::configuration::BootstrapChallengeKey::load().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let mut stream = accept_bounded(&listener);
        let request = read_headers(&mut stream);
        let nonce = header(&request, "x-nemo-relay-bootstrap-nonce");
        let proof = key.proof("fingerprint", &nonce);
        let body = format!(
            "{{\"status\":\"ok\",\"service\":\"nemo-relay\",\"version\":\"{}\",\"bootstrap_protocol\":{},\"instance_id\":\"existing-instance\"}}",
            "compatible-other-version", BOOTSTRAP_PROTOCOL_VERSION
        );
        stream
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nX-NeMo-Relay-Bootstrap-Proof: {proof}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .unwrap();
    });

    let endpoint = GatewaySpec::new(address)
        .with_fingerprint("fingerprint")
        .acquire()
        .unwrap();

    server.join().unwrap();
    assert_eq!(endpoint.address, address);
    assert_eq!(endpoint.instance_id, "existing-instance");
}

#[test]
fn foreign_and_incompatible_listeners_are_never_adopted() {
    for (status, body, expected) in [
        ("200 OK", "{}", "not a compatible"),
        (
            "409 Conflict",
            "{\"status\":\"incompatible\",\"service\":\"nemo-relay\",\"version\":\"other\",\"bootstrap_protocol\":2,\"instance_id\":\"other\"}",
            "different version",
        ),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let body = body.to_string();
        listener.set_nonblocking(true).unwrap();
        let (stop_tx, stop_rx) = std::sync::mpsc::channel();
        let server = std::thread::spawn(move || {
            loop {
                if stop_rx.try_recv().is_ok() {
                    break;
                }
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).unwrap();
                        stream
                            .set_read_timeout(Some(Duration::from_secs(5)))
                            .unwrap();
                        let _ = read_headers(&mut stream);
                        stream
                            .write_all(
                                format!(
                                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                                    body.len()
                                )
                                .as_bytes(),
                            )
                            .unwrap();
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("failed to accept test connection: {error}"),
                }
            }
        });

        let error = GatewaySpec::new(address).acquire().unwrap_err();
        stop_tx.send(()).unwrap();
        server.join().unwrap();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn readiness_file_requires_the_existing_server_identity() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("ready.json");
    assert!(read_ready_file(&path).unwrap().is_none());

    std::fs::write(
        &path,
        format!(
            "{{\"service\":\"nemo-relay\",\"version\":\"{}\",\"bootstrap_protocol\":{},\"address\":\"127.0.0.1:47632\",\"instance_id\":\"ready\"}}",
            env!("CARGO_PKG_VERSION"),
            BOOTSTRAP_PROTOCOL_VERSION
        ),
    )
    .unwrap();
    let endpoint = read_ready_file(&path).unwrap().unwrap();
    assert_eq!(endpoint.url, DEFAULT_URL);
    assert_eq!(endpoint.instance_id, "ready");

    std::fs::write(&path, "{}").unwrap();
    let error = read_ready_file(&path).unwrap_err();
    assert!(error.contains("failed to parse"), "{error}");
}

#[test]
fn persistent_gateway_resolution_keeps_server_configuration_in_one_spec() {
    let temp = tempfile::tempdir().unwrap();
    let _environment = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(temp.path().as_os_str())),
        ("HOME", Some(temp.path().as_os_str())),
        ("USERPROFILE", None),
    ]);
    let bind = DEFAULT_BIND.parse().unwrap();
    let resolved = resolve_plugin_gateway(&GatewayOverrides::default(), bind).unwrap();

    assert_eq!(resolved.gateway.bind(), bind);
    assert_eq!(
        resolved.max_hook_payload_bytes,
        crate::configuration::DEFAULT_MAX_HOOK_PAYLOAD_BYTES
    );
    assert!(resolved.gateway.bootstrap_fingerprint.is_some());
    assert!(resolved.gateway.user_config_scope);
    assert!(
        resolved
            .gateway
            .launch_args
            .iter()
            .any(|arg| arg == "--max-hook-payload-bytes")
    );
}

#[test]
fn idle_timeout_drives_heartbeat_and_rejects_invalid_values() {
    let _environment = EnvScope::set(&[(
        crate::configuration::PLUGIN_IDLE_TIMEOUT_ENV,
        Some(OsStr::new("9")),
    )]);
    assert_eq!(plugin_idle_timeout().unwrap(), Duration::from_secs(9));
    assert_eq!(plugin_heartbeat_interval().unwrap(), Duration::from_secs(3));
    drop(_environment);

    let _environment = EnvScope::set(&[(
        crate::configuration::PLUGIN_IDLE_TIMEOUT_ENV,
        Some(OsStr::new("0")),
    )]);
    assert!(
        plugin_idle_timeout()
            .unwrap_err()
            .contains("greater than 0")
    );
}

#[test]
fn binary_override_is_explicit_and_validated() {
    let temp = tempfile::tempdir().unwrap();
    let binary = temp.path().join("nemo-relay");
    std::fs::write(&binary, "").unwrap();
    let _environment = EnvScope::set(&[("NEMO_RELAY_PLUGIN_BINARY", Some(binary.as_os_str()))]);
    assert_eq!(relay_binary().unwrap(), binary);
    drop(_environment);

    let missing = temp.path().join("missing");
    let _environment = EnvScope::set(&[("NEMO_RELAY_PLUGIN_BINARY", Some(missing.as_os_str()))]);
    assert!(relay_binary().unwrap_err().contains("does not exist"));
}

#[test]
fn windows_detachment_requests_only_supported_breakaway_flags() {
    let base = WINDOWS_CREATE_NEW_PROCESS_GROUP | WINDOWS_CREATE_NO_WINDOW;
    assert_eq!(windows_detached_creation_flags(false, None), (base, false));
    assert_eq!(
        windows_detached_creation_flags(true, Some(WINDOWS_JOB_OBJECT_LIMIT_BREAKAWAY_OK)),
        (base | WINDOWS_CREATE_BREAKAWAY_FROM_JOB, false)
    );
    assert_eq!(
        windows_detached_creation_flags(true, Some(WINDOWS_JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK)),
        (base, false)
    );
    assert_eq!(windows_detached_creation_flags(true, Some(0)), (base, true));
}
