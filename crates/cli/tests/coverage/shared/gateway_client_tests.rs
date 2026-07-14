// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use super::*;
use crate::test_support::{EnvScope, header, read_headers};

fn serve_once(response: &[u8]) -> (String, mpsc::Receiver<Vec<u8>>, thread::JoinHandle<()>) {
    let response = response.to_vec();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (sender, receiver) = mpsc::channel();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            match stream.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => request.extend_from_slice(&buffer[..read]),
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    break;
                }
                Err(error) => panic!("failed to read request: {error}"),
            }
        }
        let _ = sender.send(request);
        stream.write_all(&response).unwrap();
    });
    (url, receiver, server)
}

fn serve_verified_shutdown(
    key: crate::configuration::BootstrapChallengeKey,
    response: &[u8],
) -> (String, mpsc::Receiver<String>, thread::JoinHandle<()>) {
    let response = response.to_vec();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (sender, receiver) = mpsc::channel();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let challenge = read_headers(&mut stream);
        let nonce = header(&challenge, "x-nemo-relay-bootstrap-nonce");
        let proof = key.proof("fingerprint", &nonce);
        let body = format!(
            "{{\"status\":\"ok\",\"service\":\"nemo-relay\",\"version\":\"{}\",\"bootstrap_protocol\":{},\"instance_id\":\"test-instance\"}}",
            env!("CARGO_PKG_VERSION"),
            BOOTSTRAP_PROTOCOL_VERSION
        );
        stream
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nX-NeMo-Relay-Bootstrap-Proof: {proof}\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .unwrap();
        let _ = sender.send(read_headers(&mut stream));
        stream.write_all(&response).unwrap();
    });
    (url, receiver, server)
}

#[test]
fn shutdown_request_sends_the_private_token_and_accepts_no_content() {
    let temp = tempfile::tempdir().unwrap();
    let _environment = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(temp.path().as_os_str())),
        ("HOME", Some(temp.path().as_os_str())),
    ]);
    let key = crate::configuration::BootstrapChallengeKey::load().unwrap();
    let (url, request, server) = serve_verified_shutdown(
        key,
        b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
    );

    request_shutdown(&url, "fingerprint", "private-token").unwrap();

    let request = request.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(
        request.starts_with("POST /bootstrap/shutdown HTTP/1.1"),
        "{request}"
    );
    assert!(
        request.contains("X-NeMo-Relay-Bootstrap-Token: private-token"),
        "{request}"
    );
    server.join().unwrap();
}

#[test]
fn shutdown_request_reports_rejection_without_hiding_the_status() {
    let temp = tempfile::tempdir().unwrap();
    let _environment = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(temp.path().as_os_str())),
        ("HOME", Some(temp.path().as_os_str())),
    ]);
    let key = crate::configuration::BootstrapChallengeKey::load().unwrap();
    let (url, _, server) = serve_verified_shutdown(
        key,
        b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
    );

    let error = request_shutdown(&url, "fingerprint", "wrong-token").unwrap_err();

    assert!(error.contains("rejected shutdown"), "{error}");
    assert!(error.contains("HTTP/1.1 403 Forbidden"), "{error}");
    server.join().unwrap();
}

#[test]
fn shutdown_request_rejects_a_malformed_http_response() {
    let temp = tempfile::tempdir().unwrap();
    let _environment = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(temp.path().as_os_str())),
        ("HOME", Some(temp.path().as_os_str())),
    ]);
    let key = crate::configuration::BootstrapChallengeKey::load().unwrap();
    let (url, _, server) = serve_verified_shutdown(key, b"not-http");

    let error = request_shutdown(&url, "fingerprint", "private-token").unwrap_err();

    assert!(error.contains("malformed shutdown response"), "{error}");
    server.join().unwrap();
}

#[test]
fn shutdown_request_reports_connection_failure() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);

    let error = request_shutdown(&url, "fingerprint", "private-token").unwrap_err();

    assert!(error.contains("failed to connect"), "{error}");
}

#[test]
fn health_probe_classifies_invalid_and_malformed_endpoints_as_unavailable_or_foreign() {
    assert_eq!(probe("not a URL", None), RelayHealth::Unavailable);

    let (url, _, server) = serve_once(b"not-http");
    assert_eq!(probe(&url, None), RelayHealth::Foreign);
    server.join().unwrap();

    let body = format!(
        r#"{{"status":"starting","service":"nemo-relay","version":"{}","bootstrap_protocol":{}}}"#,
        env!("CARGO_PKG_VERSION"),
        BOOTSTRAP_PROTOCOL_VERSION
    );
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let (url, _, server) = serve_once(response.as_bytes());
    assert_eq!(probe(&url, None), RelayHealth::Foreign);
    server.join().unwrap();
}

#[test]
fn loopback_helpers_normalize_localhost_and_ipv6_authorities() {
    assert_eq!(
        loopback_bind("http://localhost:47632").unwrap(),
        "127.0.0.1:47632".parse().unwrap()
    );
    assert_eq!(loopback_authority("::1", 47632), "[::1]:47632");
}

#[test]
fn verified_hook_payload_is_not_sent_before_the_tls_tunnel_is_authenticated() {
    let temp = tempfile::tempdir().unwrap();
    let _environment = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(temp.path().as_os_str())),
        ("HOME", Some(temp.path().as_os_str())),
    ]);
    crate::configuration::BootstrapChallengeKey::load().unwrap();
    crate::gateway::tls::RelayTlsIdentity::load_or_create().unwrap();
    let (url, request, server) = serve_once(
        b"HTTP/1.1 101 Switching Protocols\r\nConnection: upgrade\r\nUpgrade: nemo-relay-tls\r\nContent-Length: 0\r\n\r\n",
    );

    let error = post_verified(
        &url,
        "fingerprint",
        "/hooks/codex",
        &[],
        b"secret-hook-payload",
        Duration::from_secs(2),
        1024,
    )
    .unwrap_err();

    let request = request.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(
        !request
            .windows(19)
            .any(|window| window == b"secret-hook-payload")
    );
    assert!(error.to_string().contains("authenticated Relay TLS tunnel"));
    server.join().unwrap();
}

#[test]
fn verified_transport_reuses_loaded_bootstrap_credentials() {
    let temp = tempfile::tempdir().unwrap();
    let _environment = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(temp.path().as_os_str())),
        ("HOME", Some(temp.path().as_os_str())),
    ]);
    crate::configuration::BootstrapChallengeKey::load().unwrap();
    crate::gateway::tls::RelayTlsIdentity::load_or_create().unwrap();

    let first_key = cached_bootstrap_challenge_key().unwrap();
    let second_key = cached_bootstrap_challenge_key().unwrap();
    assert!(std::sync::Arc::ptr_eq(&first_key, &second_key));

    let first_identity = cached_tls_identity().unwrap();
    let second_identity = cached_tls_identity().unwrap();
    assert!(std::sync::Arc::ptr_eq(&first_identity, &second_identity));
}
