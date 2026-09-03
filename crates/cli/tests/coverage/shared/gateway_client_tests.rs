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
fn gateway_client_rejects_non_loopback_urls_and_classifies_health_payloads() {
    for url in [
        "https://127.0.0.1:47632",
        "http://example.test:47632",
        "http://127.0.0.1",
    ] {
        assert!(parse_loopback_url(url).is_err(), "{url}");
    }
    assert_eq!(
        http_header(b"HTTP/1.1 200 OK\r\nX-Test: value\r\n", "x-test"),
        Some("value")
    );
    assert_eq!(http_status(b"HTTP/1.1 204 No Content"), Some(204));
    assert_eq!(http_status(b"HTTP/2 200"), None);

    let body = format!(
        r#"{{"status":"ok","service":"nemo-relay","bootstrap_protocol":{},"instance_id":"instance-1"}}"#,
        BOOTSTRAP_PROTOCOL_VERSION
    );
    assert_eq!(
        classify_health_response(b"HTTP/1.1 200 OK", body.as_bytes(), None),
        (RelayHealth::Compatible, Some("instance-1".into()))
    );
    assert_eq!(
        classify_health_response(b"HTTP/1.1 409 Conflict", body.as_bytes(), None),
        (RelayHealth::Incompatible, None)
    );
    assert_eq!(
        classify_health_response(
            b"HTTP/1.1 200 OK",
            br#"{"status":"ok","service":"other"}"#,
            None,
        ),
        (RelayHealth::Foreign, None)
    );
}

#[test]
fn http_message_reader_enforces_verified_transport_boundaries() {
    let mut response = std::io::Cursor::new(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok");
    assert_eq!(
        read_http_message(&mut response, 2).unwrap(),
        (
            b"HTTP/1.1 200 OK\r\nContent-Length: 2".to_vec(),
            b"ok".to_vec()
        )
    );
    for response in [
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".as_slice(),
        b"HTTP/1.1 200 OK\r\n\r\n".as_slice(),
        b"HTTP/1.1 200 OK\r\nContent-Length: invalid\r\n\r\n".as_slice(),
    ] {
        assert!(read_http_message(&mut std::io::Cursor::new(response), 2).is_err());
    }
    assert!(
        read_http_message(
            &mut std::io::Cursor::new(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nok"),
            2,
        )
        .is_err()
    );
    assert!(connect_loopback(std::iter::empty(), Duration::from_millis(1)).is_err());
}

#[test]
fn gateway_client_helpers_reject_invalid_protocol_details() {
    assert_eq!(
        VerifiedHttpError::before_payload("before").to_string(),
        "before"
    );
    assert_eq!(
        VerifiedHttpError::after_payload("after").to_string(),
        "after"
    );
    assert!(
        VerifiedHttpError::missing_fingerprint()
            .to_string()
            .contains("missing its bootstrap fingerprint")
    );
    assert_eq!(http_header(b"Header-without-value\r\n", "header"), None);
    assert_eq!(http_header(b"X-Test: \xff\r\n", "x-test"), None);
    for response in [
        b"".as_slice(),
        b"HTTP/2 200 OK".as_slice(),
        b"HTTP/1.1 not-a-status".as_slice(),
        b"NOT HTTP".as_slice(),
    ] {
        assert_eq!(http_status(response), None, "{response:?}");
    }

    assert_eq!(split_http_response(b"invalid"), None);
    assert_eq!(
        split_http_response(b"HTTP/1.1 200 OK\r\n\r\nbody"),
        Some((&b"HTTP/1.1 200 OK"[..], &b"body"[..]))
    );
    assert_eq!(
        parse_loopback_url("http://[::1]:47632").unwrap(),
        ("::1".to_string(), 47632)
    );
    assert!(loopback_bind("http://[::1]:47632").is_ok());
}

#[test]
fn health_response_validation_rejects_every_untrusted_shape() {
    let valid = format!(
        r#"{{"status":"ok","service":"nemo-relay","bootstrap_protocol":{},"instance_id":"instance-1"}}"#,
        BOOTSTRAP_PROTOCOL_VERSION
    );
    let headers = b"HTTP/1.1 200 OK";
    for body in [
        b"not-json".as_slice(),
        br#"{"status":"ok","service":"nemo-relay"}"#.as_slice(),
        br#"{"status":"starting","service":"nemo-relay","bootstrap_protocol":1}"#.as_slice(),
        br#"{"status":"ok","service":"nemo-relay","bootstrap_protocol":1,"instance_id":""}"#
            .as_slice(),
    ] {
        assert_eq!(
            classify_health_response(headers, body, None),
            (RelayHealth::Foreign, None),
            "{body:?}"
        );
    }
    let too_long = format!(
        r#"{{"status":"ok","service":"nemo-relay","bootstrap_protocol":{},"instance_id":"{}"}}"#,
        BOOTSTRAP_PROTOCOL_VERSION,
        "x".repeat(129)
    );
    assert_eq!(
        classify_health_response(headers, too_long.as_bytes(), None),
        (RelayHealth::Foreign, None)
    );

    let temp = tempfile::tempdir().unwrap();
    let _environment = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(temp.path().as_os_str())),
        ("HOME", Some(temp.path().as_os_str())),
    ]);
    let key = crate::configuration::BootstrapChallengeKey::load().unwrap();
    assert_eq!(
        classify_health_response(
            headers,
            valid.as_bytes(),
            Some(("fingerprint", "nonce", &key))
        ),
        (RelayHealth::Foreign, None)
    );
    let invalid_proof = "HTTP/1.1 200 OK\r\nX-NeMo-Relay-Bootstrap-Proof: invalid\r\n".to_string();
    assert_eq!(
        classify_health_response(
            invalid_proof.as_bytes(),
            valid.as_bytes(),
            Some(("fingerprint", "nonce", &key)),
        ),
        (RelayHealth::Foreign, None)
    );
    let proof = key.proof("fingerprint", "nonce");
    let authenticated_headers =
        format!("HTTP/1.1 200 OK\r\nX-NeMo-Relay-Bootstrap-Proof: {proof}\r\n");
    assert_eq!(
        classify_health_response(
            authenticated_headers.as_bytes(),
            valid.as_bytes(),
            Some(("fingerprint", "nonce", &key)),
        ),
        (RelayHealth::Compatible, Some("instance-1".into()))
    );
}

#[test]
fn http_message_reader_handles_upgrade_and_incomplete_messages() {
    let mut upgraded = std::io::Cursor::new(
        b"HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\n\r\n".to_vec(),
    );
    assert_eq!(
        read_http_message(&mut upgraded, 0).unwrap(),
        (
            b"HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade".to_vec(),
            Vec::new()
        )
    );
    assert!(read_http_message(&mut std::io::Cursor::new(b""), 1).is_err());
    assert!(
        read_http_message(
            &mut std::io::Cursor::new(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\no"),
            2,
        )
        .is_err()
    );
    let oversized = vec![b'x'; 16 * 1024];
    assert!(read_http_message(&mut std::io::Cursor::new(oversized), 1).is_err());
}

#[test]
fn loopback_url_helpers_cover_address_and_port_rejections() {
    for url in ["http://example.test:80", "http://127.0.0.1", "http://[::1]"] {
        assert!(parse_loopback_url(url).is_err(), "{url}");
    }
    assert_eq!(loopback_authority("localhost", 80), "localhost:80");
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let closed = listener.local_addr().unwrap();
    drop(listener);
    assert!(connect_loopback([closed], Duration::from_millis(5)).is_err());
}

#[test]
fn verified_hook_payload_is_not_sent_before_the_tls_tunnel_is_authenticated() {
    let temp = tempfile::tempdir().unwrap();
    let _environment = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(temp.path().as_os_str())),
        ("HOME", Some(temp.path().as_os_str())),
    ]);
    let key = crate::configuration::BootstrapChallengeKey::load().unwrap();
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

    let request = String::from_utf8(request.recv_timeout(Duration::from_secs(2)).unwrap()).unwrap();
    assert_eq!(
        header(
            &request,
            crate::configuration::BOOTSTRAP_CLIENT_TOKEN_HEADER
        ),
        key.client_token()
    );
    assert!(
        !request
            .as_bytes()
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
