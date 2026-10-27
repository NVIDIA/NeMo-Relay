// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for file-backed observability headers.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use opentelemetry_http::HttpClient;
use tonic::service::Interceptor;

use super::*;

#[test]
fn reads_current_value_and_strips_only_trailing_whitespace() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("token");
    fs::write(&path, "Bearer first\n").unwrap();
    let files: HeaderFiles = HashMap::from([(
        "authorization".to_string(),
        path.to_string_lossy().into_owned(),
    )]);

    assert_eq!(
        resolve_header_files(&files).unwrap()["authorization"],
        "Bearer first"
    );
    fs::write(&path, "Bearer second\n\n").unwrap();
    assert_eq!(
        resolve_header_files(&files).unwrap()["authorization"],
        "Bearer second"
    );
}

#[test]
fn activation_requires_existing_files_and_unique_sources() {
    let headers = HashMap::from([("Authorization".to_string(), "static".to_string())]);
    let files = HashMap::from([("authorization".to_string(), "/missing".to_string())]);
    assert!(
        validate_header_files(&headers, &HashMap::new(), &files)
            .unwrap_err()
            .contains("unique across headers and header_env")
    );

    let files = HashMap::from([("authorization".to_string(), "/missing".to_string())]);
    assert!(
        validate_header_files(&HashMap::new(), &HashMap::new(), &files)
            .unwrap_err()
            .contains("unavailable")
    );

    let files = HashMap::from([("invalid header".to_string(), "/missing".to_string())]);
    assert!(
        validate_header_files(&HashMap::new(), &HashMap::new(), &files)
            .unwrap_err()
            .contains("invalid header name")
    );

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("token");
    fs::write(&path, "not valid\nvalue").unwrap();
    let files = HashMap::from([(
        "authorization".to_string(),
        path.to_string_lossy().into_owned(),
    )]);
    assert!(validate_header_files(&HashMap::new(), &HashMap::new(), &files).is_ok());
    let header_env = HashMap::from([("Authorization".to_string(), "TOKEN".to_string())]);
    assert!(
        validate_header_files(&HashMap::new(), &header_env, &files)
            .unwrap_err()
            .contains("unique across headers and header_env")
    );

    let files = HashMap::from([(
        "authorization".to_string(),
        directory.path().to_string_lossy().into_owned(),
    )]);
    let error = validate_header_files(&HashMap::new(), &HashMap::new(), &files).unwrap_err();
    assert!(error.contains("regular file"));
    assert!(!error.contains(&directory.path().display().to_string()));
}

fn hosts(entries: &[&str]) -> Vec<String> {
    entries.iter().map(|entry| entry.to_string()).collect()
}

#[test]
fn insecure_http_hosts_trust_only_exact_listed_hosts() {
    let trusted = PlaintextPolicy::from_hosts(&hosts(&[
        "Host.Docker.Internal.",
        "release-api",
        "10.0.0.5",
        "fd00::1",
    ]))
    .unwrap();
    for endpoint in [
        "http://host.docker.internal:8082/v1/traces",
        "http://HOST.DOCKER.INTERNAL./v1/traces",
        "http://release-api:8080/anything",
        "http://10.0.0.5:4318/v1/traces",
        "http://[fd00::1]:4318/v1/traces",
        "http://[fd00:0::1]:4318/v1/traces",
        "http://localhost:4318/v1/traces",
    ] {
        assert!(
            validate_header_http_endpoint(endpoint, &trusted).is_ok(),
            "{endpoint}"
        );
        assert!(
            validate_otlp_transport(endpoint, &trusted, "trace").is_ok(),
            "{endpoint}"
        );
    }
    for endpoint in [
        "http://api.release-api:8080/",
        "http://release-api.example:8080/",
        "http://10.0.0.6:4318/v1/traces",
        "http://010.0.0.5:4318/v1/traces",
        "http://0xa.0.0.5:4318/v1/traces",
        "http://host.docker.internal@collector.example/v1/traces",
    ] {
        let error = validate_header_http_endpoint(endpoint, &trusted).unwrap_err();
        assert!(
            error.contains("hosts listed in insecure_http_hosts"),
            "{error}"
        );
        assert!(
            validate_otlp_transport(endpoint, &trusted, "trace").is_err(),
            "{endpoint}"
        );
    }
}

#[test]
fn listed_ip_hosts_never_match_alternate_ipv4_spellings() {
    let trusted = PlaintextPolicy::from_hosts(&hosts(&["10.0.0.8"])).unwrap();
    for endpoint in ["http://10.0.0.010:4317", "http://10.0.0.8.:4317"] {
        let allowed = validate_header_http_endpoint(endpoint, &trusted).is_ok();
        assert_eq!(allowed, endpoint.ends_with("8.:4317"), "{endpoint}");
    }
}

#[test]
fn plaintext_rejections_name_the_redacted_origin_and_the_remedy() {
    let error = validate_header_http_endpoint(
        "http://user:secret@collector.example:4318/v1/logs?token=abc",
        &PlaintextPolicy::LoopbackOnly,
    )
    .unwrap_err();
    assert!(error.contains("http://collector.example:4318"), "{error}");
    assert!(error.contains("allow_insecure"), "{error}");
    for secret in ["secret", "token", "/v1/logs"] {
        assert!(!error.contains(secret), "{error}");
    }

    for endpoint in ["ftp://collector.example/", "https://collector.example/v1 "] {
        let error =
            validate_otlp_transport(endpoint, &PlaintextPolicy::AnyHost, "trace").unwrap_err();
        assert!(error.contains("valid https:// or http:// URI"), "{error}");
        assert!(!error.contains("insecure_http_hosts"), "{error}");
        assert!(!error.contains("collector.example"), "{error}");
    }
}

#[test]
fn any_host_policy_allows_plaintext_everywhere_and_listed_hosts_cover_websockets() {
    for endpoint in ["http://collector.example/v1/traces", "http://10.0.0.6:4318"] {
        assert!(validate_otlp_transport(endpoint, &PlaintextPolicy::AnyHost, "trace").is_ok());
        assert!(validate_header_http_endpoint(endpoint, &PlaintextPolicy::AnyHost).is_ok());
    }
    assert!(
        validate_header_websocket_endpoint(
            "ws://collector.example/events",
            &PlaintextPolicy::AnyHost
        )
        .is_ok()
    );

    let listed = PlaintextPolicy::from_hosts(&hosts(&["release-api"])).unwrap();
    assert!(validate_header_websocket_endpoint("ws://release-api:9000/events", &listed).is_ok());
    let error =
        validate_header_websocket_endpoint("ws://collector.example/events", &listed).unwrap_err();
    assert!(
        error.contains("hosts listed in insecure_http_hosts"),
        "{error}"
    );
    assert_eq!(
        PlaintextPolicy::from_hosts(&[]).unwrap(),
        PlaintextPolicy::LoopbackOnly
    );
    assert_eq!(
        PlaintextPolicy::from_allow_insecure(true),
        PlaintextPolicy::AnyHost
    );
}

#[test]
fn insecure_http_hosts_reject_entries_that_are_not_bare_hosts() {
    assert!(
        validate_insecure_http_hosts(&hosts(&[
            "host.docker.internal",
            "otel_collector",
            "10.0.0.5",
            "[fd00::1]",
            "fd00::1",
        ]))
        .is_ok()
    );
    for entry in [
        "",
        ".",
        "http://release-api",
        "http://otel:glc_TOKEN@collector.example",
        "release-api:8080",
        "[fd00::1]:80",
        "release-api/v1",
        "collector.example/..",
        "*.svc.cluster.local",
        ".svc.cluster.local",
        "%2A.svc.cluster.local",
        "user@release-api",
        "@release-api",
        " release-api",
        "release-api?x",
        "010.000.000.005",
        "10.0.0.010",
        "10.1",
        "1234",
        "0x7f.0.0.1",
        "fe80::1%eth0",
    ] {
        let error = validate_insecure_http_hosts(&hosts(&["release-api", entry])).unwrap_err();
        assert!(
            error.starts_with("insecure_http_hosts[1] must be a bare hostname or IP address"),
            "{entry:?}: {error}"
        );
        assert!(!error.contains("TOKEN"), "{error}");
    }
}

#[test]
fn http_client_consults_insecure_http_hosts_before_resolving_headers() {
    let directory = tempfile::tempdir().unwrap();
    let files = HashMap::from([(
        "authorization".to_string(),
        directory.path().join("missing").display().to_string(),
    )]);
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let send = |client: HeaderFileHttpClient| {
        runtime
            .block_on(
                client.send_bytes(
                    Request::builder()
                        .uri("http://release-api.invalid:4318/v1/logs")
                        .body(Bytes::new())
                        .unwrap(),
                ),
            )
            .unwrap_err()
            .to_string()
    };

    let error = send(HeaderFileHttpClient::new(
        reqwest::Client::builder().build().unwrap(),
        HeaderFileResolver::new(files.clone()),
        PlaintextPolicy::LoopbackOnly,
    ));
    assert!(
        error.contains("configured headers require https"),
        "{error}"
    );

    let error = send(HeaderFileHttpClient::new(
        reqwest::Client::builder().build().unwrap(),
        HeaderFileResolver::new(files),
        PlaintextPolicy::from_hosts(&hosts(&["release-api.invalid"])).unwrap(),
    ));
    assert!(!error.contains("require https"), "{error}");
    assert!(error.contains("authorization"), "{error}");
}

#[test]
fn configured_headers_require_protected_remote_destinations() {
    assert!(
        validate_header_http_endpoint(
            "https://collector.example/v1/logs",
            &PlaintextPolicy::LoopbackOnly
        )
        .is_ok()
    );
    assert!(
        validate_header_http_endpoint(
            "http://localhost:4318/v1/logs",
            &PlaintextPolicy::LoopbackOnly
        )
        .is_ok()
    );
    assert!(
        validate_header_http_endpoint(
            "http://127.0.0.1:4318/v1/logs",
            &PlaintextPolicy::LoopbackOnly
        )
        .is_ok()
    );
    assert!(
        validate_header_http_endpoint(
            "http://collector.example/v1/logs",
            &PlaintextPolicy::LoopbackOnly
        )
        .is_err()
    );

    assert!(
        validate_header_websocket_endpoint(
            "wss://collector.example/events",
            &PlaintextPolicy::default()
        )
        .is_ok()
    );
    assert!(
        validate_header_websocket_endpoint("ws://[::1]:4318/events", &PlaintextPolicy::default())
            .is_ok()
    );
    assert!(
        validate_header_websocket_endpoint(
            "ws://collector.example/events",
            &PlaintextPolicy::default()
        )
        .is_err()
    );

    assert!(has_configured_headers(
        &HashMap::from([("authorization".to_string(), "Bearer static".to_string())]),
        &HashMap::new(),
        &HashMap::new(),
    ));
    assert!(has_configured_headers(
        &HashMap::new(),
        &HashMap::from([("authorization".to_string(), "TOKEN".to_string())]),
        &HashMap::new(),
    ));
    assert!(!has_configured_headers(
        &HashMap::new(),
        &HashMap::new(),
        &HashMap::new(),
    ));
}

#[test]
fn runtime_rejects_blank_or_invalid_values_without_echoing_them() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("token");
    let files: HeaderFiles = HashMap::from([(
        "authorization".to_string(),
        path.to_string_lossy().into_owned(),
    )]);
    fs::write(&path, "\n").unwrap();
    assert!(resolve_header_files(&files).unwrap_err().contains("blank"));
    fs::write(&path, "bad\nvalue").unwrap();
    assert!(
        resolve_header_files(&files)
            .unwrap_err()
            .contains("invalid value")
    );
    fs::write(&path, [0xff]).unwrap();
    assert!(
        resolve_header_files(&files)
            .unwrap_err()
            .contains("could not read")
    );
    fs::remove_file(&path).unwrap();
    assert!(
        resolve_header_files(&files)
            .unwrap_err()
            .contains("could not read")
    );
}

#[test]
fn grpc_interceptor_reads_the_current_value_for_each_request() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("token");
    let files: HeaderFiles = HashMap::from([(
        "authorization".to_string(),
        path.to_string_lossy().into_owned(),
    )]);
    let resolver = HeaderFileResolver::new(files);
    let mut interceptor = HeaderFileInterceptor::new(resolver);

    fs::write(&path, "Bearer first\n").unwrap();
    let request = interceptor.call(tonic::Request::new(())).unwrap();
    assert_eq!(
        request
            .metadata()
            .get("authorization")
            .unwrap()
            .to_str()
            .unwrap(),
        "Bearer first"
    );

    fs::write(&path, "Bearer second\n").unwrap();
    let request = interceptor.call(tonic::Request::new(())).unwrap();
    assert_eq!(
        request
            .metadata()
            .get("authorization")
            .unwrap()
            .to_str()
            .unwrap(),
        "Bearer second"
    );
}

#[test]
fn http_client_reads_current_values_and_stops_before_network_on_resolution_failure() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("token");
    let files: HeaderFiles = HashMap::from([(
        "authorization".to_string(),
        path.to_string_lossy().into_owned(),
    )]);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let received = thread::spawn(move || {
        let mut headers = Vec::new();
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buffer = [0; 4_096];
            let count = stream.read(&mut buffer).unwrap();
            let request = String::from_utf8_lossy(&buffer[..count]);
            headers.push(
                request
                    .lines()
                    .find_map(|line| line.strip_prefix("authorization: "))
                    .unwrap()
                    .to_string(),
            );
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .unwrap();
        }
        headers
    });
    let client = HeaderFileHttpClient::new(
        reqwest::Client::builder().build().unwrap(),
        HeaderFileResolver::new(files),
        PlaintextPolicy::LoopbackOnly,
    );
    let endpoint = format!("http://{address}/v1/logs");
    let runtime = tokio::runtime::Runtime::new().unwrap();

    fs::write(&path, "Bearer first\n").unwrap();
    runtime
        .block_on(
            client.send_bytes(
                Request::builder()
                    .uri(&endpoint)
                    .body(Bytes::new())
                    .unwrap(),
            ),
        )
        .unwrap();
    fs::write(&path, "Bearer second\n").unwrap();
    runtime
        .block_on(
            client.send_bytes(
                Request::builder()
                    .uri(&endpoint)
                    .body(Bytes::new())
                    .unwrap(),
            ),
        )
        .unwrap();
    assert_eq!(received.join().unwrap(), ["Bearer first", "Bearer second"]);

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    fs::remove_file(&path).unwrap();
    let endpoint = format!("http://{}/v1/logs", listener.local_addr().unwrap());
    assert!(
        runtime
            .block_on(
                client.send_bytes(Request::builder().uri(endpoint).body(Bytes::new()).unwrap(),)
            )
            .is_err()
    );
    assert!(listener.accept().is_err());
}

#[tokio::test]
async fn http_client_resolves_file_backed_headers_in_a_tokio_runtime() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("token");
    fs::write(&path, "Bearer token\n").unwrap();
    let files = HashMap::from([(
        "authorization".to_string(),
        path.to_string_lossy().into_owned(),
    )]);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1/logs", listener.local_addr().unwrap());
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buffer = [0; 4_096];
        let bytes_read = stream.read(&mut buffer).unwrap();
        assert!(bytes_read > 0);
        assert!(
            String::from_utf8_lossy(&buffer[..bytes_read]).contains("authorization: Bearer token")
        );
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .unwrap();
    });
    let client = HeaderFileHttpClient::new(
        reqwest::Client::builder().build().unwrap(),
        HeaderFileResolver::new(files),
        PlaintextPolicy::LoopbackOnly,
    );

    client
        .send_bytes(Request::builder().uri(endpoint).body(Bytes::new()).unwrap())
        .await
        .unwrap();
    server.join().unwrap();
}

#[test]
fn transport_errors_preserve_policy_and_endpoint_causes_without_secrets() {
    for signal in ["trace", "log", "metric"] {
        let plaintext = PlaintextPolicy::default();
        let endpoint = "http://private-user:private-password@collector.example:4318/private-path?token=private-query";
        let transport_error =
            validate_otlp_transport_typed(endpoint, &plaintext, signal).unwrap_err();
        assert!(
            transport_error
                .to_string()
                .contains(&format!("OTLP {signal} endpoints require HTTPS"))
        );
        for error in [
            transport_error,
            validate_header_http_endpoint_typed(endpoint, &plaintext).unwrap_err(),
        ] {
            assert!(matches!(
                error,
                EndpointTransportError::PlaintextRejected(_)
            ));
            let message = error.to_string();
            assert!(message.contains("http://collector.example:4318"));
            for secret in [
                "private-user",
                "private-password",
                "private-path",
                "private-query",
            ] {
                assert!(!message.contains(secret));
            }
            assert!(matches!(
                crate::observability::otel::ExporterInitializationError::from(error),
                crate::observability::otel::ExporterInitializationError::PlaintextRejected(_),
            ));
        }
        for endpoint in ["not an endpoint", "ftp://collector.example:4318"] {
            assert!(matches!(
                validate_otlp_transport_typed(endpoint, &plaintext, signal).unwrap_err(),
                EndpointTransportError::InvalidEndpoint(_),
            ));
            assert!(matches!(
                validate_header_http_endpoint_typed(endpoint, &plaintext).unwrap_err(),
                EndpointTransportError::InvalidEndpoint(_),
            ));
        }
        assert_eq!(
            validate_otlp_transport(endpoint, &plaintext, signal).unwrap_err(),
            validate_otlp_transport_typed(endpoint, &plaintext, signal)
                .unwrap_err()
                .to_string(),
        );
    }
}
