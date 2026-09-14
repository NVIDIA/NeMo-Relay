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

#[test]
fn configured_headers_require_protected_remote_destinations() {
    assert!(validate_header_http_endpoint("https://collector.example/v1/logs").is_ok());
    assert!(validate_header_http_endpoint("http://localhost:4318/v1/logs").is_ok());
    assert!(validate_header_http_endpoint("http://127.0.0.1:4318/v1/logs").is_ok());
    assert!(validate_header_http_endpoint("http://collector.example/v1/logs").is_err());

    assert!(validate_header_websocket_endpoint("wss://collector.example/events").is_ok());
    assert!(validate_header_websocket_endpoint("ws://[::1]:4318/events").is_ok());
    assert!(validate_header_websocket_endpoint("ws://collector.example/events").is_err());

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
        reqwest_otel::blocking::Client::builder().build().unwrap(),
        HeaderFileResolver::new(files),
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

#[test]
fn http_client_works_without_a_tokio_runtime() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1/logs", listener.local_addr().unwrap());
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buffer = [0; 4_096];
        let bytes_read = stream.read(&mut buffer).unwrap();
        assert!(bytes_read > 0);
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .unwrap();
    });
    let client = HeaderFileHttpClient::new(
        reqwest_otel::blocking::Client::builder().build().unwrap(),
        HeaderFileResolver::new(HashMap::new()),
    );

    futures::executor::block_on(
        client.send_bytes(Request::builder().uri(endpoint).body(Bytes::new()).unwrap()),
    )
    .unwrap();
    server.join().unwrap();
}
