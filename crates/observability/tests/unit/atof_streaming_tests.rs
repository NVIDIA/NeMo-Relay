// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ATOF transport and delivery tests.

use super::*;
use futures_util::StreamExt;
use nemo_relay_types::api::event::{BaseEvent, Event, MarkEvent};
use serde_json::json;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_tungstenite::tungstenite::handshake::server::{
    Request as WebSocketRequest, Response as WebSocketResponse,
};
use uuid::Uuid;

fn enable_operational_logs() {
    log::set_max_level(log::LevelFilter::Info);
}

fn temp_dir(prefix: &str) -> PathBuf {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("nemo-relay-{prefix}-{id}"));
    fs::create_dir_all(&path).unwrap();
    path.canonicalize().unwrap()
}

fn make_mark_event(name: &str) -> Event {
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name(name)
            .data(json!({"step": 1}))
            .build(),
        None,
        None,
    ))
}

fn read_http_request(stream: &mut std::net::TcpStream) -> String {
    read_http_request_parts(stream).1
}

fn read_http_request_parts(stream: &mut std::net::TcpStream) -> (String, String) {
    let mut data = Vec::new();
    let mut buf = [0_u8; 1];
    while !data.ends_with(b"\r\n\r\n") {
        stream.read_exact(&mut buf).unwrap();
        data.push(buf[0]);
    }
    let headers = String::from_utf8_lossy(&data).to_string();
    if let Some(length) = headers
        .lines()
        .find_map(|line| {
            line.split_once(':').and_then(|(name, value)| {
                name.eq_ignore_ascii_case("content-length")
                    .then_some(value.trim())
            })
        })
        .and_then(|value| value.trim().parse::<usize>().ok())
    {
        let mut body = vec![0_u8; length];
        stream.read_exact(&mut body).unwrap();
        return (headers, String::from_utf8(body).unwrap());
    }
    if headers
        .lines()
        .any(|line| line.eq_ignore_ascii_case("Transfer-Encoding: chunked"))
    {
        let mut body = Vec::new();
        loop {
            let mut size_line = Vec::new();
            loop {
                stream.read_exact(&mut buf).unwrap();
                size_line.push(buf[0]);
                if size_line.ends_with(b"\r\n") {
                    break;
                }
            }
            let size_text = String::from_utf8_lossy(&size_line);
            let size = usize::from_str_radix(size_text.trim(), 16).unwrap();
            if size == 0 {
                let mut trailer = [0_u8; 2];
                stream.read_exact(&mut trailer).unwrap();
                break;
            }
            let mut chunk = vec![0_u8; size];
            stream.read_exact(&mut chunk).unwrap();
            body.extend(chunk);
            let mut crlf = [0_u8; 2];
            stream.read_exact(&mut crlf).unwrap();
        }
        return (headers, String::from_utf8(body).unwrap());
    }
    (headers, String::new())
}

fn start_http_capture_server(expected_requests: usize) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let captures = Arc::new(Mutex::new(Vec::new()));
    let thread_captures = Arc::clone(&captures);
    std::thread::spawn(move || {
        for _ in 0..expected_requests {
            let (mut stream, _) = listener.accept().unwrap();
            let request_captures = Arc::clone(&thread_captures);
            std::thread::spawn(move || {
                let body = read_http_request(&mut stream);
                request_captures.lock().unwrap().push(body);
                stream
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .unwrap();
            });
        }
    });
    (url, captures)
}

fn wait_for_captures(captures: &Arc<Mutex<Vec<String>>>, expected: usize) -> Vec<String> {
    for _ in 0..100 {
        let snapshot = captures.lock().unwrap().clone();
        if snapshot.len() >= expected {
            return snapshot;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    captures.lock().unwrap().clone()
}

fn start_websocket_capture_server(
    listener: TcpListener,
    captures: Arc<Mutex<Vec<String>>>,
    expected_messages: usize,
) {
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let Ok(mut websocket) = tokio_tungstenite::accept_async(stream).await else {
                    continue;
                };
                while let Some(message) = websocket.next().await {
                    let Ok(message) = message else {
                        break;
                    };
                    if message.is_text() {
                        captures
                            .lock()
                            .unwrap()
                            .push(message.into_text().unwrap().to_string());
                        if captures.lock().unwrap().len() >= expected_messages {
                            return;
                        }
                    }
                }
            }
        });
    });
}

fn validate_endpoint_config(config: AtofEndpointConfig) -> Result<ActivatedAtofEndpoint> {
    super::validate_endpoint_config(config, &PlaintextPolicy::LoopbackOnly)
}

fn test_endpoint_diagnostics(
    transport: AtofEndpointTransport,
) -> (EndpointDiagnostics, ExporterRuntimeDiagnosticsRecorder) {
    let recorder = ExporterRuntimeDiagnosticsRecorder::new(None)
        .with_host(crate::test_support::detached_host());
    let diagnostics = EndpointDiagnostics {
        recorder: recorder.clone(),
        transport,
        origin: "http://127.0.0.1:0".to_string(),
    };
    (diagnostics, recorder)
}

fn diagnostic_count(recorder: &ExporterRuntimeDiagnosticsRecorder, code: &str) -> u64 {
    recorder
        .snapshot()
        .get(code)
        .map_or(0, |diagnostic| diagnostic.count)
}

fn diagnostic_message(recorder: &ExporterRuntimeDiagnosticsRecorder, code: &str) -> String {
    recorder
        .snapshot()
        .get(code)
        .unwrap_or_else(|| panic!("diagnostic {code} was not recorded"))
        .message
        .clone()
}

#[test]
fn endpoint_field_name_policy_replaces_dots_recursively() {
    let config =
        AtofEndpointConfig::new("http://127.0.0.1:9/events", AtofEndpointTransport::HttpPost)
            .with_field_name_policy(AtofEndpointFieldNamePolicy::ReplaceDots);
    let transformed = endpoint_event_json(
        &config,
        json!({
            "kind": "scope",
            "metadata": {
                "otel_status_code": "existing",
                "otel.status_code": "OK",
                "nested": [{"a.b": true}]
            }
        })
        .to_string(),
    );
    let value: Json = serde_json::from_str(&transformed).unwrap();
    assert_eq!(value["metadata"]["otel_status_code"], json!("OK"));
    assert_eq!(value["metadata"]["otel_status_code_2"], json!("existing"));
    assert_eq!(value["metadata"]["nested"][0]["a_b"], json!(true));
}

#[test]
fn endpoint_field_name_policy_preserves_raw_json_and_falls_back_for_invalid_json() {
    let preserve =
        AtofEndpointConfig::new("http://127.0.0.1:9/events", AtofEndpointTransport::HttpPost);
    let raw = "{\"metadata\":{\"otel.status_code\":\"OK\"}}";
    assert_eq!(endpoint_event_json(&preserve, raw.into()), raw);

    let replace =
        AtofEndpointConfig::new("http://127.0.0.1:9/events", AtofEndpointTransport::HttpPost)
            .with_field_name_policy(AtofEndpointFieldNamePolicy::ReplaceDots);
    assert_eq!(endpoint_event_json(&replace, "not-json".into()), "not-json");
}

#[test]
fn endpoint_http_helper_edges_are_safe() {
    assert_eq!(
        AtofEndpointFieldNamePolicy::parse("unknown"),
        None,
        "unknown field name policies should be rejected"
    );
}

#[test]
fn streaming_sink_receives_raw_atof_events() {
    enable_operational_logs();
    let dir = temp_dir("atof-streaming-http");
    let (url, captures) = start_http_capture_server(3);
    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl")
            .with_endpoint(AtofEndpointConfig::new(
                url.clone(),
                AtofEndpointTransport::HttpPost,
            )),
    )
    .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
    .unwrap();
    let subscriber = exporter.subscriber();

    subscriber(&make_mark_event("first"));
    subscriber(&make_mark_event("second"));
    exporter.force_flush().unwrap();
    subscriber(&make_mark_event("after-flush"));
    exporter.shutdown().unwrap();

    assert_eq!(exporter.path(), None);
    let bodies = wait_for_captures(&captures, 3);
    assert_eq!(bodies.len(), 3, "captured bodies: {bodies:?}");
    let all_streamed = bodies.join("");
    assert!(all_streamed.contains("\"name\":\"first\""));
    assert!(all_streamed.contains("\"name\":\"second\""));
    assert!(all_streamed.contains("\"name\":\"after-flush\""));
    assert_eq!(
        all_streamed
            .lines()
            .filter(|line| line.contains("\"kind\":\"mark\""))
            .count(),
        3,
        "three HTTP POST records plus three NDJSON records: {bodies:?}"
    );
}

#[test]
fn websocket_endpoint_receives_fifo_json_text_events() {
    enable_operational_logs();
    let dir = temp_dir("atof-streaming-websocket");
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("ws://{}", listener.local_addr().unwrap());
    let captures = Arc::new(Mutex::new(Vec::new()));
    start_websocket_capture_server(listener, Arc::clone(&captures), 2);

    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl")
            .with_endpoint(AtofEndpointConfig::new(
                url,
                AtofEndpointTransport::Websocket,
            )),
    )
    .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
    .unwrap();
    let subscriber = exporter.subscriber();

    subscriber(&make_mark_event("first"));
    subscriber(&make_mark_event("second"));
    exporter.force_flush().unwrap();

    let messages = wait_for_captures(&captures, 2);
    assert_eq!(messages.len(), 2);
    assert!(messages[0].contains("\"name\":\"first\""));
    assert!(messages[1].contains("\"name\":\"second\""));
}

#[test]
fn websocket_flush_drains_events_queued_before_reconnect() {
    enable_operational_logs();
    let dir = temp_dir("atof-streaming-websocket-reconnect");
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let local_addr = listener.local_addr().unwrap();
    drop(listener);

    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl")
            .with_endpoint(
                AtofEndpointConfig::new(
                    format!("ws://{local_addr}"),
                    AtofEndpointTransport::Websocket,
                )
                .with_timeout_millis(1_000),
            ),
    )
    .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
    .unwrap();
    let subscriber = exporter.subscriber();

    subscriber(&make_mark_event("first"));
    subscriber(&make_mark_event("second"));
    std::thread::sleep(std::time::Duration::from_millis(300));

    let listener = TcpListener::bind(local_addr).unwrap();
    listener.set_nonblocking(true).unwrap();
    let captures = Arc::new(Mutex::new(Vec::new()));
    start_websocket_capture_server(listener, Arc::clone(&captures), 2);

    exporter.force_flush().unwrap();

    let messages = wait_for_captures(&captures, 2);
    assert_eq!(messages.len(), 2);
    assert!(messages[0].contains("\"name\":\"first\""));
    assert!(messages[1].contains("\"name\":\"second\""));
    exporter.shutdown().unwrap();
}

#[test]
fn invalid_endpoint_config_errors_cleanly() {
    let dir = temp_dir("atof-invalid-endpoint");

    let error = match AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl")
            .with_endpoint(AtofEndpointConfig::new(
                "not a url",
                AtofEndpointTransport::HttpPost,
            )),
    )
    .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
    {
        Ok(_) => panic!("expected invalid endpoint config error"),
        Err(error) => error,
    };

    match error {
        AtofExporterError::InvalidEndpoint(message) => {
            assert!(message.contains("endpoints[0]"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn invalid_endpoint_scheme_errors_cleanly() {
    let dir = temp_dir("atof-invalid-endpoint-scheme");

    let cases = [
        (
            AtofEndpointTransport::HttpPost,
            "ws://localhost:8080/events",
            "http_post",
            "ws",
        ),
        (
            AtofEndpointTransport::Ndjson,
            "ws://localhost:8080/events",
            "ndjson",
            "ws",
        ),
        (
            AtofEndpointTransport::Websocket,
            "http://localhost:8080/events",
            "websocket",
            "http",
        ),
    ];

    for (transport, url, transport_name, scheme) in cases {
        let error = match AtofExporter::new(
            AtofExporterConfig::new()
                .with_output_directory(&dir)
                .with_filename(format!("{transport_name}.jsonl"))
                .with_endpoint(AtofEndpointConfig::new(url, transport)),
        )
        .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
        {
            Ok(_) => panic!("expected invalid endpoint scheme error"),
            Err(error) => error,
        };

        match error {
            AtofExporterError::InvalidEndpoint(message) => {
                assert!(message.contains("endpoints[0]"));
                assert!(message.contains(transport_name));
                assert!(message.contains(scheme));
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}

#[test]
fn plaintext_stream_headers_require_a_listed_host_or_allow_insecure() {
    let listed = PlaintextPolicy::from_hosts(&["release-api".to_string()]).unwrap();
    for (url, transport) in [
        (
            "http://release-api:8080/events",
            AtofEndpointTransport::HttpPost,
        ),
        (
            "ws://release-api:8080/events",
            AtofEndpointTransport::Websocket,
        ),
    ] {
        let config =
            || AtofStreamSinkConfig::new(url, transport).with_header("authorization", "Bearer t");
        assert!(validate_endpoint_config(config()).is_err(), "{url}");
        assert!(
            super::validate_endpoint_config(config(), &listed).is_ok(),
            "{url}"
        );
        assert!(
            validate_endpoint_config(config().with_allow_insecure(true)).is_ok(),
            "{url}"
        );
    }
}

#[test]
fn endpoint_validation_rejects_empty_timeout_and_invalid_headers() {
    let mut headers = std::collections::HashMap::new();
    headers.insert("x-test".to_string(), "ok".to_string());
    validate_endpoint_config(AtofEndpointConfig {
        url: "http://127.0.0.1:9/events".into(),
        transport: AtofEndpointTransport::HttpPost,
        headers: headers.clone(),
        header_env: std::collections::HashMap::new(),
        header_file: std::collections::HashMap::new(),
        timeout_millis: 1,
        field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
        allow_insecure: false,
    })
    .unwrap();
    assert_eq!(
        resolved_header_map(&headers, &Default::default())
            .unwrap()
            .len(),
        1
    );
    let env_headers = std::collections::HashMap::from([("x-path".into(), "PATH".into())]);
    assert_eq!(
        resolved_header_map(&Default::default(), &env_headers)
            .unwrap()
            .len(),
        1
    );
    let missing_env = std::collections::HashMap::from([(
        "authorization".into(),
        "NEMO_RELAY_TEST_MISSING_ATOF_SECRET".into(),
    )]);
    assert!(resolved_header_map(&Default::default(), &missing_env).is_err());
    let conflict = std::collections::HashMap::from([("x-test".into(), "PATH".into())]);
    assert!(resolved_header_map(&headers, &conflict).is_err());
    let mixed_case_headers =
        std::collections::HashMap::from([("Authorization".into(), "Bearer literal".into())]);
    let mixed_case_conflict =
        std::collections::HashMap::from([("authorization".into(), "PATH".into())]);
    assert!(resolved_header_map(&mixed_case_headers, &mixed_case_conflict).is_err());

    let empty_url = AtofEndpointConfig {
        url: "  ".into(),
        transport: AtofEndpointTransport::HttpPost,
        headers: std::collections::HashMap::new(),
        header_env: std::collections::HashMap::new(),
        header_file: std::collections::HashMap::new(),
        timeout_millis: 1,
        field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
        allow_insecure: false,
    };
    assert!(
        validate_endpoint_config(empty_url)
            .unwrap_err()
            .to_string()
            .contains("endpoint url must be non-empty")
    );

    let zero_timeout = AtofEndpointConfig {
        url: "http://127.0.0.1:9/events".into(),
        transport: AtofEndpointTransport::HttpPost,
        headers: std::collections::HashMap::new(),
        header_env: std::collections::HashMap::new(),
        header_file: std::collections::HashMap::new(),
        timeout_millis: 0,
        field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
        allow_insecure: false,
    };
    assert!(
        validate_endpoint_config(zero_timeout)
            .unwrap_err()
            .to_string()
            .contains("timeout_millis")
    );

    let mut bad_header_name = std::collections::HashMap::new();
    bad_header_name.insert("bad header".to_string(), "ok".to_string());
    assert!(resolved_header_map(&bad_header_name, &Default::default()).is_err());

    let mut bad_header_value = std::collections::HashMap::new();
    bad_header_value.insert("x-test".to_string(), "bad\nvalue".to_string());
    assert!(resolved_header_map(&bad_header_value, &Default::default()).is_err());
    assert!(
        validate_endpoint_config(AtofEndpointConfig {
            url: "http://127.0.0.1:9/events".into(),
            transport: AtofEndpointTransport::Ndjson,
            headers: bad_header_value,
            header_env: std::collections::HashMap::new(),
            header_file: std::collections::HashMap::new(),
            timeout_millis: 1,
            field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
            allow_insecure: false,
        })
        .is_err()
    );
}

#[test]
fn configured_headers_require_protected_remote_atof_destinations() {
    let _guard = crate::test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let variable = "NEMO_RELAY_TEST_ATOF_TRANSPORT_TOKEN";
    // SAFETY: this uniquely named environment variable is serialized by the observability mutex.
    unsafe { std::env::set_var(variable, "Bearer environment") };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("token");
    fs::write(&path, "Bearer token\n").unwrap();
    let header_file = std::collections::HashMap::from([(
        "authorization".to_string(),
        path.to_string_lossy().into_owned(),
    )]);
    let header_sources = [
        (
            std::collections::HashMap::from([(
                "authorization".to_string(),
                "Bearer static".to_string(),
            )]),
            Default::default(),
            Default::default(),
        ),
        (
            Default::default(),
            std::collections::HashMap::from([("authorization".to_string(), variable.to_string())]),
            Default::default(),
        ),
        (Default::default(), Default::default(), header_file),
    ];

    for (transport, remote, loopback) in [
        (
            AtofEndpointTransport::HttpPost,
            "http://collector.example/events",
            "http://127.0.0.1:4318/events",
        ),
        (
            AtofEndpointTransport::Ndjson,
            "http://collector.example/events",
            "http://127.0.0.1:4318/events",
        ),
        (
            AtofEndpointTransport::Websocket,
            "ws://collector.example/events",
            "ws://127.0.0.1:4318/events",
        ),
    ] {
        let expected = match transport {
            AtofEndpointTransport::HttpPost | AtofEndpointTransport::Ndjson => "require https",
            AtofEndpointTransport::Websocket => "require wss",
        };
        for (headers, header_env, header_file) in &header_sources {
            let remote = AtofEndpointConfig {
                url: remote.into(),
                transport,
                headers: headers.clone(),
                header_env: header_env.clone(),
                header_file: header_file.clone(),
                timeout_millis: 1,
                field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
                allow_insecure: false,
            };
            let error = validate_endpoint_config(remote).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");

            let loopback = AtofEndpointConfig {
                url: loopback.into(),
                transport,
                headers: headers.clone(),
                header_env: header_env.clone(),
                header_file: header_file.clone(),
                timeout_millis: 1,
                field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
                allow_insecure: false,
            };
            assert!(validate_endpoint_config(loopback).is_ok());
        }
    }
    assert!(
        validate_endpoint_config(AtofEndpointConfig::new(
            "http://collector.example/events",
            AtofEndpointTransport::HttpPost,
        ))
        .is_ok()
    );
    // SAFETY: cleanup of the test-only environment variable.
    unsafe { std::env::remove_var(variable) };
}

#[test]
fn file_backed_http_post_headers_reload_for_each_event() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("token");
    fs::write(&path, "Bearer first\n").unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/events", listener.local_addr().unwrap());
    let (headers_tx, headers_rx) = std::sync::mpsc::channel();
    let server = std::thread::spawn(move || {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let (headers, _) = read_http_request_parts(&mut stream);
            let authorization = headers
                .lines()
                .find_map(|line| {
                    line.split_once(':').and_then(|(name, value)| {
                        name.eq_ignore_ascii_case("authorization")
                            .then_some(value.trim().to_string())
                    })
                })
                .unwrap();
            headers_tx.send(authorization).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .unwrap();
        }
    });
    let endpoint = validate_endpoint_config(AtofEndpointConfig {
        url,
        transport: AtofEndpointTransport::HttpPost,
        headers: Default::default(),
        header_env: Default::default(),
        header_file: std::collections::HashMap::from([(
            "authorization".into(),
            path.to_string_lossy().into_owned(),
        )]),
        timeout_millis: 5_000,
        field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
        allow_insecure: false,
    })
    .unwrap();
    let (diagnostics, _recorder) = test_endpoint_diagnostics(AtofEndpointTransport::HttpPost);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let worker = std::thread::spawn(move || {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(run_http_post_endpoint(0, endpoint, rx, &diagnostics));
    });

    tx.send(EndpointMessage::Event("{\"sequence\":1}".into()))
        .unwrap();
    assert_eq!(
        headers_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap(),
        "Bearer first"
    );
    fs::write(&path, "Bearer second\n").unwrap();
    tx.send(EndpointMessage::Event("{\"sequence\":2}".into()))
        .unwrap();
    assert_eq!(
        headers_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap(),
        "Bearer second"
    );
    let (close_tx, close_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Close(close_tx)).unwrap();
    close_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    worker.join().unwrap();
    server.join().unwrap();
}

#[test]
fn file_backed_ndjson_headers_refresh_when_a_stream_starts() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("token");
    fs::write(&path, "Bearer first\n").unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/events", listener.local_addr().unwrap());
    let (headers_tx, headers_rx) = std::sync::mpsc::channel();
    let server = std::thread::spawn(move || {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let (headers, _) = read_http_request_parts(&mut stream);
            let authorization = headers
                .lines()
                .find_map(|line| {
                    line.split_once(':').and_then(|(name, value)| {
                        name.eq_ignore_ascii_case("authorization")
                            .then_some(value.trim().to_string())
                    })
                })
                .unwrap();
            headers_tx.send(authorization).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .unwrap();
        }
    });
    let endpoint = validate_endpoint_config(AtofEndpointConfig {
        url: url.clone(),
        transport: AtofEndpointTransport::Ndjson,
        headers: Default::default(),
        header_env: Default::default(),
        header_file: std::collections::HashMap::from([(
            "authorization".into(),
            path.to_string_lossy().into_owned(),
        )]),
        timeout_millis: 5_000,
        field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
        allow_insecure: false,
    })
    .unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime
        .block_on(build_ndjson_client(&endpoint).unwrap().post(&url).send())
        .unwrap();
    assert_eq!(
        headers_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap(),
        "Bearer first"
    );
    fs::write(&path, "Bearer second\n").unwrap();
    runtime
        .block_on(build_ndjson_client(&endpoint).unwrap().post(&url).send())
        .unwrap();
    assert_eq!(
        headers_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap(),
        "Bearer second"
    );
    server.join().unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    fs::remove_file(&path).unwrap();
    assert!(build_ndjson_client(&endpoint).is_err());
    assert!(matches!(
        listener.accept(),
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock
    ));
}

#[test]
#[allow(clippy::result_large_err)]
fn file_backed_websocket_headers_refresh_on_reconnect() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("token");
    fs::write(&path, "Bearer first\n").unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("ws://{}/events", listener.local_addr().unwrap());
    let (headers_tx, headers_rx) = std::sync::mpsc::channel();
    let server = std::thread::spawn(move || {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async move {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                for _ in 0..2 {
                    let (stream, _) = listener.accept().await.unwrap();
                    let headers_tx = headers_tx.clone();
                    let mut socket = tokio_tungstenite::accept_hdr_async(
                        stream,
                        move |request: &WebSocketRequest, response: WebSocketResponse| {
                            let authorization = request
                                .headers()
                                .get("authorization")
                                .unwrap()
                                .to_str()
                                .unwrap()
                                .to_string();
                            headers_tx.send(authorization).unwrap();
                            Ok(response)
                        },
                    )
                    .await
                    .unwrap();
                    socket.close(None).await.unwrap();
                }
            });
    });
    let endpoint = validate_endpoint_config(AtofEndpointConfig {
        url,
        transport: AtofEndpointTransport::Websocket,
        headers: Default::default(),
        header_env: Default::default(),
        header_file: std::collections::HashMap::from([(
            "authorization".into(),
            path.to_string_lossy().into_owned(),
        )]),
        timeout_millis: 5_000,
        field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
        allow_insecure: false,
    })
    .unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let socket = runtime.block_on(connect_websocket(&endpoint)).unwrap();
    assert_eq!(
        headers_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap(),
        "Bearer first"
    );
    drop(socket);
    fs::write(&path, "Bearer second\n").unwrap();
    let socket = runtime.block_on(connect_websocket(&endpoint)).unwrap();
    assert_eq!(
        headers_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap(),
        "Bearer second"
    );
    drop(socket);
    server.join().unwrap();
}

#[test]
fn endpoint_activation_snapshots_header_env() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let variable = format!("NEMO_RELAY_TEST_ATOF_HEADER_ENV_{}", std::process::id());
    // SAFETY: The test mutex serializes environment access for this process.
    unsafe { std::env::set_var(&variable, "Bearer relay-499") };

    let endpoint = validate_endpoint_config(AtofEndpointConfig {
        url: "http://127.0.0.1:9/events".into(),
        transport: AtofEndpointTransport::HttpPost,
        headers: std::collections::HashMap::new(),
        header_env: std::collections::HashMap::from([("authorization".into(), variable.clone())]),
        header_file: std::collections::HashMap::new(),
        timeout_millis: 1,
        field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
        allow_insecure: false,
    })
    .unwrap();

    // SAFETY: The activated endpoint must not read this environment variable again.
    unsafe { std::env::remove_var(&variable) };
    assert_eq!(
        endpoint.headers.get("authorization").unwrap(),
        "Bearer relay-499"
    );
}

#[test]
fn endpoint_worker_helpers_acknowledge_flush_and_close_error_paths() {
    enable_operational_logs();
    let (diagnostics, recorder) = test_endpoint_diagnostics(AtofEndpointTransport::Ndjson);
    let (body_tx, body, _written) = ndjson_body_channel();
    drop(body);
    assert!(!send_ndjson_event(0, &body_tx, "{}".into(), &diagnostics));
    let (flush_tx, flush_rx) = std::sync::mpsc::channel();
    send_ndjson_flush(0, &body_tx, flush_tx, &diagnostics);
    flush_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .unwrap();
    assert_eq!(
        diagnostic_count(&recorder, "atof.stream_delivery_failed"),
        1
    );
    assert_eq!(diagnostic_count(&recorder, "atof.stream_flush_failed"), 1);
    assert_eq!(
        diagnostic_message(&recorder, "atof.stream_flush_failed"),
        "ATOF ndjson stream to http://127.0.0.1:0 ended its upload before a flush was acknowledged"
    );

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tx.send(EndpointMessage::Event("{}".into())).unwrap();
    let (flush_tx, flush_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Flush(flush_tx)).unwrap();
    let (close_tx, close_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Close(close_tx)).unwrap();
    drop(tx);

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async { drain_closed(rx, &diagnostics).await });
    flush_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .unwrap();
    close_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .unwrap();
}

#[test]
fn endpoint_worker_reports_flush_and_close_timeouts() {
    enable_operational_logs();
    let (diagnostics, recorder) = test_endpoint_diagnostics(AtofEndpointTransport::HttpPost);
    let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
    let worker = AtofEndpointWorker {
        sender,
        timeout: std::time::Duration::from_millis(1),
        index: 7,
        diagnostics,
    };

    worker.flush();
    worker.close();
    assert_eq!(
        diagnostic_message(&recorder, "atof.stream_flush_failed"),
        "ATOF http_post stream to http://127.0.0.1:0 did not acknowledge a flush within 1 ms"
    );
    assert_eq!(
        diagnostic_message(&recorder, "atof.stream_close_failed"),
        "ATOF http_post stream to http://127.0.0.1:0 did not acknowledge close within 1 ms"
    );
}

#[test]
fn endpoint_worker_reports_drops_flush_and_close_after_its_worker_stopped() {
    enable_operational_logs();
    let (diagnostics, recorder) = test_endpoint_diagnostics(AtofEndpointTransport::HttpPost);
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
    drop(receiver);
    let timeout = std::time::Duration::from_secs(5);
    let worker = AtofEndpointWorker {
        sender,
        timeout,
        index: 7,
        diagnostics,
    };

    let started = std::time::Instant::now();
    worker.enqueue("{}".into());
    worker.flush();
    worker.close();
    assert!(started.elapsed() < timeout);
    assert_eq!(
        diagnostic_message(&recorder, "atof.stream_events_dropped"),
        "ATOF http_post stream to http://127.0.0.1:0 dropped an event because its worker stopped"
    );
    assert_eq!(
        diagnostic_message(&recorder, "atof.stream_flush_failed"),
        "ATOF http_post stream to http://127.0.0.1:0 did not acknowledge a flush because its worker stopped"
    );
    assert_eq!(
        diagnostic_message(&recorder, "atof.stream_close_failed"),
        "ATOF http_post stream to http://127.0.0.1:0 did not acknowledge close because its worker stopped"
    );
}

fn worker_that_drops_one_request(
    diagnostics: EndpointDiagnostics,
    timeout: std::time::Duration,
) -> (AtofEndpointWorker, std::thread::JoinHandle<()>) {
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let thread = std::thread::spawn(move || drop(receiver.blocking_recv()));
    let worker = AtofEndpointWorker {
        sender,
        timeout,
        index: 7,
        diagnostics,
    };
    (worker, thread)
}

#[test]
fn endpoint_worker_reports_flush_and_close_dropped_by_a_stopping_worker() {
    enable_operational_logs();
    let (diagnostics, recorder) = test_endpoint_diagnostics(AtofEndpointTransport::HttpPost);
    let timeout = std::time::Duration::from_secs(5);

    let started = std::time::Instant::now();
    let (worker, thread) = worker_that_drops_one_request(diagnostics.clone(), timeout);
    worker.flush();
    thread.join().unwrap();
    let (worker, thread) = worker_that_drops_one_request(diagnostics, timeout);
    worker.close();
    thread.join().unwrap();
    assert!(started.elapsed() < timeout);
    assert_eq!(
        diagnostic_message(&recorder, "atof.stream_flush_failed"),
        "ATOF http_post stream to http://127.0.0.1:0 did not acknowledge a flush because its worker stopped"
    );
    assert_eq!(
        diagnostic_message(&recorder, "atof.stream_close_failed"),
        "ATOF http_post stream to http://127.0.0.1:0 did not acknowledge close because its worker stopped"
    );
}

#[test]
fn http_endpoint_worker_acknowledges_flush_close_and_logs_http_errors() {
    enable_operational_logs();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let _ = read_http_request(&mut stream);
        stream
            .write_all(
                b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .unwrap();
    });

    let (diagnostics, recorder) = test_endpoint_diagnostics(AtofEndpointTransport::HttpPost);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let worker = std::thread::spawn(move || {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            run_http_post_endpoint(
                0,
                validate_endpoint_config(
                    AtofEndpointConfig::new(url, AtofEndpointTransport::HttpPost)
                        .with_timeout_millis(5_000),
                )
                .unwrap(),
                rx,
                &diagnostics,
            )
            .await;
        });
    });

    tx.send(EndpointMessage::Event("{\"kind\":\"mark\"}".into()))
        .unwrap();
    let (flush_tx, flush_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Flush(flush_tx)).unwrap();
    flush_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    let (close_tx, close_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Close(close_tx)).unwrap();
    close_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    worker.join().unwrap();
    server.join().unwrap();
    assert_eq!(
        diagnostic_count(&recorder, "atof.stream_delivery_failed"),
        1
    );
    assert_eq!(
        diagnostic_message(&recorder, "atof.stream_delivery_failed"),
        "ATOF http_post stream to http://127.0.0.1:0 returned status 500 Internal Server Error"
    );
}

#[test]
fn http_endpoint_worker_reports_request_transport_failure() {
    enable_operational_logs();
    let (diagnostics, recorder) = test_endpoint_diagnostics(AtofEndpointTransport::HttpPost);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let worker = std::thread::spawn(move || {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            run_http_post_endpoint(
                0,
                validate_endpoint_config(
                    AtofEndpointConfig::new(
                        "http://127.0.0.1:0/events",
                        AtofEndpointTransport::HttpPost,
                    )
                    .with_timeout_millis(100),
                )
                .unwrap(),
                rx,
                &diagnostics,
            )
            .await;
        });
    });

    tx.send(EndpointMessage::Event("{\"kind\":\"mark\"}".into()))
        .unwrap();
    let (close_tx, close_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Close(close_tx)).unwrap();
    close_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    worker.join().unwrap();
    assert_eq!(
        diagnostic_count(&recorder, "atof.stream_delivery_failed"),
        1
    );
    assert!(
        diagnostic_message(&recorder, "atof.stream_delivery_failed").starts_with(
            "ATOF http_post stream to http://127.0.0.1:0 could not complete a request: "
        )
    );
}

#[test]
fn ndjson_endpoint_worker_streams_events_flushes_and_closes() {
    enable_operational_logs();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let body = read_http_request(&mut stream);
        assert_eq!(body, "{\"kind\":\"mark\"}\n");
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .unwrap();
    });

    let endpoint = validate_endpoint_config(
        AtofEndpointConfig::new(url, AtofEndpointTransport::Ndjson).with_timeout_millis(5_000),
    )
    .unwrap();
    let (diagnostics, _recorder) = test_endpoint_diagnostics(AtofEndpointTransport::Ndjson);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let worker = std::thread::spawn(move || run_endpoint_worker(0, endpoint, rx, diagnostics));

    tx.send(EndpointMessage::Event("{\"kind\":\"mark\"}".into()))
        .unwrap();
    let (flush_tx, flush_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Flush(flush_tx)).unwrap();
    flush_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    let (close_tx, close_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Close(close_tx)).unwrap();
    close_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    worker.join().unwrap();
    server.join().unwrap();
}

fn read_until_contains(stream: &mut std::net::TcpStream, needle: &str) -> String {
    let mut data = Vec::new();
    let mut buf = [0_u8; 1];
    while !String::from_utf8_lossy(&data).contains(needle) {
        stream.read_exact(&mut buf).unwrap();
        data.push(buf[0]);
    }
    String::from_utf8_lossy(&data).to_string()
}

fn wait_for_diagnostic_count(
    recorder: &ExporterRuntimeDiagnosticsRecorder,
    code: &str,
    count: u64,
) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while diagnostic_count(recorder, code) < count {
        assert!(
            std::time::Instant::now() < deadline,
            "{code} did not reach {count}"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[test]
fn ndjson_rejection_while_open_records_queued_events_before_close() {
    enable_operational_logs();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (release_server, server_released) = std::sync::mpsc::channel::<()>();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        // Read through the chunk terminator so the close below sends FIN, not RST.
        read_until_contains(&mut stream, "{\"kind\":\"first\"}\n\r\n");
        stream
            .write_all(
                b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .unwrap();
        stream.flush().unwrap();
        stream.shutdown(std::net::Shutdown::Write).unwrap();
        server_released
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
    });

    let endpoint = validate_endpoint_config(
        AtofEndpointConfig::new(url, AtofEndpointTransport::Ndjson).with_timeout_millis(5_000),
    )
    .unwrap();
    let (diagnostics, recorder) = test_endpoint_diagnostics(AtofEndpointTransport::Ndjson);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let worker = std::thread::spawn(move || run_endpoint_worker(0, endpoint, rx, diagnostics));

    tx.send(EndpointMessage::Event("{\"kind\":\"first\"}".into()))
        .unwrap();
    wait_for_diagnostic_count(&recorder, "atof.stream_delivery_failed", 1);
    assert_eq!(
        diagnostic_message(&recorder, "atof.stream_delivery_failed"),
        "ATOF ndjson stream to http://127.0.0.1:0 returned status 503 Service Unavailable"
    );
    release_server.send(()).unwrap();
    server.join().unwrap();

    tx.send(EndpointMessage::Event("{\"kind\":\"second\"}".into()))
        .unwrap();
    let (flush_tx, flush_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Flush(flush_tx)).unwrap();
    flush_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    assert_eq!(
        diagnostic_message(&recorder, "atof.stream_flush_failed"),
        "ATOF ndjson stream to http://127.0.0.1:0 ended its upload before a flush was acknowledged"
    );
    let (close_tx, close_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Close(close_tx)).unwrap();
    close_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    worker.join().unwrap();
    assert_eq!(
        diagnostic_count(&recorder, "atof.stream_delivery_failed"),
        2
    );
    assert!(
        diagnostic_message(&recorder, "atof.stream_delivery_failed")
            .contains("ended its upload before an event was written")
    );
    assert!(
        recorder
            .snapshot()
            .get("atof.stream_close_failed")
            .is_none()
    );
}

#[test]
fn websocket_error_detail_omits_the_endpoint_url() {
    use tokio_tungstenite::tungstenite::Error;
    use tokio_tungstenite::tungstenite::error::UrlError;
    assert_eq!(
        websocket_error_detail(&Error::Url(UrlError::UnableToConnect(
            "ws://127.0.0.1:1/events?token=secret".into()
        ))),
        "unable to connect"
    );
    assert_eq!(
        websocket_error_detail(&Error::Url(UrlError::UnsupportedUrlScheme)),
        "URL error: URL scheme not supported"
    );
    assert_eq!(
        websocket_error_detail(&Error::Io(std::io::ErrorKind::ConnectionRefused.into())),
        "IO error: connection refused"
    );
}

#[test]
fn websocket_helpers_cover_invalid_headers_and_timeout_reconnect_path() {
    enable_operational_logs();
    let endpoint = validate_endpoint_config(AtofEndpointConfig {
        url: "ws://127.0.0.1:9/events".into(),
        transport: AtofEndpointTransport::Websocket,
        headers: std::collections::HashMap::new(),
        header_env: std::collections::HashMap::new(),
        header_file: std::collections::HashMap::new(),
        timeout_millis: 1,
        field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
        allow_insecure: false,
    })
    .unwrap();
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        assert!(connect_websocket(&endpoint).await.is_err());

        let (diagnostics, recorder) = test_endpoint_diagnostics(AtofEndpointTransport::Websocket);
        let mut socket = None;
        let mut pending = std::collections::VecDeque::from(["{\"kind\":\"mark\"}".to_string()]);
        let mut retry = WebSocketRetryState::default();
        assert!(
            !drain_websocket_pending(
                0,
                &endpoint,
                &mut socket,
                &mut pending,
                &mut retry,
                &diagnostics
            )
            .await
        );
        assert_eq!(pending.len(), 1);
        assert_eq!(retry.attempts, 1);
        assert_eq!(
            diagnostic_count(&recorder, "atof.stream_connection_failed"),
            1
        );
        assert!(
            diagnostic_message(&recorder, "atof.stream_connection_failed")
                .starts_with("ATOF websocket stream to http://127.0.0.1:0 could not connect: ")
        );
    });
}

#[test]
fn endpoint_validation_helpers_cover_header_retry_and_collision_edges() {
    enable_operational_logs();

    let invalid_name = std::collections::HashMap::from([("bad header".into(), "value".into())]);
    assert!(resolved_header_map(&invalid_name, &std::collections::HashMap::new()).is_err());

    let invalid_value = std::collections::HashMap::from([("x-test".into(), "bad\nvalue".into())]);
    assert!(resolved_header_map(&invalid_value, &std::collections::HashMap::new()).is_err());

    let headers = std::collections::HashMap::from([("x-test".into(), "value".into())]);
    let header_env = std::collections::HashMap::from([("x-test".into(), "UNUSED_ENV".into())]);
    assert!(resolved_header_map(&headers, &header_env).is_err());

    let (diagnostics, recorder) = test_endpoint_diagnostics(AtofEndpointTransport::Websocket);
    let mut retry = WebSocketRetryState::default();
    retry.record_failure(7, &diagnostics, "could not connect: refused");
    retry.record_failure(7, &diagnostics, "could not connect: refused");
    assert_eq!(retry.attempts, 2);
    assert_eq!(
        diagnostic_count(&recorder, "atof.stream_connection_failed"),
        2
    );
    assert!(retry.warning_emitted);
    retry.record_recovered(7);
    assert_eq!(retry.attempts, 0);
    assert!(!retry.warning_emitted);
    assert!(retry.access_validated);
    retry.record_recovered(7);

    let object =
        serde_json::Map::from_iter([("key".into(), Json::Null), ("key_2".into(), Json::Null)]);
    assert_eq!(collision_free_key(&object, "new".into()), "new");
    assert_eq!(collision_free_key(&object, "key".into()), "key_3");

    for (transport, name) in [
        (AtofEndpointTransport::HttpPost, "http_post"),
        (AtofEndpointTransport::Websocket, "websocket"),
        (AtofEndpointTransport::Ndjson, "ndjson"),
    ] {
        assert_eq!(AtofEndpointTransport::parse(name), Some(transport));
        assert_eq!(transport.as_str(), name);
    }
    assert_eq!(AtofEndpointTransport::parse("invalid"), None);
}

#[test]
fn ndjson_upload_close_timeout_acknowledges_close() {
    enable_operational_logs();
    let request = tokio::runtime::Runtime::new().unwrap().block_on(async {
        let request: tokio::task::JoinHandle<reqwest::Result<reqwest::Response>> =
            tokio::spawn(async {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                unreachable!("timeout should finish before this task completes")
            });
        let (diagnostics, recorder) = test_endpoint_diagnostics(AtofEndpointTransport::Ndjson);
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        finish_ndjson_upload(
            0,
            request,
            std::time::Duration::from_millis(1),
            done_tx,
            &diagnostics,
            2,
            &std::sync::atomic::AtomicU64::new(0),
        )
        .await;
        done_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        assert_eq!(
            diagnostic_count(&recorder, "atof.stream_delivery_failed"),
            0
        );
        assert_eq!(
            diagnostic_message(&recorder, "atof.stream_close_failed"),
            "ATOF ndjson stream to http://127.0.0.1:0 did not finish its upload within 1 ms"
        );
        true
    });
    assert!(request);
}

#[test]
fn ndjson_upload_completion_reports_success_http_upload_and_task_failures() {
    enable_operational_logs();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let (diagnostics, recorder) = test_endpoint_diagnostics(AtofEndpointTransport::Ndjson);

    let ok = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    let unavailable =
        "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    for (response, queued, written) in [(ok, 3, 1), (unavailable, 0, 0), (unavailable, 3, 3)] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_http_request(&mut stream);
            stream.write_all(response.as_bytes()).unwrap();
        });
        runtime.block_on(async {
            let request = tokio::spawn(async move { reqwest::get(url).await });
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            finish_ndjson_upload(
                2,
                request,
                std::time::Duration::from_secs(5),
                done_tx,
                &diagnostics,
                queued,
                &std::sync::atomic::AtomicU64::new(written),
            )
            .await;
            done_rx
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap();
        });
        server.join().unwrap();
    }
    assert_eq!(
        diagnostic_count(&recorder, "atof.stream_delivery_failed"),
        3
    );
    assert_eq!(
        diagnostic_message(&recorder, "atof.stream_delivery_failed"),
        "ATOF ndjson stream to http://127.0.0.1:0 returned status 503 Service Unavailable"
    );
    assert_eq!(diagnostic_count(&recorder, "atof.stream_events_dropped"), 2);
    assert_eq!(
        diagnostic_message(&recorder, "atof.stream_events_dropped"),
        "ATOF ndjson stream to http://127.0.0.1:0 ended upload with events queued"
    );

    runtime.block_on(async {
        let upload_error = tokio::spawn(async { reqwest::get("http://127.0.0.1:0").await });
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        finish_ndjson_upload(
            3,
            upload_error,
            std::time::Duration::from_secs(5),
            done_tx,
            &diagnostics,
            2,
            &std::sync::atomic::AtomicU64::new(2),
        )
        .await;
        done_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        assert_eq!(
            diagnostic_count(&recorder, "atof.stream_delivery_failed"),
            5
        );

        let task_error = tokio::spawn(async {
            std::future::pending::<reqwest::Result<reqwest::Response>>().await
        });
        task_error.abort();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        finish_ndjson_upload(
            4,
            task_error,
            std::time::Duration::from_secs(5),
            done_tx,
            &diagnostics,
            1,
            &std::sync::atomic::AtomicU64::new(1),
        )
        .await;
        done_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        assert_eq!(
            diagnostic_count(&recorder, "atof.stream_delivery_failed"),
            6
        );
        assert!(
            diagnostic_message(&recorder, "atof.stream_delivery_failed")
                .contains("lost its upload task: ")
        );
    });
}

struct UnresponsiveServer {
    address: std::net::SocketAddr,
    stop: std::sync::mpsc::Sender<()>,
    thread: std::thread::JoinHandle<()>,
}

impl UnresponsiveServer {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (stop, stop_rx) = std::sync::mpsc::channel::<()>();
        let thread = std::thread::spawn(move || {
            let mut held = Vec::new();
            while matches!(
                stop_rx.try_recv(),
                Err(std::sync::mpsc::TryRecvError::Empty)
            ) {
                let (stream, _) = listener.accept().unwrap();
                held.push(stream);
            }
        });
        Self {
            address,
            stop,
            thread,
        }
    }

    fn stop(self) {
        self.stop.send(()).unwrap();
        drop(std::net::TcpStream::connect(self.address).unwrap());
        self.thread.join().unwrap();
    }
}

#[test]
fn delivered_stream_events_record_no_runtime_diagnostics() {
    enable_operational_logs();
    let (url, captures) = start_http_capture_server(2);
    let exporter = AtofExporter::new(AtofExporterConfig::new().with_stream_sink(
        AtofStreamSinkConfig::new(url, AtofEndpointTransport::HttpPost),
    ))
    .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
    .unwrap();
    assert!(exporter.runtime_diagnostics().entries().is_empty());
    let subscriber = exporter.subscriber();
    subscriber(&make_mark_event("first"));
    subscriber(&make_mark_event("second"));
    exporter.force_flush().unwrap();
    exporter.shutdown().unwrap();
    assert_eq!(wait_for_captures(&captures, 2).len(), 2);
    assert!(exporter.runtime_diagnostics().entries().is_empty());
}

#[test]
fn rejected_http_post_events_record_delivery_failures() {
    enable_operational_logs();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let url = format!("{origin}/events");
    let server = std::thread::spawn(move || {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_http_request(&mut stream);
            stream
                .write_all(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
        }
    });
    let exporter = AtofExporter::new(AtofExporterConfig::new().with_stream_sink(
        AtofStreamSinkConfig::new(url, AtofEndpointTransport::HttpPost).with_timeout_millis(5_000),
    ))
    .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
    .unwrap();
    let subscriber = exporter.subscriber();
    subscriber(&make_mark_event("first"));
    subscriber(&make_mark_event("second"));
    exporter.force_flush().unwrap();

    let diagnostics = exporter.runtime_diagnostics();
    let failed = diagnostics.get("atof.stream_delivery_failed").unwrap();
    assert_eq!(failed.count, 2);
    assert_eq!(
        failed.message,
        format!("ATOF http_post stream to {origin} returned status 503 Service Unavailable")
    );
    assert_eq!(diagnostics.entries().len(), 1);
    exporter.shutdown().unwrap();
    server.join().unwrap();
}

#[test]
fn unanswered_http_post_requests_record_flush_and_close_timeouts() {
    enable_operational_logs();
    let server = UnresponsiveServer::start();
    let address = server.address;
    let exporter = AtofExporter::new(
        AtofExporterConfig::new().with_stream_sink(
            AtofStreamSinkConfig::new(
                format!("http://{address}/events"),
                AtofEndpointTransport::HttpPost,
            )
            .with_timeout_millis(250),
        ),
    )
    .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
    .unwrap();
    let subscriber = exporter.subscriber();
    for _ in 0..3 {
        subscriber(&make_mark_event("held"));
    }

    exporter.force_flush().unwrap();
    assert_eq!(
        exporter
            .runtime_diagnostics()
            .get("atof.stream_flush_failed")
            .unwrap()
            .message,
        format!(
            "ATOF http_post stream to http://{address} did not acknowledge a flush within 250 ms"
        )
    );

    exporter.shutdown().unwrap();
    let closed = exporter.runtime_diagnostics();
    assert_eq!(closed.get("atof.stream_flush_failed").unwrap().count, 1);
    assert_eq!(
        closed.get("atof.stream_close_failed").unwrap().message,
        format!(
            "ATOF http_post stream to http://{address} did not acknowledge close within 250 ms"
        )
    );
    server.stop();
}

#[test]
fn websocket_close_without_a_handshake_drops_pending_events() {
    enable_operational_logs();
    let server = UnresponsiveServer::start();
    let address = server.address;
    let endpoint = validate_endpoint_config(
        AtofEndpointConfig::new(format!("ws://{address}"), AtofEndpointTransport::Websocket)
            .with_timeout_millis(200),
    )
    .unwrap();
    let recorder = ExporterRuntimeDiagnosticsRecorder::new(None)
        .with_host(crate::test_support::detached_host());
    let diagnostics = EndpointDiagnostics {
        recorder: recorder.clone(),
        transport: AtofEndpointTransport::Websocket,
        origin: endpoint.origin.clone(),
    };
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let worker = std::thread::spawn(move || run_endpoint_worker(0, endpoint, rx, diagnostics));
    tx.send(EndpointMessage::Event("{\"kind\":\"first\"}".into()))
        .unwrap();
    tx.send(EndpointMessage::Event("{\"kind\":\"second\"}".into()))
        .unwrap();
    let (close_tx, _close_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Close(close_tx)).unwrap();
    worker.join().unwrap();

    let snapshot = recorder.snapshot();
    let dropped = snapshot.get("atof.stream_events_dropped").unwrap();
    assert_eq!(dropped.count, 2);
    assert_eq!(
        dropped.message,
        format!("ATOF websocket stream to ws://{address} closed with undelivered events")
    );
    assert_eq!(
        snapshot.get("atof.stream_connection_failed").unwrap().count,
        4
    );
    server.stop();
}
