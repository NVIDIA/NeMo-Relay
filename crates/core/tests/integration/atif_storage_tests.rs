// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! S3 storage integration tests for the ATIF observability exporter.
//!
//! These tests require an S3-compatible object store reachable through the
//! standard AWS environment variables (`AWS_ACCESS_KEY_ID`,
//! `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, optionally `AWS_ENDPOINT_URL` and
//! `AWS_ALLOW_HTTP`). They only execute when `NEMO_RELAY_RUN_S3_TESTS=1` is
//! set. The destination bucket must be supplied via
//! `NEMO_RELAY_S3_TEST_BUCKET`. A test-scoped key prefix containing a UUID is
//! used so concurrent runs cannot collide; an override is available via
//! `NEMO_RELAY_S3_TEST_KEY_PREFIX` for environments that pin a prefix.

#![cfg(feature = "object-store")]

use std::time::Duration;

use nemo_relay::api::runtime::{
    NemoRelayContextState, create_scope_stack, global_context, set_thread_scope_stack,
};
use nemo_relay::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use nemo_relay::api::subscriber::flush_subscribers;
use nemo_relay::observability::plugin_component::OBSERVABILITY_PLUGIN_KIND;
use nemo_relay::plugin::{
    PluginComponentSpec, PluginConfig, clear_plugin_configuration, initialize_plugins,
};
use object_store::{ObjectStore, ObjectStoreExt as _};
use serde_json::{Value as Json, json};
use uuid::Uuid;

#[derive(Debug)]
struct CapturedHttpRequest {
    method: String,
    path: String,
    headers: std::collections::HashMap<String, String>,
    body: Vec<u8>,
}

struct TestHttpServer {
    base_url: String,
    received: std::sync::Arc<std::sync::Mutex<Vec<CapturedHttpRequest>>>,
    stop_tx: Option<std::sync::mpsc::Sender<()>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

const RUN_ENV: &str = "NEMO_RELAY_RUN_S3_TESTS";
const BUCKET_ENV: &str = "NEMO_RELAY_S3_TEST_BUCKET";
const KEY_PREFIX_ENV: &str = "NEMO_RELAY_S3_TEST_KEY_PREFIX";
const HTTP_SERVER_HARD_TIMEOUT: Duration = Duration::from_secs(10);
static PLUGIN_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

impl TestHttpServer {
    fn stop(&mut self) {
        if let Some(stop_tx) = self.stop_tx.take() {
            let _ = stop_tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            handle.join().expect("HTTP server thread should finish");
        }
    }
}

fn env_value_is_truthy(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some(value) if !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
    )
}

fn run_tests_enabled() -> bool {
    let raw = std::env::var_os(RUN_ENV).map(|value| value.to_string_lossy().into_owned());
    env_value_is_truthy(raw.as_deref())
}

fn read_bucket() -> Option<String> {
    let value = std::env::var(BUCKET_ENV).ok()?;
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
}

fn build_test_key_prefix() -> String {
    let base = std::env::var(KEY_PREFIX_ENV)
        .unwrap_or_else(|_| "nemo-relay-atif-integration/".to_string());
    let trimmed = base.trim_end_matches('/');
    let run_id = Uuid::now_v7();
    if trimmed.is_empty() {
        format!("{run_id}/")
    } else {
        format!("{trimmed}/{run_id}/")
    }
}

fn reset_runtime() {
    let _ = clear_plugin_configuration();
    let stack = create_scope_stack();
    set_thread_scope_stack(stack);
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoRelayContextState::new();
}

fn build_observability_config(bucket: &str, key_prefix: &str) -> PluginConfig {
    let Json::Object(component_config) = json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "s3",
                "bucket": bucket,
                "key_prefix": key_prefix,
            }]
        }
    }) else {
        unreachable!("config builder produced non-object root")
    };
    PluginConfig {
        version: 1,
        components: vec![PluginComponentSpec {
            kind: OBSERVABILITY_PLUGIN_KIND.to_string(),
            enabled: true,
            config: component_config,
        }],
        policy: Default::default(),
    }
}

fn build_http_observability_config(endpoints: &[String]) -> PluginConfig {
    let storage = endpoints
        .iter()
        .map(|endpoint| {
            json!({
                "type": "http",
                "endpoint": endpoint,
                "headers": {"x-static": "static-value"},
                "header_env": {"authorization": "NEMO_RELAY_ATIF_HTTP_TEST_TOKEN"},
                "timeout_millis": 3000
            })
        })
        .collect::<Vec<_>>();
    let Json::Object(component_config) = json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": storage
        }
    }) else {
        unreachable!("config builder produced non-object root")
    };
    PluginConfig {
        version: 1,
        components: vec![PluginComponentSpec {
            kind: OBSERVABILITY_PLUGIN_KIND.to_string(),
            enabled: true,
            config: component_config,
        }],
        policy: Default::default(),
    }
}

fn read_http_request(stream: &mut std::net::TcpStream) -> CapturedHttpRequest {
    use std::io::Read;

    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("set read timeout");
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 1024];
    let mut expected_len = None;
    loop {
        let read = stream.read(&mut chunk).expect("read HTTP request");
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if expected_len.is_none()
            && let Some(header_end) = buffer.windows(4).position(|window| window == b"\r\n\r\n")
        {
            let header_text = String::from_utf8_lossy(&buffer[..header_end]);
            let content_length = header_text
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            expected_len = Some(header_end + 4 + content_length);
        }
        if let Some(len) = expected_len
            && buffer.len() >= len
        {
            break;
        }
    }

    let header_end = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("request should contain headers");
    let header_text = String::from_utf8_lossy(&buffer[..header_end]);
    let mut lines = header_text.lines();
    let request_line = lines.next().expect("request line");
    let method = request_line
        .split_whitespace()
        .next()
        .expect("request method")
        .to_string();
    let path = request_line
        .split_whitespace()
        .nth(1)
        .expect("request path")
        .to_string();
    let mut headers = std::collections::HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.to_ascii_lowercase(), value.trim().to_string());
        }
    }
    CapturedHttpRequest {
        method,
        path,
        headers,
        body: buffer[header_end + 4..].to_vec(),
    }
}

fn write_http_response(stream: &mut std::net::TcpStream, status: u16) {
    use std::io::Write;

    let reason = match status {
        200 => "OK",
        204 => "No Content",
        500 => "Internal Server Error",
        _ => "OK",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )
    .expect("write HTTP response");
}

fn start_http_server(
    expected_requests: usize,
    statuses: Vec<(&'static str, u16)>,
) -> TestHttpServer {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind test HTTP server");
    listener.set_nonblocking(true).expect("set nonblocking");
    let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
    let statuses = statuses
        .into_iter()
        .map(|(path, status)| (path.to_string(), status))
        .collect::<std::collections::HashMap<_, _>>();
    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
    let received = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let thread_received = std::sync::Arc::clone(&received);
    let handle = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + HTTP_SERVER_HARD_TIMEOUT;
        loop {
            if thread_received.lock().unwrap().len() >= expected_requests {
                break;
            }
            match stop_rx.try_recv() {
                Ok(()) | Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            if std::time::Instant::now() >= deadline {
                break;
            }
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream.set_nonblocking(false).expect("set stream blocking");
                    let request = read_http_request(&mut stream);
                    let status = statuses.get(&request.path).copied().unwrap_or(204);
                    write_http_response(&mut stream, status);
                    thread_received.lock().unwrap().push(request);
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(err) => panic!("test HTTP server accept failed: {err}"),
            }
        }
    });

    TestHttpServer {
        base_url,
        received,
        stop_tx: Some(stop_tx),
        handle: Some(handle),
    }
}

fn read_object_with_retries(
    runtime: &tokio::runtime::Runtime,
    store: &dyn ObjectStore,
    key: &str,
) -> Vec<u8> {
    runtime.block_on(async {
        let path = object_store::path::Path::from(key);
        // The dispatcher uploads from a different runtime thread; allow a brief
        // grace window for S3-compatible backends with eventual consistency.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            match store.get(&path).await {
                Ok(result) => {
                    return result
                        .bytes()
                        .await
                        .expect("uploaded payload should be readable")
                        .to_vec();
                }
                Err(err) if std::time::Instant::now() < deadline => {
                    eprintln!("waiting for upload to settle: {err}");
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                Err(err) => panic!("failed to read uploaded ATIF object '{key}': {err}"),
            }
        }
    })
}

fn build_verification_store(bucket: &str) -> (tokio::runtime::Runtime, Box<dyn ObjectStore>) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("verification runtime should build");
    let store = object_store::aws::AmazonS3Builder::from_env()
        .with_bucket_name(bucket)
        .build()
        .expect("verification S3 client should build");
    (runtime, Box::new(store))
}

fn cleanup_prefix(runtime: &tokio::runtime::Runtime, store: &dyn ObjectStore, key_prefix: &str) {
    use futures::stream::StreamExt;
    runtime.block_on(async {
        let prefix_path = object_store::path::Path::from(key_prefix.trim_end_matches('/'));
        let mut listing = store.list(Some(&prefix_path));
        while let Some(entry) = listing.next().await {
            match entry {
                Ok(meta) => {
                    if let Err(err) = store.delete(&meta.location).await {
                        eprintln!("cleanup: failed to delete {}: {err}", meta.location);
                    }
                }
                Err(err) => {
                    eprintln!("cleanup: list error: {err}");
                    break;
                }
            }
        }
    });
}

#[test]
fn atif_storage_uploads_trajectory_to_s3() {
    let _guard = PLUGIN_TEST_LOCK.lock().unwrap();
    if !run_tests_enabled() {
        eprintln!(
            "SKIP: set {RUN_ENV} to a truthy value (for example, {RUN_ENV}=1) to run ATIF S3 storage tests"
        );
        return;
    }
    let Some(bucket) = read_bucket() else {
        eprintln!("SKIP: set {BUCKET_ENV} to the destination bucket for ATIF S3 storage tests");
        return;
    };

    let key_prefix = build_test_key_prefix();
    reset_runtime();

    let config = build_observability_config(&bucket, &key_prefix);
    futures::executor::block_on(initialize_plugins(config))
        .expect("observability plugin should initialize with S3 storage");

    let handle = push_scope(
        PushScopeParams::builder()
            .name("atif-storage-integration")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .expect("push agent scope");
    let session_id = handle.uuid;
    pop_scope(PopScopeParams::builder().handle_uuid(&handle.uuid).build())
        .expect("pop agent scope");

    clear_plugin_configuration().expect("plugin teardown should flush the trajectory");

    let key = format!("{key_prefix}trajectory-{session_id}.json");
    let (runtime, store) = build_verification_store(&bucket);
    let body = read_object_with_retries(&runtime, store.as_ref(), &key);
    let value: Json = serde_json::from_slice(&body).expect("uploaded payload should be JSON");
    assert_eq!(
        value["schema_version"].as_str(),
        Some("ATIF-v1.7"),
        "uploaded artifact should be an ATIF trajectory"
    );
    let expected_session_id = session_id.to_string();
    assert_eq!(
        value["session_id"].as_str(),
        Some(expected_session_id.as_str())
    );

    cleanup_prefix(&runtime, store.as_ref(), &key_prefix);
}

#[test]
fn atif_storage_posts_trajectory_to_http_endpoints() {
    let _guard = PLUGIN_TEST_LOCK.lock().unwrap();
    reset_runtime();
    let mut server = start_http_server(2, vec![("/primary", 204), ("/secondary", 204)]);
    // SAFETY: this uniquely named env var is only touched by this test.
    unsafe {
        std::env::set_var("NEMO_RELAY_ATIF_HTTP_TEST_TOKEN", "Bearer test-token");
    }

    let config = build_http_observability_config(&[
        format!("{}/primary", server.base_url),
        format!("{}/secondary", server.base_url),
    ]);
    futures::executor::block_on(initialize_plugins(config))
        .expect("observability plugin should initialize with HTTP storage");

    let handle = push_scope(
        PushScopeParams::builder()
            .name("atif-http-storage-integration")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .expect("push agent scope");
    let session_id = handle.uuid;
    pop_scope(PopScopeParams::builder().handle_uuid(&handle.uuid).build())
        .expect("pop agent scope");
    flush_subscribers().expect("HTTP upload subscriber should flush");

    clear_plugin_configuration().expect("plugin teardown should succeed after HTTP uploads");
    server.stop();
    // SAFETY: cleanup of test-only env var.
    unsafe {
        std::env::remove_var("NEMO_RELAY_ATIF_HTTP_TEST_TOKEN");
    }

    let requests = server.received.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let mut paths = requests
        .iter()
        .map(|request| request.path.as_str())
        .collect::<Vec<_>>();
    paths.sort_unstable();
    assert_eq!(paths, vec!["/primary", "/secondary"]);
    for request in requests.iter() {
        assert_eq!(request.method, "POST");
        assert_eq!(
            request.headers.get("content-type").map(String::as_str),
            Some("application/json")
        );
        assert_eq!(
            request
                .headers
                .get("x-nemo-relay-atif-filename")
                .map(String::as_str),
            Some(format!("trajectory-{session_id}.json").as_str())
        );
        assert_eq!(
            request
                .headers
                .get("x-nemo-relay-atif-session-id")
                .map(String::as_str),
            Some(session_id.to_string().as_str())
        );
        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some("Bearer test-token")
        );
        assert_eq!(
            request.headers.get("x-static").map(String::as_str),
            Some("static-value")
        );
        let value: Json = serde_json::from_slice(&request.body).expect("HTTP body should be JSON");
        assert_eq!(value["schema_version"].as_str(), Some("ATIF-v1.7"));
        assert_eq!(
            value["session_id"].as_str(),
            Some(session_id.to_string().as_str())
        );
    }
}

#[test]
fn atif_storage_http_non_2xx_marks_sink_unhealthy() {
    let _guard = PLUGIN_TEST_LOCK.lock().unwrap();
    reset_runtime();
    let mut server = start_http_server(2, vec![("/fail", 500)]);
    // SAFETY: this uniquely named env var is only touched by this test.
    unsafe {
        std::env::set_var("NEMO_RELAY_ATIF_HTTP_TEST_TOKEN", "Bearer test-token");
    }

    let config = build_http_observability_config(&[format!("{}/fail", server.base_url)]);
    futures::executor::block_on(initialize_plugins(config))
        .expect("observability plugin should initialize with HTTP storage");

    let handle = push_scope(
        PushScopeParams::builder()
            .name("atif-http-storage-failure")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .expect("push agent scope");
    pop_scope(PopScopeParams::builder().handle_uuid(&handle.uuid).build())
        .expect("pop agent scope");
    flush_subscribers().expect("HTTP upload subscriber should flush");

    let second = push_scope(
        PushScopeParams::builder()
            .name("atif-http-storage-after-failure")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .expect("push second agent scope");
    pop_scope(PopScopeParams::builder().handle_uuid(&second.uuid).build())
        .expect("pop second agent scope");
    flush_subscribers().expect("HTTP upload subscriber should flush after failure");

    server.stop();
    {
        let requests = server.received.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].method, "POST");
        assert_eq!(requests[0].path, "/fail");
    }
    clear_plugin_configuration().expect("plugin teardown should ignore unhealthy sink errors");
    // SAFETY: cleanup of test-only env var.
    unsafe {
        std::env::remove_var("NEMO_RELAY_ATIF_HTTP_TEST_TOKEN");
    }

    reset_runtime();
}

#[test]
fn s3_test_env_truthy_parsing() {
    assert!(!env_value_is_truthy(None));
    assert!(!env_value_is_truthy(Some("")));
    assert!(!env_value_is_truthy(Some("   ")));
    assert!(!env_value_is_truthy(Some("0")));
    assert!(!env_value_is_truthy(Some(" false ")));
    assert!(!env_value_is_truthy(Some("FALSE")));
    assert!(env_value_is_truthy(Some("1")));
    assert!(env_value_is_truthy(Some("true")));
    assert!(env_value_is_truthy(Some("yes")));
}
