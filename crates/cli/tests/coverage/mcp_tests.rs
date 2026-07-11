// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};
use std::ffi::OsString;
use std::io::{BufReader as StdBufReader, Cursor};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

use super::*;
use crate::install_generation::{
    GENERATION_FILE_NAME, GenerationRetirement, InstallGeneration, write_new_generation,
};

struct BootstrapConfigHome {
    _guard: std::sync::MutexGuard<'static, ()>,
    previous: Option<OsString>,
}

impl BootstrapConfigHome {
    fn enter(path: &std::path::Path) -> Self {
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = std::env::var_os("XDG_CONFIG_HOME");
        unsafe { std::env::set_var("XDG_CONFIG_HOME", path) };
        Self {
            _guard: guard,
            previous,
        }
    }
}

impl Drop for BootstrapConfigHome {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            unsafe { std::env::set_var("XDG_CONFIG_HOME", previous) };
        } else {
            unsafe { std::env::remove_var("XDG_CONFIG_HOME") };
        }
    }
}

#[test]
fn bounded_mcp_reader_accepts_the_limit_and_preserves_following_frames() {
    let mut input = vec![b'a'; MAX_MCP_FRAME_BYTES - 1];
    input.push(b'\n');
    input.extend_from_slice(b"{}\n");
    let mut reader = StdBufReader::new(Cursor::new(input));
    let mut frame = Vec::new();

    assert_eq!(
        read_bounded_frame(&mut reader, &mut frame, MAX_MCP_FRAME_BYTES).unwrap(),
        MAX_MCP_FRAME_BYTES
    );
    assert_eq!(frame.last(), Some(&b'\n'));
    frame.clear();
    assert_eq!(
        read_bounded_frame(&mut reader, &mut frame, MAX_MCP_FRAME_BYTES).unwrap(),
        3
    );
    assert_eq!(frame, b"{}\n");
}

#[test]
fn bounded_mcp_reader_rejects_one_oversized_unterminated_frame() {
    let input = vec![b'a'; MAX_MCP_FRAME_BYTES + 1];
    let mut reader = StdBufReader::new(Cursor::new(input));
    let mut frame = Vec::new();

    let error = read_bounded_frame(&mut reader, &mut frame, MAX_MCP_FRAME_BYTES).unwrap_err();

    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert!(error.to_string().contains("MCP frame exceeds"));
}

#[test]
fn initialize_reports_native_server_and_supported_protocol() {
    let response = response_for(&json!({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "initialize",
        "params": { "protocolVersion": "2024-11-05" }
    }))
    .unwrap();

    assert_eq!(response["jsonrpc"], json!("2.0"));
    assert_eq!(response["id"], json!(7));
    assert_eq!(
        response["result"]["protocolVersion"],
        json!(MCP_PROTOCOL_VERSION)
    );
    assert_eq!(response["result"]["capabilities"], json!({}));
    assert_eq!(
        response["result"]["serverInfo"]["name"],
        json!("nemo-relay")
    );
    assert_eq!(
        response["result"]["serverInfo"]["version"],
        json!(env!("CARGO_PKG_VERSION"))
    );
}

#[test]
fn supported_requests_and_notifications_have_minimal_mcp_behavior() {
    assert_eq!(
        response_for(&json!({"jsonrpc":"2.0", "id":"tools", "method":"tools/list"})),
        Some(json!({"jsonrpc":"2.0", "id":"tools", "result":{"tools":[]}}))
    );
    assert_eq!(
        response_for(&json!({"jsonrpc":"2.0", "id":2, "method":"ping"})),
        Some(json!({"jsonrpc":"2.0", "id":2, "result":{}}))
    );
    assert_eq!(
        response_for(&json!({
            "jsonrpc":"2.0",
            "method":"notifications/initialized"
        })),
        None
    );
}

#[test]
fn invalid_and_unknown_requests_return_jsonrpc_errors() {
    assert_eq!(
        response_for(&json!({"jsonrpc":"2.0", "id":3})),
        Some(jsonrpc_error(json!(3), -32600, "Invalid Request"))
    );
    assert_eq!(
        response_for(&json!({"jsonrpc":"2.0", "id":4, "method":"resources/list"})),
        Some(jsonrpc_error(json!(4), -32601, "Method not found"))
    );
    assert_eq!(
        response_for(&json!({"jsonrpc":"2.0", "id":5, "method":"initialize", "params":{}})),
        Some(jsonrpc_error(json!(5), -32602, "Missing protocolVersion"))
    );
}

#[test]
fn invalid_jsonrpc_returns_invalid_request() {
    let action = crate::mcp::protocol::evaluate_frame(
        r#"{"jsonrpc":"1.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}"#,
    );

    assert_eq!(
        action.response,
        Some(jsonrpc_error(json!(1), -32600, "Invalid Request"))
    );
}

#[tokio::test]
async fn stdio_loop_recovers_from_parse_errors_and_ignores_notifications() {
    let (mut client, server) = tokio::io::duplex(4096);
    let (server_reader, server_writer) = tokio::io::split(server);
    let task = tokio::spawn(serve_stdio(BufReader::new(server_reader), server_writer));

    client
        .write_all(
            b"not-json\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"ping\"}\n",
        )
        .await
        .unwrap();
    client.shutdown().await.unwrap();
    let mut output = String::new();
    client.read_to_string(&mut output).await.unwrap();
    task.await.unwrap().unwrap();

    let responses = output
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(responses.len(), 2);
    assert_eq!(
        responses[0],
        jsonrpc_error(Value::Null, -32700, "Parse error")
    );
    assert_eq!(responses[1], json!({"jsonrpc":"2.0", "id":5, "result":{}}));
}

#[tokio::test]
async fn mcp_session_serves_stdio_and_stops_heartbeat_on_eof() {
    let (client, server_io) = tokio::io::duplex(4096);
    let (client_reader, mut client_writer) = tokio::io::split(client);
    let (server_reader, server_writer) = tokio::io::split(server_io);
    let task = tokio::spawn(run_session(
        "127.0.0.1:9".parse().unwrap(),
        "http://127.0.0.1:9".into(),
        Vec::new(),
        "test-fingerprint".into(),
        Duration::from_secs(60),
        BufReader::new(server_reader),
        server_writer,
    ));

    client_writer
        .write_all(
            b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\"}}\n",
        )
        .await
        .unwrap();
    let mut client_reader = BufReader::new(client_reader);
    let mut response = String::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        client_reader.read_line(&mut response),
    )
    .await
    .expect("MCP initialization response timed out")
    .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&response).unwrap()["result"]["serverInfo"]["name"],
        json!("nemo-relay")
    );

    client_writer.shutdown().await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), task)
        .await
        .expect("MCP session did not stop after stdin EOF")
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn heartbeat_keeps_a_compatible_gateway_session_alive() {
    let _plugin_guard = crate::test_support::PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let _bootstrap_home = BootstrapConfigHome::enter(&temp.path().join("xdg"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bind = listener.local_addr().unwrap();
    let config = crate::config::GatewayConfig {
        bind,
        ..crate::config::GatewayConfig::default()
    };
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let fingerprint = "test-fingerprint";
    let gateway = tokio::spawn(crate::server::serve_listener_with_bootstrap(
        listener,
        config,
        fingerprint.into(),
        Some(shutdown_rx),
    ));
    let url = format!("http://{bind}");
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let probe_url = url.clone();
            if tokio::task::spawn_blocking(move || {
                crate::sidecar::healthz_compatible(&probe_url, fingerprint)
            })
            .await
            .unwrap()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("compatible gateway did not become healthy");

    let health_calls = Arc::new(AtomicUsize::new(0));
    let restart_calls = Arc::new(AtomicUsize::new(0));
    let (observed_tx, mut observed_rx) = tokio::sync::mpsc::unbounded_channel();
    let heartbeat = tokio::spawn(maintain_gateway_with(
        bind,
        url,
        Duration::from_millis(10),
        {
            let health_calls = health_calls.clone();
            move |url| {
                let health_calls = health_calls.clone();
                let observed_tx = observed_tx.clone();
                async move {
                    let healthy = tokio::task::spawn_blocking(move || {
                        crate::sidecar::healthz_compatible(&url, fingerprint)
                    })
                    .await
                    .map_err(|error| {
                        CliError::Launch(format!("gateway heartbeat task failed: {error}"))
                    })?;
                    if healthy {
                        let call = health_calls.fetch_add(1, Ordering::SeqCst) + 1;
                        if call == 3 {
                            let _ = observed_tx.send(());
                        }
                    }
                    Ok(healthy)
                }
            }
        },
        {
            let restart_calls = restart_calls.clone();
            move |address, _expected_instance| {
                restart_calls.fetch_add(1, Ordering::SeqCst);
                async move {
                    Ok(crate::sidecar::GatewayEndpoint {
                        address,
                        url: "http://unexpected-restart".into(),
                        instance_id: "unexpected".into(),
                    })
                }
            }
        },
    ));

    tokio::time::timeout(Duration::from_secs(5), observed_rx.recv())
        .await
        .expect("heartbeat did not complete three compatible health checks")
        .expect("heartbeat stopped before completing three compatible health checks");
    assert!(!heartbeat.is_finished());
    assert!(health_calls.load(Ordering::SeqCst) >= 3);
    assert_eq!(restart_calls.load(Ordering::SeqCst), 0);
    heartbeat.abort();
    assert!(heartbeat.await.unwrap_err().is_cancelled());

    let _ = shutdown_tx.send(());
    tokio::time::timeout(Duration::from_secs(5), gateway)
        .await
        .expect("compatible gateway did not stop")
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn heartbeat_performs_one_restart_and_tracks_the_recovered_gateway() {
    let bind = "127.0.0.1:47632".parse().unwrap();
    let restart_calls = Arc::new(AtomicUsize::new(0));
    let observed_urls = Arc::new(Mutex::new(Vec::new()));
    let (recovered_tx, mut recovered_rx) = tokio::sync::mpsc::unbounded_channel();
    let heartbeat = tokio::spawn(maintain_gateway_with(
        bind,
        "http://dead-gateway".into(),
        Duration::from_millis(1),
        {
            let observed_urls = observed_urls.clone();
            move |url| {
                let observed_urls = observed_urls.clone();
                let recovered_tx = recovered_tx.clone();
                async move {
                    let recovered = url == "http://recovered-gateway";
                    observed_urls.lock().unwrap().push(url);
                    if recovered {
                        let _ = recovered_tx.send(());
                    }
                    Ok(recovered)
                }
            }
        },
        {
            let restart_calls = restart_calls.clone();
            move |address, _expected_instance| {
                let restart_calls = restart_calls.clone();
                async move {
                    restart_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(crate::sidecar::GatewayEndpoint {
                        address,
                        url: "http://recovered-gateway".into(),
                        instance_id: "recovered".into(),
                    })
                }
            }
        },
    ));

    tokio::time::timeout(Duration::from_secs(5), recovered_rx.recv())
        .await
        .expect("heartbeat did not observe the recovered gateway")
        .expect("heartbeat stopped before observing the recovered gateway");
    assert!(!heartbeat.is_finished());
    assert_eq!(restart_calls.load(Ordering::SeqCst), 1);
    assert!(
        observed_urls
            .lock()
            .unwrap()
            .iter()
            .any(|url| url == "http://recovered-gateway")
    );
    heartbeat.abort();
    assert!(heartbeat.await.unwrap_err().is_cancelled());
}

#[tokio::test]
async fn heartbeat_ignores_isolated_transient_health_failures() {
    let health_calls = Arc::new(AtomicUsize::new(0));
    let restart_calls = Arc::new(AtomicUsize::new(0));
    let (observed_tx, mut observed_rx) = tokio::sync::mpsc::unbounded_channel();
    let heartbeat = tokio::spawn(maintain_gateway_with(
        "127.0.0.1:47632".parse().unwrap(),
        "http://gateway".into(),
        Duration::from_millis(1),
        {
            let health_calls = health_calls.clone();
            move |_url| {
                let observed_tx = observed_tx.clone();
                let call = health_calls.fetch_add(1, Ordering::SeqCst) + 1;
                async move {
                    if call == 9 {
                        let _ = observed_tx.send(());
                    }
                    Ok(call.is_multiple_of(3))
                }
            }
        },
        {
            let restart_calls = restart_calls.clone();
            move |address, _expected_instance| {
                restart_calls.fetch_add(1, Ordering::SeqCst);
                async move {
                    Ok(crate::sidecar::GatewayEndpoint {
                        address,
                        url: "http://gateway".into(),
                        instance_id: "gateway".into(),
                    })
                }
            }
        },
    ));

    tokio::time::timeout(Duration::from_secs(5), observed_rx.recv())
        .await
        .expect("heartbeat did not complete three transient-failure cycles")
        .expect("heartbeat stopped before completing three transient-failure cycles");
    assert!(!heartbeat.is_finished());
    assert!(health_calls.load(Ordering::SeqCst) >= 9);
    assert_eq!(restart_calls.load(Ordering::SeqCst), 0);
    heartbeat.abort();
    assert!(heartbeat.await.unwrap_err().is_cancelled());
}

#[tokio::test]
async fn heartbeat_rediscovery_consumes_the_shared_restart_allowance() {
    let restart_calls = Arc::new(AtomicUsize::new(0));
    let error = maintain_gateway_with(
        "127.0.0.1:47632".parse().unwrap(),
        "http://gateway".into(),
        Duration::from_millis(1),
        |_url| async { Ok(false) },
        {
            let restart_calls = restart_calls.clone();
            move |address, _expected_instance| {
                let restart_calls = restart_calls.clone();
                async move {
                    let attempt = restart_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(crate::sidecar::GatewayEndpoint {
                        address,
                        url: "http://gateway".into(),
                        instance_id: format!("gateway-{attempt}"),
                    })
                }
            }
        },
    )
    .await
    .unwrap_err();

    assert_eq!(restart_calls.load(Ordering::SeqCst), 1);
    assert!(error.to_string().contains("after its coordinated restart"));
}

#[tokio::test]
async fn heartbeat_exits_with_the_restart_failure() {
    let error = maintain_gateway_with(
        "127.0.0.1:47632".parse().unwrap(),
        "http://dead-gateway".into(),
        Duration::from_millis(1),
        |_url| async { Ok(false) },
        |_bind, _expected_instance| async {
            Err(CliError::Launch("coordinated restart failed".into()))
        },
    )
    .await
    .unwrap_err();

    assert!(error.to_string().contains("coordinated restart failed"));
}

#[tokio::test]
async fn heartbeat_attempts_at_most_one_successful_restart() {
    let restart_calls = Arc::new(AtomicUsize::new(0));
    let error = maintain_gateway_with(
        "127.0.0.1:47632".parse().unwrap(),
        "http://dead-gateway".into(),
        Duration::from_millis(1),
        |_url| async { Ok(false) },
        {
            let restart_calls = restart_calls.clone();
            move |address, _expected_instance| {
                let restart_calls = restart_calls.clone();
                async move {
                    restart_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(crate::sidecar::GatewayEndpoint {
                        address,
                        url: "http://still-unhealthy".into(),
                        instance_id: "still-unhealthy".into(),
                    })
                }
            }
        },
    )
    .await
    .unwrap_err();

    assert_eq!(restart_calls.load(Ordering::SeqCst), 1);
    assert!(error.to_string().contains("after its coordinated restart"));
}

#[tokio::test]
async fn old_mcp_maintenance_loop_exits_when_install_generation_is_replaced() {
    let dir = tempfile::tempdir().unwrap();
    let generation_path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&generation_path).unwrap();
    let generation = InstallGeneration::capture(generation_path.clone()).unwrap();
    let health_calls = Arc::new(AtomicUsize::new(0));
    let restart_calls = Arc::new(AtomicUsize::new(0));
    let (observed_tx, observed_rx) = tokio::sync::oneshot::channel();
    let observed_tx = Arc::new(Mutex::new(Some(observed_tx)));
    let heartbeat = tokio::spawn(maintain_gateway_with_generation(
        "127.0.0.1:47632".parse().unwrap(),
        "http://old-gateway".into(),
        Duration::from_millis(100),
        {
            let health_calls = health_calls.clone();
            move |_url| {
                health_calls.fetch_add(1, Ordering::SeqCst);
                let observed_tx = observed_tx.clone();
                async move {
                    if let Some(sender) = observed_tx.lock().unwrap().take() {
                        let _ = sender.send(());
                    }
                    Ok(false)
                }
            }
        },
        {
            let restart_calls = restart_calls.clone();
            move |address, _expected_instance| {
                restart_calls.fetch_add(1, Ordering::SeqCst);
                async move {
                    Ok(crate::sidecar::GatewayEndpoint {
                        address,
                        url: "http://unexpected-restart".into(),
                        instance_id: "unexpected".into(),
                    })
                }
            }
        },
        move || {
            let generation = generation.clone();
            async move { generation.verify_current().map_err(CliError::Launch) }
        },
    ));

    tokio::time::timeout(Duration::from_secs(5), observed_rx)
        .await
        .expect("old MCP maintenance loop did not perform its first health check")
        .expect("old MCP maintenance loop stopped before its first health check");
    let mut retirement = GenerationRetirement::acquire(&generation_path)
        .unwrap()
        .expect("installed generation should be retired");
    retirement.invalidate_for_replacement().unwrap();
    std::fs::rename(&generation_path, dir.path().join("retired-generation")).unwrap();
    write_new_generation(&generation_path).unwrap();
    drop(retirement);

    let error = tokio::time::timeout(Duration::from_secs(5), heartbeat)
        .await
        .expect("old MCP maintenance loop did not observe generation replacement")
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("has been retired"));
    assert_eq!(health_calls.load(Ordering::SeqCst), 1);
    assert_eq!(restart_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn invalidated_install_generation_can_be_restored_before_rollback_registration() {
    let dir = tempfile::tempdir().unwrap();
    let generation_path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&generation_path).unwrap();
    let original = InstallGeneration::capture(generation_path.clone()).unwrap();
    let mut retirement = GenerationRetirement::acquire(&generation_path)
        .unwrap()
        .expect("installed generation should be retired");

    retirement.invalidate_for_replacement().unwrap();
    let error = InstallGeneration::capture(generation_path.clone()).unwrap_err();
    assert!(error.contains("has been retired"), "{error}");

    retirement.restore_after_rollback().unwrap();
    original.verify_current().unwrap();
    InstallGeneration::capture(generation_path).unwrap();
}

#[test]
fn retired_install_generation_remains_retryable_but_not_adoptable() {
    let dir = tempfile::tempdir().unwrap();
    let generation_path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&generation_path).unwrap();
    let mut retirement = GenerationRetirement::acquire(&generation_path)
        .unwrap()
        .expect("installed generation should be retired");
    retirement.invalidate_for_replacement().unwrap();
    drop(retirement);

    let mut resumed = GenerationRetirement::acquire(&generation_path)
        .unwrap()
        .expect("a retired generation should support cleanup retry");
    resumed.invalidate_for_replacement().unwrap();
    drop(resumed);

    let error = InstallGeneration::capture(generation_path).unwrap_err();
    assert!(error.contains("has been retired"), "{error}");
}

#[test]
fn default_mcp_gateway_uses_plugin_provider_port() {
    assert_eq!(default_mcp_bind().to_string(), "127.0.0.1:47632");
}
