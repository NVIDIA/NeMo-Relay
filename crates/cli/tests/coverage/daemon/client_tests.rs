// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use super::*;

#[tokio::test]
#[allow(
    clippy::result_large_err,
    reason = "Tungstenite fixes the upgrade callback error type"
)]
async fn secure_control_connects_without_a_preinstalled_crypto_provider() {
    const ORIGIN_ENV: &str = "NEMO_RELAY_TEST_FRESH_WSS_ORIGIN";
    if let Ok(origin) = std::env::var(ORIGIN_ENV) {
        assert!(rustls::crypto::CryptoProvider::get_default().is_none());
        for role in [ComponentRole::Mcp, ComponentRole::Worker] {
            Client::default().connect(&origin, role).await.unwrap();
        }
        return;
    }

    let certificate = rcgen::generate_simple_self_signed(vec!["127.0.0.1".into()]).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let certificate_path = temp.path().join("daemon.pem");
    std::fs::write(&certificate_path, certificate.cert.pem()).unwrap();
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(
        vec![certificate.cert.der().clone()],
        rustls::pki_types::PrivateKeyDer::Pkcs8(certificate.key_pair.serialize_der().into()),
    )
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("https://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
        for path in [MCP_SOCKET_PATH, WORKER_SOCKET_PATH] {
            let (stream, _) = listener.accept().await.unwrap();
            let stream = acceptor.accept(stream).await.unwrap();
            tokio_tungstenite::accept_hdr_async(
                stream,
                |request: &tokio_tungstenite::tungstenite::handshake::server::Request, response| {
                    assert_eq!(request.uri().path(), path);
                    Ok(response)
                },
            )
            .await
            .unwrap();
        }
    });
    // Isolate the process-wide provider from other tests and trust only this fixture's CA.
    let output = tokio::time::timeout(
        Duration::from_secs(20),
        tokio::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "daemon::common::socket::tests::secure_control_connects_without_a_preinstalled_crypto_provider",
                "--nocapture",
            ])
            .env(ORIGIN_ENV, origin)
            .env("SSL_CERT_FILE", certificate_path)
            .env("SSL_CERT_DIR", temp.path())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .expect("fresh WSS client timed out")
    .unwrap();
    assert!(
        output.status.success(),
        "fresh WSS client failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .expect("both control roles must complete their WSS upgrade")
        .unwrap();
}

#[tokio::test(start_paused = true)]
async fn reconnect_attempts_are_bounded_by_one_monotonic_grace_window() {
    let started = tokio::time::Instant::now();
    let attempts = std::sync::atomic::AtomicUsize::new(0);
    let result: Result<(), CliError> = retry(|| {
        attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        async { Err(failure("unavailable")) }
    })
    .await;
    assert!(result.is_err());
    assert_eq!(started.elapsed(), GRACE);
    assert!((5..25).contains(&attempts.load(std::sync::atomic::Ordering::Relaxed)));
}

#[tokio::test(start_paused = true)]
async fn credential_rejection_is_not_retried() {
    assert!(!is_retryable(&CliError::RouteCredentialRejected(
        "rejected".into()
    )));
    for transient in [
        failure("daemon WebSocket connection failed"),
        failure("control operation timed out"),
        CliError::Unauthorized("unknown, expired, or replayed challenge".into()),
    ] {
        assert!(is_retryable(&transient));
    }
    let started = tokio::time::Instant::now();
    let attempts = std::sync::atomic::AtomicUsize::new(0);
    let result: Result<(), CliError> = retry(|| {
        attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        async { Err(CliError::RouteCredentialRejected("rejected".into())) }
    })
    .await;
    assert!(matches!(result, Err(CliError::RouteCredentialRejected(_))));
    assert_eq!(attempts.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(started.elapsed(), Duration::ZERO);
}

#[tokio::test]
async fn only_coded_unauthorized_replies_are_credential_rejections() {
    let (client, mut socket) = connected_control_pair().await;
    for (status, error, coded) in [
        (
            401,
            serde_json::json!({ "message": "bound elsewhere", "code": ROUTE_CREDENTIAL_REJECTED_CODE }),
            true,
        ),
        (
            401,
            serde_json::json!({ "message": "expired challenge" }),
            false,
        ),
        (
            503,
            serde_json::json!({ "message": "restarting", "code": ROUTE_CREDENTIAL_REJECTED_CODE }),
            false,
        ),
    ] {
        let request = client.acknowledge("reply".into());
        let server = async {
            let request = receive_control_request(&mut socket).await;
            send_control_event(
                &mut socket,
                Event::Reply {
                    request_id: request.request_id,
                    status,
                    payload: serde_json::json!({ "error": error }),
                },
            )
            .await;
        };
        let (result, ()) = tokio::join!(request, server);
        let error = result.unwrap_err();
        assert_eq!(matches!(error, CliError::RouteCredentialRejected(_)), coded);
        assert_eq!(is_retryable(&error), !coded);
    }
}

#[tokio::test(start_paused = true)]
async fn a_hung_reconnect_attempt_does_not_extend_grace() {
    let started = tokio::time::Instant::now();
    let result: Result<(), CliError> = retry(std::future::pending).await;
    assert!(result.is_err());
    assert_eq!(started.elapsed(), GRACE);
}

#[tokio::test]
async fn remote_cleartext_is_rejected_before_connecting() {
    for origin in ["http://192.0.2.1:80", "http://daemon.example:80"] {
        let error = Client::default()
            .connect(origin, ComponentRole::Mcp)
            .await
            .unwrap_err();
        assert!(
            matches!(error, CliError::Config(ref message) if message == "non-loopback daemon addresses must use https")
        );
    }
}

#[tokio::test(start_paused = true)]
async fn delayed_recovery_uses_only_the_remaining_grace() {
    let deadline = tokio::time::Instant::now() + GRACE;
    tokio::time::advance(Duration::from_secs(12)).await;
    let started = tokio::time::Instant::now();
    let result: Result<(), CliError> = retry_until(deadline, std::future::pending).await;
    assert!(result.is_err());
    assert_eq!(started.elapsed(), Duration::from_secs(18));
    let result: Result<(), CliError> = retry_until(deadline, || async {
        panic!("expired recovery must not start an attempt")
    })
    .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn queued_disconnect_keeps_its_original_recovery_deadline() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        socket.send(Message::Close(None)).await.unwrap();
    });
    let client = Client::default();
    client.connect(&origin, ComponentRole::Mcp).await.unwrap();
    server.await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if client.0.lock().await.as_ref().unwrap().task.is_finished() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let deadline = client.recovery_deadline().await;
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(12)).await;
    assert!(client.next().await.is_err());
    assert_eq!(client.recovery_deadline().await, deadline);
}

async fn connected_control_pair() -> (
    Client,
    tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let client = Client::default();
    let (connected, socket) = tokio::join!(client.connect(&origin, ComponentRole::Mcp), async {
        let (stream, _) = listener.accept().await.unwrap();
        tokio_tungstenite::accept_async(stream).await.unwrap()
    });
    connected.unwrap();
    (client, socket)
}

async fn receive_control_request(
    socket: &mut tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
) -> Request {
    let message = socket.next().await.unwrap().unwrap();
    serde_json::from_str(message.to_text().unwrap()).unwrap()
}

async fn send_control_event(
    socket: &mut tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    event: Event,
) {
    socket
        .send(Message::Text(serde_json::to_string(&event).unwrap().into()))
        .await
        .unwrap();
}

#[tokio::test]
async fn idle_event_receiver_does_not_block_concurrent_requests_or_reorder_replies() {
    let (client, mut socket) = connected_control_pair().await;
    let mut next = std::pin::pin!(client.next());
    assert!(futures_util::poll!(&mut next).is_pending());
    let first = client.request::<String>(Command::Acknowledge {
        request_id: "first".into(),
    });
    let second = client.request::<String>(Command::Acknowledge {
        request_id: "second".into(),
    });
    let server = async {
        let first = receive_control_request(&mut socket).await;
        let second = receive_control_request(&mut socket).await;
        send_control_event(
            &mut socket,
            Event::Directive {
                request_id: "directive".into(),
                directive: BrokerDirective::UsePassThrough,
            },
        )
        .await;
        // Reply in the reverse order and interleave an event with those replies.
        for request in [second, first] {
            let Command::Acknowledge { request_id: label } = request.command else {
                panic!("expected acknowledgment");
            };
            send_control_event(
                &mut socket,
                Event::Reply {
                    request_id: request.request_id,
                    status: 200,
                    payload: Value::String(label),
                },
            )
            .await;
        }
    };
    let (first, second, event, ()) = tokio::time::timeout(Duration::from_secs(2), async {
        tokio::join!(first, second, next, server)
    })
    .await
    .expect("idle event reception must not block command dispatch");
    assert_eq!(first.unwrap(), "first");
    assert_eq!(second.unwrap(), "second");
    assert!(
        matches!(event.unwrap(), Event::Directive { request_id, .. } if request_id == "directive")
    );
}

#[tokio::test]
async fn idle_event_receiver_does_not_delay_request_timeout_or_consume_late_replies() {
    let (client, mut socket) = connected_control_pair().await;
    let mut next = std::pin::pin!(client.next());
    assert!(futures_util::poll!(&mut next).is_pending());
    let mut request = std::pin::pin!(client.acknowledge("unanswered".into()));
    assert!(futures_util::poll!(&mut request).is_pending());
    let expired = receive_control_request(&mut socket).await;
    tokio::time::pause();
    tokio::time::advance(ATTEMPT_TIMEOUT).await;
    let error = request.await.unwrap_err();
    assert!(matches!(error, CliError::Launch(message) if message == "control operation timed out"));
    assert!(futures_util::poll!(&mut next).is_pending());
    tokio::time::resume();
    let server = async {
        let current = receive_control_request(&mut socket).await;
        for (request_id, payload) in [
            (expired.request_id, Value::String("late".into())),
            (current.request_id, Value::Null),
        ] {
            send_control_event(
                &mut socket,
                Event::Reply {
                    request_id,
                    status: 200,
                    payload,
                },
            )
            .await;
        }
    };
    let (result, ()) = tokio::time::timeout(Duration::from_secs(2), async {
        tokio::join!(client.acknowledge("current".into()), server)
    })
    .await
    .unwrap();
    result.unwrap();
    assert!(futures_util::poll!(&mut next).is_pending());
}

#[tokio::test]
async fn replacement_releases_old_waiters_without_consuming_new_connection_events() {
    let (client, mut socket) = connected_control_pair().await;
    let mut next = std::pin::pin!(client.next());
    assert!(futures_util::poll!(&mut next).is_pending());
    let mut request = std::pin::pin!(client.acknowledge("old".into()));
    assert!(futures_util::poll!(&mut request).is_pending());
    receive_control_request(&mut socket).await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let server = async {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        send_control_event(
            &mut socket,
            Event::Directive {
                request_id: "replacement".into(),
                directive: BrokerDirective::UsePassThrough,
            },
        )
        .await;
        socket
    };
    let (connected, _replacement_socket) = tokio::time::timeout(Duration::from_secs(2), async {
        tokio::join!(client.connect(&origin, ComponentRole::Mcp), server)
    })
    .await
    .expect("replacement must not wait for the old event receiver");
    connected.unwrap();
    let (old_request, old_event) = tokio::time::timeout(Duration::from_secs(2), async {
        tokio::join!(request, next)
    })
    .await
    .unwrap();
    assert!(old_request.is_err());
    assert!(old_event.is_err());
    let event = tokio::time::timeout(Duration::from_secs(2), client.next())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(event, Event::Directive { request_id, .. } if request_id == "replacement"));
}

#[tokio::test]
async fn pending_events_are_available_before_the_registration_reply_is_observed() {
    let (client, mut socket) = connected_control_pair().await;
    let server = async {
        let request = receive_control_request(&mut socket).await;
        send_control_event(
            &mut socket,
            Event::Directive {
                request_id: "pending".into(),
                directive: BrokerDirective::UsePassThrough,
            },
        )
        .await;
        send_control_event(
            &mut socket,
            Event::Reply {
                request_id: request.request_id,
                status: 200,
                payload: Value::Null,
            },
        )
        .await;
    };
    let (result, ()) = tokio::join!(client.acknowledge("registration".into()), server);
    result.unwrap();
    assert!(
        matches!(client.pending_event().await, Some(Event::Directive { request_id, .. }) if request_id == "pending")
    );
    assert!(client.pending_event().await.is_none());
}

#[tokio::test]
async fn cancelled_requests_do_not_exhaust_reply_capacity() {
    let (client, mut socket) = connected_control_pair().await;
    tokio::time::timeout(Duration::from_secs(2), async {
        for _ in 0..=QUEUE_CAPACITY {
            let mut request = std::pin::pin!(client.acknowledge("cancelled".into()));
            assert!(futures_util::poll!(&mut request).is_pending());
            receive_control_request(&mut socket).await;
            // Drop the caller without a response; the next request must reclaim its slot.
        }
        let server = async {
            let request = receive_control_request(&mut socket).await;
            send_control_event(
                &mut socket,
                Event::Reply {
                    request_id: request.request_id,
                    status: 200,
                    payload: Value::Null,
                },
            )
            .await;
        };
        let (result, ()) = tokio::join!(client.acknowledge("active".into()), server);
        result.unwrap();
    })
    .await
    .expect("cancelled requests must not fill the pending reply map");
}

#[tokio::test]
async fn event_overflow_closes_the_connection_and_fails_pending_requests() {
    let (client, mut socket) = connected_control_pair().await;
    let mut request = std::pin::pin!(client.acknowledge("unanswered".into()));
    assert!(futures_util::poll!(&mut request).is_pending());
    receive_control_request(&mut socket).await;
    for _ in 0..=QUEUE_CAPACITY {
        send_control_event(
            &mut socket,
            Event::Directive {
                request_id: "overflow".into(),
                directive: BrokerDirective::UsePassThrough,
            },
        )
        .await;
    }
    let error = tokio::time::timeout(Duration::from_secs(2), request)
        .await
        .unwrap()
        .unwrap_err();
    assert!(matches!(error, CliError::Launch(message) if message == "control connection lost"));
    for _ in 0..QUEUE_CAPACITY {
        assert!(client.pending_event().await.is_some());
    }
    assert!(client.pending_event().await.is_none());
    assert!(client.next().await.is_err());
}

#[tokio::test]
async fn outstanding_reply_limit_disconnects_and_releases_every_waiter() {
    let (client, mut socket) = connected_control_pair().await;
    let mut requests = Vec::new();
    for _ in 0..QUEUE_CAPACITY {
        let mut request = Box::pin(client.acknowledge("unanswered".into()));
        assert!(futures_util::poll!(&mut request).is_pending());
        receive_control_request(&mut socket).await;
        requests.push(request);
    }
    let error = tokio::time::timeout(
        Duration::from_secs(2),
        client.acknowledge("overflow".into()),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(matches!(error, CliError::Launch(message) if message == "control connection lost"));
    for request in requests {
        assert!(
            tokio::time::timeout(Duration::from_secs(2), request)
                .await
                .unwrap()
                .is_err()
        );
    }
    assert!(client.next().await.is_err());
}
