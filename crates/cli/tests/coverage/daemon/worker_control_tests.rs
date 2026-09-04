// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use std::sync::{Arc, Mutex};

use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[test]
fn daemon_heartbeat_interval_is_bounded() {
    assert!(validate_heartbeat_interval(99).is_err());
    assert_eq!(
        validate_heartbeat_interval(5_000).expect("normal interval"),
        Duration::from_secs(5)
    );
    assert!(validate_heartbeat_interval(20_001).is_err());
}

#[tokio::test]
async fn readiness_retries_the_exact_request_after_a_lost_response() {
    let (origin, received) = lost_response_server().await;
    let mut registration = test_registration("data", "session");

    registration
        .ready(&origin, "worker-one")
        .await
        .expect("readiness retry");

    assert_exact_retry(received);
    assert_eq!(registration.next_sequence, 2);
    assert!(registration.pending_ready.is_none());
}

#[tokio::test]
async fn heartbeat_retries_the_exact_request_after_a_lost_response() {
    let (origin, received) = lost_response_server().await;
    let mut registration = test_registration("data", "session");

    registration
        .heartbeat(&origin, "worker-one")
        .await
        .expect("heartbeat retry");

    assert_exact_retry(received);
    assert_eq!(registration.next_sequence, 2);
    assert!(registration.pending_heartbeat.is_none());
}

fn assert_exact_retry(received: Arc<Mutex<Vec<Bytes>>>) {
    let received = received.lock().expect("received request bodies");
    assert_eq!(received.len(), 2);
    assert_eq!(received[0], received[1]);
}

async fn lost_response_server() -> (String, Arc<Mutex<Vec<Bytes>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("local address");
    let received = Arc::new(Mutex::new(Vec::new()));
    let server_received = Arc::clone(&received);
    tokio::spawn(async move {
        for attempt in 0..2 {
            let (mut stream, _) = listener.accept().await.expect("accept request");
            let body = read_http_body(&mut stream).await;
            server_received
                .lock()
                .expect("received request bodies")
                .push(body);
            if attempt == 0 {
                // The daemon applied the request but its response was lost. Closing the socket
                // makes the client retry the same authenticated envelope on a new connection.
                continue;
            }
            stream
                .write_all(b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n")
                .await
                .expect("write response");
        }
    });
    (format!("http://{address}"), received)
}

async fn read_http_body(stream: &mut TcpStream) -> Bytes {
    let mut request = Vec::new();
    let (body_offset, content_length) = loop {
        let mut chunk = [0_u8; 1024];
        let count = stream.read(&mut chunk).await.expect("read request");
        assert_ne!(count, 0, "request ended before its headers");
        request.extend_from_slice(&chunk[..count]);
        if let Some(offset) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            let body_offset = offset + 4;
            let headers = std::str::from_utf8(&request[..offset]).expect("HTTP headers");
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().expect("content length"))
                })
                .expect("content length header");
            break (body_offset, content_length);
        }
    };
    while request.len() < body_offset + content_length {
        let mut chunk = [0_u8; 1024];
        let count = stream.read(&mut chunk).await.expect("read body");
        assert_ne!(count, 0, "request ended before its body");
        request.extend_from_slice(&chunk[..count]);
    }
    Bytes::copy_from_slice(&request[body_offset..body_offset + content_length])
}
