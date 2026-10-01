// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use super::automatic_otlp_http_client;

#[test]
fn automatic_http_client_does_not_follow_redirects() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let destination = TcpListener::bind("127.0.0.1:0").unwrap();
    destination.set_nonblocking(true).unwrap();
    let location = format!("http://{}/leak", destination.local_addr().unwrap());
    let redirector = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", redirector.local_addr().unwrap());
    let server = thread::spawn(move || {
        let (mut stream, _) = redirector.accept().unwrap();
        let mut buffer = [0; 4_096];
        assert!(stream.read(&mut buffer).unwrap() > 0);
        write!(
                stream,
                "HTTP/1.1 307 Redirect\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            .unwrap();
    });

    let client = automatic_otlp_http_client().unwrap();
    let response = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(client.get(endpoint).send())
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::TEMPORARY_REDIRECT);
    server.join().unwrap();
    assert!(
        matches!(destination.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock),
        "redirect destination must not receive a connection"
    );
}
