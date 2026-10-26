// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use axum::response::IntoResponse;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test]
async fn truncated_hook_error_bodies_preserve_status_and_failure_policy() {
    for status in [403, 404] {
        for fail_closed in [false, true] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = [0; 4096];
                let count = socket.read(&mut request).await.unwrap();
                assert!(count > 0, "hook request arrived before responding");
                socket.write_all(format!(
                    "HTTP/1.1 {status} Error\r\nContent-Length: 100\r\nConnection: close\r\n\r\npartial"
                ).as_bytes()).await.unwrap();
            });
            let response = reqwest::Client::builder()
                .no_proxy()
                .build()
                .unwrap()
                .get(format!("http://{address}"))
                .send()
                .await;
            let result = handle_hook_forward_response_with_context(
                response,
                fail_closed,
                &OperationalContext::new(),
            )
            .await;
            if fail_closed {
                let error = result.unwrap_err();
                assert_eq!(error.log_kind(), "hook_http");
                let response = error.into_response();
                assert_eq!(response.status().as_u16(), status);
                let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                let body: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(body["error"]["type"], "nemo_relay_gateway_error");
                assert!(
                    body["error"]["message"]
                        .as_str()
                        .unwrap()
                        .starts_with("nemo-relay hook forward failed: hook gateway returned HTTP")
                );
            } else {
                assert_eq!(result.unwrap(), HookDeliveryOutcome::FailedOpen);
            }
            server.await.unwrap();
        }
    }
}
