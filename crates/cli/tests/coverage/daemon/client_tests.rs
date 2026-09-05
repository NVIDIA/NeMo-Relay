// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use bytes::Bytes;
use serde_json::{Value, json};
use tokio::net::TcpListener;

#[test]
fn control_client_has_a_bounded_configuration() {
    control_client().expect("control client");
}

#[tokio::test]
async fn rejects_an_oversized_control_response_without_collecting_it() {
    async fn oversized() -> Response {
        let chunks = futures_util::stream::iter([
            Ok::<_, Infallible>(Bytes::from(vec![b'a'; MAX_CONTROL_RESPONSE_BYTES])),
            Ok(Bytes::from_static(b"b")),
        ]);
        Response::new(Body::from_stream(chunks))
    }

    let origin = spawn(Router::new().route("/control", post(oversized))).await;
    let result: Result<Value, CliError> = post_json(
        &control_client().expect("client"),
        &format!("{origin}/control"),
        &json!({"request": true}),
        None,
    )
    .await;

    let error = result.expect_err("oversized response must be rejected");
    assert!(
        error
            .to_string()
            .contains("daemon control response exceeded 262144 bytes")
    );
}

#[derive(Default)]
struct RetryState {
    json_bodies: Mutex<Vec<Bytes>>,
    empty_bodies: Mutex<Vec<Bytes>>,
}

#[tokio::test]
async fn idempotent_json_retry_reuses_the_exact_encoded_request() {
    async fn endpoint(State(state): State<Arc<RetryState>>, body: Bytes) -> Response {
        let attempt = {
            let mut bodies = state.json_bodies.lock().expect("json bodies");
            bodies.push(body);
            bodies.len()
        };
        if attempt == 1 {
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
        Json(json!({"accepted": true})).into_response()
    }

    let state = Arc::new(RetryState::default());
    let origin = spawn(
        Router::new()
            .route("/control", post(endpoint))
            .with_state(Arc::clone(&state)),
    )
    .await;
    let result: Value = post_json_idempotent(
        &control_client().expect("client"),
        &format!("{origin}/control"),
        &json!({"sequence": 7, "request_id": "same"}),
        None,
        fast_retry_policy(),
    )
    .await
    .expect("transient response should be retried");

    assert_eq!(result, json!({"accepted": true}));
    let bodies = state.json_bodies.lock().expect("json bodies");
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[0], bodies[1]);
}

#[tokio::test]
async fn idempotent_empty_retry_reuses_the_exact_encoded_request() {
    async fn endpoint(State(state): State<Arc<RetryState>>, body: Bytes) -> StatusCode {
        let attempt = {
            let mut bodies = state.empty_bodies.lock().expect("empty bodies");
            bodies.push(body);
            bodies.len()
        };
        if attempt == 1 {
            StatusCode::BAD_GATEWAY
        } else {
            StatusCode::NO_CONTENT
        }
    }

    let state = Arc::new(RetryState::default());
    let origin = spawn(
        Router::new()
            .route("/control", post(endpoint))
            .with_state(Arc::clone(&state)),
    )
    .await;
    post_empty_idempotent(
        &control_client().expect("client"),
        &format!("{origin}/control"),
        &json!({"sequence": 8, "request_id": "same"}),
        fast_retry_policy(),
    )
    .await
    .expect("transient response should be retried");

    let bodies = state.empty_bodies.lock().expect("empty bodies");
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[0], bodies[1]);
}

fn fast_retry_policy() -> ControlRetryPolicy {
    ControlRetryPolicy::new(
        Duration::from_secs(1),
        Duration::from_secs(2),
        Duration::ZERO,
    )
}

async fn spawn(router: Router) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("local address");
    tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve");
    });
    format!("http://{address}")
}
