// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use axum::{Router, routing::post};
use http_body_util::BodyExt;
use nemo_relay_runtime::api::event::{Event, ScopeCategory};
use nemo_relay_runtime::api::scope::ScopeType;
use nemo_relay_runtime::api::subscriber::{
    deregister_subscriber, flush_subscribers, register_subscriber,
};
use serde_json::json;
use std::time::Duration;
use tokio::sync::Notify;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn managed_receipts_keep_exact_llm_parent_across_concurrent_response_outcomes() {
    const CHILD_ENV: &str = "NEMO_RELAY_TEST_GATEWAY_RECEIPTS";
    if std::env::var_os(CHILD_ENV).is_none() {
        // Runtime registrations are process-wide; keep other gateway tests isolated.
        let output = tokio::time::timeout(
            Duration::from_secs(30),
            tokio::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "gateway::receipt_tests::managed_receipts_keep_exact_llm_parent_across_concurrent_response_outcomes",
                    "--nocapture",
                ])
                .env(CHILD_ENV, "1")
                .kill_on_drop(true)
                .output(),
        )
        .await
        .expect("gateway receipt test timed out")
        .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "gateway-receipts",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let cases = ["complete", "denied", "truncated", "without-receipt"];
    futures_util::future::join_all(
        cases
            .into_iter()
            .enumerate()
            .map(|(index, case)| receipt_response_case(index, case)),
    )
    .await;
    flush_subscribers().unwrap();
    deregister_subscriber("gateway-receipts").unwrap();

    let events = events.lock().unwrap();
    let receipts: Vec<_> = events
        .iter()
        .filter(|event| event.name() == "openshell.request.receipt")
        .collect();
    let starts: Vec<_> = events
        .iter()
        .filter(|event| {
            event.scope_type() == Some(ScopeType::Llm)
                && event.scope_category() == Some(ScopeCategory::Start)
        })
        .collect();
    assert_eq!(receipts.len(), 3, "one receipt per valid response header");
    assert_eq!(starts.len(), cases.len(), "one managed LLM per request");
    for (index, case) in cases.iter().enumerate() {
        let model = format!("receipt-model-{index}");
        let start = starts
            .iter()
            .find(|event| {
                event
                    .input()
                    .and_then(|input| input["content"]["model"].as_str())
                    == Some(model.as_str())
            })
            .expect("each concurrent request must have its own model start");
        let request_id = receipt_id(index);
        let receipt = receipts.iter().find(|event| {
            event.data().and_then(|data| data["request_id"].as_str()) == Some(request_id.as_str())
        });
        if *case == "without-receipt" {
            assert!(
                receipt.is_none(),
                "missing header must not produce a receipt"
            );
            continue;
        }
        let receipt = receipt.expect("each valid response must have its own receipt");
        assert_eq!(receipt.parent_uuid(), Some(start.uuid()), "{model}");
        assert_eq!(
            receipt.data(),
            Some(&json!({
                "request_id": request_id,
                "http_status": if *case == "denied" { 403 } else { 200 },
            })),
            "receipt must not copy unrelated response headers or bodies",
        );
    }
}

fn receipt_id(index: usize) -> String {
    format!("a9d998c6-9604-4c87-a03e-{index:012}")
}

async fn receipt_response_case(index: usize, case: &str) {
    let streaming = case == "truncated";
    let status = if case == "denied" { 403 } else { 200 };
    let with_receipt = case != "without-receipt";
    let release = Arc::new(Notify::new());
    let fixture_release = release.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let release = fixture_release.clone();
            async move {
                let choice = if streaming {
                    json!({"index": 0, "delta": {"content": "receipt-output"}, "finish_reason": null})
                } else {
                    json!({"index": 0, "message": {"role": "assistant", "content": "receipt-output"}, "finish_reason": "stop"})
                };
                let payload = json!({"id": "chatcmpl-receipt", "model": "receipt-model", "choices": [choice]}).to_string();
                let body = if streaming {
                    Body::from_stream(stream! {
                        yield Ok::<_, std::io::Error>(Bytes::from(format!("data: {payload}\n\n")));
                        release.notified().await;
                        yield Err(std::io::Error::other("fixture stream truncated"));
                    })
                } else {
                    Body::from(if status == 403 { "not-json".to_owned() } else { payload })
                };
                let mut response = Response::builder()
                    .status(status)
                    .header(header::CONTENT_TYPE, if streaming { "text/event-stream" } else { "application/json" })
                    .header("x-unrelated", "do-not-copy");
                if with_receipt {
                    response = response.header("x-openshell-request-id", receipt_id(index));
                }
                response.body(body).unwrap()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let config = crate::configuration::GatewayConfig {
        openai_base_url: format!("http://{}/v1", listener.local_addr().unwrap()),
        ..Default::default()
    };
    let upstream = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let mut state = AppState::new(config);
    state.http = reqwest::Client::builder().no_proxy().build().unwrap();
    let request = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        // Prevent ambient provider credentials from reaching this loopback fixture.
        .header(header::AUTHORIZATION, "Bearer receipt-fixture")
        .body(Body::from(
            json!({
                "model": format!("receipt-model-{index}"),
                "messages": [{"role": "user", "content": "capture the receipt"}],
                "stream": streaming,
            })
            .to_string(),
        ))
        .unwrap();
    let response = passthrough(State(state), request).await.unwrap();
    assert_eq!(response.status().as_u16(), status);
    release.notify_one();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    upstream.abort();
    if streaming {
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("receipt-output"), "{text}");
        assert!(text.contains("stream_error"), "{text}");
    }
}
