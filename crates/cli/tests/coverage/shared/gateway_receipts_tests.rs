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

    let forwarded = futures_util::future::join_all((0..9).map(receipt_response_case)).await;
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
    assert_eq!(receipts.len(), 6, "one receipt per valid response header");
    assert_eq!(starts.len(), 9, "one managed LLM per request");
    for (index, traceparent) in forwarded.iter().enumerate() {
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
        assert_eq!(
            traceparent.split('-').nth(2).unwrap(),
            nemo_relay_types::trace::relay_span_id(start.uuid()).to_string(),
            "outbound context must identify the actual model call",
        );
        let request_id = receipt_id(index);
        let receipt = receipts.iter().find(|event| {
            event.data().and_then(|data| data["request_id"].as_str()) == Some(request_id.as_str())
        });
        if index >= 6 {
            assert!(
                receipt.is_none(),
                "missing, malformed and duplicate headers must not produce a receipt",
            );
            continue;
        }
        let receipt = receipt.expect("each valid response must have its own receipt");
        assert_eq!(receipt.parent_uuid(), Some(start.uuid()), "{model}");
        assert_eq!(
            receipt.data(),
            Some(&json!({
                "request_id": request_id,
                "http_status": if index == 1 { 403 } else { 200 },
            })),
            "receipt must not copy unrelated response headers or bodies",
        );
    }
}

fn receipt_id(index: usize) -> String {
    format!("a9d998c6-9604-4c87-a03e-{index:012}")
}

fn response_payload(index: usize) -> String {
    let choice = if index == 0 || index >= 6 {
        json!({"index": 0, "message": {"role": "assistant", "content": "receipt-output"}, "finish_reason": "stop"})
    } else {
        json!({"index": 0, "delta": {"role": "assistant", "content": "receipt-output"}, "finish_reason": null})
    };
    let response = json!({"id": "chatcmpl-receipt", "model": "receipt-model", "choices": [choice]});
    match index {
        0 | 6.. => response.to_string(),
        1 | 2 => "not-json".to_owned(),
        5 => "data: not-json\n\n".to_owned(),
        _ => {
            let mut body = format!("data: {response}\n\n");
            if index == 4 {
                let end = json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]});
                body.push_str(&format!("data: {end}\n\ndata: [DONE]\n\n"));
            }
            body
        }
    }
}

fn fixture_response(index: usize, headers: HeaderMap, release: Arc<Notify>) -> Response<Body> {
    let payload = response_payload(index);
    let body = if index == 3 {
        Body::from_stream(stream! {
            yield Ok::<_, std::io::Error>(Bytes::from(payload));
            release.notified().await;
            yield Err(std::io::Error::other("fixture stream truncated"));
        })
    } else {
        Body::from(payload)
    };
    let response = Response::builder()
        .status(if index == 1 { 403 } else { 200 })
        .header(
            header::CONTENT_TYPE,
            if (3..6).contains(&index) {
                "text/event-stream"
            } else {
                "application/json"
            },
        )
        .header("x-forwarded-traceparent", &headers["traceparent"])
        .header("x-unrelated", "do-not-copy");
    let response = match index {
        6 => response,
        7 => response.header("x-openshell-request-id", "not-a-uuid"),
        8 => response
            .header("x-openshell-request-id", receipt_id(index))
            .header("x-openshell-request-id", receipt_id(index)),
        _ => response.header("x-openshell-request-id", receipt_id(index)),
    };
    response.body(body).unwrap()
}

async fn receipt_response_case(index: usize) -> String {
    let release = Arc::new(Notify::new());
    let fixture_release = release.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |headers: HeaderMap| {
            let release = fixture_release.clone();
            async move { fixture_response(index, headers, release) }
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
                "stream": (3..6).contains(&index),
            })
            .to_string(),
        ))
        .unwrap();
    let response = passthrough(State(state), request).await.unwrap();
    assert_eq!(
        response.status().as_u16(),
        if index == 1 { 403 } else { 200 }
    );
    let traceparent = response.headers()["x-forwarded-traceparent"]
        .to_str()
        .unwrap()
        .to_owned();
    // Managed streaming pre-reads the first SSE event before returning a response.
    // Only now may the fixture abort its actual HTTP body, retaining the receipt.
    release.notify_one();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    upstream.abort();
    match index {
        0 | 6.. => assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap(),
            serde_json::from_str::<Value>(&response_payload(index)).unwrap()
        ),
        1 | 2 => assert_eq!(body.as_ref(), b"not-json"),
        5 => assert_failed_first_event(&body),
        _ => {
            let text = std::str::from_utf8(&body).unwrap();
            assert!(text.contains("receipt-output"), "{text}");
            assert!(
                text.contains(if index == 3 { "stream_error" } else { "[DONE]" }),
                "{text}"
            );
        }
    }
    traceparent
}

fn assert_failed_first_event(body: &[u8]) {
    let text = std::str::from_utf8(body).unwrap();
    let events: Vec<Value> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert_eq!(events.len(), 1, "{text}");
    assert_eq!(events[0]["error"]["type"], "server_error", "{text}");
    assert_eq!(events[0]["error"]["code"], "stream_error", "{text}");
    assert_eq!(
        events[0]["error"]["message"], "NeMo Relay could not complete the response stream.",
        "{text}",
    );
    assert!(!text.contains("[DONE]"), "{text}");
    assert!(!text.contains("receipt-output"), "{text}");
    assert!(!text.contains("not-json"), "{text}");
}
