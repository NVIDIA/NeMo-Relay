// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for response-cache streaming commit behavior.

use std::time::Duration;

use serde_json::json;
use tokio::sync::{oneshot, watch};
use tokio_stream::StreamExt;

use super::*;

#[test]
fn chat_stream_fidelity_gate_rejects_every_uncollected_non_null_shape() {
    let supported = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000_u64,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": "hello",
                "tool_calls": [{
                    "index": 0,
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}
                }]
            },
            "finish_reason": null,
            "logprobs": null
        }],
        "usage": null
    });
    assert!(!chunk_has_uncollected_response_fields(&supported));

    for unsafe_chunk in [
        json!({"choices": [], "system_fingerprint": "fp_123"}),
        json!({"choices": [{"index": 0, "delta": {"content": ["not", "text"]}}]}),
        json!({"choices": [{"index": 0, "delta": {"tool_calls": "not-an-array"}}]}),
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0,
            "function": {"name": "lookup", "arguments": "{}", "extension": true}
        }]}}]}),
    ] {
        assert!(
            chunk_has_uncollected_response_fields(&unsafe_chunk),
            "unsupported response data must veto aggregate storage: {unsafe_chunk}"
        );
    }
}

#[tokio::test]
async fn write_behind_returns_eof_before_cache_commit_completes() {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let (cancel, _) = watch::channel(false);
    let (_, closed) = watch::channel(None::<FlowResult<()>>);
    let (release, wait) = oneshot::channel();
    let (commit_done, committed) = oneshot::channel();
    let commit: CacheCommit = Box::pin(async move {
        let _ = wait.await;
        let _ = commit_done.send(());
    });
    assert!(tx.send(TeeMessage::Commit(commit)).await.is_ok());

    let mut stream = ResponseCacheReceiver {
        receiver: ReceiverStream::new(rx),
        cancel,
        closed,
        finished: false,
    };
    assert!(
        tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("write-behind cache publication must not delay stream completion")
            .is_none()
    );
    release
        .send(())
        .expect("detached cache commit must still be waiting");
    tokio::time::timeout(Duration::from_secs(1), committed)
        .await
        .expect("detached cache commit must resume after release")
        .expect("detached cache commit must run to completion");
}

#[test]
fn response_cache_fidelity_helpers_cover_all_rejection_shapes() {
    assert_uncollected_response_field_shapes();
    assert_aggregate_replay_and_stream_completion();
    assert_inband_error_and_content_detection();
    assert_error_response_and_bypass_detection();
}

fn assert_uncollected_response_field_shapes() {
    for chunk in [
        json!(false),
        json!({"choices": false}),
        json!({"choices": [false]}),
        json!({"choices": [{"index": "zero"}]}),
        json!({"choices": [{"finish_reason": false}]}),
        json!({"choices": [{"delta": false}]}),
        json!({"choices": [{"logprobs": {}}]}),
        json!({"choices": [{"extension": true}]}),
        json!({"choices": [{"delta": {"tool_calls": [false]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": "zero"}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"id": false}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"function": false}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"extension": true}]}}]}),
    ] {
        assert!(chunk_has_uncollected_response_fields(&chunk), "{chunk}");
    }
    assert!(!chunk_has_uncollected_response_fields(&json!({
        "choices": null,
        "metadata": null
    })));
}

fn assert_aggregate_replay_and_stream_completion() {
    assert!(aggregate_replay_lossy(&json!({
        "content": [{"type": "thinking"}]
    })));
    assert!(aggregate_replay_lossy(&json!({
        "choices": [{"message": {"content": null, "tool_calls": []}}]
    })));
    assert!(!aggregate_replay_lossy(&json!({
        "choices": [{"message": {"content": "answer"}}]
    })));

    let mut completion = StreamCompletion::default();
    completion.observe(&json!({
        "choices": [
            {"index": 0, "finish_reason": "stop"},
            {"index": 1, "finish_reason": null}
        ]
    }));
    assert!(!completion.is_terminal());
    completion.observe(&json!({"choices": [{"index": 1, "finish_reason": "stop"}]}));
    assert!(completion.is_terminal());

    let mut stopped = StreamCompletion::default();
    stopped.observe(&json!({"type": "response.completed"}));
    assert!(stopped.is_terminal());
}

fn assert_inband_error_and_content_detection() {
    assert!(chunk_is_inband_error(&json!({"error": "bad"})));
    assert!(chunk_is_inband_error(&json!({"type": "response.failed"})));
    assert!(!chunk_is_inband_error(&json!({"error": null})));
    assert!(aggregate_has_no_content(&json!({})));
    assert!(!aggregate_has_no_content(&json!({"output": [1]})));
}

fn assert_error_response_and_bypass_detection() {
    assert!(!is_error_response(&json!(false)));
    assert!(is_error_response(&json!({"error": "bad"})));
    for status in [
        "failed",
        "cancelled",
        "canceled",
        "incomplete",
        "in_progress",
        "queued",
    ] {
        assert!(is_error_response(&json!({"status": status})));
    }
    assert!(!is_error_response(&json!({"status": "completed"})));
    assert!(!should_bypass(0.0));
    assert!(should_bypass(1.0));
    let unit = next_unit_f64();
    assert!((0.0..1.0).contains(&unit), "{unit}");
}
