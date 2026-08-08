// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for response-cache streaming commit behavior.

use std::time::Duration;

use nemo_relay::api::runtime::LlmJsonStream;
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

#[test]
fn malformed_stream_shapes_are_not_aggregated() {
    for malformed in [
        json!(null),
        json!({"choices": {}}),
        json!({"choices": [null]}),
        json!({"choices": [{"index": "first"}]}),
        json!({"choices": [{"finish_reason": 1}]}),
        json!({"choices": [{"unsupported": true}]}),
        json!({"choices": [{"delta": "not-an-object"}]}),
        json!({"choices": [{"delta": {"tool_calls": [null]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"id": 1}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"unsupported": true}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"function": "not-an-object"}]}}]}),
    ] {
        assert!(
            chunk_has_uncollected_response_fields(&malformed),
            "malformed stream chunk must not be cached: {malformed}"
        );
    }

    assert!(!chunk_has_uncollected_response_fields(&json!({
        "type": "message_delta"
    })));
    assert!(!chunk_has_uncollected_response_fields(&json!({
        "choices": null
    })));
    assert!(
        !chunk_has_uncollected_response_fields(&json!({
            "choices": [{"delta": {"tool_calls": [{"id": null}]}}]
        })),
        "null tool-call metadata is harmless when no uncollectable fields are present"
    );
}

#[test]
fn replay_and_error_guards_reject_unfaithful_or_failed_responses() {
    assert!(aggregate_replay_lossy(&json!({
        "choices": [{
            "message": {"role": "assistant", "content": null, "tool_calls": []}
        }]
    })));
    assert!(chunk_is_inband_error(&json!({"type": "response.failed"})));
    assert!(chunk_is_inband_error(
        &json!({"error": {"message": "upstream failed"}})
    ));
    assert!(!chunk_is_inband_error(&json!({"error": null})));
    assert!(!is_error_response(&json!("not-an-object")));
}

#[test]
fn sampled_bypass_uses_a_unit_interval_rng() {
    assert_eq!(rng_seed() & 1, 1, "xorshift state must never be zero");

    RNG_STATE.with(|state| state.set(1));
    let expected = next_unit_f64() < 0.5;
    RNG_STATE.with(|state| state.set(1));
    assert_eq!(should_bypass(0.5), expected);
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
    assert!(
        stream.next().await.is_none(),
        "finished streams stay finished"
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

#[tokio::test]
async fn stream_close_reports_when_the_cleanup_task_ends_early() {
    let (cancel, _) = watch::channel(false);
    let (closed_tx, closed) = watch::channel(None::<FlowResult<()>>);
    drop(closed_tx);
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    drop(tx);
    let mut stream = LlmJsonStream::from_closeable(ResponseCacheReceiver {
        receiver: ReceiverStream::new(rx),
        cancel,
        closed,
        finished: false,
    });

    let error = stream
        .close()
        .await
        .expect_err("an unavailable cleanup result must be reported");
    assert!(
        error
            .to_string()
            .contains("response-cache stream cleanup task ended early")
    );
}
