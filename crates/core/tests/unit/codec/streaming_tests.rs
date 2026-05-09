// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for streaming in the NeMo Flow core crate.

use super::*;
use serde_json::json;

#[test]
fn decodes_complete_frames_in_one_push() {
    let mut decoder = SseEventDecoder::new();
    let events = decoder
        .push_bytes(
            b"event: ping\ndata: {\"type\":\"ping\"}\n\nevent: msg\ndata: {\"text\":\"hi\"}\n\n",
        )
        .unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].event.as_deref(), Some("ping"));
    assert_eq!(events[0].data, json!({"type": "ping"}));
    assert_eq!(events[1].event.as_deref(), Some("msg"));
    assert_eq!(events[1].data, json!({"text": "hi"}));
}

#[test]
fn buffers_partial_frames_across_pushes() {
    let mut decoder = SseEventDecoder::new();
    assert!(decoder.push_bytes(b"event: m\ndata: ").unwrap().is_empty());
    assert!(decoder.push_bytes(b"{\"a\":1").unwrap().is_empty());
    let events = decoder.push_bytes(b"}\n\n").unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].data, json!({"a": 1}));
}

#[test]
fn drops_frames_without_data_lines() {
    let mut decoder = SseEventDecoder::new();
    // A heartbeat-style comment frame plus a real one.
    let events = decoder
        .push_bytes(b": keepalive\n\nevent: real\ndata: {\"v\":2}\n\n")
        .unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].data, json!({"v": 2}));
}

#[test]
fn surfaces_final_partial_frame_on_finish() {
    let mut decoder = SseEventDecoder::new();
    decoder
        .push_bytes(b"event: tail\ndata: {\"end\":true}")
        .unwrap();
    let trailing = decoder.finish().unwrap().expect("trailing frame present");
    assert_eq!(trailing.data, json!({"end": true}));
}

#[test]
fn drops_openai_chat_done_sentinel() {
    let mut decoder = SseEventDecoder::new();
    let events = decoder
        .push_bytes(
            b"data: {\"id\":\"chatcmpl-1\"}\n\ndata: [DONE]\n\ndata: {\"id\":\"chatcmpl-2\"}\n\n",
        )
        .unwrap();
    // [DONE] is dropped; surrounding JSON events still come through.
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].data, json!({"id": "chatcmpl-1"}));
    assert_eq!(events[1].data, json!({"id": "chatcmpl-2"}));
}

#[test]
fn surfaces_parse_errors_with_payload_context() {
    let mut decoder = SseEventDecoder::new();
    let error = decoder
        .push_bytes(b"event: bad\ndata: {not valid json}\n\n")
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("SSE data payload"), "{message}");
    assert!(message.contains("not valid json"), "{message}");
}
