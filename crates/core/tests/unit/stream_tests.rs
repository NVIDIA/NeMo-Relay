// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for stream in the NeMo Relay core crate.

use super::*;
use serde_json::json;

fn assert_no_top_level_fields(data: &Json, fields: &[&str]) {
    let object = data.as_object().expect("mark data must be an object");
    for field in fields {
        assert!(
            !object.contains_key(*field),
            "unexpected top-level field {field}"
        );
    }
}

#[test]
fn openai_chat_chunk_summary_keeps_only_compact_metadata() {
    let data = llm_chunk_mark_data(
        7,
        &json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "classified text",
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "{\"secret\":true}"}
                    }]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 3}
            }
        }),
    );

    assert_eq!(
        data,
        json!({
            "chunk_index": 7,
            "provider": "openai_chat_completions",
            "event_type": "chat.completion.chunk",
            "choice_indices": [0],
            "finish_reasons": [{"choice_index": 0, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "cache_read_tokens": 3
            }
        })
    );
    assert_no_top_level_fields(&data, &["choices", "delta", "content", "tool_calls"]);
    assert!(!data.to_string().contains("classified text"));
    assert!(!data.to_string().contains("secret"));
}

#[test]
fn openai_responses_chunk_summary_omits_response_output_item_and_delta() {
    let delta_data = llm_chunk_mark_data(
        0,
        &json!({
            "type": "response.output_text.delta",
            "output_index": 1,
            "content_index": 2,
            "delta": "classified text"
        }),
    );
    assert_eq!(
        delta_data,
        json!({
            "chunk_index": 0,
            "provider": "openai_responses",
            "event_type": "response.output_text.delta",
            "indices": {"output_index": 1, "content_index": 2}
        })
    );
    assert_no_top_level_fields(&delta_data, &["response", "output", "item", "delta"]);
    assert!(!delta_data.to_string().contains("classified text"));

    let completed_data = llm_chunk_mark_data(
        1,
        &json!({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": "classified text"}]
                }],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "total_tokens": 17,
                    "input_tokens_details": {"cached_tokens": 8}
                }
            }
        }),
    );
    assert_eq!(
        completed_data,
        json!({
            "chunk_index": 1,
            "provider": "openai_responses",
            "event_type": "response.completed",
            "status": "completed",
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17,
                "cache_read_tokens": 8
            }
        })
    );
    assert_no_top_level_fields(&completed_data, &["response", "output", "item", "delta"]);
    assert!(!completed_data.to_string().contains("classified text"));
}

#[test]
fn anthropic_chunk_summary_omits_content_blocks_and_delta_payloads() {
    let delta_data = llm_chunk_mark_data(
        3,
        &json!({
            "type": "content_block_delta",
            "index": 0,
            "content_block": {"type": "text", "text": "classified text"},
            "delta": {"type": "input_json_delta", "partial_json": "{\"secret\":true}"}
        }),
    );
    assert_eq!(
        delta_data,
        json!({
            "chunk_index": 3,
            "provider": "anthropic_messages",
            "event_type": "content_block_delta",
            "indices": {"index": 0}
        })
    );
    assert_no_top_level_fields(
        &delta_data,
        &["content_block", "delta", "text", "partial_json"],
    );
    assert!(!delta_data.to_string().contains("classified text"));
    assert!(!delta_data.to_string().contains("secret"));

    let usage_data = llm_chunk_mark_data(
        4,
        &json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {
                "input_tokens": 100,
                "output_tokens": 5,
                "cache_read_input_tokens": 11,
                "cache_creation_input_tokens": 2
            }
        }),
    );
    assert_eq!(
        usage_data,
        json!({
            "chunk_index": 4,
            "provider": "anthropic_messages",
            "event_type": "message_delta",
            "stop_reason": "end_turn",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "total_tokens": 105,
                "cache_read_tokens": 11,
                "cache_write_tokens": 2
            }
        })
    );
}

#[test]
fn unknown_chunk_summary_only_records_receipt() {
    let data = llm_chunk_mark_data(
        2,
        &json!({
            "delta": "classified text",
            "choices": [{"index": 0, "delta": {"content": "still unknown"}}],
            "response": {"output": [{"item": {"text": "secret"}}]}
        }),
    );

    assert_eq!(data, json!({"chunk_index": 2, "provider": "unknown"}));
}
