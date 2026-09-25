// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for Anthropic compaction cache eligibility and response contracts.

use super::*;
use serde_json::{Map, json};

fn request(body: Json, beta: &str) -> LlmRequest {
    LlmRequest {
        headers: Map::from_iter([
            (
                "anthropic-version".to_string(),
                json!(ANTHROPIC_API_VERSION),
            ),
            ("anthropic-beta".to_string(), json!(beta)),
        ]),
        content: body,
    }
}

fn threshold_edit() -> Json {
    json!({
        "edits": [{
            "type": "compact_20260112",
            "pause_after_compaction": true
        }]
    })
}

fn threshold_block() -> Json {
    json!({"type": "compaction", "content": "summary", "encrypted_content": null})
}

fn ordinary_response() -> Json {
    json!({
        "id": "msg_ordinary",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 2}
    })
}

fn threshold_response() -> Json {
    json!({
        "id": "msg_threshold",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [threshold_block()],
        "stop_reason": "compaction",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "iterations": [{"type": "compaction", "input_tokens": 10, "output_tokens": 5}]
        }
    })
}

#[test]
fn unknown_null_response_metadata_is_not_cacheable() {
    let context = AnthropicCacheContext {
        version: ANTHROPIC_API_VERSION.to_string(),
        beta_tokens: vec![THRESHOLD_BETA.to_string()],
        protocol: AnthropicProtocol::ThresholdV1,
        operation: AnthropicOperation::PausedThreshold,
    };
    let mut response = threshold_response();
    response["future_response_field"] = Json::Null;

    assert_eq!(classify_aggregate(&response, &context), None);
}

#[test]
fn request_shapes_map_to_distinct_compaction_operations() {
    let cases = [
        (
            request(
                json!({
                    "messages": [{"role": "user", "content": "hello"}],
                    "context_management": threshold_edit()
                }),
                THRESHOLD_BETA,
            ),
            AnthropicOperation::PausedThreshold,
        ),
        (
            request(
                json!({
                    "messages": [
                        {"role": "assistant", "content": [threshold_block()]},
                        {"role": "user", "content": "continue"}
                    ]
                }),
                THRESHOLD_BETA,
            ),
            AnthropicOperation::ThresholdContinuation,
        ),
        (
            request(
                json!({
                    "messages": [
                        {"role": "assistant", "content": [threshold_block()]},
                        {"role": "user", "content": "continue"}
                    ],
                    "context_management": threshold_edit()
                }),
                THRESHOLD_BETA,
            ),
            AnthropicOperation::ThresholdRecompact,
        ),
    ];

    for (request, expected) in cases {
        assert_eq!(
            cache_context(&request).unwrap().unwrap().operation,
            expected
        );
    }
}

#[test]
fn response_kind_must_match_the_request_operation() {
    let ordinary = ordinary_response();
    let threshold = threshold_response();
    let cases = [
        (
            AnthropicOperation::PausedThreshold,
            Some(AnthropicResponseKind::Ordinary),
            Some(AnthropicResponseKind::ThresholdCompaction),
        ),
        (
            AnthropicOperation::ThresholdContinuation,
            Some(AnthropicResponseKind::Ordinary),
            None,
        ),
        (
            AnthropicOperation::ThresholdRecompact,
            Some(AnthropicResponseKind::Ordinary),
            Some(AnthropicResponseKind::ThresholdCompaction),
        ),
    ];

    for (operation, ordinary_kind, threshold_kind) in cases {
        let context = AnthropicCacheContext {
            version: ANTHROPIC_API_VERSION.to_string(),
            beta_tokens: vec![THRESHOLD_BETA.to_string()],
            protocol: AnthropicProtocol::ThresholdV1,
            operation,
        };

        assert_eq!(classify_aggregate(&ordinary, &context), ordinary_kind);
        assert_eq!(classify_aggregate(&threshold, &context), threshold_kind);
    }
}

#[test]
fn on_demand_compaction_remains_live() {
    let request = request(
        json!({
            "messages": [{"role": "user", "content": "hello"}],
            "compaction": {"type": "summarize"}
        }),
        "compact-2026-09-04",
    );

    assert_eq!(
        cache_context(&request),
        Err(CacheReason::AnthropicCompaction)
    );
}

#[test]
fn unsupported_anthropic_api_versions_bypass_compaction_caching() {
    let mut request = request(
        json!({
            "messages": [{"role": "user", "content": "hello"}],
            "context_management": threshold_edit()
        }),
        THRESHOLD_BETA,
    );
    request
        .headers
        .insert("anthropic-version".to_string(), json!("future-version"));

    assert_eq!(
        cache_context(&request),
        Err(CacheReason::AnthropicCompaction)
    );
}

fn threshold_stream_with_start_message(start_message: Json) -> Vec<Json> {
    vec![
        json!({"type": "message_start", "message": start_message}),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "compaction", "content": null}
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "compaction_delta", "content": "summary"}
        }),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "compaction", "stop_sequence": null},
            "usage": {
                "output_tokens": 5,
                "iterations": [{"type": "compaction", "input_tokens": 10, "output_tokens": 5}]
            }
        }),
        json!({"type": "message_stop"}),
    ]
}

#[test]
fn stream_start_requires_null_stop_fields() {
    let context = AnthropicCacheContext {
        version: ANTHROPIC_API_VERSION.to_string(),
        beta_tokens: vec![THRESHOLD_BETA.to_string()],
        protocol: AnthropicProtocol::ThresholdV1,
        operation: AnthropicOperation::PausedThreshold,
    };
    let base = json!({
        "id": "msg_threshold",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [],
        "stop_reason": null,
        "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 0}
    });

    let mut cases = Vec::new();
    let mut non_null_reason = base.clone();
    non_null_reason["stop_reason"] = json!("end_turn");
    cases.push(non_null_reason);
    let mut non_null_sequence = base.clone();
    non_null_sequence["stop_sequence"] = json!("DONE");
    cases.push(non_null_sequence);
    let mut missing_reason = base.clone();
    missing_reason
        .as_object_mut()
        .unwrap()
        .remove("stop_reason");
    cases.push(missing_reason);
    let mut missing_sequence = base;
    missing_sequence
        .as_object_mut()
        .unwrap()
        .remove("stop_sequence");
    cases.push(missing_sequence);

    for start_message in cases {
        let mut validator = AnthropicStreamValidator::new(&context);
        for chunk in threshold_stream_with_start_message(start_message) {
            validator.observe(&chunk);
        }
        assert!(!validator.is_valid_for(AnthropicResponseKind::ThresholdCompaction));
    }
}
