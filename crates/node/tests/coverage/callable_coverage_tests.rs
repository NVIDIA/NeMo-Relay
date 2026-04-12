// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::json;

fn make_request() -> LLMRequest {
    LLMRequest {
        headers: serde_json::Map::from_iter([("x-trace".into(), json!("1"))]),
        content: json!({"model": "test-model"}),
    }
}

#[test]
fn test_recv_helpers_cover_error_paths() {
    let (tx, rx) = std::sync::mpsc::channel();
    drop(tx);
    assert_eq!(recv_json_or_null(rx, "tool"), Json::Null);

    let (tx, rx) = std::sync::mpsc::channel();
    drop(tx);
    assert_eq!(
        recv_json_or_value(rx, "resp", json!({"ok": true})),
        json!({"ok": true})
    );

    let (tx, rx) = std::sync::mpsc::channel();
    drop(tx);
    assert!(
        recv_option_string_result(rx, "cond")
            .unwrap_err()
            .to_string()
            .contains("cond")
    );

    let request = make_request();
    let (tx, rx) = std::sync::mpsc::channel();
    drop(tx);
    assert!(
        recv_llm_request_result(rx, "llm")
            .unwrap_err()
            .to_string()
            .contains("llm")
    );

    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(serde_json::to_value(&request).unwrap()).unwrap();
    assert_eq!(
        recv_llm_request_result(rx, "llm").unwrap().content,
        request.content
    );

    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(json!(42)).unwrap();
    assert!(
        recv_option_string_result(rx, "cond")
            .unwrap_err()
            .to_string()
            .contains("expected string or null")
    );

    let fallback = make_request();
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(json!(["bad"])).unwrap();
    let restored = recv_llm_request_or_value(rx, "llm", fallback.clone());
    assert_eq!(restored.content, fallback.content);

    let (tx, rx) = std::sync::mpsc::channel();
    drop(tx);
    assert_eq!(
        recv_json_result(rx, "tool").unwrap_err().to_string(),
        "internal error: tool: receiving on a closed channel"
    );
}
