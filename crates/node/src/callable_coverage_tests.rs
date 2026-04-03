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
fn test_recv_fallback_helpers_cover_error_paths() {
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
    assert_eq!(recv_option_string_or_none(rx, "cond"), None);

    let request = make_request();
    let (tx, rx) = std::sync::mpsc::channel();
    drop(tx);
    assert_eq!(
        recv_llm_request_or_value(rx, "llm", request.clone()).content,
        request.content
    );
}
