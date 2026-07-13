// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal MCP/JSON-RPC protocol implemented by the lifecycle client.

use serde_json::{Value, json};

pub(super) const MCP_PROTOCOL_VERSION: &str = "2025-06-18";

/// Result of decoding one newline-delimited MCP frame.
pub(super) struct FrameAction {
    pub(super) response: Option<Value>,
}

/// Parse a frame once and derive its protocol response.
pub(super) fn evaluate_frame(frame: &str) -> FrameAction {
    match serde_json::from_str::<Value>(frame) {
        Ok(message) => FrameAction {
            response: response_for(&message),
        },
        Err(_) => FrameAction {
            response: Some(jsonrpc_error(Value::Null, -32700, "Parse error")),
        },
    }
}

pub(super) fn response_for(message: &Value) -> Option<Value> {
    let raw_id = message.get("id");
    let response_id = raw_id
        .filter(|id| valid_request_id(id))
        .cloned()
        .unwrap_or(Value::Null);
    if !message.is_object() || message.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Some(jsonrpc_error(response_id, -32600, "Invalid Request"));
    }
    let method = message.get("method").and_then(Value::as_str);
    if raw_id.is_some_and(|id| !valid_request_id(id)) {
        return Some(jsonrpc_error(Value::Null, -32600, "Invalid Request"));
    }
    if method.is_none() {
        return Some(jsonrpc_error(response_id, -32600, "Invalid Request"));
    }
    let id = raw_id?.clone();
    match method {
        Some("initialize") => {
            let Some(requested_protocol) = message
                .pointer("/params/protocolVersion")
                .and_then(Value::as_str)
            else {
                return Some(jsonrpc_error(id, -32602, "Missing protocolVersion"));
            };
            let protocol_version = if requested_protocol == MCP_PROTOCOL_VERSION {
                requested_protocol
            } else {
                MCP_PROTOCOL_VERSION
            };
            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": protocol_version,
                    "capabilities": {},
                    "serverInfo": {
                        "name": "nemo-relay",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }
            }))
        }
        Some("tools/list") => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": { "tools": [] }
        })),
        Some("ping") => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {}
        })),
        Some(_) => Some(jsonrpc_error(id, -32601, "Method not found")),
        None => Some(jsonrpc_error(id, -32600, "Invalid Request")),
    }
}

fn valid_request_id(id: &Value) -> bool {
    id.is_string() || id.as_i64().is_some() || id.as_u64().is_some()
}

pub(super) fn jsonrpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message
        }
    })
}
