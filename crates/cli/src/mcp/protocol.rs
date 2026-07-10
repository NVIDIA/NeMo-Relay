// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal MCP/JSON-RPC protocol implemented by the lifecycle client.

use serde_json::{Value, json};

pub(super) const MCP_PROTOCOL_VERSION: &str = "2025-06-18";

/// Result of decoding one newline-delimited MCP frame.
pub(super) struct FrameAction {
    pub(super) response: Option<Value>,
    pub(super) requires_gateway: bool,
}

/// Parse a frame once and derive both its protocol response and bootstrap requirement.
pub(super) fn evaluate_frame(frame: &str) -> FrameAction {
    match serde_json::from_str::<Value>(frame) {
        Ok(message) => FrameAction {
            requires_gateway: is_valid_initialize(&message),
            response: response_for(&message),
        },
        Err(_) => FrameAction {
            response: Some(jsonrpc_error(Value::Null, -32700, "Parse error")),
            requires_gateway: false,
        },
    }
}

fn is_valid_initialize(message: &Value) -> bool {
    valid_jsonrpc_request(message)
        && message.get("method").and_then(Value::as_str) == Some("initialize")
        && message
            .pointer("/params/protocolVersion")
            .and_then(Value::as_str)
            .is_some()
}

fn valid_jsonrpc_request(message: &Value) -> bool {
    message.is_object()
        && message.get("jsonrpc").and_then(Value::as_str) == Some("2.0")
        && message.get("id").is_some()
}

pub(super) fn response_for(message: &Value) -> Option<Value> {
    let id = message.get("id").cloned();
    if !message.is_object() || message.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Some(jsonrpc_error(
            id.unwrap_or(Value::Null),
            -32600,
            "Invalid Request",
        ));
    }
    let id = id?;
    let method = message.get("method").and_then(Value::as_str);
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
                    "capabilities": { "tools": {} },
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
