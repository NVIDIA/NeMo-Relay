// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};

use crate::mcp::SERVER_NAME;

pub(crate) fn marketplace_manifest(marketplace: &str, plugin: &str) -> Value {
    json!({
        "name": marketplace,
        "metadata": { "description": "Local NeMo Relay plugins for Claude Code." },
        "owner": { "name": "NVIDIA Corporation and Affiliates", "email": "noreply@nvidia.com" },
        "plugins": [{
            "name": plugin,
            "description": "Run the shared native Relay gateway and capture Claude Code lifecycle events.",
            "source": "./plugins/nemo-relay-plugin",
            "category": "development"
        }]
    })
}

pub(crate) fn plugin_manifest(plugin: &str) -> Value {
    json!({
        "name": plugin,
        "version": env!("CARGO_PKG_VERSION"),
        "description": "Native Relay gateway lifecycle and Claude Code hooks for complete local observability.",
        "author": { "name": "NVIDIA Corporation and Affiliates", "url": "https://github.com/NVIDIA/NeMo-Relay" },
        "homepage": "https://github.com/NVIDIA/NeMo-Relay",
        "repository": "https://github.com/NVIDIA/NeMo-Relay",
        "license": "Apache-2.0",
        "keywords": ["nemo-relay", "claude-code", "hooks", "observability"],
        "mcpServers": "./.mcp.json"
    })
}

pub(crate) fn mcp_config(mut server: Value) -> Value {
    server
        .as_object_mut()
        .expect("persistent MCP server is a JSON object")
        .insert("alwaysLoad".into(), json!(true));
    json!({ "mcpServers": { (SERVER_NAME): server } })
}
