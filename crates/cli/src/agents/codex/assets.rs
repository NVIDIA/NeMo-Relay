// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};

use crate::mcp::SERVER_NAME;

pub(crate) fn marketplace_manifest(marketplace: &str, plugin: &str) -> Value {
    json!({
        "name": marketplace,
        "interface": { "displayName": "NeMo Relay Local" },
        "plugins": [{
            "name": plugin,
            "source": { "source": "local", "path": "./plugins/nemo-relay-plugin" },
            "policy": { "installation": "AVAILABLE", "authentication": "ON_INSTALL" },
            "category": "Coding"
        }]
    })
}

pub(crate) fn plugin_manifest(plugin: &str) -> Value {
    json!({
        "name": plugin,
        "version": env!("CARGO_PKG_VERSION"),
        "description": "Native Relay gateway lifecycle and Codex hooks for complete local observability.",
        "author": { "name": "NVIDIA Corporation and Affiliates", "url": "https://github.com/NVIDIA/NeMo-Relay" },
        "homepage": "https://github.com/NVIDIA/NeMo-Relay",
        "repository": "https://github.com/NVIDIA/NeMo-Relay",
        "license": "Apache-2.0",
        "keywords": ["nemo-relay", "codex", "hooks", "observability"],
        "mcpServers": "./.mcp.json",
        "interface": {
            "displayName": "NeMo Relay Plugin",
            "shortDescription": "Run the native Relay gateway and capture Codex lifecycle events.",
            "longDescription": "Starts the native nemo-relay gateway through a required lifecycle-bound MCP server, routes model traffic through it, and installs command hooks that preserve canonical Codex lifecycle payloads.",
            "developerName": "NVIDIA",
            "category": "Coding",
            "capabilities": ["Read"],
            "defaultPrompt": ["Capture this Codex session with NeMo Relay observability."],
            "websiteURL": "https://github.com/NVIDIA/NeMo-Relay",
            "brandColor": "#76B900"
        }
    })
}

pub(crate) fn mcp_config(mut server: Value) -> Result<Value, String> {
    let fields = server
        .as_object_mut()
        .expect("persistent MCP server is a JSON object");
    fields.insert("env_vars".into(), json!(mcp_env_vars()?));
    fields.insert("required".into(), json!(true));
    fields.insert("startup_timeout_sec".into(), json!(20));
    Ok(json!({ (SERVER_NAME): server }))
}

pub(crate) fn mcp_env_vars() -> Result<Vec<String>, String> {
    let environment = std::env::vars_os().filter_map(|(name, _)| name.into_string().ok());
    let config =
        crate::configuration::user_plugin_runtime_config().map_err(|error| error.to_string())?;
    Ok(mcp_env_vars_from(environment, config.as_ref()))
}

pub(crate) fn mcp_env_vars_from(
    environment: impl IntoIterator<Item = String>,
    config: Option<&Value>,
) -> Vec<String> {
    crate::mcp_environment::forwarded_names(environment, config)
}
