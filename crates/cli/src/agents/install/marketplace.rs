// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generated local marketplace and plugin manifest files.

use std::env;
use std::fs;
use std::path::Path;

use serde_json::{Value, json};

use crate::agents::CodingAgent;
use crate::hooks::generated_hooks;
use crate::installation::generation::{
    write_new_generation_with_token_at, write_staged_generation_with_token,
};

use super::state::{PluginInstallOptions, PluginLayout, remove_path, write_json};
use super::{MARKETPLACE_NAME, PLUGIN_NAME};

pub(super) fn write_plugin_marketplace(
    host: CodingAgent,
    layout: &PluginLayout,
    relay: &Path,
    options: &PluginInstallOptions,
) -> Result<(), String> {
    write_plugin_marketplace_for_generation(
        host,
        layout,
        relay,
        &layout.generation_fence,
        &layout.generation_lock,
        true,
        options,
    )
}

pub(super) fn write_plugin_marketplace_for_generation(
    host: CodingAgent,
    layout: &PluginLayout,
    relay: &Path,
    active_generation_fence: &Path,
    active_generation_lock: &Path,
    initialize_generation_lock: bool,
    options: &PluginInstallOptions,
) -> Result<(), String> {
    if options.dry_run {
        println!("write {}", layout.marketplace_manifest.display());
        println!("write {}", layout.plugin_manifest.display());
        println!("write {}", layout.mcp_config.display());
        println!("write {}", layout.generation_fence.display());
        println!("write {}", layout.hooks_path.display());
        return Ok(());
    }
    remove_path(&layout.plugin_root, options)?;
    fs::create_dir_all(
        layout
            .plugin_root
            .parent()
            .unwrap_or(&layout.marketplace_root),
    )
    .map_err(|error| format!("failed to create {}: {error}", layout.plugin_root.display()))?;
    fs::create_dir_all(layout.hooks_path.parent().unwrap_or(&layout.plugin_root))
        .map_err(|error| format!("failed to create {}: {error}", layout.hooks_path.display()))?;
    write_json(&layout.marketplace_manifest, &marketplace_manifest(host))?;
    write_json(&layout.plugin_manifest, &plugin_manifest(host))?;
    let generation_token = if initialize_generation_lock {
        write_new_generation_with_token_at(&layout.generation_fence, active_generation_lock)
    } else {
        write_staged_generation_with_token(&layout.generation_fence, active_generation_lock)
    }?;
    write_json(
        &layout.mcp_config,
        &plugin_mcp_config(host, relay, active_generation_fence, &generation_token)?,
    )?;
    write_json(
        &layout.hooks_path,
        &plugin_hooks(host, relay, active_generation_fence, &generation_token)?,
    )?;
    Ok(())
}

pub(super) fn marketplace_manifest(host: CodingAgent) -> Value {
    match host {
        CodingAgent::Codex => json!({
            "name": MARKETPLACE_NAME,
            "interface": {
                "displayName": "NeMo Relay Local"
            },
            "plugins": [{
                "name": PLUGIN_NAME,
                "source": {
                    "source": "local",
                    "path": "./plugins/nemo-relay-plugin"
                },
                "policy": {
                    "installation": "AVAILABLE",
                    "authentication": "ON_INSTALL"
                },
                "category": "Coding"
            }]
        }),
        CodingAgent::ClaudeCode => json!({
            "name": MARKETPLACE_NAME,
            "metadata": {
                "description": "Local NeMo Relay plugins for Claude Code."
            },
            "owner": {
                "name": "NVIDIA Corporation and Affiliates",
                "email": "noreply@nvidia.com"
            },
            "plugins": [{
                "name": PLUGIN_NAME,
                "description": "Run the shared native Relay gateway and capture Claude Code lifecycle events.",
                "source": "./plugins/nemo-relay-plugin",
                "category": "development"
            }]
        }),
        CodingAgent::Hermes => {
            unreachable!("all is expanded before manifest generation")
        }
    }
}

pub(super) fn plugin_manifest(host: CodingAgent) -> Value {
    let description = match host {
        CodingAgent::Codex => {
            "Native Relay gateway lifecycle and Codex hooks for complete local observability."
        }
        CodingAgent::ClaudeCode => {
            "Native Relay gateway lifecycle and Claude Code hooks for complete local observability."
        }
        CodingAgent::Hermes => {
            unreachable!("all is expanded before manifest generation")
        }
    };
    let keywords = match host {
        CodingAgent::Codex => json!(["nemo-relay", "codex", "hooks", "observability"]),
        CodingAgent::ClaudeCode => {
            json!(["nemo-relay", "claude-code", "hooks", "observability"])
        }
        CodingAgent::Hermes => {
            unreachable!("all is expanded before manifest generation")
        }
    };
    let mut manifest = json!({
        "name": PLUGIN_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "description": description,
        "author": {
            "name": "NVIDIA Corporation and Affiliates",
            "url": "https://github.com/NVIDIA/NeMo-Relay"
        },
        "homepage": "https://github.com/NVIDIA/NeMo-Relay",
        "repository": "https://github.com/NVIDIA/NeMo-Relay",
        "license": "Apache-2.0",
        "keywords": keywords
    });
    manifest["mcpServers"] = json!("./.mcp.json");
    if matches!(host, CodingAgent::Codex) {
        manifest["interface"] = json!({
            "displayName": "NeMo Relay Plugin",
            "shortDescription": "Run the native Relay gateway and capture Codex lifecycle events.",
            "longDescription": "Starts the native nemo-relay gateway through a required lifecycle-bound MCP server, routes model traffic through it, and installs command hooks that preserve canonical Codex lifecycle payloads.",
            "developerName": "NVIDIA",
            "category": "Coding",
            "capabilities": ["Read"],
            "defaultPrompt": ["Capture this Codex session with NeMo Relay observability."],
            "websiteURL": "https://github.com/NVIDIA/NeMo-Relay",
            "brandColor": "#76B900"
        });
    }
    manifest
}

pub(super) fn plugin_mcp_config(
    host: CodingAgent,
    relay: &Path,
    generation_fence: &Path,
    generation_token: &str,
) -> Result<Value, String> {
    let generation_fence = absolute_or_self(generation_fence);
    let mut server = crate::mcp::persistent_server(relay, &generation_fence, generation_token);
    let fields = server
        .as_object_mut()
        .expect("persistent MCP server is a JSON object");
    match host {
        CodingAgent::Codex => {
            fields.insert("env_vars".into(), json!(plugin_mcp_env_vars()?));
            fields.insert("required".into(), json!(true));
            fields.insert("startup_timeout_sec".into(), json!(20));
        }
        CodingAgent::ClaudeCode => {
            fields.insert("alwaysLoad".into(), json!(true));
        }
        CodingAgent::Hermes => {
            unreachable!("all is expanded before MCP generation")
        }
    }
    Ok(match host {
        CodingAgent::Codex => json!({ (crate::mcp::SERVER_NAME): server }),
        CodingAgent::ClaudeCode => {
            json!({ "mcpServers": { (crate::mcp::SERVER_NAME): server } })
        }
        CodingAgent::Hermes => {
            unreachable!("all is expanded before MCP generation")
        }
    })
}

fn absolute_or_self(path: &Path) -> std::path::PathBuf {
    if path.is_absolute() {
        return path.to_owned();
    }
    env::current_dir()
        .map(|current| current.join(path))
        .unwrap_or_else(|_| path.to_owned())
}

pub(super) fn plugin_hooks(
    host: CodingAgent,
    relay: &Path,
    generation_fence: &Path,
    generation_token: &str,
) -> Result<Value, String> {
    let agent = host;
    let generation_fence = absolute_or_self(generation_fence);
    Ok(generated_hooks(
        agent,
        &crate::hooks::persistent_hook_forward_command(
            relay,
            agent,
            &generation_fence,
            generation_token,
        )?,
    ))
}

pub(super) fn plugin_mcp_env_vars() -> Result<Vec<String>, String> {
    let environment = env::vars_os().filter_map(|(name, _)| name.into_string().ok());
    let config =
        crate::configuration::user_plugin_runtime_config().map_err(|error| error.to_string())?;
    Ok(plugin_mcp_env_vars_from(environment, config.as_ref()))
}

pub(super) fn plugin_mcp_env_vars_from(
    environment: impl IntoIterator<Item = String>,
    config: Option<&Value>,
) -> Vec<String> {
    crate::mcp_environment::forwarded_names(environment, config)
}
