// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generated local marketplace and plugin manifest files.

use std::env;
use std::fs;
use std::path::Path;

use serde_json::{Value, json};

use crate::config::{CodingAgent, PluginHost};
use crate::install_generation::{GENERATION_FILE_ENV, write_new_generation};
use crate::installer::generated_hooks;

use super::state::{PluginInstallOptions, PluginLayout, remove_path, write_json};
use super::{MARKETPLACE_NAME, PLUGIN_NAME};

pub(super) fn write_plugin_marketplace(
    host: PluginHost,
    layout: &PluginLayout,
    relay: &Path,
    options: &PluginInstallOptions,
) -> Result<(), String> {
    write_plugin_marketplace_for_generation(host, layout, relay, &layout.generation_fence, options)
}

pub(super) fn write_plugin_marketplace_for_generation(
    host: PluginHost,
    layout: &PluginLayout,
    relay: &Path,
    active_generation_fence: &Path,
    options: &PluginInstallOptions,
) -> Result<(), String> {
    if options.dry_run {
        println!("write {}", layout.marketplace_manifest.display());
        println!("write {}", layout.plugin_manifest.display());
        if matches!(host, PluginHost::Codex) {
            println!("write {}", layout.mcp_config.display());
            println!("write {}", layout.generation_fence.display());
        }
        if plugin_has_hooks_template(host) {
            println!("write {}", layout.hooks_path.display());
        }
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
    if plugin_has_hooks_template(host) {
        fs::create_dir_all(layout.hooks_path.parent().unwrap_or(&layout.plugin_root)).map_err(
            |error| format!("failed to create {}: {error}", layout.hooks_path.display()),
        )?;
    }
    write_json(&layout.marketplace_manifest, &marketplace_manifest(host))?;
    write_json(&layout.plugin_manifest, &plugin_manifest(host))?;
    if matches!(host, PluginHost::Codex) {
        write_new_generation(&layout.generation_fence)?;
    }
    if let Some(mcp_config) = plugin_mcp_config(host, relay, active_generation_fence)? {
        write_json(&layout.mcp_config, &mcp_config)?;
    }
    if plugin_has_hooks_template(host) {
        write_json(&layout.hooks_path, &plugin_hooks(host, relay))?;
    }
    Ok(())
}

pub(super) fn marketplace_manifest(host: PluginHost) -> Value {
    match host {
        PluginHost::Codex => json!({
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
        PluginHost::ClaudeCode => json!({
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
                "description": "Forward Claude Code lifecycle hooks to a local NeMo Relay sidecar.",
                "source": "./plugins/nemo-relay-plugin",
                "category": "development"
            }]
        }),
        PluginHost::All => unreachable!("all is expanded before manifest generation"),
    }
}

pub(super) fn plugin_manifest(host: PluginHost) -> Value {
    let description = match host {
        PluginHost::Codex => {
            "Native Relay gateway lifecycle and Codex hooks for complete local observability."
        }
        PluginHost::ClaudeCode => {
            "Claude Code hooks that forward canonical lifecycle payloads to nemo-relay."
        }
        PluginHost::All => unreachable!("all is expanded before manifest generation"),
    };
    let keywords = match host {
        PluginHost::Codex => json!(["nemo-relay", "codex", "hooks", "observability"]),
        PluginHost::ClaudeCode => json!(["nemo-relay", "claude-code", "hooks", "observability"]),
        PluginHost::All => unreachable!("all is expanded before manifest generation"),
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
    if matches!(host, PluginHost::Codex) {
        manifest["mcpServers"] = json!("./.mcp.json");
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
    host: PluginHost,
    relay: &Path,
    generation_fence: &Path,
) -> Result<Option<Value>, String> {
    if !matches!(host, PluginHost::Codex) {
        return Ok(None);
    }
    let generation_fence = absolute_or_self(generation_fence);
    Ok(Some(json!({
        "nemo-relay": {
            "command": relay,
            "args": ["mcp"],
            "env": {
                "NEMO_RELAY_GATEWAY_BIND": "127.0.0.1:47632",
                (GENERATION_FILE_ENV): generation_fence
            },
            "env_vars": plugin_mcp_env_vars()?,
            "required": true,
            "startup_timeout_sec": 20
        }
    })))
}

fn absolute_or_self(path: &Path) -> std::path::PathBuf {
    if path.is_absolute() {
        return path.to_owned();
    }
    env::current_dir()
        .map(|current| current.join(path))
        .unwrap_or_else(|_| path.to_owned())
}

pub(super) fn plugin_hooks(host: PluginHost, relay: &Path) -> Value {
    match host {
        PluginHost::Codex => generated_hooks(
            CodingAgent::Codex,
            &crate::plugin_shim::codex_plugin_hook_command(relay),
        ),
        PluginHost::ClaudeCode => generated_hooks(
            CodingAgent::ClaudeCode,
            "nemo-relay plugin-shim hook claude",
        ),
        PluginHost::All => unreachable!("all is expanded before hook generation"),
    }
}

pub(super) fn plugin_mcp_env_vars() -> Result<Vec<String>, String> {
    let environment = env::vars_os().filter_map(|(name, _)| name.into_string().ok());
    let config = crate::config::user_plugin_runtime_config().map_err(|error| error.to_string())?;
    Ok(plugin_mcp_env_vars_from(environment, config.as_ref()))
}

pub(super) fn plugin_mcp_env_vars_from(
    environment: impl IntoIterator<Item = String>,
    config: Option<&Value>,
) -> Vec<String> {
    crate::mcp_environment::forwarded_names(environment, config)
}

pub(super) fn plugin_has_hooks_template(host: PluginHost) -> bool {
    match host {
        PluginHost::Codex | PluginHost::ClaudeCode => true,
        PluginHost::All => unreachable!("all is expanded before hook generation"),
    }
}
