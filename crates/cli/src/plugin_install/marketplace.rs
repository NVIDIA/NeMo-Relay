// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generated local marketplace and plugin manifest files.

use std::fs;

use serde_json::{Value, json};

use crate::config::{CodingAgent, PluginHost};
use crate::installer::generated_hooks;

use super::state::{PluginInstallOptions, PluginLayout, remove_path, write_json};
use super::{MARKETPLACE_NAME, PLUGIN_NAME};

pub(super) fn write_plugin_marketplace(
    host: PluginHost,
    layout: &PluginLayout,
    options: &PluginInstallOptions,
) -> Result<(), String> {
    if options.dry_run {
        println!("write {}", layout.marketplace_manifest.display());
        println!("write {}", layout.plugin_manifest.display());
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
    if plugin_has_hooks_template(host) {
        write_json(&layout.hooks_path, &plugin_hooks(host))?;
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
        PluginHost::Codex => "Codex hooks that forward canonical lifecycle payloads to nemo-relay.",
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
        manifest["interface"] = json!({
            "displayName": "NeMo Relay Plugin",
            "shortDescription": "Forward Codex lifecycle hooks to a local NeMo Relay sidecar.",
            "longDescription": "Installs command hooks that preserve Codex hook payloads and forward them to nemo-relay for agent, subagent, tool, and lifecycle observability. Full LLM capture also requires sidecar provider routing.",
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

pub(super) fn plugin_hooks(host: PluginHost) -> Value {
    match host {
        PluginHost::Codex => {
            generated_hooks(CodingAgent::Codex, "nemo-relay plugin-shim hook codex")
        }
        PluginHost::ClaudeCode => generated_hooks(
            CodingAgent::ClaudeCode,
            "nemo-relay plugin-shim hook claude",
        ),
        PluginHost::All => unreachable!("all is expanded before hook generation"),
    }
}

pub(super) fn plugin_has_hooks_template(host: PluginHost) -> bool {
    match host {
        PluginHost::Codex => false,
        PluginHost::ClaudeCode => true,
        PluginHost::All => unreachable!("all is expanded before hook generation"),
    }
}
