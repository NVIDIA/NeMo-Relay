// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generated local marketplace and plugin manifest files.

use std::env;
use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::installation::generation::{
    write_new_generation_with_token_at, write_staged_generation_with_token,
};

use super::state::{
    MarketplaceHostIdentity, PluginInstallOptions, PluginLayout, remove_path, write_json,
};
use super::{MARKETPLACE_NAME, PLUGIN_NAME};

pub(super) fn write_plugin_marketplace(
    host: impl MarketplaceHostIdentity,
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
    host: impl MarketplaceHostIdentity,
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

pub(super) fn marketplace_manifest(host: impl MarketplaceHostIdentity) -> Value {
    host.marketplace_manifest(MARKETPLACE_NAME, PLUGIN_NAME)
}

pub(super) fn plugin_manifest(host: impl MarketplaceHostIdentity) -> Value {
    host.plugin_manifest(PLUGIN_NAME)
}

pub(super) fn plugin_mcp_config(
    host: impl MarketplaceHostIdentity,
    relay: &Path,
    generation_fence: &Path,
    generation_token: &str,
) -> Result<Value, String> {
    let generation_fence = absolute_or_self(generation_fence);
    let server = crate::mcp::persistent_server(relay, &generation_fence, generation_token);
    host.plugin_mcp_config(server)
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
    host: impl MarketplaceHostIdentity,
    relay: &Path,
    generation_fence: &Path,
    generation_token: &str,
) -> Result<Value, String> {
    let generation_fence = absolute_or_self(generation_fence);
    host.plugin_hooks(relay, &generation_fence, generation_token)
}

#[cfg(test)]
pub(super) fn plugin_mcp_env_vars_from(
    environment: impl IntoIterator<Item = String>,
    config: Option<&Value>,
) -> Vec<String> {
    crate::agents::codex_mcp_env_vars_from(environment, config)
}
