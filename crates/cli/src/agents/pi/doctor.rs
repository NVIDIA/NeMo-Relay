// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostics for the pi integration.
//!
//! Codex and Claude Code can be checked by inspecting files NeMo Relay wrote
//! (generated hook config, a settings base URL). pi's hooks live inside an
//! extension the user loads, so what is checkable from here is where that
//! extension sits and whether pi will actually load it.
//!
//! **The failure this module exists for is silent.** pi adds project-scoped
//! extensions to its candidate set only when the project is trusted
//! (`core/package-manager.ts:2394`), and `-p`, `--mode json` and `--mode rpc`
//! never prompt for trust (`docs/security.md:29`). Under the default policy a
//! project-scoped extension is therefore dropped by a bare conditional -- not
//! by an error path, so it never reaches pi's extension-load error list and pi
//! does not consider it a failure. Nothing reports it, and **the extension
//! cannot report it either**: by construction it is not running. A preflight
//! that reads the filesystem is the only place this can be caught.

use std::path::{Path, PathBuf};

use super::launch::{PI_EXTENSION_PATH_ENV, PI_GATEWAY_URL_ENV};

/// pi's per-user configuration root, `~/.pi/agent` unless overridden.
///
/// Mirrors `getAgentDir()` (pi `config.ts:515-522`), including the environment
/// override, so the preflight looks where pi will actually look.
const PI_AGENT_DIR_ENV: &str = "PI_CODING_AGENT_DIR";

/// pi's configuration directory name, from its `piConfig.configDir`.
const PI_CONFIG_DIR: &str = ".pi";

/// Gateway URL the extension falls back to when nothing else resolves one.
/// Kept in step with `configFromEnv` in `integrations/pi/src/gateway-client.ts`.
const DEFAULT_GATEWAY_URL: &str = "http://127.0.0.1:4040";

/// How pi reaches an extension, which is what decides whether trust gates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExtensionScope {
    /// Passed with `-e`, which is what `nemo-relay run --agent pi` does. Loads
    /// first in precedence and is never trust-gated, so it works in every mode.
    Explicit,
    /// Auto-discovered under the user's own config directory. Not trust-gated.
    User,
    /// Auto-discovered under the project's `.pi/`. **Trust-gated**, and
    /// therefore silently skipped in every non-interactive mode.
    Project,
}

/// A place a pi extension was found, and how pi would reach it.
#[derive(Debug, Clone)]
pub(crate) struct ExtensionSite {
    pub(crate) path: PathBuf,
    pub(crate) scope: ExtensionScope,
}

/// Human-readable hook status for `nemo-relay doctor`.
pub(crate) fn hook_status() -> Result<String, String> {
    match extension_location() {
        Some(path) => Ok(format!(
            "pi extension resolved at {} (hooks are emitted by the extension, not by pi itself)",
            path.display()
        )),
        None => Ok(format!(
            "pi extension not located; set {PI_EXTENSION_PATH_ENV}, or install the extension with \
             `pi install <source>` or into an auto-discovered directory \
             (`~/.pi/agent/extensions/`, `.pi/extensions/`)"
        )),
    }
}

/// Whether the extension entry point can be found.
pub(crate) fn extension_configured() -> bool {
    extension_location().is_some()
}

fn extension_location() -> Option<PathBuf> {
    std::env::var_os(PI_EXTENSION_PATH_ENV)
        .map(PathBuf::from)
        .filter(|path| path.exists())
}

/// The gateway URL the extension will post to.
///
/// Precedence matters here, and the obvious shortcut is wrong: the extension
/// itself only knows `NEMO_RELAY_PI_GATEWAY_URL` and its own hard-coded default,
/// but the launcher sets that variable *from the resolved configuration*. A
/// preflight that only read the variable would probe `127.0.0.1:4040` for a user
/// who configured a different `bind`, and report their working gateway as down.
///
/// `bind` is a `SocketAddr`, so it is always a concrete host and port -- a
/// wildcard bind such as `0.0.0.0:4040` is reachable on loopback, which is where
/// pi runs.
pub(crate) fn gateway_url(bind: Option<std::net::SocketAddr>) -> String {
    if let Some(url) = std::env::var(PI_GATEWAY_URL_ENV)
        .ok()
        .map(|url| url.trim_end_matches('/').to_string())
        .filter(|url| !url.is_empty())
    {
        return url;
    }
    match bind {
        Some(bind) if bind.ip().is_unspecified() => format!("http://127.0.0.1:{}", bind.port()),
        Some(bind) => format!("http://{bind}"),
        None => DEFAULT_GATEWAY_URL.to_string(),
    }
}

/// Every place pi could load an extension from, that currently holds one.
///
/// Deliberately reports *any* auto-discovered entry rather than trying to
/// recognize the NeMo Relay extension by filename: `pi install` renames and
/// nests what it writes, so a filename match would miss the installed layout
/// and quietly report nothing -- which is the failure mode being guarded
/// against. The trust question is a property of the directory, not of the file.
pub(crate) fn extension_sites(cwd: &Path) -> Vec<ExtensionSite> {
    let mut sites = Vec::new();
    if let Some(path) = extension_location() {
        sites.push(ExtensionSite {
            path,
            scope: ExtensionScope::Explicit,
        });
    }
    if let Some(dir) = user_extensions_dir()
        && directory_has_entries(&dir)
    {
        sites.push(ExtensionSite {
            path: dir,
            scope: ExtensionScope::User,
        });
    }
    let project_dir = cwd.join(PI_CONFIG_DIR).join("extensions");
    if directory_has_entries(&project_dir) {
        sites.push(ExtensionSite {
            path: project_dir,
            scope: ExtensionScope::Project,
        });
    }
    sites
}

/// `~/.pi/agent/extensions`, honoring pi's own directory override.
fn user_extensions_dir() -> Option<PathBuf> {
    let agent_dir = match std::env::var_os(PI_AGENT_DIR_ENV) {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => crate::agents::shared::host::home_dir()
            .ok()?
            .join(PI_CONFIG_DIR)
            .join("agent"),
    };
    Some(agent_dir.join("extensions"))
}

fn directory_has_entries(path: &Path) -> bool {
    std::fs::read_dir(path).is_ok_and(|mut entries| entries.next().is_some())
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/pi_doctor_tests.rs"]
mod tests;
