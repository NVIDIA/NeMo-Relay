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
pub(crate) const PI_AGENT_DIR_ENV: &str = "PI_CODING_AGENT_DIR";

/// pi's configuration directory name, from its `piConfig.configDir`.
const PI_CONFIG_DIR: &str = ".pi";

/// Where pi records installed packages, in both scopes.
const PI_SETTINGS_FILE: &str = "settings.json";

/// This extension's package name -- how it is told apart from anyone else's.
/// Must match `integrations/pi/package.json`.
const RELAY_PACKAGE_NAME: &str = "nemo-relay-pi";

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

impl ExtensionScope {
    /// How this route behaves, in the words the check reports.
    pub(crate) fn describe(self) -> &'static str {
        match self {
            Self::Explicit => "passed with `-e`, which loads first and is never trust-gated",
            Self::User => "user scope, which is never trust-gated",
            Self::Project => "project scope, which pi loads only for a trusted project",
        }
    }
}

/// A place a pi extension was found, and how pi would reach it.
#[derive(Debug, Clone)]
pub(crate) struct ExtensionSite {
    pub(crate) path: PathBuf,
    pub(crate) scope: ExtensionScope,
}

/// Human-readable hook status for `nemo-relay doctor`.
///
/// Shares its answer with the load-path check, deliberately. While this read only
/// the environment variable and that scanned directories, one `doctor` run could
/// report "pi extension not located" *and* a passing load path for the same
/// machine, in the same output.
pub(crate) fn hook_status() -> Result<String, String> {
    match relay_extension_sites(&current_dir()).first() {
        Some(site) => Ok(format!(
            "NeMo Relay pi extension resolved at {} ({}); hooks are emitted by the extension, \
             not by pi itself",
            site.path.display(),
            site.scope.describe()
        )),
        None => Ok(format!(
            "NeMo Relay pi extension not located; set {PI_EXTENSION_PATH_ENV}, run \
             `pi install <path to integrations/pi>`, or copy it into `~/.pi/agent/extensions/`"
        )),
    }
}

/// Whether *this* extension -- not merely some pi extension -- can be found.
pub(crate) fn extension_configured() -> bool {
    !relay_extension_sites(&current_dir()).is_empty()
}

fn current_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// The explicitly configured extension, but only when it is *this* extension.
///
/// The name check is not redundant with the directory scans below. A stale or
/// mistyped variable that happens to name an existing path -- somebody else's
/// extension, or a checkout that no longer holds ours -- otherwise reported a
/// Pass here *and* made the launcher hand that path to `-e`, so pi loaded code
/// that emits no hooks while every Relay check said the setup was ready.
fn extension_location() -> Option<PathBuf> {
    std::env::var_os(PI_EXTENSION_PATH_ENV)
        .map(PathBuf::from)
        .filter(|path| path.exists() && is_relay_extension(path))
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
/// **Two unrelated routes, and missing either one makes this check lie.**
///
/// *Auto-discovery* reads `<agent dir>/extensions` and `<cwd>/.pi/extensions`.
/// Any entry counts; the trust question is a property of the directory, not of
/// the file, so there is nothing to recognize by name.
///
/// *`pi install`* does not touch those directories at all. It appends the source
/// to a `packages` array in `settings.json` -- `<agent dir>/settings.json` for
/// user scope, `<cwd>/.pi/settings.json` for `--local` -- and for a local path
/// copies nothing whatsoever. Scanning only the extension directories therefore
/// reported "no pi extension found" to a user who had just run the install
/// command the docs recommend, and could not see a trust-gated `--local` entry
/// at all, which is the case this whole module exists to catch.
pub(crate) fn relay_extension_sites(cwd: &Path) -> Vec<ExtensionSite> {
    let mut sites = Vec::new();
    if let Some(path) = extension_location() {
        sites.push(ExtensionSite {
            path,
            scope: ExtensionScope::Explicit,
        });
    }
    if let Some(dir) = user_extensions_dir()
        && let Some(path) = relay_entry_in_directory(&dir)
    {
        sites.push(ExtensionSite {
            path,
            scope: ExtensionScope::User,
        });
    }
    if let Some(settings) = user_settings_path()
        && let Some(path) = relay_package_in_settings(&settings)
    {
        sites.push(ExtensionSite {
            path,
            scope: ExtensionScope::User,
        });
    }
    if let Some(path) = relay_entry_in_directory(&cwd.join(PI_CONFIG_DIR).join("extensions")) {
        sites.push(ExtensionSite {
            path,
            scope: ExtensionScope::Project,
        });
    }
    if let Some(path) = relay_package_in_settings(&cwd.join(PI_CONFIG_DIR).join(PI_SETTINGS_FILE)) {
        sites.push(ExtensionSite {
            path,
            scope: ExtensionScope::Project,
        });
    }
    sites
}

/// The path `nemo-relay run --agent pi` hands to `-e`, when there is one.
///
/// Explicit first, then user scope -- the order `relay_extension_sites` already
/// returns. **Project scope is excluded on purpose.** `-e` is not trust-gated, so
/// promoting a project-scoped extension to it would run repository code pi itself
/// declined to trust: the launcher would be undoing the very gate this module
/// exists to report. A site that is not a path on disk is excluded for a related
/// reason -- `pi install` can record an npm or git specifier, and pi resolves an
/// `-e` argument as a package *source*, so handing one back could make a launch
/// fetch and install from the network.
///
/// Handing pi a path it would have discovered anyway is safe: pi canonicalizes and
/// de-duplicates the merged command-line and discovered sets before loading
/// (`mergePaths`, pi `v0.84.0`, `core/resource-loader.ts:845`), and both routes
/// resolve a package directory through the same `pi.extensions` manifest, so the
/// extension loads -- and registers its hooks -- exactly once. Passing the
/// directory is also why nothing here reads that manifest: pi does it, and its
/// entry-point precedence is pi's to change.
pub(crate) fn launchable_extension_path(cwd: &Path) -> Option<PathBuf> {
    relay_extension_sites(cwd)
        .into_iter()
        .find(|site| site.scope != ExtensionScope::Project && site.path.exists())
        .map(|site| site.path)
}

/// The NeMo Relay extension inside a pi auto-discovery directory, if it is there.
///
/// Matched on the package name, not on "the directory is non-empty". A user with
/// somebody else's pi extension installed was otherwise told their *Relay*
/// extension was fine -- and, worse, a project-scoped install of an unrelated
/// package raised a Relay trust warning about a file that has nothing to do with
/// Relay.
fn relay_entry_in_directory(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| is_relay_extension(path))
}

/// Whether a path is this extension: a package directory whose manifest names it,
/// or a file sitting inside one.
fn is_relay_extension(path: &Path) -> bool {
    if manifest_names_relay(&path.join("package.json")) {
        return true;
    }
    path.parent()
        .is_some_and(|parent| manifest_names_relay(&parent.join("package.json")))
}

fn manifest_names_relay(manifest: &Path) -> bool {
    std::fs::read_to_string(manifest)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|value| {
            value
                .get("name")
                .and_then(serde_json::Value::as_str)
                .map(|name| name == RELAY_PACKAGE_NAME)
        })
        .unwrap_or(false)
}

/// The NeMo Relay package among the sources `pi install` recorded, if any.
///
/// Each entry is a source string, and a local one is a path relative to the
/// settings file's own directory. Only a local source can be resolved from here
/// -- an npm or git source is a name, not a location -- so those fall back to
/// matching the package name inside the specifier, which is the best signal
/// available without fetching anything.
fn relay_package_in_settings(settings: &Path) -> Option<PathBuf> {
    let base = settings.parent()?;
    let raw = std::fs::read_to_string(settings).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value
        .get("packages")?
        .as_array()?
        .iter()
        .filter_map(serde_json::Value::as_str)
        .find_map(|source| {
            let resolved = base.join(source);
            if is_relay_extension(&resolved) {
                return Some(resolved);
            }
            source
                .contains(RELAY_PACKAGE_NAME)
                .then(|| PathBuf::from(source))
        })
}

/// `<agent dir>/settings.json`, where `pi install` records a user-scope package.
fn user_settings_path() -> Option<PathBuf> {
    Some(pi_agent_dir()?.join(PI_SETTINGS_FILE))
}

/// `~/.pi/agent`, honoring pi's own directory override.
fn pi_agent_dir() -> Option<PathBuf> {
    match std::env::var_os(PI_AGENT_DIR_ENV) {
        Some(dir) if !dir.is_empty() => Some(PathBuf::from(dir)),
        _ => Some(
            crate::agents::shared::host::home_dir()
                .ok()?
                .join(PI_CONFIG_DIR)
                .join("agent"),
        ),
    }
}

/// `~/.pi/agent/extensions`, the auto-discovery directory.
fn user_extensions_dir() -> Option<PathBuf> {
    Some(pi_agent_dir()?.join("extensions"))
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/pi_doctor_tests.rs"]
mod tests;
