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
//! extensions to its candidate set only when the project is trusted (pi
//! `v0.84.0`, `core/package-manager.ts:2395`), and `-p`, `--mode json` and
//! `--mode rpc` never prompt for trust (`docs/security.md:29`). Under the
//! default policy a
//! project-scoped extension is therefore dropped by a bare conditional -- not
//! by an error path, so it never reaches pi's extension-load error list and pi
//! does not consider it a failure. Nothing reports it, and **the extension
//! cannot report it either**: by construction it is not running. A preflight
//! that reads the filesystem is the only place this can be caught.

use std::path::{Path, PathBuf};

use super::launch::{PI_EXTENSION_PATH_ENV, PI_GATEWAY_URL_ENV};

/// pi's per-user configuration root, `~/.pi/agent` unless overridden.
///
/// Mirrors `getAgentDir()` (pi `v0.84.0`, `config.ts:515-521`), including the
/// environment override, so the preflight looks where pi will actually look.
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
    /// Whether pi's own settings switch this copy off.
    ///
    /// An object-form `packages` entry carries per-resource filters, and two shapes leave nothing
    /// enabled for extensions: an empty `extensions` array, and `autoload: false` with no
    /// extension patterns (pi `v0.84.0`, `core/package-manager.ts:2208` and `:2232`). The copy is
    /// installed and pi still never loads it, which is the same silent drop as the trust gate and
    /// must not read as a plain Pass.
    ///
    /// A launch is unaffected on purpose: `-e` resolves its argument with no filter at all, so
    /// `launchable_extension_path` ignores this flag. The user asked for instrumentation by
    /// running the launcher; the warning is for their own `pi` sessions.
    pub(crate) disabled_by_settings: bool,
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
            disabled_by_settings: false,
        });
    }
    if let Some(dir) = user_extensions_dir()
        && let Some(path) = relay_entry_in_directory(&dir)
    {
        sites.push(ExtensionSite {
            path,
            scope: ExtensionScope::User,
            disabled_by_settings: false,
        });
    }
    if let Some(settings) = user_settings_path()
        && let Some(install) = relay_package_in_settings(&settings)
    {
        sites.push(ExtensionSite {
            path: install.path,
            scope: ExtensionScope::User,
            disabled_by_settings: install.disabled,
        });
    }
    if let Some(path) = relay_entry_in_directory(&cwd.join(PI_CONFIG_DIR).join("extensions")) {
        sites.push(ExtensionSite {
            path,
            scope: ExtensionScope::Project,
            disabled_by_settings: false,
        });
    }
    if let Some(install) =
        relay_package_in_settings(&cwd.join(PI_CONFIG_DIR).join(PI_SETTINGS_FILE))
    {
        sites.push(ExtensionSite {
            path: install.path,
            scope: ExtensionScope::Project,
            disabled_by_settings: install.disabled,
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
/// entry-point precedence is pi's to change. A copy pi would *not* have
/// discovered is a different matter, and is what `conflicting_extension_site`
/// is for.
pub(crate) fn launchable_extension_path(cwd: &Path) -> Option<PathBuf> {
    relay_extension_sites(cwd)
        .into_iter()
        .find(|site| site.scope != ExtensionScope::Project && site.path.exists())
        .map(|site| site.path)
}

/// A *second* copy of this extension that pi would load beside the launched one.
///
/// `-e` adds to pi's extension set; it does not replace it. pi merges the
/// command-line and discovered sets and de-duplicates them by canonicalized path
/// alone (`mergePaths`, pi `v0.84.0`, `core/resource-loader.ts:845`), and the
/// identity it gives a local package is that same path (`getPackageIdentity`,
/// `core/package-manager.ts:1660`) -- so **nothing in pi notices that two
/// directories hold one package**. Each copy gets its own factory call and its own
/// handler map (`core/extensions/loader.ts:506`), and the runner walks every
/// extension for every hook (`core/extensions/runner.ts:805`), so every hook is
/// posted twice. A duplicated `turn_start` closes the turn its twin just opened as
/// superseded, and the inline-shell gate decides one command twice under two
/// spans, with the second verdict the one the user gets.
///
/// Compared by *package root* rather than by path, because one install is
/// reachable both as its directory and as the entry file inside it, and pi
/// resolves both to the same file through the `pi.extensions` manifest. Symlinks
/// are resolved because pi resolves them too -- its `canonicalizePath` is
/// `realpathSync` (`utils/paths.ts:28`) -- so a symlinked copy is one copy to pi
/// and must be one copy here.
///
/// Project scope is excluded: pi loads a project-scoped extension only for a
/// trusted project, so it is not reliably a second load, and refusing on it would
/// block launches that are fine. The existing trust warning already names it.
pub(crate) fn conflicting_extension_site(cwd: &Path, launched: &Path) -> Option<PathBuf> {
    let launched_root = package_root(launched);
    relay_extension_sites(cwd)
        .into_iter()
        .filter(|site| site.scope != ExtensionScope::Project)
        .find(|site| package_root(&site.path) != launched_root)
        .map(|site| site.path)
}

/// The package directory a site belongs to, or `None` when it is not on disk.
///
/// A source that is not a path -- an npm or git specifier `pi install` recorded --
/// has no root to compare and is a separate installed copy by construction, so
/// `None` is the honest answer and makes it compare unequal to a real checkout.
fn package_root(path: &Path) -> Option<PathBuf> {
    let canonical = std::fs::canonicalize(path).ok()?;
    if manifest_names_relay(&canonical.join("package.json")) {
        return Some(canonical);
    }
    canonical.parent().map(Path::to_path_buf)
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

/// A `packages` entry that records this extension, and whether pi will load it.
struct RecordedInstall {
    path: PathBuf,
    disabled: bool,
}

/// The NeMo Relay package among the sources `pi install` recorded, if any.
///
/// Each entry is **either a source string or an object** carrying that same source
/// under `source` alongside per-resource filters (pi `v0.84.0`,
/// `core/settings-manager.ts:72-87`); both shapes resolve through one code path in
/// pi. Reading only the string shape was not a theoretical gap: pi's own
/// configuration selector rewrites a string entry into the object form the moment a
/// user toggles any resource of that package
/// (`interactive/components/config-selector.ts:595-598`), so one keystroke in pi's
/// own UI made this check report an installed extension as missing -- and the
/// launcher, which shares this resolution, refuse to start.
///
/// A local source is a path relative to the settings file's own directory, which is
/// where pi resolves it from too (`getBaseDirForScope`,
/// `core/package-manager.ts:2107-2115`). Only a local source can be resolved from
/// here -- an npm or git source is a name, not a location -- so those fall back to
/// matching the package name inside the specifier, which is the best signal
/// available without fetching anything.
fn relay_package_in_settings(settings: &Path) -> Option<RecordedInstall> {
    let base = settings.parent()?;
    let raw = std::fs::read_to_string(settings).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value.get("packages")?.as_array()?.iter().find_map(|entry| {
        let source = package_source(entry)?;
        let disabled = entry_disables_extensions(entry);
        let resolved = base.join(source);
        if is_relay_extension(&resolved) {
            return Some(RecordedInstall {
                path: resolved,
                disabled,
            });
        }
        source
            .contains(RELAY_PACKAGE_NAME)
            .then(|| RecordedInstall {
                path: PathBuf::from(source),
                disabled,
            })
    })
}

/// The source string of one `packages` entry, whichever shape it was written in.
fn package_source(entry: &serde_json::Value) -> Option<&str> {
    entry
        .as_str()
        .or_else(|| entry.get("source").and_then(serde_json::Value::as_str))
}

/// Whether an object-form entry's filters switch that package's extensions off.
///
/// Only the two shapes pi decides without consulting the package manifest are
/// recognized: an empty `extensions` array disables every extension file in the
/// package (`applyPackageFilter`, pi `v0.84.0`, `core/package-manager.ts:2208`),
/// and `autoload: false` starts from nothing, so an entry adding no `extensions`
/// patterns adds nothing back (`applyPackageDeltaFilter`, `:2232`).
///
/// A non-empty pattern list is matched against the manifest, which this module does
/// not read, so it counts as enabled. Guessing wrong in that direction produces the
/// false negative this module exists to prevent.
fn entry_disables_extensions(entry: &serde_json::Value) -> bool {
    let Some(object) = entry.as_object() else {
        return false;
    };
    match object
        .get("extensions")
        .and_then(serde_json::Value::as_array)
    {
        Some(patterns) => patterns.is_empty(),
        None => object.get("autoload") == Some(&serde_json::Value::Bool(false)),
    }
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
