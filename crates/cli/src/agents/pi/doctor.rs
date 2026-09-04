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
    /// What pi's own settings do to this copy.
    ///
    /// A copy pi will not load is installed and still silently absent -- the same failure as the
    /// trust gate, from another direction -- so it must not read as a plain Pass. A copy whose
    /// filter this check cannot evaluate must not read as one either.
    ///
    /// `-e` applies no settings filter, so a launch can still use an `Excluded` copy; that is
    /// deliberate, and it is why `launchable_extension_path` merely *prefers* a loading one.
    pub(crate) filter: SettingsFilter,
}

impl ExtensionSite {
    /// Whether pi loads this copy without help -- what makes it a second copy beside `-e`.
    fn loads_on_its_own(&self) -> bool {
        self.scope != ExtensionScope::Project && self.filter == SettingsFilter::Loads
    }
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
            "NeMo Relay pi extension not located; run `nemo-relay install pi`, or set \
             {PI_EXTENSION_PATH_ENV} to a copy you manage yourself"
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
            filter: SettingsFilter::Loads,
        });
    }
    let discovered = |sites: &mut Vec<ExtensionSite>, dir: &Path, scope| {
        sites.extend(
            relay_entries_in_directory(dir)
                .into_iter()
                .map(|path| ExtensionSite {
                    path,
                    scope,
                    filter: SettingsFilter::Loads,
                }),
        );
    };
    let recorded = |sites: &mut Vec<ExtensionSite>, settings: &Path, scope| {
        sites.extend(
            relay_packages_in_settings(settings)
                .into_iter()
                .map(|install| ExtensionSite {
                    path: install.path,
                    scope,
                    filter: install.filter,
                }),
        );
    };
    let listed = |sites: &mut Vec<ExtensionSite>, settings: &Path, base: &Path, scope| {
        sites.extend(
            relay_extensions_listed_in_settings(settings, base)
                .into_iter()
                .map(|install| ExtensionSite {
                    path: install.path,
                    scope,
                    filter: install.filter,
                }),
        );
    };
    if let Some(dir) = user_extensions_dir() {
        discovered(&mut sites, &dir, ExtensionScope::User);
    }
    if let Some(agent_dir) = pi_agent_dir() {
        listed(
            &mut sites,
            &agent_dir.join(PI_SETTINGS_FILE),
            &agent_dir,
            ExtensionScope::User,
        );
    }
    if let Some(settings) = user_settings_path() {
        recorded(&mut sites, &settings, ExtensionScope::User);
    }
    discovered(
        &mut sites,
        &cwd.join(PI_CONFIG_DIR).join("extensions"),
        ExtensionScope::Project,
    );
    let project_config = cwd.join(PI_CONFIG_DIR);
    listed(
        &mut sites,
        &project_config.join(PI_SETTINGS_FILE),
        &project_config,
        ExtensionScope::Project,
    );
    recorded(
        &mut sites,
        &project_config.join(PI_SETTINGS_FILE),
        ExtensionScope::Project,
    );
    sites
}

/// Copies named by `settings.json`'s own `extensions` array, in either scope.
///
/// A third registration route, and the one easiest to miss: it is a *sibling* of
/// `packages`, read by a generic loop over pi's four resource types
/// (`resolve`, pi `v0.84.0`, `core/package-manager.ts:906-931`) rather than by
/// anything named after extensions -- `SettingsManager::getExtensionPaths` exists
/// and has no callers, which makes the key look dead until that loop is read.
/// Entries resolve against the settings file's own directory, exactly as
/// `packages` entries do.
///
/// pi treats a *file* entry as the extension and walks a *directory* entry as a
/// container of them (`collectResourceFiles` -> `collectAutoExtensionEntries`,
/// `core/package-manager.ts:618-625`), so both shapes are checked: the entry
/// itself, then its children.
///
/// A pattern entry (`+`, `-`, `!`) filters the collected set through the same
/// globbing `packages` filters use, over files this module does not enumerate the
/// way pi does -- so any pattern present makes the verdict `Undecided` rather than
/// a guess in either direction.
fn relay_extensions_listed_in_settings(settings: &Path, base: &Path) -> Vec<RecordedInstall> {
    let Some(value) = std::fs::read_to_string(settings)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
    else {
        return Vec::new();
    };
    let Some(entries) = value
        .get("extensions")
        .and_then(serde_json::Value::as_array)
    else {
        return Vec::new();
    };
    let entries: Vec<&str> = entries
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    let filtered = entries
        .iter()
        .any(|entry| entry.starts_with(['+', '-', '!']));
    let filter = if filtered {
        SettingsFilter::Undecided
    } else {
        SettingsFilter::Loads
    };
    let mut found = Vec::new();
    for entry in entries
        .iter()
        .filter(|entry| !entry.starts_with(['+', '-', '!']))
    {
        let resolved = base.join(entry);
        if is_relay_extension(&resolved) {
            found.push(RecordedInstall {
                path: resolved,
                filter,
            });
            continue;
        }
        found.extend(
            relay_entries_in_directory(&resolved)
                .into_iter()
                .map(|path| RecordedInstall { path, filter }),
        );
    }
    found
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
/// A copy pi already loads is preferred over one its settings switch off, even
/// when the disabled one comes first. `-e` applies no settings filter, so passing
/// the disabled copy would *re-enable* it -- and then the enabled one is a genuine
/// second copy and the launch is refused over a duplicate the choice manufactured.
/// Choosing the enabled copy leaves the disabled one disabled and loads exactly
/// once. A disabled copy is still used when it is all there is, because `-e` makes
/// it work.
pub(crate) fn launchable_extension_path(cwd: &Path) -> Option<PathBuf> {
    let sites = relay_extension_sites(cwd);
    let usable = || {
        sites
            .iter()
            .filter(|site| site.scope != ExtensionScope::Project && site.path.exists())
    };
    usable()
        .find(|site| site.filter == SettingsFilter::Loads)
        .or_else(|| usable().next())
        .map(|site| site.path.clone())
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
/// block every launch in an untrusted project over a copy that will not run. The
/// launch note and the trust warning name it instead.
///
/// A copy pi's own settings switch off is excluded for the opposite reason -- it is
/// reliably *not* a load. A filtered-off package's files are added with
/// `enabled: false` and dropped before the merge (`applyPackageFilter`, pi
/// `v0.84.0`, `core/package-manager.ts:2208`), so counting it here refused a launch
/// over a copy that was never going to register a hook.
///
/// So this refuses only on copies pi is certain to load, and never on one it merely
/// might. The remaining false negative -- a trusted project holding a second copy --
/// is a doubled trace with a note rather than a blocked launch, which is the right
/// side to be wrong on: `-p`, `--mode json` and `--mode rpc` never prompt for trust,
/// so untrusted is the common state.
pub(crate) fn conflicting_extension_site(cwd: &Path, launched: &Path) -> Option<PathBuf> {
    let launched_root = package_root(launched);
    relay_extension_sites(cwd)
        .into_iter()
        .filter(ExtensionSite::loads_on_its_own)
        .find(|site| package_root(&site.path) != launched_root)
        .map(|site| site.path)
}

/// Project-scoped copies that would load *beside* the launched one, if the project
/// is trusted.
///
/// Two exclusions, both for the same reason the launcher does not refuse on these:
/// a copy pi's settings switch off never loads, and a copy that canonicalizes to
/// the launched package is the same package -- pi de-duplicates by path, so it
/// loads once. Warning about either would describe a doubled trace that cannot
/// happen.
pub(crate) fn project_copies_beside(cwd: &Path, launched: &Path) -> Vec<PathBuf> {
    let launched_root = package_root(launched);
    relay_extension_sites(cwd)
        .into_iter()
        .filter(|site| {
            site.scope == ExtensionScope::Project
                && site.filter != SettingsFilter::Excluded
                && package_root(&site.path) != launched_root
        })
        .map(|site| site.path)
        .collect()
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

/// Every copy of the NeMo Relay extension inside a pi auto-discovery directory.
///
/// Matched on the package name, not on "the directory is non-empty". A user with
/// somebody else's pi extension installed was otherwise told their *Relay*
/// extension was fine -- and, worse, a project-scoped install of an unrelated
/// package raised a Relay trust warning about a file that has nothing to do with
/// Relay.
///
/// **Every** copy, not the first. pi walks the whole directory and resolves each
/// subdirectory's entry points independently (`collectAutoExtensionEntries`, pi
/// `v0.84.0`, `core/package-manager.ts:560`), so two copies here are two packages
/// to pi and both register hooks. Stopping at the first hid exactly the doubled
/// trace the duplicate check exists to refuse.
///
/// Only the shapes pi accepts as an entry are considered -- a directory, or a
/// `.ts`/`.js` file. A flat copy puts a `package.json` beside `index.ts`, and that
/// manifest makes its own sibling look like this extension, so handing that file to
/// `-e` would give pi something it cannot import. Sorted because `read_dir` order is
/// undefined and the launcher's choice of copy must not vary run to run.
fn relay_entries_in_directory(dir: &Path) -> Vec<PathBuf> {
    let Ok(read) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = read
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            let loadable = path.is_dir()
                || matches!(
                    path.extension().and_then(std::ffi::OsStr::to_str),
                    Some("ts" | "js")
                );
            loadable && is_relay_extension(path)
        })
        .collect();
    found.sort();
    found
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

/// Whether pi would load *anything* from this directory as an extension.
///
/// Deliberately wider than [`is_relay_extension`], and wider than the manifest test this
/// started as. pi resolves a package directory through its `package.json` when there is
/// one, and **falls back to a bare `index.ts`/`index.js` when there is not**
/// (`collectAutoExtensionEntries`, pi `v0.84.0`, `core/package-manager.ts:496-566`). A
/// check that asked only "does the manifest name Relay" answered "no" for somebody else's
/// package and for a manifest-less extension alike, and the installer read that as
/// "nothing of value here".
///
/// Any manifest counts, not just ours: a directory holding someone else's package is not
/// Relay's to touch either.
pub(super) fn loadable_extension_dir(dir: &Path) -> bool {
    dir.join("package.json").is_file()
        || dir.join("index.ts").is_file()
        || dir.join("index.js").is_file()
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
    filter: SettingsFilter,
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
/// Every entry, not the first: two entries naming different directories are two
/// identities to pi (`getPackageIdentity`, `core/package-manager.ts:1660`), so
/// `dedupePackages` keeps both and both load.
fn relay_packages_in_settings(settings: &Path) -> Vec<RecordedInstall> {
    let Some(base) = settings.parent() else {
        return Vec::new();
    };
    let Some(value) = std::fs::read_to_string(settings)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
    else {
        return Vec::new();
    };
    let Some(entries) = value.get("packages").and_then(serde_json::Value::as_array) else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let source = package_source(entry)?;
            let resolved = base.join(source);
            if is_relay_extension(&resolved) {
                let filter = entry_filters_extensions(entry, &resolved);
                return Some(RecordedInstall {
                    path: resolved,
                    filter,
                });
            }
            source
                .contains(RELAY_PACKAGE_NAME)
                .then(|| RecordedInstall {
                    path: PathBuf::from(source),
                    filter: entry_filters_extensions(entry, Path::new(source)),
                })
        })
        .collect()
}

/// The source string of one `packages` entry, whichever shape it was written in.
fn package_source(entry: &serde_json::Value) -> Option<&str> {
    entry
        .as_str()
        .or_else(|| entry.get("source").and_then(serde_json::Value::as_str))
}

/// What an object-form entry's filters do to that package's extensions.
///
/// pi sorts a pattern list into four buckets: `+x` force-include and `-x`
/// force-exclude, both compared as exact strings, and `!x` exclude and bare `x`
/// include, both matched as globs (`applyPatterns`, pi `v0.84.0`,
/// `core/package-manager.ts:712-756`). Only the exact ones can be decided from
/// here without reimplementing `minimatch`, which is why the answer is a tri-state
/// rather than a bool: reporting a package as loaded because a glob could not be
/// read is the same silent Pass this module exists to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SettingsFilter {
    /// pi loads this package's extensions.
    Loads,
    /// pi loads none of them -- the entry point is filtered out for certain.
    Excluded,
    /// The filter turns on a glob this check does not evaluate. pi has the files
    /// and the matcher; this does not, so it says so rather than guessing.
    Undecided,
}

fn entry_filters_extensions(entry: &serde_json::Value, path: &Path) -> SettingsFilter {
    let Some(object) = entry.as_object() else {
        return SettingsFilter::Loads;
    };
    let autoload_off = object.get("autoload") == Some(&serde_json::Value::Bool(false));
    let Some(patterns) = object
        .get("extensions")
        .and_then(serde_json::Value::as_array)
    else {
        // `autoload: false` with no patterns starts from nothing and adds nothing back
        // (`applyPackageDeltaFilter`, `core/package-manager.ts:2232`).
        return if autoload_off {
            SettingsFilter::Excluded
        } else {
            SettingsFilter::Loads
        };
    };
    let patterns: Vec<&str> = patterns
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    // An empty array explicitly disables every resource of the type
    // (`applyPackageFilter`, `core/package-manager.ts:2208`).
    if patterns.is_empty() {
        return SettingsFilter::Excluded;
    }
    let declared = manifest_extension_entries(path);
    if declared.is_empty() {
        return SettingsFilter::Undecided;
    }
    let verdicts = declared
        .iter()
        .map(|entry| entry_verdict(entry, &patterns, autoload_off));
    let mut excluded = true;
    for verdict in verdicts {
        match verdict {
            SettingsFilter::Undecided => return SettingsFilter::Undecided,
            SettingsFilter::Loads => excluded = false,
            SettingsFilter::Excluded => {}
        }
    }
    if excluded {
        SettingsFilter::Excluded
    } else {
        SettingsFilter::Loads
    }
}

/// What a filter's patterns do to one declared entry point.
///
/// Under a normal filter the order is include, exclude, force-include,
/// force-exclude, and only the last step is unconditional -- so a `-` naming the
/// entry settles it, and otherwise the answer turns on whether any include or
/// exclude glob could match. Under an `autoload: false` delta the patterns are
/// replayed in order and the last one naming the entry wins, with an entry no
/// pattern names never added at all (`applyAutoloadDisabledPatterns`,
/// `core/package-manager.ts:760-777`).
fn entry_verdict(entry: &str, patterns: &[&str], autoload_off: bool) -> SettingsFilter {
    if autoload_off {
        return autoload_disabled_entry_verdict(entry, patterns);
    }

    normal_entry_verdict(entry, patterns)
}

fn autoload_disabled_entry_verdict(entry: &str, patterns: &[&str]) -> SettingsFilter {
    let mut decided = SettingsFilter::Excluded;
    for pattern in patterns {
        match classify_pattern(pattern, entry) {
            PatternEffect::Unknown => return SettingsFilter::Undecided,
            PatternEffect::Names(enabled) => {
                decided = if enabled {
                    SettingsFilter::Loads
                } else {
                    SettingsFilter::Excluded
                };
            }
            PatternEffect::Silent => {}
        }
    }
    decided
}

fn normal_entry_verdict(entry: &str, patterns: &[&str]) -> SettingsFilter {
    let mut force_excluded = false;
    let mut force_included = false;
    let mut includes = 0_usize;
    let mut included = false;
    let mut excluded = false;
    for pattern in patterns {
        let (marker, target) = split_pattern(pattern);
        let literal = literal_target(target);
        match marker {
            Some('+') => force_included |= target_matches(target, entry),
            Some('-') => force_excluded |= target_matches(target, entry),
            Some('!') => match literal {
                Some(literal) => excluded |= literal == entry,
                // A glob exclude could remove the entry, and only a force-include
                // would bring it back.
                None if !force_included => return SettingsFilter::Undecided,
                None => {}
            },
            _ => {
                includes += 1;
                match literal {
                    Some(literal) => included |= literal == entry,
                    None => return SettingsFilter::Undecided,
                }
            }
        }
    }
    if force_excluded {
        return SettingsFilter::Excluded;
    }
    if force_included {
        return SettingsFilter::Loads;
    }
    // With no includes at all, step 1 keeps everything.
    if (includes == 0 || included) && !excluded {
        SettingsFilter::Loads
    } else {
        SettingsFilter::Excluded
    }
}

/// What one `autoload: false` delta pattern does to a given entry.
enum PatternEffect {
    /// Names the entry, and either enables or disables it.
    Names(bool),
    /// Cannot say -- the pattern is a glob.
    Unknown,
    /// Names something else.
    Silent,
}

fn classify_pattern(pattern: &str, entry: &str) -> PatternEffect {
    let (marker, target) = split_pattern(pattern);
    let enabled = !matches!(marker, Some('-') | Some('!'));
    if matches!(marker, Some('+') | Some('-')) {
        return if target_matches(target, entry) {
            PatternEffect::Names(enabled)
        } else {
            PatternEffect::Silent
        };
    }
    match literal_target(target) {
        Some(literal) if literal == entry => PatternEffect::Names(enabled),
        Some(_) => PatternEffect::Silent,
        None => PatternEffect::Unknown,
    }
}

/// The `+`/`-`/`!` marker and the rest of a pattern.
fn split_pattern(pattern: &str) -> (Option<char>, &str) {
    match pattern.chars().next() {
        Some(marker @ ('+' | '-' | '!')) => (Some(marker), &pattern[marker.len_utf8()..]),
        _ => (None, pattern),
    }
}

/// A pattern's target with a leading `./` removed, unless it carries glob syntax.
///
/// pi strips that prefix before comparing (`normalizeExactPattern`, pi `v0.84.0`,
/// `core/package-manager.ts:656`), which is what makes an exact pattern decidable
/// from here at all.
fn literal_target(target: &str) -> Option<&str> {
    if target.contains(['*', '?', '[', '{']) {
        return None;
    }
    Some(target.strip_prefix("./").unwrap_or(target))
}

/// Whether an exact pattern names this entry. Exact patterns are never globs.
fn target_matches(target: &str, entry: &str) -> bool {
    target.strip_prefix("./").unwrap_or(target) == entry
}

/// The extension entry points a package declares, relative to its own root.
///
/// Empty when they cannot be pinned down -- a source that is not on disk, or a
/// manifest entry carrying a glob or an override marker, which pi expands with
/// `globSync` rather than comparing as a string. The caller reads empty as
/// "cannot tell" and says so rather than reporting a Pass.
fn manifest_extension_entries(path: &Path) -> Vec<String> {
    let Some(root) = package_root(path) else {
        return Vec::new();
    };
    let entries: Vec<String> = std::fs::read_to_string(root.join("package.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .as_ref()
        .and_then(|manifest| manifest.get("pi")?.get("extensions")?.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(|entry| entry.strip_prefix("./").unwrap_or(entry).to_string())
                .collect()
        })
        .unwrap_or_default();
    if entries
        .iter()
        .any(|entry| entry.contains(['*', '?']) || entry.starts_with(['+', '-', '!']))
    {
        return Vec::new();
    }
    entries
}

/// `<agent dir>/settings.json`, where `pi install` records a user-scope package.
fn user_settings_path() -> Option<PathBuf> {
    Some(pi_agent_dir()?.join(PI_SETTINGS_FILE))
}

/// `~/.pi/agent`, honoring pi's own directory override.
pub(super) fn pi_agent_dir() -> Option<PathBuf> {
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
pub(super) fn user_extensions_dir() -> Option<PathBuf> {
    Some(pi_agent_dir()?.join("extensions"))
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/pi_doctor_tests.rs"]
mod tests;
